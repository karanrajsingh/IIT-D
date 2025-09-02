import numpy as np
import time
import math
from env import FootballSkillsEnv


def policy_iteration(env, gamma=0.95, theta=1e-6):
    """
    Core Policy Iteration algorithm.
    Returns the optimal policy, value function, number of iterations.
    """

    n_states = env.grid_size * env.grid_size * 2
    policy = np.full(n_states, -1, dtype=int)
    V = np.zeros(n_states, dtype=np.float64)

    # --- Initialization ---
    for s in range(n_states):
        state = env.index_to_state(s)
        if state[2] == 1:  # terminal
            continue
        for a in range(env.action_space.n):
            if env.get_transitions_at_time(state, a):
                policy[s] = a
                break

    stable, outer_iters, last_eval_iters = False, 0, 0
    start_time = time.time()

    # --- Policy Iteration loop ---
    while not stable:
        outer_iters += 1

        # Policy Evaluation
        while True:
            delta = 0.0
            last_eval_iters += 1
            for s in range(n_states):
                if policy[s] == -1:
                    continue
                state = env.index_to_state(s)
                action = int(policy[s])
                transitions = env.get_transitions_at_time(state, action)
                v_new = 0.0
                for prob, ns in transitions:
                    reward = env._get_reward((ns[0], ns[1]), action, (state[0], state[1]))
                    v_new += prob * (reward + gamma * V[env.state_to_index(ns)])
                delta = max(delta, abs(V[s] - v_new))
                V[s] = v_new
            if delta < theta:
                break

        # Policy Improvement
        new_policy = policy.copy()
        for s in range(n_states):
            state = env.index_to_state(s)
            if state[2] == 1:
                continue
            action_values = []
            for a in range(env.action_space.n):
                transitions = env.get_transitions_at_time(state, a)
                if not transitions:
                    action_values.append(-math.inf)
                    continue
                q_sa = 0.0
                for prob, ns in transitions:
                    reward = env._get_reward((ns[0], ns[1]), a, (state[0], state[1]))
                    q_sa += prob * (reward + gamma * V[env.state_to_index(ns)])
                action_values.append(q_sa)
            new_policy[s] = int(np.argmax(action_values))

        stable = np.array_equal(policy, new_policy)
        policy = new_policy

    elapsed = time.time() - start_time
    return policy, V, outer_iters, last_eval_iters, elapsed


def evaluate_policy(env, policy, episodes=20):
    """Run episodes using the learned policy."""
    rewards = []
    seeds = [100 + i for i in range(episodes)]
    for i in range(episodes):
        obs, _ = env.reset(seed=seeds[i])
        done, total_reward, steps = False, 0.0, 0
        while not done and steps < 1000:
            s_idx = env.state_to_index(obs)
            act = int(policy[s_idx]) if policy[s_idx] != -1 else 0
            obs, r, done, _, _ = env.step(act)
            total_reward += r
            steps += 1
        rewards.append(total_reward)
    return rewards


def main():
    env = FootballSkillsEnv(render_mode="gif", degrade_pitch=False)

    # --- Run Policy Iteration ---
    policy, V, iters, eval_sweeps, runtime = policy_iteration(env, gamma=0.95, theta=1e-6)

    # --- Evaluate learned policy ---
    rewards = evaluate_policy(env, policy, episodes=20)
    print(f"\nEvaluation mean={np.mean(rewards):.4f}, std={np.std(rewards):.4f}")
    print(f"All rewards: {rewards}")

    # --- Generate GIF rollout ---
    steps, total_reward = env.get_gif(policy.tolist(), seed=1234, filename="pi_policy.gif")

    # --- Report ---
    print("\n--- Policy Iteration Results ---")
    print(f"Policy outer iterations: {iters}")
    print(f"Last evaluation sweeps: {eval_sweeps}")
    print(f"Runtime: {runtime:.2f}s")
    print(f"GIF rollout: steps={steps}, total_reward={total_reward}")
    print(f"GIF saved to: {env.default_save_path}/pi_policy.gif")


if __name__ == "__main__":
    main()

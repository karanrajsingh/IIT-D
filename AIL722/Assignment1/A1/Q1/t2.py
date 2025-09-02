from env import FootballSkillsEnv
import numpy as np
import math
import time


def policy_iteration(envr=FootballSkillsEnv):
    """
    Implements the Policy Iteration algorithm to find the optimal policy for the 
    Football Skills Environment.

    Args:
        envr (class, optional): Environment class to instantiate. Defaults to FootballSkillsEnv.

    Returns:
        tuple: (optimal_policy, value_function, num_iterations)
            - optimal_policy (numpy.ndarray): Maps state indices to optimal actions
            - value_function (numpy.ndarray): Value of each state under optimal policy  
            - num_iterations (int): Number of iterations until convergence
    """

    # --- Setup ---
    env = envr(render_mode=None, degrade_pitch=False)
    gamma, theta = 0.95, 1e-6

    n_states = env.grid_size * env.grid_size * 2
    n_actions = env.action_space.n

    # --- Initialization ---
    policy = np.full(n_states, -1, dtype=int)
    V = np.zeros(n_states, dtype=np.float64)

    for s in range(n_states):
        state = env.index_to_state(s)
        if env._is_terminal(state):
            continue
        for a in range(n_actions):
            if env.get_transitions_at_time(state, a):
                policy[s] = a
                break

    stable, num_iterations = False, 0
    start_time = time.time()

    # --- Policy Iteration loop ---
    while not stable:
        num_iterations += 1

        # --- Policy Evaluation ---
        while True:
            delta = 0.0
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

        # --- Policy Improvement ---
        new_policy = policy.copy()
        for s in range(n_states):
            state = env.index_to_state(s)
            if env._is_terminal(state):
                continue
            action_values = []
            for a in range(n_actions):
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
    print(f"Policy Iteration converged in {num_iterations} iterations, time={elapsed:.2f}s")
    return policy, V, num_iterations


def value_iteration(envr=FootballSkillsEnv):
    """Placeholder for Value Iteration (to be implemented)."""
    pass


if __name__ == "__main__":
    env_normal = FootballSkillsEnv(render_mode="gif", degrade_pitch=False)
    optimal_policy, value_function, num_iterations = policy_iteration(env_normal.__class__)

    print("Value function:", value_function)
    print("Policy function:", optimal_policy)
    print(f"Converged in {num_iterations} iterations")

    # Generate GIF rollout
    env_normal.get_gif(optimal_policy.tolist(), seed=1234, filename="pi_policy.gif")

import numpy as np
from or_gym.envs.classic_or.knapsack import OnlineKnapsackEnv


class ValueIterationOnlineKnapsack:
    def __init__(self, env, gamma=0.95, epsilon=1e-6):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        # Environment properties
        self.max_weight = env.max_weight
        self.weights = env.item_weights
        self.values = env.item_values
        self.probs = env.item_probs
        self.N = len(self.weights)
        self.T = env.step_limit
        # Initialize value function and policy
        self.V = None
        self.policy = None
        # Count iterations
        self.value_iterations = 0

    def value_iteration(self):
        """Time-dependent Value Iteration with Δ-convergence."""
        W, N, T = self.max_weight, self.N, self.T
        p, weights, values, gamma = self.probs, self.weights, self.values, self.gamma

        V = np.zeros((T+1, W+1, N))
        policy = np.zeros((T, W+1, N), dtype=int)

        self.value_iterations = 0
        while True:
            delta = 0
            self.value_iterations += 1
            # one sweep over all states
            for t in range(T-1, -1, -1):
                S = V[t+1].dot(p)  # expected value for next items
                for w in range(W+1):
                    if w == W:
                        continue
                    for i in range(N):
                        v_old = V[t, w, i]
                        V_rej = gamma * S[w]  # reject action
                        if w + weights[i] <= W:
                            Q_acc = values[i] + gamma * S[w + weights[i]]
                            V[t, w, i] = max(V_rej, Q_acc)
                            policy[t, w, i] = int(Q_acc >= V_rej)
                        else:
                            V[t, w, i] = V_rej
                            policy[t, w, i] = 0
                        delta = max(delta, abs(v_old - V[t, w, i]))
            if delta < self.epsilon:
                print(f"Value Iteration converged in {self.value_iterations} sweeps (Δ={delta:.6f})")
                break

        self.V = V
        self.policy = policy
        return self.value_iterations

    def get_action(self, state, t=0):
        """Return action according to learned policy at time t."""
        if isinstance(state, dict):
            arr = state['state']
        else:
            arr = state
        current_weight = int(arr[0])
        item_idx = int(arr[1])
        return int(self.policy[t, current_weight, item_idx])


class PolicyIterationOnlineKnapsack:
    def __init__(self, env, gamma=0.95, epsilon=1e-6):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        # Environment properties
        self.max_weight = env.max_weight
        self.weights = env.item_weights
        self.values = env.item_values
        self.probs = env.item_probs
        self.N = len(self.weights)
        self.T = env.step_limit
        # Initialize policy to reject all items
        self.policy = np.zeros((self.T, self.max_weight+1, self.N), dtype=int)
        # Initialize value function array
        self.V = np.zeros((self.T+1, self.max_weight+1, self.N))
        # Count iterations
        self.policy_iterations = 0
        self.evaluation_sweeps = 0

    def policy_evaluation(self):
        """Evaluate current policy until Δ < ε. Count sweeps."""
        W, N, T = self.max_weight, self.N, self.T
        p, weights, values, gamma = self.probs, self.weights, self.values, self.gamma
        policy = self.policy
        V = np.zeros_like(self.V)

        sweeps = 0
        while True:
            delta = 0
            sweeps += 1
            for t in range(T-1, -1, -1):
                S = V[t+1].dot(p)
                for w in range(W+1):
                    if w == W:
                        continue
                    for i in range(N):
                        v_old = V[t, w, i]
                        if policy[t, w, i] == 0:  # reject
                            V[t, w, i] = gamma * S[w]
                        else:  # accept
                            w_new = w + weights[i]
                            if w_new <= W:
                                V[t, w, i] = values[i] + gamma * S[w_new]
                            else:
                                V[t, w, i] = 0
                        delta = max(delta, abs(v_old - V[t, w, i]))
            if delta < self.epsilon:
                break

        self.evaluation_sweeps += sweeps
        self.V = V
        print(f"Policy Evaluation converged in {sweeps} sweeps (Δ={delta:.6f})")

    def policy_improvement(self):
        """Improve policy. Return True if stable."""
        W, N, T = self.max_weight, self.N, self.T
        p, weights, values, gamma = self.probs, self.weights, self.values, self.gamma
        V = self.V

        policy_stable = True
        for t in range(T):
            S = V[t+1].dot(p)
            for w in range(W+1):
                if w == W:
                    for i in range(N):
                        if self.policy[t, w, i] != 0:
                            policy_stable = False
                            self.policy[t, w, i] = 0
                    continue
                V_rej = gamma * S[w]
                for i in range(N):
                    if w + weights[i] <= W:
                        Q_acc = values[i] + gamma * S[w + weights[i]]
                        best_action = 1 if Q_acc >= V_rej else 0
                    else:
                        best_action = 0
                    if self.policy[t, w, i] != best_action:
                        policy_stable = False
                        self.policy[t, w, i] = best_action
        return policy_stable

    def run_policy_iteration(self):
        """Run policy iteration until policy stabilizes. Count iterations."""
        self.policy_iterations = 0
        self.evaluation_sweeps = 0
        while True:
            self.policy_iterations += 1
            self.policy_evaluation()
            stable = self.policy_improvement()
            if stable:
                print(f"Policy Iteration converged in {self.policy_iterations} iterations "
                      f"with {self.evaluation_sweeps} total evaluation sweeps")
                break
        return self.policy_iterations, self.evaluation_sweeps

    def get_action(self, state, t=0):
        """Return action from current policy at time t."""
        if isinstance(state, dict):
            arr = state['state']
        else:
            arr = state
        current_weight = int(arr[0])
        item_idx = int(arr[1])
        return int(self.policy[t, current_weight, item_idx])


if __name__ == "__main__":
    env = OnlineKnapsackEnv()
    state = env._RESET()
    print(f"Initial State for OnlineKnapsackEnv {state}")

    # ---------- Run Value Iteration ----------
    print("\nRunning Value Iteration...")
    vi_agent = ValueIterationOnlineKnapsack(env)
    vi_iters = vi_agent.value_iteration()

    total_reward = 0
    done = False
    t = 0
    state = env._RESET()
    while not done:
        action = vi_agent.get_action(state, t)
        next_state, reward, done, info = env._STEP(action)
        total_reward += reward
        print(f"[VI] Step {t}: Action: {action} | Reward: {reward} | Next State: {next_state} | Done: {done}")
        state = next_state
        t += 1
    print(f"[VI] Episode finished. Total reward: {total_reward}")
    print(f"[VI] Total value iterations = {vi_iters}\n")

    # ---------- Run Policy Iteration ----------
    print("Running Policy Iteration...")
    pi_agent = PolicyIterationOnlineKnapsack(env)
    pi_iters, pe_sweeps = pi_agent.run_policy_iteration()

    total_reward = 0
    done = False
    t = 0
    state = env._RESET()
    while not done:
        action = pi_agent.get_action(state, t)
        next_state, reward, done, info = env._STEP(action)
        total_reward += reward
        print(f"[PI] Step {t}: Action: {action} | Reward: {reward} | Next State: {next_state} | Done: {done}")
        state = next_state
        t += 1
    print(f"[PI] Episode finished. Total reward: {total_reward}")
    print(f"[PI] Total policy iterations = {pi_iters}, total evaluation sweeps = {pe_sweeps}")

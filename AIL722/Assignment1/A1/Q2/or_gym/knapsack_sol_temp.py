import numpy as np
from or_gym.envs.classic_or.knapsack import OnlineKnapsackEnv


class ValueIterationOnlineKnapsack:
    def __init__(self, env, gamma=0.95, epsilon=1e-6):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon

        self.max_weight = env.max_weight
        self.item_weights = env.item_weights
        self.item_values = env.item_values
        self.item_probs = env.item_probs
        self.num_items = len(self.item_weights)
        self.horizon = env.step_limit

        self.value_function = np.zeros((self.horizon+1, self.max_weight+1, self.num_items))
        self.policy_function = np.zeros((self.horizon, self.max_weight+1, self.num_items), dtype=int)

        self.value_iterations = 0

    def value_iteration(self):


        self.value_iterations = 0
        while True:
            delta = 0
            self.value_iterations += 1
            # one sweep over all states
            for time_step in reversed(range(self.horizon)):
                v_next = np.dot(self.value_function[time_step+1], self.item_probs)  # expected value for next items
                for curr_weight in range(self.max_weight+1):
                    if curr_weight == self.max_weight:
                        self.value_function[time_step, curr_weight, :] = 0.0
                        self.policy_function[time_step, curr_weight, :] = 0
                        continue
                    for item_idx in range(self.num_items):
                        v_old = self.value_function[time_step, curr_weight, item_idx]
                        q0 = self.gamma*v_next[curr_weight]  # reject action
                        if curr_weight + self.item_weights[item_idx] <= self.max_weight:
                            q1 = self.item_values[item_idx] + self.gamma*v_next[curr_weight + self.item_weights[item_idx]]
                            self.value_function[time_step, curr_weight, item_idx] = max(q0, q1)
                            if q1 >= q0:
                                self.policy_function[time_step, curr_weight, item_idx] = 1
                            else:
                                self.policy_function[time_step, curr_weight, item_idx] = 0
                        else:
                            self.value_function[time_step, curr_weight, item_idx] = q0
                            self.policy_function[time_step, curr_weight, item_idx] = 0
                        delta = max(delta, abs(v_old - self.value_function[time_step, curr_weight, item_idx]))
            if delta < self.epsilon:
                print(f"Value Iteration converged in {self.value_iterations} sweeps (Δ={delta:.6f})")
                break

    def get_action(self, state, time_step=0):
        arr = state['state']
        current_weight, item_idx = int(arr[0]), int(arr[1])
        return int(self.policy_function[time_step, current_weight, item_idx])


class PolicyIterationOnlineKnapsack:
    def __init__(self, env, gamma=0.95, epsilon=1e-6):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon

        self.item_values = env.item_values
        self.item_weights = env.item_weights
        self.item_probs = env.item_probs
        self.num_items = len(self.item_weights)
        self.horizon = env.step_limit
        self.max_weight = env.max_weight

        self.value_function = np.zeros((self.horizon+1, self.max_weight+1, self.num_items))
        self.policy_function = np.zeros((self.horizon, self.max_weight+1, self.num_items), dtype=int)

        self.policy_iterations = 0
        self.evaluation_iters = 0

    def policy_evaluation(self):
        self.evaluation_iters = 0
        while True:
            delta = 0
            self.evaluation_iters += 1
            for time_step in reversed(range(self.horizon)):
                v_next = np.dot(self.value_function[time_step+1], self.item_probs)  # expected value for next items
                for curr_weight in range(self.max_weight+1):
                    if curr_weight == self.max_weight:
                        self.value_function[time_step, curr_weight, :] = 0.0
                        self.policy_function[time_step, curr_weight, :] = 0
                        continue
                    for item_idx in range(self.num_items):
                        v_old = self.value_function[time_step, curr_weight, item_idx]
                        if self.policy_function[time_step, curr_weight, item_idx] == 0:  # reject
                            self.value_function[time_step, curr_weight, item_idx] = self.gamma*v_next[curr_weight]
                        else:  # accept
                            if curr_weight + self.item_weights[item_idx] <= self.max_weight:
                                self.value_function[time_step, curr_weight, item_idx] = self.item_values[item_idx] + self.gamma*v_next[curr_weight + self.item_weights[item_idx]]
                            else:
                                self.value_function[time_step, curr_weight, item_idx] = 0
                        delta = max(delta, abs(v_old - self.value_function[time_step, curr_weight, item_idx]))
            if delta < self.epsilon:
                break

        print(f"Policy Evaluation converged in {self.evaluation_iters} sweeps (Δ={delta:.6f})")

    def policy_improvement(self):

        policy_stable = True
        for time_step in range(self.horizon):
            v_next = np.dot(self.value_function[time_step+1], self.item_probs)  # expected value for next items
            for curr_weight in range(self.max_weight+1):
                if curr_weight == self.max_weight:
                    for item_idx in range(self.num_items):
                        if self.policy_function[time_step, curr_weight, item_idx] != 0:
                            policy_stable = False
                            self.policy_function[time_step, curr_weight, item_idx] = 0
                    continue
                q0 = self.gamma*v_next[curr_weight]
                for item_idx in range(self.num_items):
                    if curr_weight + self.item_weights[item_idx] <= self.max_weight:
                        q1 = self.item_values[item_idx] + self.gamma*v_next[curr_weight + self.item_weights[item_idx]]
                        if q1 >= q0:
                            max_action = 1
                        else:
                            max_action = 0
                    else:
                        max_action = 0
                    if self.policy_function[time_step, curr_weight, item_idx] != max_action:
                        policy_stable = False
                        self.policy_function[time_step, curr_weight, item_idx] = max_action
        return policy_stable

    def run_policy_iteration(self):
        while True:
            self.policy_iterations += 1
            self.policy_evaluation()
            stable = self.policy_improvement()
            if stable:
                print(f"Policy Iteration converged in {self.policy_iterations} iterations "
                      f"with {self.evaluation_iters} total evaluation sweeps")
                break
        return self.policy_iterations, self.evaluation_iters

    def get_action(self, state, time_step=0):
        arr = state['state']
        current_weight, item_idx = int(arr[0]), int(arr[1])
        return int(self.policy_function[time_step, current_weight, item_idx])


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

import numpy as np
from or_gym.envs.classic_or.knapsack import OnlineKnapsackEnv
import matplotlib.pyplot as plt
from scipy.stats import mode
import imageio
import os


import numpy as np

class ValueIterationOnlineKnapsack:
    def __init__(self, env, gamma=0.95, epsilon=1e-4):
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
        self.policy_fucntion = np.zeros((self.horizon, self.max_weight+1, self.num_items), dtype=int)

        self.iterations = 0


    def value_iteration(self):

        while True:
            delta = 0
            self.iterations += 1
            # one sweep over all states
            for time_step in reversed(range(self.horizon)):
                v_next = np.dot(self.value_function[time_step+1], self.item_probs)  # expected value for next items
                for curr_weight in range(self.max_weight+1):
                    if curr_weight == self.max_weight:
                        self.value_function[time_step, curr_weight, :] = 0.0
                        self.policy_fucntion[time_step, curr_weight, :] = 0
                        continue
                    for item_idx in range(self.num_items):
                        v_old = self.value_function[time_step, curr_weight, item_idx]
                        q0 = self.gamma*v_next[curr_weight]  # reject action
                        if curr_weight + self.item_weights[item_idx] <= self.max_weight:
                            q1 = self.item_values[item_idx] + self.gamma*v_next[curr_weight + self.item_weights[item_idx]]
                            self.value_function[time_step, curr_weight, item_idx] = max(q0, q1)
                            if q1 >= q0:
                                self.policy_fucntion[time_step, curr_weight, item_idx] = 1
                            else:
                                self.policy_fucntion[time_step, curr_weight, item_idx] = 0
                        else:
                            self.value_function[time_step, curr_weight, item_idx] = q0
                            self.policy_fucntion[time_step, curr_weight, item_idx] = 0
                        delta = max(delta, abs(v_old - self.value_function[time_step, curr_weight, item_idx]))
            if delta < self.epsilon:
                print(f"Value Iteration converged in {self.iterations} sweeps (delta={delta:.6f})")
                break
    
    def get_action(self, state, t=0):
        arr = state['state']
        current_weight, item_idx = int(arr[0]), int(arr[1])
        return int(self.policy_fucntion[t, current_weight, item_idx])


class PolicyIterationOnlineKnapsack:
    def __init__(self, env, gamma=0.95, epsilon=1e-4, eval_iterations=1000):
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

        print(f"Policy Evaluation converged in {self.evaluation_iters} sweeps (delta={delta:.6f})")

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

def evaluate_and_record_policy(env, agent, seed):
    env.set_seed(seed)
    state = env._RESET()
    total_reward = 0
    done = False
    t = 0
    rewards_over_time = []
    while not done:
        action = agent.get_action(state, t)
        next_state, reward, done, _ = env._STEP(action)
        total_reward += reward
        rewards_over_time.append(total_reward)
        state = next_state
        t += 1
    return total_reward, rewards_over_time

def evaluate_vi():
    seeds = [38, 490, 743, 1024, 49674]
    results = []
    trajectories = {}

    for seed in seeds:
        print(f"Running Value Iteration with seed {seed}")
        env = OnlineKnapsackEnv()
        env.set_seed(seed)

        # Train via Value Iteration
        vi = ValueIterationOnlineKnapsack(env)
        vi.value_iteration()

        # Evaluate policy for this seed
        reward, rewards_over_time = evaluate_and_record_policy(env, vi, seed)
        results.append(reward)
        trajectories[seed] = rewards_over_time
        print(f"Seed {seed},  Total knapsack value: {reward}")

    for seed, value in zip(seeds, results):
        print(f"Seed {seed}: {value}")
    print(f"Mean value: {np.mean(results):.2f}, Std Dev: {np.std(results):.2f}")

    # Plot results
    plt.figure(figsize=(10, 6))
    for seed, rewards in trajectories.items():
        plt.plot(range(1, len(rewards) + 1), rewards, label=f"Seed {seed}")
    plt.xlabel("Time step")
    plt.ylabel("Knapsack Value")
    plt.title("Knapsack Value Progression During Evaluation (5 Seeds)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def evaluate_pi():
    seeds = [38, 490, 743, 1024, 49674]
    results = []
    trajectories = {}

    for seed in seeds:
        print(f"Running Policy Iteration with seed {seed}")
        env = OnlineKnapsackEnv()
        env.set_seed(seed)

        # Train via Value Iteration
        pi = PolicyIterationOnlineKnapsack(env)
        pi.run_policy_iteration()

        # Evaluate policy for this seed
        reward, rewards_over_time = evaluate_and_record_policy(env, pi, seed)
        results.append(reward)
        trajectories[seed] = rewards_over_time
        print(f"Seed {seed}, Total knapsack value: {reward}")

    for seed, value in zip(seeds, results):
        print(f"Seed {seed}: {value}")
    print(f"Mean value: {np.mean(results):.2f}, Std Dev: {np.std(results):.2f}")

    # Plot results
    plt.figure(figsize=(10, 6))
    for seed, rewards in trajectories.items():
        plt.plot(range(1, len(rewards) + 1), rewards, label=f"Seed {seed}")
    plt.xlabel("Time Step")
    plt.ylabel("Knapsack Value")
    plt.title("Knapsack Value Progression During Evaluation (5 Seeds)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_value_heatmap_frame(agent, env, time_step, sort_by="weight", save_path="frame.png"):
    item_weights = env.item_weights
    item_values = env.item_values
    ratios = item_values / np.maximum(item_weights, 1)

    if sort_by == "weight":
        sorted_idx = np.argsort(item_weights)
        title = "Sorted by Item Weight"
    elif sort_by == "value":
        sorted_idx = np.argsort(item_values)
        title = "Sorted by Item Value"
    elif sort_by == "ratio":
        sorted_idx = np.argsort(ratios)
        title = "Sorted by Value/Weight Ratio"

    vf = agent.value_function[time_step]
    max_weight = env.max_weight

    plt.figure(figsize=(7, 5))
    plt.imshow(
        vf[:, sorted_idx].T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[0, max_weight, 0, len(sorted_idx)]
    )
    plt.colorbar(label="Value")
    plt.title(f"{title}\nTime Step {time_step}")
    plt.xlabel("Current Knapsack Weight")
    plt.ylabel("Items")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def create_gif(agent, env, sort_by="weight", gif_name="value_function.gif"):
    frames = []
    temp_dir = "frames"
    os.makedirs(temp_dir, exist_ok=True)

    for t in range(env.step_limit):
        frame_path = os.path.join(temp_dir, f"frame_{t:03d}.png")
        plot_value_heatmap_frame(agent, env, t, sort_by=sort_by, save_path=frame_path)
        frames.append(imageio.imread(frame_path))

    imageio.mimsave(gif_name, frames, duration=0.3)  # 0.3s per frame
    print(f"GIF saved as {gif_name}")

def vi_gif():
    seed = 42
    env = OnlineKnapsackEnv()  # you can also try 10 or 500
    env.set_seed(seed)

    vi_agent = ValueIterationOnlineKnapsack(env)
    vi_agent.value_iteration()

    # Create GIF for each sorting option
    create_gif(vi_agent, env, sort_by="weight", gif_name="50_50_value_function_weight.gif")
    create_gif(vi_agent, env, sort_by="value", gif_name="50_50_value_function_value.gif")
    create_gif(vi_agent, env, sort_by="ratio", gif_name="50_50_value_function_ratio.gif")


def pi_gif():
    seed = 42
    env = OnlineKnapsackEnv()
    env.set_seed(seed)

    vi_agent = PolicyIterationOnlineKnapsack(env)
    vi_agent.run_policy_iteration()

    # Create GIF for each sorting option
    create_gif(vi_agent, env, sort_by="weight", gif_name="pol_value_function_weight.gif")
    create_gif(vi_agent, env, sort_by="value", gif_name="pol_value_function_value.gif")
    create_gif(vi_agent, env, sort_by="ratio", gif_name="pol_value_function_ratio.gif")
  

if __name__=="__main__":
    env=OnlineKnapsackEnv()
    state=env._RESET()
    print(f"State for OnlineKnapsackEnv {state}")

    value_iteration = ValueIterationOnlineKnapsack(env)
    value_iteration.value_iteration()

    total_reward = 0
    done = False
    t = 0
    state = env._RESET()
    while not done:
        action = value_iteration.get_action(state, t)
        next_state, reward, done, info = env._STEP(action)
        total_reward += reward
        print(f"[VI] Step {t}: Action: {action} | Reward: {reward} | Next State: {next_state} | Done: {done}")
        state = next_state
        t += 1
    print(f"[VI] Episode finished. Total reward: {total_reward}")
    print(f"[VI] Total value iterations = {value_iteration.iterations}\n")


    # ###################### Random sampling #####################################################

    # print("Starting Online Knapsack Simulation")
    # print("Initial state:", state)

    # # Run simulation until episode ends
    # while not done:
    #     # Choose an action randomly: 0 (reject) or 1 (accept)
    #     action = env.sample_action()
    #     print(action)
    #     # Take a step in the environment
    #     next_state, reward, done, info = env._STEP(action)
    #     total_reward += reward
    #     print(f"Step: {env.step_counter} Action: {'Accept' if action == 1 else 'Reject'} | Reward: {reward} | Next state: {next_state} | Done: {done}")
    #     state = next_state
    #     # exit()
    # print(f"Episode finished. Total reward: {total_reward}")

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

    evaluate_vi()
    vi_gif()

    evaluate_pi()
    pi_gif()


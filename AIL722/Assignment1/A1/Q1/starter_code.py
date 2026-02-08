from env import FootballSkillsEnv
import numpy as np
from collections import deque

gamma = 0.95
# gamma = 0.3
# gamma = 0.5

def policy_iteration(envr=FootballSkillsEnv):
    '''
    Implements the Policy Iteration algorithm to find the optimal policy for the 
    Football Skills Environment.
    
    Args:
        envr (class, optional): Environment class to instantiate. Defaults to FootballSkillsEnv.
    
    Returns:
        tuple: (optimal_policy, value_function, num_iterations)
            - optimal_policy (dict): Maps state indices to optimal actions
            - value_function (numpy.ndarray): Value of each state under optimal policy  
            - num_iterations (int): Number of iterations until convergence
    
    Algorithm:
    1. Initialize arbitrary policy and value function
    2. Policy Evaluation: Iteratively update value function until convergence
    3. Policy Improvement: Update policy greedily based on current values  
    4. Repeat steps 2-3 until policy converges
    
    Key Environment Methods to Use:
    - env.state_to_index(state_tuple): Converts (x, y, has_shot) tuple to integer index
    - env.index_to_state(index): Converts integer index back to (x, y, has_shot) tuple
    - env.get_transitions_at_time(state, action, time_step=None): Default method for accessing transitions.
    - env._is_terminal(state): Check if state is terminal (has_shot=True)
    - env._get_reward(ball_pos, action, player_pos): Get reward for transition
    - env.reset(seed=None): Reset environment to initial state, returns (observation, info)
    - env.step(action): Execute action, returns (obs, reward, done, truncated, info)
    - env.get_gif(policy, seed=20, filename="output.gif"): Generate GIF visualization 
      of policy execution from given seed
    
    Key Env Variables Notes:
    - env.observation_space.n: Total number of states (use env.grid_size^2 * 2)
    - env.action_space.n: Total number of actions (7 actions: 4 movement + 3 shooting)
    - env.grid_size: Total number of rows in the grid
    '''

    convergence = 1e-6
    total_states = envr.grid_size*envr.grid_size*2
    total_actions = envr.action_space.n

    # Step 1: Initialization

    # Initializing value function as zero
    value_function = np.zeros(total_states)
    
    # Initializing policy function to random valid actions
    policy_function = np.random.randint(0, total_actions, size=total_states)

    transition_calls = 0
    
    for state_index in range(total_states):
        state = envr.index_to_state(state_index)
        if envr._is_terminal(state):
            policy_function[state_index] = -1
            continue
        for action in range(total_actions):
            if envr.get_transitions_at_time(state, action):
                transition_calls += 1
                policy_function[state_index] = action
                break

    policy_stable, num_iterations = False, 0

    # Core policy iteration logic
    while not policy_stable:
        num_iterations += 1

        # Step 2: Policy Evaluation
        while True:
            delta = 0.0
            for curr_state_index in range(total_states):
                # Skipping terminal states, else negative rewards would accrue
                if policy_function[curr_state_index] == -1:
                    continue
                v_old = value_function[curr_state_index]
                action = policy_function[curr_state_index]
                curr_state_tuple = envr.index_to_state(curr_state_index)
                transitions = envr.get_transitions_at_time(curr_state_tuple, action)
                transition_calls += 1
                v_new = 0.0
                if transitions:
                    # Looping over all next states through the chance node                    
                    for transition in transitions:
                        probability, next_state_tuple = transition
                        next_state_index = envr.state_to_index(next_state_tuple)
                        v_next = value_function[next_state_index]
                        # Ball position is always next state
                        # Player position at shot is previous state always
                        ball_pos = (next_state_tuple[0], next_state_tuple[1])
                        player_pos = (curr_state_tuple[0], curr_state_tuple[1])
                        reward = envr._get_reward(ball_pos, action, player_pos)
                        v_new += probability*(reward + gamma*v_next)
                    value_function[curr_state_index] = v_new
                delta = max(delta, abs(v_old - value_function[curr_state_index])) 

            if delta < convergence:
                break

        # Step 3: Policy Improvement
        new_policy_function = policy_function.copy()
        for curr_state_index in range(total_states):
            curr_state_tuple = envr.index_to_state(curr_state_index)
            if envr._is_terminal(curr_state_tuple):
                continue
            state_action_to_value = {}
            for action in range(total_actions):
                transitions = envr.get_transitions_at_time(curr_state_tuple, action)
                transition_calls += 1
                if transitions:
                    q_val = 0.0
                    for transition in transitions:
                        probability, next_state_tuple = transition
                        next_state_index = envr.state_to_index(next_state_tuple)
                        v_next = value_function[next_state_index]
                        ball_pos = (next_state_tuple[0], next_state_tuple[1])
                        player_pos = (curr_state_tuple[0], curr_state_tuple[1])
                        reward = envr._get_reward(ball_pos, action, player_pos)
                        q_val += probability*(reward + gamma*v_next)
                    state_action_to_value[action] = q_val
            if state_action_to_value:
                max_action = max(state_action_to_value, key=state_action_to_value.get)
                new_policy_function[curr_state_index] = max_action

        policy_stable = np.array_equal(policy_function, new_policy_function)
        policy_function = new_policy_function


    return dict(enumerate(policy_function)), value_function, num_iterations, transition_calls

def value_iteration(envr=FootballSkillsEnv):

    convergence = 1e-6
    total_states = envr.grid_size*envr.grid_size*2
    total_actions = envr.action_space.n

    # Step 1: Initialization
    value_function = np.zeros(total_states)
    policy_function = np.random.randint(0, total_actions, size=total_states)

    transition_calls = 0

    for state_index in range(total_states):
        state = envr.index_to_state(state_index)
        if envr._is_terminal(state):
            policy_function[state_index] = -1
            continue
        for action in range(total_actions):
            if envr.get_transitions_at_time(state, action):
                transition_calls += 1
                policy_function[state_index] = action
                break

    num_iterations = 0

    # Core value iteration logic
    while True:
        num_iterations += 1
        delta = 0.0
        new_policy_function = policy_function.copy()

        for curr_state_index in range(total_states):
            curr_state_tuple = envr.index_to_state(curr_state_index)

            # Skip terminal states
            if envr._is_terminal(curr_state_tuple):
                continue

            v_old = value_function[curr_state_index]

            # Evaluate all possible actions
            state_action_to_value = {}
            for action in range(total_actions):
                transitions = envr.get_transitions_at_time(curr_state_tuple, action)
                transition_calls += 1
                if transitions:
                    q_val = 0.0
                    for transition in transitions:
                        probability, next_state_tuple = transition
                        next_state_index = envr.state_to_index(next_state_tuple)
                        v_next = value_function[next_state_index]
                        ball_pos = (next_state_tuple[0], next_state_tuple[1])
                        player_pos = (curr_state_tuple[0], curr_state_tuple[1])
                        reward = envr._get_reward(ball_pos, action, player_pos)
                        q_val += probability * (reward + gamma * v_next)
                    state_action_to_value[action] = q_val

            if state_action_to_value:
                max_action = max(state_action_to_value, key=state_action_to_value.get)
                max_q_val = state_action_to_value[max_action]
                value_function[curr_state_index] = max_q_val
                new_policy_function[curr_state_index] = max_action

            delta = max(delta, abs(v_old - value_function[curr_state_index]))

        policy_function = new_policy_function

        if delta < convergence:
            break

    return dict(enumerate(policy_function)), value_function, num_iterations, transition_calls


def value_iteration_nonstationary(envr=FootballSkillsEnv):
    '''
    Implements Value Iteration for the non-stationary Football Skills Environment.
    Similar to stationary case, but policy is indexed by (time, state).
    '''

    convergence = 1e-6
    horizon = 40
    total_states = envr.grid_size * envr.grid_size * 2
    total_actions = envr.action_space.n

    # Step 1: Initialization
    value_function = np.zeros((horizon + 1, total_states))
    policy_function = {}

    transition_calls = 0

    for t in range(horizon):
        for state_index in range(total_states):
            state = envr.index_to_state(state_index)
            if envr._is_terminal(state):
                policy_function[(t, state_index)] = -1
                continue
            # choose the first valid action
            for action in range(total_actions):
                if envr.get_transitions_at_time(state, action, time_step=t):
                    transition_calls += 1
                    policy_function[(t, state_index)] = action
                    break
            else:
                # no valid action found
                policy_function[(t, state_index)] = -1

    num_iterations = 0

    # Core value iteration logic
    while True:
        num_iterations += 1
        delta = 0.0
        new_policy_function = policy_function.copy()

        for t in range(horizon):
            for state_index in range(total_states):
                state_tuple = envr.index_to_state(state_index)

                if envr._is_terminal(state_tuple):
                    continue

                v_old = value_function[t, state_index]
                state_action_to_value = {}

                # Evaluate all possible actions
                for action in range(total_actions):
                    transitions = envr.get_transitions_at_time(state_tuple, action, time_step=t)
                    transition_calls += 1
                    if transitions:
                        q_val = 0.0
                        for probability, next_state_tuple in transitions:
                            next_state_index = envr.state_to_index(next_state_tuple)
                            v_next = value_function[t + 1, next_state_index]
                            ball_pos = (next_state_tuple[0], next_state_tuple[1])
                            player_pos = (state_tuple[0], state_tuple[1])
                            reward = envr._get_reward(ball_pos, action, player_pos)
                            q_val += probability * (reward + gamma * v_next)
                        state_action_to_value[action] = q_val

                if state_action_to_value:
                    max_action = max(state_action_to_value, key=state_action_to_value.get)
                    max_q_val = state_action_to_value[max_action]
                    value_function[t, state_index] = max_q_val
                    new_policy_function[(t, state_index)] = max_action

                delta = max(delta, abs(v_old - value_function[t, state_index]))

        policy_function = new_policy_function

        if delta < convergence:
            break

    return policy_function, value_function, num_iterations, transition_calls


def value_iteration_bfs(envr=FootballSkillsEnv):
    """
    BFS-based Value Iteration.
    Only expands and updates states reachable from the start state.
    Returns (policy, value_function, num_iterations, transition_calls).
    """

    convergence = 1e-6
    total_states = envr.grid_size * envr.grid_size * 2
    total_actions = envr.action_space.n

    # Step 1: Initialization
    value_function = np.zeros(total_states)
    policy_function = np.full(total_states, -1)  # invalid by default
    transition_calls = 0

    # Step 2: BFS traversal to collect reachable states
    visited = [False] * total_states
    reachable_states = []

    start_state, _ = envr.reset(seed=0)
    start_index = envr.state_to_index(tuple(start_state))
    q = deque([start_index])
    visited[start_index] = True

    while q:
        s_idx = q.popleft()
        reachable_states.append(s_idx)
        s_tuple = envr.index_to_state(s_idx)

        if envr._is_terminal(s_tuple):
            continue

        for action in range(total_actions):
            transitions = envr.get_transitions_at_time(s_tuple, action)
            transition_calls += 1
            if not transitions:
                continue
            for _, next_state_tuple in transitions:
                next_state_idx = envr.state_to_index(next_state_tuple)
                if not visited[next_state_idx]:
                    visited[next_state_idx] = True
                    q.append(next_state_idx)

    # Step 3: Value Iteration loop
    num_iterations = 0
    while True:
        num_iterations += 1
        delta = 0.0
        new_policy_function = policy_function.copy()

        for curr_state_index in reachable_states:
            curr_state_tuple = envr.index_to_state(curr_state_index)
            if envr._is_terminal(curr_state_tuple):
                continue

            v_old = value_function[curr_state_index]
            state_action_to_value = {}

            for action in range(total_actions):
                transitions = envr.get_transitions_at_time(curr_state_tuple, action)
                transition_calls += 1
                if transitions:
                    q_val = 0.0
                    for prob, next_state_tuple in transitions:
                        next_state_index = envr.state_to_index(next_state_tuple)
                        v_next = value_function[next_state_index]
                        ball_pos = (next_state_tuple[0], next_state_tuple[1])
                        player_pos = (curr_state_tuple[0], curr_state_tuple[1])
                        reward = envr._get_reward(ball_pos, action, player_pos)
                        q_val += prob * (reward + gamma * v_next)
                    state_action_to_value[action] = q_val

            if state_action_to_value:
                max_action = max(state_action_to_value, key=state_action_to_value.get)
                max_q_val = state_action_to_value[max_action]
                value_function[curr_state_index] = max_q_val
                new_policy_function[curr_state_index] = max_action

            delta = max(delta, abs(v_old - value_function[curr_state_index]))

        policy_function = new_policy_function
        if delta < convergence:
            break

    return dict(enumerate(policy_function)), value_function, num_iterations, transition_calls


def policy_iteration_bfs(envr=FootballSkillsEnv):
    """
    BFS-based Policy Iteration.
    Evaluates and improves policy only on reachable states (BFS expansion).
    Returns (policy, value_function, num_iterations, transition_calls).
    """

    convergence = 1e-6

    total_states = envr.grid_size * envr.grid_size * 2
    total_actions = envr.action_space.n

    # Step 1: Initialization
    value_function = np.zeros(total_states)
    policy_function = np.full(total_states, -1)
    transition_calls = 0

    # Step 2: BFS to get reachable states
    visited = [False] * total_states
    reachable_states = []

    start_state, _ = envr.reset(seed=0)
    start_index = envr.state_to_index(tuple(start_state))
    q = deque([start_index])
    visited[start_index] = True

    while q:
        s_idx = q.popleft()
        reachable_states.append(s_idx)
        s_tuple = envr.index_to_state(s_idx)

        if envr._is_terminal(s_tuple):
            continue

        for action in range(total_actions):
            transitions = envr.get_transitions_at_time(s_tuple, action)
            transition_calls += 1
            if not transitions:
                continue
            for _, next_state_tuple in transitions:
                ns_idx = envr.state_to_index(next_state_tuple)
                if not visited[ns_idx]:
                    visited[ns_idx] = True
                    q.append(ns_idx)

    # Initialize policy with first valid action
    for s_idx in reachable_states:
        s_tuple = envr.index_to_state(s_idx)
        if envr._is_terminal(s_tuple):
            continue
        for a in range(total_actions):
            transitions = envr.get_transitions_at_time(s_tuple, a)
            transition_calls += 1
            if transitions:
                policy_function[s_idx] = a
                break

    # Step 3: Iteration
    policy_stable, num_iterations = False, 0
    while not policy_stable:
        num_iterations += 1

        # --- Policy Evaluation (only reachable states) ---
        while True:
            delta = 0.0
            for s_idx in reachable_states:
                if policy_function[s_idx] == -1:
                    continue
                v_old = value_function[s_idx]
                s_tuple = envr.index_to_state(s_idx)
                action = policy_function[s_idx]
                transitions = envr.get_transitions_at_time(s_tuple, action)
                transition_calls += 1
                v_new = 0.0
                if transitions:
                    for prob, ns_tuple in transitions:
                        ns_idx = envr.state_to_index(ns_tuple)
                        v_next = value_function[ns_idx]
                        ball_pos = (ns_tuple[0], ns_tuple[1])
                        player_pos = (s_tuple[0], s_tuple[1])
                        reward = envr._get_reward(ball_pos, action, player_pos)
                        v_new += prob * (reward + gamma * v_next)
                value_function[s_idx] = v_new
                delta = max(delta, abs(v_old - v_new))
            if delta < convergence:
                break

        # --- Policy Improvement ---
        new_policy = policy_function.copy()
        for s_idx in reachable_states:
            s_tuple = envr.index_to_state(s_idx)
            if envr._is_terminal(s_tuple):
                continue

            state_action_to_value = {}
            for action in range(total_actions):
                transitions = envr.get_transitions_at_time(s_tuple, action)
                transition_calls += 1
                if transitions:
                    q_val = 0.0
                    for prob, ns_tuple in transitions:
                        ns_idx = envr.state_to_index(ns_tuple)
                        v_next = value_function[ns_idx]
                        ball_pos = (ns_tuple[0], ns_tuple[1])
                        player_pos = (s_tuple[0], s_tuple[1])
                        reward = envr._get_reward(ball_pos, a, player_pos)
                        q_val += prob * (reward + gamma * v_next)
                    state_action_to_value[a] = q_val
            if state_action_to_value:
                best_action = max(state_action_to_value, key=state_action_to_value.get)
                new_policy[s_idx] = best_action

        policy_stable = np.array_equal(policy_function, new_policy)
        policy_function = new_policy

    return dict(enumerate(policy_function)), value_function, num_iterations, transition_calls

def evaluate_policy(env, policy, num_episodes=20, time_dependent=False):
    rewards = []
    horizon = 40 if time_dependent else None

    # Generate random seeds for evaluation
    seeds = np.random.randint(0, 10000, size=num_episodes)

    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        total_reward, done, t = 0, False, 0

        while not done:
            state_idx = env.state_to_index(tuple(obs))

            if time_dependent:
                action = policy[t][state_idx]
            else:
                action = policy[state_idx]

            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            t += 1
            if time_dependent and t >= horizon:
                break

        rewards.append(total_reward)

    return np.mean(rewards), np.std(rewards)


if __name__ == "__main__":

    
    env_normal = FootballSkillsEnv(render_mode="gif", degrade_pitch=False)

    # Policy Iteration 
    optimal_policy_pi, value_function_pi, num_iterations_pi, calls_pi = policy_iteration(env_normal)
    print("Policy Iteration Results")
    print("Value function (truncated):", value_function_pi[:10])
    print("Policy function (sample):", dict(list(optimal_policy_pi.items())[:10]))
    print(f"Converged in {num_iterations_pi} iterations")
    print(f"Transition calls: {calls_pi}")
    policy_list_pi = [optimal_policy_pi[s] for s in range(len(optimal_policy_pi))]
    env_normal.get_gif(policy_list_pi, seed=1234, filename=f"{gamma}_starter_code_policy.gif")
    print("Policy Iteration GIF saved as starter_code_policy.gif")
    mean_r, std_r = evaluate_policy(env_normal, policy_list_pi, num_episodes=20)
    print(f"Standard Policy Iteration, Mean Reward: {mean_r:.2f}, Std: {std_r:.2f}")

   # Value Iteration
    optimal_policy_vi, value_function_vi, num_iterations_vi, calls_vi = value_iteration(env_normal)

    print("Value Iteration Results")
    print("Value function (truncated):", value_function_vi[:10])
    print("Policy function (sample):", dict(list(optimal_policy_vi.items())[:10]))
    print(f"Converged in {num_iterations_vi} iterations")
    print(f"Transition calls: {calls_vi}")
    policy_list_vi = [optimal_policy_vi[s] for s in range(len(optimal_policy_vi))]
    env_normal.get_gif(policy_list_vi, seed=1234, filename=f"{gamma}_starter_code_value.gif")
    print("Value Iteration GIF saved as starter_code_value.gif")
    mean_r, std_r = evaluate_policy(env_normal, policy_list_vi, num_episodes=20)
    print(f"Standard Value Iteration, Mean Reward: {mean_r:.2f}, Std: {std_r:.2f}")

    print("Policy Comparison: Policy Iteration vs Value Iteration")

    pi_arr = np.array(policy_list_pi)
    vi_arr = np.array(policy_list_vi)

    diff_norm = np.linalg.norm(pi_arr - vi_arr, ord=0)
    total_states = len(pi_arr)

    print(f"Policy mismatch count: {diff_norm} out of {total_states} states")
    print(f"Agreement: {(1 - diff_norm/total_states) * 100:.2f}% | Disagreement: {(diff_norm/total_states) * 100:.2f}%")

    if diff_norm > 0:
        print("Note: Mismatches usually come from tie-breaking in argmax; both policies are still optimal.")

# -------- BFS Policy Iteration --------
    optimal_policy_pi, value_function_pi, num_iterations_pi, calls_pi = policy_iteration_bfs(env_normal)
    print("\n=== BFS Policy Iteration Results ===")
    print("Value function (truncated):", value_function_pi[:10])
    print("Policy function (sample):", dict(list(optimal_policy_pi.items())[:10]))
    print(f"Converged in {num_iterations_pi} iterations")
    print(f"Transition calls: {calls_pi}")
    policy_list_pi = [optimal_policy_pi[s] for s in range(len(optimal_policy_pi))]
    env_normal.get_gif(policy_list_pi, seed=1234, filename="bfs_policy_iteration.gif")
    print("BFS Policy Iteration GIF saved as bfs_policy_iteration.gif")
    mean_r, std_r = evaluate_policy(env_normal, policy_list_pi, num_episodes=20)
    print(f"BFS Policy Iteration -> Mean Reward: {mean_r:.2f}, Std: {std_r:.2f}")

    # -------- BFS Value Iteration --------
    optimal_policy_vi, value_function_vi, num_iterations_vi, calls_vi = value_iteration_bfs(env_normal)
    print("BFS Value Iteration Results")
    print("Value function (truncated):", value_function_vi[:10])
    print("Policy function (sample):", dict(list(optimal_policy_vi.items())[:10]))
    print(f"Converged in {num_iterations_vi} iterations")
    print(f"Transition calls: {calls_vi}")
    policy_list_vi = [optimal_policy_vi[s] for s in range(len(optimal_policy_vi))]
    env_normal.get_gif(policy_list_vi, seed=1234, filename=f"{gamma}_bfs_value_iteration.gif")
    print("BFS Value Iteration GIF saved as bfs_value_iteration.gif")
    # mean_r, std_r = evaluate_policy(env_normal, policy_list_vi, num_episodes=20)
    # print(f"BFS Value Iteration -> Mean Reward: {mean_r:.2f}, Std: {std_r:.2f}")

    # Degraded

    env_degraded = FootballSkillsEnv(render_mode="gif", degrade_pitch=True)

    optimal_policy_vi, value_function_vi, num_iterations_vi, calls_vi = value_iteration(env_degraded)
    print("Value Iteration Results")
    print("Value function (truncated):", value_function_vi[:10])
    print("Policy function (sample):", dict(list(optimal_policy_vi.items())[:10]))
    print(f"Converged in {num_iterations_vi} iterations")
    print(f"Transition calls: {calls_vi}")
    policy_list_vi = [optimal_policy_vi[s] for s in range(len(optimal_policy_vi))]
    env_normal.get_gif_stationary_degraded(policy_list_vi, seed=1234, filename=f"{gamma}_starter_code_value_stationary_degraded.gif")
    print("Value Iteration GIF saved as starter_code_value_stationary_degraded.gif")
    mean_r, std_r = evaluate_policy(env_normal, policy_list_vi, num_episodes=20)
    print(f"Standard Value Iteration, Mean Reward: {mean_r:.2f}, Std: {std_r:.2f}")

    optimal_policy_ns, value_function_ns, num_iterations_ns, calls_vi = value_iteration_nonstationary(env_degraded)
    print("Non-Stationary Value Iteration Results")
    print("Value function shape:", value_function_ns.shape)
    print("Policy function sample:", dict(list(optimal_policy_ns.items())[:10]))
    print(f"Completed {num_iterations_ns} backward sweeps")
    print(f"Transition calls: {calls_vi}")
    horizon = 40
    total_states = env_degraded.grid_size * env_degraded.grid_size * 2
    policy_list_ns = []
    for t in range(horizon):
        policy_at_t = [optimal_policy_ns.get((t, s)) for s in range(total_states)]
        policy_list_ns.append(policy_at_t)
    mean_r, std_r = evaluate_policy(env_degraded, policy_list_ns, num_episodes=20, time_dependent=True)
    print(f"Non-Stationary Value Iteration, Mean Reward: {mean_r:.2f}, Std: {std_r:.2f}")
    env_degraded.get_gif(policy_list_ns, seed=1234, filename=f"{gamma}_starter_code_nonstationary_value.gif")
    print("Non-Stationary Value Iteration GIF saved as starter_code_nonstationary_value.gif")

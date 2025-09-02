from env import FootballSkillsEnv
import numpy as np
import math


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
    gamma = 0.95
    total_states = envr.grid_size*envr.grid_size*2
    total_actions = envr.action_space.n

    # Step 1: Initialization

    # Initializing value function as zero
    value_function = np.zeros(total_states)
    
    # Initializing policy function to random valid actions
    policy_function = np.random.randint(0, total_actions, size=total_states)
    
    for state_index in range(total_states):
        state = envr.index_to_state(state_index)
        if envr._is_terminal(state):
            policy_function[state_index] = -1
            continue
        for action in range(total_actions):
            if envr.get_transitions_at_time(state, action):
                policy_function[state_index] = action
                break

    stable, num_iterations = False, 0

    # --- Policy Iteration loop ---
    while not stable:
        num_iterations += 1

        # --- Policy Evaluation ---
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

        # --- Policy Improvement ---
        new_policy_function = policy_function.copy()
        for curr_state_index in range(total_states):
            curr_state_tuple = envr.index_to_state(curr_state_index)
            if envr._is_terminal(curr_state_tuple):
                continue
            state_action_to_value = {}
            for action in range(total_actions):
                transitions = envr.get_transitions_at_time(curr_state_tuple, action)
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

        stable = np.array_equal(policy_function, new_policy_function)
        policy_function = new_policy_function


    return policy_function, value_function, num_iterations


def value_iteration(envr=FootballSkillsEnv):
    """Placeholder for Value Iteration (to be implemented)."""
    pass


if __name__ == "__main__":
    env_normal = FootballSkillsEnv(render_mode="gif", degrade_pitch=False)
    optimal_policy, value_function, num_iterations = policy_iteration(env_normal)

    print("Value function:", value_function)
    print("Policy function:", optimal_policy)
    print(f"Converged in {num_iterations} iterations")

    # Generate GIF rollout
    env_normal.get_gif(optimal_policy.tolist(), seed=1234, filename="starter_code_policy.gif")       



            
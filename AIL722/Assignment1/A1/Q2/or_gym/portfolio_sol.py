import numpy as np
from or_gym.envs.finance.discrete_portfolio_opt import DiscretePortfolioOptEnv
import time
from collections import defaultdict
from typing import Dict, Tuple, List, Set
import math


# BFS based state expansion, significantly reduces the state space to explore.

class SolveIteration:
    def __init__(self, env, gamma=1.0, threshold=1e-2):
        
        self.env = env
        self.gamma = gamma
        self.threshold = threshold
        self.max_iterations = 1000
        
        self.initial_cash = env.initial_cash
        self.prices_mu = np.array(env.asset_price_means).flatten()
        self.horizon = env.step_limit
        
        self.hold_limit = env.holding_limit[0]
        self.lot_limit = env.lot_size
        self.buy_cost = env.buy_cost
        self.sell_cost = env.sell_cost

        self.value_function_V = {}
        self.policy_function_V = {}

        self.value_function_P = {}
        self.policy_function_P = {}
        
        self.bfs_states, self.transitions = self.bfs_expansion()

    def get_next_state(self, time_step, curr_cash, curr_holdings, action):
        curr_price = self.prices_mu[time_step]
        # Buy action
        if action > 0:
            cost = (curr_price + self.buy_cost)*action
            if cost <= curr_cash:
                next_cash = curr_cash - cost
                next_holdings = curr_holdings + action
        elif action < 0:
            num_sale = min(-action, curr_holdings)
            next_cash = curr_cash + (curr_price - self.sell_cost)*num_sale
            next_holdings = curr_holdings - num_sale
        else:
            next_cash, next_holdings = curr_cash, curr_holdings
        return int(next_cash), int(next_holdings)

    # Returns a list of valid buy/sale action as per constraints for the current state
    def valid_actions(self, time_step, curr_cash, curr_holdings):
        acts = []        
        # At max stocks held can be sold if holding < lot_limit, or lot_limit otherwise
        max_num_sale = min(curr_holdings, self.lot_limit)
        # At max stocks can be bought if cash allows, or lot_limit or as holding limit allows
        curr_price = self.prices_mu[time_step]
        max_num_buy = min(math.floor(curr_cash/(curr_price + self.buy_cost)), self.lot_limit, max(0, self.hold_limit - curr_holdings))
        for a in range(-max_num_sale, max_num_buy + 1):
            acts.append(a)
        return acts

    def bfs_expansion(self):
        
        # Maintain a dictionary from time step to state: (cash, holdings).
        # We are not exploring all states, only those reachable from the initial state.
        bfs_states = {}
        transitions = {}

        bfs_states[0] = {(self.initial_cash, 0)}

        for time_step in range(self.horizon):
            for curr_cash, curr_holdings in bfs_states[time_step]:
                actions = self.valid_actions(time_step=time_step, curr_cash=curr_cash, curr_holdings=curr_holdings)
                for action in actions:
                    next_cash, next_holdings = self.get_next_state(time_step=time_step, curr_cash=curr_cash, curr_holdings=curr_holdings, action=action)
                    transitions[(time_step, curr_cash, curr_holdings, action)] = (time_step+1, next_cash, next_holdings)
                    bfs_states.setdefault(time_step+1, set()).add((next_cash, next_holdings))
        return bfs_states, transitions

    def vi_solve(self, max_iters=1000):
        # Initializing value functions for terminal states
        goal_price = self.prices_mu[-1]
        for curr_cash, curr_holdings in self.bfs_states[self.horizon]:
            self.value_function_V[(self.horizon, curr_cash, curr_holdings)] = curr_cash + curr_holdings*goal_price
        for iteration in range(self.max_iterations):
            delta = 0.0
            for time_step in reversed(range(self.horizon)):
                for curr_cash, curr_holdings in self.bfs_states[time_step]:
                    v_old = self.value_function_V.get((time_step, curr_cash, curr_holdings), 0)
                    actions = self.valid_actions(time_step=time_step, curr_cash=curr_cash, curr_holdings=curr_holdings)
                    state_action_to_val = {}
                    for action in actions:
                        _, next_cash, next_holdings = self.transitions[(time_step, curr_cash, curr_holdings, action)]
                        q_val = self.gamma*self.value_function_V[(time_step+1, next_cash, next_holdings)]
                        state_action_to_val[action] = q_val
                    max_action = max(state_action_to_val, key=state_action_to_val.get)
                    self.policy_function_V[(time_step, curr_cash, curr_holdings)] = max_action
                    self.value_function_V[(time_step, curr_cash, curr_holdings)] = state_action_to_val[max_action]
                    # breakpoint()
                    delta = max(delta, abs(v_old - self.value_function_V[(time_step, curr_cash, curr_holdings)]))
            if delta < self.threshold:
                print(f"Value Iteration converged in {iteration+1} iterations.")
                break
        return self.value_function_V, self.policy_function_V

    def policy_evaluation(self):
        goal_price = self.prices_mu[-1]
        for curr_cash, curr_holdings in self.bfs_states[self.horizon]:
            self.value_function_P[(self.horizon, curr_cash, curr_holdings)] = curr_cash + curr_holdings*goal_price
        delta = 0.0
        for time_step in reversed(range(self.horizon)):
            for curr_cash, curr_holdings in self.bfs_states[time_step]:
                v_old = self.value_function_P.get((time_step, curr_cash, curr_holdings), 0)
                action = self.policy_function_P.get((time_step, curr_cash, curr_holdings), 0)
                _, next_cash, next_holdings = self.transitions[(time_step, curr_cash, curr_holdings, action)]
                self.value_function_P[(time_step, curr_cash, curr_holdings)] = self.gamma*self.value_function_P[(time_step+1, next_cash, next_holdings)]
                delta = max(delta, abs(v_old - self.value_function_P[(time_step, curr_cash, curr_holdings)]))
        return delta

    def policy_improvement(self):
        stable = True
        for time_step in range(self.horizon):
            for curr_cash, curr_holdings in self.bfs_states[time_step]:
                a_old = self.policy_function_P[(time_step, curr_cash, curr_holdings)]
                state_action_to_val = {}
                for action in self.valid_actions(time_step=time_step, curr_cash=curr_cash, curr_holdings=curr_holdings):
                    _, next_cash, next_holdings = self.transitions[(time_step, curr_cash, curr_holdings, action)]
                    q_val = self.gamma*self.value_function_P[(time_step+1, next_cash, next_holdings)]
                    state_action_to_val[action] = q_val
                max_action = max(state_action_to_val, key=state_action_to_val.get)
                self.policy_function_P[(time_step, curr_cash, curr_holdings)] = max_action
                self.value_function_P[(time_step, curr_cash, curr_holdings)] = state_action_to_val[max_action]
                if max_action != a_old:
                    stable = False
        return stable

    def pi_solve(self, max_iters=1000):
        for time_step in range(self.horizon):
            for curr_cash, curr_holdings in self.bfs_states[time_step]:
                self.policy_function_P[(time_step, curr_cash, curr_holdings)] = 0          
        for iteration in range(self.max_iterations):
            delta = self.policy_evaluation()
            stable = self.policy_improvement()
            if stable and delta < self.threshold:
                print(f"Policy Iteration converged in {iteration+1} iterations.")
                break
        return self.value_function_P, self.policy_function_P

    def simulate(self, solver = 'VI'):
        state = self.env.reset()
        cash, current_price, current_holdings = state[0], state[1], state[2]

        trajectory = []
        done = False
        while not done:
            if solver == 'VI':
                action = self.policy_function_V.get((env.step_count, cash, current_holdings), 0)
            elif solver == 'PI':
                action = self.policy_function_P.get((env.step_count, cash, current_holdings), 0)
            next_state, reward, done, _ = env.step([action])
            trajectory.append((env.step_count-1, current_price, cash, current_holdings, action, next_state[0], next_state[2]))
            cash, current_price, current_holdings = next_state[0], next_state[1], next_state[2]
            if done:
                wealth = reward

        return trajectory, wealth

if __name__=="__main__":
    start_time=time.time()


    ###Part 1 and Part 2
    ####Please train the value and policy iteration training algo for the given three sequences of prices
    ####Config1
    prices = [1, 3, 5, 5 , 4, 3, 2, 3, 5, 8]
    env = DiscretePortfolioOptEnv(prices=prices, variance=0)
    solve_iteration = SolveIteration(env)
    solve_iteration.vi_solve()
    solve_iteration.pi_solve()
    trajectory, final_wealth = solve_iteration.simulate(solver="VI")
    print("Val Iteration: Prices: ", prices, "Final Wealth: ", final_wealth)
    trajectory, final_wealth = solve_iteration.simulate(solver="PI")
    print("Pol Iteration: Prices: ", prices, "Final Wealth: ", final_wealth)

    ####Config2
    prices = [2, 2, 2, 4 ,2, 2, 4, 2, 2, 2]
    env = DiscretePortfolioOptEnv(prices=prices, variance=0)  
    solve_iteration = SolveIteration(env)
    solve_iteration.vi_solve()
    solve_iteration.pi_solve()
    trajectory, final_wealth = solve_iteration.simulate(solver="VI")
    print("Val Iteration: Prices: ", prices, "Final Wealth: ", final_wealth)
    trajectory, final_wealth = solve_iteration.simulate(solver="PI")
    print("Pol Iteration: Prices: ", prices, "Final Wealth: ", final_wealth)

    ####Config3
    prices = [4, 1, 4, 1 ,4, 4, 4, 1, 1, 4]
    env = DiscretePortfolioOptEnv(prices=[4, 1, 4, 1 ,4, 4, 4, 1, 1, 4])
    solve_iteration = SolveIteration(env)
    solve_iteration.vi_solve()
    solve_iteration.pi_solve()
    trajectory, final_wealth = solve_iteration.simulate(solver="VI")
    print("Val Iteration: Prices: ", prices, "Final Wealth: ", final_wealth)
    trajectory, final_wealth = solve_iteration.simulate(solver="PI")
    print("Pol Iteration: Prices: ", prices, "Final Wealth: ", final_wealth)


    ####Run the evaluation on the following prices and save the plots.



    ###Part 3. (Portfolio Optimizaton)
    env = DiscretePortfolioOptEnv(variance=1)

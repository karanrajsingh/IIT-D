import numpy as np
from or_gym.envs.finance.discrete_portfolio_opt import DiscretePortfolioOptEnv
import time
import math
import matplotlib.pyplot as plt



# BFS based state expansion, significantly reduces the state space to explore.

class SolveIteration:
    def __init__(self, env, gamma=1.0, threshold=1e-2, num_samples=50):
        
        self.env = env
        self.gamma = gamma
        self.threshold = threshold
        self.max_iterations = 1000
        
        self.initial_cash = env.initial_cash
        self.prices_mu = np.array(env.asset_price_means).flatten()
        self.horizon = env.step_limit
        self.sigma = np.sqrt(np.array(env.asset_price_var).flatten()[0])
        self.num_samples = num_samples
        
        self.hold_limit = env.holding_limit[0]
        self.lot_limit = env.lot_size
        self.buy_cost = env.buy_cost.item()
        self.sell_cost = env.sell_cost.item()

        self.value_function_V = {}
        self.policy_function_V = {}

        self.value_function_P = {}
        self.policy_function_P = {}
        
        self.bfs_states, self.transitions = self.bfs_expansion()

        self.wealth_over_sweeps = []
        self.deltas_pi = []

    def get_next_state(self, time_step, curr_cash, curr_holdings, action):
        curr_price = self.prices_mu[time_step]
        # Buy action
        if action < 0:
            num_sale = min(-action, curr_holdings)
            next_cash = curr_cash + (curr_price - self.sell_cost)*num_sale
            next_holdings = curr_holdings - num_sale
        # Sell action
        elif action > 0:
            cost = (curr_price + self.buy_cost)*action
            if cost <= curr_cash:
                next_cash = curr_cash - cost
                next_holdings = curr_holdings + action
        else:
            next_cash, next_holdings = curr_cash, curr_holdings
        return next_cash, next_holdings

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
                    # breakpoint()
                    bfs_states.setdefault(time_step+1, set()).add((next_cash, next_holdings))
        return bfs_states, transitions

    def vi_solve(self, max_iters=1000):
        # Initializing value functions for terminal states
        self.wealth_over_sweeps = []
        self.wealth_over_sweeps.append(self.env.initial_cash)
        if self.sigma == 0:
            goal_price = self.prices_mu[-1]
            for curr_cash, curr_holdings in self.bfs_states[self.horizon]:
                self.value_function_V[(self.horizon, curr_cash, curr_holdings)] = curr_cash + curr_holdings*goal_price
        else:
            for curr_cash, curr_holdings in self.bfs_states[self.horizon]:
                samples = []
                for _ in range(self.num_samples):
                    sampled_price = max(0, int(np.random.normal(self.prices_mu[-1], self.sigma)))
                    samples.append(curr_cash + curr_holdings*sampled_price)
                self.value_function_V[(self.horizon, curr_cash, curr_holdings)] = np.mean(samples)
            
        for iteration in range(self.max_iterations):
            delta = 0.0
            for time_step in reversed(range(self.horizon)):
                for curr_cash, curr_holdings in self.bfs_states[time_step]:
                    v_old = self.value_function_V.get((time_step, curr_cash, curr_holdings), 0)
                    actions = self.valid_actions(time_step=time_step, curr_cash=curr_cash, curr_holdings=curr_holdings)
                    state_action_to_val = {}
                    for action in actions:
                        _, next_cash, next_holdings = self.transitions[(time_step, curr_cash, curr_holdings, action)]
                        if self.sigma == 0:
                            q_val = self.gamma*self.value_function_V[(time_step+1, next_cash, next_holdings)]
                        else:
                            samples = []
                            for _ in range(self.num_samples):
                                if time_step < self.horizon-1:
                                    sampled_price = max(0, int(np.random.normal(self.prices_mu[time_step+1], self.sigma)))
                                else:
                                    sampled_price = max(0, int(np.random.normal(self.prices_mu[-1], self.sigma)))
                                samples.append(next_cash + next_holdings*sampled_price)
                            q_val = self.gamma*np.mean(samples)    
                        state_action_to_val[action] = q_val
                    max_action = max(state_action_to_val, key=state_action_to_val.get)
                    self.policy_function_V[(time_step, curr_cash, curr_holdings)] = max_action
                    self.value_function_V[(time_step, curr_cash, curr_holdings)] = state_action_to_val[max_action]
                    # breakpoint()
                    delta = max(delta, abs(v_old - self.value_function_V[(time_step, curr_cash, curr_holdings)]))
            _, wealth = self.simulate('VI')
            self.wealth_over_sweeps.append(wealth)
            if delta < self.threshold:
                print(f"Value Iteration converged in {iteration+1} iterations.")
                break
        return self.value_function_V, self.policy_function_V

    def policy_evaluation(self):
        if self.sigma == 0:
            goal_price = self.prices_mu[-1]
            for curr_cash, curr_holdings in self.bfs_states[self.horizon]:
                self.value_function_P[(self.horizon, curr_cash, curr_holdings)] = curr_cash + curr_holdings*goal_price
        else:
            for curr_cash, curr_holdings in self.bfs_states[self.horizon]:
                samples = []
                for _ in range(self.num_samples):
                    sampled_price = max(0, int(np.random.normal(self.prices_mu[-1], self.sigma)))
                    samples.append(curr_cash + curr_holdings*sampled_price)
                self.value_function_P[(self.horizon, curr_cash, curr_holdings)] = np.mean(samples)
    
        delta = 0.0
        for time_step in reversed(range(self.horizon)):
            for curr_cash, curr_holdings in self.bfs_states[time_step]:
                v_old = self.value_function_P.get((time_step, curr_cash, curr_holdings), 0)
                action = self.policy_function_P.get((time_step, curr_cash, curr_holdings), 0)
                _, next_cash, next_holdings = self.transitions[(time_step, curr_cash, curr_holdings, action)]
                if self.sigma == 0:
                    self.value_function_P[(time_step, curr_cash, curr_holdings)] = self.gamma*self.value_function_P[(time_step+1, next_cash, next_holdings)]
                else:
                    samples = []
                    for _ in range(self.num_samples):
                        if time_step < self.horizon-1:
                            sampled_price = max(0, int(np.random.normal(self.prices_mu[time_step+1], self.sigma)))
                        else:
                            sampled_price = max(0, int(np.random.normal(self.prices_mu[-1], self.sigma)))
                        samples.append(next_cash + next_holdings*sampled_price)
                    self.value_function_P[(time_step, curr_cash, curr_holdings)] = self.gamma*np.mean(samples)
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
                    if self.sigma == 0:
                        q_val = self.gamma*self.value_function_P[(time_step+1, next_cash, next_holdings)]
                    else:
                        samples = []
                        for _ in range(self.num_samples):
                            if time_step < self.horizon-1:
                                sampled_price = max(0, int(np.random.normal(self.prices_mu[time_step+1], self.sigma)))
                            else:
                                sampled_price = max(0, int(np.random.normal(self.prices_mu[-1], self.sigma)))
                            samples.append(next_cash + next_holdings*sampled_price)
                        q_val = self.gamma*np.mean(samples)                   
                    state_action_to_val[action] = q_val
                max_action = max(state_action_to_val, key=state_action_to_val.get)
                self.policy_function_P[(time_step, curr_cash, curr_holdings)] = max_action
                self.value_function_P[(time_step, curr_cash, curr_holdings)] = state_action_to_val[max_action]
                if max_action != a_old:
                    stable = False
        return stable

    def pi_solve(self):
        self.wealth_over_sweeps = []
        self.wealth_over_sweeps.append(self.env.initial_cash)
        self.deltas_pi = []
        self.deltas_pi.append(0.0)
        for time_step in range(self.horizon):
            for curr_cash, curr_holdings in self.bfs_states[time_step]:
                self.policy_function_P[(time_step, curr_cash, curr_holdings)] = 0 
        iter = 0         
        for iteration in range(self.max_iterations):
            iter += 1
            delta = self.policy_evaluation()
            stable = self.policy_improvement()
            self.deltas_pi.append(delta)
            if self.sigma == 0:
                _, wealth = self.simulate('PI')
                self.wealth_over_sweeps.append(wealth)
            if stable and delta < self.threshold:
                print(f"Policy Iteration converged in {iteration+1} iterations.")
                break
            print(f"Policy iteration: ", iteration, "Delta: ", delta)
        return self.value_function_P, self.policy_function_P
    

    def simulate(self, solver = 'VI'):
        state = self.env.reset()
        cash, current_price, current_holdings = state[0], state[1], state[2]

        trajectory = []

        # breakpoint()

        done = False
        while not done:
            if solver == 'VI':
                action = self.policy_function_V.get((env.step_count, cash, current_holdings), 0)
            elif solver == 'PI':
                action = self.policy_function_P.get((env.step_count, cash, current_holdings), 0)
            next_state, reward, done, _ = env.step([action])
            # breakpoint()
            trajectory.append((env.step_count-1, current_price, cash, current_holdings, action, next_state[0], next_state[2]))
            cash, current_price, current_holdings = next_state[0], next_state[1], next_state[2]
            if done:
                wealth = reward

        return trajectory, wealth
    
def plot(env, solver = 'VI', str = '1'):
    for gamma in [0.999, 1.0]:
        print(f"Running Solve Iteration with gamma={gamma}")

        start_time = time.time()
        solve_iteration = SolveIteration(env, gamma=gamma)
        if solver == 'VI':
            solve_iteration.vi_solve()
        elif solver == 'PI':
            solve_iteration.pi_solve()
        trajectory, final_wealth = solve_iteration.simulate(solver)
        wealth_over_sweeps = solve_iteration.wealth_over_sweeps
        exec_time = time.time() - start_time
        print(f"Execution time (gamma={gamma}): {exec_time:.2f} sec")

        # Plot wealth progression over sweeps
        plt.figure()
        plt.plot(wealth_over_sweeps, marker='o')
        plt.xlabel("Sweeps")
        plt.ylabel("Final Portfolio Wealth")
        plt.title(f"Total Portfolio Wealth vs Sweeps (gamma={gamma})")
        plt.grid(True)
        plt.savefig(f"{solver}_{str}_wealth_vs_sweeps_gamma_{gamma}.png")

        # After training, simulate one episode
        trajectory, _ = solve_iteration.simulate(solver=solver)
        timesteps = [x[0] for x in trajectory]
        cash_vals = [x[2] for x in trajectory]
        holdings = [x[3] for x in trajectory]
        wealth_vals = [c + h*p for (_, p, c, h, _, _, _) in trajectory]

        # Plot evolution of wealth, cash, holdings
        plt.figure()
        plt.plot(timesteps, wealth_vals, label="Total Wealth")
        plt.plot(timesteps, cash_vals, label="Cash")
        plt.plot(timesteps, holdings, label="Holdings")
        plt.xlabel("Time step")
        plt.ylabel("Value")
        plt.title(f"Evolution of Wealth, Cash, Holdings (gamma={gamma})")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{solver}_{str}_evolution_gamma_{gamma}.png")

def plot_3():
    # Part 3: variance = 1.0
    env = DiscretePortfolioOptEnv(variance=1.0)
    solve_iteration = SolveIteration(env, gamma=1.0)
    solve_iteration.pi_solve()
    deltas = solve_iteration.deltas_pi

    plt.figure()
    plt.plot(range(len(deltas)), deltas, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Maximum Value Difference: delta")
    plt.title("Policy Iteration Convergence (variance=1.0)")
    plt.grid(True)
    plt.savefig("pi_convergence_variance1.png")

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
    plot(env, str='1')
    plot(env, solver='PI', str='1')

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
    plot(env, str='2')
    plot(env, solver='PI', str='2')

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
    plot(env, str='3')
    plot(env, solver='PI', str='3')


    ####Run the evaluation on the following prices and save the plots.



    ###Part 3. (Portfolio Optimizaton)
    env = DiscretePortfolioOptEnv(variance=1)
    solve_iteration = SolveIteration(env)
    solve_iteration.vi_solve()
    solve_iteration.pi_solve()
    trajectory, final_wealth = solve_iteration.simulate(solver="VI")
    print("Val Iteration: Prices: ", prices, "Final Wealth: ", final_wealth)
    trajectory, final_wealth = solve_iteration.simulate(solver="PI")
    print("Pol Iteration: Prices: ", prices, "Final Wealth: ", final_wealth)
    plot(env)
    plot(env, solver='PI')


    prices = [4, 1, 4, 1 ,4, 4, 4, 1, 1, 4]   # example sequence
    env = DiscretePortfolioOptEnv(prices=prices, variance=0)
    plot_3()
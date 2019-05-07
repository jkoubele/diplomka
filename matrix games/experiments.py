import numpy as np
from agents import QAgent, UCBAgent, Exp3Agent, GradientBasedAgent
from game_solver import solve_game
import matplotlib.pyplot as plt


def num_neqs(game: np.ndarray) -> int:
    neqs = 0
    m, n = game.shape
    for i in range(m):
        for j in range(n):
            if game[i][j] >= np.max(game[:,j]) and game[i][j] <= np.min(game[i,:]):
                neqs += 1
    return neqs
            


if __name__ == "__main__":
    
    NUM_EXPERIMENTS = 100
    games = []
    epsilons = []
    strategies_1_neq = []
    strategies_2_neq = []
    strategies_1_avg = []
    strategies_2_avg = []
    game_values = []
    for i in range(NUM_EXPERIMENTS):
        game = np.random.random((3,3))        
        
        #game[0,:] = np.random.random(50)
        #game[1,:] = game[0,:] + np.random.random(50) / 10
        game -= game.min()
        game /= (game.max() - game.min())
        print(i)
        
        num_rounds = 250000
        
        player1 = QAgent(game.shape[0], epsilon = 0.1, diminish=False)
        player2 = QAgent(game.shape[1], epsilon = 0.1, diminish=False)
        player1 = GradientBasedAgent(game.shape[0], alpha=0.01)
        player2 = GradientBasedAgent(game.shape[1], alpha=0.01)
    
        
                    
        for t in range(num_rounds):
            action1 = player1.select_action()
            action2 = player2.select_action()
            outcome = game[action1, action2]
            reward1 = outcome
            reward2 = 1 - outcome  
            
            player1.get_reward_and_update(reward1)
            player2.get_reward_and_update(reward2)
       
        avg_strategy_1 = player1.greedy_trials / np.sum(player1.greedy_trials)
        avg_strategy_2 = player2.greedy_trials / np.sum(player2.greedy_trials)        
        
        estimated_v = np.dot(avg_strategy_1 ,np.dot(game, avg_strategy_2))
        br_value_1 = np.max(np.dot(game, avg_strategy_2)) 
        br_value_2 = np.min(np.dot(avg_strategy_1, game))
        
        
        print(f"BR value 1 = {br_value_1}")
        print(f"BR value 2 = {br_value_2}")
        print(f"Estimated v = {estimated_v}")
        epsilon = max(br_value_1 - estimated_v, estimated_v - br_value_2)
        print(f"epsilon = {epsilon}")
        
        
        v, s1, s2 = solve_game(game)
        game_values.append(v)
        strategies_1_neq.append(s1)
        strategies_2_neq.append(s2)
        strategies_1_avg.append(avg_strategy_1)
        strategies_2_avg.append(avg_strategy_2)
        games.append(game)
        
        epsilons.append(epsilon)
        
        
    indices = list(range(NUM_EXPERIMENTS))
    indices.sort(key = lambda x: epsilons[x], reverse=True)
    games = [games[index] for index in indices]
    game_values = [game_values[index] for index in indices]
    strategies_1_avg = [strategies_1_avg[index] for index in indices]
    strategies_2_avg = [strategies_2_avg[index] for index in indices]
    strategies_1_neq = [strategies_1_neq[index] for index in indices]
    strategies_2_neq = [strategies_2_neq[index] for index in indices]
    epsilons.sort(reverse=True)
    
    avg_regret_bound_1 = 2.63 * np.sqrt(num_rounds * player1.num_actions * np.log(player1.num_actions)) / num_rounds
    avg_regret_bound_2 = 2.63 * np.sqrt(num_rounds * player2.num_actions * np.log(player2.num_actions)) / num_rounds
    regret_bound = max(avg_regret_bound_1, avg_regret_bound_2)
    epsilon_bound = 2 * regret_bound    
    
    plt.plot(epsilons)
    plt.plot(np.ones(NUM_EXPERIMENTS) * epsilon_bound, color = 'r')
    plt.show()
    
        


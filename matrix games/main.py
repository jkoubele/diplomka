import numpy as np
from agents import QAgent, UCBAgent, Exp3Agent
from game_solver import solve_game
import matplotlib.pyplot as plt

if __name__ == "__main__":
    game = np.asarray([[1,0],
                       [0,2]])
    """
    UCB konverguje, kdyz -0.4765260 zmenime na -0.4765261 tak diverguje
    game = np.asarray([[0,-0.4765260], 
                       [11,-10],
                       [-10,11]], dtype = np.float64)
    
    
    game = np.asarray([[0,1], 
                       [-1,0],
                       ], dtype = np.float64)
    """
    game = np.asarray([[0,-0.4765261], 
                       [11,-10],
                       [-10,11]], dtype = np.float64)
    
    
    
    game -= game.min()
    game /= (game.max() - game.min())
    v, s1, s2 = solve_game(game)
    num_rounds = 100000
    """
    player1 = Exp3Agent(game.shape[0], num_rounds)
    player2 = Exp3Agent(game.shape[1], num_rounds)
    """ 
    player1 = UCBAgent(game.shape[0], c = 1.0)
    player2 = UCBAgent(game.shape[1], c = 1.0)
    
    reward1 = 0
    reward2 = 0
    
    epsilons = []
    v_estimates_error = []
    
                      
    avg_v = 0
        
    for t in range(num_rounds):
        action1 = player1.select_action(reward1)
        action2 = player2.select_action(reward2)
        outcome = game[action1, action2]
        reward1 = outcome
        reward2 = 1 - outcome  
        avg_v = (avg_v * t + reward1) / (t + 1)
        v_estimates_error.append(np.abs(avg_v - v))
        
        if (t % 1 == 0 and t > 0):
            avg_strategy_1 = player1.trials / np.sum(player1.trials)
            avg_strategy_2 = player2.trials / np.sum(player2.trials)
            br_value_1 = np.max(np.dot(game, avg_strategy_2))
            br_value_2 = np.min(np.dot(avg_strategy_1, game))
            epsilon1 = np.abs(br_value_1 - v)
            epsilon2 = np.abs(br_value_2 - v)
            epsilons.append((epsilon1 + epsilon2) / 2)
        if t % 1000 == 0:
            print(f"{int(t/1000)}k")
                        
            
        
    print("Player 1 average strategy:", player1.trials / np.sum(player1.trials))
    print("Player 2 average strategy:", player2.trials / np.sum(player2.trials))
    plt.plot(epsilons)
    plt.xlabel("t")
    plt.ylabel("avg. epsilon")
    plt.show()
    plt.plot(v_estimates_error)
    plt.xlabel("t")
    plt.ylabel("v estimate error")
    plt.show()
    
    
    avg_regret_bound_1 = 2.63 * np.sqrt(num_rounds * player1.num_actions * np.log(player1.num_actions)) / num_rounds
    avg_regret_bound_2 = 2.63 * np.sqrt(num_rounds * player2.num_actions * np.log(player2.num_actions)) / num_rounds
    regret_bound = max(avg_regret_bound_1, avg_regret_bound_2)
    epsilon_bound = 2 * regret_bound
    epsilon = epsilons[-1]
    print("Bound:", epsilon_bound)
    print("Epsilon", epsilon)


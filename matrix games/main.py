import numpy as np
from agents import QAgent, UCBAgent, Exp3Agent, GradientBasedAgent
from game_solver import solve_game
import matplotlib.pyplot as plt

if __name__ == "__main__":
    game = np.asarray([[1,0],
                       [0,2]])
    """
    UCB konverguje, kdyz -0.4765260 zmenime na -0.4765261 tak diverguje
    """
    game = np.asarray([[0,-0.476526], 
                       [11,-10],
                       [-10,11]], dtype = np.float64)
    game = np.asarray([[0.9740483 , 0.63154835, 0.09830574, 0.        ],
       [0.17752728, 0.65709828, 0.09961379, 0.92046434],
       [0.35022567, 0.8916837 , 1.        , 0.53076941],
       [0.49642977, 0.80655912, 0.68135012, 0.37950726]], dtype = np.float64)
    
    game = np.asarray([[0,-0.476526], 
                       [11,-10],
                       [-10,11]], dtype = np.float64)
    
    """    
    game = np.asarray([[0.9740483 , 0.63154835, 0.09830574, 0.       ],
       [0.17752728, 0.65709828, 0.09961379, 0.92046434],
       [0.35022567, 0.8916837 , 1.        , 0.53076941],
       [0.49642977, 0.80655912, 0.68135012, 0.37950726]], dtype = np.float64)
    
    game =  np.asarray([
                [0,2],
                [4,1]                
                ], dtype = np.float64)
    
    
   
    
    
    
    
    Hure konvergujici hra
    game = np.asarray([[0.9740483 , 0.63154835, 0.09830574, 0.        ],
       [0.17752728, 0.65709828, 0.09961379, 0.92046434],
       [0.35022567, 0.8916837 , 1.        , 0.53076941],
       [0.49642977, 0.80655912, 0.68135012, 0.37950726]], dtype = np.float64)
    
    
    
    
    
    
    
    game = np.asarray([[0.9740483 , 0.63154835, 0.09830574, 0.       ],
       [0.17752728, 0.65709828, 0.09961379, 0.92046434],
       [0.35022567, 0.8916837 , 1.        , 0.53076941],
       [0.49642977, 0.80655912, 0.68135012, 0.37950726]], dtype = np.float64)
    
    game = np.array([[0.96699878, 0.884819  , 0.27870048, 0.        ],
       [0.20757165, 0.15412761, 0.16627226, 0.65769824],
       [0.07109748, 0.24257018, 0.33444664, 0.61388564],
       [0.94783685, 0.07108995, 1.        , 0.08121038]])
    
    game = np.array([[1.0, 0.9, 0.3, 0.0],
                     [0.2, 0.2, 0.2, 0.7],
                     [0.8, 0.3, 0.3, 0.6],
                     [1.0, 0.1, 1.0, 0.1]])
    
    game = np.array([[0.9, 0.3, 0.0],
                     [0.2, 0.2, 0.7],
                     [0.3, 0.3, 0.6],
                     [0.1, 1.0, 0.1]])
    
    game = np.array([[0.9, 0.3, 0.0],                     
                     [0.3, 0.3, 0.6],
                     [0.1, 1.0, 0.1]])
    """
    
    
    
    
        
    outcomes = np.zeros_like(game)
    
    game -= game.min()
    game /= (game.max() - game.min())
    v, s1, s2 = solve_game(game)
    num_rounds = 500000
    
    player1 = Exp3Agent(game.shape[0], num_rounds)
    player2 = Exp3Agent(game.shape[1], num_rounds)
    """
    player1 = QAgent(game.shape[0], epsilon = 0.1, diminish=True)
    player2 = QAgent(game.shape[1], epsilon = 0.1, diminish=True)
    
    player1 = UCBAgent(game.shape[0], c=1)
    player2 = UCBAgent(game.shape[1], c=1)
    """  
    player1 = GradientBasedAgent(game.shape[0], alpha=0.01)
    player2 = GradientBasedAgent(game.shape[1], alpha=0.01)
    
    epsilons = []
    v_estimates_error = []
    
                      
    avg_v = 0
        
    for t in range(num_rounds):
        action1 = player1.select_action()
        action2 = player2.select_action()
        
        outcomes[action1, action2] += 1
        outcome = game[action1, action2]
        
        reward1 = outcome
        reward2 = 1 - outcome
        
        player1.get_reward_and_update(reward1)
        player2.get_reward_and_update(reward2)
        
        avg_v = (avg_v * t + reward1) / (t + 1)
        v_estimates_error.append(np.abs(avg_v - v))
        
        if (t % 1 == 0 and t > 0):
            avg_strategy_1 = player1.greedy_trials / np.sum(player1.greedy_trials)
            avg_strategy_2 = player2.greedy_trials / np.sum(player2.greedy_trials)
            br_value_1 = np.max(np.dot(game, avg_strategy_2))
            br_value_2 = np.min(np.dot(avg_strategy_1, game))
            empirical_v = np.dot(np.dot(avg_strategy_1, game), avg_strategy_2)
            epsilon1 = np.abs(br_value_1 - empirical_v)
            epsilon2 = np.abs(br_value_2 - empirical_v)
            epsilons.append(max(epsilon1, epsilon2))
        if t % 10000 == 0:
            print(f"{int(t/1000)}k")
                        
            
        
    print("Player 1 average strategy:", player1.greedy_trials / np.sum(player1.greedy_trials))
    print("Player 2 average strategy:", player2.greedy_trials / np.sum(player2.greedy_trials))
    plt.plot(epsilons)
    plt.xlabel("t")
    plt.ylabel("epsilon")
    plt.show()
    plt.plot(v_estimates_error)
    plt.xlabel("t")
    plt.ylabel("v estimate error")
    plt.show()
    
    outcomes /= np.sum(outcomes)
    dist_neq = np.zeros_like(outcomes)
    for i in range(len(s1)):
        for j in range(len(s2)):
            dist_neq[i,j] = s1[i] * s2[j]
    avg_regret_bound_1 = 2.63 * np.sqrt(num_rounds * player1.num_actions * np.log(player1.num_actions)) / num_rounds
    avg_regret_bound_2 = 2.63 * np.sqrt(num_rounds * player2.num_actions * np.log(player2.num_actions)) / num_rounds
    regret_bound = max(avg_regret_bound_1, avg_regret_bound_2)
    epsilon_bound = 2 * regret_bound
    epsilon = epsilons[-1]
    print("Bound:", epsilon_bound)
    print("Epsilon", epsilon)


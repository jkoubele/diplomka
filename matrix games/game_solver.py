import numpy as np
from pulp import *
from typing import Tuple


def solve_game(game: np.ndarray) -> Tuple[np.ndarray]:
    
    v, strategy1 = solve_strategy_for_row_player(game)
    _, strategy2 = solve_strategy_for_row_player(-np.transpose(game))
    return (v, strategy1, strategy2)
    
    
        
def solve_strategy_for_row_player(game: np.ndarray) -> Tuple[float, np.ndarray]:
    prob = LpProblem("Solving the game", LpMaximize)
    v = LpVariable("Game Value")
    strategy = [LpVariable(f"p_{i}", 0) for i in range(game.shape[0])]
    
    prob += v
    prob += lpSum([p for p in strategy]) == 1.0
    for j in range(game.shape[1]):
        prob += (lpSum(strategy[i] * game[i,j] for i in range(game.shape[0])) >= v)
    prob.solve()    
    return (v.varValue, np.asarray([p.varValue for p in strategy]))
    
    
if __name__ == "__main__":
    game = np.asarray([[1,0], [0,99]])
    v, s1, s2 = solve_game(game)
    print(v)
    print(s1)
    print(s2)
    



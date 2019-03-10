import numpy as np
from abc import ABC, abstractmethod

class Agent(ABC):
    
    def __init__(self, num_actions: int):
        self.num_actions = num_actions
        self.trials = np.zeros(num_actions)
        self.greedy_trials = np.zeros(num_actions)        
    
    @abstractmethod
    def select_action(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_reward_and_update(self, reward: float) -> None:
        raise NotImplementedError


class QAgent(Agent):

    def __init__(self, num_actions: int, epsilon: float = 0.1, diminish: bool = True,
                 diminishing_parameter: float = 1000) -> None:
        super().__init__(num_actions)        
        self.estimates = np.zeros(num_actions)
        self.epsilon = epsilon        
        self.diminish = diminish
        self.diminishing_parameter = diminishing_parameter
        self.exploratory_move = False

    def select_action(self) -> int:
        if np.any(self.trials == 0):
            action = np.argmin(self.trials)
        else:
            self.exploratory_move = False
            if np.random.random() < self.epsilon:
                action = np.random.randint(self.num_actions)
                self.exploratory_move = True
            else:
                action = np.argmax(self.estimates)
                
        self.last_action = action
        return action
        
    def get_reward_and_update(self, reward: float) -> None:        
        self.estimates[self.last_action] = (reward + self.estimates[self.last_action] *
            self.trials[self.last_action]) / (self.trials[self.last_action] + 1)
        self.trials[self.last_action] += 1
        if not self.exploratory_move:
            self.greedy_trials[self.last_action] += 1        



class UCBAgent(Agent):

    def __init__(self, num_actions: int, c: float = 1) -> None:
        super().__init__(num_actions)        
        self.estimates = np.zeros(num_actions)
        self.c = c
        self.last_action = None        
    
    def select_action(self) -> int:
        if np.any(self.trials == 0):
            action = np.argmin(self.trials)
        else:
            ucb = self.estimates + self.c * np.sqrt(np.log(np.sum(self.trials)) / self.trials)
            action = np.argmax(ucb)            
        self.last_action = action   
        return action
    
    def get_reward_and_update(self, reward: float) -> None:
        self.estimates[self.last_action] = (reward + self.estimates[self.last_action] *
            self.trials[self.last_action]) / (self.trials[self.last_action] + 1)
        self.trials[self.last_action] += 1
        self.greedy_trials[self.last_action] += 1
    


class Exp3Agent(Agent):

    def __init__(self, num_actions: int, g: int) -> None:
        super().__init__(num_actions)       
        self.gamma = min(1, np.sqrt((num_actions * np.log(num_actions)) / (g * (np.e - 1))))
        self.last_action = None
        self.weights = np.ones(num_actions)
        self.p = np.ones(num_actions) / num_actions
        
    def select_action(self) -> int:
        action = np.random.choice(self.num_actions, p=self.p)
        self.last_action = action
        return action
    
    def get_reward_and_update(self, reward: float) -> None:
        self.trials[self.last_action] += 1
        self.greedy_trials[self.last_action] += 1
        
        estimated_reward = reward / self.p[self.last_action]
        self.weights[self.last_action] *= np.exp(self.gamma * estimated_reward / self.num_actions)
        self.weights /= self.weights.max() # for numerical stability
        self.p = (1 - self.gamma) * self.weights / np.sum(self.weights) + self.gamma / self.num_actions
    

class GradientBasedAgent(Agent):

    def __init__(self, num_actions: int, alpha: float = 0.1) -> None:
        super().__init__(num_actions)        
        self.alpha = alpha
        self.last_action = None
        self.weights = np.zeros(num_actions)
        self.p = np.ones(num_actions) / num_actions
        self.avg_reward = 0        
        
    def select_action(self) -> int:
        action = np.random.choice(self.num_actions, p=self.p)
        self.last_action = action
        return action
    
    def get_reward_and_update(self, reward: float) -> None:
        self.avg_reward = (np.sum(self.trials) * self.avg_reward + reward) / (np.sum(self.trials) + 1)
        self.trials[self.last_action] += 1
        self.greedy_trials[self.last_action] += 1
        self.weights[self.last_action] += self.alpha * (reward - self.avg_reward) * (1 - self.p[self.last_action])
        
        unplayed_actions = np.arange(len(self.weights)) != self.last_action
        self.weights[unplayed_actions] -= self.alpha * (reward - self.avg_reward) * self.p[unplayed_actions]
        
        self.weights -= self.weights.max() # for numerical stability        
        self.p = np.exp(self.weights) / np.sum(np.exp(self.weights))


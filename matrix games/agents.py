import numpy as np


class QAgent:
    
    def __init__(self, num_actions: int, epsilon: float = 0.1, diminish: bool = True,
                 diminishing_parameter: float = 1000) -> None:
        self.num_actions = num_actions        
        self.trials = np.zeros(num_actions)
        self.estimates = np.zeros(num_actions)
        self.epsilon = epsilon
        self.last_action = None
        self.diminish = diminish
        self.diminishing_parameter = diminishing_parameter
        self.exploratory_move = False
        
    def select_action(self, reward_for_last_action: float) -> int:
        
        if not self.last_action == None:
            self.estimates[self.last_action] = (reward_for_last_action + self.estimates[self.last_action] * 
            self.trials[self.last_action]) / (self.trials[self.last_action] + 1)
            if not self.exploratory_move:
                self.trials[self.last_action] += 1
        if np.any(self.trials == 0):
            action = np.argmin(self.trials)
            self.last_action = action            
        else:
            self.exploratory_move = False
            epsilon = self.epsilon
            if self.diminish:
                epsilon = epsilon * (self.diminishing_parameter / (self.diminishing_parameter + np.sum(self.trials)))
            if np.random.random() < self.epsilon:
                action = np.random.randint(self.num_actions)
                self.exploratory_move = True
            else:
                action = np.argmax(self.estimates)
            self.last_action = action
        return action
        
        
class UCBAgent:
    
    def __init__(self, num_actions: int, c: float = 1) -> None:
        self.num_actions = num_actions        
        self.trials = np.zeros(num_actions)
        self.estimates = np.zeros(num_actions)
        self.c = c
        self.last_action = None
        
        
    def select_action(self, reward_for_last_action: float) -> int:
        
        if not self.last_action == None:
            self.estimates[self.last_action] = (reward_for_last_action + self.estimates[self.last_action] * 
            self.trials[self.last_action]) / (self.trials[self.last_action] + 1)
            self.trials[self.last_action] += 1
        if np.any(self.trials == 0):
            action = np.argmin(self.trials)
            self.last_action = action            
            return action
        else:
            ucb = self.estimates + self.c * np.sqrt(np.log(np.sum(self.trials)) / self.trials)
            action = np.argmax(ucb)
            self.last_action = action
        return action

    
class Exp3Agent:
    
    def __init__(self, num_actions: int, g: int) -> None:
        self.num_actions = num_actions          
        self.trials = np.zeros(num_actions)
        self.gamma = min(1, np.sqrt((num_actions * np.log(num_actions)) / (g * (np.e - 1))))        
        self.last_action = None
        self.weights = np.ones(num_actions)
        self.p = np.ones(num_actions) / num_actions
        
    def select_action(self, reward_for_last_action: float) -> int:
        if not self.last_action == None:
            self.trials[self.last_action] += 1
            estimated_reward = reward_for_last_action / self.p[self.last_action]
            self.weights[self.last_action] *= np.exp(self.gamma * estimated_reward / self.num_actions)
            self.weights /= self.weights.max()
        self.p = (1 - self.gamma) * self.weights / np.sum(self.weights) + self.gamma / self.num_actions
        action = np.random.choice(self.num_actions, p=self.p)
        self.last_action = action
        return action
    
class GradientBasedAgent:
    
    def __init__(self, num_actions: int, alpha: float) -> None:
        self.num_actions = num_actions          
        self.trials = np.zeros(num_actions)
        self.alpha = alpha
        self.last_action = None
        self.weights = np.zeros(num_actions)
        
        
    def select_action(self, reward_for_last_action: float) -> int:
        if not self.last_action == None:
            self.trials[self.last_action] += 1
            estimated_reward = reward_for_last_action / self.p[self.last_action]
            self.weights[self.last_action] *= np.exp(self.alpha * estimated_reward / self.num_actions)
            self.weights /= self.weights.max()
        self.p = (1 - self.alpha) * self.weights / np.sum(self.weights) + self.alpha / self.num_actions
        action = np.random.choice(self.num_actions, p=self.p)
        self.last_action = action
        return action
        
        
            
import numpy as np
from collections import defaultdict


class Logger(object):
    """
    A class for logging the agent states and actions
    """
    def __init__(self,goal,num_agent:int=1):
        self.num_agent = num_agent
        self.history = [defaultdict(list) for _ in range(self.num_agent)]
        if self.num_agent == 1:
            self.history[0]['goal'].append(goal)
        else:
            for i in range(self.num_agent):
                self.history[i]['goal'].append(goal)

    def log(self,state,action):
        if self.num_agent == 1:
            self.history[0]['state'].append(state)
            self.history[0]['action'].append(action)
        else:
            for i in range(self.num_agent):
                self.history[i]['state'].append(state[i])
                self.history[i]['action'].append(action[i])





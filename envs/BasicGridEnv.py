from maps.example import WALLS
import gym
import gym.spaces
import numpy as np


class BasicGridEnv(gym.Env):
    """Basic class for 2D navigation environments"""
    """
    Args:
        Action:Discrete(8) 0:right 1:upper right 2:up 3:upper left 4:left 5:down left 6:down 7:down right
        Obs:MultiDiscrete([(0,self._height-1),(0,self._width-1)])
    """
    def __init__(self,walls='FourRooms',threshold_distance=1):
        self._walls = WALLS[walls]
        (height,width) = self._walls.shape
        self._height = height
        self._width = width
        self.threshold_distance = threshold_distance
        self.action_space = gym.spaces.Discrete(8)
        self.observation_space = gym.spaces.Box(
            low = np.array([0.0,0.0,0.0,0.0]),
            high = np.array([1.0,1.0,1.0,1.0]),
            dtype = np.float32
        )
        self.action_dic = [(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1)]
        self.state = None
        self.goal = None
        self.count = 0
        self.reset()

    def reset(self):
        self.state,self.goal = self._sample_random_state(),self._sample_random_state()
        self.count = 0
        return self._normalize_obs(np.concatenate([self.state,self.goal],0))


    def _sample_random_state(self):
        candidate_states = np.where(self._walls==0)
        num_candidate_states = len(candidate_states[0])
        state_index = np.random.choice(num_candidate_states)
        state = np.array([
            candidate_states[0][state_index],
            candidate_states[1][state_index]
        ])
        return state

    def _normalize_obs(self,obs):
        return np.array([
            obs[0]/float(self._height),obs[1]/float(self._width),
            obs[2]/float(self._height),obs[3]/float(self._width)
        ])

    def step(self,action):
        move_tuple = self.action_dic[action]
        new_state = self.state + move_tuple
        self.count += 1
        if not self._is_blocked(new_state):
            self.state = new_state
        done = self._check_done()
        obs = self._normalize_obs(np.concatenate([self.state, self.goal], 0))
        rew = -1
        return obs,rew,done,{}

    def _is_blocked(self,state):
        if (state[0]<0) or (state[0]>=self._height) or (state[1]<0) or (state[1]>=self._width):
            return True
        i, j = state[0], state[1]
        return (self._walls[i,j]==1)

    def _get_state(self):
        return self.state

    def _get_goal(self):
        return self.goal

    def _check_done(self):
        return np.linalg.norm(self.state-self.goal) < self.threshold_distance or self.count >= 2*(self._height+self._width)














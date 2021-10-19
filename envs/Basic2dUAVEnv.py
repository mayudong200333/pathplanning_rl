import gym
import gym.spaces
import numpy as np
from maps.example import WALLS
from utils.tools import resize_maps
from utils.lidar import Lidar

MAP = WALLS['FourRooms']

class Basic2dUAVEnv(gym.Env):
    """ Basic class of 2D navegation UAV env
    Actions:
        v:[-0.2,1.0] linear velocity of the UAV
        tau_yaw: [-1.0,1.0] angular velocity of t he UAV

    States:
        x,y,yaw,goalsets
    """

    def __init__(self,init_yaw=0,map=MAP,resize_factor=1,freq=100,threshold_distance=1):
        if resize_factor>1:
            self._map = resize_maps(map,resize_factor)
        else:
            self._map = map
        (height,width) = self._map.shape
        self._height = height - 1
        self._width = width - 1
        self.SIMFREQ = freq
        self.dt = 1/self.SIMFREQ
        self.init_yaw = np.array([init_yaw])
        self.action_space = gym.spaces.Box(
            low = np.array([-0.2,-1.0]),
            high= np.array([1.0,1.0]),
            dtype=np.float32
        )
        self.threshold_distance = threshold_distance
        self.lidar = Lidar(map=self._map)
        self.num_obs = 5 + self.lidar.num_beams
        self.observation_space = gym.spaces.Box(
            low = np.array([0.0 for _ in range(self.num_obs)]),
            high= np.array([1.0 for _ in range(self.num_obs)]),
            dtype= np.float32
        )
        self.reset()

    def reset(self):
        self.count = 0
        self.start,self.goal = self._sample_random_state(),self._sample_random_state()
        self.obs = np.concatenate([self.start,self.init_yaw,self.goal],0)
        self.lidar_obs = self.lidar.get_lidar_obs(self.obs[:3])
        return np.concatenate([self._normalize_obs(self.obs),self.lidar_obs],0)

    def _sample_random_state(self):
        '''
        sample the random position (x-y)
        '''
        candidate_states = np.where(self._map==0)
        num_candidate_states = len(candidate_states[0])
        state_index = np.random.choice(num_candidate_states)
        state = np.array([
            candidate_states[0][state_index],
            candidate_states[1][state_index],
        ],dtype=np.float)
        state += np.random.uniform(low=0.0, high=0.5, size=2)
        if state[0] >= self._height:
            state[0] -= 1
        elif state[1] >= self._width:
            state[1] -= 1
        assert not self._is_blocked(state)
        return state

    def _is_blocked(self,state):
        if (state[0]<0) or (state[0]>=self._height) or (state[1]<0) or (state[1]>=self._width):
            return True
        (i,j) = self._discretize_state(state)
        return (self._map[i,j]==1)


    def _discretize_state(self,state):
        (i,j) = np.round(state).astype(np.int)
        return (i,j)

    def step(self,action):
        dobs = self._dynamics(action)
        self.obs[0:3] = self.obs[0:3] + self.dt * dobs
        self.obs[2] = self._clip_angle(self.obs[2])
        self.lidar_obs = self.lidar.get_lidar_obs(self.obs[:3])
        self.count += 1
        obs = self._normalize_obs(self.obs)
        obs_all = np.concatenate([obs,self.lidar_obs],0)
        done = self._check_done()
        reward = self._compute_reward()
        return obs_all,reward,done,{}

    def _dynamics(self,action):
        dx = action[0]*np.cos(self.obs[2])
        dy = action[0]*np.sin(self.obs[2])
        dyaw = action[1]
        return np.array([dx,dy,dyaw])

    def _clip_angle(self,angle):
        while angle < -np.pi or angle > np.pi:
            if angle < -np.pi:
                angle += 2*np.pi
            if angle > np.pi:
                angle -= 2*np.pi
        return angle

    def _normalize_obs(self,obs):
        return np.array([
            obs[0]/float(self._height),obs[1]/float(self._width),
            obs[2]/np.pi,
            obs[3]/float(self._height),obs[4]/float(self._width)
        ])

    def _check_done(self):
        return np.linalg.norm(self.obs[:2]-self.goal) < self.threshold_distance or self.count/self.SIMFREQ >= 2*(self._height+self._width)

    def _compute_reward(self):
        return -1

    def _get_goal(self):
        return self.goal

    def _get_position(self):
        return np.copy(self.obs[:2])













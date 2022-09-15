import gym
import gym.spaces
import numpy as np
import networkx as nx
from maps.example import WALLS
from maps.map_generate import map_generate
from utils.tools import resize_maps
from utils.lidar import Lidar

MAP = WALLS['Random']

class Basic2dUAVEnv2(gym.Env):
    """ Basic class of 2D navegation UAV env
    Actions:
        v:[0.0,1.0] linear velocity of the UAV
        tau_yaw: [-1.0,1.0] angular velocity of t he UAV

    States:
        x,y,yaw,goalsets
    """

    def __init__(self,map=MAP,resize_factor=1,freq=100,threshold_distance=0.1,map_random=False):
        if map_random:
            self._map = map_generate([4,4],3)
        else:
            self._map = map
        if resize_factor>1:
            self._map = resize_maps(map,resize_factor)
        self._map_random = map_random
        (height,width) = self._map.shape
        self._height = height - 1
        self._width = width - 1
        self.g,self.g_dist = self._construct_graph()
        self.SIMFREQ = freq
        self.dt = 1/self.SIMFREQ
        self.init_yaw = np.array([0.,])
        self.action_space = gym.spaces.Box(
            low = np.array([0.0,-1.0]),
            high= np.array([1.0,1.0]),
            dtype=np.float64
        )
        self.last_pos = None
        self.threshold_distance = threshold_distance
        self.lidar = Lidar(map=self._map)
        self.num_obs = 7 + self.lidar.num_beams
        self.observation_space = gym.spaces.Box(
            low = np.array([0.0 for _ in range(self.num_obs)]),
            high= np.array([1.0 for _ in range(self.num_obs)]),
            dtype= np.float64
        )
        self.reset()

    def reset(self):
        self.count = 0
        self._if_goal=False
        self._if_collision=False
        self.last_pos = None
        if self._map_random:
            self._map = map_generate([4,4],3)
        self.start,self.goal = self._sample_constrained_state()
        self.init_angle = self._clip_angle(np.pi/2-np.arctan2(self.goal[1]-self.start[1],self.goal[0]-self.start[0]))
        self.init_yaw = np.array([self.init_angle]) + np.array([np.random.uniform(low=-np.pi/2,high=np.pi/2)])
        #self.init_yaw = np.array([0.,])
        #self.init_yaw = np.array([np.random.uniform(low=-np.pi,high=np.pi)])
        self.obs = np.concatenate([self.start,self.init_yaw,self.goal],0)
        self.lidar_obs = self.lidar.get_lidar_obs(self.obs[:3])
        return np.concatenate([self._normalize_obs(self.obs),np.array([0.0,0.0],dtype=np.float32),self.lidar_obs],0)


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
        if state[0] - 1 >= 0 and state[0] + 1 <= self._height and state[1] -1 >= 0 and state[1] + 1 <= self._width:
            state += np.random.uniform(low=-0.25, high=0.25, size=2)
        assert not self._is_blocked(state)
        return state

    def _sample_constrained_state(self,min_dist = 1.0,max_dist = 3.0):
        start = self._sample_random_state()
        (i,j) = self._discretize_state(start)
        mask = np.logical_and(self.g_dist[i,j] >=min_dist,self.g_dist[i,j]<=max_dist)
        mask = np.logical_and(mask,self._map==0)
        candidate_states = np.where(mask)
        num_candidate_states = len(candidate_states[0])
        assert num_candidate_states != 0
        goal_index = np.random.choice(num_candidate_states)
        goal = np.array([candidate_states[0][goal_index],
                         candidate_states[1][goal_index]],
                        dtype=np.float)
        if goal[0] - 1 >= 0 and goal[0] + 1 <= self._height and goal[1] - 1 >= 0 and goal[1] + 1 <= self._width:
            goal += np.random.uniform(low=-0.125, high=0.125, size=2)
        assert not self._is_blocked(goal)
        return start,goal

    def _is_blocked(self,state):
        if (state[0]<0) or (state[0]>self._height) or (state[1]<0) or (state[1]>self._width):
            return True
        (i,j) = self._discretize_state(state)
        return (self._map[i,j]==1)


    def _discretize_state(self,state):
        (i,j) = np.round(state).astype(np.int)
        return (i,j)

    def step(self,action):
        dobs = self._dynamics(action)
        self.last_pos = np.copy(self.obs[0:2])
        self.obs[0:3] = self.obs[0:3] + self.dt * dobs
        self.obs[2] = self._clip_angle(self.obs[2])
        self.lidar_obs = self.lidar.get_lidar_obs(self.obs[:3])
        self.count += 1
        obs = self._normalize_obs(self.obs)
        self.obs_all = np.concatenate([obs,action,self.lidar_obs],0)
        reward = self._compute_reward(action)
        done = self._check_done()
        return self.obs_all,reward,done,{}

    def _dynamics(self,action):
        dx = action[0]*np.sin(self.obs[2])
        dy = action[0]*np.cos(self.obs[2])
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

        return (self._if_goal or self._if_collision or self.count/self.SIMFREQ >= 1/4*(self._height+self._width))

    def _compute_reward(self,action):
        omegag = 0.25
        omegaw = -0.05
        omegal = 0.01


        if np.linalg.norm(self.obs[:2]-self.goal) < self.threshold_distance:
            r1 = 100.0
            self._if_goal=True
        else:
            r1 = omegag * (np.linalg.norm(self.last_pos-self.goal)-np.linalg.norm(self.obs[:2]-self.goal))

        if min(self.lidar_obs) < 0.1:
            r2 = -15.0
            self._if_collision=True
        else:
            r2 = -omegal*(1.-min(self.lidar_obs)) #implemented in num4

        if abs(action[1]) > 0.7:
            r3 = omegaw * abs(action[1])
        else:
            r3 = 0

        r4 = omegal*np.linalg.norm(self.lidar_obs)

        norm_now = self.obs_all[0:2]
        norm_goal = self.obs_all[3:5]

        r5 = -0.1*np.linalg.norm(norm_now-norm_goal) #implemented in num 5

        r6 = -1

        reward = r1 + r2 + r3 + r6


        return reward

    def _construct_graph(self):
        g = nx.Graph()
        height,width = self._height + 1,self._width+1
        # Add nodes
        for i in range(height):
            for j in range(width):
                if self._map[i,j] == 0:
                    g.add_node((i,j))

        # Add all edges
        for i in range(height):
            for j in range(width):
                for di in [-1,0,1]:
                    for dj in [-1,0,1]:
                        if di == dj == 0: continue
                        if i + di < 0 or i + di > height -1:continue
                        if j + dj < 0 or j + dj > width -1:continue
                        if not 0 in [di,dj]: continue
                        if self._map[i,j] == 1:continue
                        if self._map[i+di,j+dj] == 1:continue
                        g.add_edge((i,j),(i+di,j+dj))

        # least dis dist dist[i,j,k,l] is distance  from (i,j) to (k,l)
        dist = np.full((height,width,height,width),np.float('inf'))
        for ((i1,j1),dist_dict) in nx.shortest_path_length(g):
            for ((i2,j2),d) in dist_dict.items():
                dist[i1,j1,i2,j2] = d
        return g,dist


    def _get_goal(self):
        return self.goal

    def _get_position(self):
        return np.copy(self.obs[:2])
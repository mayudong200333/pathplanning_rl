import gym
import gym.spaces
import numpy as np
from maps.example import WALLS
from utils.tools import resize_maps

MAP = WALLS['FourRooms']

class Basic2dUAVEnv(gym.Env):
    """ Basic class of 2D navegation UAV env
    Actions:
        v:[-0.2,1.0] linear velocity of the UAV
        tau_yaw: [-1.0,1.0] angular velocity of t he UAV

    States:
        x,y,xdot,ydot,yaw,yawdot,goalsets
    """

    def __init__(self,map=MAP,resize_factor=1,freq=100):
        if resize_factor>1:
            self._map = resize_maps(map,resize_factor)
        else:
            self._map = map
        (height,width) = self._map.shape
        self._height = height
        self._width = width
        self.SIMFREQ = freq
        self.dt = 1/self.SIMFREQ
        self.action_space = gym.spaces.Box(
            low = np.array([-0.2,-1.0]),
            high= np.array([1.0,1.0]),
            dtype=np.float32
        )
        # not complimented yet (1D lidar)
        self.observation_space = gym.spaces.Box(
            low = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,]),
            high= np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,]),
            dtype= np.float32
        )

        self.reset()

    def reset(self):
        self.count = 0










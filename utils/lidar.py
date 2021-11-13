import numpy as np

class Lidar:
    ''' the simple 1Dlidar to estimate the distance from the obstacle
    Args:
        num_beams: the number of beams that generated from the lidar
        degrees: the range of the beams
        length: the length of the beams
    '''
    def __init__(self,map,num_beams=5,degrees=150,length=2.0,discrete_length=0.05,obstacle_radius=0.5):
        self.num_beams = num_beams
        self.degrees = degrees
        self.discrete_rad = np.deg2rad(self.degrees/self.num_beams)
        self.length = length
        self.obstacle_radius = obstacle_radius
        self.discrete_length = discrete_length
        self.map = map
        self.map_shape = self.map.shape

    def get_lidar_obs(self,current_states):
        current_position = current_states[:2]
        current_yaw = current_states[2]
        lidar_obs = np.zeros(self.num_beams)
        angle_start = -np.deg2rad(self.degrees)/2 + current_yaw
        for i in range(self.num_beams):
            angle = angle_start + i*self.discrete_rad
            for j in range(1,int(self.length/self.discrete_length)+1):
                length = j*self.discrete_length
                lidar_position = current_position + np.array([np.cos(angle)*length,np.sin(angle)*length])
                if self._touch_obstacle(lidar_position):
                    lidar_obs[i] = length - self.discrete_length
                    break
            else:
                lidar_obs[i] = length
        return self._normlize_obs(lidar_obs)

    def _touch_obstacle(self,position):
        position = np.round(position).astype(np.int)
        if position[0] >= self.map_shape[0] or position[1] >= self.map_shape[1] or position[0] < 0 or position[1] < 0:
            return True
        return (self.map[position[0],position[1]] == 1)

    def _normlize_obs(self,obs):
        for i in range(len(obs)):
            obs[i] = obs[i]/self.length
        return obs










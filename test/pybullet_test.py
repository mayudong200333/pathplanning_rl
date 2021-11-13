import pybullet as p
import time
import os

if __name__ == '__main__':
    phisicsClient = p.connect(p.GUI)
    p.disconnect()
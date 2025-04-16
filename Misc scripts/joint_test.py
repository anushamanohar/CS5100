import pybullet as p
import pybullet_data
import os
from env_setup import VisualRoboticArmEnv

env = VisualRoboticArmEnv(render=False)
env.reset()

for i in range(p.getNumJoints(env.robot_id)):
    joint_info = p.getJointInfo(env.robot_id, i)
    print(f"Joint {i}: {joint_info[1].decode('utf-8')}, type: {joint_info[2]}, axis: {joint_info[13]}")

env.close()

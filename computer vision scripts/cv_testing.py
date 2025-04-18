import time
import numpy as np
from stable_baselines3 import SAC
from env_setup_multiobject_cv import VisualRoboticArmEnv
import matplotlib.pyplot as plt

if __name__ == "__main__":
    env = VisualRoboticArmEnv(render=True)

    obs, _ = env.reset()  

    detected_pos = env.get_obj_position()
    true_pos = np.array([1.1, 0, 0.4])
    print(f'Box detected at world coordinates: {detected_pos}')
    print(f'3D Triangulation Error: {np.linalg.norm(detected_pos - true_pos) * 100:.4f}cm')

    

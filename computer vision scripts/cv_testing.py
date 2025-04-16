import time
import numpy as np
from stable_baselines3 import SAC
from env_setup_multiobject_cv import VisualRoboticArmEnv
import matplotlib.pyplot as plt

if __name__ == "__main__":
    env = VisualRoboticArmEnv(render=True)
   # model = SAC.load("sac_grasping_state")

    obs, _ = env.reset()  

    # terminated = False
    # truncated = False
    # episode_reward = 0

    # while not (terminated or truncated):
    #     action, _ = model.predict(obs, deterministic=True)
    #     obs, reward, terminated, truncated, _ = env.step(action) 
    #     episode_reward += reward
    #     time.sleep(1 / 600)

    # print("Test episode reward:", episode_reward)
    # env.close()

    #rgb_left, depth_left, segmentation_left, rgb_right, depth_right, segmentation_right, P1, P2 = env.get_camera_data()

    detected_pos = env.get_obj_position()
    true_pos = np.array([1.1, 0, 0.4])
    print(f'Box detected at world coordinates: {detected_pos}')
    print(f'3D Triangulation Error: {np.linalg.norm(detected_pos - true_pos) * 100:.4f}cm')
    # plt.close()
    # fig, ax = plt.subplots(2, 2)
    # ax[0, 0].imshow(rgb_left)
    # ax[0, 1].imshow(rgb_right)
    # ax[1, 0].imshow(segmentation_left)
    # ax[1, 1].imshow(segmentation_right)
    # plt.show()

    

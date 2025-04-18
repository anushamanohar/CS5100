from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np
import torch

# Tensor Board Logging 

class CustomMetricsLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        obs = self.locals['new_obs']  
        state = obs['state']

        
        if len(state.shape) > 1:
            state = state[0]  

        cube_3d_pred = state[-3:]
        ee_pos = state[12:15]
        rel_pos = state[15:18]

        distance = np.linalg.norm(rel_pos)
        grasp_success = float(self.training_env.envs[0].unwrapped._check_grasp())
        class_id = getattr(self.training_env.envs[0].unwrapped, 'last_detected_class_id', -1)

        # === Log to TensorBoard
        self.logger.record("env/distance_to_obj", distance)
        self.logger.record("env/grasp_success", grasp_success)
        self.logger.record("env/cube_3d_pred_x", cube_3d_pred[0])
        self.logger.record("env/cube_3d_pred_y", cube_3d_pred[1])
        self.logger.record("env/cube_3d_pred_z", cube_3d_pred[2])
        self.logger.record("env/cube_class_id", class_id)

        return True

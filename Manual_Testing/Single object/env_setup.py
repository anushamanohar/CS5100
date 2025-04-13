#env_setup.py

# === Core Imports ===
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import os
import time
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit
import random
import torch
import cv2
import sys

# === YOLOv5 Setup ===
YOLOV5_PATH = r"/home/shruti/CS5100/RL_training/fai_data_set/fai_data_set/yolov5"
sys.path.insert(0, YOLOV5_PATH)
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.augmentations import letterbox
from utils.torch_utils import select_device


class VisualRoboticArmEnv(gym.Env):
    def __init__(self, render=False):
        super().__init__()
        self.render = render
        self.urdf_dir = os.path.join(os.path.dirname(__file__), "urdf_files")
        self.max_steps = 50
        self.step_counter = 0
        self.grasp_counter = 0
        self.stable_grasp_steps = 0
        self.max_grasp_stability = 30
        self.grasped = False
        
        # Define the reachable workspace based on workspace analysis
        self.reachable_workspace = {
            'x_min': 0.79, 'x_max': 0.91,
            'y_min': -0.2, 'y_max': 0.3,
            'z_min': 0.4, 'z_max': 0.6
        }

        if p.isConnected():
            p.disconnect()
        self.client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        self.arm_joint_indices = [0, 1, 2, 3, 4, 5, 6]
        self.gripper_joints = [7, 8]

        self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8),
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)
        })

        # === YOLOv5 Setup ===
        self.yolo_model_path = os.path.join(YOLOV5_PATH, "runs/train/exp/weights/best.pt")
        self.device = select_device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.yolo = DetectMultiBackend(self.yolo_model_path, device=self.device)
        self.yolo_stride = self.yolo.stride
        self.yolo_img_size = 640
        self.conf_thres = 0.25
        self.iou_thres = 0.45

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1 / 240)
        p.loadURDF("plane.urdf")

        # Source table
        self.table_id = p.loadURDF(os.path.join(self.urdf_dir, "table.urdf"), [1.0, 0, 0.3], useFixedBase=True)
        
        # Tray 1
        self.tray_id = p.loadURDF(os.path.join(self.urdf_dir, "tray.urdf"), [1.1, 0.0, 0.35], useFixedBase=True)
        
        # Position the robot slightly forward for better reach
        self.robot_id = p.loadURDF(os.path.join(self.urdf_dir, "robotic_arm.urdf"),
                                   basePosition=[0.79, -0.3, 0.35],  # Moved further forward from 0.85 to 0.79
                                   baseOrientation=p.getQuaternionFromEuler([0, 0, -1.57]),
                                   useFixedBase=True)

        # Table 2 (drop here) – placed further back to avoid overlap
        self.drop_table_id = p.loadURDF(os.path.join(self.urdf_dir, "table.urdf"), [1.0, -1.0, 0.3], useFixedBase=True)
        
        self.drop_tray_id = p.loadURDF(os.path.join(self.urdf_dir, "tray.urdf"), [1.0, -0.7, 0.35], useFixedBase=True)
        # Add visual markers for reachable workspace
        for x in np.linspace(self.reachable_workspace['x_min'], 
                             self.reachable_workspace['x_max'], 5):
            for y in np.linspace(self.reachable_workspace['y_min'], 
                                 self.reachable_workspace['y_max'], 5):
                p.addUserDebugLine([x, y, 0.4], [x, y, 0.5], [0, 1, 0], 1, 3)

        # Set a better initial pose for reaching
        initial_joint_positions = [0, -0.5, 0.5, -0.5, 0.0, 0.0, 0.0]
        for i in range(7):
            p.resetJointState(self.robot_id, i, initial_joint_positions[i])

        self.ee_link_index = self._get_ee_index()
        self.gripper_joints = self._get_gripper_joint_indices()
        self.num_arm_joints = 7

        
        color_map = {
            "red": [1, 0, 0, 1],
            "green": [0, 1, 0, 1],
            "blue": [0, 0, 1, 1],
            "yellow": [1, 1, 0, 1],
        }
        selected_class = random.choice(list(color_map.keys()))
        
        # Position the cube in a known reachable position
        # This position is based on our testing and workspace analysis
        cube_pos = [0.83, 0.21, 0.45]  # Position where the robot can actually reach
                
        self.obj_id = p.loadURDF(os.path.join(self.urdf_dir, "box.urdf"), cube_pos)
        p.changeVisualShape(self.obj_id, -1, rgbaColor=color_map[selected_class])

        print("Actual cube pos:", cube_pos)

        self.step_counter = 0
        self.grasp_counter = 0
        self.stable_grasp_steps = 0
        self.grasped = False

        # Step the simulation a few times to let cube settle
        for _ in range(10):
            p.stepSimulation()

        return self._get_obs(), {}

    def _get_obs(self):
        joint_angles = [p.getJointState(self.robot_id, i)[0] for i in self.arm_joint_indices]
        joint_vels = [p.getJointState(self.robot_id, i)[1] for i in self.arm_joint_indices]
        ee_pos = p.getLinkState(self.robot_id, self.ee_link_index)[0]
        obj_pos, _ = p.getBasePositionAndOrientation(self.obj_id)
        rel_pos = np.array(obj_pos) - np.array(ee_pos)
        state = np.array(joint_angles + joint_vels + list(ee_pos) + list(rel_pos) + list(obj_pos))

        width, height = 128, 128
        fov = 60
        eye = [1.0, -0.2, 0.8]
        target = [1.0, 0.0, 0.35]
        up = [0, 0, 1]

        view_matrix = p.computeViewMatrix(eye, target, up)
        proj_matrix = p.computeProjectionMatrixFOV(fov, width / height, 0.1, 3.1)

        _, _, rgb, depth_buffer, _ = p.getCameraImage(width, height, view_matrix, proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        image = np.reshape(rgb, (height, width, 4))[:, :, :3].astype(np.uint8)
        depth = np.reshape(depth_buffer, (height, width))

        img_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_letterboxed = letterbox(img_rgb, self.yolo_img_size, stride=self.yolo_stride, auto=True)[0]

        img_input = img_letterboxed.transpose((2, 0, 1))[::-1]
        img_input = np.ascontiguousarray(img_input)
        img_tensor = torch.from_numpy(img_input).to(self.device).float() / 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        pred = self.yolo(img_tensor, augment=False, visualize=False)
        det = non_max_suppression(pred, self.conf_thres, self.iou_thres)[0]

        if det is not None and len(det):
            x1, y1, x2, y2, conf, cls = det[0].tolist()
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cx = np.clip(cx, 0, width - 1)
            cy = np.clip(cy, 0, height - 1)

            near = 0.1
            far = 3.1
            z_buffer = depth[cy, cx]
            Z = (far * near) / (far - (far - near) * z_buffer)

            fx = fy = width / (2.0 * np.tan(np.radians(fov / 2)))
            cx0, cy0 = width / 2.0, height / 2.0
            X = (cx - cx0) * Z / fx
            Y = (cy - cy0) * Z / fy
            cube_estimated_pos = np.array([X + eye[0], Y + eye[1], Z + eye[2]])

            self.last_detected_class_id = int(cls)
            p.addUserDebugLine(cube_estimated_pos, cube_estimated_pos + np.array([0, 0, 0.05]), [1, 0, 0], 1.0, 2)
        else:
            cube_estimated_pos = np.array([0.0, 0.0, 0.0])
            self.last_detected_class_id = -1

        state = np.concatenate((state, cube_estimated_pos))
        return {"image": image, "state": state}

    def _get_ee_index(self):
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            if joint_info[12].decode("utf-8") == "gripper_palm":
                return i
        return p.getNumJoints(self.robot_id) - 1

    def _get_gripper_joint_indices(self):
        names = ["left_finger_joint", "right_finger_joint"]
        return [i for i in range(p.getNumJoints(self.robot_id))
                if p.getJointInfo(self.robot_id, i)[1].decode("utf-8") in names]

    def _check_grasp(self):
        contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=self.obj_id)
        
        # More robust grasp detection - check if both fingers are in contact
        left_touching = any(c[3] == self.gripper_joints[0] for c in contacts)
        right_touching = any(c[3] == self.gripper_joints[1] for c in contacts)
        
        # Visual debug for contacts
        if contacts:
            for c in contacts:
                pos = c[6]  # Contact position on object
                p.addUserDebugLine(pos, [pos[0], pos[1], pos[2] + 0.03], [0, 1, 0], 2, 0.1)
            
        # Count as grasp if at least one finger is touching
        good_grasp = left_touching or right_touching
        
        # Update grasp counter
        self.grasp_counter = self.grasp_counter + 1 if good_grasp else max(0, self.grasp_counter - 1)
        
        # Require more consistent contact to count as grasped
        return self.grasp_counter > 3

    def _compute_reward(self):
        reward = 0.0
        ee_pos = np.array(p.getLinkState(self.robot_id, self.ee_link_index)[0])
        obj_pos, _ = p.getBasePositionAndOrientation(self.obj_id)
        dist_ee_to_obj = np.linalg.norm(ee_pos - obj_pos)
        reward -= dist_ee_to_obj * 5.0

        if self._check_grasp():
            reward += 50.0
            reward += (obj_pos[2] - 0.4) * 50.0  # Reward lifting
            drop_target_pos = np.array([0.6, -0.3, 0.3])
            dist_obj_to_drop = np.linalg.norm(obj_pos - drop_target_pos)
            reward += 10.0 * (1 - np.tanh(dist_obj_to_drop))

        if (np.linalg.norm(np.array(obj_pos[:2]) - np.array([0.6, -0.3])) < 0.1 and
                obj_pos[2] < 0.35 and not self._check_grasp()):
            reward += 200.0
            reward += 10.0 * ((self.max_steps - self.step_counter) / self.max_steps)
            return reward, True

        joint_velocities = [p.getJointState(self.robot_id, i)[1] for i in self.arm_joint_indices]
        reward -= 0.5 * np.linalg.norm(joint_velocities)
        return reward, False

    def step(self, action):
        joint_targets = np.clip(action[:7], -1, 1) * np.pi
        for idx, joint_idx in enumerate(self.arm_joint_indices):
            p.setJointMotorControl2(self.robot_id, joint_idx, p.POSITION_CONTROL,
                                    targetPosition=joint_targets[idx], force=500)

        grip = action[6]
        grip_pos = 0.04 * (1 - np.clip(grip, -1, 1)) / 2
        for gripper_joint in self.gripper_joints:
            p.setJointMotorControl2(self.robot_id, gripper_joint, p.POSITION_CONTROL, grip_pos, force=500)

        for _ in range(8):
            p.stepSimulation()

        obs = self._get_obs()
        reward, done = self._compute_reward()
        self.step_counter += 1
        return obs, reward, done or self.step_counter >= self.max_steps, False, {}

    def close(self):
        p.disconnect()


def make_env(seed=0, render=False):
    def _init():
        env = VisualRoboticArmEnv(render=render)
        env = Monitor(env)
        env = TimeLimit(env, max_episode_steps=200)
        env.reset(seed=seed)
        return env
    return _init
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

# Color mapping for visualization and YOLO detection
COLOR_MAP = {
    'red': [1, 0, 0, 1],
    'green': [0, 1, 0, 1],
    'blue': [0, 0, 1, 1],
    'yellow': [1, 1, 0, 1],
    'pink': [1, 0, 1, 1]
}


class VisualRoboticArmEnv(gym.Env):
    def __init__(self, render=False, num_objects=1):
        super().__init__()
        self.render = render
        self.urdf_dir = os.path.join(os.path.dirname(__file__), "urdf_files")
        self.max_steps = 50
        self.step_counter = 0
        self.grasp_counter = 0
        self.stable_grasp_steps = 0
        self.max_grasp_stability = 30
        self.grasped = False
        self.num_objects = num_objects  # Store number of objects to create
        
        # Define the reachable workspace based on workspace analysis
        # Restricted to values that are known to be reachable
        self.reachable_workspace = {
            'x_min': 0.79, 'x_max': 0.84,  # Narrower x range for reliability
            'y_min': 0.1, 'y_max': 0.25,    # y range that robot can reach
            'z_min': 0.4, 'z_max': 0.5
        }

        if p.isConnected():
            p.disconnect()
        self.client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        self.arm_joint_indices = [0, 1, 2, 3, 4, 5, 6]
        self.gripper_joints = [7, 8]
        
        # List to store all object IDs
        self.object_ids = []

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

        self.table_id = p.loadURDF(os.path.join(self.urdf_dir, "table.urdf"), [0.83, 0.21, 0.35], useFixedBase=True)
        
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

        # Position source tray closer to robot's reachable area
        self.tray_id = p.loadURDF(os.path.join(self.urdf_dir, "tray.urdf"), 
                                 [0.82, 0.18, 0.35],  # More reachable position
                                 useFixedBase=True)
        
        # Label trays
        try:
            source_pos, _ = p.getBasePositionAndOrientation(self.tray_id)
            dest_pos, _ = p.getBasePositionAndOrientation(self.drop_tray_id)
            
            p.addUserDebugText("SOURCE TRAY", 
                              [source_pos[0], source_pos[1], source_pos[2] + 0.1], 
                              [0, 0.8, 0.8], 
                              1.2, 
                              lifeTime=0)
            
            p.addUserDebugText("DESTINATION TRAY", 
                              [dest_pos[0], dest_pos[1], dest_pos[2] + 0.1], 
                              [0, 0, 1], 
                              1.2, 
                              lifeTime=0)
        except:
            pass
        
        # Create objects at reliably reachable positions
        self.create_multiple_objects()

        self.step_counter = 0
        self.grasp_counter = 0
        self.stable_grasp_steps = 0
        self.grasped = False

        # Step the simulation to let objects settle
        for _ in range(50):
            p.stepSimulation()
            if self.render:
                time.sleep(1/240)

        # Set the first object as the current target
        if self.object_ids:
            self.obj_id = self.object_ids[0]
        else:
            self.obj_id = None

        return self._get_obs(), {}

    def create_multiple_objects(self):
        """Create multiple objects at positions that are reliably reachable by the robot"""
        # Clear existing objects
        self.object_ids = []
    
        # Limit number of objects to a reasonable range
        num_objects = max(1, min(3, self.num_objects))
    
        # Predefined positions that are KNOWN to be reachable
        # These are specifically based on your successful position
        reliable_positions = [
            [0.83, 0.21, 0.45],  # Position 1 - Your proven position
            [0.83, 0.195, 0.45],  # Position 2 - Very close to Position 1
            [0.83, 0.225, 0.45]   # Position 3 - Also close to Position 1
        ]
    
        # Color options for easy identification
        colors = [
            [1, 0, 0, 1],  # Red
            [0, 1, 0, 1],  # Green
            [0, 0, 1, 1],  # Blue
        ]
    
        # Create objects at predefined positions
        for i in range(min(num_objects, len(reliable_positions))):
            pos = reliable_positions[i]
        
            # Create object
            obj_id = p.loadURDF(os.path.join(self.urdf_dir, "box.urdf"), pos)
        
            # Apply color
            p.changeVisualShape(obj_id, -1, rgbaColor=colors[i % len(colors)])
        
            # Add to list
            self.object_ids.append(obj_id)
        
            # Label the object
            try:
                color_name = ["RED", "GREEN", "BLUE"][i % 3]
                p.addUserDebugText(color_name, 
                            [pos[0], pos[1], pos[2] + 0.05], 
                            colors[i % len(colors)][:3],  # RGB without alpha
                            1.0, 
                            lifeTime=0)
            except:
                pass
        
            print(f"Created object {i+1} (ID: {obj_id}) at position: {pos}")
    
        # Allow objects to settle
        for _ in range(20):
            p.stepSimulation()
            if self.render:
                time.sleep(1/240)

    def _get_obs(self):
        """Get current observation state"""
        joint_angles = [p.getJointState(self.robot_id, i)[0] for i in self.arm_joint_indices]
        joint_vels = [p.getJointState(self.robot_id, i)[1] for i in self.arm_joint_indices]
        ee_pos = p.getLinkState(self.robot_id, self.ee_link_index)[0]
        
        # Get current object position
        if self.obj_id is not None:
            obj_pos, _ = p.getBasePositionAndOrientation(self.obj_id)
            rel_pos = np.array(obj_pos) - np.array(ee_pos)
        else:
            obj_pos = [0, 0, 0]
            rel_pos = [0, 0, 0]
            
        state = np.array(joint_angles + joint_vels + list(ee_pos) + list(rel_pos) + list(obj_pos))

        # Camera setup for vision
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

        # YOLO processing for object detection
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
            try:
                p.addUserDebugLine(cube_estimated_pos, cube_estimated_pos + np.array([0, 0, 0.05]), [1, 0, 0], 1.0, 2)
            except:
                pass
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
        if self.obj_id is None:
            return False
            
        contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=self.obj_id)
        
        # More robust grasp detection - check if both fingers are in contact
        left_touching = any(c[3] == self.gripper_joints[0] for c in contacts)
        right_touching = any(c[3] == self.gripper_joints[1] for c in contacts)
        
        # Visual debug for contacts
        if contacts:
            for c in contacts:
                pos = c[6]  # Contact position on object
                try:
                    p.addUserDebugLine(pos, [pos[0], pos[1], pos[2] + 0.03], [0, 1, 0], 2, 0.1)
                except:
                    pass
            
        # Count as grasp if at least one finger is touching
        good_grasp = left_touching or right_touching
        
        # Update grasp counter
        self.grasp_counter = self.grasp_counter + 1 if good_grasp else max(0, self.grasp_counter - 1)
        
        # Require more consistent contact to count as grasped
        return self.grasp_counter > 3

    def _compute_reward(self):
        reward = 0.0
        
        if self.obj_id is None:
            return reward, False
            
        ee_pos = np.array(p.getLinkState(self.robot_id, self.ee_link_index)[0])
        obj_pos, _ = p.getBasePositionAndOrientation(self.obj_id)
        dist_ee_to_obj = np.linalg.norm(ee_pos - obj_pos)
        reward -= dist_ee_to_obj * 5.0

        if self._check_grasp():
            reward += 50.0
            reward += (obj_pos[2] - 0.4) * 50.0  # Reward lifting
            
            # Target the drop tray instead of a fixed position
            drop_tray_pos, _ = p.getBasePositionAndOrientation(self.drop_tray_id)
            dist_obj_to_drop = np.linalg.norm(np.array(obj_pos[:2]) - np.array(drop_tray_pos[:2]))
            reward += 10.0 * (1 - np.tanh(dist_obj_to_drop))

        # Successful drop in destination tray
        drop_tray_pos, _ = p.getBasePositionAndOrientation(self.drop_tray_id)
        if (np.linalg.norm(np.array(obj_pos[:2]) - np.array(drop_tray_pos[:2])) < 0.3 and
                abs(obj_pos[2] - drop_tray_pos[2] - 0.04) < 0.1 and 
                not self._check_grasp()):
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
    
    def set_target_object(self, index):
        """Set the target object to a specific index in the object_ids list"""
        if 0 <= index < len(self.object_ids):
            self.obj_id = self.object_ids[index]
            return True
        return False
        
    def get_tray_positions(self):
        """Get positions of source and destination trays"""
        source_pos, _ = p.getBasePositionAndOrientation(self.tray_id)
        dest_pos, _ = p.getBasePositionAndOrientation(self.drop_tray_id)
        return source_pos, dest_pos
        
    def get_object_ids(self):
        """Get list of all object IDs"""
        return self.object_ids

    def close(self):
        p.disconnect()


def make_env(seed=0, render=False, num_objects=1):
    def _init():
        env = VisualRoboticArmEnv(render=render, num_objects=num_objects)
        env = Monitor(env)
        env = TimeLimit(env, max_episode_steps=200)
        env.reset(seed=seed)
        return env
    return _init
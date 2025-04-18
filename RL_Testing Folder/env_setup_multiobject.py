# env_setup_multiobject.py
# Outline:
# 1.  Set object position at stage 1
# 2.  Implemented staged reward system
# 3.  Defined transition conditions between stages
# 4.  Applied approach positioning rewards
# 5.  Added emergency takeover for hybrid controller
# 6.  Added forced transition after 40 steps
# 7.  Added height penalty if too high
# 8.  Added large bonus for reaching stage 1
# 9. Added stronger downward incentive in stage 1
# 10. Defined lenient takeover conditions 
# 11. Integrated yolo-based 3d object detection
# 12. Included rgb image, state vector, and yolo-estimated position in observation
# 13. Created both RL-only and hybrid control support
# 14. Added debug visuals and logging

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
import types
import datetime

from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.augmentations import letterbox
from utils.torch_utils import select_device

# === YOLOv5 Setup ===
# Note : Please make sure you change the path to the yolov5 model before you start running this code
YOLOV5_PATH = r"/home/shruti/CS5100/RL_training/fai_data_set/fai_data_set/yolov5"
sys.path.insert(0, YOLOV5_PATH)


# Color mapping for visualization and YOLO detection
COLOR_MAP = {
    'red': [1, 0, 0, 1],
    'green': [0, 1, 0, 1],
    'blue': [0, 0, 1, 1],
    'yellow': [1, 1, 0, 1],
    'pink': [1, 0, 1, 1]
}

# Class defines the main RL environment for Robotic Arm Grasping
class VisualRoboticArmEnv(gym.Env):
    def __init__(self, render=False, num_objects=1, stage=1, use_staged_rewards=True):
        super().__init__()
        self.render = render
        self.urdf_dir = os.path.join(os.path.dirname(__file__), "urdf_files")
        self.max_steps = 50
        self.step_counter = 0
        self.grasp_counter = 0
        self.stable_grasp_steps = 0
        self.grasped = False
        # Store number of objects to create
        self.num_objects = num_objects  
        self.stage = stage
        self.use_staged_rewards = use_staged_rewards
        
        # Defining the reachable workspace based on workspace analysis
        # Restricted to values that are known to be reachable
        # Run test_limitations.py script present in Misc scripts to see the limitations of the arm's reachability
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

        # Defining the action space and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)
        })

        # Setting up YOLOV5 best.pt model
        self.yolo_model_path = os.path.join(YOLOV5_PATH, "runs/train/exp/weights/best.pt")
        self.device = select_device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.yolo = DetectMultiBackend(self.yolo_model_path, device=self.device)
        self.yolo_stride = self.yolo.stride
        self.yolo_img_size = 320
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        
        #  Stage-based reward variables 
        self.reset_stage_variables()

    # Function - Calculates how close the end effector is to the approach position
    def _get_approach_progress(self, ee_pos, obj_pos):
        """Calculate progress towards ideal approach position (0 to 1)"""
        target_pos = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + 0.1])
        
        # Calculate distances
        xy_distance = np.linalg.norm(ee_pos[:2] - obj_pos[:2])
        z_diff = abs(ee_pos[2] - target_pos[2])
        
        # Calculate individual progress scores (1.0 = perfect)
        xy_score = max(0.0, 1.0 - (xy_distance / 0.2))  # Full score within 0.0m, zero at 0.2m
        z_score = max(0.0, 1.0 - (z_diff / 0.15))       # Full score within 0.0m, zero at 0.15m
        
        # Combined progress (weighted more toward XY alignment)
        progress = (xy_score * 0.7) + (z_score * 0.3)
        
        return progress

    # Resets all tracking variables for stage-based rewards
    def reset_stage_variables(self):
        """Initialize stage-based tracking variables"""
        # Stage tracking (0: approach, 1: descent, 2: grasp, 3: lift, 4: success)
        self.current_stage = 0
        
        # Position tracking
        self.initial_obj_pos = None
        self.last_ee_pos = None
        self.last_obj_pos = None
        self.distance_to_object = float('inf')
        self.min_distance_to_object = float('inf')
        self.above_object = False
        self.attempted_grasp = False
        self.grasp_progress = 0
        self.lift_progress = 0
        self.max_lift_height = 0
        
        # Stability counters
        self.stable_hover_steps = 0
        self.stable_grasp_steps = 0
        
        # Action smoothness
        self.last_action = None
        self.action_smoothness = 0
        
        # Debug visualization
        self.debug_text_id = None
        
        # Stage 1 tracking
        self.steps_in_stage1 = 0
        self.stage1_start_height = 0
    
    # Resets the whole environment and sets up urdfs    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1 / 120)
        p.loadURDF("plane.urdf")

        self.table_id = p.loadURDF(os.path.join(self.urdf_dir, "table.urdf"), [0.83, 0.21, 0.35], useFixedBase=True)
        
        # Position the robot slightly forward for better reach
        self.robot_id = p.loadURDF(os.path.join(self.urdf_dir, "robotic_arm.urdf"),
                                   basePosition=[0.79, -0.3, 0.35],  # Moved further forward from 0.85 to 0.79
                                   baseOrientation=p.getQuaternionFromEuler([0, 0, -1.57]),
                                   useFixedBase=True)

        # Table 2 (drop here) – placed further back to avoid overlap
        self.drop_table_id = p.loadURDF(os.path.join(self.urdf_dir, "table.urdf"), [1.0, -1.0, 0.3], useFixedBase=True)
        
        # Tray 2 drop zone
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
            
        # Occasionally set a better initial position closer to the target pose
        if np.random.random() < 0.2:  # 20% of the time
            print("Setting demonstration-inspired initial position")
            # Create objects first so we have their positions
            self.create_multiple_objects(stage=1)
            
            # Get object position
            if self.object_ids:
                obj_pos, _ = p.getBasePositionAndOrientation(self.object_ids[0])
                
                # Calculate a good approach position
                demo_approach_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.15]
                
                # Calculate IK for this position
                ik_solution = p.calculateInverseKinematics(
                    self.robot_id, 
                    self._get_ee_index(), 
                    demo_approach_pos,
                    maxNumIterations=100
                )
                
                # Apply the IK solution with small random noise
                for i, joint_idx in enumerate(self.arm_joint_indices):
                    if i < len(ik_solution):
                        # Add small noise to avoid deterministic behavior
                        noise = np.random.uniform(-0.1, 0.1) 
                        p.resetJointState(self.robot_id, joint_idx, 
                                          ik_solution[i] + noise)
        else:
            self.create_multiple_objects(stage=1)  

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

        self.step_counter = 0
        self.grasp_counter = 0
        self.stable_grasp_steps = 0
        self.grasped = False

        # Reset stage variables
        self.reset_stage_variables()

        # Step the simulation to let objects settle
        for _ in range(50):
            p.stepSimulation()
            if self.render:
                time.sleep(1/240)

        # Set the first object as the current target
        if self.object_ids:
            self.obj_id = self.object_ids[0]
            obj_pos, _ = p.getBasePositionAndOrientation(self.obj_id)
            approach_target = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.1]
            p.addUserDebugLine(
                [obj_pos[0], obj_pos[1], obj_pos[2]], 
                approach_target, 
                [0, 1, 0], 
                2,
                lifeTime=0
            )
            self.initial_obj_pos = np.array(obj_pos)
        else:
            self.obj_id = None
            self.initial_obj_pos = None

        return self._get_obs(), {}
    
    # Spawns objects
    def create_multiple_objects(self, stage=None):
        """Create objects with a FIXED position for more consistent training"""
        if stage is None:
            stage = self.stage
            
        self.object_ids = []
    
        base_pos = [0.83, 0.21, 0.45]
    
        obj_id = p.loadURDF(os.path.join(self.urdf_dir, "box.urdf"), base_pos)
        p.changeVisualShape(obj_id, -1, rgbaColor=[1, 0, 0, 1])
        self.object_ids.append(obj_id)
    
        return self.object_ids
    
    # Returns the current observation (image + state vector)
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
    
        # Determine if we should run YOLO in this step
        run_yolo = hasattr(self, 'step_counter') and self.step_counter % 3 == 0  # Only every 3 steps
    
        # Camera setup for vision - consistent size
        width, height = 64, 64 
        fov = 60
        eye = [1.0, -0.2, 0.8]
        target = [1.0, 0.0, 0.35]
        up = [0, 0, 1]

        view_matrix = p.computeViewMatrix(eye, target, up)
        proj_matrix = p.computeProjectionMatrixFOV(fov, width / height, 0.1, 3.1)

        _, _, rgb, depth_buffer, _ = p.getCameraImage(width, height, view_matrix, proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        image = np.reshape(rgb, (height, width, 4))[:, :, :3].astype(np.uint8)
        depth = np.reshape(depth_buffer, (height, width))

        # Initialize cube position
        cube_estimated_pos = np.array([0.0, 0.0, 0.0])
    
        # Only run YOLO if needed
        if run_yolo:
            try:
                # YOLO processing for object detection
                img_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                img_letterboxed = letterbox(img_rgb, self.yolo_img_size, stride=self.yolo_stride, auto=True)[0]

                img_input = img_letterboxed.transpose((2, 0, 1))[::-1]
                img_input = np.ascontiguousarray(img_input)
                img_tensor = torch.from_numpy(img_input).to(self.device).float() / 255.0
                if img_tensor.ndimension() == 3:
                    img_tensor = img_tensor.unsqueeze(0)
                
                # Use mixed precision for faster inference
                with torch.amp.autocast('cuda', enabled=True):
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
                    self.last_detected_pos = cube_estimated_pos
                
                    if self.render:
                        try:
                            p.addUserDebugLine(cube_estimated_pos, cube_estimated_pos + np.array([0, 0, 0.05]), [1, 0, 0], 1.0, 2)
                        except:
                            pass
                else:
                    self.last_detected_class_id = -1
                    if hasattr(self, 'last_detected_pos'):
                        # Keep using the last detection if nothing new is found
                        cube_estimated_pos = self.last_detected_pos
                    else:
                        self.last_detected_pos = np.array([0.0, 0.0, 0.0])
                    
            except Exception as e:
                print(f"YOLO error: {e} - Using ground truth position")
                # Fall back to ground truth if YOLO fails
                if self.obj_id is not None:
                    cube_estimated_pos = np.array(obj_pos)
                if hasattr(self, 'last_detected_pos'):
                    cube_estimated_pos = self.last_detected_pos
            
        else:
            # Use the last detected position when not running YOLO
            if hasattr(self, 'last_detected_pos'):
                cube_estimated_pos = self.last_detected_pos
    
        # Always concatenate the state with position estimate
        state = np.concatenate((state, cube_estimated_pos))
        return {"image": image, "state": state}
    
    # Gets index of end effector link
    def _get_ee_index(self):
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            if joint_info[12].decode("utf-8") == "gripper_palm":
                return i
        return p.getNumJoints(self.robot_id) - 1

    # returns joint indices for gripper fingers
    def _get_gripper_joint_indices(self):
        names = ["left_finger_joint", "right_finger_joint"]
        return [i for i in range(p.getNumJoints(self.robot_id))
                if p.getJointInfo(self.robot_id, i)[1].decode("utf-8") in names]

    # Checks if object is currently grasped by both fingers
    def _check_grasp(self):
        contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=self.obj_id)
    
        # Check for contacts on both fingers
        left_finger_contacts = [c for c in contacts if c[3] == self.gripper_joints[0]]
        right_finger_contacts = [c for c in contacts if c[3] == self.gripper_joints[1]]
    
        # Only count as grasp if both fingers have contact
        good_grasp = len(left_finger_contacts) > 0 and len(right_finger_contacts) > 0
    
        # Also check if object is moving with end effector
        if good_grasp:
            ee_pos = p.getLinkState(self.robot_id, self.ee_link_index)[0]
            obj_pos, _ = p.getBasePositionAndOrientation(self.obj_id)
            if np.linalg.norm(np.array(ee_pos) - np.array(obj_pos)) > 0.15:
                return False  # Too far from end effector to be a real grasp
            
        return good_grasp

    # Updates variables used to track current reward stage
    def _update_stage_variables(self, action, ee_pos, obj_pos, is_grasping, lift_height):
        """Update stage tracking for the staged reward system - MODIFIED FOR EASIER TRANSITIONS"""
        # Calculate metrics
        distance_to_object = np.linalg.norm(ee_pos - obj_pos)
        self.distance_to_object = distance_to_object
        self.min_distance_to_object = min(self.min_distance_to_object, distance_to_object)
    
        # Check if end effector is above object - MORE LENIENT
        xy_distance = np.linalg.norm(ee_pos[:2] - obj_pos[:2])
        z_relation = ee_pos[2] > obj_pos[2]
        self.above_object = xy_distance < 0.10 and z_relation  # Increased from 0.08
    
        # Track maximum lift height
        self.max_lift_height = max(self.max_lift_height, lift_height)
    
        # Stage progression logic with MORE LENIENT conditions
        if self.current_stage == 0:  # Approach stage
            # FORCE TRANSITION AFTER 40 STEPS IN STAGE 0
            if self.step_counter > 40 and xy_distance < 0.15:
                print(f"FORCED STAGE TRANSITION after {self.step_counter} steps: Approach → Descent")
                print(f"  XY Distance: {xy_distance:.4f}, Z-diff: {ee_pos[2] - obj_pos[2]:.4f}")
                self.current_stage = 1  # Force transition to descent stage
                self.stable_hover_steps = 0
                # Initialize Stage 1 tracking variables
                self.steps_in_stage1 = 0
                self.stage1_start_height = ee_pos[2]
                print("Stage 1: Positioning above object -> COMPLETE")
                print("Stage 2: Descending to grasp -> ACTIVE")
            elif self.above_object:
                # MUCH MORE LENIENT XY alignment requirement
                xy_distance = np.linalg.norm(ee_pos[:2] - obj_pos[:2])
            
                # Wider range for proper height - any reasonable height above object
                proper_height = ee_pos[2] > (obj_pos[2] + 0.03) and ee_pos[2] < (obj_pos[2] + 0.15)
            
                if xy_distance < 0.10 and proper_height:  # Increased from 0.07
                    self.stable_hover_steps += 1
                    if self.stable_hover_steps > 2:  # Reduced from 3 for faster transition
                        self.current_stage = 1  # Move to descent stage
                        # Initialize Stage 1 tracking variables
                        self.steps_in_stage1 = 0
                        self.stage1_start_height = ee_pos[2]
                        print("Stage 1: Positioning above object -> COMPLETE")
                        print("Stage 2: Descending to grasp -> ACTIVE")
                        self.stable_hover_steps = 0
                else:
                    # Reset if drifting away
                    self.stable_hover_steps = 0
    
        elif self.current_stage == 1:  # Descent stage
            # Update Stage 1 tracking
            if hasattr(self, 'steps_in_stage1'):
                self.steps_in_stage1 += 1
            else:
                self.steps_in_stage1 = 1
                self.stage1_start_height = ee_pos[2]
            
            # Check if we're descending properly
            if self.steps_in_stage1 > 10:
                total_descent = self.stage1_start_height - ee_pos[2]
                if total_descent < 0.05:  # Less than 5cm descent in 10 steps
                    print(f"WARNING: Insufficient descent in Stage 1 ({total_descent:.4f}m)")
                    
            
            # Much more lenient distance threshold
            if self.distance_to_object < 0.12:  # Increased from 0.07
                # Any closing gripper action when close should trigger transition
                if action[6] < 0.0:  # Any closing action
                    self.attempted_grasp = True
                    self.current_stage = 2
                    print("Stage 2: Descending to grasp -> COMPLETE")
                    print("Stage 3: Grasping object -> ACTIVE")
            
                    # Call explicit gripper close to help demonstration
                    for gripper_joint in self.gripper_joints:
                        p.setJointMotorControl2(self.robot_id, gripper_joint, 
                                        p.POSITION_CONTROL, 0.0, force=20000)
            
                    # Step simulation a few times for the close to take effect
                    for _ in range(5):
                        p.stepSimulation()
    
        elif self.current_stage == 2:  # Grasp stage
            if is_grasping:
                self.stable_grasp_steps += 1
                if self.stable_grasp_steps > 3:  # Reduced from 5 for faster transition
                    self.current_stage = 3  # Move to lift stage
                    print("Stage 3: Grasping object -> COMPLETE")
                    print("Stage 4: Lifting object -> ACTIVE")
            else:
                self.stable_grasp_steps = 0  # Reset if grasp lost
    
        elif self.current_stage == 3:  # Lift stage
            if lift_height > 0.01:  # Success threshold
                self.current_stage = 4  # Success stage
                print("Stage 4: Lifting object -> COMPLETE")
                print("Stage 5: SUCCESS! Object lifted")
            
        # Store state for next step
        self.last_ee_pos = ee_pos
        self.last_obj_pos = obj_pos
        self.last_action = action.copy() if action is not None else None
    
        # Update action smoothness if we have a previous action
        if self.last_action is not None and action is not None:
            self.action_smoothness = np.linalg.norm(action - self.last_action)
        
        # Visualize current stage
        self._visualize_stage(ee_pos)
        
    # Shows debug text and lines to visualize current stage
    def _visualize_stage(self, ee_pos):
        """Add enhanced visual indicators of current stage for debugging"""
        if not self.render:
            return
            
        stage_names = [
            "STAGE 1: APPROACH",
            "STAGE 2: DESCENT",
            "STAGE 3: GRASP", 
            "STAGE 4: LIFT",
            "SUCCESS!"
        ]
        
        stage_colors = [
            [0, 0.8, 0.8],  # Cyan
            [0, 0.8, 0],    # Green
            [0.8, 0.8, 0],  # Yellow
            [0.8, 0.4, 0],  # Orange
            [0, 1, 0]       # Bright Green
        ]
        
        # Remove previous debug text
        if self.debug_text_id is not None:
            try:
                p.removeUserDebugItem(self.debug_text_id)
            except:
                pass
        
        # Add new debug text
        try:
            # First - add text label for current stage
            self.debug_text_id = p.addUserDebugText(
                stage_names[self.current_stage],
                [ee_pos[0], ee_pos[1], ee_pos[2] + 0.1],
                stage_colors[self.current_stage],
                textSize=1.5,
                lifeTime=0.2
            )
            
            # Add visual indicator at object position
            if self.obj_id is not None:
                obj_pos, _ = p.getBasePositionAndOrientation(self.obj_id)
                
                # Add a vertical line to show target approach position
                if self.current_stage == 0:
                    approach_target = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.1]
                    p.addUserDebugLine(
                        obj_pos,
                        approach_target,
                        stage_colors[self.current_stage],
                        2,
                        lifeTime=0.2
                    )
                
                # Draw boundary circle for acceptable XY position
                radius = 0.1  # Matches our updated threshold
                for angle in range(0, 360, 30):  # 12 points around the circle
                    rad = angle * 3.14159 / 180
                    x1 = obj_pos[0] + radius * np.cos(rad)
                    y1 = obj_pos[1] + radius * np.sin(rad)
                    x2 = obj_pos[0] + radius * np.cos(rad + 3.14159/6)
                    y2 = obj_pos[1] + radius * np.sin(rad + 3.14159/6)
                    p.addUserDebugLine(
                        [x1, y1, obj_pos[2]],
                        [x2, y2, obj_pos[2]],
                        stage_colors[self.current_stage],
                        1,
                        lifeTime=0.2
                    )
                    
                # Add hover target position
                if self.current_stage <= 1:  # For approach and descent stages
                    hover_pos = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + 0.1])
                    p.addUserDebugText(
                        "TARGET",
                        hover_pos,
                        stage_colors[self.current_stage],
                        textSize=1.0,
                        lifeTime=0.2
                    )
        except:
            pass  
    
    # Gives reward based on current stage and task progress
    def _calculate_staged_reward(self, action, ee_pos, obj_pos, is_grasping, lift_height):
        """Calculate reward based on current stage and state - MODIFIED REWARD STRUCTURE"""
        reward = 0.0
    
        # Base penalty for time - reduced to encourage exploration
        reward -= 0.02  # Reduced from 0.05
    
        # === Stage 0: Approach object from above  ===
        if self.current_stage == 0:
            # Target position is directly above object with clearer height offset
            target_pos = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + 0.1])
            distance_to_target = np.linalg.norm(ee_pos - target_pos)
        
            xy_distance = np.linalg.norm(ee_pos[:2] - obj_pos[:2])

            # Stronger reward for approaching hover position
            reward += (1.0 - min(1.0, distance_to_target / 0.1)) * 15.0
        
            # Separate reward for proper height
            z_diff = abs(ee_pos[2] - target_pos[2])
            reward += (1.0 - min(1.0, z_diff / 0.1)) * 10.0
        
            # Extra reward for being above object
            if self.above_object:
                reward += 10.0
            
                # Bonus for stable hovering
                reward += min(10.0, self.stable_hover_steps * 1.0)
        
            # Add progress-based reward
            approach_progress = self._get_approach_progress(ee_pos, obj_pos)
            progress_reward = 100.0 * approach_progress * approach_progress
            reward += progress_reward
        
            # Keep gripper open in approach stage
            if action[6] > 0.5:  # Positive values open gripper
                reward += 3.0
            else:
                reward -= 2.0  # Penalize closed gripper in approach
                
            # Add High penalty if too high above the object
            if ee_pos[2] > obj_pos[2] + 0.3:  # If more than 30cm above the object
                height_penalty = min(50.0, (ee_pos[2] - (obj_pos[2] + 0.3)) * 100.0)  # Penalty scales with height
                reward -= height_penalty
                try:
                    if self.render:
                        p.addUserDebugText("TOO HIGH!", 
                                         [ee_pos[0], ee_pos[1], ee_pos[2] + 0.05], 
                                         [1, 0, 0], 
                                         1.5, 
                                         lifeTime=0.5)
                except:
                    pass
            
            # This prevents the agent from staying in Stage 0 indefinitely
            # Limit total Stage 0 reward to 150 plus bonuses for being in position
            # This way it's still better to move to Stage 1 than stay in Stage 0
            max_stage0_reward = 150.0
        
            # Add bonus for being very close to proper positioning
            if xy_distance < 0.10 and z_diff < 0.07:  # Made more lenient
                stage0_bonus = 80.0  # Large spike in reward for near-perfect positioning
                # Also add bonus for opening gripper when in position
                if action[6] > 0.5:  # Open gripper
                    stage0_bonus += 20.0
                
                # At this point, strongly encourage stage transition by comparing reward
                # If we're ready to transition, make the base reward slightly less
                # than what Stage 1 would offer, so there's incentive to move on
                if self.stable_hover_steps >= 2:  # Almost ready to transition
                    max_stage0_reward = 120.0  # Lower than normal Stage 1 reward
                
                reward = min(max_stage0_reward, reward) + stage0_bonus

        # === Stage 1: Descend to grasp ===
        elif self.current_stage == 1:
            # Base reward for Stage 1 - higher than max Stage 0 reward
            base_reward = 200.0
            reward += base_reward
            
            # Add large bonus for reaching stage 1
            reward += 500.0  # Big bonus just for reaching Stage 1
            print("Adding +500 reward for reaching Stage 1!")
        
            # Strong reward for alignment in XY plane 
            xy_distance = np.linalg.norm(ee_pos[:2] - obj_pos[:2])
            reward += (1.0 - min(1.0, xy_distance / 0.1)) * 15.0
        
            # Target Z-position is slightly above object
            z_target = obj_pos[2] + 0.01  # Small offset for better grasp
            z_diff = abs(ee_pos[2] - z_target)
            reward += (1.0 - min(1.0, z_diff / 0.1)) * 15.0
        
            # Reward for getting closer to object - stronger
            reward += (1.0 - min(1.0, self.distance_to_object / 0.1)) * 15.0
        
            # Strong Incentive to Downward in Stage 1
            if self.last_ee_pos is not None:
                # Reward for moving downward (negative z direction)
                z_movement = self.last_ee_pos[2] - ee_pos[2]
                if z_movement > 0:  # Moving downward
                    downward_reward = z_movement * 300.0  # Very strong reward for moving down
                    reward += downward_reward
                    if self.render and z_movement > 0.01:
                        try:
                            p.addUserDebugText(f"DOWN: +{downward_reward:.1f}", 
                                            [ee_pos[0], ee_pos[1], ee_pos[2] - 0.05], 
                                            [0, 1, 0], 
                                            1.2, 
                                            lifeTime=0.5)
                        except:
                            pass
                else:  # Moving upward or staying at same height
                    upward_penalty = abs(z_movement) * 150.0  # Penalty for moving up
                    reward -= upward_penalty
                    if self.render and z_movement < -0.01:
                        try:
                            p.addUserDebugText(f"UP: -{upward_penalty:.1f}", 
                                            [ee_pos[0], ee_pos[1], ee_pos[2] + 0.05], 
                                            [1, 0, 0], 
                                            1.2, 
                                            lifeTime=0.5)
                        except:
                            pass
                
            # Penalize being too high in Stage 1
            height_above_obj = ee_pos[2] - obj_pos[2]
            if height_above_obj > 0.2:  # If more than 20cm above object in descent stage
                height_penalty = min(100.0, (height_above_obj - 0.2) * 200.0)  # Even stronger penalty
                reward -= height_penalty
                try:
                    if self.render:
                        p.addUserDebugText(f"TOO HIGH: -{height_penalty:.1f}", 
                                        [ee_pos[0], ee_pos[1], ee_pos[2] + 0.05], 
                                        [1, 0, 0], 
                                        1.5, 
                                        lifeTime=0.5)
                except:
                    pass
                        
            # Keep gripper open during descent until very close
            if self.distance_to_object > 0.07:
                if action[6] > 0.0:  # Reward for keeping gripper open
                    reward += 5.0
            # When very close, strongly reward closing gripper
            else:
                if action[6] < 0.0:  # Strong reward for closing gripper when close
                    reward += 80.0  # Increased from 50.0
            
        # === Stage 2: Grasp object ===
        elif self.current_stage == 2:
            # Base reward for reaching Stage 2
            base_reward = 300.0
            reward += base_reward
        
            # Reward for closing gripper when close - increased
            if action[6] < -0.5 and self.distance_to_object < 0.12:  # Negative values close gripper
                reward += 40.0  # Increased from 20.0
        
            # Reward for successful grasp - increased
            if is_grasping:
                reward += 150.0  # Increased from 100.0
                reward += self.stable_grasp_steps * 10  # Increased from 5
            
            # Keep rewarding proximity to object
            reward += (1.0 - min(1.0, self.distance_to_object / 0.1)) * 15.0
            
        # === Stage 3: Lift object ===
        elif self.current_stage == 3:
            # Base reward for reaching Stage 3
            base_reward = 400.0
            reward += base_reward
        
            # Base reward for maintaining grasp - increased
            if is_grasping:
                reward += 50.0  # Increased from 30.0
            else:
                reward -= 20.0  # Same penalty for dropping
            
            # Substantial reward for lifting 
            reward += lift_height * 500.0  # Increased from 300.0
        
            # Extra reward for faster lifting
            if self.last_ee_pos is not None:
                lift_velocity = ee_pos[2] - self.last_ee_pos[2]
                if lift_velocity > 0:
                    reward += lift_velocity * 300.0  # Increased from 200.0
        
        # === Stage 4: Success ===
        elif self.current_stage == 4:
            # Large success reward 
            reward += 800.0  # Increased from 500.0
        
            # Continue rewarding lifting higher
            reward += lift_height * 300.0  # Increased from 200.0
        
            # Extra reward for keeping object stable
            if is_grasping:
                reward += 100.0  # Increased from 50.0
    
        # Global penalties to discourage unwanted behavior 
    
        # Penalty for pushing object without grasping
        if not is_grasping and self.initial_obj_pos is not None:
            xy_displacement = np.linalg.norm(obj_pos[:2] - self.initial_obj_pos[:2])
            if xy_displacement > 0.03:  # Object has moved
                reward -= xy_displacement * 20.0
    
        # Extra penalty for pushing object during approach
        if self.current_stage == 0 and not is_grasping and self.initial_obj_pos is not None:
            obj_xy_displacement = np.linalg.norm(obj_pos[:2] - self.initial_obj_pos[:2])
            if obj_xy_displacement > 0.01:  # Even slight movement is penalized
                reward -= obj_xy_displacement * 100.0
                try:
                    if self.render:
                        p.addUserDebugText("PUSHING PENALTY!", 
                                        [obj_pos[0], obj_pos[1], obj_pos[2] + 0.15], 
                                        [1, 0, 0], 
                                        1.5, 
                                        lifeTime=0.5)
                except:
                    pass
            
        # Penalty for extreme joint motions - REDUCED
        joint_velocities = np.array([p.getJointState(self.robot_id, i)[1] 
                                for i in self.arm_joint_indices])
        reward -= min(0.5, np.sum(np.abs(joint_velocities)) * 0.003)
    
        return reward
    
    # Computes the reward for each step (staged or basic)
    def _compute_reward(self):
        """Compute reward for the current state"""
        # Get positions
        ee_pos = np.array(p.getLinkState(self.robot_id, self.ee_link_index)[0])
        obj_pos, _ = p.getBasePositionAndOrientation(self.obj_id)
        obj_pos = np.array(obj_pos)
        
        # Check for grasp
        is_grasping = self._check_grasp()
        
        # Calculate lift height
        lift_height = obj_pos[2] - self.initial_obj_pos[2] if self.initial_obj_pos is not None else 0
        
        # Update stage variables if using staged rewards
        last_action = None if not hasattr(self, 'last_action') else self.last_action
        if self.use_staged_rewards:
            self._update_stage_variables(last_action, ee_pos, obj_pos, is_grasping, lift_height)
            
            # Use the staged reward calculation
            reward = self._calculate_staged_reward(last_action, ee_pos, obj_pos, is_grasping, lift_height)
            
            # Check for success condition
            done = (self.current_stage == 4)
            
            return reward, done
        
        # Original reward calculation if not using staged rewards
        else:
            # Base reward components
            reward = -np.linalg.norm(ee_pos - obj_pos) * 3.0  # Distance penalty
        
            # Detect pushing behavior - penalize horizontal motion without grasping
            if not is_grasping and obj_pos[2] < 0.47:  # Object still on surface
                obj_xy_displacement = np.linalg.norm(np.array(obj_pos[:2]) - np.array([0.83, 0.21]))
                if obj_xy_displacement > 0.05:  # Object pushed away from initial position
                    reward -= obj_xy_displacement * 20.0  # Penalty for pushing
        
            if is_grasping:
                reward += 50.0  # Reward for successful grasp
            
                # Calculate vertical lift with the gripper
                lift_height = obj_pos[2] - 0.45
            
                # Only reward lift if object moves with end effector
                ee_obj_dist = np.linalg.norm(ee_pos - obj_pos)
                if ee_obj_dist < 0.1 and lift_height > 0:  # Object is close to gripper while lifting
                    reward += lift_height * 100.0
                
                    # Success condition
                    if lift_height > 0.05:
                        reward += 200.0
                        return reward, True
        
            return reward, False

    # Applies the given action and steps simulation forward
    def step(self, action):
        # Store the action for reward calculation
        self.last_action = action.copy()
        
        # Apply joint control with smoother, lower force for more natural motion
        joint_targets = np.clip(action[:7], -1, 1) * np.pi
        for idx, joint_idx in enumerate(self.arm_joint_indices):
            p.setJointMotorControl2(self.robot_id, joint_idx, p.POSITION_CONTROL,
                                    targetPosition=joint_targets[idx], force=400)  # Reduced force

        # Apply gripper control - increased force for better grasping
        grip = action[6]
        grip_pos = 0.04 * (1 - np.clip(grip, -1, 1)) / 2
        for gripper_joint in self.gripper_joints:
            p.setJointMotorControl2(self.robot_id, gripper_joint, p.POSITION_CONTROL, grip_pos, force=800)  # Increased force

        # Step simulation with more substeps for smoother motion
        for _ in range(2):  # Increased from 8 to 10
            p.stepSimulation()
            if self.render:
                time.sleep(1/240)

        # Get observation and calculate reward
        obs = self._get_obs()
        reward, done = self._compute_reward()
        self.step_counter += 1
        
        # Prepare info dictionary with stage information if using staged rewards
        info = {}
        if self.use_staged_rewards:
            info['stage'] = self.current_stage
            info['max_lift_height'] = self.max_lift_height
            info['is_grasping'] = self._check_grasp()
            
            # Add debug info for better monitoring
            ee_pos = p.getLinkState(self.robot_id, self.ee_link_index)[0]
            obj_pos, _ = p.getBasePositionAndOrientation(self.obj_id)
            info['distance_to_object'] = np.linalg.norm(np.array(ee_pos) - np.array(obj_pos))
            info['above_object'] = self.above_object
            info['xy_distance'] = np.linalg.norm(np.array(ee_pos[:2]) - np.array(obj_pos[:2]))
        
        return obs, reward, done or self.step_counter >= self.max_steps, False, info
    
    # Changes the current target object to another index
    def set_target_object(self, index):
        """Set the target object to a specific index in the object_ids list"""
        if 0 <= index < len(self.object_ids):
            self.obj_id = self.object_ids[index]
            obj_pos, _ = p.getBasePositionAndOrientation(self.obj_id)
            self.initial_obj_pos = np.array(obj_pos)
            return True
        return False
    
    # Returns positions of source and destination trays    
    def get_tray_positions(self):
        """Get positions of source and destination trays"""
        source_pos, _ = p.getBasePositionAndOrientation(self.tray_id)
        dest_pos, _ = p.getBasePositionAndOrientation(self.drop_tray_id)
        return source_pos, dest_pos
    
    # Returns the list of all object ids in the scene
    def get_object_ids(self):
        """Get list of all object IDs"""
        return self.object_ids
    
    # Closes the simulation and disconnects from pybullet
    def close(self):
        p.disconnect()
        
# Function to create the base environment
def make_env(seed=0, render=False, num_objects=1, use_staged_rewards=True):
    def _init():
        env = VisualRoboticArmEnv(render=render, num_objects=num_objects, use_staged_rewards=use_staged_rewards)
        env = Monitor(env)
        env = TimeLimit(env, max_episode_steps=200)
        env.reset(seed=seed)
        return env
    return _init

        
# Class defines manual grasp logic over RL control using hybrid approach       
class HybridRLEnv(gym.Wrapper):
    """Environment wrapper that uses RL for positioning and manual control for grasping"""
    
    def __init__(self, env):
        super().__init__(env)
        self.transition_triggered = False
        self.manual_sequence_step = 0
        self.constraint_id = None
        self.takeover_attempts = 0  # Track takeover attempts for statistics
        
        # Create debug log file
        import datetime
        import os
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = open(f"logs/hybrid_debug_{timestamp}.csv", "w")
        self.log_file.write("step,stage,ee_x,ee_y,ee_z,obj_x,obj_y,obj_z,xy_distance,z_difference,is_good_position,is_emergency,action_grip\n")
    
    # Resets hybrid controller state variables
    def reset(self, **kwargs):
        self.transition_triggered = False
        self.manual_sequence_step = 0
        self.constraint_id = None
        self.takeover_attempts = 0
        return self.env.reset(**kwargs)
    
    # Takes a step in environment using hybrid policy logic
    def step(self, action):
        # Get current positions for debugging
        ee_pos = p.getLinkState(self.env.unwrapped.robot_id, self.env.unwrapped.ee_link_index)[0]
        if self.env.unwrapped.obj_id is not None:
            obj_pos, _ = p.getBasePositionAndOrientation(self.env.unwrapped.obj_id)
            distance = np.linalg.norm(np.array(ee_pos[:2]) - np.array(obj_pos[:2]))
            z_difference = ee_pos[2] - obj_pos[2]
            stage = self.env.unwrapped.current_stage
        else:
            obj_pos = [0, 0, 0]
            distance = float('inf')
            z_difference = 0
            stage = -1
        
        # Print action information more frequently
        if self.env.unwrapped.step_counter % 5 == 0:  # Every 5 steps
            stage_names = ["Approach", "Descent", "Grasp", "Lift", "Success"]
            stage_name = stage_names[stage] if 0 <= stage < len(stage_names) else "Unknown"
            print(f"Step {self.env.unwrapped.step_counter}: Stage {stage} ({stage_name}), "
                  f"Distance: {distance:.4f}m, Z-diff: {z_difference:.4f}m, Gripper: {'Close' if action[6] < 0 else 'Open'}")
            
            # Add additional position information
            print(f"  EE pos: {np.round(ee_pos, 4)}, Object pos: {np.round(obj_pos, 4)}")
        
        # Evaluate takeover conditions before proceeding
        should_take_over, is_emergency = self._should_take_over()
        takeover_type = "EMERGENCY" if is_emergency else "NORMAL" if should_take_over else "NONE"
        
        # Add visualization for takeover zone
        if self.env.unwrapped.render and self.env.unwrapped.obj_id is not None:
            # Draw line showing distance between EE and object
            p.addUserDebugLine(
                ee_pos,
                obj_pos,
                [1.0, 0.5, 0.5],  # Light red
                1,
                lifeTime=0.1
            )
            
            # Show takeover zone if close enough
            if distance < 0.18:  # Larger than takeover threshold for visibility
                # Color based on takeover conditions
                if should_take_over:
                    color = [0, 1, 0]  # Green for takeover ready
                elif is_emergency:
                    color = [1, 0, 0]  # Red for emergency takeover
                else:
                    color = [1, 1, 0]  # Yellow for close but not taking over
                    
                # Draw vertical line from EE showing height difference
                p.addUserDebugLine(
                    ee_pos,
                    [ee_pos[0], ee_pos[1], obj_pos[2]],
                    color,
                    2,
                    lifeTime=0.1
                )
        
        # Log data regardless of takeover
        if self.env.unwrapped.obj_id is not None:
            self.log_file.write(f"{self.env.unwrapped.step_counter},{stage},"
                               f"{ee_pos[0]:.4f},{ee_pos[1]:.4f},{ee_pos[2]:.4f},"
                               f"{obj_pos[0]:.4f},{obj_pos[1]:.4f},{obj_pos[2]:.4f},"
                               f"{distance:.4f},{z_difference:.4f},"
                               f"{should_take_over},{is_emergency},{action[6]:.4f}\n")
            self.log_file.flush()  # Force write to ensure data is saved
    
        # Let RL handle everything except when we're in position to grasp
        if not should_take_over or self.transition_triggered:
            return self.env.step(action)
    
        # If in position for grasping and not already triggered
        if should_take_over and not self.transition_triggered:
            self.transition_triggered = True
            self.takeover_attempts += 1
            print(f"HYBRID: Triggering {takeover_type} takeover at step {self.env.unwrapped.step_counter}")
            print(f"  EE pos: {np.round(ee_pos, 4)}, Object pos: {np.round(obj_pos, 4)}")
            print(f"  XY distance: {distance:.4f}m, Z difference: {z_difference:.4f}m")
        
            # Execute the proven manual grasping sequence
            obs, reward, terminated, truncated, info = self.execute_proven_grasp_sequence()
            return obs, reward, terminated, truncated, info
        
    # Decides if hybrid controller should take over
    def _should_take_over(self):
        """Determine if the controller should take over with more lenient criteria
        Returns: (should_take_over, is_emergency)"""
        if self.env.unwrapped.obj_id is None:
            return False, False
    
        # Don't take over in the first 5 steps to allow agent to position
        if self.env.unwrapped.step_counter < 5:
            return False, False
    
        ee_pos = p.getLinkState(self.env.unwrapped.robot_id, self.env.unwrapped.ee_link_index)[0]
        obj_pos, _ = p.getBasePositionAndOrientation(self.env.unwrapped.obj_id)

        # Calculate distances with lenient thresholds
        xy_distance = np.linalg.norm(np.array(ee_pos[:2]) - np.array(obj_pos[:2]))
        z_difference = ee_pos[2] - obj_pos[2]
    
        good_position = (
            xy_distance < 0.18 and      # Very wide XY alignment (was 0.16)
            z_difference < 0.45 and     # Only check if too high (don't check too low)
            z_difference > -0.08        # Allow more below object (was -0.05)
        )
    
        # Emergency takeover if poorly positioned but close
        emergency_takeover = (
            xy_distance < 0.2 and       # Close enough horizontally
            (z_difference > 0.5 or      # Way too high
             z_difference < -0.1)       # Way too low
        )
        
        # Add Stage-based hybrid takeover criteria
        # Force takeover if we've been in Stage 1 for too long
        if (self.env.unwrapped.current_stage == 1 and 
            hasattr(self.env.unwrapped, 'step_counter') and 
            self.env.unwrapped.step_counter > 40):  # Reduced from 45
            
            # Force takeover if been in Stage 1 for a while
            if xy_distance < 0.25:  # Very lenient XY threshold
                print("FORCED EMERGENCY TAKEOVER - Been in Stage 1 too long")
                return True, True
        
        # Also check for contacts - immediate takeover if touching object
        contacts = p.getContactPoints(bodyA=self.env.unwrapped.robot_id, 
                                    bodyB=self.env.unwrapped.obj_id)
        if contacts:
            print(f"FORCE TAKEOVER: Contact detected with object ({len(contacts)} points)")
            return True, False

        #  If we're in approach stage for too long, take over
        if (self.env.unwrapped.current_stage == 0 and 
            hasattr(self.env.unwrapped, 'step_counter') and 
            self.env.unwrapped.step_counter > 30 and
            xy_distance < 0.2):
            print("FORCED TAKEOVER - Been in approach stage too long")
            return True, True

        # Log takeover criteria evaluation if we're somewhat close
        if xy_distance < 0.25:  # Only log when somewhat close
            print(f"  Takeover evaluation: xy_dist={xy_distance:.4f}, z_diff={z_difference:.4f}, "
                  f"stage={self.env.unwrapped.current_stage}, good_pos={good_position}, emergency={emergency_takeover}")

        # Visual indications
        if emergency_takeover and self.env.unwrapped.render:
            try:
                p.addUserDebugText("EMERGENCY TAKEOVER", 
                              [ee_pos[0], ee_pos[1], ee_pos[2] + 0.05], 
                              [1, 0, 0], 
                              1.5, 
                              lifeTime=1.0)
            except:
                pass

        if good_position and self.env.unwrapped.render:
            try:
                p.addUserDebugText("READY TO GRASP", 
                              [ee_pos[0], ee_pos[1], ee_pos[2] + 0.05], 
                              [0, 1, 0.5], 
                              1.5, 
                              lifeTime=0.5)
            except:
                pass

        return good_position or emergency_takeover, emergency_takeover
    
    # Runs manual grasping routine step-by-step
    def execute_proven_grasp_sequence(self):
        """Execute grasp using the proven minimal approach from final_grasp_test.py"""
        print("HYBRID: Starting proven minimal grasp sequence")
        
        # Get object and robot info
        robot_id = self.env.unwrapped.robot_id
        obj_id = self.env.unwrapped.obj_id
        ee_index = self.env.unwrapped.ee_link_index
        joint_indices = self.env.unwrapped.arm_joint_indices
        gripper_joints = self.env.unwrapped.gripper_joints
        
        # 1. Stabilize the object with higher mass/damping
        try:
            p.changeDynamics(obj_id, -1,
                           mass=15.0,           # Fixed reasonable mass
                           linearDamping=0.95,   # Reduce sliding
                           angularDamping=0.95)  # Reduce rotation
            print("HYBRID: Stabilized object with fixed mass")
        except Exception as e:
            print(f"HYBRID: Warning - couldn't stabilize object: {e}")
        
        # 2. Get object position
        obj_pos, obj_orn = p.getBasePositionAndOrientation(obj_id)
        print(f"HYBRID: Object position: {np.round(obj_pos, 4)}")
        
        
        # EMERGENCY: Reset object position if it was pushed
        if abs(obj_pos[0] - 0.83) > 0.03 or abs(obj_pos[1] - 0.21) > 0.03:
            print("EMERGENCY: Object was pushed, resetting position")
            p.resetBasePositionAndOrientation(obj_id, [0.83, 0.21, 0.45], [0, 0, 0, 1])
            # Let it settle
            for _ in range(20):
                p.stepSimulation()
                if self.env.render:
                    time.sleep(1/240)
            # Update obj_pos after reset
            obj_pos, obj_orn = p.getBasePositionAndOrientation(obj_id)
        
        # Add visual markers for target positions
        try:
            p.addUserDebugText("TARGET OBJECT", 
                             [obj_pos[0], obj_pos[1], obj_pos[2] + 0.05], 
                             [0, 0.8, 0.8], 
                             1.2, 
                             lifeTime=5.0)
        except:
            pass
            
        
        # STEP 1: ENSURE GRIPPER IS OPEN WIDE
        print("HYBRID: Step 1: Opening gripper wide")
        for gripper_joint in gripper_joints:
            p.setJointMotorControl2(robot_id, gripper_joint, 
                                  p.POSITION_CONTROL, 0.04, force=2000)
                               
        # Step simulation to fully open gripper
        for _ in range(80):
            p.stepSimulation()
            if self.env.render:
                time.sleep(1/240)
        
        # STEP 2: POSITION DIRECTLY ABOVE OBJECT
        print("HYBRID: Step 2: Moving above object")
        above_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.08]  # 8cm above
        
        # Try to reach this position
        ik_solution = p.calculateInverseKinematics(robot_id, ee_index, above_pos)
        for i, joint_idx in enumerate(joint_indices):
            if i < len(ik_solution):
                p.setJointMotorControl2(robot_id, joint_idx, 
                                      p.POSITION_CONTROL, 
                                      ik_solution[i], 
                                      force=500)
        
        # Step simulation for positioning
        for _ in range(100):
            p.stepSimulation()
            if self.env.render:
                time.sleep(1/240)
                
        # STEP 3: DESCEND TO OBJECT WITH SLIGHT OFFSET
        print("HYBRID: Step 3: Descending to grasp position")
        # Calculate grasp position - right at the object with slight offset
        grasp_pos = [
            obj_pos[0],  # Slight offset for better grip
            obj_pos[1],
            obj_pos[2] + 0.01    # Slightly above surface
        ]
        
        ik_solution = p.calculateInverseKinematics(robot_id, ee_index, grasp_pos)
        for i, joint_idx in enumerate(joint_indices):
            if i < len(ik_solution):
                p.setJointMotorControl2(robot_id, joint_idx, 
                                      p.POSITION_CONTROL, 
                                      ik_solution[i], 
                                      force=400)
        
        # Step simulation for approach
        for _ in range(100):
            p.stepSimulation()
            if self.env.render:
                time.sleep(1/240)
                
        # STEP 4: CLOSE GRIPPER WITH HIGH FORCE
        print("HYBRID: Step 4: Closing gripper with high force")
        for gripper_joint in gripper_joints:
            p.setJointMotorControl2(robot_id, gripper_joint, 
                                  p.POSITION_CONTROL, 
                                  0.0,  # Fully closed
                                  force=40000)  # Very high force
        
        # Step simulation for gripping
        for _ in range(120):
            p.stepSimulation()
            if self.env.render:
                time.sleep(1/240)
            
        # Check if grasping
        is_grasping = self._check_grasp()
        if is_grasping:
            print("HYBRID: Grasp successful!")
        else:
            print("HYBRID: First grasp attempt failed, trying with more force...")
            # Try with even higher force
            for gripper_joint in gripper_joints:
                p.setJointMotorControl2(robot_id, gripper_joint, 
                                      p.POSITION_CONTROL, 
                                      0.0,  # Fully closed
                                      force=35000)  # Maximum force
            
            # Step simulation for gripping
            for _ in range(100):
                p.stepSimulation()
                if self.env.render:
                    time.sleep(1/240)
                
            is_grasping = self._check_grasp()
        
        # STEP 5: CREATE CONSTRAINT FOR LIFTING
        print("HYBRID: Step 5: Creating constraint for lift")
        constraint_id = self._create_fixed_constraint()
        if constraint_id is None:
            print("HYBRID: Failed to create constraint. Trying direct lift without constraint.")
        else:
            self.constraint_id = constraint_id
        
        # STEP 6: LIFT OBJECT WITH REDUCED GRAVITY
        print("HYBRID: Step 6: Lifting object with reduced gravity")
        start_pos, _ = p.getBasePositionAndOrientation(obj_id)
        
        # Reduce gravity temporarily for easier lifting
        p.setGravity(0, 0, -1.0)  # Reduced gravity
        
        # Calculate lift position
        lift_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.15]  # 15cm up
        
        ik_solution = p.calculateInverseKinematics(robot_id, ee_index, lift_pos)
        for i, joint_idx in enumerate(joint_indices):
            if i < len(ik_solution):
                p.setJointMotorControl2(robot_id, joint_idx, 
                                      p.POSITION_CONTROL, 
                                      ik_solution[i], 
                                      force=700)  # Higher force for lifting
        
        # Step simulation for lift
        for _ in range(150):
            p.stepSimulation()
            if self.env.render:
                time.sleep(1/240)
        
        # Restore gravity
        p.setGravity(0, 0, -9.81)  # Normal gravity
        
        # STEP 7: Hold and Check lift success
        print("HYBRID: Step 7: Checking lift success")
        for _ in range(60):
            p.stepSimulation()
            if self.env.render:
                time.sleep(1/240)
        
        # Check lift success
        lifted_pos, _ = p.getBasePositionAndOrientation(obj_id)
        lift_height = lifted_pos[2] - start_pos[2]
        print(f"HYBRID: Initial height: {start_pos[2]:.4f}m")
        print(f"HYBRID: Current height: {lifted_pos[2]:.4f}m")
        print(f"HYBRID: Lift height achieved: {lift_height:.4f}m")
        
        # Determine success
        success = lift_height > 0.01  # 5cm threshold
        
        # Visual feedback
        if success:
            print(f"HYBRID: Lift successful! Final height: {lift_height:.4f}m")
            print("HYBRID: Holding position for 3 seconds to view success...")
            try:
                p.addUserDebugText("GRASP SUCCESS!", 
                                 [lifted_pos[0], lifted_pos[1], lifted_pos[2] + 0.1], 
                                 [0, 1, 0], 
                                 1.5, 
                                 lifeTime=5.0)
                for _ in range(180):  # 3 seconds at 60 Hz
                    p.stepSimulation()
                    if self.env.render:
                        time.sleep(1/60)  # Slower simulation for better viewing
            except:
                pass
            
            # Update environment state
            self.env.unwrapped.current_stage = 4  # Success stage
            self.env.unwrapped.max_lift_height = lift_height
            
            # Release hybrid controller for next time
            self.transition_triggered = False
            
            # Get updated observation
            obs = self.env.unwrapped._get_obs()
            
            # Return with high reward and success
            return obs, 500.0, True, False, {
                'stage': 4,
                'max_lift_height': lift_height,
                'is_grasping': True,
                'success': True
            }
        else:
            print("HYBRID: Lift failed - insufficient height")
            try:
                p.addUserDebugText("GRASP FAILED", 
                                [obj_pos[0], obj_pos[1], obj_pos[2] + 0.1], 
                                [1, 0, 0], 
                                1.5, 
                                lifeTime=5.0)
            except:
                pass
            
            # Remove constraint if created
            if self.constraint_id is not None:
                try:
                    p.removeConstraint(self.constraint_id)
                    self.constraint_id = None
                except:
                    pass
            
            # Reset for next time
            self.transition_triggered = False
            
            # Get updated observation
            obs = self.env.unwrapped._get_obs()
            
            # Return with negative reward
            return obs, -10.0, False, False, {
                'stage': self.env.unwrapped.current_stage,
                'grasp_failed': True
            }
    
    # Checks grasp using contact points from both fingers
    def _check_grasp(self):
        """Check if object is grasped using contact points"""
        contacts = p.getContactPoints(bodyA=self.env.unwrapped.robot_id, bodyB=self.env.unwrapped.obj_id)
    
        # Check for contacts on both fingers
        left_finger_contacts = [c for c in contacts if c[3] == self.env.unwrapped.gripper_joints[0]]
        right_finger_contacts = [c for c in contacts if c[3] == self.env.unwrapped.gripper_joints[1]]
    
        # Only count as grasp if both fingers have contact
        good_grasp = len(left_finger_contacts) > 0 and len(right_finger_contacts) > 0
    
        # Also check if object is moving with end effector
        if good_grasp:
            ee_pos = p.getLinkState(self.env.unwrapped.robot_id, self.env.unwrapped.ee_link_index)[0]
            obj_pos, _ = p.getBasePositionAndOrientation(self.env.unwrapped.obj_id)
            if np.linalg.norm(np.array(ee_pos) - np.array(obj_pos)) > 0.15:
                return False  # Too far from end effector to be a real grasp
            
        return good_grasp
    
    # creates a fixed constraint between end effector and object# creates a fixed constraint between end effector and object
    def _create_fixed_constraint(self):
        """Create a fixed constraint between gripper and object"""
        robot_id = self.env.unwrapped.robot_id
        obj_id = self.env.unwrapped.obj_id
        ee_index = self.env.unwrapped.ee_link_index
        
        # Get positions
        ee_pos = p.getLinkState(robot_id, ee_index)[0]
        obj_pos, obj_orn = p.getBasePositionAndOrientation(obj_id)
        
        # Calculate relative position
        rel_pos = [
            obj_pos[0] - ee_pos[0], 
            obj_pos[1] - ee_pos[1],
            obj_pos[2] - ee_pos[2]
        ]
        
        try:
            # Create constraint
            constraint_id = p.createConstraint(
                parentBodyUniqueId=robot_id,
                parentLinkIndex=ee_index,
                childBodyUniqueId=obj_id,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=rel_pos,
                childFramePosition=[0, 0, 0],
                parentFrameOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                childFrameOrientation=obj_orn,
                
            )
            p.changeConstraint(constraint_id, maxForce=100.0)
            print(f"HYBRID: Created constraint ID: {constraint_id}")
            return constraint_id
        except Exception as e:
            print(f"HYBRID: Error creating constraint: {e}")
            return None
        
    # Closes hybrid controller and logs
    def close(self):
        """Clean up resources when environment is closed"""
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()
            
        # Remove constraint if it exists
        if hasattr(self, 'constraint_id') and self.constraint_id is not None:
            try:
                p.removeConstraint(self.constraint_id)
            except:
                pass
                
        super().close()
        
# Function to create the hybrid environment
def make_hybrid_env(seed=0, render=False, num_objects=1, use_staged_rewards=True):
    """Create an environment with the improved hybrid control approach"""
    def _init():
        # Create the base environment
        env = VisualRoboticArmEnv(render=render, num_objects=num_objects, use_staged_rewards=use_staged_rewards)
        
        # Wrap with the hybrid controller
        env = HybridRLEnv(env)
        
        # Add standard wrappers
        env = Monitor(env)
        env = TimeLimit(env, max_episode_steps=200)
        env.reset(seed=seed)
        return env
    return _init

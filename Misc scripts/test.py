import time
import pybullet as p
import numpy as np
from env_setup import make_env

# === Config ===
TRAY_HEIGHT = 0.4
DESCENT_OFFSET = 0.005
MAX_ERROR_THRESHOLD = 0.025
MAX_IK_RETRIES = 15

def modify_robot_position(env, target_object_pos, reach_distance=0.25):
    """
    Modify the robot's base position to ensure the target object is within reach.
    
    Args:
        env: The simulation environment
        target_object_pos: Position of the target object [x, y, z]
        reach_distance: Desired distance from robot base to target (meters)
        
    Returns:
        dict: Information about the position adjustment
    """
    robot_id = env.unwrapped.robot_id
    
    # Get current robot base position and orientation
    original_pos, original_orn = p.getBasePositionAndOrientation(robot_id)
    print(f"Original robot base position: {np.round(original_pos, 4)}")
    
    # Calculate optimal position based on target
    # For this robot, we want to be closer in X direction
    # and properly positioned in Y direction
    
    # We'll keep the same height (Z) and orientation
    desired_x = target_object_pos[0] - reach_distance  # Closer in X direction
    desired_y = target_object_pos[1]  # Aligned in Y direction
    desired_z = original_pos[2]  # Same height
    
    new_position = [desired_x, desired_y, desired_z]
    
    # Apply the new position
    p.resetBasePositionAndOrientation(robot_id, new_position, original_orn)
    
    # Let simulation stabilize
    for _ in range(20):
        p.stepSimulation()
        if env.render:
            time.sleep(1/240)
    
    # Get and print the new position
    new_pos, new_orn = p.getBasePositionAndOrientation(robot_id)
    print(f"New robot base position: {np.round(new_pos, 4)}")
    
    # Calculate distance to target
    distance_to_target = np.linalg.norm(np.array(target_object_pos) - np.array(new_pos))
    print(f"Distance to target object: {distance_to_target:.4f}m")
    
    # Return adjustment info
    return {
        "original_position": original_pos,
        "new_position": new_pos,
        "distance_to_target": distance_to_target
    }

def move_to_position(env, target_pos, max_retries=20, threshold=0.02, debug=True):
    """
    Move the end effector to a target position.
    
    Args:
        env: The simulation environment
        target_pos: Target position for end effector [x, y, z]
        max_retries: Maximum number of attempts
        threshold: Success distance threshold
        debug: Whether to print debug information
        
    Returns:
        bool: Success or failure
    """
    robot_id = env.unwrapped.robot_id
    ee_index = env.unwrapped.ee_link_index
    joint_indices = env.unwrapped.arm_joint_indices
    
    # Get joint limits for the robot
    joint_limits_lower = []
    joint_limits_upper = []
    for i in joint_indices:
        info = p.getJointInfo(robot_id, i)
        joint_limits_lower.append(info[8])  # Lower limit
        joint_limits_upper.append(info[9])  # Upper limit
    
    if debug:
        print(f"\n🚀 Moving to target: {np.round(target_pos, 4)}")
    
    # Step-by-step approach
    for i in range(max_retries):
        # Get current end effector position
        ee_pos = p.getLinkState(robot_id, ee_index)[0]
        
        # Calculate error
        error = np.array(target_pos) - np.array(ee_pos)
        distance = np.linalg.norm(error)
        
        if debug and (i % 2 == 0 or distance < threshold * 2):
            print(f"  Iteration {i}: EE at {np.round(ee_pos, 4)}, distance: {distance:.4f}")
        
        # Check if we've reached the target
        if distance < threshold:
            if debug:
                print(f"✅ Reached target position! Final distance: {distance:.4f}")
            return True
        
        # Calculate IK for target position
        ik_solution = p.calculateInverseKinematics(
            robot_id, 
            ee_index, 
            target_pos,
            lowerLimits=joint_limits_lower,
            upperLimits=joint_limits_upper,
            jointRanges=[upper - lower for lower, upper in zip(joint_limits_lower, joint_limits_upper)],
            maxNumIterations=200,
            residualThreshold=1e-5
        )
        
        # Apply the joint positions
        for j, joint_idx in enumerate(joint_indices):
            # Limit to valid joint range
            target_angle = max(min(ik_solution[j], joint_limits_upper[j]), joint_limits_lower[j])
            p.setJointMotorControl2(
                robot_id, 
                joint_idx, 
                p.POSITION_CONTROL, 
                targetPosition=target_angle,
                force=500  # High force for precise control
            )
        
        # Step simulation
        for _ in range(10):
            p.stepSimulation()
            if env.render:
                time.sleep(1/240)
        
        # If we're getting stuck, try a different approach
        if i > max_retries // 2 and distance > threshold * 2:
            # Try a slightly different target on direct line to target
            if debug:
                print("⚠️ Progress slow - adjusting approach")
            
            # Get current joint positions
            current_joints = [p.getJointState(robot_id, j)[0] for j in joint_indices]
            
            # Try more incremental joint change
            blend_factor = 0.2  # Take smaller steps in joint space
            for j, joint_idx in enumerate(joint_indices):
                current = current_joints[j]
                target = ik_solution[j]
                # Take small step toward target joint position
                adjusted = current + blend_factor * (target - current)
                adjusted = max(min(adjusted, joint_limits_upper[j]), joint_limits_lower[j])
                p.setJointMotorControl2(robot_id, joint_idx, p.POSITION_CONTROL, adjusted, force=300)
                
            for _ in range(10):
                p.stepSimulation()
                if env.render:
                    time.sleep(1/240)
    
    # If we get here, we failed to reach the target
    if debug:
        print(f"❌ Failed to reach target position. Final distance: {distance:.4f}")
    return False

def control_gripper(env, closing=True, grip_force=600, steps=100):
    """
    Control the gripper to open or close.
    
    Args:
        env: The simulation environment
        closing: True to close, False to open
        grip_force: Force to apply (higher for closing)
        steps: Simulation steps to run
    """
    left, right = env.unwrapped.gripper_joints
    pos = 0.0 if closing else 0.04
    p.setJointMotorControl2(env.unwrapped.robot_id, left, p.POSITION_CONTROL, pos, force=grip_force)
    p.setJointMotorControl2(env.unwrapped.robot_id, right, p.POSITION_CONTROL, pos, force=grip_force)
    
    for _ in range(steps):
        p.stepSimulation()
        if env.render:
            time.sleep(1/240)

def check_contact(env):
    """
    Check for contact between the robot and the target object.
    
    Returns:
        bool: True if contact detected
    """
    contacts = p.getContactPoints(env.unwrapped.robot_id, env.unwrapped.obj_id)
    
    if contacts:
        print(f"✅ Found {len(contacts)} contact points")
        
        # Visualize contacts
        for contact in contacts:
            pos = contact[5]  # Contact position
            p.addUserDebugLine(
                pos, 
                [pos[0], pos[1], pos[2] + 0.02], 
                [0, 1, 0], 
                2, 
                lifeTime=1.0
            )
        return True
    else:
        print("⚠️ No contact detected")
        return False

def run_adjusted_base_grasp():
    """
    Run the complete grasping sequence with robot base position adjustment.
    """
    # Initialize environment
    env = make_env(render=True)()
    obs, _ = env.reset()
    
    # Get object position
    obj_id = env.unwrapped.obj_id
    cube_pos, _ = p.getBasePositionAndOrientation(obj_id)
    print(f"\n==== Grasp Sequence ====")
    print(f"Target object position: {np.round(cube_pos, 4)}")
    
    # Step 1: Adjust robot base position to ensure cube is reachable
    print("\n🚀 Step 1: Adjusting robot base position")
    adjust_info = modify_robot_position(env, cube_pos, reach_distance=0.25)
    
    # Initialize robot joints to a good starting position
    print("\n🚀 Step 2: Setting initial joint positions")
    initial_joints = [0, -0.4, 0.6, -0.6, 0.0, 0.0, 0.0]
    for i, joint_idx in enumerate(env.unwrapped.arm_joint_indices):
        p.resetJointState(env.unwrapped.robot_id, joint_idx, initial_joints[i])
    
    # Step simulation to stabilize
    for _ in range(30):
        p.stepSimulation()
        if env.render:
            time.sleep(1/240)
    
    # Step 3: Open gripper
    print("\n🚀 Step 3: Opening gripper")
    control_gripper(env, closing=False)
    
    # Step 4: Move above the cube
    print("\n🚀 Step 4: Moving above the cube")
    above_cube = np.array(cube_pos) + np.array([0, 0, 0.15])
    if not move_to_position(env, above_cube):
        print("❌ Failed to position above cube")
        env.close()
        return False
    
    # Step 5: Descend to grasp
    print("\n🚀 Step 5: Descending to grasp position")
    grasp_position = np.array(cube_pos) + np.array([0, 0, -0.005])
    if not move_to_position(env, grasp_position):
        print("❌ Failed to reach grasp position")
        env.close()
        return False
    
    # Step 6: Check for contact
    print("\n🚀 Step 6: Checking for contact")
    has_contact = check_contact(env)
    
    # Step 7: Close gripper
    print("\n🚀 Step 7: Closing gripper")
    control_gripper(env, closing=True, grip_force=800, steps=200)
    
    # Step 8: Lift object
    print("\n🚀 Step 8: Lifting object")
    lift_position = np.array(cube_pos) + np.array([0, 0, 0.2])
    move_to_position(env, lift_position)
    
    # Check result
    final_cube_pos, _ = p.getBasePositionAndOrientation(obj_id)
    lift_height = final_cube_pos[2] - cube_pos[2]
    
    print("\n==== Grasp Result ====")
    print(f"Initial cube height: {cube_pos[2]:.4f}")
    print(f"Final cube height: {final_cube_pos[2]:.4f}")
    print(f"Lift distance: {lift_height:.4f}")
    
    if lift_height > 0.1:
        print("✅ Grasp successful!")
        success = True
    elif lift_height > 0.05:
        print("⚠️ Partial grasp - cube lifted but not securely")
        success = True
    else:
        print("❌ Grasp failed - cube remains on table")
        success = False
    
    # Keep simulation open until user exits
    input("\n🔚 Press ENTER to exit")
    env.close()
    
    return success

if __name__ == "__main__":
    run_adjusted_base_grasp()
import time
import pybullet as p
import numpy as np
from env_setup import make_env

def simple_grasp(env):
    """
    A robust grasp approach that acknowledges kinematic limitations
    """
    robot_id = env.unwrapped.robot_id
    obj_id = env.unwrapped.obj_id
    
    # Get object position
    cube_pos, _ = p.getBasePositionAndOrientation(obj_id)
    print(f"\n📊 Cube position: {np.round(cube_pos, 4)}")
    
    # Get robot configuration information
    joint_indices = env.unwrapped.arm_joint_indices
    ee_index = env.unwrapped.ee_link_index
    gripper_joints = env.unwrapped.gripper_joints
    
    print(f"📊 Robot base position: {p.getBasePositionAndOrientation(robot_id)[0]}")
    print(f"📊 Joint indices: {joint_indices}")
    print(f"📊 End effector index: {ee_index}")
    print(f"📊 Gripper joints: {gripper_joints}")
    
    # ------------------------------
    # Step 1: Reset to a known position
    # ------------------------------
    print("\n🚀 Step 1: Reset to home position")
    
    # Reset to a known good configuration - adjust these values as needed
    reset_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    for i, joint_idx in enumerate(joint_indices):
        p.resetJointState(robot_id, joint_idx, reset_angles[i])
    
    # Let the simulation settle
    for _ in range(20):
        p.stepSimulation()
        if env.render:
            time.sleep(1/240)
    
    # Get current EE position after reset
    ee_pos = p.getLinkState(robot_id, ee_index)[0]
    print(f"  EE position after reset: {np.round(ee_pos, 4)}")
    
    # ------------------------------
    # Step 2: Open gripper
    # ------------------------------
    print("\n🚀 Step 2: Opening gripper")
    
    for j in gripper_joints:
        p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, 0.04, force=300)
    
    for _ in range(20):
        p.stepSimulation()
        if env.render:
            time.sleep(1/240)
    
    # ------------------------------
    # Step 3: Calculate best approach angle
    # ------------------------------
    print("\n🚀 Step 3: Calculating approach angle")
    
    # Calculate relative position (cube to robot base)
    robot_base_pos = p.getBasePositionAndOrientation(robot_id)[0]
    relative_pos = np.array(cube_pos) - np.array(robot_base_pos)
    
    # Calculate angle to object in XY plane (for base rotation)
    angle_to_obj = np.arctan2(relative_pos[1], relative_pos[0])
    distance_xy = np.sqrt(relative_pos[0]**2 + relative_pos[1]**2)
    
    print(f"  Relative position: {np.round(relative_pos, 4)}")
    print(f"  Angle to object: {np.round(angle_to_obj, 4)} radians")
    print(f"  XY distance: {np.round(distance_xy, 4)} meters")
    
    # ------------------------------
    # Step 4: Multi-stage approach using pre-calculated joint positions
    # ------------------------------
    print("\n🚀 Step 4: Multi-stage approach")
    
    # Stage 1: Orient base toward object
    print("  Stage 1: Orient toward object")
    target_joints = reset_angles.copy()
    target_joints[0] = angle_to_obj  # Base rotation joint
    
    for i, joint_idx in enumerate(joint_indices):
        p.setJointMotorControl2(robot_id, joint_idx, p.POSITION_CONTROL, target_joints[i], force=300)
    
    for _ in range(40):
        p.stepSimulation()
        if env.render:
            time.sleep(1/240)
    
    # Check the end effector position
    ee_pos = p.getLinkState(robot_id, ee_index)[0]
    print(f"  EE position after orientation: {np.round(ee_pos, 4)}")
    
    # Stage 2: Position arm for reach
    print("  Stage 2: Position arm for reach")
    
    # Suggested joint configuration based on typical robot arm kinematics
    # These need to be tuned for your specific robot
    reach_joints = [
        angle_to_obj,       # Base rotation
        0.4,                # Shoulder angle - raised
        0.8,                # Elbow angle - extended 
        -0.5,               # Wrist angle
        0.0,                # Wrist roll
        0.5,                # Wrist pitch
        0.0                 # Any additional joint
    ]
    
    for i, joint_idx in enumerate(joint_indices):
        if i < len(reach_joints):
            p.setJointMotorControl2(robot_id, joint_idx, p.POSITION_CONTROL, reach_joints[i], force=400)
    
    for _ in range(60):
        p.stepSimulation()
        if env.render:
            time.sleep(1/240)
    
    # Stage A: Get current EE position
    ee_pos = p.getLinkState(robot_id, ee_index)[0]
    ee_to_cube = np.array(cube_pos) - np.array(ee_pos)
    distance = np.linalg.norm(ee_to_cube)
    
    print(f"  Current EE position: {np.round(ee_pos, 4)}")
    print(f"  Distance to cube: {np.round(distance, 4)}")
    
    # Get current joint positions
    current_joints = [p.getJointState(robot_id, j)[0] for j in joint_indices]
    
    # Stage 3: Extend arm toward object
    print("  Stage 3: Extend toward object")
    
    extend_joints = current_joints.copy()
    
    # Modify the joint angles to move closer to the cube
    # Focus on elbow extension and shoulder angle
    extend_joints[1] += 0.2  # Adjust shoulder joint
    extend_joints[2] += 0.3  # Extend elbow joint
    extend_joints[3] -= 0.2  # Adjust wrist to compensate
    
    for i, joint_idx in enumerate(joint_indices):
        if i < len(extend_joints):
            p.setJointMotorControl2(robot_id, joint_idx, p.POSITION_CONTROL, extend_joints[i], force=300)
    
    for _ in range(50):
        p.stepSimulation()
        if env.render:
            time.sleep(1/240)
    
    # Stage B: Get current EE position
    ee_pos = p.getLinkState(robot_id, ee_index)[0]
    ee_to_cube = np.array(cube_pos) - np.array(ee_pos)
    distance = np.linalg.norm(ee_to_cube)
    
    print(f"  Current EE position: {np.round(ee_pos, 4)}")
    print(f"  Distance to cube: {np.round(distance, 4)}")
    
    # Get current joint positions
    current_joints = [p.getJointState(robot_id, j)[0] for j in joint_indices]
    
    # Stage 4: Fine-tune position 
    print("  Stage 4: Fine-tune position")
    
    # Try multiple minor adjustments to get closer
    for attempt in range(3):
        # Get current state
        ee_pos = p.getLinkState(robot_id, ee_index)[0]
        ee_to_cube = np.array(cube_pos) - np.array(ee_pos)
        distance = np.linalg.norm(ee_to_cube)
        
        # Skip if we're already close
        if distance < 0.08:
            print(f"  Already close enough: {np.round(distance, 4)}")
            break
            
        # Get current joint positions
        current_joints = [p.getJointState(robot_id, j)[0] for j in joint_indices]
        fine_tune_joints = current_joints.copy()
        
        # Make adjustments based on relative position
        if ee_to_cube[0] > 0.05:  # Need to move forward
            fine_tune_joints[2] += 0.1  # Extend elbow
        
        if ee_to_cube[2] > 0.05:  # Need to move up
            fine_tune_joints[1] += 0.1  # Raise shoulder
            fine_tune_joints[3] -= 0.1  # Adjust wrist
        elif ee_to_cube[2] < -0.05:  # Need to move down
            fine_tune_joints[1] -= 0.1  # Lower shoulder
            fine_tune_joints[3] += 0.1  # Adjust wrist
            
        if abs(ee_to_cube[1]) > 0.05:  # Need to adjust laterally
            fine_tune_joints[0] += 0.1 * np.sign(ee_to_cube[1])  # Rotate base
            
        # Apply adjustments
        for i, joint_idx in enumerate(joint_indices):
            if i < len(fine_tune_joints):
                p.setJointMotorControl2(robot_id, joint_idx, p.POSITION_CONTROL, fine_tune_joints[i], force=200)
        
        # Let the adjustments take effect
        for _ in range(30):
            p.stepSimulation()
            if env.render:
                time.sleep(1/240)
                
        # Log result
        ee_pos = p.getLinkState(robot_id, ee_index)[0]
        distance = np.linalg.norm(np.array(ee_pos) - np.array(cube_pos))
        print(f"  Fine-tune attempt {attempt+1}: distance = {np.round(distance, 4)}")
    
    # ------------------------------
    # Step 5: Check for contact and close gripper
    # ------------------------------
    print("\n🚀 Step 5: Checking for contact")
    
    contacts = p.getContactPoints(robot_id, obj_id)
    
    if contacts:
        print(f"  ✅ Found {len(contacts)} contact points")
        
        # Visualize contacts
        for contact in contacts:
            pos = contact[5]
            p.addUserDebugLine(
                pos, 
                [pos[0], pos[1], pos[2] + 0.05], 
                [0, 1, 0], 
                2, 
                lifeTime=2.0
            )
    else:
        print("  ⚠️ No contact points found - trying final approach")
        
        # Final approach - very small step
        current_joints = [p.getJointState(robot_id, j)[0] for j in joint_indices]
        final_approach = current_joints.copy()
        
        # Slight extension of arm
        final_approach[2] += 0.05  # Tiny elbow extension
        
        for i, joint_idx in enumerate(joint_indices):
            if i < len(final_approach):
                p.setJointMotorControl2(robot_id, joint_idx, p.POSITION_CONTROL, final_approach[i], force=100)
        
        for _ in range(30):
            p.stepSimulation()
            if env.render:
                time.sleep(1/240)
    
    # ------------------------------
    # Step 6: Close gripper
    # ------------------------------
    print("\n🚀 Step 6: Closing gripper")
    
    for j in gripper_joints:
        p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, 0.0, force=1000)
    
    for _ in range(60):
        p.stepSimulation()
        if env.render:
            time.sleep(1/240)
    
    # Verify gripper closure
    gripper_pos = [p.getJointState(robot_id, j)[0] for j in gripper_joints]
    print(f"  Gripper positions: {np.round(gripper_pos, 4)}")
    
    # ------------------------------
    # Step 7: Lift object (if grasped)
    # ------------------------------
    print("\n🚀 Step 7: Lifting object")
    
    # Record initial cube position for comparison
    initial_cube_pos = p.getBasePositionAndOrientation(obj_id)[0]
    
    # Get current joint positions
    current_joints = [p.getJointState(robot_id, j)[0] for j in joint_indices]
    lift_joints = current_joints.copy()
    
    # Modify joints to lift
    lift_joints[1] -= 0.3  # Retract shoulder
    
    for i, joint_idx in enumerate(joint_indices):
        if i < len(lift_joints):
            p.setJointMotorControl2(robot_id, joint_idx, p.POSITION_CONTROL, lift_joints[i], force=500)
    
    for _ in range(80):
        p.stepSimulation()
        if env.render:
            time.sleep(1/240)
    
    # ------------------------------
    # Step 8: Check result
    # ------------------------------
    final_cube_pos = p.getBasePositionAndOrientation(obj_id)[0]
    lift_height = final_cube_pos[2] - initial_cube_pos[2]
    
    print("\n==== Grasp Result ====")
    print(f"📍 Initial cube height: {initial_cube_pos[2]:.4f}")
    print(f"📍 Final cube height: {final_cube_pos[2]:.4f}")
    print(f"📍 Lift distance: {lift_height:.4f}")
    
    if lift_height > 0.1:
        print("✅ Grasp successful!")
    elif lift_height > 0.05:
        print("⚠️ Partial grasp - cube lifted but not securely")
    else:
        print("❌ Grasp failed - cube remains on table")
        
    return lift_height > 0.05  # Return True if at least partially successful

def run_grasp_demo():
    """Run the grasp demo"""
    env = make_env(render=True)()
    obs, _ = env.reset()
    
    # Run the grasp
    success = simple_grasp(env)
    
    # Keep simulation running until user exits
    input("\n🔚 Press ENTER to exit")
    env.close()
    
    return success

if __name__ == "__main__":
    run_grasp_demo()
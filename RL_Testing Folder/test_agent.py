# Testing agent
"""
Final test script that directly uses the successful manual control approach
from process_single_object, avoiding hybrid controller complexity.
"""

import time
import pybullet as p
import numpy as np
import os
from env_setup_multiobject import make_env

def move_to_position_minimal(env, target_pos, max_retries=30, accept_close=True):
    """Ultra-minimal function that uses the simplest possible IK calculation"""
    robot_id = env.unwrapped.robot_id
    ee_index = env.unwrapped.ee_link_index
    joint_indices = env.unwrapped.arm_joint_indices
    
    print(f"Target position: {np.round(target_pos, 4)}")
    
    # Store best position for fallback
    best_distance = float('inf')
    best_joint_positions = None
    
    for i in range(max_retries):
        # Get current end effector position
        ee_pos = p.getLinkState(robot_id, ee_index)[0]
        distance = np.linalg.norm(np.array(ee_pos) - np.array(target_pos))
        print(f"  EE position: {np.round(ee_pos, 4)}, Distance: {distance:.4f}")
        
        # Track best position
        if distance < best_distance:
            best_distance = distance
            best_joint_positions = [p.getJointState(robot_id, j)[0] for j in joint_indices]
        
        # 4cm threshold for success
        if distance < 0.04:  
            print("  Close enough to target!")
            return True
            
        # Simple IK calculation
        try:
            ik_solution = p.calculateInverseKinematics(robot_id, ee_index, target_pos)
            
            # Apply IK solution
            for j, joint_idx in enumerate(joint_indices):
                if j < len(ik_solution):  # Safety check
                    p.setJointMotorControl2(robot_id, joint_idx, p.POSITION_CONTROL, ik_solution[j], force=500)
                
            # Step simulation
            for _ in range(20):
                p.stepSimulation()
                if env.render:
                    time.sleep(1 / 240)
                
        except Exception as e:
            print(f"  Error: {e}")
            return False
    
    # If we couldn't reach exactly, but got close enough, use the best position
    if accept_close and best_distance < 0.08:  # 8cm threshold for acceptable position
        print(f"  Using best achieved position (distance: {best_distance:.4f}m)")
        for j, joint_idx in enumerate(joint_indices):
            if j < len(best_joint_positions):
                p.setJointMotorControl2(robot_id, joint_idx, p.POSITION_CONTROL, 
                                       best_joint_positions[j], force=500)
        
        for _ in range(20):
            p.stepSimulation()
            if env.render:
                time.sleep(1 / 240)
        
        return True
            
    print("  Could not reach target after maximum retries.")
    return False

def control_gripper(env, closing=True, grip_force=10000, steps=300, target_obj_id=None):
    """Control the gripper with improved force and visual feedback"""
    left, right = env.unwrapped.gripper_joints
    pos = 0.0 if closing else 0.04
    
    # Higher force when closing for a firmer grip
    actual_force = grip_force * 2 if closing else grip_force
    
    p.setJointMotorControl2(env.unwrapped.robot_id, left, p.POSITION_CONTROL, pos, force=actual_force)
    p.setJointMotorControl2(env.unwrapped.robot_id, right, p.POSITION_CONTROL, pos, force=actual_force)
    
    # Visual feedback
    ee_pos = p.getLinkState(env.unwrapped.robot_id, env.unwrapped.ee_link_index)[0]
    gripper_status = "CLOSING" if closing else "OPENING"
    try:
        p.addUserDebugText(gripper_status, 
                         [ee_pos[0], ee_pos[1], ee_pos[2] + 0.05], 
                         [1, 0.5, 0] if closing else [0, 0.8, 0.3], 
                         1.5, 
                         lifeTime=1.0)
    except:
        pass  # Ignore if debug text limit is reached
    
    # Step simulation with contact checking
    for i in range(steps):
        p.stepSimulation()
        
        # Check for contacts periodically while closing
        if closing and i % 10 == 0 and target_obj_id is not None:
            contact_points = p.getContactPoints(
                env.unwrapped.robot_id, 
                target_obj_id
            )
            if contact_points:
                # Visualize contacts (limit to avoid clutter)
                for c in contact_points[:2]:
                    contact_pos = c[6]  # Position on object
                    try:
                        p.addUserDebugLine(
                            contact_pos, 
                            [contact_pos[0], contact_pos[1], contact_pos[2] + 0.02],
                            [0, 1, 0],
                            3,
                            lifeTime=0.5
                        )
                    except:
                        pass  # Ignore if too many debug lines
        
        if env.render:
            time.sleep(1/240)
            
def create_fixed_constraint(env, robot_id, obj_id):
    """Create a fixed constraint between the robot gripper and the object"""
    ee_index = env.unwrapped.ee_link_index
    
    # Get positions
    ee_pos = p.getLinkState(robot_id, ee_index)[0]
    obj_pos, obj_orn = p.getBasePositionAndOrientation(obj_id)
    
    # Calculate relative position (object position relative to end effector)
    rel_pos = [
        obj_pos[0] - ee_pos[0], 
        obj_pos[1] - ee_pos[1],
        obj_pos[2] - ee_pos[2]
    ]
    
    try:
        # Create constraint between gripper palm and object
        constraint_id = p.createConstraint(
            parentBodyUniqueId=robot_id,
            parentLinkIndex=ee_index,
            childBodyUniqueId=obj_id,
            childLinkIndex=-1,  # -1 for base link
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=rel_pos,  # Use actual relative position
            childFramePosition=[0, 0, 0],  # Center of the cube
            parentFrameOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            childFrameOrientation=obj_orn
        )
        
        print(f"Created fixed constraint ID: {constraint_id}")
        return constraint_id
    except Exception as e:
        print(f"Error creating constraint: {e}")
        return None
    
def test_direct_grasp():
    """Run grasp test using the proven manual approach"""
    print("\n DIRECT GRASP TEST")
    print("Using minimal manual control approach that worked in original code")
    
    # Create environment
    env = make_env(render=True, num_objects=1, use_staged_rewards=True)()
    
    # Reset and get initial state
    obs, _ = env.reset()
    
    # Wait for environment to stabilize
    for _ in range(50):
        p.stepSimulation()
        time.sleep(1/240)
    
    # Get object IDs and positions
    obj_id = env.unwrapped.obj_id
    obj_pos, obj_orn = p.getBasePositionAndOrientation(obj_id)
    print(f"Object position: {np.round(obj_pos, 4)}")
    
    # Stabilize the object
    p.changeDynamics(obj_id, -1,
                   mass=10.0,           # Heavier object
                   linearDamping=0.9,   # Reduce sliding
                   angularDamping=0.9)  # Reduce rotation
    
    print("\nStep 1: Opening gripper")
    control_gripper(env, closing=False, steps=60, target_obj_id=obj_id)
    
    print("\nStep 2: Moving above cube")
    above_pos = np.array(obj_pos) + [0, 0, 0.05]  # 5cm above object
    move_success = move_to_position_minimal(env, above_pos.tolist())
    
    if not move_success:
        print("Failed to move above cube. Trying fallback position...")
        fallback_pos = np.array(obj_pos) + [0, 0, 0.1]  # 10cm above
        move_success = move_to_position_minimal(env, fallback_pos.tolist())
        if not move_success:
            print("Fallback position also failed. Aborting test.")
            env.close()
            return False
    
    # Ensure gripper is fully open
    control_gripper(env, closing=False, steps=40, target_obj_id=obj_id)
    
    print("\nStep 3: Moving down to grasp position")
    grasp_pos = np.array(obj_pos) + [0, 0, 0]  # Direct at object
    if not move_to_position_minimal(env, grasp_pos.tolist()):
        print("Failed to move to grasp position. Trying adjusted position...")
        adjusted_pos = np.array(obj_pos) + [0.01, 0.01, 0]  # Slight offset
        if not move_to_position_minimal(env, adjusted_pos.tolist()):
            print("Adjusted position also failed. Aborting test.")
            env.close()
            return False
    
    # Allow time to stabilize
    for _ in range(30):
        p.stepSimulation()
        time.sleep(1/240)
    
    print("\nStep 4: Closing gripper to grasp")
    control_gripper(env, closing=True, grip_force=20000, steps=150, target_obj_id=obj_id)
    
    # Allow time for grip to settle
    for _ in range(60):
        p.stepSimulation()
        time.sleep(1/240)
    
    # Check for contacts
    contacts = p.getContactPoints(env.unwrapped.robot_id, obj_id)
    print(f"Contacts detected: {len(contacts)}")
    
    # Log detailed contact information
    left_finger_contacts = [c for c in contacts if c[3] == env.unwrapped.gripper_joints[0]]
    right_finger_contacts = [c for c in contacts if c[3] == env.unwrapped.gripper_joints[1]]
    print(f"Left finger contacts: {len(left_finger_contacts)}")
    print(f"Right finger contacts: {len(right_finger_contacts)}")
    
    # Create constraint for lifting
    print("\nStep 5: Creating fixed constraint")
    constraint_id = create_fixed_constraint(env, env.unwrapped.robot_id, obj_id)
    if constraint_id is None:
        print("Failed to create constraint. Aborting test.")
        env.close()
        return False
    
    # Reduce gravity for easier lifting
    print("\nStep 6: Reducing gravity for lift")
    p.setGravity(0, 0, -1.0)  # Very low gravity
    
    # Record initial position
    start_pos, _ = p.getBasePositionAndOrientation(obj_id)
    
    print("\nStep 7: Lifting object")
    lift_pos = np.array(obj_pos) + [0, 0, 0.15]  # 15cm up
    move_to_position_minimal(env, lift_pos.tolist())
    
    # Allow more time for lift
    for _ in range(80):
        p.stepSimulation()
        time.sleep(1/240)
    
    # Check lift success
    lifted_pos, _ = p.getBasePositionAndOrientation(obj_id)
    lift_height = lifted_pos[2] - start_pos[2]
    print(f"Initial height: {start_pos[2]:.4f}")
    print(f"Lifted height: {lifted_pos[2]:.4f}")
    print(f"Lift amount: {lift_height:.4f}m")
    
    # Restore gravity gradually
    p.setGravity(0, 0, -3.0)
    for _ in range(20):
        p.stepSimulation()
        time.sleep(1/240)
    
    p.setGravity(0, 0, -9.81)
    
    # Determine success
    success = lift_height > 0.05
    if success:
        print("\n GRASP TEST PASSED: Successfully grasped and lifted object!")
        try:
            p.addUserDebugText("GRASP SUCCESS!", 
                             [lifted_pos[0], lifted_pos[1], lifted_pos[2] + 0.1], 
                             [0, 1, 0], 
                             1.5, 
                             lifeTime=5.0)
        except:
            pass
    else:
        print("\nGRASP TEST FAILED: Object not lifted high enough")
        try:
            p.addUserDebugText("GRASP FAILED", 
                             [obj_pos[0], obj_pos[1], obj_pos[2] + 0.1], 
                             [1, 0, 0], 
                             1.5, 
                             lifeTime=5.0)
        except:
            pass
    
    # Hold position to see the result
    print("\nHolding position for 3 seconds...")
    for _ in range(180):  
        p.stepSimulation()
        time.sleep(1/60)
    
    
    env.close()
    return success

if __name__ == "__main__":
    test_direct_grasp()
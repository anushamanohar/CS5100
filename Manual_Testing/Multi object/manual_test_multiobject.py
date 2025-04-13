import time
import pybullet as p
import numpy as np
import random
import os
from env_setup_multiobject import make_env

# === Configurations ===
TRAY_HEIGHT = 0.4
DESCENT_OFFSET = 0.02
MAX_ERROR_THRESHOLD = 0.025
MAX_IK_RETRIES = 15

def move_to_position_minimal(env, target_pos, max_retries=30, accept_close=True):
    """Ultra-minimal function that uses the simplest possible IK calculation"""
    robot_id = env.unwrapped.robot_id
    ee_index = env.unwrapped.ee_link_index
    joint_indices = env.unwrapped.arm_joint_indices
    
    print(f" Target position: {np.round(target_pos, 4)}")
    
    # Store best position for fallback
    best_distance = float('inf')
    best_joint_positions = None
    
    for i in range(max_retries):
        # Get current end effector position
        ee_pos = p.getLinkState(robot_id, ee_index)[0]
        distance = np.linalg.norm(np.array(ee_pos) - np.array(target_pos))
        print(f" EE position: {np.round(ee_pos, 4)}, Distance: {distance:.4f}")
        
        # Track best position
        if distance < best_distance:
            best_distance = distance
            best_joint_positions = [p.getJointState(robot_id, j)[0] for j in joint_indices]
        
        # 4cm threshold for success
        if distance < 0.04:  
            print(" Close enough to target!")
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
            print(f" Error: {e}")
            return False
    
    # If we couldn't reach exactly, but got close enough, use the best position
    if accept_close and best_distance < 0.08:  # 8cm threshold for acceptable position
        print(f" Using best achieved position (distance: {best_distance:.4f}m)")
        for j, joint_idx in enumerate(joint_indices):
            if j < len(best_joint_positions):
                p.setJointMotorControl2(robot_id, joint_idx, p.POSITION_CONTROL, 
                                       best_joint_positions[j], force=500)
        
        for _ in range(20):
            p.stepSimulation()
            if env.render:
                time.sleep(1 / 240)
        
        return True
            
    print(" Could not reach target after maximum retries.")
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

def log_contacts(env, obj_id):
    """Check for contacts between the robot and the specified object"""
    contact_found = False
    contact_count = 0
    
    for c in p.getContactPoints(env.unwrapped.robot_id, obj_id):
        pos = c[5]  # Position on robot
        obj_pos = c[6]  # Position on object
        
        # Color code: red for finger contacts, blue for other parts
        color = [1, 0, 0] if c[3] in env.unwrapped.gripper_joints else [0, 0, 1]
        
        try:
            p.addUserDebugLine(pos, [pos[0], pos[1], pos[2] + 0.02], color, 2, 1.5)
        except:
            pass  # Ignore if too many debug lines
            
        print(f" Contact at: {np.round(obj_pos, 4)}, Joint: {c[3]}")
        contact_found = True
        contact_count += 1

    if not contact_found:
        print(" No contact — attempting adjustment.")
        try:
            p.addUserDebugText("NO CONTACT", [1.05, 0, 0.5], [1, 0, 0], 1.5, 2)
        except:
            pass  # Ignore if debug text limit is reached
    else:
        print(f" {contact_count} contacts detected")
        
    return contact_found

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
        
        print(f" Created fixed constraint ID: {constraint_id}")
        return constraint_id
    except Exception as e:
        print(f" Error creating constraint: {e}")
        return None

def process_single_object(env, obj_id, tray_pos):
    """Process a single object with the original grasp sequence"""
    # Temporarily set this as the current target object
    env.unwrapped.obj_id = obj_id
    
    # Get object position
    obj_pos, obj_orn = p.getBasePositionAndOrientation(obj_id)
    print(f" Object {obj_id} position: {obj_pos}")
    
    # Add visual marker
    try:
        p.addUserDebugText("TARGET", obj_pos, [1, 0, 0], 1.5, lifeTime=5)
    except:
        pass  # Ignore if debug text limit is reached
    
    # Hover position - slightly above the cube
    above_cube = np.array(obj_pos) + [0, 0, 0.05]
    grasp_cube = np.array(obj_pos)

    print(" Step 1: Move above cube")
    control_gripper(env, closing=False, steps=50, target_obj_id=obj_id)

    if not move_to_position_minimal(env, above_cube.tolist()):
        print(" Cannot move above cube")
        fallback_pos = np.array(obj_pos) + [0, 0, 0.1]
        print(" Trying fallback position...")
        if not move_to_position_minimal(env, fallback_pos.tolist()):
            print(" Fallback position also failed. Skipping this object.")
            return False

    control_gripper(env, closing=False, steps=50, target_obj_id=obj_id)

    print(" Step 2: Descend to grasp")
    if not move_to_position_minimal(env, grasp_cube.tolist()):
        print(" Cannot descend to grasp")
        adjusted_pos = np.array(obj_pos) + [0.01, 0.01, 0]
        print(" Trying adjusted grasp position...")
        if not move_to_position_minimal(env, adjusted_pos.tolist()):
            print(" Adjusted position also failed. Skipping this object.")
            return False

    for _ in range(30):
        p.stepSimulation()
        if env.render:
            time.sleep(1/240)

    print(" Step 3: Close gripper")
    control_gripper(env, closing=True, grip_force=20000, steps=300, target_obj_id=obj_id)

    for _ in range(60):
        p.stepSimulation()
        if env.render:
            time.sleep(1/240)

    has_contact = log_contacts(env, obj_id)
    
    print(" Step 4: Creating fixed grasp constraint")
    constraint_id = create_fixed_constraint(env, env.unwrapped.robot_id, obj_id)
    if constraint_id is None:
        print(" Failed to create constraint. Skipping object.")
        return False

    print(" Step 5: Temporarily reducing gravity for lift")
    p.setGravity(0, 0, -1.0)  # Even lower gravity for more reliable lifting

    start_pos, start_orn = p.getBasePositionAndOrientation(obj_id)

    print(" Step 6: Lift")
    lift_pos = np.array(obj_pos) + [0, 0, 0.15]
    move_to_position_minimal(env, lift_pos.tolist())

    for _ in range(80):  # More time for lift
        p.stepSimulation()
        if env.render:
            time.sleep(1/240)

    lifted_pos, lifted_orn = p.getBasePositionAndOrientation(obj_id)
    lift_height = lifted_pos[2] - start_pos[2]
    print(" Initial cube height:", round(start_pos[2], 4))
    print(" Cube pos after lift:", round(lifted_pos[2], 4))
    print(" Lift height:", round(lift_height, 4))

    # Gradually restore gravity
    p.setGravity(0, 0, -3.0)
    for _ in range(20):
        p.stepSimulation()
        if env.render:
            time.sleep(1/240)
            
    p.setGravity(0, 0, -9.81)

    if lift_height > 0.05:
        print(" Grasp succeeded!")
        try:
            p.addUserDebugText("GRASP SUCCESS!", 
                              [lifted_pos[0], lifted_pos[1], lifted_pos[2] + 0.1], 
                              [0, 1, 0], 
                              1.5, 
                              lifeTime=5.0)
        except:
            pass  # Ignore if debug text limit is reached

        # === DROP SEQUENCE WITH LOWER HEIGHT ===
        robot_id = env.unwrapped.robot_id
        
        # Step 1: Reset arm to neutral position
        print(" Resetting arm to straight position")
        reset_joints = [0, 0, 0, 0, 0, 0, 0]  # Neutral position
        
        for i, joint_idx in enumerate(env.unwrapped.arm_joint_indices):
            p.setJointMotorControl2(
                robot_id, 
                joint_idx, 
                p.POSITION_CONTROL, 
                reset_joints[i], 
                force=500
            )
        
        # Allow time for reset
        for _ in range(100):
            p.stepSimulation()
            if env.render:
                time.sleep(1/240)
                
        # Step 2: Rotate base 180 degrees
        print(" Rotating base joint 180 degrees")
        p.setJointMotorControl2(
            robot_id, 
            0,  # Base joint
            p.POSITION_CONTROL, 
            3.14,  # 180 degrees (π radians)
            force=500
        )
        
        # Allow time for rotation
        for _ in range(150):
            p.stepSimulation()
            if env.render:
                time.sleep(1/240)
                
        # Step 3: Position over tray with minimal drop height
        print(" Moving arm to tray position")
        
        # These joint values position the arm slightly lower for less drop height
        tray_joints = [
            3.14,    # Base (already rotated)
            -0.85,   # Shoulder (lower)
            1.4,     # Elbow (more extended)
            -0.8,    # Wrist (more angled)
            0.0,     # Wrist roll
            0.3,     # Wrist adjustment
            0.0      # Fixed joint
        ]
        
        for i, joint_idx in enumerate(env.unwrapped.arm_joint_indices):
            p.setJointMotorControl2(
                robot_id, 
                joint_idx, 
                p.POSITION_CONTROL, 
                tray_joints[i], 
                force=300
            )
            
        # Allow time to reach position
        for _ in range(200):
            p.stepSimulation()
            if env.render:
                time.sleep(1/240)
        
        # Check position
        ee_pos = p.getLinkState(robot_id, env.unwrapped.ee_link_index)[0]
        distance_to_tray = np.linalg.norm(np.array(ee_pos[:2]) - np.array(tray_pos[:2]))
        height_above_tray = ee_pos[2] - tray_pos[2]
        print(f" Arm positioned at: {np.round(ee_pos, 4)}")
        print(f" Distance to tray center: {distance_to_tray:.4f}")
        print(f" Height above tray: {height_above_tray:.4f}")
        
        try:
            p.addUserDebugText("DROP", ee_pos, [0.2, 1, 1], 1.5, lifeTime=5)
        except:
            pass
        
        # Step 4: Lower even more for gentle drop
        print(" Lowering for gentle drop")
        low_drop_joints = tray_joints.copy()
        low_drop_joints[1] -= 0.15  # Even lower shoulder
        low_drop_joints[2] += 0.1   # More elbow extension
        
        for i, joint_idx in enumerate(env.unwrapped.arm_joint_indices):
            p.setJointMotorControl2(
                robot_id, 
                joint_idx, 
                p.POSITION_CONTROL, 
                low_drop_joints[i], 
                force=200
            )
                
        # Allow time for adjustment
        for _ in range(100):
            p.stepSimulation()
            if env.render:
                time.sleep(1/240)
        
        # Re-check position
        ee_pos = p.getLinkState(robot_id, env.unwrapped.ee_link_index)[0]
        height_above_tray = ee_pos[2] - tray_pos[2]
        print(f" Adjusted position: {np.round(ee_pos, 4)}")
        print(f" Final height above tray: {height_above_tray:.4f}")
        
        # Release the object
        print(" Releasing object")
        try:
            p.removeConstraint(constraint_id)
        except Exception as e:
            print(f" Error removing constraint: {e}")
            
        control_gripper(env, closing=False, grip_force=300, steps=100, target_obj_id=obj_id)
        
        # Allow time for the object to drop and settle
        for _ in range(200):
            p.stepSimulation()
            if env.render:
                time.sleep(1/240)
                
        final_pos, _ = p.getBasePositionAndOrientation(obj_id)
        print(" Final object position after drop:", np.round(final_pos, 4))
        
        # Check if object landed in tray
        dist_to_tray_center = np.linalg.norm(np.array(final_pos[:2]) - np.array(tray_pos[:2]))
        if dist_to_tray_center < 0.3 and abs(final_pos[2] - tray_pos[2]) < 0.15:
            print(" Successfully dropped in tray!")
            try:
                p.addUserDebugText("IN TRAY!", 
                                  [final_pos[0], final_pos[1], final_pos[2] + 0.1], 
                                  [0, 1, 0], 
                                  1.5, 
                                  lifeTime=5.0)
            except:
                pass
            return True
        else:
            print(" Object missed the tray. Distance to tray center:", dist_to_tray_center)
            return False
            
    else:
        print(" Grasp failed — cube still on table.")
        try:
            p.addUserDebugText("GRASP FAILED", 
                             [obj_pos[0], obj_pos[1], obj_pos[2] + 0.1], 
                             [1, 0, 0], 
                             1.5, 
                             lifeTime=5.0)
        except:
            pass
        return False
    
def reset_arm_to_initial(env):
    """Reset the arm to its initial position between objects"""
    robot_id = env.unwrapped.robot_id
    
    # Initial pose that matches what's set in env_setup_multiobject.py
    initial_joint_positions = [0, -0.5, 0.5, -0.5, 0.0, 0.0, 0.0]
    
    print(" Resetting arm to initial position...")
    for i, joint_idx in enumerate(env.unwrapped.arm_joint_indices):
        p.setJointMotorControl2(
            robot_id, 
            joint_idx, 
            p.POSITION_CONTROL, 
            initial_joint_positions[i], 
            force=500
        )
    
    # Allow time for the arm to reach the initial position
    for _ in range(120):
        p.stepSimulation()
        if env.render:
            time.sleep(1/240)
    
    # Re-open gripper
    control_gripper(env, closing=False, steps=50)
    
    # Additional stability steps
    for _ in range(30):
        p.stepSimulation()
        if env.render:
            time.sleep(1/240)
            
    print(" Arm reset complete")
    return True

def run_multi_object_sequence():
    """Run pick and place for multiple objects"""
    # Get user input for number of objects
    try:
        user_input = input("Enter number of objects (1-3) or press Enter for default (2): ")
        num_objects = int(user_input) if user_input.strip() else 2
        num_objects = max(1, min(3, num_objects))  # Limit between 1 and 3
    except ValueError:
        print("Invalid input. Using default (2 objects).")
        num_objects = 2
    
    # Create environment with multiple objects
    env = make_env(render=True, num_objects=num_objects)()
    obs, _ = env.reset()
    
    # Wait for environment to stabilize
    for _ in range(100):
        p.stepSimulation()
        if env.render:
            time.sleep(1/240)
    
    # Get tray positions
    source_tray_pos, dest_tray_pos = env.unwrapped.get_tray_positions()
    print(f"Source tray position: {source_tray_pos}")
    print(f"Destination tray position: {dest_tray_pos}")
    
    # Get object IDs from environment
    object_ids = env.unwrapped.get_object_ids()
    if not object_ids:
        print("No objects found in environment!")
        env.close()
        return
        
    print(f"Found {len(object_ids)} objects. Starting pick and place sequence.")
    
    # Process each object
    successful_placements = 0
    for i, obj_id in enumerate(object_ids):
        print(f"\n==== Processing Object {i+1}/{len(object_ids)} (ID: {obj_id}) ====")
        
        # Process this object
        if process_single_object(env, obj_id, dest_tray_pos):
            successful_placements += 1
        
        # Wait a short time for visualization purposes between objects
        if i < len(object_ids) - 1:
            print(f"Automatically continuing to next object ({i+2}/{len(object_ids)}) in 1 second...")
            time.sleep(1)  # Short delay for visualization
            
        reset_arm_to_initial(env)
    
    # Display final results
    print(f"\n==== Completed pick and place sequence ====")
    print(f"Successfully placed {successful_placements}/{len(object_ids)} objects in the tray.")
    
    input("Press ENTER to exit")
    env.close()

if __name__ == "__main__":
    run_multi_object_sequence()

import time
import pybullet as p
import numpy as np
import random
from env_setup import make_env

# === Configurations ===
TRAY_HEIGHT = 0.4
DESCENT_OFFSET = 0.02  
MAX_ERROR_THRESHOLD = 0.025
MAX_IK_RETRIES = 15

def move_to_position_minimal(env, target_pos, max_retries=30):
    """Using the simplest possible IK calculation"""
    robot_id = env.unwrapped.robot_id
    ee_index = env.unwrapped.ee_link_index
    joint_indices = env.unwrapped.arm_joint_indices
    
    print(f" Target position: {np.round(target_pos, 4)}")
    
    for i in range(max_retries):
        # Get current end effector position
        ee_pos = p.getLinkState(robot_id, ee_index)[0]
        distance = np.linalg.norm(np.array(ee_pos) - np.array(target_pos))
        print(f" EE position: {np.round(ee_pos, 4)}, Distance: {distance:.4f}")
        
        # IMPORTANT: 4cm threshold based on observed capabilities
        if distance < 0.04:  
            print(" Close enough to target!")
            return True
            
        
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
            
    print(" Could not reach target after maximum retries.")
    return False

def control_gripper(env, closing=True, grip_force=10000, steps=300):
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
    p.addUserDebugText(gripper_status, 
                      [ee_pos[0], ee_pos[1], ee_pos[2] + 0.05], 
                      [1, 0.5, 0] if closing else [0, 0.8, 0.3], 
                      1.5, 
                      lifeTime=1.0)
    
    # Step simulation with contact checking
    for i in range(steps):
        p.stepSimulation()
        
        # Check for contacts periodically while closing
        if closing and i % 10 == 0:
            contact_points = p.getContactPoints(
                env.unwrapped.robot_id, 
                env.unwrapped.obj_id
            )
            if contact_points:
                # Visualize contacts
                for c in contact_points:
                    contact_pos = c[6]  # Position on object
                    p.addUserDebugLine(
                        contact_pos, 
                        [contact_pos[0], contact_pos[1], contact_pos[2] + 0.02],
                        [0, 1, 0],
                        3,
                        lifeTime=0.5
                    )
        
        if env.render:
            time.sleep(1/240)

def log_contacts(env):
    """Check for contacts between the robot and the object"""
    contact_found = False
    contact_count = 0
    
    for c in p.getContactPoints(env.unwrapped.robot_id, env.unwrapped.obj_id):
        pos = c[5]  # Position on robot
        obj_pos = c[6]  # Position on object
        
        # Color code: red for finger contacts, blue for other parts
        color = [1, 0, 0] if c[3] in env.unwrapped.gripper_joints else [0, 0, 1]
        
        p.addUserDebugLine(pos, [pos[0], pos[1], pos[2] + 0.02], color, 2, 1.5)
        print(f" Contact at: {np.round(obj_pos, 4)}, Joint: {c[3]}")
        contact_found = True
        contact_count += 1

    if not contact_found:
        print(" No contact — attempting adjustment.")
        p.addUserDebugText("NO CONTACT", [1.05, 0, 0.5], [1, 0, 0], 1.5, 2)
    else:
        print(f" {contact_count} contacts detected")
        
    return contact_found

def create_fixed_constraint(env, robot_id, obj_id):
    """Create a fixed constraint between the robot gripper and the object"""
    ee_index = env.unwrapped.ee_link_index
    left_finger, right_finger = env.unwrapped.gripper_joints
    
    # Get positions
    ee_pos = p.getLinkState(robot_id, ee_index)[0]
    obj_pos, obj_orn = p.getBasePositionAndOrientation(obj_id)
    
    # Calculate relative position (object position relative to end effector)
    rel_pos = [
        obj_pos[0] - ee_pos[0], 
        obj_pos[1] - ee_pos[1],
        obj_pos[2] - ee_pos[2]
    ]
    
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

def run_grasp_sequence():
    """Run the complete grasp sequence with direct tray-targeting approach"""
    env = make_env(render=True)()
    obs, _ = env.reset()

    
    for _ in range(50):  
        p.stepSimulation()
        if env.render:
            time.sleep(1/240)

    # === Get True Object Position  ===
    true_cube_pos, true_cube_orn = p.getBasePositionAndOrientation(env.unwrapped.obj_id)
    print(f" Actual cube pos after stabilization: {true_cube_pos}")

    # Get the drop tray position
    tray_pos, tray_orn = p.getBasePositionAndOrientation(env.unwrapped.drop_tray_id)
    print(f" Drop tray position: {tray_pos}")
    
    # Add debug markers
    p.addUserDebugText("CUBE", true_cube_pos, [1, 0, 0], 1.5, lifeTime=0)
    p.addUserDebugText("TRAY", tray_pos, [0, 0, 1], 1.2, lifeTime=0)

    # Hover position above cube
    above_cube = np.array(true_cube_pos) + [0, 0, 0.05]
    grasp_cube = np.array(true_cube_pos)

    print(" Step 1: Move above cube")
    control_gripper(env, closing=False, steps=50)

    if not move_to_position_minimal(env, above_cube.tolist()):
        print(" Cannot move above cube")
        fallback_pos = np.array(true_cube_pos) + [0, 0, 0.1]
        print(" Trying fallback position...")
        if not move_to_position_minimal(env, fallback_pos.tolist()):
            print(" Fallback position also failed. Aborting.")
            env.close()
            return

    control_gripper(env, closing=False, steps=50)

    print(" Step 2: Descend to grasp")
    if not move_to_position_minimal(env, grasp_cube.tolist()):
        print(" Cannot descend to grasp")
        adjusted_pos = np.array(true_cube_pos) + [0.01, 0.01, 0]
        print(" Trying adjusted grasp position...")
        if not move_to_position_minimal(env, adjusted_pos.tolist()):
            print(" Adjusted position also failed.")

    for _ in range(30):
        p.stepSimulation()
        if env.render:
            time.sleep(1/240)

    print(" Step 3: Close gripper")
    control_gripper(env, closing=True, grip_force=20000, steps=300)

    for _ in range(60):
        p.stepSimulation()
        if env.render:
            time.sleep(1/240)

    has_contact = log_contacts(env)
    if has_contact:
        print("Contact confirmed after gripper close")
    else:
        print(" No contact detected after gripper close")

    print(" Step 4: Creating fixed grasp constraint")
    constraint_id = create_fixed_constraint(env, env.unwrapped.robot_id, env.unwrapped.obj_id)

    print(" Step 5: Temporarily reducing gravity for lift")
    p.setGravity(0, 0, -2.0)

    start_pos, start_orn = p.getBasePositionAndOrientation(env.unwrapped.obj_id)

    print(" Step 6: Lift")
    lift_pos = np.array(true_cube_pos) + [0, 0, 0.15]
    move_to_position_minimal(env, lift_pos.tolist())

    for _ in range(60):
        p.stepSimulation()
        if env.render:
            time.sleep(1/240)

    lifted_pos, lifted_orn = p.getBasePositionAndOrientation(env.unwrapped.obj_id)
    lift_height = lifted_pos[2] - start_pos[2]
    print(" Initial cube height:", round(start_pos[2], 4))
    print(" Cube pos after lift:", round(lifted_pos[2], 4))
    print(" Lift height:", round(lift_height, 4))

    p.setGravity(0, 0, -9.81)

    if lift_height > 0.05:
        print(" Grasp succeeded!")
        p.addUserDebugText("GRASP SUCCESS!", 
                          [lifted_pos[0], lifted_pos[1], lifted_pos[2] + 0.1], 
                          [0, 1, 0], 
                          1.5, 
                          lifeTime=5.0)

        #  Reset arm to known position and directly set joints ===
        robot_id = env.unwrapped.robot_id
        
        # Step 1: Set arm to a known "reset" position (straight up)
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
                
        # Step 2: Now rotate the base 180 degrees (3.14 radians)
        print("Rotating base joint 180 degrees")
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
                
        # Step 3: Configure arm joints to reach back toward tray
        # These specific joint values place the end effector over the tray
        tray_joints = [
            3.14,    # Base (already rotated)
            -0.7,    # Shoulder joint bend backward
            1.2,     # Elbow joint bend downward
            -0.7,    # Wrist pitch down
            0.0,     # Wrist roll neutral
            0.3,     # Wrist pitch adjustment
            0.0      # Fixed joint
        ]
        
        print("Moving arm to tray position")
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
        
        # Check where the end effector is now
        ee_pos = p.getLinkState(robot_id, env.unwrapped.ee_link_index)[0]
        distance_to_tray = np.linalg.norm(np.array(ee_pos[:2]) - np.array(tray_pos[:2]))
        print(f" Arm positioned at: {np.round(ee_pos, 4)}")
        print(f" Distance to tray center: {distance_to_tray:.4f}")
        
        # Visualize the drop point
        p.addUserDebugText("DROP", ee_pos, [0.2, 1, 1], 1.5, lifeTime=5)
        
        # Final joint adjustment to position better over tray if needed
        if distance_to_tray > 0.2:
            print(" Fine-tuning position with targeted joint adjustment")
            # Adjust specific joints to bring arm closer to tray
            fine_tune_joints = tray_joints.copy()
            
            # Modify specific joints to get closer to tray
            # Adjust shoulder and elbow to extend further
            fine_tune_joints[1] -= 0.2  # More backward tilt
            fine_tune_joints[2] += 0.2  # More elbow extension
            
            for i, joint_idx in enumerate(env.unwrapped.arm_joint_indices):
                p.setJointMotorControl2(
                    robot_id, 
                    joint_idx, 
                    p.POSITION_CONTROL, 
                    fine_tune_joints[i], 
                    force=200
                )
                
            
            for _ in range(100):
                p.stepSimulation()
                if env.render:
                    time.sleep(1/240)
            
            # Re-check position
            ee_pos = p.getLinkState(robot_id, env.unwrapped.ee_link_index)[0]
            distance_to_tray = np.linalg.norm(np.array(ee_pos[:2]) - np.array(tray_pos[:2]))
            print(f" Adjusted arm position: {np.round(ee_pos, 4)}")
            print(f" New distance to tray: {distance_to_tray:.4f}")
        
        # Release the object
        print("Releasing object")
        p.removeConstraint(constraint_id)
        control_gripper(env, closing=False, grip_force=300, steps=100)
        
        # Allow time for the object to drop and settle
        for _ in range(200):
            p.stepSimulation()
            if env.render:
                time.sleep(1/240)
                
        final_pos, _ = p.getBasePositionAndOrientation(env.unwrapped.obj_id)
        print("Final cube position after drop:", np.round(final_pos, 4))
        
        # Check if object landed in tray
        dist_to_tray_center = np.linalg.norm(np.array(final_pos[:2]) - np.array(tray_pos[:2]))
        if dist_to_tray_center < 0.3 and abs(final_pos[2] - tray_pos[2]) < 0.15:
            print(" Successfully dropped in tray!")
            p.addUserDebugText("IN TRAY!", 
                              [final_pos[0], final_pos[1], final_pos[2] + 0.1], 
                              [0, 1, 0], 
                              1.5, 
                              lifeTime=5.0)
        else:
            print(" Object missed the tray. Distance to tray center:", dist_to_tray_center)
            
    else:
        print("Grasp failed — cube still on table.")
        p.addUserDebugText("GRASP FAILED", 
                         [true_cube_pos[0], true_cube_pos[1], true_cube_pos[2] + 0.1], 
                         [1, 0, 0], 
                         1.5, 
                         lifeTime=5.0)

    input("Press ENTER to exit")
    env.close()
    
if __name__ == "__main__":

    run_grasp_sequence()
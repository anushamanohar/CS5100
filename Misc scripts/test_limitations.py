import time
import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from env_setup import make_env
import random

def analyze_robot_workspace(env, samples=500, visualize=True, save_path="workspace_analysis.png"):
    """
    Analyze the robot's reachable workspace by sampling random joint configurations
    and plotting the resulting end-effector positions.
    
    Args:
        env: The simulation environment
        samples: Number of random joint configurations to sample
        visualize: Whether to render the simulation during analysis
        save_path: Path to save the workspace visualization
    """
    robot_id = env.unwrapped.robot_id
    joint_indices = env.unwrapped.arm_joint_indices
    ee_index = env.unwrapped.ee_link_index
    
    # Get joint limits
    joint_limits = []
    for joint_idx in joint_indices:
        info = p.getJointInfo(robot_id, joint_idx)
        lower = info[8]  # Lower limit
        upper = info[9]  # Upper limit
        joint_limits.append((lower, upper))
    
    # Get current joint positions (to restore later)
    original_positions = [p.getJointState(robot_id, j)[0] for j in joint_indices]
    
    # Storage for positions
    ee_positions = []
    successful_joint_configs = []
    cube_position, _ = p.getBasePositionAndOrientation(env.unwrapped.obj_id)
    
    print(f"Analyzing robot workspace with {samples} random configurations...")
    print(f"Target cube position: {np.round(cube_position, 4)}")
    
    # Create visualization points in the simulation
    point_ids = []
    
    for i in range(samples):
        # Generate random joint configuration
        joint_config = []
        for j, (lower, upper) in enumerate(joint_limits):
            # Sample within limits with some margin
            margin = (upper - lower) * 0.1  # 10% margin
            rand_pos = random.uniform(lower + margin, upper - margin)
            joint_config.append(rand_pos)
        
        # Apply joint configuration
        for j, joint_idx in enumerate(joint_indices):
            p.resetJointState(robot_id, joint_idx, joint_config[j])
        
        # Step the simulation to update positions
        p.stepSimulation()
        if visualize:
            time.sleep(1/480)  # Faster than normal, just to see progress
        
        # Get end effector position
        ee_pos = p.getLinkState(robot_id, ee_index)[0]
        ee_positions.append(ee_pos)
        successful_joint_configs.append(joint_config)
        
        # Add a visual marker at this position in the simulation
        if i % 10 == 0:  # Only visualize every 10th point to avoid clutter
            point_id = p.addUserDebugPoints([ee_pos], [[0, 0, 1]], pointSize=3)
            point_ids.append(point_id)
    
    # Reset robot to original position
    for j, joint_idx in enumerate(joint_indices):
        p.resetJointState(robot_id, joint_idx, original_positions[j])
    
    # Analyze results
    ee_positions = np.array(ee_positions)
    
    # Find closest point to cube
    distances = np.linalg.norm(ee_positions - np.array(cube_position), axis=1)
    closest_idx = np.argmin(distances)
    closest_distance = distances[closest_idx]
    closest_joint_config = successful_joint_configs[closest_idx]
    closest_ee_pos = ee_positions[closest_idx]
    
    print(f"\nWorkspace Analysis Results:")
    print(f"  Number of sampled positions: {len(ee_positions)}")
    print(f"  Min X: {ee_positions[:, 0].min():.4f}, Max X: {ee_positions[:, 0].max():.4f}")
    print(f"  Min Y: {ee_positions[:, 1].min():.4f}, Max Y: {ee_positions[:, 1].max():.4f}")
    print(f"  Min Z: {ee_positions[:, 2].min():.4f}, Max Z: {ee_positions[:, 2].max():.4f}")
    print(f"\nDistance to Target Analysis:")
    print(f"  Closest point to cube: {np.round(closest_ee_pos, 4)}")
    print(f"  Closest distance achieved: {closest_distance:.4f}")
    print(f"  Joint configuration for closest point:")
    for i, val in enumerate(closest_joint_config):
        print(f"    Joint {i}: {val:.4f}")
    
    # Demonstrate the closest position
    print("\nDemonstrating closest achievable position to cube...")
    
    # Visual marker for cube
    p.addUserDebugPoints([cube_position], [[1, 0, 0]], pointSize=10)
    
    # Set joints to the closest configuration
    for j, joint_idx in enumerate(joint_indices):
        p.resetJointState(robot_id, joint_idx, closest_joint_config[j])
    
    # Step simulation
    for _ in range(30):
        p.stepSimulation()
        if visualize:
            time.sleep(1/240)
    
    # Confirm actual position
    actual_pos = p.getLinkState(robot_id, ee_index)[0]
    actual_distance = np.linalg.norm(np.array(actual_pos) - np.array(cube_position))
    
    print(f"  Actual closest position achieved: {np.round(actual_pos, 4)}")
    print(f"  Actual distance to cube: {actual_distance:.4f}")
    
    # Create a visual line between closest position and cube
    p.addUserDebugLine(actual_pos, cube_position, [1, 0, 0], 3, lifeTime=15.0)
    p.addUserDebugText(f"Distance: {actual_distance:.3f}m", 
                       [(actual_pos[0]+cube_position[0])/2, 
                        (actual_pos[1]+cube_position[1])/2, 
                        (actual_pos[2]+cube_position[2])/2], 
                       [1, 1, 1], 1.5, lifeTime=15.0)
    
    # Plot 3D scatter of workspace with cube position
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot workspace points
    ax.scatter(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], 
               c='blue', alpha=0.3, s=10, label='Workspace Points')
    
    # Plot cube position
    ax.scatter(cube_position[0], cube_position[1], cube_position[2], 
               c='red', s=100, marker='*', label='Cube Position')
    
    # Plot closest achievable point
    ax.scatter(closest_ee_pos[0], closest_ee_pos[1], closest_ee_pos[2], 
               c='green', s=100, marker='o', label='Closest Achievable Point')
    
    # Connect closest point to cube with a line
    ax.plot([closest_ee_pos[0], cube_position[0]], 
            [closest_ee_pos[1], cube_position[1]], 
            [closest_ee_pos[2], cube_position[2]], 
            'r--', linewidth=2, label=f'Distance: {closest_distance:.3f}m')
    
    # Draw the robot base position
    base_pos = p.getBasePositionAndOrientation(robot_id)[0]
    ax.scatter(base_pos[0], base_pos[1], base_pos[2], 
               c='black', s=150, marker='^', label='Robot Base')
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Robot Workspace Analysis')
    
    # Add a legend
    ax.legend()
    
    # Set consistent axis limits for better perspective
    max_range = 1.5
    mid_x = (ee_positions[:, 0].max() + ee_positions[:, 0].min()) / 2
    mid_y = (ee_positions[:, 1].max() + ee_positions[:, 1].min()) / 2
    mid_z = (ee_positions[:, 2].max() + ee_positions[:, 2].min()) / 2
    
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Workspace visualization saved to: {save_path}")
    
    # Create a direct approach with the best configuration
    print("\nAttempting direct approach with best configuration...")
    
    # Reset joint positions
    for j, joint_idx in enumerate(joint_indices):
        p.resetJointState(robot_id, joint_idx, closest_joint_config[j])
    
    # Open gripper
    left, right = env.unwrapped.gripper_joints
    p.setJointMotorControl2(robot_id, left, p.POSITION_CONTROL, 0.04, force=300)
    p.setJointMotorControl2(robot_id, right, p.POSITION_CONTROL, 0.04, force=300)
    
    for _ in range(50):
        p.stepSimulation()
        if visualize:
            time.sleep(1/240)
    
    # Check how close we get to the cube
    ee_pos = p.getLinkState(robot_id, ee_index)[0]
    distance = np.linalg.norm(np.array(ee_pos) - np.array(cube_position))
    print(f"  Final approach position: {np.round(ee_pos, 4)}")
    print(f"  Final distance to cube: {distance:.4f}")
    
    # Check for contact
    contacts = p.getContactPoints(robot_id, env.unwrapped.obj_id)
    if contacts:
        print(f"   Contact established with {len(contacts)} points!")
    else:
        print("   No contact detected with best configuration.")
    
    # Close gripper to test grasp
    p.setJointMotorControl2(robot_id, left, p.POSITION_CONTROL, 0.0, force=1000)
    p.setJointMotorControl2(robot_id, right, p.POSITION_CONTROL, 0.0, force=1000)
    
    for _ in range(60):
        p.stepSimulation()
        if visualize:
            time.sleep(1/240)
    
    # Try lifting
    for j, joint_idx in enumerate(joint_indices):
        if j == 1:  # Typically the shoulder joint
            target = closest_joint_config[j] - 0.3  # Lift up
        else:
            target = closest_joint_config[j]
        p.setJointMotorControl2(robot_id, joint_idx, p.POSITION_CONTROL, target, force=500)
    
    for _ in range(80):
        p.stepSimulation()
        if visualize:
            time.sleep(1/240)
    
    # Check lift result
    final_cube_pos = p.getBasePositionAndOrientation(env.unwrapped.obj_id)[0]
    lift_height = final_cube_pos[2] - cube_position[2]
    
    print("\n== Lift Test Results ==")
    print(f"  Initial cube height: {cube_position[2]:.4f}")
    print(f"  Final cube height: {final_cube_pos[2]:.4f}")
    print(f"  Lift distance: {lift_height:.4f}")
    
    if lift_height > 0.1:
        print("  Grasp and lift successful!")
    elif lift_height > 0.05:
        print("   Partial grasp - cube lifted but not securely")
    else:
        print("   Grasp failed - cube remains on table")
    
    print("\nWorkspace analysis complete!")
    return ee_positions, closest_distance, lift_height > 0.05

def test_alternative_cube_positions(env, num_positions=3, samples_per_position=250):
    """
    Test different cube positions to find one that can be reached by the robot.
    
    Args:
        env: The simulation environment
        num_positions: Number of alternative positions to test
        samples_per_position: Number of samples for workspace analysis per position
    """
    robot_id = env.unwrapped.robot_id
    obj_id = env.unwrapped.obj_id
    
    # Get original cube position
    original_pos, original_orn = p.getBasePositionAndOrientation(obj_id)
    print(f"Original cube position: {np.round(original_pos, 4)}")
    
    # Define test positions (more likely to be reachable)
    test_positions = [
        [0.95, 0.0, 0.4],    # Directly in front, closer
        [0.9, 0.1, 0.4],     # Closer, slightly to the side
        [0.85, -0.1, 0.4],   # Even closer, other side
        [1.0, 0.2, 0.4],     # Original distance, more to the side
        [0.8, 0.0, 0.5],     # Closer, higher up
    ]
    
    # Ensure we have enough test positions
    while len(test_positions) < num_positions:
        # Generate random positions within reasonable range
        x = random.uniform(0.8, 1.0)
        y = random.uniform(-0.2, 0.2)
        z = 0.4  # Keep height constant for table
        test_positions.append([x, y, z])
    
    # Limit to requested number
    test_positions = test_positions[:num_positions]
    
    results = []
    
    # Test each position
    for i, position in enumerate(test_positions):
        
        print(f"Testing cube position {i+1}/{len(test_positions)}: {np.round(position, 4)}")
        
        # Reset cube position
        p.resetBasePositionAndOrientation(obj_id, position, original_orn)
        
        # Step simulation to stabilize
        for _ in range(20):
            p.stepSimulation()
        
        # Analyze workspace for this position
        _, distance, grasp_success = analyze_robot_workspace(
            env, 
            samples=samples_per_position, 
            save_path=f"workspace_position_{i+1}.png"
        )
        
        results.append({
            'position': position,
            'distance': distance,
            'grasp_success': grasp_success
        })
        
        # If the grasp was successful, we've found a good position
        if grasp_success:
            print(f"Found a reachable position! {np.round(position, 4)}")
            break
    
    # Restore original position
    p.resetBasePositionAndOrientation(obj_id, original_pos, original_orn)
    
    # Report results
    print("\n=== Alternative Position Test Results ===")
    for i, result in enumerate(results):
        status = "SUCCESS" if result['grasp_success'] else "FAILED"
        print(f"Position {i+1}: {np.round(result['position'], 4)}")
        print(f"  Closest distance: {result['distance']:.4f}m")
        print(f"  Grasp test: {status}")
    
    # Find the best position (prioritize success, then closest distance)
    successful = [r for r in results if r['grasp_success']]
    if successful:
        best = min(successful, key=lambda x: x['distance'])
        status = "successful"
    else:
        best = min(results, key=lambda x: x['distance'])
        status = "unsuccessful but closest"
    
    print(f"\nRecommended position (most {status}):")
    print(f"  {np.round(best['position'], 4)}")
    print(f"  Distance: {best['distance']:.4f}m")
    
    return results

def run_workspace_analysis():
    """Run the complete workspace analysis"""
    env = make_env(render=True)()
    obs, _ = env.reset()
    
    print("\n ROBOT WORKSPACE ANALYSIS ")
    print("Analyzing the robot's reachable workspace to verify kinematic limits...")
    
    # First analyze the workspace with the original cube position
    ee_positions, closest_distance, grasp_success = analyze_robot_workspace(env)
    
    if not grasp_success and closest_distance > 0.1:
        print("\nThe cube appears to be outside the robot's reachable workspace.")
        print("Testing alternative positions to find a reachable location...")
        
        # Test alternative positions
        results = test_alternative_cube_positions(env)
        
        # Find the best position
        successful = [r for r in results if r['grasp_success']]
        if successful:
            best = min(successful, key=lambda x: x['distance'])
            best_pos = best['position']
            
            # Demonstrate the best position
            print("\nDemonstrating grasp with the best position...")
            obj_id = env.unwrapped.obj_id
            _, original_orn = p.getBasePositionAndOrientation(obj_id)
            p.resetBasePositionAndOrientation(obj_id, best_pos, original_orn)
            
            # Step simulation to stabilize
            for _ in range(20):
                p.stepSimulation()
                time.sleep(1/240)
                
            # Analyze this position again with more detailed output
            analyze_robot_workspace(env, samples=200, save_path="best_position_workspace.png")
    
    print("\nWorkspace analysis complete. Refer to the generated plots for visualization.")
    
    # Keep simulation running until user exits
    input("Press ENTER to exit")
    env.close()

if __name__ == "__main__":
    run_workspace_analysis()
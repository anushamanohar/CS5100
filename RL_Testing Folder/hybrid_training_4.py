# improved_training.py
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
from gymnasium import spaces
import argparse
import pybullet as p
import datetime
import json

from gymnasium.wrappers import TimeLimit

# Import the modified environment and hybrid wrapper
from env_setup_multiobject import VisualRoboticArmEnv, make_env
from env_setup_multiobject import VisualRoboticArmEnv, make_env, make_hybrid_env

# Import policy and utils from existing code
from cnn_policy import CustomSACPolicy
from RL_train import DimensionAdapter, TrainingMetricsCallback, plot_training_results

# Create directories for logs and models
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ================== VERBOSE CALLBACK ==================
class VerboseCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(VerboseCallback, self).__init__(verbose)
        self.episode_count = 0
        self.episode_rewards = []
        self.current_reward = 0
        self.takeover_attempts = 0
        self.takeover_success = 0
        
    def _on_step(self) -> bool:
        # Get current reward
        if 'reward' in self.locals:
            self.current_reward += self.locals['reward'][0]
            
        # Check if episode is done
        if self.locals['dones'][0]:
            self.episode_count += 1
            self.episode_rewards.append(self.current_reward)
            
            # Get the last info for this episode
            info = self.locals['infos'][0]
            max_stage = info.get('stage', 0)
            max_lift = info.get('max_lift_height', 0)
            is_success = max_stage == 4
            
            # Track takeovers
            env = self.locals['env'].envs[0]
            if hasattr(env, 'transition_triggered') and hasattr(env, 'takeover_attempts'):
                takeovers = env.takeover_attempts
                takeover_success = 1 if is_success and takeovers > 0 else 0
                self.takeover_attempts += takeovers
                self.takeover_success += takeover_success
                takeover_info = f", Takeovers: {takeovers}, Success: {takeover_success}"
            else:
                takeover_info = ""
            
            # Print episode summary
            print(f"\nEpisode {self.episode_count} complete: Reward={self.current_reward:.2f}")
            print(f"Max Stage: {max_stage}, Success: {is_success}, Max Lift: {max_lift:.4f}m{takeover_info}")
            print(f"Average reward over {min(10, self.episode_count)} episodes: "
                  f"{np.mean(self.episode_rewards[-10:]):.2f}")
            
            # If hybrid is being used, report takeover stats
            if self.takeover_attempts > 0:
                success_rate = (self.takeover_success / self.takeover_attempts) * 100
                print(f"Hybrid takeover success rate: {success_rate:.1f}% ({self.takeover_success}/{self.takeover_attempts})")
                  
            # Reset current reward
            self.current_reward = 0
            
        return True

# ================== TRAINING FUNCTION ==================
def train_with_optimized_params(total_timesteps=150000, log_dir="logs/optimized_training", 
                               seed=0, render=False, learning_rate=0.0001, batch_size=1024,
                               buffer_size=300000, checkpoint=None, continue_training=False,
                               use_hybrid=True):
    """Train the agent with optimized parameters and the option to use hybrid control"""
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Use render flag passed in
    print(f"Creating environment with render={render}")
    
    # Choose environment based on hybrid flag
    if use_hybrid:
        print("Using hybrid RL/manual control environment")
        env_fn = make_hybrid_env(seed=seed, render=render, num_objects=1, use_staged_rewards=True)
    else:
        print("Using standard RL environment")
        env_fn = make_env(seed=seed, render=render, num_objects=1, use_staged_rewards=True)
    
    env = env_fn()
    
    # Ensure we're using fixed object positions
    print("Using FIXED object position for reliable learning")
    
    # Add dimension adapter for CNN policy
    env = DimensionAdapter(env)
    
    # Vectorize environment
    vec_env = DummyVecEnv([lambda: env])
    
    # Use VecTransposeImage for image data
    if isinstance(vec_env.observation_space, gym.spaces.Dict) and 'image' in vec_env.observation_space.spaces:
        vec_env = VecTransposeImage(vec_env)
    
    # Initialize the model
    if checkpoint:
        print(f"Loading model from checkpoint: {checkpoint}")
        model = SAC.load(checkpoint, env=vec_env)
        model.learning_rate = learning_rate
    else:
        print(f"Initializing new model with learning rate: {learning_rate}")
        model = SAC(
            policy=CustomSACPolicy,
            env=vec_env,
            learning_rate=learning_rate,
            batch_size=64,  # Smaller batch size for more frequent updates
            buffer_size=100000,
            train_freq=4,
            gradient_steps=1,
            ent_coef="auto",  # Automatically adjust exploration-exploitation balance
            gamma=0.98,       # Higher discount factor for better long-term planning
            tau=0.02,         # Faster target network updates
            action_noise=None,
            target_update_interval=2000,
            verbose=1,
            tensorboard_log=log_dir,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,  # Save more frequently
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="sac_grasp",
        save_replay_buffer=True,
    )
    
    metrics_callback = TrainingMetricsCallback(
        check_freq=5000,
        log_dir=log_dir,
        filename="training_metrics_optimized.csv"
    )
    
    # Add verbose callback for detailed progress tracking
    verbose_callback = VerboseCallback()
    callbacks = [checkpoint_callback, metrics_callback, verbose_callback]
    
    # Train the model
    print(f"Starting training with optimized parameters for {total_timesteps} timesteps...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name="sac_optimized_training",
        reset_num_timesteps=not continue_training
    )
    
    # Save the final model
    final_model_path = os.path.join(log_dir, "final_model")
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")
    
    # Calculate training time
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Plot the results
    plot_training_results(log_dir)
    
    return model, final_model_path

# ================== TESTING FUNCTION ==================
def test_model(model_path, num_episodes=10, render=True, seed=0, use_hybrid=True):
    """Test the trained agent with comprehensive metrics and visualization"""
    # Create logs directories
    os.makedirs("logs/test_details", exist_ok=True)
    
    # Load environment with appropriate wrapper
    if use_hybrid:
        print("Testing with hybrid RL/manual control environment")
        env = make_hybrid_env(seed=seed, render=render, num_objects=1, use_staged_rewards=True)()
    else:
        print("Testing with standard RL environment")
        env = make_env(seed=seed, render=render, num_objects=1, use_staged_rewards=True)()
    
    # Add dimension adapter
    env = DimensionAdapter(env)
    
    # Add monitor for logging
    env = Monitor(env, "logs/test_results")
    
    # Load the model
    model = SAC.load(model_path, env=env)
    
    # Test the model
    print(f"Testing model for {num_episodes} episodes...")
    
    # Prepare arrays to collect metrics
    rewards = []
    successes = []
    max_stages = []
    grasp_attempts = []
    lift_heights = []
    time_to_success = []
    stage_times = {0: [], 1: [], 2: [], 3: [], 4: []}
    joint_velocities = []
    
    # New metrics for hybrid takeover
    takeover_attempts = []
    takeover_success = []
    takeover_xy_distances = []
    takeover_z_diffs = []
    emergency_takeovers = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        max_stage = 0
        grasp_attempt = 0
        max_lift = 0
        
        # Track when each stage is reached
        stage_start_times = {0: 0, 1: None, 2: None, 3: None, 4: None}
        current_stage = 0
        
        # Track episode-level metrics
        episode_joint_velocities = []
        
        # Track takeover metrics for this episode
        episode_takeovers = 0
        episode_takeover_success = 0
        episode_emergency_takeovers = 0
        episode_takeover_positions = []
        
        # Detailed episode tracking
        episode_trajectory = []
        
        while not done and steps < 200:  # Cap at 200 steps
            # Get action
            action, _ = model.predict(obs, deterministic=True)
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Track metrics
            episode_reward += reward
            steps += 1
            
            # Track position data for analysis
            if hasattr(env.unwrapped, 'robot_id') and hasattr(env.unwrapped, 'obj_id'):
                ee_pos = p.getLinkState(env.unwrapped.robot_id, env.unwrapped.ee_link_index)[0]
                if env.unwrapped.obj_id is not None:
                    obj_pos, _ = p.getBasePositionAndOrientation(env.unwrapped.obj_id)
                    xy_distance = np.linalg.norm(np.array(ee_pos[:2]) - np.array(obj_pos[:2]))
                    z_diff = ee_pos[2] - obj_pos[2]
                else:
                    obj_pos = [0, 0, 0]
                    xy_distance = float('inf')
                    z_diff = 0
                
                # Save trajectory point
                episode_trajectory.append({
                    'step': steps,
                    'ee_pos': list(ee_pos),
                    'obj_pos': list(obj_pos),
                    'xy_distance': float(xy_distance),
                    'z_diff': float(z_diff),
                    'reward': float(reward),
                    'stage': info.get('stage', 0),
                    'action': action.tolist(),
                })
            
            # Track joint velocities for smoothness analysis
            if hasattr(env.unwrapped, 'robot_id') and hasattr(env.unwrapped, 'arm_joint_indices'):
                joint_vels = [abs(p.getJointState(env.unwrapped.robot_id, i)[1]) 
                            for i in env.unwrapped.arm_joint_indices]
                episode_joint_velocities.append(np.mean(joint_vels))
            
            # Track stage transitions
            if 'stage' in info:
                if info['stage'] > current_stage:
                    # Stage transition detected
                    new_stage = info['stage']
                    stage_start_times[new_stage] = steps
                    
                    # Calculate time spent in previous stage
                    if current_stage > 0 and stage_start_times[current_stage] is not None:
                        stage_duration = steps - stage_start_times[current_stage]
                        stage_times[current_stage].append(stage_duration)
                    
                    current_stage = new_stage
                
                max_stage = max(max_stage, info['stage'])
            
            # Track grasp attempts
            try:
                if env.unwrapped._check_grasp():
                    grasp_attempt = 1
            except:
                # Use info if _check_grasp method is not accessible
                if 'is_grasping' in info and info['is_grasping']:
                    grasp_attempt = 1
            
            # Track maximum lift height
            if 'max_lift_height' in info:
                max_lift = max(max_lift, info['max_lift_height'])
            
            # Track hybrid takeover statistics - special case for our wrapper
            if hasattr(env, 'transition_triggered') and env.transition_triggered:
                # Only count each takeover once per episode
                if len(episode_takeover_positions) == episode_takeovers:
                    episode_takeovers += 1
                    # Get positions at takeover time
                    ee_pos = p.getLinkState(env.unwrapped.robot_id, env.unwrapped.ee_link_index)[0]
                    obj_pos, _ = p.getBasePositionAndOrientation(env.unwrapped.obj_id)
                    xy_dist = np.linalg.norm(np.array(ee_pos[:2]) - np.array(obj_pos[:2]))
                    z_diff = ee_pos[2] - obj_pos[2]
                    
                    # Record takeover position
                    episode_takeover_positions.append({
                        'step': steps,
                        'ee_pos': list(ee_pos),
                        'obj_pos': list(obj_pos),
                        'xy_distance': float(xy_dist),
                        'z_diff': float(z_diff),
                        'stage': info.get('stage', 0)
                    })
                    
                    # Record distances for analysis
                    takeover_xy_distances.append(xy_dist)
                    takeover_z_diffs.append(z_diff)
                    
                    # Check if emergency takeover
                    if hasattr(env, '_should_take_over'):
                        _, is_emergency = env._should_take_over()
                        if is_emergency:
                            episode_emergency_takeovers += 1
        
        # Record time to success if successful
        if max_stage == 4:  # Success
            time_to_success.append(steps)
            # Count successful takeover if any takeover led to success
            if episode_takeovers > 0:
                episode_takeover_success = 1
        else:
            time_to_success.append(None)
        
        # Record episode results
        rewards.append(episode_reward)
        
        # Consider success if max stage is 4 (full success)
        success = (max_stage == 4)
        successes.append(success)
        max_stages.append(max_stage)
        grasp_attempts.append(grasp_attempt)
        lift_heights.append(max_lift)
        
        # Record average joint velocities for this episode
        if episode_joint_velocities:
            joint_velocities.append(np.mean(episode_joint_velocities))
        
        # Record takeover statistics
        takeover_attempts.append(episode_takeovers)
        takeover_success.append(episode_takeover_success)
        emergency_takeovers.append(episode_emergency_takeovers)
        
        # Save detailed episode data
        with open(f"logs/test_details/episode_{episode+1}_data.json", "w") as f:
            json.dump({
                'trajectory': episode_trajectory,
                'takeovers': episode_takeover_positions,
                'reward': float(episode_reward),
                'max_stage': int(max_stage),
                'success': bool(success),
                'max_lift': float(max_lift),
                'takeover_attempts': int(episode_takeovers),
                'takeover_success': bool(episode_takeover_success > 0),
            }, f, indent=2)
        
        print(f"Episode {episode+1}: Reward={episode_reward:.2f}, Max Stage={max_stage}, " 
              f"Success={success}, Max Lift={max_lift:.4f}m, Steps={steps}, "
              f"Takeovers={episode_takeovers}")
    
    # Calculate overall statistics
    success_rate = np.mean(successes) * 100
    avg_reward = np.mean(rewards)
    avg_stage = np.mean(max_stages)
    avg_lift = np.mean([h for h in lift_heights if h is not None and h > 0] or [0])
    avg_grasps = np.mean(grasp_attempts)
    
    # Calculate average time to success (excluding failed episodes)
    successful_times = [t for t in time_to_success if t is not None]
    avg_time_to_success = np.mean(successful_times) if successful_times else float('inf')
    
    # Calculate average stage durations
    avg_stage_times = {}
    for stage, times in stage_times.items():
        avg_stage_times[stage] = np.mean(times) if times else 0
    
    # Calculate takeover statistics
    total_takeovers = sum(takeover_attempts)
    takeover_success_rate = 0
    if total_takeovers > 0:
        takeover_success_rate = (sum(takeover_success) / total_takeovers) * 100
    
    # Save aggregated test results
    test_results = {
        'success_rate': float(success_rate),
        'avg_reward': float(avg_reward),
        'avg_stage': float(avg_stage),
        'avg_lift': float(avg_lift),
        'avg_grasps': float(avg_grasps),
        'total_takeovers': int(total_takeovers),
        'takeover_success_rate': float(takeover_success_rate),
        'emergency_takeovers': int(sum(emergency_takeovers)),
        'takeover_positions': {
            'xy_distances': [float(d) for d in takeover_xy_distances],
            'z_diffs': [float(d) for d in takeover_z_diffs]
        }
    }
    
    # Save summary statistics
    with open("logs/test_results_summary.json", "w") as f:
        json.dump(test_results, f, indent=2)
    
    print("\nTest Results:")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Max Stage: {avg_stage:.2f}/4")
    print(f"Average Max Lift Height: {avg_lift:.4f}m")
    print(f"Average Grasp Attempts: {avg_grasps:.2f}")
    if successful_times:
        print(f"Average Time to Success: {avg_time_to_success:.2f} steps")
    
    # Print hybrid takeover statistics
    print("\nHybrid Takeover Statistics:")
    print(f"Total takeover attempts: {total_takeovers}")
    print(f"Takeover success rate: {takeover_success_rate:.2f}%")
    print(f"Total emergency takeovers: {sum(emergency_takeovers)}")
    
    if takeover_xy_distances:
        print(f"Average XY distance at takeover: {np.mean(takeover_xy_distances):.4f}m")
        print(f"Average Z difference at takeover: {np.mean(takeover_z_diffs):.4f}m")
    
    # Print stage time statistics
    print("\nAverage time spent in each stage:")
    for stage, avg_time in avg_stage_times.items():
        stage_names = ["Approach", "Descent", "Grasp", "Lift", "Success"]
        if avg_time > 0:
            print(f"  Stage {stage} ({stage_names[stage]}): {avg_time:.2f} steps")
    
    # Plot test results
    plt.figure(figsize=(15, 12))
    
    # Plot rewards
    plt.subplot(3, 3, 1)
    plt.bar(range(1, num_episodes+1), rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Rewards per Episode')
    plt.grid(True)
    
    # Plot success rate
    plt.subplot(3, 3, 2)
    plt.bar(['Failure', 'Success'], [100-success_rate, success_rate], color=['red', 'green'])
    plt.ylabel('Percentage')
    plt.title(f'Success Rate: {success_rate:.2f}%')
    
    # Plot maximum stages reached
    plt.subplot(3, 3, 3)
    stage_counts = [0, 0, 0, 0, 0]  # Counts for stages 0-4
    for stage in max_stages:
        stage_counts[stage] += 1
    
    stage_names = ['Approach', 'Descent', 'Grasp', 'Lift', 'Success']
    plt.bar(stage_names, [count/num_episodes*100 for count in stage_counts])
    plt.ylabel('Percentage of Episodes')
    plt.title('Maximum Stage Reached')
    
    # Plot lift heights
    plt.subplot(3, 3, 4)
    plt.bar(range(1, num_episodes+1), lift_heights)
    plt.axhline(y=0.05, color='r', linestyle='--', label='Success Threshold (5cm)')
    plt.xlabel('Episode')
    plt.ylabel('Max Lift Height (m)')
    plt.title('Maximum Lift Height per Episode')
    plt.legend()
    plt.grid(True)
    
    # # Plot grasp attempts
    # plt.subplot(3, 3, 5)
    # plt.bar(range(1, num_episodes+1), grasp_attempts)
    # plt.xlabel('Episode')
    # plt.ylabel('Number of Attempts')
    # plt.title('Grasp Attempts per Episode')
    # plt.grid(True)
    
    # Plot time to success
    plt.subplot(3, 3, 6)
    valid_times = [(i+1, t) for i, t in enumerate(time_to_success) if t is not None]
    if valid_times:
        episodes, times = zip(*valid_times)
        plt.bar(episodes, times)
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.title('Time to Success')
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, "No successful episodes", horizontalalignment='center')
        plt.title('Time to Success')
    
    # # Plot average time per stage
    # plt.subplot(3, 3, 7)
    # stages = []
    # times = []
    # for stage, avg_time in avg_stage_times.items():
    #     if avg_time > 0:
    #         stages.append(stage_names[stage])
    #         times.append(avg_time)
    
    # if times:
    #     plt.bar(stages, times)
    #     plt.xlabel('Stage')
    #     plt.ylabel('Average Steps')
    #     plt.title('Average Time per Stage')
    #     plt.grid(True)
    # else:
    #     plt.text(0.5, 0.5, "No stage time data", horizontalalignment='center')
    #     plt.title('Average Time per Stage')
    
    # Plot joint velocities
    plt.subplot(3, 3, 7)
    if joint_velocities:
        plt.bar(range(1, len(joint_velocities)+1), joint_velocities)
        plt.xlabel('Episode')
        plt.ylabel('Average Joint Velocity')
        plt.title('Motion Smoothness')
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, "No joint velocity data", horizontalalignment='center')
        plt.title('Motion Smoothness')
    
    # Plot stage distribution
    plt.subplot(3, 3, 9)
    plt.pie(
        stage_counts, 
        labels=[f"{stage_names[i]}" if stage_counts[i] > 0 else "" for i in range(5)],
        autopct=lambda p: f'{p:.1f}%' if p > 5 else '',
        colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0']
    )
    plt.title('Distribution of Maximum Stages')
    plt.legend(
        [f"{stage_names[i]}" for i in range(5)], 
        loc="center left", 
        bbox_to_anchor=(1, 0.5)
    )
    
    plt.tight_layout()
    plt.savefig("logs/optimized_test_results.png")
    plt.close()
    
    print(f"Test results plot saved to logs/optimized_test_results.png")
    
    return success_rate, avg_reward, avg_stage

# ================== MAIN FUNCTION ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test RL agent with optimized parameters')
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--test', type=str, help='Test a trained model (provide model path)', default=None)
    parser.add_argument('--timesteps', type=int, default=150000, help='Total timesteps for training')
    parser.add_argument('--render', action='store_true', help='Render the environment during training')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes for testing')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for training')
    parser.add_argument('--checkpoint', type=str, default=None, help='Resume training from checkpoint')
    parser.add_argument('--continue_training', action='store_true', 
                       help='Continue training metrics from checkpoint (don\'t reset timestep count)')
    parser.add_argument('--hybrid', action='store_true', help='Use hybrid RL/manual control approach')
    parser.add_argument('--no_hybrid', action='store_true', help='Use pure RL approach (no manual control)')
    args = parser.parse_args()
    
    # Determine whether to use hybrid approach
    use_hybrid = not args.no_hybrid  # Default to hybrid unless explicitly disabled
    
    if args.train:
        print(f"Training with optimized parameters for {args.timesteps} timesteps with seed {args.seed}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Using hybrid control: {use_hybrid}")
        
        if args.checkpoint:
            print(f"Resuming from checkpoint: {args.checkpoint}")
            print(f"Continue training metrics: {args.continue_training}")
            
        model, model_path = train_with_optimized_params(
            total_timesteps=args.timesteps,
            log_dir="logs/optimized_training",
            seed=args.seed,
            render=args.render,
            learning_rate=args.learning_rate,
            checkpoint=args.checkpoint,
            continue_training=args.continue_training,
            use_hybrid=use_hybrid
        )
        
    if args.test:
        if args.test == "last" and args.train:
            # Use the model we just trained
            model_path = os.path.join("logs/optimized_training", "final_model")
        else:
            model_path = args.test
            
        print(f"Testing model {model_path} for {args.episodes} episodes")
        print(f"Using hybrid control: {use_hybrid}")
        
        success_rate, avg_reward, avg_stage = test_model(
            model_path=model_path,
            num_episodes=args.episodes,
            render=True,  # Always render during testing
            seed=args.seed,
            use_hybrid=use_hybrid
        )
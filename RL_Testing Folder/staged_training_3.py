# staged_training.py
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

# Import your existing environment and policy
from env_setup_multiobject import make_env, VisualRoboticArmEnv
from cnn_policy import CustomSACPolicy

# Import dimension adapter from your existing code
from RL_train import DimensionAdapter, TrainingMetricsCallback, plot_training_results

# Create directories for logs and models
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

def train_with_staged_rewards(total_timesteps=100000, log_dir="logs/staged_training", 
                             seed=0, render=False, learning_rate=0.0003, batch_size=256,
                             buffer_size=100000, checkpoint=None, continue_training=False):
    """Train the agent with the integrated staged rewards"""
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    render=False
    print(f"Creating environment with render={render}")
    # Create environment with staged rewards (already integrated)
    env = make_env(seed=seed, render=render, num_objects=1, use_staged_rewards=True)()
    
    # Add dimension adapter if needed
    env = DimensionAdapter(env)
    
    # Vectorize environment
    vec_env = DummyVecEnv([lambda: env])
    
    # Use VecTransposeImage if needed
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
            buffer_size=buffer_size,
            batch_size=batch_size,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            verbose=1,
            tensorboard_log=log_dir,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="sac_grasp"
    )
    
    metrics_callback = TrainingMetricsCallback(
        check_freq=5000,
        log_dir=log_dir,
        filename="training_metrics_staged.csv"  # Add this parameter
    )
    
    # Train the model
    print(f"Starting training with staged rewards for {total_timesteps} timesteps...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, metrics_callback],
        tb_log_name="sac_staged_training",
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

def test_staged_rewards_model(model_path, num_episodes=10, render=True, seed=0):
    """Test the trained agent with staged rewards"""
    # Load environment with staged rewards
    env = make_env(seed=seed, render=render, num_objects=1, use_staged_rewards=True)()
    
    # Add dimension adapter if needed
    env = DimensionAdapter(env)
    
    # Add monitor for logging
    env = Monitor(env, "logs/staged_test")
    
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
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        max_stage = 0
        grasp_attempt = 0
        max_lift = 0
        
        while not done and steps < 200:  # Cap at 200 steps
            # Get action
            action, _ = model.predict(obs, deterministic=True)
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Track metrics
            episode_reward += reward
            steps += 1
            
            # Track maximum stage reached
            if 'stage' in info:
                max_stage = max(max_stage, info['stage'])
            
            # Track grasp attempts
            if env.unwrapped._check_grasp():
                grasp_attempt += 1
            
            # Track maximum lift height
            if 'max_lift_height' in info:
                max_lift = max(max_lift, info['max_lift_height'])
        
        # Record episode results
        rewards.append(episode_reward)
        
        # Consider success if max stage is 4 (full success)
        success = (max_stage == 4)
        successes.append(success)
        max_stages.append(max_stage)
        grasp_attempts.append(grasp_attempt)
        lift_heights.append(max_lift)
        
        print(f"Episode {episode+1}: Reward={episode_reward:.2f}, Max Stage={max_stage}, " 
              f"Success={success}, Max Lift={max_lift:.2f}m, Grasp Attempts={grasp_attempt}")
    
    # Calculate overall statistics
    success_rate = np.mean(successes) * 100
    avg_reward = np.mean(rewards)
    avg_stage = np.mean(max_stages)
    avg_lift = np.mean(lift_heights)
    avg_grasps = np.mean(grasp_attempts)
    
    print("\nTest Results:")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Max Stage: {avg_stage:.2f}/4")
    print(f"Average Max Lift Height: {avg_lift:.4f}m")
    print(f"Average Grasp Attempts: {avg_grasps:.2f}")
    
    # Plot test results
    plt.figure(figsize=(15, 10))
    
    # Plot rewards
    plt.subplot(2, 3, 1)
    plt.bar(range(1, num_episodes+1), rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Rewards per Episode')
    plt.grid(True)
    
    # Plot success rate
    plt.subplot(2, 3, 2)
    plt.bar(['Failure', 'Success'], [100-success_rate, success_rate], color=['red', 'green'])
    plt.ylabel('Percentage')
    plt.title(f'Success Rate: {success_rate:.2f}%')
    
    # Plot maximum stages reached
    plt.subplot(2, 3, 3)
    stage_counts = [0, 0, 0, 0, 0]  # Counts for stages 0-4
    for stage in max_stages:
        stage_counts[stage] += 1
    
    stage_names = ['Approach', 'Descent', 'Grasp', 'Lift', 'Success']
    plt.bar(stage_names, [count/num_episodes*100 for count in stage_counts])
    plt.ylabel('Percentage of Episodes')
    plt.title('Maximum Stage Reached')
    
    # Plot lift heights
    plt.subplot(2, 3, 4)
    plt.bar(range(1, num_episodes+1), lift_heights)
    plt.axhline(y=0.05, color='r', linestyle='--', label='Success Threshold (5cm)')
    plt.xlabel('Episode')
    plt.ylabel('Max Lift Height (m)')
    plt.title('Maximum Lift Height per Episode')
    plt.legend()
    plt.grid(True)
    
    # Plot grasp attempts
    plt.subplot(2, 3, 5)
    plt.bar(range(1, num_episodes+1), grasp_attempts)
    plt.xlabel('Episode')
    plt.ylabel('Number of Attempts')
    plt.title('Grasp Attempts per Episode')
    plt.grid(True)
    
    # Plot maximum stage distribution
    plt.subplot(2, 3, 6)
    labels = [f"Stage {i}: {name}" for i, name in enumerate(stage_names)]
    plt.pie(stage_counts, labels=labels, autopct='%1.1f%%', 
            colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0'])
    plt.title('Distribution of Maximum Stages')
    
    plt.tight_layout()
    plt.savefig("logs/staged_test_results.png")
    plt.close()
    
    print(f"Test results plot saved to logs/staged_test_results.png")
    
    return success_rate, avg_reward, avg_stage

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train or test RL agent with staged rewards')
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--test', type=str, help='Test a trained model (provide model path)', default=None)
    parser.add_argument('--timesteps', type=int, default=100000, help='Total timesteps for training')
    parser.add_argument('--render', action='store_true', help='Render the environment during training')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes for testing')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='Learning rate for training')
    parser.add_argument('--checkpoint', type=str, default=None, help='Resume training from checkpoint')
    parser.add_argument('--continue_training', action='store_true', 
                       help='Continue training metrics from checkpoint (don\'t reset timestep count)')
    args = parser.parse_args()
    
    if args.train:
        print(f"Training with staged rewards for {args.timesteps} timesteps with seed {args.seed}")
        print(f"Learning rate: {args.learning_rate}")
        if args.checkpoint:
            print(f"Resuming from checkpoint: {args.checkpoint}")
            print(f"Continue training metrics: {args.continue_training}")
            
        model, model_path = train_with_staged_rewards(
            total_timesteps=args.timesteps,
            log_dir="logs/staged_training",
            seed=args.seed,
            render=args.render,
            learning_rate=args.learning_rate,
            checkpoint=args.checkpoint,
            continue_training=args.continue_training
        )
        
    if args.test:
        if args.test == "last" and args.train:
            # Use the model we just trained
            model_path = os.path.join("logs/staged_training", "final_model")
        else:
            model_path = args.test
            
        print(f"Testing model {model_path} for {args.episodes} episodes")
        success_rate, avg_reward, avg_stage = test_staged_rewards_model(
            model_path=model_path,
            num_episodes=args.episodes,
            render=True  # Always render during testing
        )
        
# RL_train.py
import os
import time
import numpy as np
import torch
import pybullet as p
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
from gymnasium import spaces

# Import your existing environment
from env_setup_multiobject import make_env, VisualRoboticArmEnv
from cnn_policy import CustomSACPolicy  # Your existing CNN policy

# Create directories for logs and models
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Wrapper to handle dimension mismatch
class DimensionAdapter(gym.Wrapper):
    """Adapts state dimensions to match what the policy expects"""
    
    def __init__(self, env, expected_dim=24):
        super().__init__(env)
        
        # Get actual dimensions from environment
        actual_dim = env.observation_space['state'].shape[0]
        self.expected_dim = expected_dim
        
        print(f"Adapting state dimensions: actual={actual_dim}, expected={expected_dim}")
        
        # Modify observation space to match expected dimensions
        self.observation_space = spaces.Dict({
            "image": self.env.observation_space["image"],
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(expected_dim,), dtype=np.float32)
        })
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Adapt state vector size
        if 'state' in obs:
            obs['state'] = self._adapt_state_dim(obs['state'])
        
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Adapt state vector size
        if 'state' in obs:
            obs['state'] = self._adapt_state_dim(obs['state'])
        
        return obs, reward, terminated, truncated, info
    
    def _adapt_state_dim(self, state):
        """Adapt state vector to expected dimensions"""
        actual_dim = len(state)
        
        if actual_dim == self.expected_dim:
            return state
        
        if actual_dim > self.expected_dim:
            # If actual is larger, trim excess dimensions
            return state[:self.expected_dim]
        else:
            # If actual is smaller, pad with zeros
            padded = np.zeros(self.expected_dim, dtype=np.float32)
            padded[:actual_dim] = state
            return padded

# Callback for saving training metrics
class TrainingMetricsCallback(BaseCallback):
    def __init__(self, check_freq=1000, log_dir="logs/", verbose=1,filename="training_metrics.csv"):
        super(TrainingMetricsCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.filename = filename
        
    def _init_callback(self):
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            # Get current statistics
            try:
                x, y = ts2xy(load_results(self.log_dir), 'timesteps')
                if len(x) > 0:
                    # Compute mean reward over last 100 episodes
                    mean_reward = np.mean(y[-100:])
                    
                    # Log to console
                    if self.verbose > 0:
                        print(f"Num timesteps: {self.num_timesteps}")
                        print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward: {mean_reward:.2f}")
                    
                    # Save best model
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        if self.verbose > 0:
                            print(f"Saving new best model to {self.save_path}")
                        self.model.save(os.path.join(self.save_path, 'best_model'))
                        
                    # Save metrics for plotting
                    with open(os.path.join(self.log_dir, 'training_metrics.csv'), 'a') as f:
                        if os.stat(os.path.join(self.log_dir, 'training_metrics.csv')).st_size == 0:
                            f.write("timesteps,mean_reward,best_reward\n")
                        f.write(f"{self.num_timesteps},{mean_reward},{self.best_mean_reward}\n")
            except Exception as e:
                print(f"Warning: Could not save metrics - {e}")
                
        return True

# Function to create a monitored environment for logging
def make_monitored_env(rank, seed=0, render=False, log_dir="logs/", stage=1):
    def _init():
        # Create environment with the specified stage
        env = VisualRoboticArmEnv(render=render, num_objects=1, stage=stage)
        
        # Add dimension adapter
        env = DimensionAdapter(env)
        
        # Add monitor for logging
        env = Monitor(env, os.path.join(log_dir, str(rank)))
        
        # Set seed through reset method
        env.reset(seed=seed + rank)
        
        return env
    return _init

# Function to plot training progress
def plot_training_results(log_dir="logs/", title="Learning Curve"):
    # Load and plot the data
    try:
        data_file = os.path.join(log_dir, 'training_metrics.csv')
        if not os.path.exists(data_file) or os.stat(data_file).st_size == 0:
            print("No training data available for plotting")
            return
            
        data = np.loadtxt(data_file, delimiter=',', skiprows=1)
        
        if data.size > 0:
            # Extract data
            if len(data.shape) == 1:  # Only one row
                timesteps = [data[0]]
                mean_rewards = [data[1]]
                best_rewards = [data[2]]
            else:
                timesteps = data[:, 0]
                mean_rewards = data[:, 1]
                best_rewards = data[:, 2]
            
            # Create plots
            plt.figure(figsize=(12, 6))
            
            # Plot mean rewards
            plt.subplot(1, 2, 1)
            plt.plot(timesteps, mean_rewards, label='Mean Reward')
            plt.xlabel('Timesteps')
            plt.ylabel('Mean Reward')
            plt.title('Training Progress: Mean Reward')
            plt.legend()
            plt.grid(True)
            
            # Plot best rewards
            plt.subplot(1, 2, 2)
            plt.plot(timesteps, best_rewards, label='Best Reward', color='green')
            plt.xlabel('Timesteps')
            plt.ylabel('Best Reward')
            plt.title('Training Progress: Best Reward')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(log_dir, 'learning_curve.png'))
            plt.close()
            
            print(f"Learning curve saved to {os.path.join(log_dir, 'learning_curve.png')}")
        else:
            print("No data available for plotting")
    except Exception as e:
        print(f"Error plotting results: {e}")

# Train the agent
def train_agent(total_timesteps=100000, log_dir="logs/", seed=0, render=False, 
                learning_rate=0.0003, checkpoint=None, continue_training=False,
                curriculum=True, stage_switch=50000):
    """
    Train the agent with curriculum learning support
    
    Args:
        total_timesteps: Total training steps
        curriculum: Whether to use curriculum learning
        stage_switch: When to switch from stage 1 to stage 2 (in timesteps)
    """
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Start with stage 1 (fixed positions)
    current_stage = 1
    
    # Create environment with stage 1
    env = DummyVecEnv([make_monitored_env(0, seed, render, log_dir, stage=current_stage)])
    
    # Use VecTransposeImage if needed
    if isinstance(env.observation_space, gym.spaces.Dict) and 'image' in env.observation_space.spaces:
        env = VecTransposeImage(env)
    
    # Initialize the model
    if checkpoint:
        print(f"Loading model from checkpoint: {checkpoint}")
        model = SAC.load(checkpoint, env=env)
        model.learning_rate = learning_rate
    else:
        print(f"Initializing new model with learning rate: {learning_rate}")
        model = SAC(
            policy=CustomSACPolicy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=100000,
            batch_size=256,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            verbose=1,
            tensorboard_log=log_dir,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="sac_grasp"
    )
    
    metrics_callback = TrainingMetricsCallback(
        check_freq=1000,
        log_dir=log_dir
    )
    
    # Train the model with curriculum
    print(f"Starting training for {total_timesteps} timesteps...")
    start_time = time.time()
    
    # If using curriculum learning
    if curriculum:
        # First stage - train with fixed positions
        first_stage_steps = min(stage_switch, total_timesteps)
        remaining_steps = total_timesteps - first_stage_steps
        
        print(f"Stage 1 (fixed positions): Training for {first_stage_steps} timesteps...")
        model.learn(
            total_timesteps=first_stage_steps,
            callback=[checkpoint_callback, metrics_callback],
            tb_log_name="sac_training_stage1",
            reset_num_timesteps=not continue_training
        )
        
        # Save stage 1 model
        stage1_model_path = os.path.join(log_dir, "stage1_model")
        model.save(stage1_model_path)
        print(f"Stage 1 model saved to {stage1_model_path}")
        
        # Switch to stage 2 if there are remaining steps
        if remaining_steps > 0:
            print(f"Switching to Stage 2 (variable positions)...")
            
            # Create new environment with stage 2
            env = DummyVecEnv([make_monitored_env(0, seed, render, log_dir, stage=2)])
            if isinstance(env.observation_space, gym.spaces.Dict) and 'image' in env.observation_space.spaces:
                env = VecTransposeImage(env)
            
            # Update model with new environment
            model.set_env(env)
            
            print(f"Stage 2: Training for {remaining_steps} timesteps...")
            model.learn(
                total_timesteps=remaining_steps,
                callback=[checkpoint_callback, metrics_callback],
                tb_log_name="sac_training_stage2",
                reset_num_timesteps=False  # Continue from previous timesteps
            )
    else:
        # Standard training without curriculum
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, metrics_callback],
            tb_log_name="sac_training",
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
    
    return model, os.path.join(log_dir, "final_model")

# Test the trained agent
def test_agent(model_path, num_episodes=10, render=True):
    # Load environment
    env = VisualRoboticArmEnv(render=render, num_objects=1)
    env = DimensionAdapter(env)
    env = Monitor(env, "logs/test")
    
    # Load the model
    model = SAC.load(model_path, env=env)
    
    # Test the model
    print(f"Testing model for {num_episodes} episodes...")
    
    # Prepare arrays to collect metrics
    rewards = []
    successes = []
    grasp_attempts = []
    lift_heights = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        max_lift_height = 0
        grasp_attempt = 0
        
        while not done and steps < 200:  # Cap at 200 steps
            # Get action
            action, _ = model.predict(obs, deterministic=True)
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Track metrics
            episode_reward += reward
            steps += 1
            
            # Track grasp attempts
            if env.unwrapped._check_grasp():
                grasp_attempt += 1
            
            # Track maximum lift height
            if env.unwrapped.obj_id is not None:
                obj_pos, _ = p.getBasePositionAndOrientation(env.unwrapped.obj_id)
                lift_height = obj_pos[2] - 0.45  # Assuming initial height is 0.45
                max_lift_height = max(max_lift_height, lift_height)
        
        # Record episode results
        rewards.append(episode_reward)
        
        # Consider success if object was lifted at least 5cm
        success = max_lift_height > 0.05
        successes.append(success)
        grasp_attempts.append(grasp_attempt)
        lift_heights.append(max_lift_height)
        
        print(f"Episode {episode+1}: Reward={episode_reward:.2f}, Success={success}, " 
              f"Max Lift={max_lift_height:.2f}m, Grasp Attempts={grasp_attempt}")
    
    # Calculate overall statistics
    success_rate = np.mean(successes) * 100
    avg_reward = np.mean(rewards)
    avg_lift = np.mean(lift_heights)
    avg_grasps = np.mean(grasp_attempts)
    
    print("\nTest Results:")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Max Lift Height: {avg_lift:.4f}m")
    print(f"Average Grasp Attempts: {avg_grasps:.2f}")
    
    # Plot test results
    plt.figure(figsize=(12, 8))
    
    # Plot rewards
    plt.subplot(2, 2, 1)
    plt.bar(range(1, num_episodes+1), rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Rewards per Episode')
    plt.grid(True)
    
    # Plot success rate
    plt.subplot(2, 2, 2)
    plt.bar(['Failure', 'Success'], [100-success_rate, success_rate], color=['red', 'green'])
    plt.ylabel('Percentage')
    plt.title(f'Success Rate: {success_rate:.2f}%')
    
    # Plot lift heights
    plt.subplot(2, 2, 3)
    plt.bar(range(1, num_episodes+1), lift_heights)
    plt.axhline(y=0.05, color='r', linestyle='--', label='Success Threshold (5cm)')
    plt.xlabel('Episode')
    plt.ylabel('Max Lift Height (m)')
    plt.title('Maximum Lift Height per Episode')
    plt.legend()
    plt.grid(True)
    
    # Plot grasp attempts
    plt.subplot(2, 2, 4)
    plt.bar(range(1, num_episodes+1), grasp_attempts)
    plt.xlabel('Episode')
    plt.ylabel('Number of Attempts')
    plt.title('Grasp Attempts per Episode')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("logs/test_results.png")
    plt.close()
    
    print(f"Test results plot saved to logs/test_results.png")
    
    return success_rate, avg_reward, avg_lift

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train or test RL agent for grasping')
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--test', type=str, help='Test a trained model (provide model path)', default=None)
    parser.add_argument('--timesteps', type=int, default=100000, help='Total timesteps for training')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes for testing')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='Learning rate for training')
    parser.add_argument('--checkpoint', type=str, default=None, help='Resume training from checkpoint')
    parser.add_argument('--continue_training', action='store_true', 
                       help='Continue training metrics from checkpoint (don\'t reset timestep count)')
    parser.add_argument('--curriculum', action='store_true', help='Use curriculum learning (fixed position -> variable)')
    parser.add_argument('--stage_switch', type=int, default=50000, help='When to switch from stage 1 to stage 2 (timesteps)')
    args = parser.parse_args()
    
    if args.train:
        print(f"Training for {args.timesteps} timesteps with seed {args.seed}")
        print(f"Learning rate: {args.learning_rate}")
        if args.checkpoint:
            print(f"Resuming from checkpoint: {args.checkpoint}")
            print(f"Continue training metrics: {args.continue_training}")
            
        model, model_path = train_agent(
            total_timesteps=args.timesteps,
            log_dir="logs/",
            seed=args.seed,
            render=args.render,
            learning_rate=args.learning_rate,
            checkpoint=args.checkpoint,
            continue_training=args.continue_training,
            curriculum=args.curriculum,
            stage_switch=args.stage_switch
        )
        
    if args.test:
        if args.test == "last" and args.train:
            # Use the model we just trained
            model_path = os.path.join("logs/", "final_model")
        else:
            model_path = args.test
            
        print(f"Testing model {model_path} for {args.episodes} episodes")
        success_rate, avg_reward, avg_lift = test_agent(
            model_path=model_path,
            num_episodes=args.episodes,
            render=False  # Always render during testing
        )
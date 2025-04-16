from stable_baselines3 import SAC
from env_setup import make_env

# Create the environment
env = make_env(render=False)()
# or if you had a vectorized env previously, replicate that setup accordingly

# Load the checkpointed model
model = SAC.load("models/robot_arm_10000_steps", env=env, device="cuda")

# Continue training for more timesteps:
# Note: set reset_num_timesteps=False so that the internal counter doesn't reset
model.learn(total_timesteps=300000, reset_num_timesteps=False)

# Save the updated model
model.save("models/robot_arm_300000_yolo_steps")

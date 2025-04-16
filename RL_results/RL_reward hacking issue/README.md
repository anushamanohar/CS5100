1. Results Summary

The agent achieved a 100% success rate according to the defined success metric (object height change > 5cm), but through an unexpected strategy that showcased the importance of correct reward structure into reward function design in RL.

2. Implementation Details

Environment:

Task: Robotic arm manipulation of objects placed on a tray
Observation Space: RGB images from a fixed camera position
Action Space: Continuous control of the robotic end effector
Success Criterion: Object lifted at least 5cm from its initial position

3. Training Methodology

Algorithm: 

Network Architecture:  SAC (Soft Actor-Critic) policy with CNN-based image processing
Computational Resources: GPU(CUDA Training - Gaming Laptop - ROG)

3. Results Analysis

Quantitative Results

Success Rate: 100% across all test episodes
Average Lift Height: ~12cm (above the 5cm threshold)
Grasp Attempts: 0 across all episodes

4. Unexpected Behavior: Reward Hacking

The agent developed an intriguing strategy that differed from what was expected from it to do:

Push-to-Edge Strategy: Instead of grasping and lifting the object, the agent learned to push the object so that it falls or slides to the edge of the tray.

Height Achievement: This strategy still satisfied the success condition because the object's height changed by more than the required 5cm threshold.

Zero Traditional Grasps: The grasp attempts metric confirmed that the agent completely bypassed learning the intended grasping behavior.

5. Results:

Showcased Reward Hacking Phenomenon:

This approach provides a clear example of "reward hacking" in reinforcement learning - a situation where an agent satisfies the defined reward criteria but in a way that doesn't align with the intended behavior. The agent found an easier path to maximize rewards by pushing rather than grasping.

6. Significance


-Demonstrates the importance of careful reward function design
-RL agents in finding optimal solutions
-Shows how unintended behaviors can emerge when success criteria are under specified

7. Lessons Learned

Reward Function Design: Success metrics must be carefully defined to prevent exploitation of loopholes.

Multi-Objective Rewards: Simple height-based metrics are insufficient for complex manipulation tasks.
Constraint Specification: All desired behaviors should be explicitly incentivized or constrained.

8. Future Improvements

To address the reward hacking and encourage true grasping behavior, future iterations could:

Tried correcting the reward structure to measure success as lifting the object and closing the gripper as well
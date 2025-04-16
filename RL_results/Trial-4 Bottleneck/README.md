1. Results Summary

After extensive training and reward tuning, the RL agent demonstrates consistent success in the initial stage of the task, but fails to progress to full task completion.

Current Performance (Post 150,000 Steps):

Stage 1 (Positioning/Descent): Mastered with 100% consistency

Stage 2 (Grasping): Not reached

Test Success Rate: 0.00%

Average Reward: ~54.0 (up from 6.2 early in training)

Grasp Attempts: 0 across all test episodes

These results indicate that the agent has reached a local optimum, exploiting the reward shaping for positioning but failing to explore further into the task sequence.

2. Implementation Details

Environment

Task: Robotic grasp-and-lift

Visual Input: YOLO-based object detection

Action Space: Continuous 6-DOF control + gripper

Reward Structure: Five-stage shaped rewards

Training Episodes: 150,000+ timesteps

Evaluation: Automated metrics (lift height, reward, grasp attempts)

3. Current Challenge: Exploration Bottleneck

The agent has reached an "exploration bottleneck," a known issue in reinforcement learning where:

The policy discovers a locally optimal behavior (e.g., hovering above the object)

The next required actions (e.g., gripper closure, fine descent) are difficult to discover due to low probability and sparse reward reinforcement

Without a clear reward incentive or guidance for attempting new action combinations, the agent fails to explore beyond its current strategy

Despite increasing total rewards and training time, the agent does not transition to grasping behavior, confirming that this is not a perception or detection issue.

4. Recommended Solution: Hybrid Control

To overcome the bottleneck while preserving the successful aspects of training, a hybrid control strategy is adapted next:

Phase 1: RL-Based Positioning

Use the trained RL policy to position the end-effector directly above the object (Stage 0 → Stage 1)


Phase 2: Manual Grasp Execution

Once the agent reaches a stable hover state above the object, trigger a manual grasping sequence

This sequence would control:

Vertical descent

Gripper closure

Lifting motion


5. Implementation Plan

To implement the hybrid system:

Stage Detection: Modify the environment to detect when the RL agent achieves Stage 1 

Control Switch: Trigger manual control logic once Stage 1 conditions are met

Manual Grasp Logic:

Descend a fixed distance

Close the gripper

Lift a fixed height

Evaluate success (object displacement, gripper status)
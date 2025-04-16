1. Problem Background

In prior experiments:

The RL agent consistently mastered Stage 1: Positioning

However, it failed to transition to Stage 2: Grasping, despite 150,000 training steps and improved reward shaping

This was identified as an exploration bottleneck, a known RL limitation when the next-stage behavior is too complex or under-incentivized

2. Hybrid Control Solution

To overcome this, we implemented a hybrid RL system:

Stage 0 → 1 (Approach/Descent): Controlled by trained RL agent

Stage 2 → 3 (Grasp & Lift): Handled via manual scripted control

Stage 4+ (Optional Placement, Return, etc.): Remains open for future extension (RL or manual)

Implementation Details:

A wrapper class (HybridRLEnv) detects when Stage 1 is reached

Control is handed off to manual grasp and lift logic at that point

After successful grasp and lift, optional control is returned to RL

3. Test Results

The hybrid system was evaluated across 5 test episodes, shown in the figure optimized_test_results.png.

Key Observations:

Success Rate: 100% across all test episodes

Grasp Attempts: Successfully initiated during manual control

Lift Height: Object consistently lifted 

Time to Success: Fixed, consistent across episodes

Motion Smoothness: Stable average joint velocity

Stage Distribution: All episodes reached final stage successfully

4. Significance

RL handles adaptable components (like noisy visual-based positioning)

Manual control ensures reliability for critical stages (like grasping)


5. Lessons Learned

Reward shaping alone isn't always enough—some actions are too complex or sensitive for RL to explore effectively without risk

Hybridization is efficient—it maximizes RL’s adaptability while minimizing development time

Modular task breakdown enables flexible control handoffs

6. Future Work

-Add final stages: object placement, release, and reset behavior

-Use RL confidence or value estimates to trigger handoffs dynamically

-Explore curriculum learning to gradually replace scripted logic with learned control

-Extend hybrid architecture to multi-object manipulation tasks
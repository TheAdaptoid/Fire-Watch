# Autonomous Flight Control with Q-Networks

Marion Forrest

School of Computing

University of North Florida

Jacksonville, United States

<the.marion.404@gmail.com>

## Abstract

**This work details a simple implementation of a Q-Network controlled autonomous cruising system for long-haul flights. Diverging from previous methods reliant on human input, the agent was designed to learn optimal flight behavior independently. After training for 1,250 episodes in a simulated environment, the agent failed to achieve and maintain an ideal cruising state and frequently attempted aggressive maneuvers.**

***Keywords – Aviation, Reinforcement Learning, Artificial Neural Network, Q-Network***

## Introduction and Related Work

A direct flight from one end of the globe to the other will never cease to be an attractive option. Leading to the proliferation of evermore efficient airframes and engines, capable of lengthy nonstop flights. These long-haul routes will become ever easier for aircraft to operate on, however, the same cannot be said for pilots of these routes. There may soon come a time in which airline pilots have to endure nonstop flights with durations exceeding 20 hours. However, in line with advancements in machine learning and transportation technologies, that potential future may not come to fruition with the introduction of fully autonomous flight systems; systems capable of circumnavigating the planet without a single human input.

The ideal implementation of such a system would be capable of taxing, taking off, cruising, navigating, and landing all under its own power. This paper however, will investigate a narrower scope and focus on an implementation of an artificial neural network controlled cruising phase.

Previous investigations into neural network controlled flight systems had agents trained on human inputs. Observing human actions across various flight scenarios and scoring the agent's decisions based on how well they aligned with expected human behavior [1]  [2]. Although, this approach has yielded substantial results, this work would like to propose an alternative method; one in which the agent would learn optimal flight behavior entirely on its own absent of human experience, feedback, or input. This agent's evaluation criteria would be based on how well its current state of flight aligns with an ideal final state.

## Implementation

In the context of an autonomous cruiser, the agent will be assessed based on how closely the current flight behavior of the aircraft aligns with the expected behavior of a cruising aircraft. For this research effort, expected cruising behavior was defined as level flight with nearly zero degrees of pitch or roll, an approximate speed of 0.85 Mach, and an altitude of 6,000 meters above sea level. Although airliners typically cruise at around 9,000 meters, the limitations of the simulated environment constrained the target cruising state to 6,000 meters.

### Network Architecture

This system is designed around a Q-Network comprised of four input nodes, nine output nodes, and only one hidden layer with eight nodes. Hyperbolic tangent was used as the activation function for the input and hidden layer to better encode negative values from the input data. This simple architecture was chosen in an attempt to replicate the model performance others have observed with less complex networks [1].

Each of the four input nodes correspond to the flight characteristics of altitude, pitch, roll, and speed. Altitude is measured in terms of meters above sea level and normalized against a maximum of 12,000 meters, which is the typical flight ceiling for commercial aircraft. This value is then scaled to the range of zero to one. The aircraft's pitch and roll are both measured in terms of degrees relative to the horizon. Pitch is normalized against the range -90 to 90 degrees and roll is normalized against the range -180 to 180 degrees. Both metrics are scaled to the range negative one to one. Lastly, speed is measured in multiples of the speed of sound, which gives this metric an effective range of zero to one, however values exceeding one are possible.

The output nodes are mapped to the nine distinct actions. The first two nodes are mapped to increasing and decreasing throttle percentage in increments of 0.01%. Nodes three through eight are mapped to increasing and decreasing pitch, yaw, and roll in increments of 0.01 degrees. The ninth node is mapped to an action that corresponds with no action being taken in the simulation. Actions are selected based on an ε-greedy policy to prevent convergence at local maximums.

### Simulation Environment

This work leverages the detailed and flexible physics-based environment provided by the popular educational spaceflight simulator, Kerbal Space Program (KSP) [3]. The agent's actions were executed through a flight manager client which made remote procedure calls to a server attached to the simulation [4]. As previously stated, the simulation presented one main limitation. The engines used on the simulated aircraft performed most optimally at around 6,000 meters above sea level and would begin to significantly decrease in power at the industry standard 9,000 meters. As such the target altitude used during training had to be adjusted down to ensure the aircraft would remain stable and operational once the agent reached the desired cruising state.

### Evaluation System

The agent's reward (1) is a linear combination of altitude, speed, pitch, and roll components. The weights of each component were adjusted manually during training to incentivize and discourage certain behavior.

$$
R=w_{\mathrm{altitude}}c_{\mathrm{altitude}}+w_{\mathrm{speed}}c_{\mathrm{speed}}+w_{\mathrm{pitch}}c_{\mathrm{pitch}}+w_{\mathrm{roll}}c_{\mathrm{roll}}
$$

The altitude component is calculated as the absolute value of the difference between the aircraft's current altitude and the target altitude, divided by the target altitude (2). This value is then multiplied by negative one to incentivize an altitude that does not deviate from the ideal. The speed component is calculated in a similar manner (3).

$$
c_{\text{altitude}} = -1 \cdot \frac{| \text{altitude}_c - \text{altitude}_t |}{\text{altitude}_t}
$$

$$
c_{\text{speed}} = -1 \cdot \frac{| \text{speed}_c - \text{speed}_t |}{\text{speed}_t}
$$

The pitch component (4) is calculated as the aircraft's current pitch angle plus 90 degrees, divided by 180, then multiplied by two. The difference of one is then calculated to produce a value in the range of negative one to one. Finally, the absolute value is multiplied by negative one to reward pitch angles closer to zero. The roll component is calculated similarly to reward level flight at near zero angles (5).

$$
c_\text{pitch}=-1 \cdot \left| \left( \frac{pitch_c + 90}{180} \cdot 2 \right) - 1 \right|
$$

$$
c_\text{roll}=-1 \cdot \left| \left( \frac{roll_c + 180}{360} \cdot 2 \right) - 1 \right|
$$

Attempting to reach and maintain a zero reward signifies that the agent is performing optimally, as the maximum reward possible within this evaluation system is zero. Consistent near zero rewards are indicators of alignment with the ideal cruising state of 6,000 meters altitude, 0.85 Mach, and near-zero pitch and roll degrees.

## Training

Episodes are initialized with the aircraft flying within range of the ideal state. This being a level flight at less than 200 meters from the target altitude, while accelerating up to 0.85 mach. If the agent crashes the aircraft, the episode is terminated and the simulation resets back to the initial state and a new episode begins. Additionally, the reward received by the agent during the terminal state is reduced by 100 to discourage crash inducing behavior. On average, an episode lasted four minutes.

### Weight Adjustments

The weights of the reward function were adjusted periodically in accordance with any observable trends in the components of the reward to prioritize certain behavior. For example, altitude was typically kept higher than that of pitch to provide sufficient leeway for the agent to increase the aircraft's angle of attack to make necessary altitude adjustments without suffering large losses during the maneuver. At the beginning of the training process the weights for pitch, roll, altitude, and speed were set to 10, 10, 12, and 10 respectively. After 1,250 episodes, the final set of adjustments settled the weights at 10, 2.5, 20, and 2.

## Results

Following 1,250 training episodes, with an average of 1,427 states observed per episode, the agent was unable to achieve and sustain an ideal cruising state. Its failure to effectively learn and adapt to the environment and action space resulted in consistent negative rewards. The agent routinely initiated aggressive maneuvers that led to unrecoverable stalls, rapid altitude loss, inverted flight, and frequent crashes.

## Conclusion

This research effort sought to develop a Q-Network controlled autonomous cruising system for long-haul flights. The proposed method diverged from previous work that relied on human input, instead opting for the agent to learn optimal flight behavior independently. The agent's performance was evaluated based on how closely its current flight behavior aligned with an ideal cruising state of level flight at 6,000 meters, 0.85 Mach, and near-zero pitch and roll.

The Q-network consisted of four input nodes for altitude, pitch, roll, and speed, one hidden layer with eight nodes, and nine output nodes mapped to throttle and control surface adjustments. The agent's reward function was a linear combination of altitude, speed, pitch, and roll components, each with manually adjusted weights to incentivize desired behavior.

Despite an extensive training cycle, the agent failed to achieve and maintain an ideal cruising state after 1,250 episodes. The agent frequently attempted aggressive maneuvers that resulted in unfavorable terminal states. While the concept may still hold promise, significant changes need to be considered and made to improve performance before this implementation can be considered viable and practical.

## Future Work

Expanding upon the concept introduced in this work, one potential approach to improving model performance would be to employ the use of a much larger and more complex Q-Network. With only four inputs, the agent likely lacked enough data to correctly identify critical flight states. Additional input nodes for lift, drag, and thrust force could enable the agent to better identify stall and near-stall states. Furthermore, increasing the number of hidden layers may improve the agent's ability to effectively analyze and respond to these states.

## References

[1] H. Baomar and P. J. Bentley, "An Intelligent Autopilot System that learns piloting skills from human pilots by imitation," 2016 International Conference on Unmanned Aircraft Systems (ICUAS), Arlington, VA, USA, 2016, pp. 1023-1031, doi: 10.1109/ICUAS.2016.7502578.

[2] Vemuru, K. V., Harbour, S. D., & Clark, J. D. (2019). Reinforcement Learning in Aviation, Either Unmanned or Manned, with an Injection of AI. 20th International Symposium on Aviation Psychology, 492-497. https://corescholar.libraries.wright.edu/isap_2019/83

[3] Squad, *Kerbal Space Program* (Version 1.12.5). Private Division. [Computer Software]. Available: https://store.steampowered.com/app/220200/Kerbal_Space_Program/

[4] kRPC Organization, *kRPC* (Version 0.5.4) [Computer software]. Available: https://github.com/krpc/krpc/releases/tag/v0.5.4

## Implementation

This implementation is based off of the `../ddpg-bipedal` implementation. The impementation includes an agent, model and training script. The network is a DDPG (extension of Q-learning for continuous spaces) with a replay buffer. The neural network models inlcude an actor and a critic. The actor implements a policy to deterministically select the best actions from the states and is trained using the gradient from maximizing the estimated Q-value obtained from the critic (when the actor's best predicted action is fed as an input to the critic). The critic, on the other hand, implements the Q function and is trained as in Q-learning.

The actor is made up of 3 fully connected layers and 3 batch normalization layers with `relu` as an activation function and `tanh` for the output layer. The critic is made up of 3 fully connected layers and 1 batch normalization layer with `relu` as an activation function.

The hyperparameters that are used during training are shown below.

| Hyperparameter |                   Description                    | Value |
| -------------- | :----------------------------------------------: | ----: |
| BUFFER_SIZE    |                replay buffer size                |   1e6 |
| NUM_UPDATES    |        number of updates per n time steps        |    10 |
| TIME_STEPS     |         how often updates should happen          |    20 |
| BATCH_SIZE     |                  minibatch size                  |   256 |
| Gamma          |                 discount factor                  |  0.99 |
| Tau            | used in soft update of target network parameters |  1e-3 |
| LR_ACTOR       |            learning rate of the actor            |  1e-3 |
| LR_CRITIC      |           learning rate of the critic            |  1e-3 |
| WEIGHT_DECAY   |                 l2 weight decay                  |     0 |
| EPSILON        |            coefficient for the noise             |   1.0 |
| EPSILON_DECAY  |      decay coefficient for the noise factor      |  1e-5 |
| fc_units       |  number of nodes in the fully-connected layers   |   128 |

I have used the environment with 20 agents, but am using only one agent for simulation. There are 4 available actions and each agent observes a state vector of size 33. During training, the environment states for each agent are acquired from `env_info`, the scores for each agent are initialized and the agent is reset. Then, at each timestep actions are inferred using the `actor_local` network, and noise is added to each one. These actions are used in the environment and new states and rewards are returned from it. Then, the agent's `step` function is executed which samples from the memory buffer and learns from its experiences at every 10 time steps and whenever there are enough samples in the memory buffer.

The actor network learns through backpropagation with an Adam optimizer. The critic depends on the predicted next-state actions to calculate its own Q targets for the next time step. Then the current Q targets are calculated with

```
Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))
```

The local critic network is used to calculate the expected Q values and the mean squared error between the expected and the target Q values gives the critic loss. Then, the critic uses backpropagation (with clipping gradients) and the Adam optimizer to learn. Finally, soft updates are applied to both networks and the noise is reset to the mean.

## Results

A training graph of the successful agent is displayed below.

![Trained Agent](saved/training.png)

## Improvement Ideas

One potential improvement would be to use 20 agents in parallel which share a replay buffer and actor-critic networks. Additionally, using a network which is more robust to hyperparameter tuning would be a plus since DDPG is very sensitive to them. The D4PG algorithm as described in [this paper](https://openreview.net/pdf?id=SyZipzbCb) can be used to improve how the network works by introducing N-step returns and prioritised experience replay.

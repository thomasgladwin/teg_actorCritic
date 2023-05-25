# teg_actorCritic
Actor-Critic reinforcement learning model with a toy environment.

Practice project for Sutton & Barto. The model uses linear parameterizations for values and preferences. The code uses separate classes to make the separation of information between elements clear: the Environment (which responds to actions with state features and rewards), the Critic (which receives rewards and features from the Environment and bootstrap-learns state-value functions) and the Actor (which receives state features from the Environment and Temporal Difference signals from the critic and bootstrap-learns state-action preferences). There's also a Simulation class for convenience.

In principle, the classes should be pretty general-purpose. The Environment just has to receive actions and produce rewards to work with them, and it should contain all the case-specific machinery connecting actions to rewards.

In this case, the environment is a 2D GridWorld, with a starting point and a terminal point that must be reached in as few steps as possible, ending the episode. Starting and terminal points can be random per episode. There are vertical crosswinds that blow the agent off-course with varying speeds, walls which cannot be moved into, and pits with an added negative reward when moved into. Pits can be fixed or randomized. The Environment has options to determine which set of features are observable: the actual grid coordinates, the pits in neighbouring states, and the relative direction of the terminal point.

The two main cases are:

- A static world (fixed start and terminal points and fixed objects like pits), with the actual grid coordinates as the observable feature. This works with Lambda = 0.5.

- A varying world (randomized start and terminal points, both fixed objects and random pits), with the neighbouring pits and relative direction of the terminal point as observed features. This works with Lambda = 0.

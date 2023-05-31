# teg_actorCritic
Actor-Critic reinforcement learning model with a toy environment.

Practice project for Sutton & Barto. The model uses linear parameterizations for values and preferences. The code uses the classes: the Environment (which responds to actions with state features and rewards, as well as allowable actions), the Critic (which receives rewards and features from the Environment and bootstrap-learns state-value functions) and the Actor (which receives state features from the Environment and Temporal Difference signals from the critic and bootstrap-learns state-action preferences). The Critic and Actor are contained in an Agent class. There's also a Simulation class for training, testing, and plotting.

In principle, the classes should be pretty general-purpose. The Environment just has to receive actions and produce rewards to work with them, and it should contain all the case-specific machinery connecting actions to rewards.

In this case, the environment is a 2D GridWorld, with a starting point and a terminal point that must be reached in as few steps as possible, ending the episode. Starting and terminal points can be random per episode. There are vertical crosswinds that blow the agent off-course with varying speeds, walls which cannot be moved into, and pits with an added negative reward when moved into. Pits can be fixed or randomized. The Environment has options to determine which set of features are observable: the actual grid coordinates, the pits in neighbouring states, and the relative direction of the terminal point.

The two main cases are shown in Examples.py:

- A static world (fixed start and terminal points and fixed objects like pits), with the actual grid coordinates as the observable feature. The static world and its features can be defined via a multiline string, e.g.,

MapString = '''
4 0 0 0 0 0 0 0 0 0
0 0 0 0 0 4 2 0 0 0
0 0 0 0 0 4 4 4 4 0
0 0 0 0 0 0 0 0 0 0
3 3 3 3 0 3 3 0 0 0
0 1 0 0 0 0 0 0 0 0
'''

- A varying world (randomized start and terminal points and randomized pits), with the neighbouring pits and relative direction of the terminal point as observed features.

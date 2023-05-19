# teg_actorCritic
Actor-critic "GridWorld" toy model for reinforcement learning. Additionally, a SARSA(lambda) model and expansion of GridWorld with (random) negative-reward pits to avoid.

Practice project for Sutton & Barto. The model has the most basic state-indicator binary parameterization for the actor's and critic's weights. The model uses continuous learning on reset-on-terminal-state episodes.

The code uses three separate classes to make the separation of information between elements clear: the Environment (which responds to actions), the Critic (which receives rewards from the Environment and bootstrap-learns state-value functions) and the Actor (which receives Temporal Difference signals from the critic and bootstrap-learns state-action preferences). There's also a Simulation class for convenience.

In principle, the Actor, Critic and Simulation classes should be pretty general-purpose. The Environment just has to receive actions and produce rewards to work with them, and it contains all the specific machinery connecting actions to rewards. In this case, the GridWorld environment has a starting point and a terminal point that must be reached in as few steps as possible, ending the episode. There are vertical crosswinds that blow the agent off-course with varying speeds and pits that reduce reward. The Environment has options to determine what features are observable: the actual grid coordinates, the neighbouring pits, and the relative direction of the terminal point.

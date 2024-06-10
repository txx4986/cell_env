# cell_env
The code implemented in this project is adapted from [1], featuring three main classes: Mind, Agent and Environment.

The Mind class represents the decision-making process of the agents, employing deep Q networks (DQN) as decision functions through a deep reinforcement learning-based approach. Unlike the original implementation, where two distinct DQNs were trained for different agent types, our adaptation utilises a single DQN due to the presence of only one agent type.

The Agent class represents the agent within the agent-based model. Communication between the Environment and Agent classes facilitates the execution of simulations. Notably, we modified the input of the DQN to utilise local concentration instead of agent ages, as per the original implementation.

The Environment class manages a grid world where simulations take place, overseeing agent movement and removal. Our modifications include transforming agent actions from directional movements to choices between dividing, remaining inactive, or dying. Additionally, we introduced random agent movement before action selection.

During a simulation, the Environment initialises a grid world with a specified number of agents based on the initial global concentration. Each agent communicates with the same mind to select actions, with subsequent updates to the grid world reflecting these actions.

## Bibliography
[1] Sert, Egemen, et al. ‘Segregation Dynamics with Reinforcement Learning and Agent Based Modeling’. Scientific Reports, vol. 10, no. 1, Springer Science and Business Media LLC, July 2020, https://doi.org10.1038/s41598-020-68447-8.

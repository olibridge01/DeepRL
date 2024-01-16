![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

# DeepRL - Deep Reinforcement Learning in Python
This repository contains implementations of deep reinforcement learning algorithms on various environments.

## Directory Structure
```
DeepRL/
├── agents/
│   └── (RL agents)
├── figs/
│   ├── gifs/
│   │   └── (gifs of agent behaviour)
│   └── plots/
│       └── (plots e.g. scores/episode durations)    
├── misc/
│   └── (miscellaneous resources e.g. fonts)
├── networks/
│   └── (neural networks used for deep RL algorithms)
├── saves/
│   └── (saved weights for trained agent networks)
├── utils/
│   └── (utility functions e.g. plotters, agent config etc.)
├── run.py (script to train agents/plot gifs etc.)
└── README.md
```

## Examples
Below are some examples of RL algorithms implemented on OpenAI's `gymnasium` environments.
- *Random agent*: Agent selects moves randomly from the action space. This is not a very useful agent but is used to show how the agent behaves in the following environments by acting at random.
- *DQN agent*: Agent learns with an epsilon-greedy policy via a feedforward neural network with an Experience Replay buffer - [Mnih *et al*, 2013](https://arxiv.org/pdf/1312.5602.pdf).
- *DDQN agent*: Agent learns with Deep Double Q-learning - [van Hasselt *et al*, 2015](https://arxiv.org/pdf/1509.06461.pdf).

## Direct Observations
In the following examples, the agent receives physical properties of the system as the current state e.g. player position/velocity etc.

### LunarLander-v2
The aim is to land the Lunar Lander between the flags. The agent is rewarded based on how close the lander is to the pad, how fast it is moving, and how little it has to use its engines. It is also given a reward of -100(+100) for crashing (landing safely) on the moon. An episode terminates if the lander crashes, leaves the frame or comes to rest on the ground. An episode is a **success** if the agent's cumulative reward is over 200.
<p float="left">
  <img src="https://github.com/olibridge01/DeepRL/assets/86416298/e8815da6-4c45-4e2d-b0f2-c1af56fe03ce" width="49.5%" />
  <img src="https://github.com/olibridge01/DeepRL/assets/86416298/27a94222-4e5b-4701-9e88-c669ac3c84fb" width="49.5%" />
</p>

### CartPole-v1
The aim is to keep the pole balanced by only moving the cart left or right. The agent is given a reward of `+1` for every timestep it keeps the pole upright. An episode ends if the pole deviates by more than ±12° from the vertical, or the cart leaves the frame. An episode is a **success** if it keeps the pole upright for 500 timesteps.
<p float="left">
  <img src="https://github.com/olibridge01/DeepRL/assets/86416298/07f63b9b-c747-4c4c-bc3b-b6b1bf1dce3c" width="49.5%" />
  <img src="https://github.com/olibridge01/DeepRL/assets/86416298/f4960390-2922-4575-b710-705ed60719ef" width="49.5%" />
</p>

### MountainCar-v0
The aim is to accelerate the car up the hill to the flag. The agent is given a reward of $-t$ where $t$ is the number of timesteps needed to reach the flag ($t$ is capped at 200). An episode is a **success** if the car reaches the flag in under 200 timesteps.
<p float="left">
  <img src="https://github.com/olibridge01/DeepRL/assets/86416298/692b03a1-6cb7-4e81-a263-8c66104b7e5f" width="49.5%" />
  <img src="https://github.com/olibridge01/DeepRL/assets/86416298/a1978627-a818-4a08-bca1-1896195a5fcd" width="49.5%" />
</p>

## Screen Observations
Here, rather than receiving physical properties of the current state (e.g. position, velocity etc.), the agent receives only a screen image (a 2d array of coloured pixels) as the current state.
### CarRacing-v2
The aim is to achieve the highest score possible by racing a car around a racetrack. The car accumulates a reward of -0.1 per frame, and a reward of +1000/N for every racetrack tile visited, where N is the number of tiles on the whole track.
<p float="left">
  <img src="https://github.com/olibridge01/DeepRL/assets/86416298/d3f9f573-5cc8-46f1-a969-05dd8d45a4a4" width="49.5%" />
  <img src="https://github.com/olibridge01/DeepRL/assets/86416298/4f407768-0155-4357-ad06-d4367c4c47ab" width="49.5%" />
</p>

A plot of the scores as a function of training episodes is shown below. The score clearly improves when the agent learns how to drive the car round the track effectively.

<p align="center">
  <img src="https://github.com/olibridge01/DeepRL/assets/86416298/8ad3f9f7-3bb5-4d27-a409-cf45f1f80349" width="70%" />
</p>

---
**Oli Bridge** (<olibridge@rocketmail.com>)

Feel free to explore the different directories, and please reach out if you have any questions or suggestions. 🚀

<!-- Add plots for direct obs environments - DQN vs DDQN (vs QLearning?) -->
<!-- Add NN architecture diagrams for direct/screen observations -->

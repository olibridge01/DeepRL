![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

# DeepRL - Deep Reinforcement Learning in Python
This repository contains implementations of various deep reinforcement learning algorithms.

## Directory Structure
```
DeepRL/
├── agents/
│   └── (RL agents)
├── networks/
│   └── (neural networks used for deep RL algorithms)
├── utils/
│   └── (utility functions e.g. plotters, agent config etc.)
├── misc/
│   └── (miscellaneous resources e.g. fonts)
├── run.py (script to train agents/plot gifs etc.)
└── README.md

```

## Examples
Below are some examples of RL algorithms implemented on OpenAI's `gymnasium` environments.
- *Random agent*: Agent selects moves randomly from the action space. This is not a very useful agent but is used to show how the agent behaves in the following environments by acting at random.
- *DQN agent*: Agent learns with an epsilon-greedy policy via a feedforward neural network with an Experience Replay buffer - [Mnih *et al*, 2013](https://arxiv.org/pdf/1312.5602.pdf).

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

---
**Oli Bridge** (<olibridge@rocketmail.com>)

Feel free to explore the different directories, and please reach out if you have any questions or suggestions. 🚀

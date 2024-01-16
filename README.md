![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

# DeepRL - Deep Reinforcement Learning in Python
This repository contains implementations of deep reinforcement learning algorithms on various environments.

## Installation
To install the required packages, run the following commands in the terminal:
```
git clone https://github.com/olibridge01/DeepRL.git
cd DeepRL
pip install -r requirements.txt
```

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
├── README.md
├── requirements.txt
└── run.py (script to train agents/plot gifs etc.)
```

## Reinforcement Learning Algorithms
Below are some examples of the RL algorithms implemented.
- *Random agent*: Agent selects moves randomly from the action space. This is not a very useful agent but is used to show how the agent behaves in the following environments by acting at random.
- *Q-learning agent*: Agent learns with an epsilon-greedy policy via a table of action value function values - [Watkins, 1989](http://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf).
- *DQN agent*: Agent learns with an epsilon-greedy policy via a feedforward neural network with an Experience Replay buffer - [Mnih *et al*, 2013](https://arxiv.org/pdf/1312.5602.pdf).
- *DDQN agent*: Agent learns with Deep Double Q-learning - [van Hasselt *et al*, 2015](https://arxiv.org/pdf/1509.06461.pdf).

Below is a brief overview of the background theory for the algorithms used.

### Background Theory - Action value functions, Q-Learning and DQN
The action value function $Q^\pi: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ for a given policy $\pi$ is defined as the expected return for taking action $a$ in state $s$ and following policy $\pi$ thereafter:

$$
Q^\pi(s,a) = \mathbb{E} \left[ G_t | S_t = s, A_t = a , \pi \right]
$$

where the return $G_t$ is the sum of discounted rewards $R$ from time $t$:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^\infty \gamma^k R_{t+k+1}
$$

The action value function satisfies the Bellman equation:

$$
Q^\pi(s,a) = \mathbb{E} \left[ R_{t+1} + \gamma Q^\pi(S_{t+1},A_{t+1}) | S_t = s, A_t = a, \pi\right]
$$

The optimal action value function $Q^*: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ is the maximum action value function over all possible policies; it satisfies the Bellman optimality equation:

$$
Q^* (s,a) = \mathbb{E} \left[ R_{t+1} + \gamma \max_{a'} Q^* (S_{t+1},a') | S_t = s, A_t = a \right]
$$

For a *deterministic environment* - where each state/action pair $(s,a)$ maps to exactly one new state $s'$ - the optimal action value function can be rewritten as:

$$
Q^* (s,a) = r(s,a) + \gamma \max_{a'} Q^* (s',a')
$$

where $r(s,a)$ is the reward for taking action $a$ in state $s$. 

In **Q-Learning**, the agent learns an estimate of the optimal action value function $Q\approx Q^*$ by performing an update at each timestep:

$$
Q(s_t,a_t) \gets Q(a_t,a_t) + \alpha \left[ r(s_t,a_t) + \gamma \max_{a'} Q(s_{t+1},a') - Q(s_t,a_t) \right]
$$

where $\alpha$ is the learning rate. This method typically makes use of a table to store the action value function values. This requires a continuous state space to be discretised, and is therefore only suitable for small state/action spaces $\mathcal{S}\times\mathcal{A}$.

In **Deep Q-Learning (DQN)**, the agent learns an estimate of the optimal action value function $Q\approx Q^*$ using a neural network. In this project, the `torch.nn.SmoothL1Loss()` (Huber) loss function was used to train the network. 

In both the cases above, performing these updates alone is known as a *greedy policy* - the agent chooses the action which maximises the action value function at each step. This typically explores the state/action space poorly, so instead an *epsilon-greedy policy* is used - the agent chooses a random action with probability $\epsilon \in [0,1]$ and the greedy action with probability $1-\epsilon$. The value of $\epsilon$ is decayed over time to encourage the agent to sufficiently explore the state/action space in earlier episodes, then act using the knowledge it has gained in later episodes.

## Results
The RL algorithms described above were implemented on a series of [OpenAI's `gymnasium` environments](https://gymnasium.farama.org/index.html) (see below). All results are shown alongside a random agent for comparison.

### Direct Observations
In the following examples, the agent receives *physical properties* of the system as the current state e.g. player position/velocity etc. This typically makes it simpler for the agent to learn an optimal policy.

For the deep RL algorithms, a **multi-layer perceptron (MLP)** neural network was used to approximate the action value function $Q: \mathcal{S}\times\mathcal{A}\to\mathbb{R}$. The network architecture used for the `Cartpole-v1` environment is shown in the corresponding section below.

#### CartPole-v1
The aim is to keep the pole balanced by only moving the cart left or right. The agent is given a reward of `+1` for every timestep it keeps the pole upright. An episode ends if the pole deviates by more than `±12°` from the vertical, or the cart leaves the frame. An episode is a **success** if it keeps the pole upright for `500` timesteps.

The agent receives the following observations:
- `p`: The position of the cart on the track.
- `v`: The velocity of the cart.
- `a`: The angle of the pole from the vertical.
- `av`: The pole's angular velocity.

The neural network takes these observations as a 4-dimensional input, and outputs the action value function $Q(s,a')$ for each action $a'\in\mathcal{A}$ (in this case $\mathcal{A} = \{0,1\}$). The DQN/DDQN algorithms also make use of an *experience replay* buffer which is described in [Mnih *et al*, 2013](https://arxiv.org/pdf/1312.5602.pdf).

<img align="right" src="https://github.com/olibridge01/DeepRL/assets/86416298/3f51fcc7-9010-4f5a-9787-52daea2c4118" width="39.5%" />

<p float="left">
  <img src="https://github.com/olibridge01/DeepRL/assets/86416298/07f63b9b-c747-4c4c-bc3b-b6b1bf1dce3c" width="56%" />
  <img src="https://github.com/olibridge01/DeepRL/assets/86416298/f4960390-2922-4575-b710-705ed60719ef" width="56%" />
</p>

A plot of the scores as a function of training episodes is shown below for DQN, DDQN and Q-learning. The agents were each run `6` times and the mean rolling score over the last `10` episodes is shown.

<p align="center">
  <img src="https://github.com/olibridge01/DeepRL/assets/86416298/5e64dee0-5c9d-4044-ad32-71f8a83b6e65" width="70%" />
</p>

In general DQN and DDQN were able to solve `Cartpole-v1` in slightly fewer episodes than Q-Learning. Q-Learning performs reasonably well here due to the relatively small state/action space, but will struggle to scale to larger environments.

#### LunarLander-v2
The aim is to land the Lunar Lander between the flags. The agent is rewarded based on how close the lander is to the pad, how fast it is moving, and how little it has to use its engines. It is also given a reward of `-100`(`+100`) for crashing (landing safely) on the moon. An episode terminates if the lander crashes, leaves the frame or comes to rest on the ground. An episode is a **success** if the agent's cumulative reward is over `200`. Gifs of the agent's behaviour are shown below. 
<p float="left">
  <img src="https://github.com/olibridge01/DeepRL/assets/86416298/e8815da6-4c45-4e2d-b0f2-c1af56fe03ce" width="49.5%" />
  <img src="https://github.com/olibridge01/DeepRL/assets/86416298/27a94222-4e5b-4701-9e88-c669ac3c84fb" width="49.5%" />
</p>

#### MountainCar-v0
The aim is to accelerate the car up the hill to the flag. The agent is given a reward of $-t$ where $t$ is the number of timesteps needed to reach the flag ($t$ is capped at `200`). An episode is a **success** if the car reaches the flag in under `200` timesteps. Gifs of the agent's behaviour are shown below.
<p float="left">
  <img src="https://github.com/olibridge01/DeepRL/assets/86416298/692b03a1-6cb7-4e81-a263-8c66104b7e5f" width="49.5%" />
  <img src="https://github.com/olibridge01/DeepRL/assets/86416298/a1978627-a818-4a08-bca1-1896195a5fcd" width="49.5%" />
</p>

---

### Screen Observations
Here, rather than receiving physical properties of the current state (e.g. position, velocity etc.), the agent receives only a *screen image* (a 2D array of coloured pixels) as the current state. In order to handle this, a few modifications were made to the DQN/DDQN algorithms implemented above. These are listed below, and are described in more detail in [Mnih *et al*, 2013](https://arxiv.org/pdf/1312.5602.pdf) where the agent was trained to play *Atari* games.

- The action value function was approximated using a **convolutional neural network (CNN)** (with some extra fully-connected layers on the end) rather than an MLP. This allows the agent to learn features from the screen image (a diagram of the architecture is shown in the following section).
- The screen image was preprocessed before being passed into the network. The images were converted to a single channel (grayscale) and cropped to `84x84`. Each observation was stacked with the previous `3` frames to give an input tensor of shape `(4,84,84)`.
- At each step, `4` frames were skipped and the action was repeated for each of these frames.

#### CarRacing-v2
The aim is to achieve the highest score possible by racing a car around a racetrack. The car accumulates a reward of `-0.1` per frame, and a reward of `+1000/N` for every racetrack tile visited, where `N` is the number of tiles on the whole track.

<img align="right" src="https://github.com/olibridge01/DeepRL/assets/86416298/692103b6-7200-4a33-8ad7-fcb9f245bf08" width="36%" />


<p float="left">
  <img src="https://github.com/olibridge01/DeepRL/assets/86416298/d3f9f573-5cc8-46f1-a969-05dd8d45a4a4" width="59%" />
  <img src="https://github.com/olibridge01/DeepRL/assets/86416298/4f407768-0155-4357-ad06-d4367c4c47ab" width="59%" />
</p>

A plot of the rolling scores (`100`-episode window) as a function of training episodes is shown below. The score clearly improves when the agent learns how to drive the car round the track effectively.

<p align="center">
  <img src="https://github.com/olibridge01/DeepRL/assets/86416298/8ad3f9f7-3bb5-4d27-a409-cf45f1f80349" width="70%" />
</p>


---
**Oli Bridge** (<olibridge@rocketmail.com>)

Feel free to explore the different directories, and please reach out if you have any questions or suggestions. 🚀

<!-- Add plots for direct obs environments - DQN vs DDQN (vs QLearning?) -->
<!-- Add NN architecture diagrams for direct/screen observations -->

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt

def preprocess_obs(obs: np.ndarray) -> np.ndarray:
    """Convert state observation to grayscale and crop to 84x84."""
    obs_grayscale = obs.mean(-1) # Convert to grayscale
    obs_cropped = obs_grayscale[:84, 6:-6] # Crop to 84x84
    return obs_cropped

class CarRacingEnv(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        frame_stack: int = 4,
        frame_skip: int = 4,
        burn_in: int = 0,
        **kwargs
        ):
        super(CarRacingEnv, self).__init__(env, **kwargs)
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        self.burn_in = burn_in

    def reset(self) -> tuple:
        """Resets the environment and returns the initial observation."""
        obs, info = self.env.reset()

        # Burn in
        for _ in range(self.burn_in):
            obs, _, _, _, _ = self.env.step(0) # Do nothing for burn_in steps

        # Initial observation is repeated frame_stack times
        self.stacked_obs = np.stack([preprocess_obs(obs)] * self.frame_stack, axis=0)

        return self.stacked_obs, info
    
    def step(self, action: int) -> tuple:
        """Performs action for frame_skip steps and returns the stacked observations."""
        total_reward = 0
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
            if done:
                break

        # Update stacked observations
        self.stacked_obs = np.concatenate((self.stacked_obs[1:], [preprocess_obs(obs)]), axis=0)
        return self.stacked_obs, total_reward, terminated, truncated, info


if __name__ == '__main__':  

    # Test the preprocessing on CarRacing-v2 environment
    env = gym.make('CarRacing-v2', continuous=False)
    env = CarRacingEnv(env)

    s, _ = env.reset()
    print("The shape of an observation: ", s.shape)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i in range(4):
        axes[i].imshow(s[i], cmap='gray')
        axes[i].axis('off')
    plt.show()

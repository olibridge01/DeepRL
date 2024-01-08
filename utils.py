import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, namedtuple

class Config(object):
    """Holds parameters for training."""
    def __init__(self):
        self.environment = None
        self.hyperparameters = None

class ReplayBuffer(object):
    def __init__(self, buffer_size: int) -> None:
        self.memory = deque([], maxlen=buffer_size)
        self.transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
    
    def add_transition(self, *args):
        """Saves a transition"""
        self.memory.append(self.transition(*args))
    
    def sample(self, batch_size: int, separate_types: bool = True) -> tuple:
        """Returns a random sample of transitions"""
        samples = random.sample(self.memory, k=batch_size)
        if separate_types:
            return self.transition(*zip(*samples))
        else:
            return samples
    
    def __len__(self) -> int:
        """Returns the length of the memory"""
        return len(self.memory)

def plot_durations(
        episode_durations: list,
        figsize: tuple = (10, 6),
        moving_average_window: int = 100
    ) -> None:
    """Plots episode durations over time."""
    plt.figure(figsize=figsize)

    # Plot raw episode durations
    plt.plot(episode_durations, 
             color='b', 
             linewidth=1.25, 
             alpha=0.75
    )

    # Compute and plot moving average
    moving_average = []

    for i in range(len(episode_durations) - moving_average_window):
        moving_average.append(np.mean(episode_durations[i:i+moving_average_window]))

    plt.plot(
        moving_average, 
        color='r', 
        linewidth=2, 
    )
    plt.xlabel('Episode')
    plt.ylabel('Episode duration')
    plt.ylim(bottom=0)
    plt.show()
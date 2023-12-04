import numpy as np
import matplotlib.pyplot as plt

class Config(object):
    """Holds parameters for training."""
    def __init__(self):
        self.environment = None
        self.hyperparameters = None

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
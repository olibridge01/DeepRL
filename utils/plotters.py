import random
import numpy as np
import matplotlib.pyplot as plt

import imageio
from PIL import Image
import PIL.ImageDraw as ImageDraw   
import PIL.ImageFont as ImageFont
import gym
import torch
from utils.preprocessing import preprocess_obs

def plot_durations(
        episode_durations: list,
        figsize: tuple = (10, 6),
        moving_average_window: int = 100,
        filename: str = None
    ) -> None:
    """
    Plots episode durations over time.
    """

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
    
    moving_avg_t = np.arange(moving_average_window, len(episode_durations))

    plt.plot(
        moving_avg_t,
        moving_average, 
        color='r', 
        linewidth=2, 
    )
    plt.xlabel('Episode')
    plt.ylabel('Episode duration')
    plt.ylim(bottom=0)
    if filename is not None:
        plt.savefig(f'{filename}.pdf', dpi=600, bbox_inches='tight')
    plt.show()

def plot_scores(
        scores: list,
        figsize: tuple = (10, 6),
        moving_average_window: int = 100,
        filename: str = None
    ) -> None:
    """
    Plots scores over time.
    """
    # Set Latex fonts
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Define figure
    plt.figure(figsize=figsize)

    # Plot raw scores
    plt.plot(scores, 
             color='b', 
             linewidth=1.25, 
             alpha=0.75,
             label='Episode score'
    )

    # Compute and plot moving average
    moving_average = []

    for i in range(len(scores) - moving_average_window):
        moving_average.append(np.mean(scores[i:i+moving_average_window]))
    
    moving_avg_t = np.arange(moving_average_window, len(scores))

    plt.plot(
        moving_avg_t,
        moving_average, 
        color='r', 
        linewidth=2,
        label=f'Average score ({moving_average_window} episodes)' 
    )
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.xlim(0, len(scores))
    plt.legend(frameon=False)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(f'{filename}.png', dpi=600, bbox_inches='tight')
    plt.show()


def label_with_timestep(frame: np.ndarray, 
                        agentname: str, 
                        timestep: int
    ) -> Image:
    """
    Draws image with timestep number in top left corner.
    """

    # Draw image
    im = Image.fromarray(frame)
    drawer = ImageDraw.Draw(im)
    drawer.fontmode = '1' # Anti-aliasing

    # Determine text color based on background of image
    if np.mean(im) < 128:
        text_color = (255,255,255)
    else:
        text_color = (0,0,0)

    # Add frame label
    font = ImageFont.truetype('misc/mc-font/MCRegular.otf', 10)
    drawer.text((im.size[0]/20,im.size[1]/18), f' Agent: {agentname}', fill=text_color, font=font)
    drawer.text((im.size[0]/20, 1.5*im.size[1]/18), f' Time step: {timestep+1}', fill=text_color, font=font)

    return im


def save_trained_agent_gif(Agent: object,
                           agentname: str,
                           filename: str,
                           n_time_steps: int = 1000,
    ) -> None:
    """
    Saves a gif of the trained agent which acts using the policy network.
    """

    frames = []
    state, info = Agent.env.reset()
    state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
    for t in range(n_time_steps):
        action =  Agent.get_action(state)
        
        frame = Agent.env.render()
        frames.append(label_with_timestep(frame, agentname, timestep=t))

        obs, _, done, _, _ = Agent.env.step(action.item())
        state = torch.tensor(obs, dtype=torch.float).unsqueeze(0)

        if done:
            break

    Agent.env.close()
    imageio.mimwrite(f'{filename}.gif', frames, fps=20, loop=0)
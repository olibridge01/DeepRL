
import gymnasium as gym

from utils.config import Config

class RandomAgent(object):
    """Agent that acts randomly."""
    def __init__(self, config: Config) -> None:
        self.env = config.environment

    def get_action(self, state: int) -> int:
        """Randomly sample from action space."""
        return self.env.action_space.sample()
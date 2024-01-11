import random
from collections import deque, namedtuple

class ReplayBuffer(object):
    """Experience replay from Mnih et al. (2013)"""
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
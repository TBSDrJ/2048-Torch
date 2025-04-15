"""
Author: TBSDrJ
Date: Spring 2025
Purpose: Trying to train a model to play 2048 better than I can.
Reference:
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""
from collections import namedtuple, deque
import random

import torch

from Env2048 import Env2048

Transition = namedtuple('Transition', 
        ('state', 'move', 'next_state', 'reward'))

torch.set_default_device("mps")
torch.manual_seed(2048) # What else?

WIDTH = 4
HEIGHT = 4
PROB_4 = 0.1
env = Env2048(WIDTH, HEIGHT, PROB_4)

BATCH_SIZE = 16

class ReplayMemory:
    def __init__(self, max_length: int):
        self.memory = deque(maxlen=max_length)
    
    def append(self, state: torch.tensor, move: torch.tensor, 
            next_state: torch.tensor, reward: torch.tensor) -> None:
        """Append a transition to memory.
        
        Expects:
        state and next_state shapes to be [m, n] where m x n is the board size.
        move and reward shapes to be [1]."""
        self.memory.append(Transition(state, move, next_state, reward))

    def random_sample(self, batch_size: int = BATCH_SIZE) -> list[Transition]:
        """Get a random sample of moves from this memory."""
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)

    def __getitem__(self: int) -> Transition:
        return self.memory[i]


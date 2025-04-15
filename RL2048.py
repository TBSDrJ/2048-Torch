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
import numpy as np

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

class DQN(torch.nn.Module):
    def __init__(self, env: Env2048):
        super().__init__()
        len_input = env.game.width * env.game.height
        len_output = int(env.action_space.shape[0])
        self.linear_0 = torch.nn.Linear(len_input, len_input)
        self.linear_1 = torch.nn.Linear(len_input, len_input)
        self.linear_2 = torch.nn.Linear(len_input, len_output)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x: "torch.tensor | np.ndarray") -> torch.tensor:
        # First, make sure input is 2-D tensor, with dim_0 = batch dim.
        if isinstance(x, np.ndarray):
            x = torch.tensor(x)
        if len(x.shape) > 2:
            x.reshape((BATCH_SIZE, -1))
        y = self.linear_0(x)
        y = self.relu(y)
        y = self.linear_1(y)
        y = self.relu(y)
        y = self.linear_2(y)
        return y

"""
Author: TBSDrJ
Date: Spring 2025
Purpose: Trying to train a model to play 2048 better than I can.
Reference:
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""
from collections import namedtuple, deque
import random
import math

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
BATCH_SIZE = 1
EPISODES = 2

class ReplayMemory:
    def __init__(self, max_length: int):
        self.memory = deque(maxlen=max_length)
    
    def append(self, state: torch.tensor, move: torch.tensor, 
            next_state: torch.tensor, reward: torch.tensor) -> None:
        """Append a transition to memory.
        
        Expects:
        state and next_state shapes to be [WIDTH, HEIGHT] and
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
        len_output = torch.numel(env.action_space)
        self.linear_0 = torch.nn.Linear(len_input, len_input)
        self.linear_1 = torch.nn.Linear(len_input, len_input)
        self.linear_2 = torch.nn.Linear(len_input, len_output)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x: "torch.tensor | np.ndarray") -> torch.tensor:
        # First, make sure input is 2-D tensor, with dim_0 = batch dim.
        if isinstance(x, np.ndarray):
            x = torch.tensor(x)
        if len(x.shape) > 2:
            x = x.reshape((BATCH_SIZE, -1))
        x = x.to(torch.float32)
        y = self.linear_0(x)
        y = self.relu(y)
        y = self.linear_1(y)
        y = self.relu(y)
        y = self.linear_2(y)
        return y

def select_move(env: Env2048, policy_net: DQN, state: torch.tensor, steps: int
        ) -> torch.int32:
    """Pick a move. Random more often early, use policy_net more later.
    
    Use an exponentially decaying threshold for the decision of which. """
    prob = random.random()
    start = 0.9
    end = 0.05
    decay_steps = 1000
    threshold = end + (start - end) * math.exp(-1. * steps / decay_steps)
    if prob > threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.reshape(1, 1)
    else:
        # numel just returns num of elements
        n = torch.randint(0, torch.numel(env.action_space), [1, 1])
        return env.action_space[n]

def optimize_model(
        policy_net: DQN, 
        target_net: DQN, 
        optimizer: torch.optim.Optimizer, 
        lr_sch: torch.optim.lr_scheduler.LRScheduler,
        memory: ReplayMemory,
    ):
    """"""

def main():
    env = Env2048(WIDTH, HEIGHT, PROB_4)
    policy_net = DQN(env)
    target_net = DQN(env)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=0.001, 
            amsgrad=True)
    lr_sch = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, 
            lambda epoch: 0.995)
    memory = ReplayMemory(10000)
    episode_scores = []
    steps = 0
    for i in range(EPISODES):
        state = env.reset()
        game_over = False
        while not game_over:
            move = select_move(env, policy_net, state, steps)
            next_state, reward, game_over = env.step(move)
            steps += 1
            memory.append(state, move, next_state, reward)
            state = next_state
            optimize_model(policy_net, target_net, optimizer, lr_sch, memory)

if __name__ == "__main__":
    main()
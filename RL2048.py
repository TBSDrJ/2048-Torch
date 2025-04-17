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
        ('state', 'move', 'next_state', 'reward', 'game_over'))

torch.set_default_device("mps")
torch.manual_seed(2048) # What else?

WIDTH = 3
HEIGHT = 3
PROB_4 = 0.1
BATCH_SIZE = 128
EPISODES = 200
BUFFER = 500

class ReplayMemory:
    def __init__(self, max_length: int):
        self.memory = deque(maxlen=max_length)
    
    def append(self, state: torch.Tensor, move: torch.Tensor, 
            next_state: torch.Tensor, reward: torch.Tensor, 
            game_over: bool) -> None:
        """Append a transition to memory.
        
        Expects:
        state and next_state shapes to be [WIDTH, HEIGHT] and
        move and reward shapes to be [1]."""
        self.memory.append(
                Transition(state, move, next_state, reward, game_over))

    def random_sample(self, batch_size: int = BATCH_SIZE) -> list[Transition]:
        """Get a random sample of moves from this memory."""
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)

    def __getitem__(self, i: int) -> Transition:
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
    
    def forward(self, x: "torch.Tensor | np.ndarray") -> torch.Tensor:
        x = x.to(torch.float32)
        y = self.linear_0(x)
        y = self.relu(y)
        y = self.linear_1(y)
        y = self.relu(y)
        y = self.linear_2(y)
        return y

def select_move(env: Env2048, policy_net: DQN, state: torch.Tensor, steps: int
        ) -> torch.int32:
    """Pick a move. Random more often early, use policy_net more later.
    
    Use an exponentially decaying threshold for the decision of which. """
    prob = random.random()
    start = 0.9
    end = 0.05
    decay_steps = 5000
    threshold = end + (start - end) * math.exp(-1. * steps / decay_steps)
    if (steps + 1) % 500 == 0: 
        print(" "*53 + f"{threshold:.4f}", end="\r")
    if prob > threshold:
        with torch.no_grad():
            state = state.flatten()
            state = state.reshape(1, -1)
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
    """Optimization step."""
    if len(memory) < BUFFER:
        return
    discount = 0.98
    transitions = memory.random_sample(BATCH_SIZE - 1)
    # I decided to make sure most recent move is included in the batch.
    transitions.append(memory[-1])
    # Change from batch of Transitions to a Transition of batches
    batch = Transition(*zip(*transitions))
    batch_state = torch.cat(batch.state).reshape((BATCH_SIZE, -1))
    batch_move = torch.cat(batch.move).to(torch.int64)
    batch_next_state = torch.cat(batch.next_state).reshape((BATCH_SIZE, -1))
    batch_reward = torch.cat(batch.reward)
    batch_game_not_over = torch.cat(batch.game_over).logical_not()
    pred_q_for_actual_move = policy_net(batch_state).gather(1, batch_move)
    target_q_future_max = torch.zeros((BATCH_SIZE))
    with torch.no_grad():
        target_q_future_max = target_net(batch_next_state).max(1).values
    target_q_future_max * batch_game_not_over
    target_q = discount * target_q_future_max + batch_reward
    target_q = target_q.reshape((BATCH_SIZE, 1))
    loss = torch.nn.HuberLoss()(pred_q_for_actual_move, target_q)
    policy_state_dict = policy_net.state_dict()
    target_state_dict = target_net.state_dict()
    for key in target_net.state_dict():
        target_state_dict[key] = (policy_state_dict[key] * 0.01 + 
                target_state_dict[key] * .99)
    target_net.load_state_dict(target_state_dict)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    lr_sch.step()

def main():
    env = Env2048(WIDTH, HEIGHT, PROB_4)
    policy_net = DQN(env)
    target_net = DQN(env)
    policy_net.train()
    target_net.train()
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=0.0005, 
            amsgrad=True)
    lr_sch = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, 
            lambda epoch: 1)
    memory = ReplayMemory(10000)
    episode_scores = []
    steps = 0
    prev_steps = 0
    buffer_reached = False
    print(f"Buffering {BUFFER} steps before starting training...")
    print("Ep   Steps   Score   Avg.Score   Last 10     LR     % rand moves")
    for i in range(EPISODES):
        state = env.reset()
        game_over = False
        while not game_over:
            move = select_move(env, policy_net, state, steps)
            next_state, reward, game_over = env.step(move)
            steps += 1
            game_over_tensor = torch.tensor([game_over], dtype=torch.bool)
            memory.append(state, move, next_state, reward, game_over_tensor)
            state = next_state
            optimize_model(policy_net, target_net, optimizer, lr_sch, memory)
            if buffer_reached:
                print(f"{i:^3}  {steps - prev_steps:^5}   ", end="")
                print(f"{env.game.score:^5}   ", end="\r")
        if buffer_reached:
            episode_scores.append((steps - prev_steps, env.game.score))
            all_scores = [e[1] for e in episode_scores]
            print(f"{i:^3}  {steps - prev_steps:^5}   ", end="")
            print(f"{env.game.score:^5}   ", end="")
            print(f"{sum(all_scores) / len(all_scores):.2f}", end="")
            if len(episode_scores) >= 10:
                ten_scores = [e[1] for e in episode_scores[-10:]]
                print(f"   {sum(ten_scores) / len(ten_scores):.2f}", end="")
            else:
                print(" " * 9, end="")
            print(f"   {lr_sch.get_last_lr()[0]:.6f}")
        if steps > BUFFER:
            buffer_reached = True
        prev_steps = steps

if __name__ == "__main__":
    main()
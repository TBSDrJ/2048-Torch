import random
import math

import torch

from Game2048 import Game2048

class Env2048:
    """Create an environment to use for RL in Torch."""
    def __init__(self, width:int=4, height:int=4, prob_4:float=0.1):
        self.width = width
        self.height = height
        self.prob_4 = prob_4
        self.reset()

    def reset(self):
        """Reset the environment with a new game."""
        self.game = Game2048(self.width, self.height, self.prob_4)
        return self.state

    @property
    def state(self):
        """Read-only attribute for the current state of the game."""
        state = self.game.board
        state = torch.tensor([state], dtype=torch.int32)
        return state

    def step(self, move: torch.tensor) -> (torch.tensor, torch.tensor, bool):
        """Receive an action, return the state, reward, game_over."""
        score_before = self.game.score
        move = move.detach().clone().to("cpu")
        move = int(move.numpy()[0])
        # Not using board_changed yet, could penalize moves that don't.
        board_changed = self.game.one_turn(move)
        turn_score = self.game.score - score_before
        if turn_score == 0:
            turn_score += 1
        reward = turn_score.bit_length()
        reward = torch.tensor([reward], dtype=torch.int32)
        return self.state, reward, self.game.game_over

    @property
    def action_space(self) -> torch.tensor:
        """Return a list of all possible actions."""
        return torch.tensor([0, 1, 2, 3], dtype=torch.int32)


if __name__ == "__main__":
    env = Env2048()
    print(env.state)
    game_over = False
    while not game_over:
        r = random.randrange(4)
        print(r)
        move = torch.tensor([r], dtype=torch.int32)
        state, reward, game_over = env.step(move)
        print(state)
        print(int(reward[0]))
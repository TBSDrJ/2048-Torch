"""
Dr. J, Spring 2024
Implementation of 2048 as an importable class/module, or playable as __main__.
"""
from __future__ import annotations
import random
from argparse import ArgumentParser

class Game2048:
    """Game Engine for 2048 with a simple API and no rendering."""
    def __init__(self, width:int=4, height:int=4, prob_4:float=0.1):
        self.width = width
        self.height = height
        self.prob_4 = prob_4
        self.game_over = False
        self.score = 0
        self.board = self._new_game()
        random.seed(2048)

    def _new_game(self) -> list[list[int]]:
        """Sets up a new board with two random tiles in it."""
        board = [[0 for i in range(self.width)] for j in range(self.height)]
        tiles_placed = 0
        while tiles_placed < 2:
            i = random.randrange(self.width)
            j = random.randrange(self.height)
            if board[j][i] == 0:
                if random.random() < self.prob_4:
                    board[j][i] = 4
                else:
                    board[j][i] = 2
                tiles_placed += 1            
        return board

    def _merge_one_left(self, row: list[int], execute: bool
            ) -> (list[int], bool):
        """Apply merge operation to one row, moving left.
        
        Other moves can be executed by setting up the row properly."""
        new_row = [0] * len(row)
        i = 0
        pvs_entry = 0
        for entry in row:
            if entry != 0 and entry != pvs_entry:
                # Move nonzero entries left if not combined
                new_row[i] = entry
                i += 1
                pvs_entry = entry
            elif entry != 0 and entry == pvs_entry:
                # Combine if possible
                new_row[i-1] = 2*entry
                if execute:
                    self.score += 2*entry
                pvs_entry = 0
        changed = False
        for i in range(len(row)):
            if row[i] != new_row[i]:
                changed = True
        return new_row, changed

    def _merge(self, dir: str, execute:bool=True) -> bool:
        """Helper function that combines the four moves.
        
        I decided to write a helper function to avoid repetitive code.
        It may decrease readability?"""
        board = [[entry for entry in row] for row in self.board]
        changed = False
        if dir == 'L' or dir == 'l' or dir == 'R' or dir == 'r':
            for i in range(self.height):
                if dir == 'L' or dir == 'l':
                    row = board[i]
                elif dir == 'R' or dir == 'r':
                    row = board[i]
                    row.reverse()
                row, ch = self._merge_one_left(row, execute)
                if dir == 'L' or dir == 'l':
                    board[i] = row
                elif dir == 'R' or dir == 'r':
                    row.reverse()
                    board[i] = row
                if ch: changed = True
        else:
            for i in range(self.width):
                if dir == 'U' or dir == 'u':
                    row = [board[j][i] for j in range(self.height)]
                elif dir == 'D' or dir == 'd':
                    row = [board[j][i] for j in range(self.height)]
                    row.reverse()
                else:
                    raise ValueError(f"Invalid direction '{dir}' received.")
                row, ch = self._merge_one_left(row, execute)
                if dir == 'U' or dir == 'u':
                    for j in range(self.height):
                        board[j][i] = row[j]
                elif dir == 'D' or dir == 'd':
                    for j in range(self.height):
                        board[self.height-j-1][i] = row[j]
                if ch: changed = True
        if execute:
            self.board = board
            return changed
        else:
            return changed

    def merge_left(self, execute:bool=True) -> bool:
        """Response to receiving a left move. Returns True if something moved."""
        return self._merge('L', execute)

    def merge_right(self, execute:bool=True) -> bool:
        """Response to receiving a right move. Returns True if something moved."""
        return self._merge('R', execute)

    def merge_up(self, execute:bool=True) -> bool:
        """Response to receiving an up move Returns True if something moved.."""
        return self._merge('U', execute)

    def merge_down(self, execute:bool=True) -> bool:
        """Response to receiving a down move. Returns True if something moved.q"""
        return self._merge('D', execute)

    def game_over_check(self) -> bool:
        """Checks if there is a possible remaining move."""
        if self.merge_left(execute=False): return False
        if self.merge_right(execute=False): return False
        if self.merge_up(execute=False): return False
        if self.merge_down(execute=False): return False
        return True

    def add_tile(self) -> bool:
        """Add a random 2 or 4 to the board somewhere.
        
        Return True if a tile was able to be added, False if not."""
        available_locations = []
        for i in range(self.width):
            for j in range(self.height):
                if self.board[j][i] == 0:
                    available_locations.append((i,j))
        if len(available_locations) > 0:
            loc = random.choice(available_locations)
            if random.random() < self.prob_4:
                self.board[loc[1]][loc[0]] = 4
            else:
                self.board[loc[1]][loc[0]] = 2
            return True
        return False
    
    def one_turn(self, move: int) -> bool:
        """Run one turn.  Returns True if something changed, False if not.
        
        Moves: 0=Up, 1=Left, 2=Down, 3=Right (following WASD)"""
        if move == 0:
            ch = self.merge_up()
        elif move == 1:
            ch = self.merge_left()
        elif move == 2:
            ch = self.merge_down()
        elif move == 3:
            ch = self.merge_right()
        if ch:
            self.add_tile()
        self.game_over = self.game_over_check()
        return ch

class Text2048:
    def __init__(self, width = 4, height = 4, prob_4 = 0.1, play_game = True):
        self.width = width
        self.height = height
        self.prob_4 = prob_4
        self.game = Game2048(width, height, prob_4)
        if play_game:
            print("Please use WASD controls: W=Up, A=Left, S=Down, D=Right.")
            print()
            self.print_board()
            print(f"SCORE: {self.game.score}")
            print()
            while not self.game.game_over:
                move = input("Move: ")
                valid_moves = ['W', 'w', 'A', 'a', 'S', 's', 'D', 'd']
                if move in valid_moves:
                    if move == 'W' or move == 'w': move = 0
                    if move == 'A' or move == 'a': move = 1
                    if move == 'S' or move == 's': move = 2
                    if move == 'D' or move == 'd': move = 3
                    self.game.one_turn(move)
                    self.print_board()
                    print(f"SCORE: {self.game.score}")
                    print()
                else:
                    print("Invalid Move, please try again.")
            print("GAME OVER.  Thanks for playing!")

    def print_board(self, **kwargs) -> None:
        """Will pass kwargs to print call other than end.
        
        Additional kwarg behavior:
        end: Will apply as end=kwargs["end"] for each entry only.
        row_start: Replace default "|" with kwargs["row_start"]
        row_end: Replace default "|\n" with kwargs["row_end"]
        board_start: Replace default "" with kwargs["board_start"]
        board_end: Replace default "" with kwargs["board_end"]
        file: Replace default None with kwargs["file"]
        """
        if "end" in kwargs.keys():
            end = kwargs["end"]
            del kwargs["end"]
        else:
            end = " "
        if "row_start" in kwargs.keys():
            row_start = kwargs["row_start"]
            del kwargs["row_start"]
        else:
            row_start = "|"
        if "row_end" in kwargs.keys():
            row_end = kwargs["row_end"]
            del kwargs["row_end"]
        else:
            row_end = "|\n"
        if "board_start" in kwargs.keys():
            board_start = kwargs["board_start"]
            del kwargs["board_start"]
        else:
            board_start = ""
        if "board_end" in kwargs.keys():
            board_end = kwargs["board_end"]
            del kwargs["board_end"]
        else:
            board_end = ""
        if "file" in kwargs.keys():
            print_file = kwargs["file"]
            del kwargs["file"]
        else:
            print_file = None
        print(board_start, end="", file=print_file, **kwargs)
        for row in self.game.board:
            print(row_start, end="", file=print_file, **kwargs)
            for entry in row:
                print(f"{entry:5}", end=end, file=print_file, **kwargs)
            print(row_end, end="", file=print_file, **kwargs)
        print(board_end, end="", file=print_file, **kwargs)

if __name__ == "__main__":
    # ap = ArgumentParser()
    # ap.add_argument("--width", default=4, type=int,
    #         help="Set the width, or number of columns of the board.")
    # ap.add_argument("--height", default=4, type=int,
    #         help="Set the height, or number of rows of the board.")
    # ap.add_argument("-p", "--prob4", default=0.1, type=float,
    #         help="Set the probability that you get a 4 in a new tile.")
    # args = vars(ap.parse_args())
    # print(args)
    # game = Text2048(width=args["width"], height=args["height"], 
    #         prob_4=args["prob4"])
    scores = []
    for i in range(100):
        game = Game2048()
        while not game.game_over:
            move = random.randrange(4)
            game.one_turn(move)
            print(game.score, end="\r")
        print()
        scores.append(game.score)
    print(sum(scores) / len(scores))
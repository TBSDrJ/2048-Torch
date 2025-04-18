from collections import namedtuple

Transition = namedtuple('Transition', 
        ('state', 'move', 'next_state', 'reward', 'game_over'))

def read_saves(width, length):
    with open(f'2048_savegame_{width}_{length}.txt') as f:
        lines = f.readlines()
    transitions = []
    started_board = False
    end_of_board = False
    game_over = False
    prev_state = None
    next_state = None
    prev_score = 0
    for line in lines:
        if line[0] == "|":
            if not started_board:
                prev_state = next_state
                next_state = []
                started_board = True
            sp = line.split()
            row = [int(n) for n in sp[1:width + 1]]
            next_state.append(row)
        elif line[0:9] == "GAME OVER":
            game_over = True
            end_of_board = True
        else:
            sp = line.split()
            move = int(sp[0])
            score = int(sp[1])
            end_of_board = True
        if end_of_board:
            end_of_board = False
            started_board = False
            if prev_state is not None:
                transition = Transition(prev_state, move, next_state, 
                        score - prev_score, game_over)
                transitions.append(transition)
                prev_state = next_state
                prev_score = score
                game_over = False
    return transitions

if __name__ == "__main__":
    transitions = read_save(3, 3)
    count = 0
    for transition in transitions:
        if transition.game_over:
            count += 1
            print(transition)
            print(transitions.index(transition))
    print(count)

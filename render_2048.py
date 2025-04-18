import time
import sys

import pygame
from Game2048 import Game2048
from Game2048 import Text2048

def draw_board(
            screen: pygame.Surface, 
            game: Game2048, 
            font: pygame.font.Font,
    ) -> None:
    for i, row in enumerate(game.board):
        for j, entry in enumerate(row):
            pygame.draw.rect(screen, (20, 20, 150), 
                    (10 + j*120, 10 + i*120, 100, 100))
            if entry != 0:
                text = font.render(str(entry), True, (255, 100, 100), 
                        (20, 20, 150))
                text_loc = text.get_rect(center=(60 + j*120, 60 + i*120))
                screen.blit(text, text_loc)

def get_keys(game: Game2048, act:bool) -> (bool, int):
    action = None
    keys = pygame.key.get_pressed()
    num_keys = sum(list(keys))
    if num_keys == 0:
        act = True
    if num_keys == 1:
        if act:
            if keys[pygame.K_w]:
                action = 0
            if keys[pygame.K_a]:
                action = 1
            if keys[pygame.K_s]:
                action = 2
            if keys[pygame.K_d]:
                action = 3
            game.one_turn(action)
        act = False
    return act, action

def draw_score(
            screen: pygame.Surface, 
            game: Game2048, 
            font: pygame.font.Font,
    ) -> None:
    text = font.render("SCORE: " + str(game.score), True, (255, 100, 100), 
            (20, 20, 150))
    text_loc = text.get_rect(center=(screen.get_width() / 2, 
            screen.get_height() - 25))
    pygame.draw.rect(screen, (20, 20, 150), (text_loc.left - 10, 
            text_loc.top - 10, text_loc.width + 20, text_loc.height + 20))
    screen.blit(text, text_loc)

def main():
    usage = "Usage: python render.py [-w WIDTH] [-h HEIGHT] [-store] [--help]"
    w_error = "\n\nFlag -w needs to be followed by an integer.\n"
    h_error = "\n\nFlag -h needs to be followed by an integer.\n"
    args = sys.argv
    if "-w" in args:
        w = args.index("-w")
        if len(args) > w:
            try:
                width = int(args[w+1])
            except ValueError:
                print(w_error, usage)
                quit()
            except IndexError:
                print(w_error, usage)
                quit()
        else:
            print(w_error, usage)
            quit()
    else:
        width = 4
    if "-h" in args:
        h = args.index("-h")
        if len(args) > h:
            try:
                height = int(args[h+1])
            except ValueError:
                print(h_error, usage)
                quit()
            except IndexError:
                print(h_error, usage)
                quit()
        else:
            print(h_error, usage)
            quit()
    else:
        height = 4
    if "-store" in args:
        transcribe = True
    text_output = Text2048(width, height, play_game=False)
    game = text_output.game
    pygame.init()
    screen = pygame.display.set_mode((120*game.width, 120*game.height + 50))
    font = pygame.font.Font(None, 40)
    act = True
    if transcribe:
        saved_games = open(f'2048_savegame_{width}_{height}.txt', 'a')
        text_output.print_board(file=saved_games)
    while not game.game_over:
        screen.fill((0, 200, 255))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.game_over = True
        draw_board(screen, game, font)
        act, action = get_keys(game, act)
        draw_score(screen, game, font)
        pygame.display.flip()
        if transcribe and action is not None:
            print(action, game.score, file=saved_games)
            text_output.print_board(file=saved_games)
    print("GAME OVER", file=saved_games)
    saved_games.close()
    print(f"FINAL SCORE: {game.score}")
    pygame.quit()

if __name__ == "__main__":
    main()
from random import randint
import time
import sys
import os
from patterns import Singleton
from fysom import Fysom

@Singleton
class CLIOutput(object):
    cli_buffer = []

    def put_2_buffer(self, str):
        self.cli_buffer.append(str)

    def show_buffer(self):
        self.show_message("\n".join(self.cli_buffer))

    def clear_buffer(self):
        self.cli_buffer = []

    def clear_screen(self):
        os.system('clear')

    def show_and_save(self, str):
        self.show_message(str)
        self.put_2_buffer(str)

    def show_message(self, str):
        sys.stdout.write(str + "\n")
        sys.stdout.flush()

@Singleton
class CLIInput(object):

    def get_user_input(self, msg):
        output.show_message(msg)
        return raw_input("\n")

# ==========================================================
class GameMapControl(object):
    # map size: 40x9 Initial int: Zero (not One!) from top-left corner
    GAME = None

    # Pieces symbols
    BLACK_PIECE_SYMBOL = '#'
    WHITE_PIECE_SYMBOL = '@'

    def __init__(self, game_instance):
        self.GAME = game_instance

    def logic2real_coords(field_number):
        _fn = str(field_number)
        coords = {
            "1": 2,
            "2": 5,
            "3": 8,
            "4": 11,
            "5": 14,
            "6": 17,
            "7": 22,
            "8": 25,
            "9": 28,
            "10": 31,
            "11": 34,
            "12": 37,
            #  Bottom side. Vertical coord value: 6 - for symbol, 5 - for counter
            #  Top side. Vertical coord value: 2 - for symbol, 3 - for counter
            "13": 37,
            "14": 34,
            "15": 31,
            "16": 28,
            "17": 25,
            "18": 22,
            "19": 17,
            "20": 14,
            "21": 11,
            "22": 8,
            "23": 5,
            "24": 2,
        }

        return coords[_fn] if coords.has_key(_fn) else None

class GameRules(object):
    pass

class GameProceed(object):
    BLACK = 0
    WHITE = 1

    IS_JACKPOT = 0

    # Pieces position map: key - number of field, value - count of pieces
    # Initial maps before start
    BLACK_STATE_MAP = {1: 15}
    WHITE_STATE_MAP = {13: 15}

    # init empty states for dices
    FIRST_DICE = SECOND_DICE = 0

    HISTORY = []

    # States of step:
    # 0 - No movement
    # 1 - one piece was moved
    # 2 - two pieces was moved
    # 3 - three pieces was moved (jackpot only)
    # 4 - four pieces was moved (jackpot only)
    CURRENT_MOVE  = None # or list [color, points on dices, state of step(0|1|2|3|4)]
    CURRENT_MAP   = None # ?????
    CURRENT_ROUND = 1 # first round, increment witn step 0.5
    MAP_CONTROL   = None

    # Service and init
    def __init__(self):
        self.HISTORY = []
        self.CURRENT_ROUND = 1
        self._log_history('Started')
        self.MAP_CONTROL = GameMapControl(self)

    def _log_history(self, log_record):
        _d = time.strftime("%d-%m-%Y %H:%M:%S", time.localtime())
        self.HISTORY.append('['+ _d + '] ' + log_record)

    
    ###########################
    def roll_the_dice(self):
        return [randint(1, 6), randint(1, 6)]

    def set_dices_state(self, dices_list):
        self.FIRST_DICE, self.SECOND_DICE = dices_list
        if self.FIRST_DICE == self.SECOND_DICE:
            self.IS_JACKPOT = 1
        else:
            self.IS_JACKPOT = 0

    def set_current_move(self, color, points, state):
        self.CURRENT_MOVE = [color, points, state]

    # Error's handling
    def error(self, err_msg):
        self._log_history('[!] Error: %s' % err_msg)
        pass

    def reset(self):
        pass

    # Getters
    def get_current_round(self):
        return (self.CURRENT_ROUND, self.CURRENT_MOVE[0])

    def get_pieces_color_name(self, color_id):
        if color_id in (self.BLACK, self.WHITE): return dict({"0": "black", "1": "white"}).get(str(color_id))
        else: self.error('Incorrect value of color')

    def get_full_current_state_map(self):
        return self.BLACK_STATE_MAP.update(self.WHITE_STATE_MAP)

    def get_current_state_map(self, color_id):
        return self.BLACK_STATE_MAP if (color_id == 0) else self.WHITE_STATE_MAP

    # Game actions
    def begin_round(self, color_id):
        self.set_current_move(color_id, 0, 0)

    def finish_round(self):
        pass

    def begin_step(self):
        pass

    def cancel_step(self):
        pass

    def check_player_step(self):
        pass



output = CLIOutput.Instance();
input  = CLIInput.Instance()
game   = GameProceed();


def get_start_points():
    output.show_and_save("Who goes first?\n")
    for x in range(0, 6):
        output.clear_screen()
        output.show_buffer()
        b_bone, w_bone = game.roll_the_dice()
        mes = "|=====||=====|\n|  %d  ||  %d  |\n|=====||=====|\n\n" % (b_bone, w_bone)
        output.show_message(mes)
        time.sleep(0.3)
        #time.sleep(2.0 - float('0.' + str(x)))

    if (b_bone > w_bone):
        output.show_message("Left dice is win! White moves first")
        game.begin_round(game.WHITE)
    elif (w_bone > b_bone):
        output.show_message("Right dice is win! Black moves first")
        game.begin_round(game.BLACK)
    else:
        output.show_message("Bang!! Dead heat! Try one else time")
        input.get_user_input('Hit [ENTER] when ready\n')
        output.clear_buffer()
        get_start_points()


def main():
    get_start_points()
    input.get_user_input('Press [ENTER] to continue...\n')

    # first round check


def show_field():
    print ' 24 23 22 21 20 19   18 17 16 15 14 13  '
    print '|--------------------------------------|'
    print '| -  -  -  -  -  - || -  -  -  -  -  # |'
    print '| -  -  -  -  -  - || -  -  -  -  - 15 |'
    print '| -  -  -  -  -  - || -  -  -  -  -  - |'
    print '| 15 -  -  -  -  - || -  -  -  -  -  - |'
    print '| @  -  -  -  -  - || -  -  -  -  -  - |'
    print '|--------------------------------------|'
    print '  1  2  3  4  5  6    7  8  9  10 11 12 '

### Run, Vasya, run!

main()
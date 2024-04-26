from karel.stanfordkarel import *

"""
File: CheckerboardKarel.py
----------------------------
When you finish writing it, CheckerboardKarel should draw
a checkerboard using beepers, as described in Assignment 1. 
You should make sure that your program works for all of the 
sample worlds supplied in the starter folder.
"""


def main():
    create_board()

# this function lets the robot create the whole checkboard
# by creating row after row in a loop till he arrives at the ceiling
def create_board():

    # check if robot already faces the wall, and if yes turn upwards to solve this special case
    if front_is_blocked():
        turn_left()
    # if not facing the whole, we can star the process by doing one row at a time
    while front_is_clear():
        create_row()

# this function lets the robot create a row and after its finished
# the robot goes up one row (if he is not facing the ceiling already)
def create_row():
    while front_is_clear():
        put_beeper()
        move()
        if front_is_clear():
            move()
            if front_is_blocked():
                put_beeper()
                move_up_n_forward()
    move_up()

# this functions lets the robot go up in a row if he arrived at the wall and do one step forward
# reason for this is to get the checkboard pattern, which means to avoid two
# markings in a row in the <<create_row>> function
def move_up_n_forward():
    if left_is_clear() and facing_east():
        turn_left()
        move()
        turn_left()
        if front_is_clear():
            move()
    elif right_is_clear() and facing_west():
        turn_right()
        move()
        turn_right()
        if front_is_clear():
            move()

# this function lets the robot move one row up, if he arrived at the wall
def move_up():
    if left_is_clear() and facing_east():
        turn_left()
        move()
        turn_left()
    elif right_is_clear() and facing_west():
        turn_right()
        move()
        turn_right()

# help function to turn right
def turn_right():
    turn_left()
    turn_left()
    turn_left()


# There is no need to edit code beyond this point

if __name__ == "__main__":
    run_karel_program()

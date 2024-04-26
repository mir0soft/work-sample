from karel.stanfordkarel import *

"""
File: StoneMasonKarel.py
------------------------
When you finish writing code in this file, StoneMasonKarel should 
solve the "repair the quad" problem from Assignment 1. You
should make sure that your program works for all of the 
sample worlds supplied in the starter folder.
"""


def main():
# we need to first turn left to make life easy,
# then we can call the most highest level fuction repair_main_quad
    turn_left()
    repair_main_quad()


def repair_main_quad():
    while right_is_clear():
        repair_arch()
    # repair last remaining column
    repair_arch()

# repair each arch and go to the next column
def repair_arch():
    while front_is_clear():
        if no_beepers_present():
            put_beeper()
        move()
    if no_beepers_present():
        put_beeper()
    go_to_next_column()


def go_to_next_column():
    get_down()
    # check if we hit the end of the Main Quad
    if front_is_clear():
        for i in range(4):
            move()
        turn_left()

# help function to get down
def get_down():
    turn_around()
    while front_is_clear():
        move()
    turn_left()

def turn_around():
    turn_left()
    turn_left()

def turn_right():
    turn_left()
    turn_left()
    turn_left()






# There is no need to edit code beyond this point

if __name__ == "__main__":
    run_karel_program()

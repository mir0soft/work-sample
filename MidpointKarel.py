from karel.stanfordkarel import * 

"""
File: MidpointKarel.py
----------------------
When you finish writing it, MidpointKarel should leave
a beeper on the corner closest to the center of 1st Street
(or either of the two central corners if 1st Street has an even
number of corners).  Karel can put down additional beepers as it
looks for the midpoint, but must pick them up again before it
stops.  The world may be of any size, but you are allowed to
assume that it is at least as tall as it is wide.
"""

def main():
    find_midpoint()

def find_midpoint():
    mark_diagonal()
    get_down()
    hit_diagonal()
    mark_midpoint()
    collect_rest_beepers()
    get_down()
    go_to_middle()

def mark_midpoint():
    put_beeper()

def go_to_middle():
    turn_right()
    while no_beepers_present():
        move()

def collect_rest_beepers():
    turn_right()
    get_down()
    turn_left()
    turn_left()
    collect_diagonal()

def collect_diagonal():
    while front_is_clear():
        pick_beeper()
        move()
        turn_left()
        move()
        turn_right()
    turn_right()

def mark_diagonal():
    while front_is_clear():
        put_beeper()
        move()
        turn_left()
        move()
        turn_right()
    turn_right()

def get_down():
    while front_is_clear():
        move()

def hit_diagonal():
    turn_right()
    while no_beepers_present() and front_is_clear():
        move()
        if beepers_present():
            turn_left()
            get_down()
        else:
            turn_right()
            move()
            turn_left()
    if front_is_clear():
        turn_left()
        get_down()


def turn_right():
    turn_left()
    turn_left()
    turn_left()

# There is no need to edit code beyond this point

if __name__ == "__main__":
    run_karel_program()

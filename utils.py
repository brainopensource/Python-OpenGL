import time
from glfw_initialize2 import *
from math import pi


# ------------ Config Variables
WHITE_COLOR = (1.0, 1.0, 1.0)
BLACK_COLOR = (0.0, 0.0, 0.0)


NEAR_PROJ = 0.1
FAR_PROJ = 1000
SWAP_INTERVAL = 1
PLAYER_SPEED = 0.05


GRID_DENSITY = 20
SURFACE_ROWS = 50
SURFACE_COLS = 50
GRID_ROWS = 1
GRID_COLS = 1
INSTANCE_AREA = GRID_ROWS * GRID_COLS
GRID_SPACING = (2*pi) * 1.8

DRAW_POLYS = 1

RESOLUTION = [1420, 800]
last_x, last_y = RESOLUTION[0] / 2, RESOLUTION[1] / 2
first_mouse = True
left, right, forward, backward, up, down = False, False, False, False, False, False






def time_record(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__}: {round(end_time - start_time, 2)} s. ")
        return result
    return wrapper

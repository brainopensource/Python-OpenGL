import ctypes
from math import sin, pi
from pyrr import Vector3, matrix44
import random
from sys import exit
#from utils import *
from numba import jit, prange, uint, float32
#import glfw
from glfw_initialize2 import *
from math import sin, cos
import numpy as np
from OpenGL.GL import (
    glClear, glClearColor, glEnableClientState, glLoadIdentity,
    glRotatef, glVertexPointer, GL_COLOR_BUFFER_BIT, GL_FLOAT, GL_TRIANGLES, GL_TRUE, GL_FALSE, GL_VERTEX_ARRAY,
    GL_COLOR_ARRAY, glDrawArrays, glColorPointer, GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, glUseProgram, glGenBuffers,
    glBindBuffer, GL_ARRAY_BUFFER, glBufferData, GL_STATIC_DRAW, glGetAttribLocation, glEnableVertexAttribArray,
    glVertexAttribPointer, GL_TRIANGLE_STRIP, glViewport, GL_ELEMENT_ARRAY_BUFFER, glDrawElements, GL_UNSIGNED_INT,
    glEnable, GL_DEPTH_TEST, glGetUniformLocation, GL_DEPTH_BUFFER_BIT, glUniformMatrix4fv, glPolygonMode,
    GL_FRONT_AND_BACK, GL_LINE, glLineWidth, GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, glBlendFunc,
    glGenVertexArrays, glBindVertexArray, glDeleteBuffers, glDeleteVertexArrays, glVertexAttribDivisor,
    glDrawElementsInstanced, glUniform3f, GL_DYNAMIC_DRAW, glUniform1f
    )
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr
from camera import Camera


# ------------------------------- GLOBAL VARIABLES --------------------------------------------------------------------

WHITE_COLOR = (1.0, 1.0, 1.0)
BLACK_COLOR = (0.0, 0.0, 0.0)
ASPECT_RATIO = RESOLUTION[0] / RESOLUTION[1]


NEAR_PROJ = 0.1
FAR_PROJ = 1000
SWAP_INTERVAL = 1
PLAYER_SPEED = 0.3


GRID_DENSITY = 500
SURFACE_ROWS = GRID_DENSITY
SURFACE_COLS = 2 * GRID_DENSITY
GRID_ROWS = 2
GRID_COLS = 1
GRID_SPACING = (2*pi) * 5

DRAW_POLYS = 0

last_x, last_y = RESOLUTION[0] / 2, RESOLUTION[1] / 2
first_mouse = True
left, right, forward, backward, up, down = False, False, False, False, False, False


cam = Camera()


# ------------------------------ Game Controls -------------------------------------------------------------------
def mouse_look_callback(window, x, y):
    global first_mouse, last_x, last_y
    if first_mouse:
        last_x = -x
        last_y = y
        first_mouse = False

    delta_x = (x - last_x) / 2
    delta_y = (last_y - y) / 3

    last_x = x
    last_y = y

    cam.process_mouse_movements(delta_x, delta_y)

    return


def keyboard_callback(window, key, scancode, action, mode):
    global left, right, forward, backward, up, down

    if key == glfw.KEY_W and action == glfw.PRESS:
        forward = True
    elif key == glfw.KEY_W and action == glfw.RELEASE:
        forward = False
    if key == glfw.KEY_S and action == glfw.PRESS:
        backward = True
    elif key == glfw.KEY_S and action == glfw.RELEASE:
        backward = False
    if key == glfw.KEY_A and action == glfw.PRESS:
        left = True
    elif key == glfw.KEY_A and action == glfw.RELEASE:
        left = False
    if key == glfw.KEY_D and action == glfw.PRESS:
        right = True
    elif key == glfw.KEY_D and action == glfw.RELEASE:
        right = False
    if key == glfw.KEY_Q and action == glfw.PRESS:
        up = True
    elif key == glfw.KEY_Q and action == glfw.RELEASE:
        up = False
    if key == glfw.KEY_E and action == glfw.PRESS:
        down = True
    elif key == glfw.KEY_E and action == glfw.RELEASE:
        down = False


def move_camera():
    if left:
        cam.process_keyboard("LEFT", PLAYER_SPEED)
    if right:
        cam.process_keyboard("RIGHT", PLAYER_SPEED)
    if forward:
        cam.process_keyboard("FORWARD", PLAYER_SPEED)
    if backward:
        cam.process_keyboard("BACKWARD", PLAYER_SPEED)
    if up:
        cam.process_keyboard("UP", PLAYER_SPEED)
    if down:
        cam.process_keyboard("DOWN", PLAYER_SPEED)


#  ------------------------------  GLSL Functions ---------------------------------------------------------------------
def vertex_shader():
    with open('./shaders/vertex100.glsl', 'r') as file:
            string_variable = file.read()
    return compileShader(string_variable, GL_VERTEX_SHADER)


def fragment_shader():
    with open('./shaders/fragment.glsl', 'r') as file:
        string_variable = file.read()
    return compileShader(string_variable, GL_FRAGMENT_SHADER)


# ------------------------------ Mesh Functions --------------------------------------
def create_program():
    shader = compileProgram(vertex_shader(), fragment_shader())
    return shader


def gen_grid_instanced(rows, cols, instances_per_row, instances_per_col, instance_spacing):
    vertices = []
    indices = []
    num_instances = instances_per_row * instances_per_col

    # Constants for the range
    x_start = -8 * np.pi
    x_end = 8 * np.pi
    z_start = -8 * np.pi
    z_end = 8 * np.pi

    # Base grid vertices and indices generation
    for i in range(rows + 1):
        for j in range(cols + 1):
            x = x_start + (j / cols) * (x_end - x_start)  # Adjust x to span from -2*pi to 2*pi
            z = z_start + (i / rows) * (z_end - z_start)  # Adjust z to span from -2*pi to 2*pi
            y = 0.0  # Flat grid on the XZ plane
            vertices.extend([x, y, z])

    for i in range(rows):
        for j in range(cols):
            start = i * (cols + 1) + j
            indices.extend([start, start + 1, start + cols + 1,
                            start + 1, start + cols + 2, start + cols + 1])

    vertices = np.array(vertices, dtype=np.float32)
    indices = np.array(indices, dtype=np.uint32)

    # Adjusted Grid of grids offsets for symmetry
    offsets = []
    offset_x_start = -(instances_per_col - 1) / 2 * instance_spacing
    offset_z_start = -(instances_per_row - 1) / 2 * instance_spacing
    for i in range(instances_per_row):
        for j in range(instances_per_col):
            offsetX = offset_x_start + j * instance_spacing
            offsetY = 0.0  # Keep Y offset as 0 to stay on the XZ plane
            offsetZ = offset_z_start + i * instance_spacing
            offsets.append([offsetX, offsetY, offsetZ])

    offsets = np.array(offsets, dtype=np.float32)

    # Fixed colors for each instance
    #colors = np.random.rand(num_instances, 4).astype(np.float32)  # RGBA
    acolor = np.array([0.0, 0.55, 0.6, 1.0], dtype=np.float32)  # Cyan with full opacity
    colors = np.tile(acolor, (num_instances, 1))

    return vertices, indices, offsets, colors


def create_buffers(vertices, indices, offsets, colors):
    # Generate VAO / VBO
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)

    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    offset_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, offset_vbo)
    glBufferData(GL_ARRAY_BUFFER, offsets.nbytes, offsets, GL_STATIC_DRAW)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(1)
    glVertexAttribDivisor(1, 1)

    # Color VBO
    color_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, color_vbo)
    glBufferData(GL_ARRAY_BUFFER, colors.nbytes, colors, GL_STATIC_DRAW)
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(2)
    glVertexAttribDivisor(2, 1)

    return vao


def clear_frame(vao):
    glDeleteVertexArrays(1, [vao])


# ---------------------------       MAIN CODE  -------------------------------------------------------------------
def main_game(gwindow):

    glfw.set_cursor_pos_callback(gwindow, mouse_look_callback)
    glfw.set_key_callback(gwindow, keyboard_callback)
    glfw.set_input_mode(gwindow, glfw.CURSOR, glfw.CURSOR_DISABLED)
    glfw.swap_interval(SWAP_INTERVAL)
    glEnable(GL_DEPTH_TEST)
    if DRAW_POLYS == True:
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    glClearColor(0.5, 0.5, 0.5, 1.0)

    shader = create_program()
    glUseProgram(shader)
    rows = SURFACE_ROWS
    cols = SURFACE_COLS
    instances_per_row = GRID_ROWS
    instances_per_col = GRID_COLS
    spacing = GRID_SPACING
    instances_area = instances_per_row * instances_per_col
    vertices, indices, offsets, colors = gen_grid_instanced(rows, cols,
                                                            instances_per_row, instances_per_col,
                                                            spacing)

    indices_count = len(indices)
    vao = create_buffers(vertices,indices, offsets, colors)

    projection = matrix44.create_perspective_projection_matrix(45.0, RESOLUTION[0] / RESOLUTION[1], NEAR_PROJ, FAR_PROJ)
    proj_location = glGetUniformLocation(shader, "projection")
    glUniformMatrix4fv(proj_location, 1, GL_FALSE, projection)
    view = cam.get_view_matrix()
    view_location = glGetUniformLocation(shader, "view")
    glUniformMatrix4fv(view_location, 1, GL_FALSE, view)

    time_location = glGetUniformLocation(shader, "time")

    playerPosLocation = glGetUniformLocation(shader, "playerPosition")

    running = True
    frame_count = 0
    zero_time = glfw.get_time()
    while running:

        start_time = glfw.get_time()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUniform1f(time_location, start_time * 10)
        move_camera()
        view = cam.get_view_matrix()
        glUniformMatrix4fv(view_location, 1, GL_FALSE, view)
        print(cam.camera_pos[0], cam.camera_pos[1], cam.camera_pos[2])
        glUniform3f(playerPosLocation, cam.camera_pos[0], cam.camera_pos[1], cam.camera_pos[2])

        glBindVertexArray(vao)
        glDrawElementsInstanced(GL_TRIANGLES, indices_count, GL_UNSIGNED_INT, None, instances_area)

        # Reset frame
        glfw.swap_buffers(gwindow)
        frame_count, zero_time, fps = handle_events(frame_count, zero_time, start_time, gwindow)
        glfw.poll_events()

        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.terminate()
            running = False
 
        frame_count += 1

    # Cleanup
    glfw.terminate()


# Starting script
if __name__ == "__main__":
    start_t = time.time()
    window = initialize_glfw(RESOLUTION)
    main_game(window)
    exit()


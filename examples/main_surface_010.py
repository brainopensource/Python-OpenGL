import ctypes
from sys import exit
import random
from utils import *
from numba import jit, prange
import time
import polars as pl
import glfw
from glfw_initialize import *
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
    glGenVertexArrays, glBindVertexArray,
    )
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr
from camera import Camera
#from player import *


# ------------------------------- GLOBAL VARIABLES --------------------------------------------------------------------
GRID_ROWS = 100
GRID_COLS = 100

WHITE_COLOR = (1.0, 1.0, 1.0)
BLACK_COLOR = (0.0, 0.0, 0.0)
ASPECT_RATIO = RESOLUTION[0]/RESOLUTION[1]

cam = Camera()
NEAR_PROJ = 0.1
FAR_PROJ = 1000

last_x, last_y = RESOLUTION[0]/2, RESOLUTION[1]/2
first_mouse = True
left, right, forward, backward = False, False, False, False


#  ------------------------------  GLSL Functions ---------------------------------------------------------------------
def vertex_shader():
    vertex_src = """
    # version 330 core
    
    layout(location = 0) in vec3 position;
    
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    
    void main() {
        gl_Position = projection * view * model * vec4(position, 1.0);
    }
    """
    return compileShader(vertex_src, GL_VERTEX_SHADER)


def fragment_shader():
    fragment_src = """
    # version 330 core
    out vec4 FragColor;
    void main() {
        FragColor = vec4(0.1, 0.6, 0.9, 1.0); // White color
    }
    """
    return compileShader(fragment_src, GL_FRAGMENT_SHADER)


# ------------------------------ Mesh Functions --------------------------------------
@time_record
def create_program():
    shader = compileProgram(vertex_shader(), fragment_shader())
    return shader

#@time_record
#@jit(nopython=True, parallel=True)
def create_grid_old(rows, cols):
    """Generates vertex and index data for a 3D surface."""
    vertices = []
    indices = []
    for i in prange(rows - 1):
        for j in range(cols):
            x = (j / (cols - 1) - 0.5) * 50  # Centering the grid at the origin
            z = (i / (rows - 1) - 0.5) * 50
            y = (sin(x + sin(z/2)) * cos(z/2 + cos(x/3)) + sin(x/2 + sin(z/4)) * cos(z/4 + cos(x/6))) * sin(2 * x) * 0.5  # Generate z based on a function of x and y
            #z = 0.1
            vertices.extend([x, y, z])

    # Generating two triangles for each square in the grid
    for i in prange(rows - 1):
        for j in range(cols - 1):
            start = i * cols + j
            indices.extend([start, start + 1, start + cols, start + 1, start + cols + 1, start + cols])

    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)


@time_record
@jit(nopython=True, cache=True)
def create_grid(rows, cols, offsetx=0, offsetz=0):
    """Generates vertex and index data for a 3D surface."""
    vertices = []
    indices = []

    # Generate vertices
    x_vals = np.linspace(-0.5 * 50, 0.5 * 50, cols)
    z_vals = np.linspace(-0.5 * 50, 0.5 * 50, rows - 1)

    for i in prange(rows):
        for j in prange(cols):
            x = x_vals[j] + offsetx
            z = z_vals[i] + offsetz if i < len(z_vals) else z_vals[-1] + offsetz # Ensure that we don't go out of bounds for z_vals
            y = (sin(x + sin(z / 2)) * cos(z / 2 + cos(x / 3)) + sin(x / 2 + sin(z / 4)) * cos(
                z / 4 + cos(x / 6))) * sin(2 * x) * 0.5
            vertices.extend([x, y, z])

    # Generate indices
    for i in prange(rows - 1):
        for j in prange(cols - 1):
            start = i * cols + j
            indices.extend([start, start + 1, start + cols])
            indices.extend([start + 1, start + cols + 1, start + cols])

    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)


def generate_chunks(player_position, chunk_size, grid_rows, grid_cols):
    # Calculate the base offset from the player's position
    base_offset_x = player_position[0] - (chunk_size * 1.5)  # Assuming player_position has x attribute
    base_offset_z = player_position[1] - (chunk_size * 1.5)  # Assuming player_position has z attribute

    chunks_data = []
    for i in range(3):  # Row in the 3x3 grid
        for j in range(3):  # Column in the 3x3 grid
            # Calculate offsets for the current chunk
            offsetX = base_offset_x + j * chunk_size
            offsetZ = base_offset_z + i * chunk_size
            # Generate chunk
            vertices, indices = create_grid(grid_rows, grid_cols, offsetX, offsetZ)
            chunks_data.append((vertices, indices))

    return chunks_data


@time_record
def create_objects():

    shader = create_program()
    vertices, indices = create_grid(GRID_ROWS, GRID_COLS, -10, 50)
    len_index = len(indices)

    # VAO, VBO, EBO
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
    # Set vertex attribute pointers
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertices.itemsize * 3, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    # Use shader and set background color
    glUseProgram(shader)
    glClearColor(0, 0, 0, 1)
    glEnable(GL_DEPTH_TEST)

    return shader, vao, len_index


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
    global left, right, forward, backward

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


def move_camera():
    if left:
        cam.process_keyboard("LEFT", 0.5)
    if right:
        cam.process_keyboard("RIGHT", 0.5)
    if forward:
        cam.process_keyboard("FORWARD", 0.5)
    if backward:
        cam.process_keyboard("BACKWARD", 0.5)


# ---------------------------       MAIN CODE  -------------------------------------------------------------------
def main_game(gwindow):

    zero_time = glfw.get_time()
    glfw.set_cursor_pos_callback(gwindow, mouse_look_callback)
    glfw.set_key_callback(window, keyboard_callback)
    glfw.set_input_mode(gwindow, glfw.CURSOR, glfw.CURSOR_DISABLED)

    #Create all objects meshes
    shader, vao, len_index = create_objects()

    # View configuring
    projection = pyrr.matrix44.create_perspective_projection_matrix(60.0, ASPECT_RATIO, NEAR_PROJ, FAR_PROJ)
    surf_pos = pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 0, -3]))
    # Uniform locations
    model_loc = glGetUniformLocation(shader, "model")
    proj_loc = glGetUniformLocation(shader, "projection")
    view_loc = glGetUniformLocation(shader, "view")
    # Uniforms update
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    #glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, surf_pos)

    running = True
    frame_count = 0
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    print('Start render', glfw.get_time() - zero_time)
    zero_time = glfw.get_time()

    while running:
        current_time = glfw.get_time()

        # Start frame
        glfw.poll_events()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        frame_count, zero_time, fps = display_fps(frame_count, zero_time, current_time, gwindow)

        # Process camera and player actions
        move_camera()
        view = cam.get_view_matrix()
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)

        # Process rendering pipelines #glViewport(200, 200, 900, 600)
        glBindVertexArray(vao)
        glDrawElements(GL_TRIANGLES, len_index, GL_UNSIGNED_INT, None)

        # Reset frame
        glfw.swap_buffers(gwindow)
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.terminate()
            exit()
        frame_count += 1

    # Cleanup
    glfw.terminate()

# Starting script
if __name__ == "__main__":
    start_t = time.time()
    window = initialize_glfw(RESOLUTION)
    main_game(window)


import ctypes
import random
from sys import exit
#from utils import *
from numba import jit, prange, uint, float32
#import glfw
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
    glGenVertexArrays, glBindVertexArray, glDeleteBuffers, glDeleteVertexArrays,
    )
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr
from camera import Camera


# ------------------------------- GLOBAL VARIABLES --------------------------------------------------------------------
GRID_ROWS = 10
GRID_COLS = GRID_ROWS
CHUNK_SIZE = GRID_ROWS
#CHUNK_SIZE = GRID_ROWS - 1   # NO GAPS MODE
GRID_SIZE = 4
GRID_CENTER = CHUNK_SIZE * (GRID_SIZE // 2)

WHITE_COLOR = (1.0, 1.0, 1.0)
BLACK_COLOR = (0.0, 0.0, 0.0)
ASPECT_RATIO = RESOLUTION[0]/RESOLUTION[1]

cam = Camera()
NEAR_PROJ = 0.1
FAR_PROJ = 1000
SWAP_INTERVAL = 1
PLAYER_SPEED = 0.9

last_x, last_y = RESOLUTION[0]/2, RESOLUTION[1]/2
first_mouse = True
left, right, forward, backward = False, False, False, False


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
        cam.process_keyboard("LEFT", PLAYER_SPEED)
    if right:
        cam.process_keyboard("RIGHT", PLAYER_SPEED)
    if forward:
        cam.process_keyboard("FORWARD", PLAYER_SPEED)
    if backward:
        cam.process_keyboard("BACKWARD", PLAYER_SPEED)


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
def create_program():
    shader = compileProgram(vertex_shader(), fragment_shader())
    return shader


@jit(nopython=True, cache=True)
def create_grid(rows, cols, offsetx=0, offsetz=0):
    """Generates vertex and index data for a 3D surface."""
    vertices = []
    indices = []
    for i in prange(rows):
        for j in prange(cols):
            x = int(2 * j + 2 * offsetx)
            z = int(2 * i + 2 * offsetz)
            y = 0  #round(sin(x/2 + sin(z/4)), 2)
            vertices.extend([x, y, z])

    # Generate indices
    for i in prange(rows - 1):
        for j in prange(cols - 1):
            start = i * cols + j
            indices.extend([start, start + 1, start + cols])
            indices.extend([start + 1, start + cols + 1, start + cols])

    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)


@jit(nopython=True, cache=True)
def generate_chunks(pos_x, pos_z, chunk_size, center_offset, grid_rows, grid_cols):

    # Calculate the base offset from the player's position
    base_offset_x = pos_x - center_offset
    base_offset_z = pos_z - center_offset

    chunks_data = []
    for i in prange(GRID_SIZE):  # Row in the 3x3 grid
        for j in prange(GRID_SIZE):  # Column in the 3x3 grid
            # Calculate offsets for the current chunk
            offset_x = base_offset_x + j * chunk_size
            offset_z = base_offset_z + i * chunk_size
            # Generate chunk
            vertices, indices = create_grid(grid_rows, grid_cols, offset_x, offset_z)
            chunk_uid = int(offset_x // chunk_size + (offset_z // chunk_size) * GRID_SIZE)
            chunks_data.append((vertices, indices, chunk_uid))

    return chunks_data


@jit(nopython=True, cache=True)
def generate_chunks_old(pos_x, pos_z, chunk_size,  center_offset, grid_rows, grid_cols):

    # Calculate the base offset from the player's position
    base_offset_x = pos_x - center_offset
    base_offset_z = pos_z - center_offset

    chunks_data = []
    for i in prange(GRID_SIZE):  # Row in the 3x3 grid
        for j in prange(GRID_SIZE):  # Column in the 3x3 grid
            # Calculate offsets for the current chunk
            offset_x = base_offset_x + j * (chunk_size)
            offset_z = base_offset_z + i * (chunk_size)
            # Generate chunk
            vertices, indices = create_grid(grid_rows, grid_cols, offset_x, offset_z)
            chunk_uid = int(offset_x // chunk_size + (offset_z // chunk_size) * GRID_SIZE)
            chunks_data.append((vertices, indices, chunk_uid))

    return chunks_data


def prepare_chunk_render(vertices, indices):
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)
    return vao, vbo, ebo


def create_world_vaos(chunks_data):
    preprocessed_chunks = {}
    for vertices, indices, chunk_uid in chunks_data:
        vao, vbo, ebo = prepare_chunk_render(vertices, indices)
        preprocessed_chunks[chunk_uid] = (vao, vbo, ebo, len(indices))
    return preprocessed_chunks


def draw_chunks(vao, vbo, ebo, len_index, model, model_loc):
    glBindVertexArray(vao)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    glDrawElements(GL_TRIANGLES, len_index, GL_UNSIGNED_INT, None)
    #glDeleteBuffers(1, [vbo])       # these slows frame time __________________________
    #glDeleteBuffers(1, [ebo])       # ________________________and reduce memory _______
    #glDeleteVertexArrays(1, [vao])   # _________________________________________ locked


# ---------------------------       MAIN CODE  -------------------------------------------------------------------
def main_game(gwindow):

    zero_time = glfw.get_time()
    glfw.set_cursor_pos_callback(gwindow, mouse_look_callback)
    glfw.set_key_callback(window, keyboard_callback)
    glfw.set_input_mode(gwindow, glfw.CURSOR, glfw.CURSOR_DISABLED)

    # Start objects preparation
    shader = create_program()
    glUseProgram(shader)
    model = pyrr.matrix44.create_identity(dtype=np.float32)
    model_loc = glGetUniformLocation(shader, "model")
    proj_loc = glGetUniformLocation(shader, "projection")
    view_loc = glGetUniformLocation(shader, "view")
    projection = pyrr.matrix44.create_perspective_projection_matrix(60.0, ASPECT_RATIO, NEAR_PROJ, FAR_PROJ)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    glEnable(GL_DEPTH_TEST)
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    glfw.swap_interval(SWAP_INTERVAL)

    # Start World preparation
    pos_x, pos_z = 0, 0
    chunk_data = generate_chunks(pos_x, pos_z, CHUNK_SIZE, GRID_CENTER, GRID_ROWS, GRID_COLS)
    world_dict = create_world_vaos(chunk_data)
    uid_iterations = list(world_dict.keys())

    print('Start render', glfw.get_time() - zero_time)
    running = True
    frame_count = 0
    zero_time = glfw.get_time()
    while running:
        current_time = glfw.get_time()

        # Start frame
        glfw.poll_events()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        frame_count, zero_time, fps = handle_events(frame_count, zero_time, current_time, gwindow,
                                                    view_loc, [move_camera, cam.get_view_matrix])


        # Process rendering pipelines #glViewport(200, 200, 900, 600)
        #for vertices, indices, chunk_uid in chunks_data:
        #    draw_chunks(*create_chunk_objects(vertices, indices), model, model_loc)
        for grid in uid_iterations:
            vao, vbo, ebo, len_idx = world_dict[grid]
            draw_chunks(vao, vbo, ebo, len_idx, model, model_loc)


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


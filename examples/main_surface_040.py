import ctypes
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
    glGenVertexArrays, glBindVertexArray, glDeleteBuffers, glDeleteVertexArrays, glVertexAttribDivisor, glDrawElementsInstanced
    )
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr
from camera import Camera


# ------------------------------- GLOBAL VARIABLES --------------------------------------------------------------------
GRID_ROWS = 4
GRID_COLS = GRID_ROWS
CHUNK_SIZE = GRID_ROWS
GRID_SIZE = 3
HALF_GRID_SIZE = GRID_SIZE // 2
CENTER_OFFSET = HALF_GRID_SIZE * CHUNK_SIZE
#CHUNK_SIZE = GRID_ROWS - 1   # NO GAPS MODE

WHITE_COLOR = (1.0, 1.0, 1.0)
BLACK_COLOR = (0.0, 0.0, 0.0)
ASPECT_RATIO = RESOLUTION[0]/RESOLUTION[1]

cam = Camera()
NEAR_PROJ = 0.1
FAR_PROJ = 1000
SWAP_INTERVAL = 1
PLAYER_SPEED = 0.1

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
    layout(location = 1) in vec3 instancePos;
    
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    
    void main() {
        vec4 pos = vec4(position, 1.0) + vec4(instancePos, 0.0);
        gl_Position = projection * view * model * pos;
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


def gen_vertices():
    vertices = np.array([
        -1.0, 0.0, -1.0,
        1.0, 0.0, -1.0,
        1.0, 0.0, 1.0,
        -1.0, 0.0, 1.0,
    ], dtype=np.float32)

    indices = np.array([
        0, 1, 2,
        2, 3, 0
    ], dtype=np.uint32)

    return vertices, indices


def gen_instance_pos(i, j):
    return np.array([[x, 0, z] for x in range(i) for z in range(j)], dtype=np.float32)


def create_buffers(vertices, indices, instances):
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    ebo = glGenBuffers(1)
    instance_vbo = glGenBuffers(1)
    # Configuração do VAO
    glBindVertexArray(vao)
    # Carregando dados de vértices
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    # Carregando índices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
    # Atributo de vértice, posição
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * vertices.itemsize, None)
    glEnableVertexAttribArray(0)
    # Carregando instâncias
    glBindBuffer(GL_ARRAY_BUFFER, instance_vbo)
    glBufferData(GL_ARRAY_BUFFER, instances.nbytes, instances, GL_STATIC_DRAW)
    # Atributo de instância, posição
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * instances.itemsize, None)
    glEnableVertexAttribArray(1)
    glVertexAttribDivisor(1, 1)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)
    return vao, vbo, ebo, instance_vbo


def config_views(shader):
    model = pyrr.matrix44.create_identity(dtype=np.float32)
    model_loc = glGetUniformLocation(shader, "model")
    proj_loc = glGetUniformLocation(shader, "projection")
    view_loc = glGetUniformLocation(shader, "view")
    projection = pyrr.matrix44.create_perspective_projection_matrix(60.0, ASPECT_RATIO, NEAR_PROJ, FAR_PROJ)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    return model, model_loc, proj_loc, view_loc


def render_instance(vao, len_idx):
    glBindVertexArray(vao)
    glDrawElementsInstanced(GL_TRIANGLES, len_idx, GL_UNSIGNED_INT, None, 9)
    glBindVertexArray(0)


def clear_frame(vao, vbo, ebo, instance_vbo):
    glDeleteVertexArrays(1, [vao])
    glDeleteBuffers(1, [vbo])
    glDeleteBuffers(1, [ebo])
    glDeleteBuffers(1, [instance_vbo])


# ---------------------------       MAIN CODE  -------------------------------------------------------------------
def main_game(gwindow):

    zero_time = glfw.get_time()
    glfw.set_cursor_pos_callback(gwindow, mouse_look_callback)
    glfw.set_key_callback(gwindow, keyboard_callback)
    glfw.set_input_mode(gwindow, glfw.CURSOR, glfw.CURSOR_DISABLED)

    glfw.swap_interval(SWAP_INTERVAL)
    glEnable(GL_DEPTH_TEST)
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

    shader = create_program()
    glUseProgram(shader)

    model, model_loc, proj_loc, view_loc = config_views(shader)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)


    instances = gen_instance_pos(3, 3)
    print(instances)
    vertices, indices = gen_vertices()
    len_idx = len(indices)
    vao, vbo, ebo, instance_vbo = create_buffers(vertices, indices, instances)


    print('Start render', glfw.get_time() - zero_time)
    running = True
    frame_count = 0
    zero_time = glfw.get_time()
    while running:

        # Get Events
        current_time = glfw.get_time()
        glfw.poll_events()

        # Update camera
        view = cam.get_view_matrix()
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
        move_camera()

        # Clear old buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        frame_count, zero_time, fps = handle_events(frame_count, zero_time, current_time, gwindow)

        # Render objects
        render_instance(vao, len_idx)

        # Reset frame
        glfw.swap_buffers(gwindow)
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            clear_frame(vao, vbo, ebo, instance_vbo)
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


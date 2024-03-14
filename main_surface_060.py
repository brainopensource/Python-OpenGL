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
PLAYER_SPEED = 0.01


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
    vertex_src = """
    #version 330
    layout (location = 0) in vec3 position;
    layout (location = 1) in vec3 color;
    uniform mat4 view;
    uniform mat4 projection;
    uniform float timeOffset;
    out vec3 newColor;
    
    void main()
    {
        vec4 pos = vec4(position.x, position.y, position.z, 1.0);
        gl_Position = projection * view * pos;
        newColor = color;
    }
    """
    return compileShader(vertex_src, GL_VERTEX_SHADER)


def fragment_shader():
    fragment_src = """
    #version 330
    in vec3 newColor;
    out vec4 outColor;
    void main()
    {
        outColor = vec4(newColor, 1.0);
    }
    """
    return compileShader(fragment_src, GL_FRAGMENT_SHADER)


# ------------------------------ Mesh Functions --------------------------------------
def create_program():
    shader = compileProgram(vertex_shader(), fragment_shader())
    return shader


def generate_mesh_data():
    # Triangle vertices
    vertices = np.array([
        -0.5, 0.0, -0.5,  0.0, 0.8, 0.6,
         0.5, 0.0, -0.5,  0.0, 0.6, 0.8,
         0.0,  0.0, 0.5,  1.0, 0.7, 0.7,
    ], dtype=np.float32)
    return vertices


def generate_grid_mesh_data_indexed(rows, cols, spacing=1.0):
    vertices = []
    for i in range(rows + 1):
        for j in range(cols + 1):
            x = (j - cols / 2) * spacing
            y = (i - rows / 2) * spacing

            vertices.extend([x, y, 0, j / cols, i / rows, 0.9])  # Color (for visual interest)
    indices = []
    for i in range(rows):
        for j in range(cols):
            start = i * (cols + 1) + j
            indices.extend([start, start + 1, start + cols + 1,
                            start + 1, start + cols + 2, start + cols + 1])

    print(vertices)
    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)



def create_buffers(vertices, indices):
    # Generate VAO / VBO
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    # Position and Color attributes
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * vertices.itemsize, None)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * vertices.itemsize, ctypes.c_void_p(3 * vertices.itemsize))
    glEnableVertexAttribArray(1)

    return vao, vbo, ebo


def config_views(shader):
    model = pyrr.matrix44.create_identity(dtype=np.float32)
    model_loc = glGetUniformLocation(shader, "model")
    proj_loc = glGetUniformLocation(shader, "projection")
    view_loc = glGetUniformLocation(shader, "view")
    projection = pyrr.matrix44.create_perspective_projection_matrix(60.0, ASPECT_RATIO, NEAR_PROJ, FAR_PROJ)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    return model, model_loc, proj_loc, view_loc


def clear_frame(vao, vbo):
    glDeleteVertexArrays(1, [vao])
    glDeleteBuffers(1, [vbo])



# ---------------------------       MAIN CODE  -------------------------------------------------------------------
def main_game(gwindow):

    glfw.set_cursor_pos_callback(gwindow, mouse_look_callback)
    glfw.set_key_callback(gwindow, keyboard_callback)
    glfw.set_input_mode(gwindow, glfw.CURSOR, glfw.CURSOR_DISABLED)
    glfw.swap_interval(SWAP_INTERVAL)
    glEnable(GL_DEPTH_TEST)
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    glClearColor(0.3, 0.3, 0.3, 1.0)

    shader = create_program()
    glUseProgram(shader)
    vertices, indices = generate_grid_mesh_data_indexed(4, 4)
    index_count = len(indices)
    vao, vbo, ebo = create_buffers(vertices, indices)

    projection = matrix44.create_perspective_projection_matrix(45.0, RESOLUTION[0] / RESOLUTION[1], 0.1, 1000.0)
    proj_location = glGetUniformLocation(shader, "projection")
    glUniformMatrix4fv(proj_location, 1, GL_FALSE, projection)

    view = cam.get_view_matrix()
    view_location = glGetUniformLocation(shader, "view")
    glUniformMatrix4fv(view_location, 1, GL_FALSE, view)


    running = True
    frame_count = 0
    zero_time = glfw.get_time()

    while running:

        start_time = glfw.get_time()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        current_time = time.time()
        time_offset_location = glGetUniformLocation(shader, "timeOffset")
        glUniform1f(time_offset_location, current_time)
        move_camera()
        view = cam.get_view_matrix()
        glUniformMatrix4fv(view_location, 1, GL_FALSE, view)

        glBindVertexArray(vao)
        glDrawElements(GL_TRIANGLES, index_count, GL_UNSIGNED_INT, None)

        # Reset frame
        glfw.swap_buffers(gwindow)
        frame_count, zero_time, fps = handle_events(frame_count, zero_time, start_time, gwindow)
        glfw.poll_events()

        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.terminate()
            clear_frame(vao, vbo)
            exit()
        frame_count += 1



    # Cleanup
    glfw.terminate()



# Starting script
if __name__ == "__main__":
    start_t = time.time()
    window = initialize_glfw(RESOLUTION)
    main_game(window)


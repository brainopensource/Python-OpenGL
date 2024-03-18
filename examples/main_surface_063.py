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
FAR_PROJ = 100
SWAP_INTERVAL = 1
PLAYER_SPEED = 0.1


SURFACE_ROWS = 50
SURFACE_COLS = 50
GRID_ROWS = 10
GRID_COLS = 10
GRID_SPACING = 1.5


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
    layout (location = 1) in vec3 offset;
    layout (location = 2) in vec4 color;   
    uniform mat4 view;
    uniform mat4 projection;
    uniform float time;
    
    
    out vec4 vertexColor;
   
void main()
{
    vec3 pos = position;
    // Base wave
    float wave = sin(0.5 * (1 + pos.x) + 0.5 * cos((1 + pos.z) * time * 5)) * 0.01;
    
    // Iteratively add complexity based on time
    int iterations = int(mod(time, 200.0)); // Change 10 to adjust how quickly new waves are added
    for(int i = 1; i <= iterations; i++)
    {
        float frequency = float(i) * 0.1; // Adjust frequency scaling as desired
        float phase = sin(time * float(i) * 0.5); // Phase changes with time
        wave += sin(pos.x * frequency + phase) * cos(pos.z * frequency + phase) * 0.01; // Adjust amplitude scaling as desired
    }
    
    pos.y += wave;
    
    gl_Position = projection * view * vec4(pos + offset, 1.0);
    vertexColor = color;
    }
    """
    return compileShader(vertex_src, GL_VERTEX_SHADER)


def fragment_shader():
    fragment_src = """
    #version 330
    in vec4 vertexColor;
    out vec4 outColor;
    void main()
    {
        outColor = vertexColor;
    }
    """
    return compileShader(fragment_src, GL_FRAGMENT_SHADER)


# ------------------------------ Mesh Functions --------------------------------------
def create_program():
    shader = compileProgram(vertex_shader(), fragment_shader())
    return shader


def gen_grid_instanced(rows, cols, instances_per_row, instances_per_col, instance_spacing):
    vertices = []
    indices = []
    num_instances = instances_per_row * instances_per_col

    # Base grid vertices and indices generation
    for i in range(rows + 1):
        for j in range(cols + 1):
            x = (j / cols) - 0.5  # Centering the grid
            z = (i / rows) - 0.5  # Centering the grid
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

    # Generate colors for each instance
    colors = np.random.rand(num_instances, 4).astype(np.float32)  # RGBA
    #acolor = np.array([0.0, 0.8, 0.9, 1.0], dtype=np.float32)
    #colors = np.tile(acolor, (num_instances, 1))

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
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    glClearColor(0.3, 0.3, 0.3, 0.1)

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

    running = True
    frame_count = 0
    zero_time = glfw.get_time()
    while running:

        start_time = glfw.get_time()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUniform1f(time_location, start_time * 5)

        move_camera()
        view = cam.get_view_matrix()
        glUniformMatrix4fv(view_location, 1, GL_FALSE, view)

        glBindVertexArray(vao)
        glDrawElementsInstanced(GL_TRIANGLES, indices_count, GL_UNSIGNED_INT, None, instances_area)

        # Reset frame
        glfw.swap_buffers(gwindow)
        frame_count, zero_time, fps = handle_events(frame_count, zero_time, start_time, gwindow)
        glfw.poll_events()

        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.terminate()
            clear_frame(vao)
            exit()
        frame_count += 1



    # Cleanup
    glfw.terminate()



# Starting script
if __name__ == "__main__":
    start_t = time.time()
    window = initialize_glfw(RESOLUTION)
    main_game(window)


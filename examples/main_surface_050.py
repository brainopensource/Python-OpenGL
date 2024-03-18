import ctypes
from math import sin, pi
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
    glDrawElementsInstanced, glUniform3f
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

SIZE = 1
INSTANCE_COUNT = SIZE * SIZE
level_of_detail = 400

WHITE_COLOR = (1.0, 1.0, 1.0)
BLACK_COLOR = (0.0, 0.0, 0.0)
ASPECT_RATIO = RESOLUTION[0]/RESOLUTION[1]

cam = Camera()
NEAR_PROJ = 0.1
FAR_PROJ = 1000
SWAP_INTERVAL = 1
PLAYER_SPEED = 0.3

last_x, last_y = RESOLUTION[0]/2, RESOLUTION[1]/2
first_mouse = True
left, right, forward, backward, up, down = False, False, False, False, False, False


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
    #version 330 core
    
    layout(location = 0) in vec3 position;
    layout(location = 1) in vec3 normal;
    layout(location = 2) in vec3 instancePos;
    
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    
    out vec3 FragPos;
    out vec3 Normal;
    
    void main() {
        FragPos = vec3(model * vec4(position, 1.0)) + instancePos;
        Normal = mat3(transpose(inverse(model))) * normal;
        gl_Position = projection * view * model * vec4(position, 1.0);
    }
    """
    return compileShader(vertex_src, GL_VERTEX_SHADER)


def fragment_shader():
    fragment_src = """
    #version 330 core
    
    out vec4 FragColor;
    in vec3 Normal;
    in vec3 FragPos;
    
    uniform vec3 lightPos; // Position of light source
    uniform vec3 viewPos; // Position of camera view
    uniform vec3 lightColor; // Color of the light
    
    void main() {
        // Ambient
        float ambientStrength = 0.1;
        vec3 ambient = ambientStrength * lightColor;
        
        // Diffuse
        vec3 norm = normalize(Normal);
        vec3 lightDir = normalize(lightPos - FragPos);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diff * lightColor;
        
        // Specular
        float specularStrength = 0.4;
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 reflectDir = reflect(-lightDir, norm);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
        vec3 specular = specularStrength * spec * lightColor;
        
        vec3 result = (ambient + diffuse + specular) * vec3(0.0, 0.6, 0.7);
        FragColor = vec4(result, 1.0);
    }
    """
    return compileShader(fragment_src, GL_FRAGMENT_SHADER)


# ------------------------------ Mesh Functions --------------------------------------
def create_program():
    shader = compileProgram(vertex_shader(), fragment_shader())
    return shader


def gen_surface_mesh(x_segments, z_segments, x_interval, z_interval):
    vertices = []
    indices = []

    for i in range(x_segments + 1):
        for j in range(z_segments + 1):
            x = x_interval[0] + (i / x_segments) * (x_interval[1] - x_interval[0])
            z = z_interval[0] + (j / z_segments) * (z_interval[1] - z_interval[0])
            y = 0.3 * (sin(x + cos(z)) * sin(z + cos(x))) + sin(x/2)*0.5 + sin(z/2)
            vertices.extend([x, y, z])

    # Generating indices for the vertex grid
    for i in range(x_segments):
        for j in range(z_segments):
            start = i * (z_segments + 1) + j
            indices.extend([start, start + 1, start + z_segments + 1,
                            start + 1, start + z_segments + 2, start + z_segments + 1])

    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)


def create_buffers(vertices, indices):
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    ebo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * vertices.itemsize, None)
    glEnableVertexAttribArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)
    return vao, vbo, ebo


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
    glDrawElements(GL_TRIANGLES, len_idx, GL_UNSIGNED_INT, None, INSTANCE_COUNT)
    glBindVertexArray(0)


def clear_frame(vao, vbo, ebo, instance_vbo=None):
    glDeleteVertexArrays(1, [vao])
    glDeleteBuffers(1, [vbo])
    glDeleteBuffers(1, [ebo])
    if instance_vbo:
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

    # Light config
    light_pos = [50.0, 5.0, 30.0]  # Example light position
    view_pos = [50.0, 5.0, 30.0]  # Camera/view position, adjust as necessary
    light_color = [1.0, 1.0, 1.0]  # White light

    # Set light and view position uniform
    lightPosLoc = glGetUniformLocation(shader, "lightPos")
    viewPosLoc = glGetUniformLocation(shader, "viewPos")
    lightColorLoc = glGetUniformLocation(shader, "lightColor")
    glUniform3f(lightPosLoc, *light_pos)
    glUniform3f(viewPosLoc, *view_pos)
    glUniform3f(lightColorLoc, *light_color)

    model, model_loc, proj_loc, view_loc = config_views(shader)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)

    distanciamento = (2*pi) * 8
    vertices, indices = gen_surface_mesh(level_of_detail, level_of_detail,
                                         [0, distanciamento], [0, distanciamento])

    print(vertices)

    len_idx = len(indices)
    vao, vbo, ebo = create_buffers(vertices, indices)

    print('Start render', glfw.get_time() - zero_time)
    running = True
    frame_count = 0
    zero_time = glfw.get_time()

    while running:
        # Get Events
        start_time = glfw.get_time()
        glfw.poll_events()

        # Update camera
        view = cam.get_view_matrix()
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
        move_camera()
        # Update the light position to the current camera position
        #glUniform3f(lightPosLoc, *cam.camera_pos)
        #glUniform3f(viewPosLoc, *cam.camera_pos)

        # Clear old buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glfw.set_window_title(gwindow, f"XYZ {[round(i,2) for i in cam.camera_pos]}")
        render_instance(vao, len_idx)

        # Reset frame
        glfw.swap_buffers(gwindow)
        frame_count, zero_time, fps = handle_events(frame_count, zero_time, start_time, gwindow)

        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            clear_frame(vao, vbo, ebo)
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


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
    glDrawElementsInstanced, glUniform3f, GL_DYNAMIC_DRAW, glUniform1f, glDeleteProgram, GL_CULL_FACE
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
SWAP_INTERVAL = 0
PLAYER_SPEED = 0.6


GRID_DENSITY = 50
SURFACE_ROWS = GRID_DENSITY
SURFACE_COLS = GRID_DENSITY
GRID_ROWS = 2
GRID_COLS = int(1e2)
INSTANCE_AREA = GRID_ROWS * GRID_COLS
GRID_SPACING = (2*pi) * 2

DRAW_POLYS = 1

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
# Class to manage different shaders and programs for a complete game and scene
class ShaderManager:
    def __init__(self, vertex_path, fragment_path):
        self.vertex_path = './shaders/' + vertex_path
        self.fragment_path = './shaders/' + fragment_path
        self.shader = self.create_program()


    def create_vertex_shader(self):
        with open(self.vertex_path, 'r') as file:
            vertex_shader = file.read()

        shader = compileShader(vertex_shader, GL_VERTEX_SHADER)
        return shader


    def create_fragment_shader(self):
        with open(self.fragment_path, 'r') as file:
            fragment_shader = file.read()
        shader = compileShader(fragment_shader, GL_FRAGMENT_SHADER)
        return shader


    def create_program(self):
        return compileProgram(self.create_vertex_shader(), self.create_fragment_shader())


    def load(self):
        glUseProgram(self.shader)


    def delete(self):
        glDeleteProgram(self.shader)


# ------------------------------ Mesh Functions --------------------------------------
# Crete a class for generating the vertices, offsets, colors and indices for the sphere
class ObjectManager:
    def __init__(self, vertices, indices, grid_rows, grid_cols, grid_spacing, base_color, shader_program):
        self.vertices = vertices
        self.indices = indices
        self.indices_count = len(indices)
        self.instances_per_row = grid_rows
        self.instances_per_col = grid_cols
        self.instance_area = self.instances_per_row * self.instances_per_col
        self.spacing = grid_spacing
        self.vao = None
        self.base_color = base_color
        self.shader_program = shader_program
        self.offsets = self.generate_offsets()
        self.colors = self.generate_colors()
        self.uniform_locs = self.gen_uniforms()


    def generate_offsets(self):
        offsets = []
        for i in range(self.instances_per_row):
            for j in range(self.instances_per_col):
                x_offset = (i - self.instances_per_row / 2) * self.spacing
                y_offset = 0  # Assuming you want them on the same plane, adjust as needed
                z_offset = (j - self.instances_per_col / 2) * self.spacing
                offsets.append([x_offset, y_offset, z_offset])
        return np.array(offsets, dtype=np.float32)


    def generate_colors(self):
        return np.full((self.instance_area, 4), self.base_color, dtype=np.float32)
        #return np.random.rand(self.instance_area, 4).astype(np.float32)


    def gen_uniforms(self):
        locations = {
            "projection": glGetUniformLocation(self.shader_program, "projection"),
            "view": glGetUniformLocation(self.shader_program, "view"),
            "time": glGetUniformLocation(self.shader_program, "time"),
            }
        return locations
    

    def create_buffers(self):
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)

        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

        offset_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, offset_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.offsets.nbytes, self.offsets, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)
        glVertexAttribDivisor(1, 1)

        # Color VBO
        color_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, color_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.colors.nbytes, self.colors, GL_STATIC_DRAW)
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(2)
        glVertexAttribDivisor(2, 1)
        self.vao = vao


    def delete_buffers(self):
        glDeleteVertexArrays(1, [self.vao])


def create_sphere_vertices(latitudes, longitudes):
    vertices = []
    for i in range(latitudes + 1):
        theta = i * np.pi / latitudes
        sinTheta = np.sin(theta)
        cosTheta = np.cos(theta)

        for j in range(longitudes + 1):
            phi = j * 2 * np.pi / longitudes
            sinPhi = np.sin(phi)
            cosPhi = np.cos(phi)

            x = cosPhi * sinTheta
            y = cosTheta
            z = sinPhi * sinTheta
            vertices.extend([x, y, z])

    indices = []
    for i in range(latitudes):
        for j in range(longitudes):
            first = (i * (longitudes + 1)) + j
            second = first + longitudes + 1

            indices.extend([first, second, first + 1, second, second + 1, first + 1])

    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)



# ---------------------------       MAIN CODE  -------------------------------------------------------------------
def main_game(gwindow):

    glfw.set_cursor_pos_callback(gwindow, mouse_look_callback)
    glfw.set_key_callback(gwindow, keyboard_callback)
    glfw.set_input_mode(gwindow, glfw.CURSOR, glfw.CURSOR_DISABLED)
    glfw.swap_interval(SWAP_INTERVAL)
    glEnable(GL_DEPTH_TEST)  # GL_MULTISAMPLE, GL_BLEND
    #glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    if DRAW_POLYS == True:
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    glClearColor(0.0, 0.1, 0.2, 1.0)

    # Manage Shaders
    shaders_list = []
    sphere_shader = ShaderManager("vertex_sph102.glsl", "fragment_sph102.glsl")
    shpere_program = sphere_shader.shader
    sphere_shader.load()
    #sphere_shader_2 = ShaderManager("vertex_sph102.glsl", "fragment_sph102.glsl")
    #shpere_program_2 = sphere_shader.shader
    #sphere_shader_2.load()
    shaders_list.extend([sphere_shader])


    # Create Objects
    objects_list = []

    vertices_sph, indices_sph = create_sphere_vertices(SURFACE_ROWS, SURFACE_COLS)
    instanced_spheres = ObjectManager(vertices_sph, indices_sph, GRID_ROWS, GRID_COLS, GRID_SPACING,
                                       [1.0, 0.0, 0.0, 1.0], shpere_program)
    instanced_spheres.create_buffers()

    vertices_sph2, indices_sph2 = create_sphere_vertices(SURFACE_ROWS, SURFACE_COLS)
    instanced_spheres2 = ObjectManager(vertices_sph2, indices_sph2, GRID_COLS, GRID_ROWS, GRID_SPACING,
                                        [1.0, 0.0, 1.0, 1.0], shpere_program)
    instanced_spheres2.create_buffers()

    objects_list.extend([instanced_spheres])
    objects_list.extend([instanced_spheres2])

    # Set up projection matrix
    projection = matrix44.create_perspective_projection_matrix(60.0, RESOLUTION_RATIO, NEAR_PROJ, FAR_PROJ)
    # Get camera view matrix
    view = cam.get_view_matrix()

    # Set up uniforms time projection and view

    for object in objects_list:
        glUniformMatrix4fv(object.uniform_locs['projection'], 1, GL_FALSE, projection)
        glUniformMatrix4fv(object.uniform_locs['view'], 1, GL_FALSE, view)

    
    #time_location = glGetUniformLocation(shpere_program, "time")
    #proj_location = glGetUniformLocation(shpere_program, "projection")
    #view_location = glGetUniformLocation(shpere_program, "view")
    #glUniformMatrix4fv(proj_location, 1, GL_FALSE, projection)
    #glUniformMatrix4fv(view_location, 1, GL_FALSE, view)

    running = True
    frame_count = 0
    zero_time = glfw.get_time()

    while running:

        start_time = glfw.get_time()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        move_camera()
        view = cam.get_view_matrix()
        glUniformMatrix4fv(objects_list[0].uniform_locs['view'], 1, GL_FALSE, view)

        for i in range(len(objects_list)):
            glUniform1f(objects_list[i].uniform_locs['time'], start_time)
            # Render objects
            glBindVertexArray(objects_list[i].vao)
            glDrawElementsInstanced(GL_TRIANGLES, objects_list[i].indices_count, GL_UNSIGNED_INT, None, INSTANCE_AREA)


        # Reset frame
        glfw.swap_buffers(gwindow)
        frame_count, zero_time, fps = handle_events(frame_count, zero_time, start_time, gwindow)
        glfw.poll_events()

        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            [shader.delete() for shader in shaders_list]
            [object.delete_buffers() for object in objects_list]
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


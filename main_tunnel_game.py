from math import sin, cos, pi
from pyrr import Vector3, matrix44
import random
from sys import exit
from utils import *
from glfw_initialize2 import *
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


# ------------------------------ Game Controls -------------------------------------------------------------------
# Create a class to handle game controls
class GameControls:
    def __init__(self, gamewindow, width, height, GameCamera, player_speed):
        self.camera = GameCamera
        self.left = self.right = self.forward = self.backward = self.up = self.down = False
        self.player_speed = player_speed
        self.last_x = width * 0.5
        self.last_y = height * 0.5
        self.window = gamewindow
        glfw.set_cursor_pos_callback(gamewindow, self.mouse_look_callback)
        glfw.set_key_callback(gamewindow, self.keyboard_callback)
        glfw.set_input_mode(gamewindow, glfw.CURSOR, glfw.CURSOR_DISABLED)

    def mouse_look_callback(self, window, x, y):
        self.camera.process_mouse_movements((x - self.last_x) / 2, (self.last_y - y) / 2)
        self.last_x = x
        self.last_y = y

    def get_view(self):
        return self.camera.get_view_matrix()

    def move_camera(self):
        if self.left:
            self.camera.process_keyboard("LEFT", self.player_speed)
        if self.right:
            self.camera.process_keyboard("RIGHT", self.player_speed)
        if self.forward:
            self.camera.process_keyboard("FORWARD", self.player_speed)
        if self.backward:
            self.camera.process_keyboard("BACKWARD", self.player_speed)
        if self.up:
            self.camera.process_keyboard("UP", self.player_speed)
        if self.down:
            self.camera.process_keyboard("DOWN", self.player_speed)

    def keyboard_callback(self, window, key, scancode, action, mode):
        if key == glfw.KEY_W and action == glfw.PRESS:
            self.forward = True
        elif key == glfw.KEY_W and action == glfw.RELEASE:
            self.forward = False
        if key == glfw.KEY_S and action == glfw.PRESS:
            self.backward = True
        elif key == glfw.KEY_S and action == glfw.RELEASE:
            self.backward = False
        if key == glfw.KEY_A and action == glfw.PRESS:
            self.left = True
        elif key == glfw.KEY_A and action == glfw.RELEASE:
            self.left = False
        if key == glfw.KEY_D and action == glfw.PRESS:
            self.right = True
        elif key == glfw.KEY_D and action == glfw.RELEASE:
            self.right = False
        if key == glfw.KEY_Q and action == glfw.PRESS:
            self.up = True
        elif key == glfw.KEY_Q and action == glfw.RELEASE:
            self.up = False
        if key == glfw.KEY_E and action == glfw.PRESS:
            self.down = True
        elif key == glfw.KEY_E and action == glfw.RELEASE:
            self.down = False


#Create a class to make the first configurations in glfw
class GameWindow:
    def __init__(self, interval, poly_test):
        self.interval = interval
        self.polygons = poly_test
        glfw.swap_interval(self.interval)
        glEnable(GL_DEPTH_TEST)
        if self.polygons:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

    @staticmethod
    def set_clear_color(color):
        glClearColor(color[0], color[1], color[2], color[3])


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
# Class to handle loaded objects from Blender
class MeshLoader:
    def __init__(self, filename):
        self.vertices = []
        self.indices = []
        self.load_obj(filename)

    def load_obj(self, filename):
        with open(filename, 'r') as file:
            for line in file:
                if line.startswith('v '):  # Vertex position
                    parts = line.split()
                    self.vertices.extend([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith('f '):  # Face
                    parts = line.split()
                    # Assuming the OBJ file uses 1-based indexing
                    idx = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                    self.indices.extend(idx)

    def get_vertices(self):
        return self.vertices

    def get_indices(self):
        return self.indices


# Crete a class for generating the vertices, offsets, colors and indices for the sphere
class ObjectManager:
    def __init__(self, vertices, indices, normals, grid_rows, grid_cols, grid_spacing, base_color, shader_program):
        self.vertices = vertices
        self.indices = indices
        self.normals = normals
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

        if self.normals is not None:
            # VBO for normals
            normal_vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, normal_vbo)
            glBufferData(GL_ARRAY_BUFFER, self.normals.nbytes, self.normals, GL_STATIC_DRAW)
            glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 0, None)  # Assuming location 3 for normals
            glEnableVertexAttribArray(3)


    def delete_buffers(self):
        glDeleteVertexArrays(1, [self.vao])


# Create a class to handle the creation of the geometries
class GeometryGenerator:
    def __init__(self, rows, cols):
        self.grid_rows = rows
        self.grid_cols = cols

    def create_sphere_vertices(self):
        vertices = []
        normals = []
        for i in range(self.grid_rows + 1):
            theta = i * np.pi / self.grid_rows
            sinTheta = sin(theta)
            cosTheta = cos(theta)
            for j in range(self.grid_cols + 1):
                phi = j * 2 * np.pi / self.grid_cols
                sinPhi = sin(phi)
                cosPhi = cos(phi)
                x = cosPhi * sinTheta * 5
                y = cosTheta
                z = sinPhi * sinTheta * 3
                vertices.extend([x, y, z])
                normals.extend([0.5, 1.0, 0.5])

        indices = []
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                first = (i * (self.grid_cols + 1)) + j
                second = first + self.grid_cols + 1

                indices.extend([first, second, first + 1, second, second + 1, first + 1])

        return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32), np.array(normals, dtype=np.float32)


# ---------------------------       MAIN CODE  -------------------------------------------------------------------
# Transforming your main_game function into a clas, enhances modularity, improves code organization, makes your game more scalable, and simplifies maintenance and debugging. By encapsulating game-related functionality within a class, you can also more easily manage game state, share common resources, and extend game features.
class GameMain:
    def __init__(self):
        self.window = initialize_glfw(RESOLUTION)
        self.controls = GameControls(self.window, RESOLUTION[0], RESOLUTION[1], Camera(), PLAYER_SPEED)
        self.game_manager = GameWindow(SWAP_INTERVAL, DRAW_POLYS)
        self.game_manager.set_clear_color((0.0, 0.1, 0.2, 1.0))

        self.shaders_list = self.setup_shaders("vertex_sph103.glsl", "fragment_sph103.glsl")
        self.objects_list = self.create_objects()

        self.projection = matrix44.create_perspective_projection_matrix(60.0, RESOLUTION_RATIO, NEAR_PROJ, FAR_PROJ)
        self.view = self.controls.get_view()
        self.setup_uniforms()

        self.running = True
        self.frame_count = 0
        self.zero_time = glfw.get_time()

    @staticmethod
    def setup_shaders(vertex, fragment):
        shaders_list = []
        sphere_shader = ShaderManager(vertex, fragment)
        sphere_shader.load()
        shaders_list.append(sphere_shader)
        return shaders_list

    def create_objects(self):
        objects_list = []
        geometries = GeometryGenerator(SURFACE_ROWS, SURFACE_COLS)
        vertices_sph, indices_sph, normals_sph = geometries.create_sphere_vertices()

        instanced_spheres = ObjectManager(vertices_sph, indices_sph, normals_sph, GRID_ROWS, GRID_COLS,
                                          GRID_SPACING, [1.0, 1.0, 1.0, 1.0], self.shaders_list[0].shader)
        instanced_spheres.create_buffers()

        vertices_sph2, indices_sph2, normals_sph2 = geometries.create_sphere_vertices()
        instanced_spheres2 = ObjectManager(vertices_sph2, indices_sph2, normals_sph2, GRID_COLS, GRID_ROWS,
                                           GRID_SPACING, [1.0, 1.0, 1.0, 1.0], self.shaders_list[0].shader)
        instanced_spheres2.create_buffers()

        objects_list.extend([instanced_spheres, instanced_spheres2])
        return objects_list

    def setup_uniforms(self):
        for obj in self.objects_list:
            glUniformMatrix4fv(obj.uniform_locs['projection'], 1, GL_FALSE, self.projection)
            glUniformMatrix4fv(obj.uniform_locs['view'], 1, GL_FALSE, self.view)

    def run(self):
        while self.running:
            start_time = glfw.get_time()
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self.controls.move_camera()
            self.view = self.controls.get_view()
            glUniformMatrix4fv(self.objects_list[0].uniform_locs['view'], 1, GL_FALSE, self.view)

            for obj in self.objects_list:
                glUniform1f(obj.uniform_locs['time'], start_time)
                glBindVertexArray(obj.vao)
                glDrawElementsInstanced(GL_TRIANGLES, obj.indices_count, GL_UNSIGNED_INT, None, INSTANCE_AREA)

            glfw.swap_buffers(self.window)
            self.frame_count, self.zero_time, fps = handle_events(self.frame_count, self.zero_time, start_time, self.window)
            glfw.poll_events()

            if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
                self.terminate()

            self.frame_count += 1

    def terminate(self):
        [shader.delete() for shader in self.shaders_list]
        [obj.delete_buffers() for obj in self.objects_list]
        glfw.terminate()
        self.running = False


# Starting script
if __name__ == "__main__":
    game = GameMain()
    game.run()
    exit()

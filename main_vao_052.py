import ctypes
import random
from utils import *
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


# ------------------------------- GLOBAL VARIABLES --------------------------------------------------------------------
WHITE_COLOR = (1.0, 1.0, 1.0)
BLACK_COLOR = (0.0, 0.0, 0.0)
ASPECT_RATIO = RESOLUTION[0]/RESOLUTION[1]

NUM_INST = int(25**3 * 1e1)   # 1kk cubes 165hz
PENDULUM_AMP = 1.0
PENDULUM_FREQ = np.pi/2

NEAR_PROJ = 0.1
FAR_PROJ = 10000


#  ------------------------------  Mesh Functions ---------------------------------------------------------------------
def vertex_shader():
    vertex_src = """
    # version 330
    
    layout(location = 0) in vec3 a_position;
    layout(location = 1) in vec3 a_color;

    uniform mat4 model;
    uniform mat4 projection;
    uniform mat4 view;
    
    out vec3 v_color;
    
    void main()
    {
        gl_Position = projection * view * model * vec4(a_position, 1.0);
        v_color = a_color;
    }
    """
    return compileShader(vertex_src, GL_VERTEX_SHADER)


def fragment_shader():
    fragment_src = """
    # version 330

    in vec3 v_color;
    out vec4 out_color;

    void main()
    {
        out_color = vec4(v_color, 1.0);
    }
    """
    return compileShader(fragment_src, GL_FRAGMENT_SHADER)


def create_program():
    shader = compileProgram(vertex_shader(), fragment_shader())
    return shader


def mesh_cube_vertices_gay():
    vertices = [
        # Front face
        -1.0, -1.0, 0.1, 0.9, 0.0, 0.9,
        1.0, -1.0, 0.1, 0.0, 0.8, 0.8,
        1.0, 1.0, 0.1, 0.8, 0.0, 0.8,
        -1.0, 1.0, 0.1, 0.5, 0.8, 0.8,
        # Back face
        -1.0, -1.0, -0.1, 0.6, 0.0, 0.6,
        1.0, -1.0, -0.1, 0.0, 1.0, 1.0,
        1.0, 1.0, -0.1, 0.9, 0.0, 0.9,
        -1.0, 1.0, -0.1, 0.0, 0.5, 0.5,
        # Top face
        -1.0, 1.0, 0.1, 0.5, 0.8, 0.8,
        1.0, 1.0, 0.1, 0.8, 0.0, 0.8,
        1.0, 1.0, -0.1, 0.0, 0.5, 0.5,
        -1.0, 1.0, -0.1, 0.9, 0.0, 0.9,
        # Bottom face
        -1.0, -1.0, 0.1, 0.0, 0.8, 0.8,
        1.0, -1.0, 0.1, 0.8, 0.0, 0.8,
        1.0, -1.0, -0.1, 0.9, 0.0, 0.9,
        -1.0, -1.0, -0.1, 0.0, 0.5, 0.5,
        # Left face
        -1.0, -1.0, 0.1, 0.6, 0.0, 0.6,
        -1.0, -1.0, -0.1, 0.0, 1.0, 1.0,
        -1.0, 1.0, -0.1, 0.9, 0.0, 0.9,
        -1.0, 1.0, 0.1, 0.0, 0.5, 0.5,
        # Right face
        1.0, -1.0, 0.1, 0.9, 0.0, 0.9,
        1.0, -1.0, -0.1, 0.0, 0.8, 0.8,
        1.0, 1.0, -0.1, 0.8, 0.0, 0.8,
        1.0, 1.0, 0.1, 0.5, 0.8, 0.8,
        ]
    return np.array(vertices, dtype=np.float32)


def mesh_cube_vertices():
    vertices = [
        # Front face
        -1.0, -1.0, 0.1, 0.9, 0.9, 0.9,
        1.0, -1.0, 0.1, 0.8, 0.8, 0.8,
        1.0, 1.0, 0.1, 0.8, 0.8, 0.8,
        -1.0, 1.0, 0.1, 0.5, 0.8, 0.8,
        # Back face
        -1.0, -1.0, -0.1, 0.6, 0.6, 0.6,
        1.0, -1.0, -0.1, 1.0, 1.0, 1.0,
        1.0, 1.0, -0.1, 0.9, 0.9, 0.9,
        -1.0, 1.0, -0.1, 0.5, 0.5, 0.5,
        # Top face
        -1.0, 1.0, 0.1, 0.5, 0.8, 0.8,
        1.0, 1.0, 0.1, 0.8, 0.8, 0.8,
        1.0, 1.0, -0.1, 0.5, 0.5, 0.5,
        -1.0, 1.0, -0.1, 0.9, 0.9, 0.9,
        # Bottom face
        -1.0, -1.0, 0.1, 0.8, 0.8, 0.8,
        1.0, -1.0, 0.1, 0.8, 0.8, 0.8,
        1.0, -1.0, -0.1, 0.9, 0.9, 0.9,
        -1.0, -1.0, -0.1, 0.5, 0.5, 0.5,
        # Left face
        -1.0, -1.0, 0.1, 0.6, 0.6, 0.6,
        -1.0, -1.0, -0.1, 1.0, 1.0, 1.0,
        -1.0, 1.0, -0.1, 0.9, 0.9, 0.9,
        -1.0, 1.0, 0.1, 0.5, 0.5, 0.5,
        # Right face
        1.0, -1.0, 0.1, 0.9, 0.9, 0.9,
        1.0, -1.0, -0.1, 0.8, 0.8, 0.8,
        1.0, 1.0, -0.1, 0.8, 0.8, 0.8,
        1.0, 1.0, 0.1, 0.5, 0.8, 0.8,
        ]
    return np.array(vertices, dtype=np.float32)


def mesh_n_cube_generation(n):

    s_t = time.time()
    all_vertices = []
    vertice_base = mesh_cube_vertices()
    total = int(np.cbrt(n))
    grid_lengh = total / 2
    step = 2.5
    count = 0
    for z in np.arange(0, step * grid_lengh, step):
        for x in np.arange(-step * grid_lengh, step * grid_lengh, step):
            for y in np.arange(-step * grid_lengh, step * grid_lengh, step):
                # Create a new array for the current cube's vertices
                vertices = np.array(vertice_base, copy=True)
                vertices[0::6] += x  # Apply x offset
                vertices[1::6] += y  # Apply y offset
                vertices[2::6] += (z * 0.8)  # Apply y offset
                all_vertices.append(vertices)
                count += 1

    # Flatten the list of all vertices arrays into a single array
    all_vertices_array = np.concatenate(all_vertices)
    return all_vertices_array


def mesh_n_cube_generation_pl(n):
    print('mesh')
    grid_size = int(np.sqrt(n))
    s_t = time.time()
    step = 2.5
    V = 24
    # Convert the NumPy array to a dictionary then to a Polars DataFrame
    vertices_array = mesh_cube_vertices()
    vertices_dict = {
        'x': vertices_array[:, 0],
        'y': vertices_array[:, 1],
        'z': vertices_array[:, 2],
        'r': vertices_array[:, 3],
        'g': vertices_array[:, 4],
        'b': vertices_array[:, 5],
    }
    df = pl.DataFrame(vertices_dict)
    df_expanded = pl.concat([df] * n)

    # Generate grid offsets
    x_offsets = np.repeat(np.arange(grid_size), grid_size) * step
    y_offsets = np.tile(np.arange(grid_size), grid_size) * step
    # Ensure the offsets are repeated for each vertex in a cube
    x_offsets_repeated = np.repeat(x_offsets, V)
    y_offsets_repeated = np.repeat(y_offsets, V)
    # Replicate the initial dataframe N times
    df_expanded = pl.concat([df] * n)
    # Apply the offsets
    df_expanded = df_expanded.with_columns([
        pl.col('x') + pl.lit(x_offsets_repeated).alias('x_offset'),
        pl.col('y') + pl.lit(y_offsets_repeated).alias('y_offset')
    ])

    # Ensure all columns are float32
    #for col in df_expanded.columns:
    #    df_expanded = df_expanded.with_column(pl.col(col).cast(pl.Float32))

    print(df_expanded)
    numpy_array = df_expanded.to_numpy().astype(np.float32).flatten()
    return numpy_array


def mesh_cube_indexes():
    cube_indices = [0, 1, 2, 2, 3, 0,
                    4, 5, 6, 6, 7, 4,
                    8, 9, 10, 10, 11, 8,
                    12, 13, 14, 14, 15, 12,
                    16, 17, 18, 18, 19, 16,
                    20, 21, 22, 22, 23, 20]
    return np.array(cube_indices, dtype=np.uint32)


@time_record
def mesh_n_cube_indexes(num_cubes):
    base_indices = mesh_cube_indexes()
    cube_indices = []
    vertices_per_cube = 24

    for i in range(num_cubes):
        offset = i * vertices_per_cube
        cube_indices.extend([idx + offset for idx in base_indices])

    return np.array(cube_indices, dtype=np.uint32)


def create_vbo_buff(vertices):
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)


def create_vao_buff():
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    return vao


def create_ebo_buff(indices):
    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)


def create_attrib_array(vertices, id_array, index, size, b_pointer):
    # Create Pointers for Cube Attributes
    glEnableVertexAttribArray(id_array)
    glVertexAttribPointer(index, size, GL_FLOAT, GL_FALSE, vertices.itemsize * 6, b_pointer)


@time_record
def initialize_object(cube_vertices, cube_indices):

    # Create Shader and define position and color
    shader = create_program()
    a_position_loc = glGetAttribLocation(shader, "a_position")
    a_color_loc = glGetAttribLocation(shader, "a_color")

    # Cube VAO, VBO and EBO
    vao = create_vao_buff()
    create_vbo_buff(cube_vertices)
    create_ebo_buff(cube_indices)

    # Attributes pointers positions and colors
    create_attrib_array(cube_vertices, a_position_loc, 0, 3, ctypes.c_void_p(0))
    create_attrib_array(cube_vertices, a_color_loc, 1, 3, ctypes.c_void_p(12))
    glUseProgram(shader)

    # Config Background standard color
    glClearColor(0.0, 0.2, 0.3, 1)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    return shader, vao


# ---------------------------       MAIN CODE  -------------------------------------------------------------------
def main_game(awindow):

    # Create mesh
    cubes = NUM_INST
    cube_indexes = mesh_n_cube_indexes(cubes)
    shader, vao = initialize_object(mesh_n_cube_generation(cubes), cube_indexes)

    # World view models
    cube_pos = pyrr.matrix44.create_from_translation(pyrr.Vector3([-0, -0, -50]))
    projection = pyrr.matrix44.create_perspective_projection_matrix(45, ASPECT_RATIO, NEAR_PROJ, FAR_PROJ)
    view = pyrr.matrix44.create_look_at(pyrr.Vector3([0, 1, 4]), pyrr.Vector3([0, 0, 0]), pyrr.Vector3([0, 1, 0]))
    model_loc = glGetUniformLocation(shader, "model")
    proj_loc = glGetUniformLocation(shader, "projection")
    view_loc = glGetUniformLocation(shader, "view")
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)

    # Game Loop
    frame_count = 0
    last_time = glfw.get_time()
    running = True
    slide_z = -50
    slide_x = 0
    slide_y = 0

    print('sleeping and loading')
    time.sleep(1)

    while running:
        current_time = glfw.get_time()
        glfw.poll_events()  # Check for user interactions
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # Reset the screen
        frame_count, last_time, fps = display_fps(frame_count, last_time, current_time, awindow)

        if 4 > current_time > 2:
            slide_x = slide_x + np.tanh(current_time/2 - 3) / 1000
            slide_y = slide_y - np.tanh(current_time/2 - 4) / 500
            slide_z = slide_z - np.tanh(current_time/2 - 4) / 500
            cube_pos = pyrr.matrix44.create_from_translation(pyrr.Vector3([slide_x, slide_y, slide_z]))

        elif 8 > current_time > 4:
            slide_y = slide_y - 0.08/(current_time)
            slide_z = (slide_z - np.tanh(current_time/5)/50)
            cube_pos = pyrr.matrix44.create_from_translation(pyrr.Vector3([0, slide_y, slide_z]))

        elif 18 > current_time > 8:
            slide_z = ((current_time-12)**2 - 20) * 5
            slide_x = slide_x + 0.005
            #slide_y = slide_y - 10
            cube_pos = pyrr.matrix44.create_from_translation(pyrr.Vector3([slide_x, slide_y, slide_z]))

        elif 19 > current_time > 18:
            slide_z = slide_z - ((current_time-12)**2 - 20) / 30
            slide_x = slide_x - 0.005
            cube_pos = pyrr.matrix44.create_from_translation(pyrr.Vector3([slide_x, slide_y, slide_z]))

        elif current_time > 19:
            slide_z = slide_z + ((current_time-12)**2 - 20) / 20000
            cube_pos = pyrr.matrix44.create_from_translation(pyrr.Vector3([slide_x, slide_y, slide_z]))



        acceleration = np.tanh(current_time/100) * 10
        rot_x = pyrr.Matrix44.from_x_rotation(acceleration)
        rot_y = pyrr.Matrix44.from_y_rotation(-acceleration)
        rot_z = pyrr.Matrix44.from_z_rotation(-acceleration)
        rotation = pyrr.matrix44.multiply(rot_x, rot_y)
        rotation = pyrr.matrix44.multiply(rotation, rot_z)

        model = pyrr.matrix44.multiply(rotation, cube_pos)
        glBindVertexArray(vao)
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
        glDrawElements(GL_TRIANGLES, len(cube_indexes), GL_UNSIGNED_INT, None)

        glfw.swap_buffers(awindow)
        running = glfw_handle_events(awindow)
        frame_count += 1
    glfw.terminate()





# Starting script
if __name__ == "__main__":
    start_t = time.time()
    window = initialize_glfw(RESOLUTION)
    main_game(window)


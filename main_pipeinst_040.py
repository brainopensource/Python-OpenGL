import ctypes
import random

import glfw
from math import sin, cos
import numpy as np
from OpenGL.GL import (
    glClear, glClearColor, glEnableClientState, glLoadIdentity,
    glRotatef, glVertexPointer, GL_COLOR_BUFFER_BIT, GL_FLOAT, GL_TRIANGLES, GL_TRUE, GL_FALSE, GL_VERTEX_ARRAY,
    GL_COLOR_ARRAY, glDrawArrays, glColorPointer, GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, glUseProgram, glGenBuffers,
    glBindBuffer, GL_ARRAY_BUFFER, glBufferData, GL_STATIC_DRAW, glGetAttribLocation, glEnableVertexAttribArray,
    glVertexAttribPointer, GL_TRIANGLE_STRIP, glViewport, GL_ELEMENT_ARRAY_BUFFER, glDrawElements, GL_UNSIGNED_INT,
    glEnable, GL_DEPTH_TEST, glGetUniformLocation, GL_DEPTH_BUFFER_BIT, glUniformMatrix4fv, glPolygonMode,
    GL_FRONT_AND_BACK, GL_LINE, glLineWidth, GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, glBlendFunc
    )
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr


# ------------------------------- GLOBAL VARIABLES --------------------------------------------------------------------
RESOLUTION = [1800, 1000]
MONITOR_1_RESOLUTION = [1920, 1080]
WHITE_COLOR = (1.0, 1.0, 1.0)
BLACK_COLOR = (0.0, 0.0, 0.0)
REFRESH_MODE = 'vsync'

NUM_INST = 10

PENDULUM_AMP = 1.0
PENDULUM_FREQ = np.pi/2

NEAR_PROJ = 0.1
FAR_PROJ = 1000
#  ------------------------------  GLFW Functions
def initialize_glfw(resolution):
    if not glfw.init():
        raise Exception("glfw can not be initialized!")
    glfw.window_hint(glfw.SAMPLES, 0)  # Disable anti-aliasing
    glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)
    glfw_window = glfw.create_window(resolution[0], resolution[1], "True OpenGL 0.0.1", None, None)
    glfw.set_window_pos(glfw_window,
                        int(MONITOR_1_RESOLUTION[0]/2 - resolution[0]/2),
                        int(MONITOR_1_RESOLUTION[1]/2 - resolution[1]/2))
    # set the callback function for window resize
    glfw.set_window_size_callback(glfw_window, window_resize)
    # check if window was created
    glfw_check_terminate(glfw_window)
    # make the context current
    glfw.make_context_current(glfw_window)
    # Disable V-Sync (vertical synchronization)
    glfw.swap_interval(choose_refresh_mode(REFRESH_MODE))

    return glfw_window


def window_resize(window, width, height):
    glViewport(0, 0, width, height)


def glfw_handle_events(glfwwindow):
    if not glfw.get_key(glfwwindow, glfw.KEY_ESCAPE) == glfw.PRESS:
        return True


def glfw_check_terminate(glfwwindow):
    if not glfwwindow:
        glfw.terminate()
        raise Exception("glfw window can not be created!")


def choose_refresh_mode(mode):
    if mode == 'vsync':
        return 1
    elif mode == 'infinite' or 'inf':
        return 0
    elif mode == 'half':
        return 2


def calculate_fps(dt, fcount):
    return fcount / dt

def display_fps(frame_count, last_time, current_time, window_bar):
    delta_time = current_time - last_time
    if delta_time >= 1.0:
        fps = frame_count / delta_time
        glfw.set_window_title(window_bar, f"True OpenGL 0.0.1 - FPS: {fps:.0f}")
        return 0, current_time, fps
    else:
        return frame_count, last_time, None


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


def mesh_cube_vertices():
    """
    x,y,z,r,g,b
    :return: np.array(dtype=np.float32)
    """
    # Example 1: Gradient Blue to Cyan Cube
    vertices = [
        -0.5, -0.5, 0.1, 0.9, 0.0, 0.9,  # Neon Purple
        0.5, -0.5, 0.1, 0.0, 0.8, 0.8,  # Neon Cyan
        0.5, 0.5, 0.1, 0.8, 0.0, 0.8,  # Neon Purple
        -0.5, 0.5, 0.1, 0.5, 0.8, 0.8,  # Neon Cyan

        -0.5, -0.5, -0.1, 0.6, 0.0, 0.6,  # Dark Neon Purple
        0.5, -0.5, -0.1, 0.0, 1.0, 1.0,  # Dark Neon Cyan
        0.5, 0.5, -0.1, 0.9, 0.0, 0.9,  # Dark Neon Purple
        -0.5, 0.5, -0.1, 0.0, 0.5, 0.5]  # Dark Neon Cyan
    return np.array(vertices, dtype=np.float32)


def mesh_cube_indexes():
    indices = [0, 1, 2, 2, 3, 0,
               4, 5, 6, 6, 7, 4,
               4, 5, 1, 1, 0, 4,
               6, 7, 3, 3, 2, 6,
               5, 6, 2, 2, 1, 5,
               7, 4, 0, 0, 3, 7]
    return np.array(indices, dtype=np.uint32)


def create_vbo_buff(vertices):
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)    # 32 = 9 verts * 4 bytes each


def create_ebo_buff(indices):
    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

def initialize_object(vertices, indices):

    shader = create_program()

    create_vbo_buff(vertices)
    create_ebo_buff(indices)

    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))            # 3,0 vertices tight
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))         # 3,0 vertices tight

    glUseProgram(shader)
    # Config Background standard color
    glClearColor(0.0, 0.2, 0.3, 1)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    return shader

def render_cube_instance(cube, rotation, total_indexes, model_loc):
    model = pyrr.matrix44.multiply(rotation, cube)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    glDrawElements(GL_TRIANGLES, total_indexes, GL_UNSIGNED_INT, None)

def create_cubes(total):
    all_cubes = []
    for i in range(total):
        x = random.uniform(-6, 6)
        y = random.uniform(-3.0, 1.0)
        z = random.uniform(-15, -3)
        all_cubes.append(pyrr.matrix44.create_from_translation(pyrr.Vector3([x, y, z])))
    return all_cubes


def oscillate_cube_zposition(original_matrix, time, amplitude, frequency):
    # Extract the translation component from the original matrix
    translation_vector = original_matrix[3, :3]
    # Oscillate the z component of the translation vector
    new_z = translation_vector[2] + amplitude * sin(frequency * time)
    # Create a new translation matrix with the oscillated z position
    new_matrix = original_matrix.copy()
    new_matrix[3][2] = new_z
    return new_matrix

def oscillate_cube_allposition(original_matrix, time, amplitude, frequency):
    # Extract the translation component from the original matrix
    translation_vector = original_matrix[3, :3]
    # Oscillate the x and y components of the translation vector, keep z constant
    new_x = translation_vector[0] + amplitude * sin(frequency * time)
    new_y = translation_vector[1] + amplitude * cos(frequency * time)
    # Create a new translation matrix with the oscillated x and y positions
    new_matrix = original_matrix.copy()
    new_matrix[3][0] = new_x
    new_matrix[3][1] = new_y
    return new_matrix

def oscillate_cube_rposition(original_matrix, time, amplitude, frequency):
    # Extract the original translation components from the matrix
    original_x, original_y, _ = original_matrix[3, :3]

    # Calculate the oscillation for x and y using a sine wave, ensuring it oscillates between -amplitude and +amplitude
    # This simulates a pendulum motion between two points (0 and amplitude) for both x and y
    oscillation_x = amplitude * np.sin(frequency * time)
    oscillation_y = amplitude * np.sin(frequency * time + np.pi / 2)  # Phase shift to vary the motion pattern

    # Apply the oscillation to the original position
    new_x = original_x + oscillation_x
    new_y = original_y + oscillation_y

    # Update the matrix with the new x and y, keeping z constant
    new_matrix = original_matrix.copy()
    new_matrix[3][0] = new_x
    new_matrix[3][1] = new_y

    return new_matrix


def oscillate_cube_position(original_matrix, time, max_angle, frequency):
    # Extract the original translation components from the matrix
    original_x, original_y, original_z = original_matrix[3, :3]
    # Simulate pendulum movement using a sine function, with decreasing amplitude towards the extremes
    # Calculate the current angle of the pendulum swing
    angle = max_angle * np.sin(frequency * time)
    # Calculate the new positions based on the current angle
    # Assuming the pendulum swings in the plane, with its pivot point above the original position
    new_x = original_x + np.sin(angle) * 9
    new_y = original_y + 6 * (1 - np.cos(angle)) - 3  # Subtract to simulate gravity pulling the pendulum down
    #new_z = original_z + random.uniform(-0.1, 0.1)
    # Update the matrix with the new x and y positions, keeping z constant
    new_matrix = original_matrix.copy()
    new_matrix[3][0] = new_x
    new_matrix[3][1] = new_y
    new_matrix[3][2] = original_z  # Keep z constant

    return new_matrix


# ---------------------------       MAIN CODE  -------------------------------------------------------------------
def main_loop(awindow):

    vtx_array = mesh_cube_vertices()
    idx_array = mesh_cube_indexes()
    len_idx = len(idx_array)
    shader = initialize_object(vtx_array, idx_array)

    # World view models
    projection = pyrr.matrix44.create_perspective_projection_matrix(90,
                                                                    RESOLUTION[0]/RESOLUTION[1],
                                                                    NEAR_PROJ, FAR_PROJ)
    view = pyrr.matrix44.create_look_at(pyrr.Vector3([0, 1, 4]),    # eye behind screen in the top
                                        pyrr.Vector3([0, 0, 0]),    # looking to the origin
                                        pyrr.Vector3([0, 1, 0]))    # camera direction to Y upright

    cubes_list = create_cubes(NUM_INST)

    proj_loc = glGetUniformLocation(shader, "projection")
    model_loc = glGetUniformLocation(shader, "model")
    view_loc = glGetUniformLocation(shader, "view")
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)

    # Scale factors for time to control the "speed" of each movement
    scale_time_x = 0.01
    scale_time_y = 0.05
    scale_time_z = 0.07

    max_speed_x = 5
    max_speed_y = 5
    max_speed_z = 0.1

    # Game Loop
    frame_count = 0
    last_time = glfw.get_time()
    running = True
    #rotation = pyrr.matrix44.create_identity(dtype=np.float32)
    while running:
        current_time = glfw.get_time()
        glfw.poll_events()                                         # Check for user interactions
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)         # Reset the screen
        frame_count, last_time, fps = display_fps(frame_count, last_time, current_time, window)

        # Using np.tanh for smooth transitions in speed variation
        speed_variation_x = np.tanh(np.sin(current_time * scale_time_x)) * max_speed_x
        speed_variation_y = np.tanh(np.cos(current_time * scale_time_y)) * max_speed_y
        speed_variation_z = np.tanh(np.sin(current_time * scale_time_z)) * max_speed_z

        rot_x = pyrr.Matrix44.from_x_rotation(speed_variation_x * glfw.get_time())
        rot_y = pyrr.Matrix44.from_y_rotation(speed_variation_y * glfw.get_time())
        rot_z = pyrr.Matrix44.from_y_rotation(speed_variation_z * glfw.get_time())
        rotation = pyrr.matrix44.multiply(rot_x, rot_y)
        rotation = pyrr.matrix44.multiply(rotation, rot_z)


        for cube in cubes_list:
            oscillated_cube = oscillate_cube_position(cube, current_time, PENDULUM_AMP, PENDULUM_FREQ)

            render_cube_instance(oscillated_cube, rotation, len_idx, model_loc)

        # Next iteration
        glfw.swap_buffers(awindow)
        running = glfw_handle_events(awindow)
        frame_count += 1

    # terminate glfw, free up allocated resources
    glfw.terminate()




# Starting script
if __name__ == "__main__":
    window = initialize_glfw(RESOLUTION)
    main_loop(window)


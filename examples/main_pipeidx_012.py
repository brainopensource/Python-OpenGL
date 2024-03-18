import ctypes
import glfw
from math import sin, cos
import numpy as np
from OpenGL.GL import (
    glClear, glClearColor, glEnableClientState, glLoadIdentity,
    glRotatef, glVertexPointer, GL_COLOR_BUFFER_BIT, GL_FLOAT, GL_TRIANGLES, GL_TRUE, GL_FALSE, GL_VERTEX_ARRAY,
    GL_COLOR_ARRAY, glDrawArrays, glColorPointer, GL_VERTEX_SHADER, GL_FRAGMENT_SHADER,
    glUseProgram, glGenBuffers, glBindBuffer, GL_ARRAY_BUFFER, glBufferData, GL_STATIC_DRAW, glGetAttribLocation,
    glEnableVertexAttribArray, glVertexAttribPointer, GL_TRIANGLE_STRIP, glViewport,
    GL_ELEMENT_ARRAY_BUFFER, glDrawElements, GL_UNSIGNED_INT, glEnable, GL_DEPTH_TEST, glGetUniformLocation,
    GL_DEPTH_BUFFER_BIT, glUniformMatrix4fv, glPolygonMode, GL_FRONT_AND_BACK, GL_LINE, glLineWidth,
    glBindVertexArray, GL_LINES, glGenVertexArrays,

    )
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr


# ------------------------------- GLOBAL VARIABLES --------------------------------------------------------------------
RESOLUTION = [800, 600]
MONITOR_1_RESOLUTION = [1920, 1080]
WHITE_COLOR = (1.0, 1.0, 1.0)
BLACK_COLOR = (0.0, 0.0, 0.0)
REFRESH_MODE = 'infinite'


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
    if delta_time >= 0.016:
        fps = frame_count / delta_time
        glfw.set_window_title(window_bar, f"True OpenGL 0.0.1 - FPS: {fps:.0f}")
        return 0, current_time, fps
    else:
        return frame_count, last_time, None


#  ------------------------------  Mesh Functions ---------------------------------------------------------------------
def vertex_shader():
    return compileShader("""
    #version 330
    layout(location = 0) in vec4 a_position;
    layout(location = 1) in vec3 a_color;
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    out vec3 v_color;
    void main() {
        gl_Position = projection * view * model * vec4(a_position.xyz, 1.0);
        v_color = a_color;
    }
    """, GL_VERTEX_SHADER)


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


def mesh_hypercube_vertices():
    """
    x,y,z,r,g,b
    :return: np.array(dtype=np.float32)
    """
    # Example 1: Gradient Blue to Cyan Cube
    vertices = [
        # Front face
        -0.5, -0.5, 0.5, 0.0, 0.9, 0.0, 0.9,  # Neon Purple
        0.5, -0.5, 0.5, 0.0, 0.0, 0.8, 0.8,  # Neon Cyan
        0.5, 0.5, 0.5, 0.0, 0.8, 0.0, 0.8,  # Neon Purple
        -0.5, 0.5, 0.5, 0.0, 0.5, 0.8, 0.8,  # Neon Cyan

        # Back face
        -0.5, -0.5, -0.5, 0.0, 0.6, 0.0, 0.6,  # Dark Neon Purple
        0.5, -0.5, -0.5, 0.0, 0.0, 1.0, 1.0,  # Dark Neon Cyan
        0.5, 0.5, -0.5, 0.0, 0.9, 0.0, 0.9,  # Dark Neon Purple
        -0.5, 0.5, -0.5, 0.0, 0.0, 0.5, 0.5,  # Dark Neon Cyan

        # Hyper face (+w)
        -0.5, -0.5, 0.5, 0.5, 1.0, 1.0, 0.0,  # Yellow
        0.5, -0.5, 0.5, 0.5, 1.0, 0.0, 1.0,  # Magenta
        0.5, 0.5, 0.5, 0.5, 0.0, 1.0, 1.0,  # Cyan
        -0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0,  # White

        # Hyper face (-w)
        -0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.0,  # Dark Yellow
        0.5, -0.5, -0.5, -0.5, 0.5, 0.0, 0.5,  # Dark Magenta
        0.5, 0.5, -0.5, -0.5, 0.0, 0.5, 0.5,  # Dark Cyan
        -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5  # Dark White
    ]
    return np.array(vertices, dtype=np.float32)


def mesh_hypercube_indices():
    # Each cube (front and back)
    front_back_indices = [
        0, 1, 1, 2, 2, 3, 3, 0,  # Front face
        4, 5, 5, 6, 6, 7, 7, 4,  # Back face
        0, 4, 1, 5, 2, 6, 3, 7  # Connect front and back faces
    ]

    # Connect corresponding vertices between the inner and outer cube (4D connections)
    inner_outer_indices = [
        8, 9, 9, 10, 10, 11, 11, 8,  # Inner front face
        12, 13, 13, 14, 14, 15, 15, 12,  # Inner back face
        8, 12, 9, 13, 10, 14, 11, 15  # Connect inner front and back faces
    ]

    # Connect outer vertices to inner vertices (4D to 3D projection)
    projection_indices = [
        0, 8, 1, 9, 2, 10, 3, 11,
        4, 12, 5, 13, 6, 14, 7, 15
    ]

    return np.array(front_back_indices + inner_outer_indices + projection_indices, dtype=np.uint32)


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
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))            # 3,0 vertices tight
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))  # 3,0 vertices tight

    glUseProgram(shader)
    # Config Background standard color
    glClearColor(0.0, 0.4, 0.5, 1)
    glEnable(GL_DEPTH_TEST)

    return shader


def initialize_hypercube():
    vertices = mesh_hypercube_vertices()  # Define your vertices function
    indices = mesh_hypercube_indices()
    shader = create_program()

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    # Create and bind buffers, upload data
    create_vbo_buff(vertices)
    create_ebo_buff(indices)

    # Vertex attrib pointers setup
    # Position
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
    # Color
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(16))

    return vao, shader, len(indices)


def render_hypercube(vao, shader, indices_count):
    glUseProgram(shader)
    glBindVertexArray(vao)
    # Set uniforms (projection, view, model matrices)
    # Assuming you have functions to create these matrices
    set_uniforms(shader)

    glDrawElements(GL_LINES, indices_count, GL_UNSIGNED_INT, None)

# ---------------------------       MAIN CODE  -------------------------------------------------------------------
def main_loop(awindow):

    #initialize_object(mesh_triangle_vertices(0.1), mesh_triangle_indexes())

    vao, shader, indices_count = initialize_hypercube()
    # Game Loop
    frame_count = 0
    last_time = glfw.get_time()
    running = True
    while running:
        glfw.poll_events()  # Check for user interactions
        # Reset the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        current_time = glfw.get_time()
        frame_count += 1
        frame_count, last_time, fps = display_fps(frame_count, last_time, current_time, window)


        # Render buffered data
        render_hypercube(vao, shader, indices_count)

        # Next iteration
        glfw.swap_buffers(awindow)
        running = glfw_handle_events(awindow)


    # terminate glfw, free up allocated resources
    glfw.terminate()




# Starting script
if __name__ == "__main__":
    window = initialize_glfw(RESOLUTION)
    main_loop(window)

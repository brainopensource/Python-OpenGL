import ctypes
import glfw
from math import sin, cos
import numpy as np
from OpenGL.GL import (
    glClear, glClearColor, glEnableClientState, glLoadIdentity,
    glRotatef, glVertexPointer, GL_COLOR_BUFFER_BIT, GL_FLOAT, GL_TRIANGLES, GL_TRUE, GL_FALSE, GL_VERTEX_ARRAY,
    GL_COLOR_ARRAY, glDrawArrays, glColorPointer, GL_VERTEX_SHADER, GL_FRAGMENT_SHADER,
    glUseProgram, glGenBuffers, glBindBuffer, GL_ARRAY_BUFFER, glBufferData, GL_STATIC_DRAW, glGetAttribLocation,
    glEnableVertexAttribArray, glVertexAttribPointer, GL_TRIANGLE_STRIP, glViewport
    )
from OpenGL.GL.shaders import compileProgram, compileShader

# ------------------------------- GLOBAL VARIABLES --------------------------------------------------------------------
RESOLUTION = [1600, 900]
MONITOR_1_RESOLUTION = [1920, 1080]
WHITE_COLOR = (1.0, 1.0, 1.0)
REFRESH_MODE = 'vsync'


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
    elif mode == 'infinite':
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
    vertex_src = """
    # version 330

    layout(location = 0) in vec3 a_position;
    layout(location = 1) in vec3 a_color;

    out vec3 v_color;

    void main()
    {
        gl_Position = vec4(a_position, 1.0);
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


def mesh_create_triangle_vertices():
    verts = [-0.5, -0.5, 0.0, 0.0, 0.0, 0.0,
             0.5, -0.5, 0.0, 0.0, 1.0, 0.0,
             -0.5, 0.5, 0.0, 0.0, 0.0, 1.0,
             0.5, 0.5, 0.0, 1.0, 1.0, 1.0]
    return np.array(verts, dtype=np.float32)


def create_buffers(vertices):
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)    # 32 = 9 verts * 4 bytes each


def initialize_object(vertices):
    shader = create_program()
    create_buffers(vertices)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))            # 3,0 vertices tight
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))  # 3,0 vertices tight

    glUseProgram(shader)


# ---------------------------       MAIN CODE  -------------------------------------------------------------------
def main_loop(awindow):


    initialize_object(mesh_create_triangle_vertices())

    # Config Background standard color
    glClearColor(0.0, 0.4, 0.5, 1)

    # Game Loop
    frame_count = 0
    last_time = glfw.get_time()
    running = True
    while running:
        glfw.poll_events()  # Check for user interactions

        current_time = glfw.get_time()
        frame_count += 1
        frame_count, last_time, fps = display_fps(frame_count, last_time, current_time, window)

        # Reset the screen
        glClear(GL_COLOR_BUFFER_BIT)

        # Render buffered data
        #glDrawArrays(GL_TRIANGLES, 0, 3)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

        # Next iteration
        glfw.swap_buffers(awindow)
        running = glfw_handle_events(awindow)


    # terminate glfw, free up allocated resources
    glfw.terminate()




# Starting script
if __name__ == "__main__":
    window = initialize_glfw(RESOLUTION)
    main_loop(window)


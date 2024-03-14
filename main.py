import ctypes
import glfw
from math import sin, cos
import numpy as np
from OpenGL.GL import (
    glClear, glClearColor, glColor3f, glEnableClientState, glLoadIdentity,
    glRotatef, glScale, glTranslatef, glVertexPointer, glRasterPos2f,
    GL_COLOR_BUFFER_BIT, GL_FLOAT, GL_TRIANGLES, GL_TRUE, GL_VERTEX_ARRAY,
    GL_COLOR_ARRAY, glDrawArrays, glColorPointer
    )
from OpenGL.GL.shaders import compileProgram, compileShader


# ------------------------------- GLOBAL VARIABLES --------------------------------------------------------------------
RESOLUTION = [1280, 720]
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
    # check if window was created
    glfw_check_terminate(glfw_window)
    # make the context current
    glfw.make_context_current(glfw_window)
    # Disable V-Sync (vertical synchronization)
    glfw.swap_interval(choose_refresh_mode(REFRESH_MODE))

    return glfw_window


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

    in vec3 a_position;
    in vec3 a_color;

    out vec3 v_color;

    void main()
    {
        gl_Position = vec4(a_position, 1.0);
        v_color = a_color;
    }
    """
    return compileShader(vertex_src)


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
    return compileShader(fragment_src)


def mesh_create_triangle_vertices():
    verts = [-0.5, -0.0, 0.5,
             0.5, -0.0, 0.5,
             0.0, 0.5, 0.0]
    return np.array(verts, dtype=np.float32)


def mesh_create_triangle_colors():
    col = [1.0, 0.0, 0.0,
           0.0, 1.0, 0.0,
           0.0, 0.0, 1.0]
    return np.array(col, dtype=np.float32)

def initialize_render():
    # initializing glfw library
    glfw_window = initialize_glfw(RESOLUTION)
    # Create vertices coordinates and colors as numpy arrays
    vertices = mesh_create_triangle_vertices()
    colors = mesh_create_triangle_colors()
    # Symbolic constant, using vertex arrays with calls to draw
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    # Number of coordinates per vertex, data type, stride, first coord of first vertex array
    glVertexPointer(3, GL_FLOAT, 0, vertices)
    glColorPointer(3, GL_FLOAT, 0, colors)
    # Reset the background with a standard color
    glClearColor(0, 0.1, 0.1, 1)

    return glfw_window



# ---------------------------       MAIN CODE  -------------------------------------------------------------------
def main_loop(awindow):
    frame_count = 0
    last_time = glfw.get_time()
    running = True
    while running:
        current_time = glfw.get_time()
        frame_count += 1
        frame_count, last_time, fps = display_fps(frame_count, last_time, current_time, window)

        # Reset the screen
        glClear(GL_COLOR_BUFFER_BIT)
        # Check for user interactions
        glfw.poll_events()

        glLoadIdentity()
        glRotatef(sin(current_time) * 45, 0, 0, 1)
        glDrawArrays(GL_TRIANGLES, 0, 3)

        glfw.swap_buffers(awindow)

        running = glfw_handle_events(awindow)


    # terminate glfw, free up allocated resources
    glfw.terminate()


# Entry point
if __name__ == "__main__":
    window = initialize_render()
    main_loop(window)


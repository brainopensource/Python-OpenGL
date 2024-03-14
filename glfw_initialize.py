import glfw
from utils import *
from OpenGL.GL import (glUniformMatrix4fv, GL_FALSE, glEnable, glPolygonMode, GL_DEPTH_TEST, GL_FRONT_AND_BACK, GL_LINE,
                       GL_FRONT)



RESOLUTION = [1420, 800]
MONITOR_1_RESOLUTION = [1920, 1080]
REFRESH_MODE = 1


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
    #glfw.swap_interval(choose_refresh_mode(REFRESH_MODE))


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
    else:
        return mode


def calculate_fps(dt, fcount):
    return fcount / dt


def display_fps_title(frame_count, delta_time, window_bar):
    fps = frame_count / delta_time
    glfw.set_window_title(window_bar, f"True OpenGL 0.0.1 - FPS: {fps:.0f}")
    return fps

def handle_events(frame_count, last_time, current_time, window_bar, view_loc, afunc_list):
    delta_time = current_time - last_time
    if delta_time >= 0.01:
        fps = display_fps_title(frame_count, delta_time, window_bar)
        afunc_list[0]()
        view = afunc_list[1]()
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
        return 0, current_time, fps
    else:
        return frame_count, last_time, None


def display_fps(frame_count, last_time, current_time, window_bar):
    delta_time = current_time - last_time
    if delta_time >= 0.01:
        fps = display_fps_title(frame_count, delta_time, window_bar)
        return 0, current_time, fps
    else:
        return frame_count, last_time, None

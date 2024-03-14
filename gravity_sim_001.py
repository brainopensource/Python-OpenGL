import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

# Shader Source Code
compute_shader_source = """
#version 430
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer SSBO {
    vec4 positions[];
};
uniform float deltaTime;
void main() {
    int id = int(gl_GlobalInvocationID.x);
    if(id == 0) {
        vec3 direction = positions[1].xyz - positions[0].xyz;
        float distance = length(direction);
        if(distance > 0.0) {
            vec3 forceDirection = normalize(direction);
            float forceMagnitude = deltaTime / (distance * distance);
            positions[0].xyz += forceDirection * forceMagnitude;
        }
    } else {
        vec3 direction = positions[0].xyz - positions[1].xyz;
        float distance = length(direction);
        if(distance > 0.0) {
            vec3 forceDirection = normalize(direction);
            float forceMagnitude = deltaTime / (distance * distance);
            positions[1].xyz += forceDirection * forceMagnitude;
        }
    }
}
"""

vertex_shader_source = """
#version 430
layout(location = 0) in vec4 position;
void main() {
    gl_Position = position;
    gl_PointSize = 50.0;
}
"""

fragment_shader_source = """
#version 430
out vec4 fragColor;
void main() {
    fragColor = vec4(1.0, 1.0, 1.0, 1.0);
}
"""

def main():
    # Initialize GLFW
    if not glfw.init():
        return
    window = glfw.create_window(800, 600, "Particle Simulation", None, None)
    if not window:
        glfw.terminate()
        return
    glfw.make_context_current(window)

    # Compile shaders
    compute_program = compileProgram(compileShader(compute_shader_source, GL_COMPUTE_SHADER))
    render_program = compileProgram(compileShader(vertex_shader_source, GL_VERTEX_SHADER),
                                    compileShader(fragment_shader_source, GL_FRAGMENT_SHADER))

    # Particle data
    particle_positions = np.array([
        [-0.1, 0.0, 0.0, 1.0],  # Particle 1
        [0.05, 0.05, 0.0, 1.0,]    # Particle 2
    ], dtype=np.float32)

    # Create a buffer for particle positions
    ssbo = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, particle_positions.nbytes, particle_positions, GL_DYNAMIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo)

    # Vertex Array Object
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    glEnableVertexAttribArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, ssbo)
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))

    # Main loop
    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT)

        # Compute shader
        glUseProgram(compute_program)
        glUniform1f(glGetUniformLocation(compute_program, "deltaTime"), 0.01)
        glDispatchCompute(2, 1, 1)

        # Make sure writing to image has finished before read
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        # Render
        glUseProgram(render_program)
        glBindVertexArray(vao)
        glDrawArrays(GL_POINTS, 0, 2)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()

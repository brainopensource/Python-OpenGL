#version 330 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 offset; // Not used for tunnel
layout (location = 2) in vec4 color;

uniform mat4 view;
uniform mat4 projection;
uniform float time; // Time for simulating movement

out vec4 vertexColor;

void main() {
    float speed = 0.5; // Speed of movement
    float tunnelLength = 50.0; // Length of the tunnel for repeating effect

    // Compute an overall phase for the cylinder based on time
    float phase = mod(speed * time, tunnelLength) * 5;

    // Apply the phase uniformly to all vertices to prevent discontinuity
    float zPos = mod(position.z - phase, tunnelLength) - (tunnelLength / 2.0);

    vec3 pos = vec3(position.xy, zPos); // Apply movement effect along Z-axis

    gl_Position = projection * view * vec4(pos, 1.0);
    vertexColor = color;
}

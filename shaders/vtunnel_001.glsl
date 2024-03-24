#version 330 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 offset; // Not used for tunnel
layout (location = 2) in vec4 color;

uniform mat4 view;
uniform mat4 projection;
uniform float time; // Time for simulating movement

out vec4 vertexColor;

void main() {
    // Parameters for tunnel movement
    float speed = 0.5; // Speed of movement
    float tunnelLength = 4.0; // Length of the tunnel for repeating effect

    // Adjust Z position based on time to simulate movement
    float zPos = mod(position.z - (speed * time), tunnelLength) - (tunnelLength / 2.0);

    vec3 pos = vec3(position.xy, zPos); // Apply movement effect along Z-axis

    gl_Position = projection * view * vec4(pos, 1.0);
    vertexColor = color;
}

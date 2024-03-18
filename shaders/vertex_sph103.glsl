#version 330
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 offset;
layout (location = 2) in vec4 color;
layout (location = 3) in vec3 normal;

uniform mat4 view;
uniform mat4 projection;
uniform float time;

out vec4 vertexColor;
out vec3 FragPos; // Pass the fragment position
out vec3 Normal; // Pass the normal

void main() {
    float amplitude = 10.0;
    float height = 10.0;
    float speed = 2.0;
    float pend_time = time * 3.14 * 0.01;

    float pendulumX = sin(pend_time * speed) * amplitude + offset.x;
    float factor = cos(pend_time * speed);
    float pendulumY = -factor * factor * height + offset.y;

    vec3 pos = position + vec3(pendulumX, pendulumY, offset.z);
    gl_Position = projection * view * vec4(pos, 1.0);

    // Update the outputs for the fragment shader
    FragPos = vec3(view * vec4(pos, 1.0)); // Convert position to view space
    Normal = mat3(transpose(inverse(view))) * normal; // Convert normal to view space

    // Pass color directly to fragment shader
    vertexColor = color;
}

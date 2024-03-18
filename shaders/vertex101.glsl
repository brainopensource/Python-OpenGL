#version 330
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 offset;
layout (location = 2) in vec4 color;

uniform mat4 view;
uniform mat4 projection;
uniform float time;

out vec4 vertexColor;

void main() {
    // Original position of the vertex
    vec3 pos = position + offset;
    float waveamp = sin(pos.x * 0.5 + time * 0.5) * 1.0;
    pos.y = waveamp;

    vec3 ambient = vec3(0.6588, 0.6588, 0.6588);
    vec3 colormap = vec3(max(0.6, pos.y), 0.0, 1.0);
    vertexColor = vec4(colormap * ambient, color.a);

    gl_Position = projection * view * vec4(pos, 1.0);
}

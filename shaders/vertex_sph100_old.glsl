#version 330
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 offset;
layout (location = 2) in vec4 color;

uniform mat4 view;
uniform mat4 projection;
uniform float time;

out vec4 vertexColor;

void main() {
    // Pendulum motion parameters
    float amplitude = 10.0; // Maximum distance from the center in XY plane
    float height = 10.0;    // Maximum height of the pendulum swing
    float speed = 2.0;     // Speed of the pendulum
    float baseHeight = 1.0; // Base height from which the pendulum swings
    float pend_time = time * 0.2;

    // The pendulum swings back and forth in an arc, so we use cos for Y to simulate the height change
    float pendulumX = sin(pend_time * speed) * amplitude + offset.x;
    float pendulumY = -abs(cos(pend_time * speed)) * height + offset.y; // abs ensures it goes up and then back down
    vec3 pos = position + vec3(pendulumX, pendulumY, offset.z); // Z remains unchanged

    vec3 ambient = vec3(0.6588, 0.6588, 0.6588);
    vec3 colormap = vec3(-pos.y, 0.0, 1.0);
    vertexColor = vec4(colormap * ambient, color.a);

    gl_Position = projection * view * vec4(pos, 1.0);
}

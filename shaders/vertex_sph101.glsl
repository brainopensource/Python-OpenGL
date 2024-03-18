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
    float speed = 2.0;      // Speed of the pendulum
    float pend_time = time * 3.14 * 0.1;

    // Adjusting the motion to simulate the pendulum's changing velocity
    // The pendulum's X position changes with a constant speed
    float pendulumX = sin(pend_time * speed) * amplitude + offset.x;
    
    // The pendulum's Y position changes, slowing down as it reaches the apex
    // This uses the squared cosine to exaggerate the slowdown at the top
    float factor = cos(pend_time * speed);
    float pendulumY = -factor * factor * height + offset.y; // Squaring cos() exaggerates the slow at the apex

    vec3 pos = position + vec3(pendulumX, pendulumY, offset.z); // Z remains unchanged

    vec3 ambient = vec3(0.6588, 0.6588, 0.6588);
    // Using -pos.y for colormap to create a gradient effect based on the pendulum's height
    vec3 colormap = vec3(-pos.y / height, 0.5, 1.0); // Normalizing -pos.y by height for a color gradient
    vertexColor = vec4(colormap * ambient, color.a);

    gl_Position = projection * view * vec4(pos, 1.0);
}

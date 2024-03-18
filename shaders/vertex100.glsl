#version 330
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 offset;
layout (location = 2) in vec4 color;

uniform mat4 view;
uniform mat4 projection;
uniform float time;
uniform vec3 camera_position; 

out vec4 vertexColor;

void main() {
    // Original position of the vertex
    vec3 pos = position + offset;
    float waveamp_time = time * 0.5;
    float waveamp = sin(pos.x * 0.5 + waveamp_time);
    pos.y += waveamp;

    // Calculate the derivative of the wave function with respect to x.
    float dwave_dx = 0.5 * cos(pos.x * 0.5 + waveamp_time);
    vec3 tangent_x = vec3(1.0, dwave_dx, 0.0);
    vec3 tangent_z = vec3(0.0, 0.0, 1.0);
    vec3 normal = normalize(cross(tangent_x, tangent_z));

    // Calculate lighting
    vec3 lightColor = vec3(1.0, 1.0, 1.0);
    vec3 lightSource = normalize(vec3(1.0, -0.1, 0.5));
    vec3 ambient = vec3(0.5, 0.5, 0.5);
    vec3 light_reflection = vec3(lightSource.x, -lightSource.y, lightSource.z);
    // Phong lighting model
    vec3 viewDir = normalize(camera_position - pos);
    vec3 reflectDir = reflect(normal, -light_reflection);

    float diffuseStrength = max(dot(normal, lightSource), 0.0);
    vec3 diffuse = diffuseStrength * lightColor;
    
    float specularStrength = pow(max(dot(normal, reflectDir), 0.0), 64.0); // Increased specular strength
    vec3 specular = specularStrength * lightColor;

    // Combine ambient, diffuse, and specular lighting
    vec3 lighting = ambient + diffuse * 0.8 + specular * 0.5; // Increased lighting intensity

    // Apply lighting effect to the color
    vertexColor = vec4(color.rgb * lighting, color.a);
    //vertexColor = vec4(normalize(normal) * 0.5 + 0.5, 1.0);

    // Set the position of the vertex
    gl_Position = projection * view * vec4(pos, 1.0);
}

#version 330
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 offset;
layout (location = 2) in vec4 color;

uniform mat4 view;
uniform mat4 projection;
uniform float time;
uniform vec3 camera_position; // For dynamic lighting


out vec4 vertexColor;

void main()
{

    // Original position of the vertex
    vec3 pos = position;
    float waveamp_time = time * 0.5;
    float waveamp = sin(pos.x * 0.5 + waveamp_time) * 1.2;
    pos.y += waveamp;

    // Calculate the derivative of the wave function with respect to x. The partial derivative of the wave function, giving us the slope at a given point.
    float dwave_dx = 0.5 * cos(pos.x * 0.5 + waveamp_time);
    vec3 tangent_x = vec3(1.0, dwave_dx, 0.0);
    vec3 tangent_z = vec3(0.0, 0.0, 1.0);
    vec3 normal = cross(tangent_x, tangent_z);
    normal = normalize(normal);

    // Calculate lighting
    vec3 lightColor = vec3(0.75, 0.75, 0.75);
    vec3 lightSource = normalize(camera_position - pos);
    vec3 ambient = vec3(0.5, 0.5, 0.5);

    // Assume normal points towards the light source for diffuse lighting
    float diffuseStrength = max(dot(normal, lightSource), 0.0);
    vec3 diffuse = diffuseStrength * lightColor;

    // Specular lighting
    vec3 viewDir = normalize(camera_position - pos);
    vec3 reflectDir = (reflect(-lightSource, normal));

    float specularStrength = max(0.0, dot(viewDir, reflectDir));
    specularStrength = pow(specularStrength, 64);
    vec3 specular = specularStrength * lightColor;

    float distance = length(camera_position - pos);
    float attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);

    // Combine ambient and diffuse lighting
    vec3 lighting = (ambient + (diffuse + specular * 0.5) * attenuation);

    // Set the position of the vertex
    gl_Position = projection * view * vec4(pos + offset, 1.0);

    // Apply lighting effect to the color
    vertexColor = color * vec4(lighting, 1.0);

}


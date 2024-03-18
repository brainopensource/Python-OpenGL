#version 330
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 offset;
layout (location = 2) in vec4 color;

uniform mat4 view;
uniform mat4 projection;
uniform float time;


out vec4 vertexColor;

void main()
{

    // Original position of the vertex
    vec3 pos = position;
    float waveamp_time = time * 0.5;
    float lightsource_time = time * 0.1;
    float waveamp = sin(pos.x * 0.5 + waveamp_time) * 1.2;
    //float waveamp_2 = cos(pos.z * 0.1 + time * 0.5) * 2.5;
    pos.y += waveamp;

    // Calculate the derivative of the wave function with respect to x. The partial derivative of the wave function, giving us the slope at a given point.
    float dwave_dx = 0.5 * cos(pos.x * 0.5 + waveamp_time);
    // Construct the tangent vector at this point on the surface, with respect to x has a y component equal to the derivative of the wave function
    vec3 tangent_x = vec3(1.0, dwave_dx, 0.0);
    // Assume a default tangent vector along the z axis (since your surface deformation is independent of z)
    vec3 tangent_z = vec3(0.0, 0.0, 1.0);
    // Compute the normal by taking the cross product of the tangent vectors
    vec3 normal = cross(tangent_x, tangent_z);
    // Normalize the normal to ensure it's a unit vector
    normal = normalize(normal);

    // Calculate lighting
    vec3 lightColor = vec3(0.75, 0.75, 0.75);
    vec3 lightSource = vec3(1.0, -0.05, 0.0);
    vec3 ambient = vec3(0.5, 0.5, 0.5);

    // Assume normal points towards the light source for diffuse lighting
    float diffuseStrength = max(dot(normal, lightSource), 0.0);
    vec3 diffuse = diffuseStrength * lightColor;

    // Specular lighting
    vec3 camera_source = vec3(-1.0, 1.0, 0.0);
    vec3 view_source = normalize(camera_source);
    vec3 reflect_source = normalize(reflect(-lightSource, normal));
    float specularStrength = max(0.0, dot(view_source, reflect_source));
    specularStrength = pow(specularStrength, 64);
    vec3 specular = specularStrength * lightColor;

    // Combine ambient and diffuse lighting
    vec3 lighting = (ambient + diffuse + specular * 0.1);

    // Set the position of the vertex
    gl_Position = projection * view * vec4(pos + offset, 1.0);

    // Apply lighting effect to the color
    vertexColor = color * vec4(lighting, 1.0);

}


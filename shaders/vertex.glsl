#version 330
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 offset;
layout (location = 2) in vec4 color;

uniform mat4 view;
uniform mat4 projection;
uniform float time;
//varying vec4 v_normal;

out vec4 vertexColor;

void main()
{
    // Position of the vertex
    vec3 pos = position;
    // Light properties
    vec3 lightColor = vec3(1.0, 1.0, 1.0);
    vec3 lightSource = vec3(1.0, 0.0, 1.0);
    // Ambient Light
    vec3 ambient = vec3(0.8, 0.7, 0.6);
    // Light Diffusion
    //vec3 normal = normalize(v_normal.xyz);
    //float diffuseStrenght = max(0.0, dot(lightSource, normal))
    //vec3 diffuse = diffuseStrenght * lightColor;

    vec3 lighting = ambient; //* 0.0 + diffuse;


    // Combining sine, cosine, and exponential functions for a chaotic ocean-like effect
    float wave1 = sin(pos.x * 0.5 + time * 0.2) * 1.2;
    float wave8 = sin(pos.z * 0.33 + time * 0.3) * 0.5;
    float wave9 = sin(pos.z * 0.13 + time * 0.4) * 0.25;
    float wave2 = sin(pos.x * 1.0 + time * 1.0) * 0.2;
    float wave3 = cos(pos.z * 1.0 + time * 0.8) * 0.05;
    float wave4 = sin(pos.x + pos.z) * 0.05;
    float wave5 = cos(pos.x * time * 0.1 - 2 * sin(pos.z * time * 0.1) + time * 0.01) * 0.01;
    float wave6 = sin(pos.z * time * 0.2 - 2 * cos(pos.x * time * 0.2) + time * 0.03) * 0.01;
    float wave7 = sin(pos.x + time * 0.1) * cos(pos.z + time * 0.1) * 0.15;

    // Combining waves for a more complex and chaotic effect
    pos.y = wave1 + wave2 + wave3 + wave4 + wave5 + wave6 + wave7 + wave8 + wave9;

    gl_Position = projection * view * vec4(pos + offset, 1.0);
    vertexColor = color * vec4(lighting, 1.0);
}

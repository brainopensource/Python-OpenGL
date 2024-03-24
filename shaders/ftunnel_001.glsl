#version 330
in vec4 vertexColor;
in vec3 FragPos;
in vec3 Normal;

uniform vec3 lightPos; // Position of the light source
uniform vec3 viewPos; // Position of the camera/view

out vec4 outColor;

void main() {
    // Ambient
    float ambientStrength = 0.5;
    vec3 ambient = ambientStrength * vertexColor.rgb;

    // Diffuse
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * vertexColor.rgb;

    vec3 result = (ambient + diffuse) * vertexColor.rgb;
    outColor = vec4(result, vertexColor.a);
}

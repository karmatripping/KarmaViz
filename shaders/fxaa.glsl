// FXAA (Fast Approximate Anti-Aliasing) Shader
// Post-processing anti-aliasing for smoother edges

#version 330 core

uniform sampler2D u_texture;
uniform vec2 u_resolution;
uniform float u_fxaa_strength;  // 0.0 = off, 1.0 = full strength

in vec2 v_texcoord;
out vec4 fragColor;

// FXAA implementation
vec4 fxaa(sampler2D tex, vec2 texCoord, vec2 resolution) {
    vec2 texelSize = 1.0 / resolution;
    
    // Sample the center pixel
    vec4 center = texture(tex, texCoord);
    
    // Sample neighboring pixels
    vec4 north = texture(tex, texCoord + vec2(0.0, -texelSize.y));
    vec4 south = texture(tex, texCoord + vec2(0.0, texelSize.y));
    vec4 east = texture(tex, texCoord + vec2(texelSize.x, 0.0));
    vec4 west = texture(tex, texCoord + vec2(-texelSize.x, 0.0));
    
    // Calculate luminance for edge detection
    float centerLuma = dot(center.rgb, vec3(0.299, 0.587, 0.114));
    float northLuma = dot(north.rgb, vec3(0.299, 0.587, 0.114));
    float southLuma = dot(south.rgb, vec3(0.299, 0.587, 0.114));
    float eastLuma = dot(east.rgb, vec3(0.299, 0.587, 0.114));
    float westLuma = dot(west.rgb, vec3(0.299, 0.587, 0.114));
    
    // Find min and max luminance
    float minLuma = min(centerLuma, min(min(northLuma, southLuma), min(eastLuma, westLuma)));
    float maxLuma = max(centerLuma, max(max(northLuma, southLuma), max(eastLuma, westLuma)));
    
    // Calculate edge contrast
    float contrast = maxLuma - minLuma;
    
    // If contrast is low, no anti-aliasing needed
    if (contrast < 0.05) {
        return center;
    }
    
    // Calculate edge direction
    float horizontal = abs(northLuma + southLuma - 2.0 * centerLuma);
    float vertical = abs(eastLuma + westLuma - 2.0 * centerLuma);
    
    bool isHorizontal = horizontal >= vertical;
    
    // Sample along the edge direction
    vec2 step = isHorizontal ? vec2(0.0, texelSize.y) : vec2(texelSize.x, 0.0);
    
    vec4 pos = texture(tex, texCoord + step);
    vec4 neg = texture(tex, texCoord - step);
    
    // Blend based on edge strength
    float blend = 0.5 * (1.0 - contrast / 0.2);
    blend = clamp(blend, 0.0, 0.5);
    
    return mix(center, 0.5 * (pos + neg), blend * u_fxaa_strength);
}

void main() {
    if (u_fxaa_strength > 0.0) {
        fragColor = fxaa(u_texture, v_texcoord, u_resolution);
    } else {
        fragColor = texture(u_texture, v_texcoord);
    }
}

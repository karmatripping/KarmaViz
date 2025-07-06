import os
import json

experimental_warp_maps = [
    {
        "name": "Plasma Tunnel",
        "category": "experimental",
        "description": "A tunnel effect with plasma-like color modulation.",
        "complexity": "high",
        "author": "AutoGen",
        "version": "1.0",
        "is_builtin": False,
        "glsl_code": """
// Plasma Tunnel Warp Map
vec2 get_pattern(vec2 pos, float t) {
    vec2 center = vec2(0.5, 0.5);
    vec2 p = pos - center;
    float angle = atan(p.y, p.x) + sin(t + length(p) * 8.0) * 0.2;
    float radius = length(p) + 0.1 * sin(t * 2.0 + angle * 6.0);
    return vec2(cos(angle), sin(angle)) * radius * 0.04;
}
"""
    },
    {
        "name": "Hex Grid Flow",
        "category": "experimental",
        "description": "Hexagonal grid with animated flow distortion.",
        "complexity": "medium",
        "author": "AutoGen",
        "version": "1.0",
        "is_builtin": False,
        "glsl_code": """
// Hex Grid Flow Warp Map
vec2 get_pattern(vec2 pos, float t) {
    float qx = pos.x * 2.0 / sqrt(3.0);
    float qy = pos.y - pos.x / sqrt(3.0);
    float hex = mod(floor(qx + 0.5) + floor(qy + 0.5), 2.0);
    float flow = sin(t * 2.0 + pos.x * 8.0 + pos.y * 8.0) * 0.02;
    return vec2(flow * hex, -flow * (1.0 - hex));
}
"""
    },
    {
        "name": "Radial Pulse",
        "category": "experimental",
        "description": "Pulsing radial waves from the center.",
        "complexity": "low",
        "author": "AutoGen",
        "version": "1.0",
        "is_builtin": False,
        "glsl_code": """
// Radial Pulse Warp Map
vec2 get_pattern(vec2 pos, float t) {
    vec2 center = vec2(0.5, 0.5);
    float dist = length(pos - center);
    float pulse = sin(20.0 * dist - t * 4.0) * 0.025 * smoothstep(0.1, 0.5, dist);
    return normalize(pos - center) * pulse;
}
"""
    },
    {
        "name": "Twist Grid",
        "category": "experimental",
        "description": "Grid pattern with a twisting animation.",
        "complexity": "medium",
        "author": "AutoGen",
        "version": "1.0",
        "is_builtin": False,
        "glsl_code": """
// Twist Grid Warp Map
vec2 get_pattern(vec2 pos, float t) {
    float grid = step(0.5, mod(floor(pos.x * 12.0) + floor(pos.y * 12.0), 2.0));
    float twist = sin((pos.x - pos.y) * 12.0 + t * 3.0) * 0.018;
    return vec2(twist * grid, twist * (1.0 - grid));
}
"""
    },
    {
        "name": "Wave Interference",
        "category": "experimental",
        "description": "Interference pattern from two moving wave sources.",
        "complexity": "high",
        "author": "AutoGen",
        "version": "1.0",
        "is_builtin": False,
        "glsl_code": """
// Wave Interference Warp Map
vec2 get_pattern(vec2 pos, float t) {
    vec2 src1 = vec2(0.3 + 0.1 * sin(t), 0.5);
    vec2 src2 = vec2(0.7 + 0.1 * cos(t), 0.5);
    float w1 = sin(30.0 * length(pos - src1) - t * 2.0);
    float w2 = sin(30.0 * length(pos - src2) + t * 2.0);
    float interference = (w1 + w2) * 0.012;
    return normalize(pos - 0.5) * interference;
}
"""
    },
    {
        "name": "Spiral Zoom",
        "category": "experimental",
        "description": "Zooming spiral effect with time-based scaling.",
        "complexity": "medium",
        "author": "AutoGen",
        "version": "1.0",
        "is_builtin": False,
        "glsl_code": """
// Spiral Zoom Warp Map
vec2 get_pattern(vec2 pos, float t) {
    vec2 center = vec2(0.5, 0.5);
    vec2 p = pos - center;
    float angle = atan(p.y, p.x) + t * 0.7;
    float radius = length(p) * (1.0 + 0.3 * sin(t));
    float spiral = sin(10.0 * angle + radius * 8.0) * 0.025;
    return vec2(cos(angle), sin(angle)) * spiral;
}
"""
    },
    {
        "name": "Noise Flow",
        "category": "experimental",
        "description": "Animated flow field using pseudo-random noise.",
        "complexity": "high",
        "author": "AutoGen",
        "version": "1.0",
        "is_builtin": False,
        "glsl_code": """
// Noise Flow Warp Map
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
vec2 get_pattern(vec2 pos, float t) {
    float n = hash(floor(pos * 10.0) + t);
    float angle = n * 6.2831;
    return vec2(cos(angle), sin(angle)) * 0.018;
}
"""
    },
    {
        "name": "Mirror Ripple",
        "category": "experimental",
        "description": "Mirrored ripple effect from both sides.",
        "complexity": "low",
        "author": "AutoGen",
        "version": "1.0",
        "is_builtin": False,
        "glsl_code": """
// Mirror Ripple Warp Map
vec2 get_pattern(vec2 pos, float t) {
    float rippleL = sin(25.0 * pos.x - t * 2.0) * 0.012;
    float rippleR = sin(25.0 * (1.0 - pos.x) - t * 2.0) * 0.012;
    return vec2(rippleL + rippleR, 0.0);
}
"""
    }
]

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

output_dir = os.path.join("warp_maps", "experimental")
ensure_dir(output_dir)

for warp in experimental_warp_maps:
    filename = warp["name"].lower().replace(" ", "_") + ".json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        json.dump(warp, f, indent=2)
    print(f"Created: {filepath}")

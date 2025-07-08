"""
Rotation Renderer for KarmaViz

This module provides rotation effects and transformations for the visualizer.
Repurposed from the old stackable shader system to handle rotation-specific effects.
"""

import moderngl
import numpy as np
import math
from typing import Optional
from modules.benchmark import benchmark


class RotationRenderer:
    """Handles rotation effects and transformations for the visualizer.
    This class was repurposed from the old StackableRenderer to handle
    our fucking awesome rotation effects because why the hell not?
    """

    def __init__(self, ctx: moderngl.Context, width: int, height: int):
        self.ctx = ctx
        self.width = width
        self.height = height

        # Create framebuffers for rotation rendering
        self.framebuffer_a = self._create_framebuffer()
        self.framebuffer_b = self._create_framebuffer()

        # Create fullscreen quad for post-processing
        self.quad_vbo = self._create_fullscreen_quad()

        # Rotation state
        self.rotation_angle = 0.0
        self.rotation_speed = 1.0
        self.rotation_direction = 1  # 1 for CW, -1 for CCW
        self.rotation_mode = 0  # 0=None, 1=CW, -1=CCW, 2=Music, -2=Random

        # Create rotation shader
        self.rotation_program = self._create_rotation_shader()

    def _create_framebuffer(self) -> moderngl.Framebuffer:
        """Create a framebuffer for intermediate rendering"""
        texture = self.ctx.texture((self.width, self.height), 4)  # RGBA
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        framebuffer = self.ctx.framebuffer(color_attachments=[texture])
        return framebuffer

    def _create_fullscreen_quad(self) -> moderngl.Buffer:
        """Create a fullscreen quad for post-processing"""
        # Fullscreen quad vertices (position + texcoord)
        vertices = np.array([
            # Position    # TexCoord
            -1.0, -1.0,   0.0, 0.0,  # Bottom-left
             1.0, -1.0,   1.0, 0.0,  # Bottom-right
            -1.0,  1.0,   0.0, 1.0,  # Top-left
             1.0,  1.0,   1.0, 1.0,  # Top-right
        ], dtype=np.float32)

        return self.ctx.buffer(vertices.tobytes())

    def _create_rotation_shader(self) -> moderngl.Program:
        """Create the rotation post-processing shader"""
        vertex_shader = """
        #version 330 core
        in vec2 in_position;
        in vec2 in_texcoord;
        out vec2 texcoord;

        void main() {
            gl_Position = vec4(in_position, 0.0, 1.0);
            texcoord = in_texcoord;
        }
        """

        fragment_shader = """
        #version 330 core
        in vec2 texcoord;
        out vec4 fragColor;

        uniform sampler2D input_texture;
        uniform float rotation_angle;

        void main() {
            vec2 uv = texcoord;

            if (rotation_angle != 0.0) {
                // Center the coordinates
                vec2 center = vec2(0.5, 0.5);
                uv = uv - center;

                // Apply rotation matrix
                float cos_angle = cos(rotation_angle);
                float sin_angle = sin(rotation_angle);

                mat2 rotation_matrix = mat2(
                    cos_angle, -sin_angle,
                    sin_angle, cos_angle
                );

                uv = rotation_matrix * uv;

                // Restore center
                uv = uv + center;
            }

            // Sample the texture
            if (uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0) {
                fragColor = texture(input_texture, uv);
            } else {
                fragColor = vec4(0.0, 0.0, 0.0, 1.0);  // Black for out-of-bounds
            }
        }
        """

        return self.ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)

    def set_rotation_mode(self, mode: int):
        """Set rotation mode: 0=None, 1=CW, -1=CCW, 2=Music, -2=Random"""
        self.rotation_mode = mode

        if mode == 1:  # Clockwise
            self.rotation_direction = 1
        elif mode == -1:  # Counter-clockwise
            self.rotation_direction = -1
        elif mode == 2:  # Music-driven
            self.rotation_direction = 1  # Will be modulated by audio
        elif mode == -2:  # Random
            self.rotation_direction = 1 if np.random.random() > 0.5 else -1

    @benchmark("update_rotation")
    def update_rotation(self, time: float, animation_speed: float = 1.0, audio_amplitude: float = 0.0, frequency_balance: float = 0.0):
        """Update rotation angle based on mode and audio input"""
        if self.rotation_mode == 0:
            return

        base_speed = self.rotation_speed * animation_speed

        if self.rotation_mode == 2:  # Music-driven
            # Use frequency balance to determine direction
            if frequency_balance > 0.1:
                self.rotation_direction = 1  # High frequencies = clockwise
            elif frequency_balance < -0.1:
                self.rotation_direction = -1  # Low frequencies = counter-clockwise

            # Modulate speed by audio amplitude
            speed_multiplier = 1.0 + (audio_amplitude * 2.0)
            base_speed *= speed_multiplier

        elif self.rotation_mode == -2:  # Random
            # Occasionally change direction randomly
            if np.random.random() < 0.001:  # 0.1% chance per frame
                self.rotation_direction *= -1

        # Update rotation angle
        self.rotation_angle += base_speed * self.rotation_direction * 0.01  # Scale factor

        # Keep angle in reasonable range
        self.rotation_angle = self.rotation_angle % (2 * math.pi)

    @benchmark("render_with_rotation")
    def render_with_rotation(self, input_texture: moderngl.Texture) -> moderngl.Texture:
        """Apply rotation post-processing to the input texture"""
        if self.rotation_mode == 0:
            return input_texture

        # Create VAO if not exists
        if not hasattr(self, 'rotation_vao'):
            self.rotation_vao = self.ctx.vertex_array(
                self.rotation_program, [(self.quad_vbo, "2f 2f", "in_position", "in_texcoord")]
            )

        # Render to framebuffer A
        self.framebuffer_a.use()
        self.framebuffer_a.clear(0.0, 0.0, 0.0, 1.0)

        # Bind input texture
        input_texture.use(location=0)

        # Set uniforms
        self.rotation_program["input_texture"].value = 0
        self.rotation_program["rotation_angle"].value = self.rotation_angle

        # Render fullscreen quad
        self.rotation_vao.render(mode=moderngl.TRIANGLE_STRIP)

        # Return the rendered texture
        return self.framebuffer_a.color_attachments[0]

    def render_to_screen(self, texture: moderngl.Texture, screen_vao: moderngl.VertexArray):
        """Render the final texture to screen using the provided VAO"""
        # Use default framebuffer (screen)
        self.ctx.screen.use()

        # Bind the texture
        texture.use(location=0)

        # Render using the provided VAO (which should have the right shader)
        screen_vao.render(mode=moderngl.TRIANGLE_STRIP)

    def set_rotation_speed(self, speed: float):
        """Set the rotation speed multiplier"""
        self.rotation_speed = max(0.0, speed)

    def get_rotation_angle(self) -> float:
        """Get the current rotation angle in radians"""
        return self.rotation_angle

    def reset_rotation(self):
        """Reset rotation angle to 0"""
        self.rotation_angle = 0.0

    def resize(self, width: int, height: int):
        """Resize the framebuffers"""
        if width != self.width or height != self.height:
            self.width = width
            self.height = height

            # Recreate framebuffers
            self.framebuffer_a.release()
            self.framebuffer_b.release()

            self.framebuffer_a = self._create_framebuffer()
            self.framebuffer_b = self._create_framebuffer()

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'framebuffer_a'):
            self.framebuffer_a.release()
        if hasattr(self, 'framebuffer_b'):
            self.framebuffer_b.release()
        if hasattr(self, 'quad_vbo'):
            self.quad_vbo.release()
        if hasattr(self, 'rotation_vao'):
            self.rotation_vao.release()
        if hasattr(self, 'rotation_program'):
            self.rotation_program.release()

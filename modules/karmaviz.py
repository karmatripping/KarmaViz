import pygame
import numpy as np
import sys
from PyQt5.QtWidgets import QApplication

# Lazy import audio modules to avoid early sounddevice issues
# from modules.audio_handler import AudioProcessor  # Moved to lazy import

qt_app = QApplication(sys.argv)

# Lazy import sounddevice to avoid early library issues
# from sounddevice import _terminate  # Moved to lazy import

from moderngl import (
    create_context,
    BLEND,
    ONE,
    SRC_ALPHA,
    ONE_MINUS_SRC_ALPHA,
    TRIANGLE_FAN,
)

from random import randint, choice
from collections import deque
from typing import Dict

import os
from time import sleep, time
from modules.preset_manager import PresetManager
from modules.logging_config import get_logger

# Get logger for this module
logger = get_logger('karmaviz')
from shaders.shaders import (
    VERTEX_SHADER,
    FRAGMENT_SHADER,
    SPECTROGRAM_VERTEX_SHADER,
    SPECTROGRAM_FRAGMENT_SHADER,
)
from modules.config_menu_qt import ConfigMenuQt  # Add this import
from modules.palette_manager import PaletteManager  # Add palette manager import
from modules.warp_map_manager import WarpMapManager
from modules.shader_compiler import ShaderCompiler, ThreadedShaderCompiler
from modules.shader_manager import ShaderManager 
from modules.benchmark import benchmark, get_performance_monitor

# Global variable to store the selected fullscreen resolution string
selected_fullscreen_res_str = "Native"

try:
    from modules.color_ops import apply_color_parallel
except ImportError:
    logger.error("Warning: Cython color_ops module not found. Using slower numpy version.")
    apply_color_parallel = None

from config.constants import WIDTH, HEIGHT, CHUNK, RATE, DATA_FORMAT

fps = 60


def get_moderngl_dtype_string(numpy_dtype):
    """Convert numpy dtype to ModernGL format string

    ModernGL expects string format specifiers for texture dtypes.
    We'll standardize on float32 for compatibility.
    """
    # Always use float32 for textures regardless of input dtype
    # This ensures compatibility and avoids precision issues
    return "f4"


# Get the ModernGL format string for the current DATA_FORMAT
TEXTURE_DTYPE_STR = get_moderngl_dtype_string(DATA_FORMAT)

# Force SDL to use X11 if not already set (prevents Wayland crashes)
if os.environ.get("SDL_VIDEODRIVER") != "x11":
    os.environ["SDL_VIDEODRIVER"] = "x11"

# Prevent fullscreen windows from minimizing when they lose focus
os.environ["SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS"] = "0"

# Prevent fullscreen windows from minimizing when they lose focus
os.environ["SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS"] = "0"


def create_glow_lookup() -> np.ndarray:
    """Create a lookup table for glow effect calculations.

    Generates a 1D array of values that decrease non-linearly from 1.0,
    raised to the power of 1.5 to create a smooth falloff effect.

    Returns:
        np.ndarray: A 1D array of size 20 containing glow intensity values.
    """
    size = 20
    lookup = np.zeros(size, dtype=DATA_FORMAT)
    values = 1.0 - np.arange(size) / 4.0
    lookup[:] = np.maximum(0.0, values) ** 1.5
    return lookup


GLOW_LOOKUP = create_glow_lookup()


class KarmaVisualizer:
    """Main visualization engine for audio-reactive graphics.

    This class handles all aspects of the visualization including:
    - OpenGL/ModernGL rendering setup
    - Audio data processing and visualization
    - Shader management and compilation
    - Visual effects like waveforms, warping, and color palettes
    - User interaction and configuration
    """

    def __init__(
        self,
        window_size: tuple[int, int],
        audio_processor,
        ctx,
        compiled_programs,
        logo_surface=None,
    ):
        """Initialize the KarmaVisualizer with required components.

        Args:
            window_size: Tuple containing (width, height) of the visualization window
            audio_processor: Object that provides audio data for visualization
            ctx: ModernGL context for rendering
            compiled_programs: Pre-compiled shader programs (can be None for auto-compilation)
            logo_surface: Optional logo surface for initial display
        """
        self.audio_processor = audio_processor

        self.window_size = window_size
        # Unpack for local use, ensures textures/buffers match window
        tex_width, tex_height = self.window_size

        self.clear_feedback_frames: int = 5

        # Initialize mouse interaction state
        self.mouse_interaction_enabled = False
        self.mouse_position = [0.0, 0.0]
        self.mouse_intensity = 1.0  # Default mouse interaction intensity
        self.resolution = [float(tex_width), float(tex_height)]
        
        # Initialize mouse click effects
        self.shockwaves = []  # List of active shockwaves: [x, y, start_time, intensity]
        self.ripples = []     # List of active ripples: [x, y, start_time, intensity]
        self.max_effects = 5  # Maximum number of simultaneous effects

        # Initialize rotation angle
        self.rotation_angle = 0.0

        # Initialize anti-aliasing setting early (needed for texture creation)
        self.anti_aliasing_samples = 4  # Default 4x MSAA for better logo quality

        # Flag to prevent redundant operations during initialization
        self._initializing = True

        # Logo overlay for initial display
        self.logo_surface = logo_surface
        self.logo_overlay_alpha = 1.0 if logo_surface else 0.0  # Start fully visible
        self.logo_overlay_start_time = (
            time() if logo_surface else None
        )  # Start timing immediately
        self.logo_overlay_duration = 3.0  # Show for 3 seconds then fade
        self.logo_fade_duration = 1.0  # Fade out over 1 second

        # Waveform fade-in during logo fade-out
        self.waveform_fade_alpha = (
            0.0 if logo_surface else 1.0
        )  # Start invisible if logo present

        # Initialize spectrogram overlay state
        self.show_spectrogram_overlay = False
        self.spectrogram_data = np.zeros(128, dtype=DATA_FORMAT)
        self.spectrogram_smooth = np.zeros(128, dtype=DATA_FORMAT)
        self.spectrogram_peak = np.zeros(128, dtype=DATA_FORMAT)
        self.spectrogram_falloff = 0.25  # Falloff rate per frame
        self.spectrogram_smoothing = 0.3  # Smoothing factor (0-1)
        self.spectrogram_texture = None
        self.spectrogram_program = None
        self.spectrogram_palette_texture = None  # For palette colors
        self.spectrogram_color_interpolation_speed = 0.1  # Speed of color cycling

        # Initialize ModernGL context first
        try:
            self.ctx = ctx
        except Exception as e:
            logger.debug(f"Failed to initialize ModernGL context: {e}")
            raise

        # Display is already set up in main.py, just setup GL state
        try:
            self.check_gl_setup()  # Setup GL state after display is set
        except Exception as e:
            logger.debug(f"Failed to setup GL state: {e}")
            raise  # Critical error if GL setup fails

        # Removed pattern tracking - now using stackable warp maps

        # Initialize state variables
        self.current_amplitude = 0.0
        self.smoothed_amplitude = 0.0
        self.waveform_style = 0
        self.waveform_scale = 1.0
        self.current_waveform_style = 0
        self.gpu_waveform_random = True  # Enable random GPU waveforms by default
        self.rotation_mode = 1  # 0=None, 1=Clockwise, 2=Counter-clockwise, 3=Beat Driven
        self.rotation_speed = 1.0  # Speed multiplier
        self.rotation_amplitude = 1.0  # Amplitude/intensity multiplier
        self.rotation_beat_direction = 1  # Current direction for beat-driven mode (1 or -1)
        self.rotation_angle = 0.0  # Current rotation angle
        self.bounce_enabled = False
        self.bounce_height = 0.2

        # Removed pattern variables - now using stackable warp maps

        vertices = np.array(
            [
                -1.0,
                -1.0,
                0.0,
                0.0,
                1.0,
                -1.0,
                1.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                -1.0,
                1.0,
                0.0,
                1.0,
            ],
            dtype=DATA_FORMAT,
        )

        self.vbo = self.ctx.buffer(vertices.tobytes())

        # GPU waveform rendering system (always enabled)
        self.waveform_texture = None  # 1D texture for waveform data
        self.waveform_buffer = None   # Buffer for waveform data

        # Initialize waveform manager and compile initial GPU waveform shader
        self.current_waveform_name = 'normal'
        self.active_warp_map_name = None
        self.waveform_program = None
        self.waveform_vao = None

        # Initialize symmetry mode early (needed for warp map selection)
        self.symmetry_mode = 0
        self.current_symmetry = 0
        # Initialize waveform cycling index
        self.warp_map_manager = WarpMapManager()
        self.shader_compiler = ShaderCompiler(self.ctx, self.warp_map_manager)
        self.threaded_shader_compiler = ThreadedShaderCompiler(
            self.ctx, self.warp_map_manager, max_workers=2, enable_shared_contexts=True
        )
        self.shader_manager = ShaderManager(
            warp_map_manager=self.warp_map_manager,
            threaded_shader_compiler=self.threaded_shader_compiler,
        )

        # Set shader manager reference in shader compiler
        self.shader_compiler.set_shader_manager(self.shader_manager)
        self.fallback_program = None  # Fallback shader program
        available_waveforms = self.shader_manager.list_waveforms()
        if available_waveforms:
            available_waveforms.sort()  # Ensure consistent ordering
            try:
                self.current_waveform_index = available_waveforms.index(self.current_waveform_name)
            except ValueError:
                # If 'normal' is not found, start with the first available waveform
                self.current_waveform_index = 0
                self.current_waveform_name = available_waveforms[0]
            logger.debug(f"Available waveforms: {', '.join(available_waveforms)}")
            logger.debug("Starting with waveform: {self.current_waveform_name} ({self.current_waveform_index + 1}/{len(available_waveforms)})")
        else:
            self.current_waveform_index = 0
            logger.error("No waveforms found!")

        # Use pre-compiled programs from splash screen
        if compiled_programs and "main" in compiled_programs:
            self.program = compiled_programs["main"]
            logger.debug("Using pre-compiled main shader from splash screen")
        else:
            # Compile main shader
            logger.debug("Compiling main shader...")
            try:
                self.cycle_to_random_warp_map()
                self.cycle_gpu_waveform()
                fragment_shader_with_waveform = (
                    self.shader_manager.build_full_fragment_shader(
                        waveform_name=self.current_waveform_name,
                        warp_map_name=self.active_warp_map_name,
                    )
                )
                self.program = self.ctx.program(
                    vertex_shader=VERTEX_SHADER,
                    fragment_shader=fragment_shader_with_waveform,
                )
                logger.debug(f"Main shader compiled with waveform: {self.current_waveform_name}")
            except Exception as e:
                logger.error(f"Failed to compile main shader: {e}")
                raise RuntimeError("Main shader compilation failed")

        # Use pre-compiled spectrogram program or compile fallback
        if compiled_programs and "spectrogram" in compiled_programs:
            self.spectrogram_program = compiled_programs["spectrogram"]
            logger.debug("Using pre-compiled spectrogram shader from splash screen")
        else:
            logger.debug("Compiling spectrogram shader...")
            self.spectrogram_program = self.ctx.program(
                vertex_shader=SPECTROGRAM_VERTEX_SHADER,
                fragment_shader=SPECTROGRAM_FRAGMENT_SHADER,
            )
            logger.debug("Spectrogram shader compiled")

        # Always compile waveform shader (needed for waveform-only rendering)
        try:
            program = self.shader_manager.compile_shader(
                self.ctx,
                waveform_name=self.current_waveform_name,
                warp_map_name=None,  # No warp map needed for pure waveform rendering
            )
            self.waveform_program = program
            self.waveform_vao = self.ctx.vertex_array(
                program, [(self.vbo, "2f 2f", "in_position", "in_texcoord")]
            )
            logger.debug(f"Loaded '{self.current_waveform_name}' waveform shader")
        except Exception as e:
            logger.debug("Failed to load '{self.current_waveform_name}' waveform shader: {e}")
            raise RuntimeError(
                "GPU waveform rendering is required but failed to initialize"
            )

        # Create VAO for spectrogram
        self.spectrogram_vao = self.ctx.vertex_array(
            self.spectrogram_program,
            [(self.vbo, "2f 2f", "in_position", "in_texcoord")],
        )

        try:
            self.textures = [
                self.ctx.texture(
                    (tex_width, tex_height),
                    3,
                    samples=self.anti_aliasing_samples,
                    dtype=TEXTURE_DTYPE_STR,
                ),
                self.ctx.texture(
                    (tex_width, tex_height),
                    3,
                    samples=self.anti_aliasing_samples,
                    dtype=TEXTURE_DTYPE_STR,
                ),
            ]
            logger.debug("Created main textures with {self.anti_aliasing_samples}x multisampling")
        except Exception as e:
            # Fallback to non-multisampled textures
            self.textures = [
                self.ctx.texture((tex_width, tex_height), 3, dtype=TEXTURE_DTYPE_STR),
                self.ctx.texture((tex_width, tex_height), 3, dtype=TEXTURE_DTYPE_STR),
            ]
            logger.error(f"Multisampling not supported for textures, using standard textures: {e}")
            # Disable multisampling for consistency
            self.anti_aliasing_samples = 0

        self.current_texture = 0

        # Feedback texture doesn't need multisampling
        self.feedback_texture = self.ctx.texture(
            (tex_width, tex_height), 3, dtype=TEXTURE_DTYPE_STR
        )
        self.feedback_fbo = self.ctx.framebuffer(
            color_attachments=[self.feedback_texture]
        )

        # Create a separate overlay texture for logo and other overlays
        # This texture will never be part of the feedback loop
        self.overlay_texture = self.ctx.texture(
            (tex_width, tex_height), 4, dtype=TEXTURE_DTYPE_STR  # RGBA for transparency
        )
        self.overlay_fbo = self.ctx.framebuffer(
            color_attachments=[self.overlay_texture]
        )

        self.main_fbo = self.ctx.framebuffer(
            color_attachments=[self.textures[0]]
        )

        self.vao = self.ctx.vertex_array(
            self.program, [(self.vbo, "2f 2f", "in_position", "in_texcoord")]
        )

        self.time = 0.0
        self.clear_texture()

        # Initialize buffers using constants and unpacked dimensions
        initial_chunk_size = (
            audio_processor.get_chunk_size()
            if hasattr(audio_processor, "get_chunk_size")
            else CHUNK
        )
        self.audio_buffer = np.empty(initial_chunk_size, dtype=DATA_FORMAT)
        self.pixel_buffer = np.empty((tex_height, tex_width, 3), dtype=DATA_FORMAT)

        # Setup Blend state
        self.ctx.enable(BLEND)
        # Start with standard alpha blending
        self.ctx.blend_func = SRC_ALPHA, ONE_MINUS_SRC_ALPHA

        # Mood and Palette initialization
        self.mood_buffer = deque(maxlen=5)
        self.current_mood = {
            "energy": 0.5,  # Low to high energy
            "warmth": 0.5,  # Cool to warm tonality
        }

        # Initialize palette manager
        self.palette_manager = PaletteManager()

        # Initialize integrated warp map system

        # Initialize shader manager with warp map manager reference

        # Warp map state
        self.active_warp_map_name = None
        self.warp_intensity = 0.5  # Default intensity (increased for testing)
        self.active_warp_map_index = -1  # -1 means no warp map active
        self.persistent_warp_map = None  # Warp map selected in config menu
        self.warp_map_locked = False  # Prevents automatic warp map changes when True
        self.waveform_locked = False

        # Beat-based transition system
        self.beats_per_change = 16  # Default beats per change
        self.beat_counter = 0  # Count beats for transitions
        self.beat_detected = False
        self.transitions_paused = False  # Can pause automatic transitions

        # Get available warp maps
        available_warp_maps = self.warp_map_manager.get_all_warp_maps()
        logger.debug(f"Loaded {len(available_warp_maps)} warp maps for integrated rendering")

        # Randomly select a warp map on startup
        if available_warp_maps:
            # Get a random filename key from the warp map manager
            warp_map_keys = list(self.warp_map_manager.warp_maps.keys())
            if warp_map_keys:
                random_key = choice(warp_map_keys)
                random_warp_map = self.warp_map_manager.get_warp_map(random_key)
                self.select_random_warp_map(random_key)
                logger.debug(f"Randomly selected warp map on startup: {random_warp_map.name} (key: {random_key})")

        # Palette selection settings
        self.palette_mode = "Mood-based"  # "Mood-based", "Fixed", "Random"
        self.selected_palette_name = "Auto (Mood-based)"
        self.fixed_palette = None  # For when a specific palette is selected

        # Color cycling variables - now using palette manager
        all_palettes = self.palette_manager.get_all_palettes()
        if all_palettes:
            random_palette = choice(all_palettes)
            self.current_palette = random_palette.colors
        else:
            # Fallback if no palettes loaded
            self.current_palette = [(255, 0, 0), (255, 100, 0), (255, 255, 0), (0, 255, 0),
                                   (0, 100, 255), (150, 0, 255), (255, 50, 0)]

        # Enhanced color interpolation system
        self.current_palette = self.current_palette
        self.target_palette = self.current_palette.copy()  # Target palette for smooth transitions
        self.palette_transition_progress = 1.0  # 1.0 = fully transitioned, 0.0 = starting transition
        self.palette_transition_speed = 0.02  # How fast to transition between palettes

        self.color_index = 0
        self.color_time = 0
        self.color_transition_smoothness = 0.1  # How smooth color transitions are (lower = smoother)
        self.current_interpolated_color = None  # Store smoothly interpolated color
        self.mood_update_counter = 0  # Counter for mood updates

        # Simplified animation speed system
        self.animation_speed = 0.0  # Base animation speed (0.0 to 5.0)
        self.audio_speed_boost = 1.0  # How much audio affects speed (1.0 to 3.0)
        self.smoothed_audio_boost = 0.0  # Smoothed audio contribution

        # Other control variables
        self.palette_rotation_speed = 1.0
        self.multiplier = 1.0  # Added for Numpad 8/2 control
        self.rotation_multiplier = 1.0
        self.invert_rotation_direction = False
        self.rotation_mode = 0
        self.pulse_enabled = False

        # Single centered waveform (no stereo separation)

        # Beat tracking
        self.last_beat_time = 0.0
        self.beat_interval = 0.2
        self.beats_since_last_change = 0  # Initialize beat counter
        self.excess_energy = 0.0

        # Effect intensities & modes
        self.pulse_intensity = 1.0
        self.pulse_intensity_multiplier = 0.5
        self.trail_intensity = 0.0
        self.glow_intensity = 0.9
        self.glow_radius = 0.065  # Default glow radius (matches previous hardcoded value)
        self.kaleidoscope_sections = 10
        self.smoke_intensity = 0.0
        self.warp_first_enabled = False  # New: Toggle warp/symmetry order

        # Debug options
        self.debug_beats = False

        # Fullscreen resolution setting (set by config menu)
        self.selected_fullscreen_resolution = "Native"

        # Set initial values for uniforms
        if "mouse_position" in self.program:
            self.program["mouse_position"] = self.mouse_position  # type: ignore
        if "resolution" in self.program:
            self.program["resolution"] = self.resolution  # type: ignore
        if "mouse_enabled" in self.program:
            self.program["mouse_enabled"] = False  # type: ignore
        if "mouse_intensity" in self.program:
            self.program["mouse_intensity"] = self.mouse_intensity  # type: ignore
        if "warp_first" in self.program:
            self.program["warp_first"] = self.warp_first_enabled  # type: ignore
        if "bounce_enabled" in self.program:
            self.program["bounce_enabled"] = self.bounce_enabled  # type: ignore
        if "bounce_height" in self.program:
            self.program["bounce_height"] = float(self.bounce_height)  # type: ignore

        # Load compiled programs
        self.compiled_programs = compiled_programs

        # Add bounce effect parameters
        self.bounce_enabled = False
        self.bounce_height = 0.0
        self.bounce_velocity = 0.0
        self.bounce_intensity_multiplier = 1.0
        self.bounce_decay = 1.2  # Even slower decay
        self.bounce_spring = 4.0  # Much softer spring for bigger motion
        self.bounce_damping = 0.3  # Less damping for more bounce
        self.last_bounce_time = time()

        # Initialize GPU waveform system
        self.initialize_gpu_waveform()

        # Silence detection and logo fade parameters
        self.silence_threshold = 0.01  # Amplitude threshold for silence detection (increased for easier testing)
        self.silence_duration_threshold = 2.0  # Seconds of silence before showing logo
        self.silence_start_time = None  # When silence started
        self.logo_fade_alpha = 0.0  # Current logo opacity (0.0 to 1.0)
        self.logo_fade_speed = 1.5  # Fade speed (alpha change per second)
        self.logo_visible = False  # Whether logo is currently visible
        self.logo_texture = None  # Logo texture for rendering
        self.logo_size = (0, 0)  # Logo dimensions
        self.logo_test_mode = False  # For testing logo display manually
        self.logo_hue_shift = 0.0  # Current hue shift for rainbow effect (0.0 to 1.0)
        self.logo_hue_speed = 0.1
        self.logo_pulse_scale = 1.0
        self.logo_heartbeat_bpm = 60
        self.logo_pulse_intensity = 0.05

        # Waveform brightness fade during silence
        self.waveform_brightness_multiplier = 1.0
        self.waveform_silence_brightness = 0.1  # Target brightness during silence (10%)

        # FPS tracking for title bar display
        self.fps_counter = 0
        self.fps_timer = 0.0
        self.current_fps = 0.0
        self.fps_update_interval = 1.0  # Update FPS display every 1 second
        self.last_frame_time = time()  # Track actual frame time
        self.frame_times = []  # Store recent frame times for rolling average
        self.max_frame_samples = 60  # Keep last 60 frames for averaging

        # Load logo texture for silence display
        self.load_logo_texture()

        # Initialization complete - allow normal operations
        self._initializing = False

    def initialize_gpu_waveform(self):
        """Initialize GPU waveform rendering system"""
        try:
            # Get actual chunk size from audio processor
            actual_chunk_size = (
                self.audio_processor.get_chunk_size()
                if hasattr(self.audio_processor, "get_chunk_size")
                else CHUNK
            )
            
            # DIAGNOSTIC: Log buffer size mismatch
            logger.debug(f"🔍 GPU Waveform Init - Audio chunk size: {actual_chunk_size}")
            
            # Use actual chunk size or minimum of 1024 for GPU efficiency
            waveform_samples = max(128, actual_chunk_size)
            logger.debug(f"🔍 GPU Waveform Init - Using waveform_samples: {waveform_samples}")
            
            self.waveform_buffer = np.zeros(waveform_samples, dtype=DATA_FORMAT)

            # Create 1D texture using 2D texture with height=1 (ModernGL doesn't support true 1D textures)
            self.waveform_texture = self.ctx.texture(
                (waveform_samples, 1), 1, dtype=TEXTURE_DTYPE_STR
            )

            # Set texture parameters for smooth sampling
            self.waveform_texture.filter = (self.ctx.LINEAR, self.ctx.LINEAR)
            self.waveform_texture.repeat_x = False
            self.waveform_texture.repeat_y = False

            # Create frequency data texture for lightning waveform (FFT data)
            fft_samples = 256  # Half of waveform samples for FFT
            self.fft_buffer = np.zeros(fft_samples, dtype=DATA_FORMAT)
            self.fft_texture = self.ctx.texture(
                (fft_samples, 1), 1, dtype=TEXTURE_DTYPE_STR
            )
            self.fft_texture.filter = (self.ctx.LINEAR, self.ctx.LINEAR)
            self.fft_texture.repeat_x = False
            self.fft_texture.repeat_y = False

            logger.debug(f"GPU waveform system initialized with {waveform_samples} waveform samples and {fft_samples} FFT samples")

        except Exception as e:
            logger.error(f"Error initializing GPU waveform system: {e}")
            raise RuntimeError("GPU waveform system initialization failed")

    @benchmark("update_gpu_waveform")
    def update_gpu_waveform(self, audio_data):
        """Update GPU waveform texture with new audio data"""
        # Recreate textures if they were released during window resize
        if self.waveform_texture is None:
            try:
                logger.debug("Recreating GPU waveform textures after resize...")

                # Get actual chunk size from audio processor
                actual_chunk_size = (
                    self.audio_processor.get_chunk_size()
                    if hasattr(self.audio_processor, "get_chunk_size")
                    else CHUNK
                )
                
                # Use actual chunk size or minimum of 128 for GPU efficiency
                waveform_samples = max(128, actual_chunk_size)
                logger.debug(f"🔍 GPU Waveform Recreate - Using waveform_samples: {waveform_samples}")
                
                self.waveform_buffer = np.zeros(waveform_samples, dtype=DATA_FORMAT)

                # Create 1D texture using 2D texture with height=1 (ModernGL doesn't support true 1D textures)
                self.waveform_texture = self.ctx.texture(
                    (waveform_samples, 1), 1, dtype=TEXTURE_DTYPE_STR
                )

                # Set texture parameters for smooth sampling
                self.waveform_texture.filter = (self.ctx.LINEAR, self.ctx.LINEAR)
                self.waveform_texture.repeat_x = False
                self.waveform_texture.repeat_y = False

                # Create frequency data texture for lightning waveform (FFT data)
                fft_samples = 256  # Half of waveform samples for FFT
                self.fft_buffer = np.zeros(fft_samples, dtype=DATA_FORMAT)
                self.fft_texture = self.ctx.texture(
                    (fft_samples, 1), 1, dtype=TEXTURE_DTYPE_STR
                )
                self.fft_texture.filter = (self.ctx.LINEAR, self.ctx.LINEAR)
                self.fft_texture.repeat_x = False
                self.fft_texture.repeat_y = False

                logger.debug("GPU waveform textures recreated successfully")

            except Exception as e:
                logger.error(f"Error recreating GPU waveform textures: {e}")
                return

        try:
            # Extract raw audio data from AudioData object
            raw_audio = audio_data.raw_data
            
            # Process audio data for GPU upload - optimized for performance
            if len(raw_audio.shape) > 1:
                # Use left channel for mono waveform
                mono_data = raw_audio[:, 0]
            else:
                mono_data = raw_audio

            # Ensure data is in the correct dtype for processing
            mono_data = np.asarray(mono_data, dtype=DATA_FORMAT)

            # Optimized resampling using vectorized operations
            if len(mono_data) != len(self.waveform_buffer):
                # Use more efficient resampling with pre-computed indices
                # Cache indices based on both input and output lengths to handle chunk size changes
                cache_key = (len(mono_data), len(self.waveform_buffer))
                if not hasattr(self, '_resample_cache_key') or self._resample_cache_key != cache_key:
                    self._resample_indices = np.linspace(0, len(mono_data) - 1, len(self.waveform_buffer), dtype=np.int32)
                    self._resample_weights = self._resample_indices - self._resample_indices.astype(np.int32)
                    self._resample_cache_key = cache_key

                # Fast linear interpolation using pre-computed indices
                idx_floor = self._resample_indices.astype(np.int32)
                idx_ceil = np.clip(idx_floor + 1, 0, len(mono_data) - 1)
                weights = self._resample_weights

                resampled_data = (mono_data[idx_floor] * (1 - weights) +
                                mono_data[idx_ceil] * weights)
            else:
                resampled_data = mono_data

            # Optimized normalization using vectorized operations
            abs_data = np.abs(resampled_data)
            max_val = np.max(abs_data)
            if max_val > 0:
                # Vectorized division and assignment
                np.divide(resampled_data, max_val, out=self.waveform_buffer)
            else:
                self.waveform_buffer.fill(0.0)

            # Upload waveform data to GPU texture (avoid unnecessary copy)
            if self.waveform_buffer.dtype != np.float32:
                gpu_buffer = self.waveform_buffer.astype(np.float32)
            else:
                gpu_buffer = self.waveform_buffer
            self.waveform_texture.write(gpu_buffer.tobytes())

            # Upload pre-calculated FFT data for lightning waveform
            if hasattr(self, "fft_texture") and self.fft_texture is not None:
                # Use pre-calculated FFT data from AudioData object (already logarithmically scaled)
                fft_data = audio_data.fft_data

                # Optimized FFT resampling
                if len(fft_data) != len(self.fft_buffer):
                    # Use more efficient resampling for FFT data
                    # Cache indices based on both input and output lengths to handle chunk size changes
                    fft_cache_key = (len(fft_data), len(self.fft_buffer))
                    if not hasattr(self, '_fft_resample_cache_key') or self._fft_resample_cache_key != fft_cache_key:
                        self._fft_resample_indices = np.linspace(0, len(fft_data) - 1, len(self.fft_buffer), dtype=np.int32)
                        self._fft_resample_weights = self._fft_resample_indices - self._fft_resample_indices.astype(np.int32)
                        self._fft_resample_cache_key = fft_cache_key

                    idx_floor = self._fft_resample_indices.astype(np.int32)
                    idx_ceil = np.clip(idx_floor + 1, 0, len(fft_data) - 1)
                    weights = self._fft_resample_weights

                    resampled_fft = (fft_data[idx_floor] * (1 - weights) +
                                   fft_data[idx_ceil] * weights)
                else:
                    resampled_fft = fft_data

                # FFT data is already logarithmically scaled and normalized from AudioData
                # Just copy to buffer without additional normalization
                np.copyto(self.fft_buffer, resampled_fft)

                # Upload FFT data to GPU texture (avoid unnecessary copy)
                if self.fft_buffer.dtype != np.float32:
                    fft_gpu_buffer = self.fft_buffer.astype(np.float32)
                else:
                    fft_gpu_buffer = self.fft_buffer
                self.fft_texture.write(fft_gpu_buffer.tobytes())

        except Exception as e:
            logger.error(f"Error updating GPU waveform: {e}")

    def load_logo_texture(self):
        """Load the KarmaViz logo texture for silence display with proper transparency and enhanced anti-aliasing"""
        try:
            logger.debug(f"Loading logo texture...")
            logger.debug(f"ModernGL context available: {self.ctx is not None}")

            import pygame

            # Load the logo image with transparency support
            logo_surface = pygame.image.load("karmaviz_logo.png").convert_alpha()
            original_size = logo_surface.get_size()

            # Calculate supersampling factor for better anti-aliasing
            # Higher factor = better quality but more memory usage
            supersample_factor = 2 if self.anti_aliasing_samples > 0 else 1

            if supersample_factor > 1:
                # Scale up the logo for supersampling anti-aliasing
                supersampled_size = (
                    original_size[0] * supersample_factor,
                    original_size[1] * supersample_factor,
                )
                logo_surface = pygame.transform.smoothscale(
                    logo_surface, supersampled_size
                )
                self.logo_size = supersampled_size
                logger.debug("Logo supersampled to {supersample_factor}x: size={self.logo_size}")
            else:
                self.logo_size = original_size
                logger.debug(f"Logo loaded at original size: {self.logo_size}")

            # Get RGBA data to preserve transparency
            logo_data = pygame.image.tostring(logo_surface, "RGBA")

            # Convert to numpy array and reshape
            logo_array = np.frombuffer(logo_data, dtype=np.uint8)
            logo_array = logo_array.reshape((self.logo_size[1], self.logo_size[0], 4))

            # Flip vertically to fix upside-down issue (OpenGL vs Pygame coordinate systems)
            logo_array = np.flipud(logo_array)

            # Normalize to 0-1 range for OpenGL (always use float32 for texture compatibility)
            logo_rgba = logo_array.astype(np.float32) / 255.0

            # Create OpenGL texture with RGBA format for transparency
            # Note: Logo texture cannot use multisampling because we need to write data directly to it
            # ModernGL doesn't allow writing to multisampled textures
            self.logo_texture = self.ctx.texture(
                self.logo_size, 4, dtype=TEXTURE_DTYPE_STR
            )
            logger.debug("Logo texture created (no MSAA - required for direct write): size={self.logo_size}")
            self.logo_texture.write(logo_rgba.tobytes())
            logger.debug(f"Logo texture data written successfully")

            # Set texture parameters for smooth anti-aliased rendering
            # Use LINEAR_MIPMAP_LINEAR for best quality when available
            self.logo_texture.filter = (
                self.ctx.LINEAR_MIPMAP_LINEAR,
                self.ctx.LINEAR,
            )  # Use mipmap filtering for better anti-aliasing
            self.logo_texture.repeat_x = False
            self.logo_texture.repeat_y = False

            # Generate mipmaps for better anti-aliasing at different scales
            try:
                self.logo_texture.build_mipmaps()
                logger.debug(f"Logo texture mipmaps generated for enhanced anti-aliasing")
            except Exception as e:
                logger.error(f"Could not generate mipmaps (using linear filtering): {e}")
                # Fallback to simple linear filtering
                self.logo_texture.filter = (
                    self.ctx.LINEAR,
                    self.ctx.LINEAR,
                )

            logger.debug("Logo texture loaded with enhanced anti-aliasing: {self.logo_size[0]}x{self.logo_size[1]}")

        except Exception as e:
            logger.error(f"Could not load logo texture: {e}")
            self.logo_texture = None
            self.logo_size = (0, 0)

    def check_gl_setup(self):
        # Reinitialize ModernGL context
        try:
            self.ctx = create_context()
            self.ctx.enable(BLEND)
            self.ctx.blend_func = (
                SRC_ALPHA,
                ONE_MINUS_SRC_ALPHA,
            )

            # Enable anti-aliasing in OpenGL context
            try:
                # Enable multisampling using the context property
                self.ctx.multisample = True
                logger.debug("ModernGL multisampling enabled via context property")

                # Enable line smoothing for better line quality
                try:
                    # Note: LINE_SMOOTH is deprecated in core profile, but MULTISAMPLE handles it
                    # We'll rely on MSAA for line smoothing
                    logger.debug("Line smoothing handled by MSAA")
                except:
                    pass

            except Exception as e:
                logger.error(f"Could not enable multisampling: {e}")

            # Set line width for better visibility
            try:
                # Set a reasonable line width (some drivers limit this)
                self.ctx.line_width = 5.0
                logger.debug("Line width set to 2.0")
            except Exception as e:
                logger.error(f"Could not set line width: {e}")

            # Recreate textures
            self.textures = [
                self.ctx.texture((WIDTH, HEIGHT), 3, dtype=TEXTURE_DTYPE_STR)
                for _ in range(2)
            ]
            self.clear_texture()
        except Exception as e:
            logger.error(f"Error reinitializing ModernGL context: {e}")
            raise  # Critical error if context fails

    def clear_texture(self):
        # For multisampled textures, we need to clear using framebuffer operations
        # instead of writing directly to the texture

        # Temporarily disable blending for clearing to prevent accumulation
        try:
            self.ctx.disable(BLEND)
        except:
            pass  # Blending might not be enabled yet

        # Clear main framebuffer (which contains multisampled textures)
        if hasattr(self, 'main_fbo') and self.main_fbo is not None:
            self.main_fbo.use()
            self.ctx.clear(0.0, 0.0, 0.0, 1.0)  # Clear to black
            logger.debug("Cleared main framebuffer (multisampled textures)")

        # Clear feedback framebuffer (usually not multisampled)
        if hasattr(self, 'feedback_fbo') and self.feedback_fbo is not None:
            self.feedback_fbo.use()
            self.ctx.clear(0.0, 0.0, 0.0, 1.0)  # Clear to black
            logger.debug("Cleared feedback framebuffer")

        # For non-multisampled textures, try direct write (fallback)
        zeros = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
        for i, tex in enumerate(self.textures):
            try:
                tex.write(zeros.tobytes())
                logger.debug(f"Cleared texture {i} directly")
            except Exception as e:
                # This is expected for multisampled textures
                logger.debug(f"Texture {i} cleared via framebuffer (multisampled): {e}")

        # Return to screen framebuffer and restore blending
        self.ctx.screen.use()
        self.ctx.enable(BLEND)
        self.ctx.blend_func = SRC_ALPHA, ONE_MINUS_SRC_ALPHA
    @benchmark("analyze_mood")
    def analyze_mood(self, audio_data):
        try:
            # Simple RMS energy calculation
            # Use numpy for efficient vectorized operations
            audio_np = np.asarray(audio_data, dtype=DATA_FORMAT)


            # Optimized RMS calculation using vectorized operations
            rms_energy = np.sqrt(np.mean(np.square(audio_np)))
            energy = np.clip(rms_energy * 4.0, 0.0, 1.0)  # Vectorized clipping

            # Optimized zero crossing detection using vectorized operations
            # Use sign changes instead of multiplication for better performance
            sign_changes = np.diff(np.signbit(audio_np))
            zero_crossings = np.count_nonzero(sign_changes)

            current_chunk_size = (
                self.audio_processor.get_chunk_size()
                if hasattr(self.audio_processor, "get_chunk_size")
                else CHUNK
            )
            warmth = 1.0 - np.clip(zero_crossings / (current_chunk_size * 0.25), 0.0, 1.0)

            self.mood_buffer.append((energy, warmth))
            if len(self.mood_buffer) > 0:
                # Vectorized average calculation using numpy
                mood_array = np.array(self.mood_buffer, dtype=DATA_FORMAT)
                self.current_mood = {
                    "energy": np.mean(mood_array[:, 0]),
                    "warmth": np.mean(mood_array[:, 1]),
                }

            return self.current_mood

        except Exception as e:
            logger.debug(f"Mood analysis error: {e}")
            return self.current_mood

    def lerp_color(self, color1, color2, t):
        """Linear interpolation between two RGB colors using optimized numpy operations"""
        t = np.clip(t, 0.0, 1.0)

        # Use pre-allocated arrays for better performance
        if not hasattr(self, '_color_temp_arrays'):
            self._color_temp_arrays = {
                'color1_arr': np.zeros(3, dtype=np.float32),
                'color2_arr': np.zeros(3, dtype=np.float32),
                'result_arr': np.zeros(3, dtype=np.float32)
            }

        temp = self._color_temp_arrays
        temp['color1_arr'][:] = color1
        temp['color2_arr'][:] = color2

        # Optimized interpolation using in-place operations
        inv_t = 1.0 - t
        np.multiply(temp['color1_arr'], inv_t, out=temp['result_arr'])
        temp['result_arr'] += temp['color2_arr'] * t

        return tuple(temp['result_arr'].astype(np.int32))

    def lerp_palette(self, palette1, palette2, t):
        """Linear interpolation between two palettes using optimized vectorized operations"""
        t = np.clip(t, 0.0, 1.0)

        # Cache palette arrays to avoid repeated conversions
        max_len = max(len(palette1), len(palette2))
        cache_key = (id(palette1), id(palette2), max_len)

        if not hasattr(self, '_palette_cache') or self._palette_cache.get('key') != cache_key:
            # Extend shorter palette by repeating last color
            p1_extended = palette1 + [palette1[-1]] * (max_len - len(palette1))
            p2_extended = palette2 + [palette2[-1]] * (max_len - len(palette2))

            self._palette_cache = {
                'key': cache_key,
                'p1': np.array(p1_extended, dtype=np.float32),
                'p2': np.array(p2_extended, dtype=np.float32),
                'result': np.zeros((max_len, 3), dtype=np.float32)
            }

        cache = self._palette_cache

        # Optimized vectorized interpolation using in-place operations
        inv_t = 1.0 - t
        np.multiply(cache['p1'], inv_t, out=cache['result'])
        cache['result'] += cache['p2'] * t

        return cache['result'].astype(np.int32).tolist()

    def smooth_color_transition(self, current_color, target_color, smoothness=0.1):
        """Smooth exponential interpolation between colors"""
        return self.lerp_color(current_color, target_color, smoothness)

    def get_mood_palette(self, mood):
        """Get a palette based on current palette mode and settings"""
        try:
            if self.palette_mode == "Fixed" and self.fixed_palette:
                # Use the fixed selected palette
                return self.fixed_palette
            elif self.palette_mode == "Random":
                # Select a random palette
                all_palettes = self.palette_manager.get_all_palettes()
                if all_palettes:
                    random_palette = choice(all_palettes)
                    return random_palette.colors
            else:
                # Default to mood-based selection
                chosen_palette_colors = self.palette_manager.get_mood_palette(mood)
                return chosen_palette_colors
        except Exception as e:
            logger.error(f"Error getting mood palette: {e}")
            # Fallback to current palette or a default
            if self.current_palette:
                return self.current_palette
            else:
                return [(255, 0, 0), (255, 100, 0), (255, 255, 0), (0, 255, 0),
                       (0, 100, 255), (150, 0, 255), (255, 50, 0)]

    def set_palette_mode(self, mode):
        """Set the palette selection mode"""
        self.palette_mode = mode
        logger.debug(f"Palette mode set to: {mode}")

    def set_selected_palette(self, palette_name):
        """Set the selected palette by name with smooth transition"""
        self.selected_palette_name = palette_name

        if palette_name == "Auto (Mood-based)":
            self.palette_mode = "Mood-based"
            self.fixed_palette = None
            # Trigger immediate mood-based palette selection
            new_palette = self.get_mood_palette(self.current_mood)
            self._start_palette_transition(new_palette)
        elif palette_name == "Random":
            self.palette_mode = "Random"
            self.fixed_palette = None
            # Select a random palette immediately
            new_palette = self.get_mood_palette(self.current_mood)
            self._start_palette_transition(new_palette)
        else:
            # It's a specific palette name
            self.palette_mode = "Fixed"
            palette_info = self.palette_manager.get_palette_by_name(palette_name)
            if palette_info:
                self.fixed_palette = palette_info.colors
                self._start_palette_transition(palette_info.colors)
                logger.debug(f"Selected fixed palette: {palette_name}")
            else:
                logger.error(f"Warning: Palette '{palette_name}' not found, falling back to mood-based")
                self.palette_mode = "Mood-based"
                self.fixed_palette = None

    def _start_palette_transition(self, new_palette):
        """Start a smooth transition to a new palette"""
        if new_palette != self.target_palette:
            self.target_palette = new_palette
            self.palette_transition_progress = 0.0

    @benchmark("draw_waveform")
    def draw_waveform(self, color):
        audio_data = self.audio_processor.get_audio_data()

        if not audio_data:  # Check if audio_data is None
            # Handle case where no audio data is available
            return  # Exit early if no audio

        # Optimized amplitude calculation using vectorized operations
        raw_data = np.asarray(audio_data.raw_data, dtype=DATA_FORMAT)
        self.current_amplitude = np.mean(np.abs(raw_data)) * 8.0

        # Optimized smoothing calculation
        amplitude_scaled = abs(self.current_amplitude * 0.2)
        self.smoothed_amplitude = 0.1 * (self.smoothed_amplitude + amplitude_scaled)

        if self.show_spectrogram_overlay:
            self.update_spectrogram_data(audio_data.raw_data)

        self.beat_detected = audio_data.beat_detected

        # Handle pulse effect on beat detection regardless of transitions
        if self.beat_detected:
            # Update pulse intensity based on beat
            self.pulse_intensity = (
                0.1 + self.audio_processor.excess_energy + self.smoothed_amplitude
            ) * self.pulse_intensity_multiplier

        # Handle bounce effect on beat detection
        if self.beat_detected and self.bounce_enabled:
            # Add upward velocity on beat with stronger initial impulse
            base_velocity = 1.0  # Much larger base velocity
            self.bounce_velocity = min(
                12.0,  # Much higher max velocity
                (
                    base_velocity
                    + self.audio_processor.excess_energy * 4.0
                    + self.smoothed_amplitude * 4.0
                )
                * self.bounce_intensity_multiplier,
            )
            self.bounce_height = 0.0  # Reset height to allow for full bounce
            self.last_bounce_time = time()

        # Handle beat-based pattern transitions (only if not locked by config menu)
        if self.beat_detected and not self.transitions_paused and not self.warp_map_locked and not self.waveform_locked:
            self.beat_counter += 1
            logger.debug(f"🎵 Beat detected: {self.beat_counter} of {self.beats_per_change}")
            if self.beat_counter >= self.beats_per_change:
                self.beat_counter = 0
                self.cycle_to_random_warp_map()
                logger.debug("🎵 Beat transition: Complete warp map change after {self.beats_per_change} beats")
                if self.gpu_waveform_random:
                    self.cycle_gpu_waveform()
                    logger.debug("🎵 Beat transition: Complete waveform change after {self.beats_per_change} beats")
                if self.symmetry_mode == -1:
                    self.current_symmetry = randint(0, 10)
                else:
                    self.current_symmetry = self.symmetry_mode

        else:
            # Optimized spring physics calculations
            time_since_bounce = time() - self.last_bounce_time
            dt = min(time_since_bounce, 1.0 / 60.0)  # Cap delta time to avoid huge jumps

            # Pre-compute common values for efficiency
            spring_force = -self.bounce_spring * self.bounce_height
            damping_factor = 1.0 - self.bounce_damping * dt
            decay_factor = np.exp(-self.bounce_decay * time_since_bounce)

            # Apply spring force to velocity with damping (vectorized)
            self.bounce_velocity = (self.bounce_velocity + spring_force * dt) * damping_factor

            # Update position
            self.bounce_height += self.bounce_velocity * dt

            # Apply decay to overall motion (vectorized)
            self.bounce_height *= decay_factor
            self.bounce_velocity *= decay_factor

        # --- Mood & Color Update ---
        if not self.pulse_enabled:
            self.pulse_intensity = 0.0
        self.mood_update_counter += 1
        if self.mood_update_counter >= 60:
            self.mood_update_counter = 0
            try:
                mood = self.analyze_mood(audio_data.raw_data)
                new_palette = self.get_mood_palette(mood)

                # Check if palette actually changed
                if new_palette != self.target_palette:
                    # Start smooth transition to new palette
                    self.target_palette = new_palette
                    self.palette_transition_progress = 0.0

            except Exception as e:
                logger.debug(f"Mood update error: {e}")

        # Update palette transition
        if self.palette_transition_progress < 1.0:
            self.palette_transition_progress = min(1.0,
                self.palette_transition_progress + self.palette_transition_speed)

            # Interpolate between current and target palette
            self.current_palette = self.lerp_palette(
                self.current_palette,
                self.target_palette,
                self.palette_transition_progress
            )
        if self.current_palette:
            base_frames = 5
            cycle_frames = max(
                3,
                int(
                    base_frames
                    * (1.0 - self.smoothed_amplitude * 3.5)
                    / (self.palette_rotation_speed * self.color_cycle_speed_multiplier)
                ),
            )
            self.color_time += 1
            if self.color_time >= cycle_frames:
                self.color_time = 0
                self.color_index = (self.color_index + 1) % len(self.current_palette)

            # Enhanced smooth color interpolation
            current_color = self.current_palette[self.color_index % len(self.current_palette)]
            next_color = self.current_palette[(self.color_index + 1) % len(self.current_palette)]

            # Vectorized smooth step calculation
            linear_blend = self.color_time / cycle_frames
            smooth_blend = linear_blend * linear_blend * (3.0 - 2.0 * linear_blend)

            # Use vectorized color interpolation
            current_color_arr = np.array(current_color, dtype=np.float32)
            next_color_arr = np.array(next_color, dtype=np.float32)
            color_arr = current_color_arr * (1 - smooth_blend) + next_color_arr * smooth_blend
            color = tuple(color_arr.astype(np.int32))
            self.current_interpolated_color = color
        else:
            # Ensure current_interpolated_color is None when no palette is available
            self.current_interpolated_color = None
        # --------------------------

        # Calculate rotation angle based on mode and audio input
        if self.rotation_mode != 0:
            # Base rotation speed with configurable multiplier (independent of animation speed)
            base_speed = (
                self.rotation_speed * 0.1
            )  # Fixed multiplier for consistent rotation

            if self.rotation_mode == 0:  # None/Disabled
                rotation_angle = 0.0
            elif self.rotation_mode == 1:  # Clockwise
                rotation_angle = base_speed * self.rotation_amplitude
            elif self.rotation_mode == 2:  # Counter-clockwise
                rotation_angle = -base_speed * self.rotation_amplitude
            elif self.rotation_mode == 3:  # Beat Driven
                # Flip direction on each beat
                if self.beat_detected:
                    self.rotation_beat_direction *= -1
                rotation_angle = base_speed * self.rotation_beat_direction * self.rotation_amplitude
            else:
                rotation_angle = 0.0
        else:
            # Process any completed shader compilations
            self.shader_manager.process_shader_compilation_results(
                self._handle_program_update
            )

            rotation_angle = 0.0

        # Store rotation angle for shader
        self.rotation_angle = rotation_angle

        # --- GPU Waveform Setup (integrated with main shader) ---
        # Update GPU waveform data
        self.update_gpu_waveform(audio_data)
        # The actual rendering will happen in the main render() method

    @benchmark("render")
    def render(self):
        try:
            # Process any completed shader compilations
            self.shader_manager.process_shader_compilation_results(
                self._handle_program_update
            )

            # Update mouse enabled state and intensity
            self.program["mouse_enabled"] = self.mouse_interaction_enabled  # type: ignore
            if "mouse_intensity" in self.program:
                self.program["mouse_intensity"] = self.mouse_intensity  # type: ignore
            
            # Update click effects and pass to shader
            self.update_click_effects()
            self._update_click_effect_uniforms()

            current_time = time()

            # FPS tracking and title bar update
            frame_time = current_time - self.last_frame_time
            self.last_frame_time = current_time

            # Store frame time for rolling average (filter out unreasonable values)
            if 0.001 < frame_time < 0.1:  # Between 1ms and 100ms
                self.frame_times.append(frame_time)

            # Keep only recent frames
            if len(self.frame_times) > self.max_frame_samples:
                self.frame_times.pop(0)

            self.fps_counter += 1
            self.fps_timer += frame_time

            if (
                hasattr(self, "clear_feedback_frames")
                and self.clear_feedback_frames > 0
            ):
                logger.debug(f"Clearing feedback frame{self.clear_feedback_frames}")
                self.feedback_fbo.use()
                self.ctx.disable(BLEND)
                self.ctx.clear(0.0, 0.0, 0.0, 1.0)
                self.ctx.enable(BLEND)
                self.clear_feedback_frames -= 1

            # Update FPS display every second
            if self.fps_timer >= self.fps_update_interval:
                # Optimized FPS calculation using numpy for better performance
                if len(self.frame_times) > 0:
                    # Use numpy mean for vectorized calculation
                    frame_times_array = np.array(self.frame_times, dtype=np.float32)
                    avg_frame_time = np.mean(frame_times_array)
                    self.current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
                else:
                    self.current_fps = 0.0

                self.fps_counter = 0
                self.fps_timer = 0.0

                # Update window title with FPS and automatic transition status
                title = f"KarmaViz - {self.current_fps:.1f} FPS"

                # Add automatic transition status
                if self.transitions_paused:
                    title += " (PAUSED)"
                elif not (self.warp_map_locked and self.waveform_locked):
                    # Show AUTO if either warp maps or waveforms can change automatically
                    title += " (AUTO)"

                pygame.display.set_caption(title)

            # Calculate unified animation speed
            # Apply audio boost multiplier (range 1.0 to 3.0)
            target_audio_boost = (
                abs(self.current_amplitude) * self.audio_speed_boost * 0.5
            )
            self.smoothed_audio_boost = (
                0.75 * self.smoothed_audio_boost + 0.15 * target_audio_boost
            )

            # Additional smoothing for audio boost
            if not hasattr(self, 'smoothed_audio_boost2'):
                self.smoothed_audio_boost2 = self.audio_speed_boost
            self.smoothed_audio_boost2 = 0.75 * getattr(
                self, 'smoothed_audio_boost2', 
                self.audio_speed_boost) + 0.15 * self.smoothed_audio_boost

            # Calculate final animation speed: base speed * audio boost multiplier
            final_animation_speed = abs(self.animation_speed + self.smoothed_audio_boost2)

            # Update time with the unified speed (much smaller increment)
            # Use actual FPS for normalization, fallback to 60fps if unavailable
            frame_time = 1.0 / self.current_fps if self.current_fps > 0 else 0.016
            time_increment = final_animation_speed * frame_time
            self.time += time_increment

            # Pass the final speed to shaders (they'll handle their own time scaling)
            self.animation_speed_uniform = final_animation_speed

            # Update silence detection and logo fade
            self.update_silence_detection(current_time)

            # Update logo overlay transition
            self.update_logo_overlay_transition(current_time)

            # Update current symmetry
            # Only update current_symmetry when NOT in random mode (-1)
            # This ensures random symmetry only changes during beat detection
            if self.symmetry_mode != -1:
                self.current_symmetry = self.symmetry_mode

            # Apply screen pulse - calculate pulsing effect if enabled
            pulse_scale = 1.0  # Start with no scaling
            if self.pulse_enabled and self.pulse_intensity > 0:

                time_since_beat = current_time - self.audio_processor.last_beat_time
                beat_interval = self.beat_interval

                # Complete one full pulse cycle based on beat interval
                cycle_progress = min(1.0, time_since_beat / beat_interval)

                # Use cosine wave for smoother bidirectional movement
                # Cosine starts at 1 and smoothly transitions to -1
                cos_value = np.cos(cycle_progress * 2 * np.pi)

                # Apply exponential decay for smoother falloff
                decay = np.exp(-3.0 * time_since_beat)

                # Combine smooth oscillation with decay
                pulse_factor = (
                    self.pulse_intensity
                    * decay
                    * (cos_value * 0.5 + 0.5)  # Normalize to 0-1 range
                )

                # Add small baseline pulse (independent of animation speed)
                baseline = 0.02 * np.sin(current_time * 2)

                # Combine main pulse with baseline
                pulse_scale = (
                    1.0 + (pulse_factor + baseline) * self.pulse_intensity_multiplier
                )

            # Optimized kaleidoscope sections calculation
            # Pre-compute constants for efficiency
            if not hasattr(self, '_kaleidoscope_constants'):
                self._kaleidoscope_constants = {
                    'transition_speed': 0.3,
                    'min_sections': 8,
                    'max_sections': 24,
                    'section_range': 16,  # 24 - 8
                    'half_range': 8.0,    # section_range / 2
                    'mid_sections': 16    # (8 + 24) / 2
                }

            constants = self._kaleidoscope_constants

            # Optimized oscillation calculation using pre-computed values
            sin_value = np.sin(self.time * constants['transition_speed'])
            # Direct calculation avoiding division: sin ranges from -1 to 1,
            # so (sin + 1) * half_range + min gives us the range we want
            self.kaleidoscope_sections = int(
                constants['mid_sections'] + sin_value * constants['half_range']
            )

            # Update shader uniforms (simplified - removed pattern uniforms)
            self.program["time"].value = self.time  # type: ignore
            self.program["animation_speed"].value = self.animation_speed_uniform  # type: ignore
            self.program["rotation"].value = self.rotation_angle  # type: ignore
            self.program["trail_intensity"].value = self.trail_intensity  # type: ignore
            self.program["glow_intensity"].value = self.glow_intensity  # type: ignore
            if "glow_radius" in self.program:
                self.program["glow_radius"].value = self.glow_radius  # type: ignore
            self.program["symmetry_mode"].value = self.current_symmetry  # type: ignore
            self.program["kaleidoscope_sections"].value = self.kaleidoscope_sections  # type: ignore
            self.program["smoke_intensity"].value = self.smoke_intensity  # type: ignore
            self.program["pulse_scale"].value = pulse_scale  # type: ignore
            self.program["warp_first"].value = self.warp_first_enabled  # type: ignore
            # Update bounce uniforms
            self.program["bounce_enabled"].value: bool = self.bounce_enabled  # type: ignore
            self.program["bounce_height"].value = float(self.bounce_height)  # type: ignore
            # Update warp uniforms (only if they exist in the shader)
            if "warp_intensity" in self.program:
                self.program["warp_intensity"].value = self.warp_intensity  # type: ignore
            if "active_warp_map" in self.program:
                self.program["active_warp_map"].value = self.active_warp_map_index  # type: ignore

            # Set GPU waveform uniforms in main shader
            if self.waveform_texture:
                # Bind waveform texture to texture unit 1 (texture0 is unit 0)
                self.waveform_texture.use(location=1)

                # Bind FFT texture to texture unit 2 for lightning waveform
                if hasattr(self, "fft_texture") and self.fft_texture is not None:
                    self.fft_texture.use(location=2)

                # Get current waveform color from palette
                waveform_color = self.get_current_waveform_color()

                # Set waveform uniforms in main shader program
                if "waveform_data" in self.program:
                    self.program["waveform_data"].value = 1  # Texture unit 1
                if hasattr(self, "fft_texture") and self.fft_texture is not None:
                    if "fft_data" in self.program:
                        self.program["fft_data"].value = 2  # Texture unit 2
                if "waveform_length" in self.program and self.waveform_buffer is not None:
                    self.program["waveform_length"].value = len(self.waveform_buffer)
                if "waveform_scale" in self.program:
                    self.program["waveform_scale"].value = self.waveform_scale
                if "waveform_color" in self.program:
                    self.program["waveform_color"].value = waveform_color  # Current palette color
                if "waveform_alpha" in self.program:
                    self.program["waveform_alpha"].value = (
                        self.waveform_fade_alpha
                    )  # Fade-in effect
                if "waveform_enabled" in self.program:
                    self.program["waveform_enabled"].value = True  # Enable GPU waveform in main shader
            else:
                # Disable GPU waveform rendering in main shader
                if "waveform_enabled" in self.program:
                    self.program["waveform_enabled"].value = False
                if "waveform_alpha" in self.program:
                    self.program["waveform_alpha"].value = (
                        self.waveform_fade_alpha
                    )  # Still set alpha for consistency

            # Use fixed vertices that don't change every frame - much more efficient
            # The pulse effect will be handled in the shader using the pulse_scale uniform
            if not hasattr(self, "fixed_vertices_set"):
                # Only set up the vertices once for better performance
                vertices = np.array(
                    [
                        # positions    # uv coordinates
                        -1.0,
                        -1.0,
                        0.0,
                        0.0,  # Bottom left
                        1.0,
                        -1.0,
                        1.0,
                        0.0,  # Bottom right
                        1.0,
                        1.0,
                        1.0,
                        1.0,  # Top right
                        -1.0,
                        1.0,
                        0.0,
                        1.0,  # Top left
                    ],
                    dtype=DATA_FORMAT,
                )
                self.vbo.write(vertices)
                self.fixed_vertices_set = True
            # No need to update vertices every frame - they're now fixed

            # Main render pass (now with integrated warp maps and GPU waveforms)
            self.main_fbo.use()
            self.ctx.disable(BLEND)
            self.ctx.clear(0.0, 0.0, 0.0, 1.0)
            self.ctx.enable(BLEND)
            self.ctx.blend_func = ONE, ONE_MINUS_SRC_ALPHA

            # GPU waveform rendering is integrated into main shader
            # The main shader uses texture0 (feedback) as input and renders to main_fbo
            self.feedback_texture.use(
                0
            )  # Bind feedback texture to texture0 for main shader
            self.vao.render(TRIANGLE_FAN)

            # Screen pass - render clean visualization (no overlays)
            self.ctx.screen.use()
            self.ctx.disable(BLEND)
            self.ctx.clear(0.0, 0.0, 0.0, 1.0)
            self.ctx.enable(BLEND)

            # Add main framebuffer content (contains GPU waveforms)
            self.main_fbo.color_attachments[0].use(0)
            # Use alpha blending to respect trail_alpha for proper dimming
            self.ctx.blend_func = SRC_ALPHA, ONE_MINUS_SRC_ALPHA
            self.vao.render(TRIANGLE_FAN)

            # CRITICAL: Copy clean screen to feedback buffer NOW, before any overlays are added
            # This preserves the feedback loop for waveforms while excluding overlays
            self.ctx.copy_framebuffer(self.feedback_fbo, self.ctx.screen)

            # Now render overlays directly to screen (they won't be in next frame's feedback)
            # Render spectrogram overlay directly to screen
            if self.show_spectrogram_overlay:
                self.ctx.screen.use()
                self.ctx.enable(BLEND)
                self.ctx.blend_func = SRC_ALPHA, ONE_MINUS_SRC_ALPHA
                self.render_spectrogram_overlay()

            # Render logo overlay for smooth transition from splash screen
            if self.logo_overlay_alpha > 0.0:
                self.ctx.screen.use()
                self.ctx.enable(BLEND)
                self.ctx.blend_func = SRC_ALPHA, ONE_MINUS_SRC_ALPHA
                self.render_logo_overlay()

            # Render regular logo (if enabled and visible)
            elif self.logo_visible and self.logo_fade_alpha > 0.0:
                self.ctx.screen.use()
                self.ctx.enable(BLEND)
                self.ctx.blend_func = SRC_ALPHA, ONE_MINUS_SRC_ALPHA
                self.render_logo()

        except Exception:
            raise
    @benchmark("update_viewport")
    def update_viewport(self, new_size):
        global WIDTH, HEIGHT

        # Update the global WIDTH and HEIGHT to match the actual window/screen size
        # This ensures all components use consistent dimensions
        WIDTH, HEIGHT = new_size

        # Update resolution uniform
        self.resolution = [float(WIDTH), float(HEIGHT)]
        if hasattr(self, "program"):
            self.program["resolution"] = self.resolution  # type: ignore

        target_aspect = WIDTH / HEIGHT
        screen_aspect = new_size[0] / new_size[1]

        if screen_aspect > target_aspect:
            new_height = new_size[1]
            new_width = int(new_height * target_aspect)
            offset_x = (new_size[0] - new_width) // 2
            offset_y = 0
        else:
            new_width = new_size[0]
            new_height = int(new_width / target_aspect)
            offset_x = 0
            offset_y = (new_size[1] - new_height) // 2

        self.viewport = (offset_x, offset_y, new_width, new_height)
        self.ctx.viewport = self.viewport
        self.window_size = new_size

        # Clean up old textures
        if (
            hasattr(self, "spectrogram_texture")
            and self.spectrogram_texture is not None
        ):
            self.spectrogram_texture.release()
            self.spectrogram_texture = None
        if (
            hasattr(self, "spectrogram_palette_texture")
            and self.spectrogram_palette_texture is not None
        ):
            self.spectrogram_palette_texture.release()
            self.spectrogram_palette_texture = None
        if hasattr(self, "spectrogram_fbo") and self.spectrogram_fbo is not None:
            self.spectrogram_fbo.release()
            self.spectrogram_fbo = None

        # Release old resources first to prevent memory leaks and white blowout
        if hasattr(self, 'main_fbo') and self.main_fbo is not None:
            self.main_fbo.release()
            self.main_fbo = None
        if hasattr(self, 'feedback_fbo') and self.feedback_fbo is not None:
            self.feedback_fbo.release()
            self.feedback_fbo = None
        if hasattr(self, 'textures'):
            for tex in self.textures:
                if tex is not None:
                    tex.release()
        if hasattr(self, 'feedback_texture') and self.feedback_texture is not None:
            self.feedback_texture.release()
            self.feedback_texture = None
        if hasattr(self, 'overlay_texture') and self.overlay_texture is not None:
            self.overlay_texture.release()
            self.overlay_texture = None
        if hasattr(self, 'overlay_fbo') and self.overlay_fbo is not None:
            self.overlay_fbo.release()
            self.overlay_fbo = None
        logger.debug("Released old GPU resources")

        # Recreate textures with new dimensions to match the pixel buffer
        # Use anti_aliasing_samples for multisampling if available
        try:
            self.textures = [
                self.ctx.texture(
                    (WIDTH, HEIGHT),
                    3,
                    samples=self.anti_aliasing_samples,
                    dtype=TEXTURE_DTYPE_STR,
                ),
                self.ctx.texture(
                    (WIDTH, HEIGHT),
                    3,
                    samples=self.anti_aliasing_samples,
                    dtype=TEXTURE_DTYPE_STR,
                ),
            ]
            logger.debug("Recreated main textures with {self.anti_aliasing_samples}x multisampling ({WIDTH}x{HEIGHT})")

        except Exception as e:
            # Fallback to non-multisampled textures
            self.textures = [
                self.ctx.texture((WIDTH, HEIGHT), 3, dtype=TEXTURE_DTYPE_STR),
                self.ctx.texture((WIDTH, HEIGHT), 3, dtype=TEXTURE_DTYPE_STR),
            ]
            logger.error(f"Multisampling not supported for textures, using standard textures: {e}")

        # Feedback texture doesn't need multisampling
        self.feedback_texture = self.ctx.texture(
            (WIDTH, HEIGHT), 3, dtype=TEXTURE_DTYPE_STR
        )
        logger.debug(f"Created feedback texture ({WIDTH}x{HEIGHT})")

        # Recreate overlay texture for logo and other overlays
        self.overlay_texture = self.ctx.texture(
            (WIDTH, HEIGHT), 4, dtype=TEXTURE_DTYPE_STR  # RGBA for transparency
        )
        logger.debug(f"Created overlay texture ({WIDTH}x{HEIGHT})")

        # Recreate framebuffers with new textures
        self.feedback_fbo = self.ctx.framebuffer(
            color_attachments=[self.feedback_texture]
        )
        self.overlay_fbo = self.ctx.framebuffer(
            color_attachments=[self.overlay_texture]
        )
        self.main_fbo = self.ctx.framebuffer(
            color_attachments=[self.textures[0]]
        )
        logger.debug(f"Created framebuffers for {WIDTH}x{HEIGHT}")

        # Immediately clear all framebuffers to prevent white blowout
        # Disable blending for clearing
        self.ctx.disable(BLEND)

        # Clear main framebuffer thoroughly
        self.main_fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)  # Clear to transparent black

        # Clear feedback framebuffer thoroughly - this is critical
        self.feedback_fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)  # Clear to transparent black

        # Clear overlay framebuffer thoroughly
        self.overlay_fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)  # Clear to transparent black

        # Clear screen framebuffer
        self.ctx.screen.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)  # Clear to opaque black

        # Restore proper blending state
        self.ctx.enable(BLEND)
        self.ctx.blend_func = SRC_ALPHA, ONE_MINUS_SRC_ALPHA
        logger.debug("Cleared all framebuffers and restored blending state")

        # Set resize recovery flags to prevent white blowout during first few frames
        self.resize_trail_reduction = 0.1  # Reduce trails by 90% during resize
        self.resize_frame_counter = 0  # Counter to track frames since resize
        self.disable_feedback_frames = 20  # Disable feedback for first 20 frames after resize
        self.disable_additive_blending_frames = 15  # Disable additive blending for first 15 frames
        logger.debug("Set resize recovery flags")

        # Recreate GPU waveform textures
        if hasattr(self, "waveform_texture") and self.waveform_texture is not None:
            self.waveform_texture.release()
            self.waveform_texture = None
        if hasattr(self, "fft_texture") and self.fft_texture is not None:
            self.fft_texture.release()
            self.fft_texture = None

        # Reset beat tracking state to ensure pulse effect works after resolution change
        self.last_beat_time = (
            time()
        )  # Use time() for consistency with the beat detection
        self.beat_interval = 0.2  # Default beat interval
        self.beat_detected = False
        self.pulse_intensity = 0.0

    def decrease_pulse_intensity(self):
        """Decrease the pulse intensity by 0.01, with a minimum of 0.01"""
        self.pulse_intensity_multiplier = max(
            0.1, self.pulse_intensity_multiplier - 0.1
        )
        logger.debug(f"pulse_intensity = {self.pulse_intensity_multiplier}")

    def increase_pulse_intensity(self):
        """Increase the pulse intensity by 0.01, with a maximum of 2.0"""
        self.pulse_intensity_multiplier = min(
            2.0, self.pulse_intensity_multiplier + 0.1
        )
        logger.debug(f"pulse_intensity = {self.pulse_intensity_multiplier}")

    def cycle_waveform_style(self):
        """Cycle through GPU waveform shaders"""
        logger.debug("Cycling through waveform shaders...")
        self.cycle_gpu_waveform()

    def cycle_gpu_waveform(self):
        """Select a random GPU waveform shader using WaveformManager."""
        available_waveforms = self.shader_manager.list_waveforms()
        if not available_waveforms:
            logger.error("No waveforms found.")
            return

        # Filter out the current waveform to ensure we get a different one
        other_waveforms = [wf for wf in available_waveforms if wf != self.current_waveform_name]

        if other_waveforms:
            next_waveform = choice(other_waveforms)
        else:
            # If only one waveform available, keep current
            logger.debug("Only one waveform available, keeping current selection")
            return

        logger.debug(f"[AsyncShader] Requesting compilation for waveform: {next_waveform}")

        # Update waveform name immediately (optimistic update)
        self.current_waveform_name = next_waveform

        # Submit async compilation request with high priority
        warp_maps = [self.active_warp_map_name] if self.active_warp_map_name else []
        request_id = self.shader_manager.compile_shader_async(
            warp_maps=warp_maps,
            current_waveform_name=self.current_waveform_name,
            purpose=f"main_shader_waveform_{next_waveform}",
            priority=8,  # High priority for user-initiated waveform changes
        )

        logger.debug(f"[AsyncShader] Submitted async compilation request {request_id} for waveform: {next_waveform}")
        logger.debug(f"Randomly selected waveform: {next_waveform} - compiling in background...")

        # The shader will be updated automatically when compilation completes
        # via process_shader_compilation_results() in the render loop

    def cycle_rotation_mode(self):
        """Cycle through rotation modes: None -> Clockwise -> Counter-clockwise -> Beat Driven"""
        modes = [0, 1, 2, 3]  # None, Clockwise, Counter-clockwise, Beat Driven
        current_index = modes.index(self.rotation_mode) if self.rotation_mode in modes else 0
        self.rotation_mode = modes[(current_index + 1) % len(modes)]

    def set_rotation_speed(self, speed: float):
        """Set rotation speed multiplier"""
        self.rotation_speed = max(0.0, min(5.0, speed))
        logger.debug(f"Rotation speed: {self.rotation_speed:.2f}")

    def set_rotation_amplitude(self, amplitude: float):
        """Set rotation amplitude/intensity multiplier"""
        self.rotation_amplitude = max(0.0, min(3.0, amplitude))
        logger.debug(f"Rotation amplitude: {self.rotation_amplitude:.2f}")

    def set_rotation_mode(self, mode: int):
        """Set rotation mode directly"""
        if mode in [0, 1, 2, 3]:
            self.rotation_mode = mode
            mode_names = {0: "None", 1: "Clockwise", 2: "Counter-clockwise", 3: "Beat Driven"}
            logger.debug(f"Rotation mode: {mode_names.get(mode, 'Unknown')}")

    def increase_rotation_speed(self):
        """Increase rotation speed by 0.1, with a maximum of 5.0"""
        self.rotation_speed = min(5.0, self.rotation_speed + 0.1)
        logger.debug(f"Rotation speed increased to: {self.rotation_speed:.2f}")

    def decrease_rotation_speed(self):
        """Decrease rotation speed by 0.1, with a minimum of 0.0"""
        self.rotation_speed = max(0.0, self.rotation_speed - 0.1)
        logger.debug(f"Rotation speed decreased to: {self.rotation_speed:.2f}")

    def increase_rotation_amplitude(self):
        """Increase rotation amplitude by 0.1, with a maximum of 3.0"""
        self.rotation_amplitude = min(3.0, self.rotation_amplitude + 0.1)
        logger.debug(f"Rotation amplitude increased to: {self.rotation_amplitude:.2f}")

    def decrease_rotation_amplitude(self):
        """Decrease rotation amplitude by 0.1, with a minimum of 0.0"""
        self.rotation_amplitude = max(0.0, self.rotation_amplitude - 0.1)
        logger.debug(f"Rotation amplitude decreased to: {self.rotation_amplitude:.2f}")

    def toggle_pulse(self):
        """Toggle screen shake effect on/off"""
        self.pulse_enabled = not self.pulse_enabled

    def decrease_trail_intensity(self):
        """Decrease the trail intensity by 0.05, with a minimum of 0.0"""
        self.trail_intensity = max(0.0, self.trail_intensity - 0.5)

    def increase_trail_intensity(self):
        """Increase the trail intensity by 0.05, with a maximum of 1.0"""
        self.trail_intensity = min(5.0, self.trail_intensity + 0.5)

    def decrease_glow_intensity(self):
        """Decrease the glow intensity by 0.01, with a minimum of 0.7"""
        self.glow_intensity = max(0.7, self.glow_intensity - 0.01)

    def increase_glow_intensity(self):
        """Increase the glow intensity by 0.1, with a maximum of 2.0"""
        self.glow_intensity = min(1.0, self.glow_intensity + 0.01)

    def decrease_glow_radius(self):
        """Decrease the glow radius by 0.005, with a minimum of 0.01"""
        self.glow_radius = max(0.01, self.glow_radius - 0.005)

    def increase_glow_radius(self):
        """Increase the glow radius by 0.005, with a maximum of 0.2"""
        self.glow_radius = min(0.2, self.glow_radius + 0.005)

    def toggle_gpu_waveform_random(self):
        """Toggle random GPU waveform selection on/off"""
        self.gpu_waveform_random = not self.gpu_waveform_random
        status = "enabled" if self.gpu_waveform_random else "disabled"
        logger.debug(f"Random GPU waveforms {status}")

        # If we just enabled random mode, immediately switch to a random waveform
        if self.gpu_waveform_random:
            self.cycle_gpu_waveform()

    def decrease_smoke_intensity(self):
        """Decrease the smoke intensity by 0.1, with a minimum of 0.0"""
        self.smoke_intensity = max(0.0, self.smoke_intensity - 0.1)
        logger.debug(f"Smoke intensity decreased to: {self.smoke_intensity:.1f}")

    def increase_smoke_intensity(self):
        """Increase the smoke intensity by 0.1, with a maximum of 1.0"""
        self.smoke_intensity = min(1.0, self.smoke_intensity + 0.1)
        logger.debug(f"Smoke intensity increased to: {self.smoke_intensity:.1f}")

    def decrease_warp_intensity(self):
        """Decrease the warp intensity by 0.05, with a minimum of 0.0"""
        self.warp_intensity = max(0.0, self.warp_intensity - 0.1)
        logger.debug(f"Warp intensity decreased to: {self.warp_intensity:.2f}")

    def increase_warp_intensity(self):
        """Increase the warp intensity by 0.05, with a maximum of 2.0"""
        self.warp_intensity = min(5.0, self.warp_intensity + 0.1)
        logger.debug(f"Warp intensity increased to: {self.warp_intensity:.2f}")

    def decrease_waveform_scale(self):
        """Decrease the waveform scale by 0.1, with a minimum of 0.1"""
        self.waveform_scale = max(0.1, self.waveform_scale - 0.1)

    def increase_waveform_scale(self):
        """Increase the waveform scale by 0.1, with a maximum of 5.0"""
        self.waveform_scale = min(5.0, self.waveform_scale + 0.1)

    def cycle_symmetry_mode(self):
        """Cycle through symmetry modes: None -> Mirror -> Quad -> Kaleidoscope -> Grid -> Spiral -> Diamond -> Fractal -> Random"""
        if self.symmetry_mode == -1:  # If currently on Random
            self.symmetry_mode = 0  # Go back to None
        else:
            # Cycle through 0-7, then wrap to Random (-1)
            self.symmetry_mode = (self.symmetry_mode + 1) % 11
            if self.symmetry_mode == 0:  # If we wrapped around to 0
                self.symmetry_mode = -1  # Set to Random instead

        # If we're in Random mode, immediately select a random mode
        if self.symmetry_mode == -1:
            # Randomly choose from 0 (None) to 7 (Fractal)
            self.current_symmetry = randint(0, 11)
        else:
            self.current_symmetry = self.symmetry_mode

    def decrease_animation_speed(self):
        """Decrease the base animation speed by 0.2, with a minimum of 0.0"""
        self.animation_speed = max(0.0, self.animation_speed - 0.2)
        logger.debug(f"Animation speed: {self.animation_speed:.1f}")

    def increase_animation_speed(self):
        """Increase the base animation speed by 0.2, with a maximum of 5.0"""
        self.animation_speed = min(2.0, self.animation_speed + 0.2)
        logger.debug(f"Animation speed: {self.animation_speed:.1f}")

    def decrease_audio_speed_boost(self):
        """Decrease the audio speed boost by 0.1, with a minimum of 0.0"""
        self.audio_speed_boost = max(0.0, self.audio_speed_boost - 0.05)
        logger.debug(f"Audio speed boost: {self.audio_speed_boost:.1f}")

    def increase_audio_speed_boost(self):
        """Increase the audio speed boost by 0.1, with a maximum of 1.0"""
        self.audio_speed_boost = min(1.0, self.audio_speed_boost + 0.05)
        logger.debug(f"Audio speed boost: {self.audio_speed_boost:.1f}")

    def get_current_waveform_color(self):
        """Get the current waveform color from the active palette"""
        try:
            if self.current_palette and len(self.current_palette) > 0:
                # Always use the current palette color as base, with interpolation as enhancement
                current_color_rgb = self.current_palette[self.color_index % len(self.current_palette)]

                # Use the smoothly interpolated color if available, otherwise use palette color directly
                if (
                    hasattr(self, "current_interpolated_color")
                    and self.current_interpolated_color is not None
                ):
                    # Use the smoothly interpolated color (already in 0-255 RGB)
                    base_color = (
                        self.current_interpolated_color[0] / 255.0,
                        self.current_interpolated_color[1] / 255.0,
                        self.current_interpolated_color[2] / 255.0,
                    )
                else:
                    # Fallback to current palette color (based on color_index)
                    base_color = (
                        current_color_rgb[0] / 255.0,
                        current_color_rgb[1] / 255.0,
                        current_color_rgb[2] / 255.0,
                    )

                # Add some brightness variation based on audio amplitude for more dynamic waveforms
                brightness_boost = 1.0 + (self.smoothed_amplitude * 0.8)  # Up to 80% brighter for more dramatic effect

                # Apply waveform brightness multiplier for silence fade
                final_brightness = (
                    brightness_boost * self.waveform_brightness_multiplier
                )

                enhanced_color = (
                    min(1.0, base_color[0] * final_brightness),
                    min(1.0, base_color[1] * final_brightness),
                    min(1.0, base_color[2] * final_brightness),
                )

                return enhanced_color
            else:
                # Fallback to a nice cyan color if no palette is available
                fallback_color = (0.3, 0.8, 1.0)
                return (
                    fallback_color[0] * self.waveform_brightness_multiplier,
                    fallback_color[1] * self.waveform_brightness_multiplier,
                    fallback_color[2] * self.waveform_brightness_multiplier,
                )
        except Exception as e:
            logger.error(f"Error getting waveform color: {e}")
            # Fallback color with brightness multiplier
            fallback_color = (0.3, 0.8, 1.0)
            return (
                fallback_color[0] * self.waveform_brightness_multiplier,
                fallback_color[1] * self.waveform_brightness_multiplier,
                fallback_color[2] * self.waveform_brightness_multiplier,
            )

    def decrease_palette_speed(self):
        """Decrease the palette rotation speed by 0.1, with a minimum of 0.1"""
        self.palette_rotation_speed = max(
            0.1, self.palette_rotation_speed - 0.1
        )

    def increase_palette_speed(self):
        """Increase the palette rotation speed by 0.1, with a maximum of 15.0"""
        self.palette_rotation_speed = min(
            15.0, self.palette_rotation_speed + 0.1
        )

    def decrease_color_cycle_speed(self):
        """Decrease the color cycle speed multiplier by 0.1, minimum 0.1"""
        self.color_cycle_speed_multiplier = max(
            0.1, self.color_cycle_speed_multiplier - 0.1
        )
        logger.debug(f"Color Cycle Speed Multiplier: {self.color_cycle_speed_multiplier:.1f}")

    def increase_color_cycle_speed(self):
        """Increase the color cycle speed multiplier by 0.1, maximum 15.0"""
        self.color_cycle_speed_multiplier = min(
            15.0, self.color_cycle_speed_multiplier + 0.1
        )
        logger.debug(f"Color Cycle Speed Multiplier: {self.color_cycle_speed_multiplier:.1f}")

    def force_palette_change(self):
        """Force a smooth palette change based on current mood"""
        new_palette = self.get_mood_palette(self.current_mood)
        self._start_palette_transition(new_palette)

        # Apply bounce effect if enabled
        if self.bounce_enabled and self.bounce_height > 0.001:
            # Calculate bounce offset based on current bounce height
            bounce_offset = int(self.bounce_height * HEIGHT * 0.3)  # 30% of height max
            y_coords += bounce_offset

            # Decay the bounce height
            time_since_bounce = time() - self.last_bounce_time
            self.bounce_height *= np.exp(-self.bounce_decay * time_since_bounce)

    def toggle_spectrogram_overlay(self):
        """Toggle the spectrogram overlay on/off"""
        self.show_spectrogram_overlay = not self.show_spectrogram_overlay

    @benchmark("update_spectrogram_data")
    def update_spectrogram_data(self, audio_data):
        """Update spectrogram data from audio buffer with logarithmic frequency mapping and smoothing"""
        if not self.show_spectrogram_overlay:
            return

        # Ensure audio data is in correct format for processing
        audio_data = np.asarray(audio_data, dtype=DATA_FORMAT)

        # Perform FFT on the audio buffer (already mono from draw_waveform processing)
        fft_data = np.abs(np.fft.rfft(audio_data))

        # Pre-compute constants to avoid repeated calculations
        if not hasattr(self, '_spectrogram_constants'):
            self._spectrogram_constants = {
                'num_bins': 128,
                'min_freq': 20,
                'max_freq': 5000,
                'rate_inv': 1.0 / RATE,
                'log_min_freq': np.log10(20),
                'log_max_freq': np.log10(5000),
                'epsilon': 1e-6
            }

        constants = self._spectrogram_constants

        # Get frequencies up to max_freq (e.g., 10kHz) - cache if possible
        if not hasattr(self, '_cached_freqs') or len(self._cached_freqs) != len(fft_data):
            self._cached_freqs = np.fft.rfftfreq(len(audio_data), d=constants['rate_inv'])

            # Pre-compute frequency indices for efficiency
            self._min_freq_idx = np.searchsorted(self._cached_freqs, constants['min_freq'], side="left")
            self._max_freq_idx = np.searchsorted(self._cached_freqs, constants['max_freq'], side="right")

            # Pre-compute target frequencies and log values
            self._target_log_freqs = np.logspace(
                constants['log_min_freq'], constants['log_max_freq'], num=constants['num_bins']
            )
            self._log_target_freqs = np.log10(self._target_log_freqs)
            self._freq_weights = np.sqrt(self._target_log_freqs / constants['min_freq'])

        # Ensure we have valid indices and data
        if self._max_freq_idx <= self._min_freq_idx or self._max_freq_idx > len(self._cached_freqs):
            # Vectorized clearing
            self.spectrogram_data.fill(0.0)
            self.spectrogram_smooth.fill(0.0)
            self.spectrogram_peak.fill(0.0)
            if self.spectrogram_texture is not None:
                self.spectrogram_texture.write(self.spectrogram_data.astype("f4").tobytes())
            return

        # Extract valid frequency range using pre-computed indices
        valid_freqs = self._cached_freqs[self._min_freq_idx:self._max_freq_idx]
        valid_fft_data = fft_data[self._min_freq_idx:self._max_freq_idx]

        # Optimized logarithmic interpolation
        log_valid_freqs = np.log10(valid_freqs + constants['epsilon'])

        # Vectorized log transformation and interpolation
        log_fft_data = np.log10(valid_fft_data + 1)
        interp_log_fft = np.interp(self._log_target_freqs, log_valid_freqs, log_fft_data)

        # Apply pre-computed frequency weighting
        new_data = interp_log_fft * self._freq_weights

        # Vectorized clamping and normalization
        new_data = np.maximum(new_data, 0.0)
        max_val = np.max(new_data)
        if max_val > 0:
            new_data *= (1.0 / max_val)  # Faster than division

        # Vectorized smoothing operations
        smoothing_factor = self.spectrogram_smoothing
        inv_smoothing = 1.0 - smoothing_factor

        # In-place operations for better performance
        self.spectrogram_smooth *= smoothing_factor
        self.spectrogram_smooth += inv_smoothing * new_data

        # Vectorized peak update with falloff
        falloff_peaks = self.spectrogram_peak - self.spectrogram_falloff
        self.spectrogram_peak = np.maximum(self.spectrogram_smooth, falloff_peaks)

        # Direct assignment instead of copy for better performance
        self.spectrogram_data[:] = self.spectrogram_smooth

        # Update texture if it exists
        if self.spectrogram_texture is not None:
            self.spectrogram_texture.write(self.spectrogram_data.astype("f4").tobytes())

    @benchmark("render_spectrogram_overlay")
    def render_spectrogram_overlay(self):
        """Render the spectrogram overlay using shader-based approach with palette colors"""
        if not self.show_spectrogram_overlay:
            return

        # Create or recreate frequency texture if needed
        if self.spectrogram_texture is None:
            self.spectrogram_texture = self.ctx.texture(
                (128, 1), 1, dtype=TEXTURE_DTYPE_STR
            )

        # Update frequency data texture
        self.spectrogram_texture.write(self.spectrogram_data.astype("f4").tobytes())

        # Create or update palette texture
        if self.current_palette and len(self.current_palette) > 0:
            palette_size = len(self.current_palette)

            # Vectorized palette conversion
            palette_array = np.array(self.current_palette, dtype=DATA_FORMAT) / 255.0

            # Create or recreate palette texture if size changed
            if (self.spectrogram_palette_texture is None or
                self.spectrogram_palette_texture.size[0] != palette_size):
                if self.spectrogram_palette_texture is not None:
                    self.spectrogram_palette_texture.release()
                self.spectrogram_palette_texture = self.ctx.texture(
                    (palette_size, 1), 3, dtype=TEXTURE_DTYPE_STR
                )

            # Update palette texture
            self.spectrogram_palette_texture.write(palette_array.tobytes())
        else:
            palette_size = 1
            # Fallback to white if no palette
            if self.spectrogram_palette_texture is None:
                self.spectrogram_palette_texture = self.ctx.texture(
                    (1, 1), 3, dtype=DATA_FORMAT
                )
            fallback_color = np.array([[1.0, 1.0, 1.0]], dtype=DATA_FORMAT)
            self.spectrogram_palette_texture.write(fallback_color.tobytes())

        # Set shader uniforms
        self.spectrogram_program["frequency_data"] = 0
        self.spectrogram_program["palette_data"] = 1
        self.spectrogram_program["palette_size"] = palette_size
        self.spectrogram_program["opacity"] = 1.0
        self.spectrogram_program["time"] = self.time
        self.spectrogram_program["color_interpolation_speed"] = self.spectrogram_color_interpolation_speed

        # Bind textures
        self.spectrogram_texture.use(0)
        self.spectrogram_palette_texture.use(1)

        # Use standard alpha blending
        self.ctx.blend_func = SRC_ALPHA, ONE_MINUS_SRC_ALPHA

        # Render the overlay
        self.spectrogram_vao.render(TRIANGLE_FAN)

    def toggle_mouse_interaction(self):
        """Toggle mouse interaction on/off"""
        self.mouse_interaction_enabled = not self.mouse_interaction_enabled
        self.program["mouse_enabled"] = self.mouse_interaction_enabled  # type: ignore
        logger.debug(f"Mouse interaction: {'enabled' if self.mouse_interaction_enabled else 'disabled'}")

    def increase_mouse_intensity(self):
        """Increase the mouse interaction intensity by 0.1, with a maximum of 3.0"""
        self.mouse_intensity = min(3.0, self.mouse_intensity + 0.1)
        if "mouse_intensity" in self.program:
            self.program["mouse_intensity"] = self.mouse_intensity  # type: ignore
        logger.debug(f"Mouse intensity increased to: {self.mouse_intensity:.1f}")

    def decrease_mouse_intensity(self):
        """Decrease the mouse interaction intensity by 0.1, with a minimum of 0.1"""
        self.mouse_intensity = max(0.1, self.mouse_intensity - 0.1)
        if "mouse_intensity" in self.program:
            self.program["mouse_intensity"] = self.mouse_intensity  # type: ignore
        logger.debug(f"Mouse intensity decreased to: {self.mouse_intensity:.1f}")

    def add_shockwave(self, x: float, y: float, intensity: float = 1.0):
        """Add a shockwave effect at the specified position"""
        current_time = time()
        
        # Remove oldest shockwave if we're at the limit
        if len(self.shockwaves) >= self.max_effects:
            self.shockwaves.pop(0)
        
        # Add new shockwave [x, y, start_time, intensity]
        self.shockwaves.append([x, y, current_time, intensity])
        logger.debug(f"Added shockwave at ({x:.0f}, {y:.0f}) with intensity {intensity:.1f}")

    def add_ripple(self, x: float, y: float, intensity: float = 1.0):
        """Add a ripple effect at the specified position"""
        current_time = time()
        
        # Remove oldest ripple if we're at the limit
        if len(self.ripples) >= self.max_effects:
            self.ripples.pop(0)
        
        # Add new ripple [x, y, start_time, intensity]
        self.ripples.append([x, y, current_time, intensity])
        logger.debug(f"Added ripple at ({x:.0f}, {y:.0f}) with intensity {intensity:.1f}")

    def update_click_effects(self):
        """Update and clean up expired click effects"""
        current_time = time()
        shockwave_duration = 2.0  # Shockwaves last 2 seconds
        ripple_duration = 3.0     # Ripples last 3 seconds
        
        # Remove expired shockwaves
        self.shockwaves = [sw for sw in self.shockwaves if (current_time - sw[2]) < shockwave_duration]
        
        # Remove expired ripples
        self.ripples = [rp for rp in self.ripples if (current_time - rp[2]) < ripple_duration]

    def _update_click_effect_uniforms(self):
        """Update shader uniforms with current click effect data"""
        current_time = time()
        
        # Prepare shockwave data for shader (up to max_effects)
        shockwave_data = []
        for i in range(self.max_effects):
            if i < len(self.shockwaves):
                sw = self.shockwaves[i]
                x, y, start_time, intensity = sw
                age = current_time - start_time
                # Normalize coordinates to [0,1] range
                norm_x = x / self.resolution[0]
                norm_y = y / self.resolution[1]
                shockwave_data.extend([norm_x, norm_y, age, intensity])
            else:
                # Inactive slot - use negative age to indicate inactive
                shockwave_data.extend([0.0, 0.0, -1.0, 0.0])
        
        # Prepare ripple data for shader (up to max_effects)
        ripple_data = []
        for i in range(self.max_effects):
            if i < len(self.ripples):
                rp = self.ripples[i]
                x, y, start_time, intensity = rp
                age = current_time - start_time
                # Normalize coordinates to [0,1] range
                norm_x = x / self.resolution[0]
                norm_y = y / self.resolution[1]
                ripple_data.extend([norm_x, norm_y, age, intensity])
            else:
                # Inactive slot - use negative age to indicate inactive
                ripple_data.extend([0.0, 0.0, -1.0, 0.0])
        
        # Update shader uniforms
        if "shockwave_data" in self.program:
            self.program["shockwave_data"] = shockwave_data  # type: ignore
        if "ripple_data" in self.program:
            self.program["ripple_data"] = ripple_data  # type: ignore

    def toggle_warp_first(self):
        """Toggle the order of warp and symmetry application."""
        self.warp_first_enabled = not self.warp_first_enabled
        logger.debug(f"Warp First Mode: {'Enabled' if self.warp_first_enabled else 'Disabled'}")

    def toggle_bounce(self):
        """Toggle the bounce effect on/off"""
        self.bounce_enabled = not self.bounce_enabled
        logger.debug(f"Bounce effect: {'enabled' if self.bounce_enabled else 'disabled'}")
        if not self.bounce_enabled:
            self.bounce_height = 0.0
            self.bounce_velocity = 0.0  # Reset velocity too
        else:
            # Initialize with some small motion
            self.bounce_height = 0.0
            self.bounce_velocity = 0.1
            self.last_bounce_time = time()

    def increase_bounce_intensity(self):
        """Increase the bounce intensity by 0.1, with a maximum of 1.0 asynchronously"""
        self.bounce_intensity_multiplier = min(
            1.0, self.bounce_intensity_multiplier + 0.05
        )
        logger.debug(f"Bounce intensity: {self.bounce_intensity_multiplier:.1f}")

    def decrease_bounce_intensity(self):
        """Decrease the bounce intensity by 0.1, with a minimum of 0.1"""
        self.bounce_intensity_multiplier = max(
            0.1, self.bounce_intensity_multiplier - 0.05
        )
        logger.debug(f"Bounce intensity: {self.bounce_intensity_multiplier:.1f}")

    def select_random_warp_map(self, warp_map_name: str):
        """Select a specific warp map and compile the shader asynchronously"""
        if warp_map_name and warp_map_name != "None":
            logger.debug(f"[AsyncShader] Requesting compilation for warp map: {warp_map_name}")

            # Update state immediately (optimistic update)
            self.active_warp_map_name = warp_map_name
            self.active_warp_map_index = 0

            # If symmetry mode is set to random, change symmetry when warp map changes
            if self.symmetry_mode == -1:
                self.current_symmetry = randint(0, 10)
                logger.debug(f"Symmetry changed to mode {self.current_symmetry} (random mode active)")

            # Submit async compilation request with high priority
            self.shader_manager.compile_shader_async(
                warp_maps=[warp_map_name],
                current_waveform_name=self.current_waveform_name,
                purpose=f"main_shader_warp_{warp_map_name}",
                priority=10,  # High priority for user-initiated changes
            )

            logger.debug("[AsyncShader] Submitted async compilation request {request_id} for warp map: {warp_map_name}")

            # The shader will be updated automatically when compilation completes
            # via process_shader_compilation_results() in the render loop
        else:
            # Clear warp map
            self.clear_warp_map()

    def cycle_to_random_warp_map(self):
        """Cycle to a random warp map"""
        # Get available warp map keys (filenames)
        warp_map_keys = list(self.warp_map_manager.warp_maps.keys())

        if warp_map_keys:
            # Filter out the current warp map to ensure we get a different one
            other_keys = [key for key in warp_map_keys if key != self.active_warp_map_name]

            if other_keys:
                random_key = choice(other_keys)
                random_warp_map = self.warp_map_manager.get_warp_map(random_key)
                logger.debug(f"Selected warp map: {random_warp_map.name}")
                self.select_random_warp_map(random_key)  # Pass filename key, not display name
            else:
                logger.debug("Only one warp map available, keeping current selection")

    def clear_warp_map(self):
        """Clear the current warp map asynchronously"""
        logger.debug("[AsyncShader] Clearing warp map - compiling shader without warp effects")

        # Update state immediately (optimistic update)
        self.active_warp_map_name = None
        self.active_warp_map_index = -1

        # If symmetry mode is set to random, change symmetry when warp map changes
        if self.symmetry_mode == -1:
            self.current_symmetry = randint(0, 10)
            logger.debug(f"Symmetry changed to mode {self.current_symmetry} (random mode active)")

        # Submit async compilation request for shader without warp maps
        self.shader_manager.compile_shader_async(
            warp_maps=[],  # Empty list = no warp maps
            current_waveform_name=self.current_waveform_name,
            purpose="main_shader_clear_warp",
            priority=10,  # High priority for user-initiated changes
        )

        logger.debug("[AsyncShader] Submitted async compilation request {request_id} to clear warp map")

        # The shader will be updated automatically when compilation completes
        # via process_shader_compilation_results() in the render loop

    def _copy_uniform_values(self, old_program, new_program):
        """Copy uniform values from old program to new program"""
        uniforms_to_preserve = [
            'time', 'animation_speed', 'rotation', 'trail_intensity',
            'glow_intensity', 'glow_radius', 'symmetry_mode', 'kaleidoscope_sections',
            'smoke_intensity', 'pulse_scale', 'mouse_position', 'resolution',
            'mouse_enabled', 'mouse_intensity', 'warp_first', 'bounce_enabled', 'bounce_height',
            'shockwave_data', 'ripple_data',
            'waveform_data', 'waveform_length', 'waveform_scale', 'waveform_style',
            'waveform_enabled', 'waveform_color'
        ]

        for uniform_name in uniforms_to_preserve:
            if uniform_name in old_program and uniform_name in new_program:
                try:
                    old_uniform = old_program[uniform_name]
                    new_uniform = new_program[uniform_name]
                    new_uniform.value = old_uniform.value
                except Exception:
                    pass  # Silently ignore uniform copy errors

    def _handle_program_update(self, purpose: str, new_program) -> None:
        """Handle program updates from shader compilation results.

        Args:
            purpose: Purpose of the compilation (e.g., "main_shader")
            new_program: The newly compiled program
        """
        # Update the main program if this was a main shader compilation
        if purpose.startswith("main_shader"):
            old_program = self.program

            # Copy uniform values from old shader to new shader
            if old_program:
                self._copy_uniform_values(old_program, new_program)

            # Update program and VAO
            self.program = new_program
            self.vao = self.ctx.vertex_array(
                self.program,
                [(self.vbo, "2f 2f", "in_position", "in_texcoord")],
            )

            # Release old resources
            if old_program:
                old_program.release()

    def increase_beats_per_change(self):
        """Increase beats per change by 1, maximum 64"""
        self.beats_per_change = min(64, self.beats_per_change + 1)
        logger.debug(f"Beats per change: {self.beats_per_change}")

    def decrease_beats_per_change(self):
        """Decrease beats per change by 1, minimum 1"""
        self.beats_per_change = max(1, self.beats_per_change - 1)
        logger.debug(f"Beats per change: {self.beats_per_change}")

    def toggle_transitions(self):
        """Toggle automatic transitions on/off"""
        self.transitions_paused = not self.transitions_paused
        logger.debug(f"Automatic transitions: {'paused' if self.transitions_paused else 'enabled'}")

    def update_silence_detection(self, current_time):
        """Update silence detection and logo fade state"""
        try:
            # Check if in test mode or if current amplitude is below silence threshold
            is_silent = (
                self.logo_test_mode or self.current_amplitude < self.silence_threshold
            )

            if is_silent:
                # Start tracking silence if not already
                if self.silence_start_time is None:
                    self.silence_start_time = current_time

                # Check if we've been silent long enough to show logo (or in test mode)
                silence_duration = current_time - self.silence_start_time
                if (
                    silence_duration >= self.silence_duration_threshold
                    or self.logo_test_mode
                ):
                    # Fade in logo
                    if not self.logo_visible:
                        self.logo_visible = True
                        if self.logo_test_mode:
                            logger.debug("Test mode - showing logo")
                        else:
                            logger.debug(f"Silence detected ({self.current_amplitude:.4f} < {self.silence_threshold:.4f}) - fading in logo")

                    # Increase logo alpha (fade in)
                    target_alpha = 1.0
                    fade_speed = self.logo_fade_speed * 0.016  # Normalize for 60fps
                    self.logo_fade_alpha = min(
                        target_alpha, self.logo_fade_alpha + fade_speed
                    )

                    # Fade waveform brightness down as logo fades in
                    target_waveform_brightness = self.waveform_silence_brightness
                    old_brightness = self.waveform_brightness_multiplier
                    self.waveform_brightness_multiplier = max(
                        target_waveform_brightness,
                        self.waveform_brightness_multiplier - fade_speed,
                    )

                    # Debug: Print when waveform brightness changes significantly
                    if abs(old_brightness - self.waveform_brightness_multiplier) > 0.05:
                        logger.debug("🔅 Waveform brightness: {self.waveform_brightness_multiplier:.2f} (fading down during silence)")
            else:
                # Audio detected - reset silence tracking and fade out logo
                self.silence_start_time = None

                if self.logo_visible and not self.logo_test_mode:
                    # Fade out logo
                    fade_speed = self.logo_fade_speed * 0.016  # Normalize for 60fps
                    self.logo_fade_alpha = max(0.0, self.logo_fade_alpha - fade_speed)
                    
                    # Fade waveform brightness back up as logo fades out
                    target_waveform_brightness = 1.0
                    old_brightness = self.waveform_brightness_multiplier
                    self.waveform_brightness_multiplier = min(
                        target_waveform_brightness,
                        self.waveform_brightness_multiplier + fade_speed,
                    )

                    # Debug: Print when waveform brightness changes significantly
                    if abs(old_brightness - self.waveform_brightness_multiplier) > 0.05:
                        logger.debug(f"🔆 Waveform brightness: {self.waveform_brightness_multiplier:.2f} (fading up as audio resumes)")

                    # Hide logo when fully faded out
                    if self.logo_fade_alpha <= 0.0:
                        self.logo_visible = False
                        logger.debug(f"🎵 Audio resumed ({self.current_amplitude:.4f} >= {self.silence_threshold:.4f}) - logo faded out")

        except Exception as e:
            logger.error(f"Error in silence detection: {e}")

    def toggle_logo_test(self):
        """Toggle logo test mode for debugging"""
        self.logo_test_mode = not self.logo_test_mode
        logger.debug(f"🧪 Logo test mode toggled: {self.logo_test_mode}")
        if self.logo_test_mode:
            logger.debug("🧪 Logo test mode enabled - logo will be visible, waveform dimmed")
            self.logo_visible = True
            self.logo_fade_alpha = 1.0
            self.waveform_brightness_multiplier = self.waveform_silence_brightness
            logger.debug(f"🧪 Set logo_visible={self.logo_visible}, logo_fade_alpha={self.logo_fade_alpha}, logo_texture={'exists' if self.logo_texture else 'None'}")
        else:
            logger.debug("🧪 Logo test mode disabled - waveform brightness restored")
            self.logo_visible = False
            self.logo_fade_alpha = 0.0
            self.waveform_brightness_multiplier = 1.0

    def update_anti_aliasing(self, samples):
        """Update anti-aliasing setting and reload logo texture if needed"""
        old_samples = self.anti_aliasing_samples
        self.anti_aliasing_samples = samples
        logger.debug(
            f"Anti-aliasing updated to: {samples}x MSAA"
            if samples > 0
            else "Anti-aliasing disabled"
        )

        # Reload logo texture to apply new anti-aliasing settings (supersampling factor may change)
        # Skip during initialization to prevent redundant loading
        if (
            self.logo_texture
            and old_samples != samples
            and not getattr(self, "_initializing", False)
        ):
            self.load_logo_texture()
            logger.debug("Logo texture reloaded with new anti-aliasing settings")

    def update_chunk_size(self, new_chunk_size):
        """Update audio chunk size and reinitialize GPU waveform system"""
        if hasattr(self.audio_processor, 'get_chunk_size') and self.audio_processor.get_chunk_size() == new_chunk_size:
            return  # No change needed
            
        logger.debug(f"Updating chunk size to: {new_chunk_size}")
        
        # Clear cached resampling indices since chunk size is changing
        if hasattr(self, '_resample_cache_key'):
            delattr(self, '_resample_cache_key')
        if hasattr(self, '_fft_resample_cache_key'):
            delattr(self, '_fft_resample_cache_key')
            
        # Release existing GPU waveform textures
        if hasattr(self, "waveform_texture") and self.waveform_texture is not None:
            self.waveform_texture.release()
            self.waveform_texture = None
        if hasattr(self, "fft_texture") and self.fft_texture is not None:
            self.fft_texture.release()
            self.fft_texture = None
            
        # The audio processor chunk size will be updated by the callback in main.py
        # The GPU waveform textures will be recreated on the next update_gpu_waveform call
        logger.debug(f"GPU waveform textures cleared for chunk size change to {new_chunk_size}")

    def calculate_logo_pulse(self, current_time):
        """Calculate smooth pulse scale with slow in and out breathing effect"""
        # Use a slower pulse rate for smooth breathing effect
        pulse_rate = 0.5  # 0.5 Hz = one complete pulse every 2 seconds

        # Create smooth sine wave pulse
        pulse_phase = current_time * pulse_rate * 2 * np.pi
        pulse_strength = np.sin(pulse_phase) * self.logo_pulse_intensity

        # Return scale multiplier (1.0 + pulse_strength)
        return 1.0 + pulse_strength

    def update_logo_overlay_transition(self, current_time: float):
        """Update logo overlay alpha and waveform fade-in for smooth transition from splash screen"""
        if not self.logo_surface or self.logo_overlay_start_time is None:
            self.logo_overlay_alpha = 0.0
            self.waveform_fade_alpha = 1.0
            return

        elapsed_time = current_time - self.logo_overlay_start_time

        if elapsed_time < self.logo_overlay_duration:
            # Show logo at full opacity for the first duration
            self.logo_overlay_alpha = 1.0
            self.waveform_fade_alpha = 0.0
        elif elapsed_time < self.logo_overlay_duration + self.logo_fade_duration:
            # Fade out logo and fade in waveform simultaneously
            fade_progress = (
                elapsed_time - self.logo_overlay_duration
            ) / self.logo_fade_duration
            self.logo_overlay_alpha = 1.0 - fade_progress
            self.waveform_fade_alpha = fade_progress
        else:
            # Logo completely faded out, waveform fully visible
            self.logo_overlay_alpha = 0.0
            self.waveform_fade_alpha = 1.0

    def render_logo_overlay(self):
        """Render logo overlay during startup using existing logo system"""
        if not self.logo_surface or self.logo_overlay_alpha <= 0.0:
            return

        # Use existing logo rendering system by temporarily setting logo_fade_alpha
        original_fade_alpha = getattr(self, "logo_fade_alpha", 0.0)
        self.logo_fade_alpha = self.logo_overlay_alpha

        # Render using existing logo system
        self.render_logo()

        # Restore original fade alpha
        self.logo_fade_alpha = original_fade_alpha

    def render_logo(self):
        """Render the KarmaViz logo with current fade alpha"""
        if not self.logo_texture:
            logger.error(f"Logo texture is None - cannot render logo")
            return
        if self.logo_fade_alpha <= 0.0:
            logger.error(f"Logo fade alpha is {self.logo_fade_alpha} - skipping render")
            return

        try:
            # Use the same size as splash screen - keep original logo proportions
            logo_width, logo_height = self.logo_size

            # Scale logo to fit nicely on screen (same as splash screen)
            screen_width, screen_height = self.window_size

            # Calculate scale to fit logo nicely (not too big, not too small) * self.logo_pulse_scale * self.logo_pulse_scale
            max_width = screen_width * 0.6  # Max 60% of screen width
            max_height = screen_height * 0.6  # Max 60% of screen height

            scale_x = max_width / logo_width
            scale_y = max_height / logo_height
            scale = (
                min(scale_x, scale_y) * 0.8
            )  # Use smaller scale to maintain aspect ratio, 80% of original size

            # Apply scale with heartbeat pulse effect
            scaled_width = logo_width * scale * self.logo_pulse_scale
            scaled_height = logo_height * scale * self.logo_pulse_scale

            # Convert to normalized coordinates for OpenGL (-1 to 1)
            norm_width = scaled_width / screen_width * 2.0
            norm_height = scaled_height / screen_height * 2.0

            # Center the logo
            norm_x = -norm_width / 2.0
            norm_y = -norm_height / 2.0

            # Create vertices for logo quad (counter-clockwise for proper face culling)
            logo_vertices = np.array(
                [
                    # positions                           # texture coords
                    norm_x,
                    norm_y,
                    0.0,
                    0.0,  # Bottom left
                    norm_x + norm_width,
                    norm_y,
                    1.0,
                    0.0,  # Bottom right
                    norm_x + norm_width,
                    norm_y + norm_height,
                    1.0,
                    1.0,  # Top right
                    norm_x,
                    norm_y + norm_height,
                    0.0,
                    1.0,  # Top left
                ],
                dtype=DATA_FORMAT,
            )

            # Create simple shader for logo rendering
            if not hasattr(self, "logo_program"):
                logo_vertex_shader = """
                #version 330 core
                in vec2 in_position;
                in vec2 in_texcoord;
                out vec2 texcoord;

                void main() {
                    gl_Position = vec4(in_position, 0.0, 1.0);
                    texcoord = in_texcoord;
                }
                """

                logo_fragment_shader = """
                #version 330 core
                in vec2 texcoord;
                out vec4 fragColor;
                uniform sampler2D logo_texture;
                uniform float alpha;
                uniform float hue_shift;

                // Convert RGB to HSV
                vec3 rgb2hsv(vec3 c) {
                    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
                    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
                    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
                    
                    float d = q.x - min(q.w, q.y);
                    float e = 1.0e-10;
                    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
                }

                // Convert HSV to RGB
                vec3 hsv2rgb(vec3 c) {
                    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
                }

                void main() {
                    // Enhanced anti-aliasing with 9-tap sampling pattern
                    vec2 texel_size = 1.0 / textureSize(logo_texture, 0);
                    
                    // 9-tap sampling pattern for superior anti-aliasing
                    vec4 center = texture(logo_texture, texcoord);
                    vec4 tl = texture(logo_texture, texcoord + vec2(-texel_size.x, -texel_size.y));
                    vec4 tm = texture(logo_texture, texcoord + vec2(0.0, -texel_size.y));
                    vec4 tr = texture(logo_texture, texcoord + vec2(texel_size.x, -texel_size.y));
                    vec4 ml = texture(logo_texture, texcoord + vec2(-texel_size.x, 0.0));
                    vec4 mr = texture(logo_texture, texcoord + vec2(texel_size.x, 0.0));
                    vec4 bl = texture(logo_texture, texcoord + vec2(-texel_size.x, texel_size.y));
                    vec4 bm = texture(logo_texture, texcoord + vec2(0.0, texel_size.y));
                    vec4 br = texture(logo_texture, texcoord + vec2(texel_size.x, texel_size.y));
                    
                    // Weighted average for better anti-aliasing
                    // Center pixel gets more weight, corners get less
                    vec4 smoothed_sample = (center * 4.0 + 
                                          (tm + ml + mr + bm) * 2.0 + 
                                          (tl + tr + bl + br) * 1.0) / 16.0;

                    // Convert to HSV for hue shifting
                    vec3 hsv = rgb2hsv(smoothed_sample.rgb);
                    
                    // Apply hue shift (only if the pixel has some saturation to avoid shifting grays)
                    if (hsv.y > 0.1) {  // Only shift colors with some saturation
                        hsv.x = fract(hsv.x + hue_shift);  // Shift hue and wrap around
                    }
                    
                    // Convert back to RGB
                    vec3 shifted_rgb = hsv2rgb(hsv);

                    // Use the smoothed texture's alpha channel and multiply by fade alpha
                    float final_alpha = smoothed_sample.a * alpha;

                    // Only output the pixel if it has some transparency
                    if (final_alpha < 0.01) {
                        discard;  // Discard fully transparent pixels
                    }

                    fragColor = vec4(shifted_rgb, final_alpha);
                }
                """

                self.logo_program = self.ctx.program(
                    vertex_shader=logo_vertex_shader,
                    fragment_shader=logo_fragment_shader,
                )

                # Create logo VBO and VAO
                self.logo_vbo = self.ctx.buffer(logo_vertices.tobytes())
                self.logo_vao = self.ctx.vertex_array(
                    self.logo_program,
                    [(self.logo_vbo, "2f 2f", "in_position", "in_texcoord")],
                )
            else:
                # Update existing VBO with new vertices
                self.logo_vbo.write(logo_vertices.tobytes())

            # Enable proper alpha blending for transparency
            self.ctx.enable(self.ctx.BLEND)
            self.ctx.blend_func = self.ctx.SRC_ALPHA, self.ctx.ONE_MINUS_SRC_ALPHA

            # Disable depth testing for overlay rendering
            self.ctx.disable(self.ctx.DEPTH_TEST)

            # Update hue shift for smooth rainbow cycling
            current_time = pygame.time.get_ticks() / 1000.0  # Convert to seconds
            self.logo_hue_shift = (
                current_time * self.logo_hue_speed
            ) % 1.0  # Cycle from 0.0 to 1.0

            # Calculate heartbeat pulse scale
            self.logo_pulse_scale = self.calculate_logo_pulse(current_time)

            # Bind logo texture and set uniforms
            self.logo_texture.use(0)
            self.logo_program["logo_texture"].value = 0
            self.logo_program["alpha"].value = self.logo_fade_alpha
            self.logo_program["hue_shift"].value = self.logo_hue_shift

            # Render logo quad using TRIANGLE_FAN
            self.logo_vao.render(mode=self.ctx.TRIANGLE_FAN)

            # Restore previous blend state
            self.ctx.blend_func = (
                self.ctx.ONE,
                self.ctx.ONE,
            )  # Restore to additive blending

        except Exception as e:
            logger.error(f"Error loading texture. {e}")

    def print_performance_stats(self) -> None:
        """Print performance statistics for all benchmarked functions."""
        from modules.benchmark import print_performance_stats

        print_performance_stats()

    def get_performance_summary(self) -> str:
        """Get a formatted performance summary string."""
        from modules.benchmark import get_performance_summary

        return get_performance_summary()

    def clear_performance_stats(self) -> None:
        """Clear all performance statistics."""
        monitor = get_performance_monitor()
        monitor.clear_stats()
        logger.debug("Performance statistics cleared.")

    def toggle_performance_monitoring(self) -> None:
        """Toggle performance monitoring on/off."""
        monitor = get_performance_monitor()
        if monitor.is_enabled():
            monitor.disable()
            logger.debug("Performance monitoring disabled.")
        else:
            monitor.enable()
            logger.debug("Performance monitoring enabled.")


# Splash screen functionality moved to modules/splash_screen.py

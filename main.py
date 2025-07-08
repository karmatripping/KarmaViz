#!/usr/bin/env python3
"""
KarmaViz - Audio Visualizer
Main entry point for the application
"""

import pygame
import sys
import os
import argparse
from PyQt5.QtWidgets import QApplication

# Add modules directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

# Import logging configuration
from modules.logging_config import init_logging, log_debug, log_info, log_error, log_critical

# Import non-audio modules first
from modules.config_menu_qt import ConfigMenuQt
from modules.preset_manager import PresetManager

# Import the main visualizer class
from modules.karmaviz import KarmaVisualizer

from moderngl import create_context
from config.constants import WIDTH, HEIGHT

# Global variables
fps = 60
selected_fullscreen_res_str = "Native"
frame_interval = 1.0 / fps

# Force SDL to use X11 if not already set (prevents Wayland crashes)
if os.environ.get("SDL_VIDEODRIVER") != "x11":
    os.environ["SDL_VIDEODRIVER"] = "x11"


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="KarmaViz - Audio Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Run with minimal logging (errors only)
  %(prog)s --debug           # Run with verbose debug logging
        """
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging (shows all messages)'
    )
    return parser.parse_args()


def run_main_loop(vis, config_menu, audio_processor, _ctx, preset_manager=None):
    """Main application loop - extracted for modularity"""
    global fps, frame_interval
    from time import sleep, time
    from modules.logging_config import get_logger
    from modules.benchmark import get_performance_monitor
    from config.constants import WIDTH, HEIGHT

    # Set the visualizer reference in the config menu for syncing settings
    config_menu.set_visualizer(vis)

    # Add fullscreen flag
    fullscreen = False
    # Persist previous window size for restoring after fullscreen
    prev_window_size = (WIDTH, HEIGHT)

    # Get successful MSAA setting from the current display
    successful_msaa = None
    try:
        # Try to detect current MSAA setting
        import pygame

        # This is a simplified approach - in practice we'd need to track this from window creation
        successful_msaa = 4  # Default assumption
    except:
        successful_msaa = None

    def try_msaa_setup(samples):
        """Try to set up MSAA with given sample count"""
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, samples)
        return samples

    def update_config_menu_setting(setting_name, value):
        """Update a setting in the config menu when changed via hotkey"""
        if config_menu and config_menu.visible:
            config_menu.update_setting_from_visualizer(setting_name, value)

    # Track last frame time for consistent updates
    frame_interval = 1.0 / fps

    # Performance monitoring variables
    last_performance_report = time()
    performance_report_interval = 60.0  # Report every 30 seconds

    try:
        while True:
            frame_start_time = time()
            # Event Handling
            # ... (rest of the main loop code)

            # Frame limiting for FPS control from config menu
            frame_end_time = time()
            elapsed = frame_end_time - frame_start_time
            # Use the latest frame_interval from the visualizer instance
            sleep_time = max(0, getattr(vis, 'frame_interval', 1.0/60) - elapsed)
            sleep(sleep_time)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise SystemExit
                elif event.type == pygame.VIDEORESIZE:
                    new_size = event.dict["size"]
                    log_debug(f"==> VIDEORESIZE event caught: {new_size}")  # Specific log

                    # Don't recreate the display mode - just update the viewport
                    # This prevents window jumping during resize
                    if not fullscreen:  # Only handle resize if NOT in fullscreen
                        # Update visualizer and menu with new size
                        vis.update_viewport(new_size)
                        config_menu.resize(new_size)  # Call menu resize

                        # Update the OpenGL viewport directly without recreating the window
                        vis.ctx.viewport = vis.viewport

                        log_debug(f"Updated viewport to: {vis.viewport}")
                    continue  # Skip other event processing for this frame
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        raise SystemExit
                    elif event.key == pygame.K_TAB:
                        log_debug("TAB key pressed - toggling menu")  # Debug print
                        config_menu.toggle()
                        log_debug(f"Menu visibility is now: {config_menu.visible}")  # Debug print
                        pygame.event.set_grab(False)  # Release input grab if any
                        continue
                    # Quick preset shortcuts
                    elif event.key in [pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4,
                                     pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9]:
                        if preset_manager is None:
                            log_error("Preset manager not available")
                            continue

                        # Get the slot number (0-9)
                        slot = event.key - pygame.K_0

                        # Check if Ctrl is pressed for saving
                        keys = pygame.key.get_pressed()
                        if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
                            # Save quick preset (Ctrl+0-9)
                            log_debug(f"💾 Saving quick preset {slot}...")
                            success = preset_manager.save_quick_preset(vis, slot)
                            if success:
                                log_debug(f"Quick preset {slot} saved!")
                            else:
                                log_error(f"Failed to save quick preset {slot}")
                        else:
                            # Load quick preset (0-9)
                            if preset_manager.quick_preset_exists(slot):
                                log_debug(f"📂 Loading quick preset {slot}...")
                                success = preset_manager.load_quick_preset(vis, slot)
                                if success:
                                    log_debug(f"Quick preset {slot} loaded!")
                                else:
                                    log_error(f"Failed to load quick preset {slot}")
                            else:
                                log_error(f"Quick preset {slot} does not exist")
                        continue
                    elif event.key == pygame.K_F11:
                        fullscreen = not fullscreen
                        target_size = None  # Reset target size
                        if fullscreen:
                            # Only update prev_window_size when entering fullscreen
                            screen = pygame.display.get_surface()
                            prev_window_size = (
                                screen.get_size() if screen else (WIDTH, HEIGHT)
                            )
                            # Get available fullscreen modes
                            available_modes = pygame.display.list_modes(
                                0, pygame.FULLSCREEN
                            )
                            if vis.selected_fullscreen_resolution != "Native":
                                log_debug(f"🎯 Using selected fullscreen resolution: {vis.selected_fullscreen_resolution}")
                                try:
                                    w_str, h_str = (
                                        vis.selected_fullscreen_resolution.split("x")
                                    )
                                    requested_size = (int(w_str), int(h_str))
                                    # Only use if available
                                    if (
                                        available_modes
                                        and requested_size in available_modes
                                    ):
                                        target_size = requested_size
                                    else:
                                        log_debug(f"[WARNING] Requested fullscreen resolution {requested_size} not available. Using closest available mode.")
                                        if available_modes and available_modes != -1:
                                            # Use the closest mode by area
                                            target_size = min(
                                                available_modes,
                                                key=lambda m: abs(
                                                    m[0] * m[1]
                                                    - requested_size[0]
                                                    * requested_size[1]
                                                ),
                                            )
                                        else:
                                            target_size = (
                                                1024,
                                                576,
                                            )  # Absolute fallback
                                            log_debug(f"list_modes() failed, using default fallback: {target_size}")
                                except ValueError:
                                    log_debug(f"Warning: Could not parse selected resolution '{vis.selected_fullscreen_resolution}'. Falling back to native.")
                                    if available_modes and available_modes != -1:
                                        target_size = available_modes[0]
                                    else:
                                        target_size = (1024, 576)  # Absolute fallback
                                        log_debug(f"list_modes() failed, using default fallback: {target_size}")
                            else:
                                # User selected Native, so use list_modes
                                log_debug("Using native fullscreen resolution (selected 'Native').")
                                if available_modes and available_modes != -1:
                                    target_size = available_modes[0]
                                else:
                                    target_size = (1024, 576)  # Absolute fallback
                                    log_debug(f"list_modes() failed, using default fallback: {target_size}")

                        # --- Now attempt to set the determined target_size ---
                        if (
                            target_size
                        ):  # Ensure we have a target size before proceeding
                            try:
                                log_debug(f"--> Attempting fullscreen set_mode({target_size})...")
                                # Preserve anti-aliasing settings for fullscreen
                                if successful_msaa:
                                    try_msaa_setup(successful_msaa)
                                    log_debug(f"Preserving {successful_msaa}x MSAA for fullscreen")

                                screen = pygame.display.set_mode(
                                    target_size,  # Use target_size here
                                    pygame.FULLSCREEN
                                    | pygame.OPENGL
                                    | pygame.DOUBLEBUF,
                                )
                                log_debug(f"Fullscreen mode set successfully to {target_size}")
                                log_debug(f"🎯 Applied resolution: {vis.selected_fullscreen_resolution}")
                                pygame.mouse.set_visible(False)
                                # Update visualizer and menu with the actual size used
                                vis.update_viewport(target_size)
                                config_menu.resize(target_size)  # Call menu resize
                            except pygame.error as e:
                                log_debug(f"*** ERROR ENTERING FULLSCREEN ({target_size}): {e} ***")
                                log_debug("*** Reverting to windowed mode. ***")
                                fullscreen = False  # Revert state
                                # Attempt to restore windowed mode
                                window_size = prev_window_size
                                try:
                                    screen = pygame.display.set_mode(
                                        window_size,
                                        pygame.OPENGL
                                        | pygame.DOUBLEBUF
                                        | pygame.RESIZABLE,
                                    )
                                    pygame.mouse.set_visible(True)
                                    vis.update_viewport(window_size)
                                    config_menu.resize(window_size)  # Call menu resize
                                except pygame.error as e2:
                                    log_debug(f"*** FAILED TO REVERT TO WINDOWED MODE: {e2} ***")
                                    # App might be in a bad state here
                        else:
                            # Return to windowed mode using previous window size if available
                            try:
                                window_size = prev_window_size
                            except NameError:
                                window_size = (WIDTH, HEIGHT)

                            # Preserve anti-aliasing settings for windowed mode
                            if successful_msaa:
                                try_msaa_setup(successful_msaa)
                                log_debug(f"Preserving {successful_msaa}x MSAA for windowed mode")

                            screen = pygame.display.set_mode(
                                window_size,
                                pygame.OPENGL
                                | pygame.DOUBLEBUF
                                | pygame.RESIZABLE,  # Added RESIZABLE flag back
                            )
                            pygame.mouse.set_visible(True)
                            vis.update_viewport(window_size)
                            config_menu.resize(window_size)  # Call menu resize

                    # Main Controls

                    elif event.key == pygame.K_p:
                        vis.toggle_pulse()
                        update_config_menu_setting("pulse_enabled", vis.pulse_enabled)
                    elif event.key == pygame.K_KP5:  # Numpad 5
                        vis.toggle_bounce()
                        update_config_menu_setting("bounce_enabled", vis.bounce_enabled)
                    elif event.key == pygame.K_i:
                        vis.toggle_mouse_interaction()
                        # No corresponding config menu setting for mouse interaction
                    elif event.key == pygame.K_s:
                        vis.toggle_spectrogram_overlay()
                        update_config_menu_setting("spectrogram_enabled", vis.show_spectrogram_overlay)
                    elif event.key == pygame.K_w:
                        vis.cycle_gpu_waveform()
                        # No corresponding config menu setting for waveform cycling
                    elif event.key == pygame.K_r:
                        vis.cycle_rotation_mode()
                        update_config_menu_setting("rotation_mode", vis.rotation_mode)
                    elif (
                        event.key == pygame.K_t
                        and pygame.key.get_mods() & pygame.KMOD_SHIFT
                    ):
                        vis.decrease_trail_intensity()
                        update_config_menu_setting("trail_intensity", vis.trail_intensity)
                    elif event.key == pygame.K_t:
                        vis.increase_trail_intensity()
                        update_config_menu_setting("trail_intensity", vis.trail_intensity)
                    elif (
                        event.key == pygame.K_g
                        and pygame.key.get_mods() & pygame.KMOD_SHIFT
                    ):
                        vis.decrease_glow_intensity()
                        update_config_menu_setting("glow_intensity", vis.glow_intensity)
                    elif event.key == pygame.K_g:
                        vis.increase_glow_intensity()
                        update_config_menu_setting("glow_intensity", vis.glow_intensity)
                    elif event.key == pygame.K_m:
                        vis.cycle_symmetry_mode()
                        update_config_menu_setting("symmetry_mode", vis.symmetry_mode)

                    # Intensity Controls
                    elif event.key == pygame.K_LEFTBRACKET:
                        vis.decrease_pulse_intensity()
                        update_config_menu_setting("pulse_intensity", vis.pulse_intensity)
                    elif event.key == pygame.K_RIGHTBRACKET:
                        vis.increase_pulse_intensity()
                        update_config_menu_setting("pulse_intensity", vis.pulse_intensity)

                    elif (
                        event.key == pygame.K_f
                        and pygame.key.get_mods() & pygame.KMOD_SHIFT
                    ):
                        vis.decrease_smoke_intensity()
                        update_config_menu_setting("smoke_intensity", vis.smoke_intensity)
                    elif event.key == pygame.K_f:
                        vis.increase_smoke_intensity()
                        update_config_menu_setting("smoke_intensity", vis.smoke_intensity)

                    # Scale Controls
                    elif (
                        event.key == pygame.K_DOWN
                        and pygame.key.get_mods() & pygame.KMOD_SHIFT
                    ):
                        vis.decrease_glow_radius()
                        update_config_menu_setting("glow_radius", vis.glow_radius)
                    elif (
                        event.key == pygame.K_UP
                        and pygame.key.get_mods() & pygame.KMOD_SHIFT
                    ):
                        vis.increase_glow_radius()
                        update_config_menu_setting("glow_radius", vis.glow_radius)
                    elif event.key == pygame.K_DOWN:
                        vis.decrease_waveform_scale()
                        update_config_menu_setting("waveform_scale", vis.waveform_scale)
                    elif event.key == pygame.K_UP:
                        vis.increase_waveform_scale()
                        update_config_menu_setting("waveform_scale", vis.waveform_scale)

                    # Speed Controls
                    elif event.key == pygame.K_KP_MINUS:  # Numpad -
                        vis.decrease_animation_speed()
                        update_config_menu_setting("animation_speed", vis.animation_speed)
                    elif event.key == pygame.K_KP_PLUS:  # Numpad +
                        vis.increase_animation_speed()
                        update_config_menu_setting("animation_speed", vis.animation_speed)
                    elif event.key == pygame.K_KP_DIVIDE:  # Numpad /
                        vis.decrease_audio_speed_boost()
                        update_config_menu_setting("audio_speed_boost", vis.audio_speed_boost)
                    elif event.key == pygame.K_KP_MULTIPLY:  # Numpad *
                        vis.increase_audio_speed_boost()
                        update_config_menu_setting("audio_speed_boost", vis.audio_speed_boost)

                    elif event.key == pygame.K_KP1:  # Numpad 1
                        vis.decrease_palette_speed()
                        update_config_menu_setting("palette_speed", vis.palette_rotation_speed)
                    elif event.key == pygame.K_KP3:  # Numpad 3
                        vis.increase_palette_speed()
                        update_config_menu_setting("palette_speed", vis.palette_rotation_speed)
                    elif event.key == pygame.K_KP4:  # Numpad 4
                        vis.decrease_color_cycle_speed()
                        update_config_menu_setting("color_cycle_speed", vis.color_cycle_speed_multiplier)
                    elif event.key == pygame.K_KP6:  # Numpad 6
                        vis.increase_color_cycle_speed()
                        update_config_menu_setting("color_cycle_speed", vis.color_cycle_speed_multiplier)

                    # Rotation Controls
                    elif event.key == pygame.K_MINUS:  # - key
                        keys = pygame.key.get_pressed()
                        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                            # Shift + - : Decrease rotation amplitude
                            vis.decrease_rotation_amplitude()
                            update_config_menu_setting("rotation_amplitude", vis.rotation_amplitude)
                        else:
                            # - : Decrease rotation speed
                            vis.decrease_rotation_speed()
                            update_config_menu_setting("rotation_speed", vis.rotation_speed)
                    elif event.key == pygame.K_EQUALS:  # = key
                        keys = pygame.key.get_pressed()
                        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                            # Shift + = : Increase rotation amplitude
                            vis.increase_rotation_amplitude()
                            update_config_menu_setting("rotation_amplitude", vis.rotation_amplitude)
                        else:
                            # = : Increase rotation speed
                            vis.increase_rotation_speed()
                            update_config_menu_setting("rotation_speed", vis.rotation_speed)

                    # Beat Controls
                    elif event.key == pygame.K_COMMA:
                        vis.decrease_beats_per_change()
                        update_config_menu_setting("beats_per_change", vis.beats_per_change)
                    elif event.key == pygame.K_PERIOD:
                        vis.increase_beats_per_change()
                        update_config_menu_setting("beats_per_change", vis.beats_per_change)
                    elif event.key == pygame.K_KP7:  # Numpad 7
                        audio_processor.decrease_beat_sensitivity()
                        update_config_menu_setting("beat_sensitivity", audio_processor.get_beat_sensitivity())
                    elif event.key == pygame.K_KP9:  # Numpad 9
                        audio_processor.increase_beat_sensitivity()
                        update_config_menu_setting("beat_sensitivity", audio_processor.get_beat_sensitivity())

                    # Warp Map Controls
                    elif event.key == pygame.K_SPACE:  # Space - Manual warp map change
                        if not vis.warp_map_locked:
                            vis.cycle_to_random_warp_map()
                        else:
                            log_debug("🔒 Warp map changes locked - close config menu to resume")
                    elif event.key == pygame.K_BACKSPACE:  # Backspace - Clear warp map
                        if not vis.warp_map_locked:
                            vis.clear_warp_map()
                        else:
                            log_debug("🔒 Warp map changes locked - close config menu to resume")
                    elif event.key == pygame.K_SLASH:  # / - Toggle automatic transitions
                        vis.toggle_transitions()
                        update_config_menu_setting("transitions_paused", vis.transitions_paused)
                    elif event.key == pygame.K_PAGEDOWN:  # Page Down - Decrease warp intensity
                        vis.decrease_warp_intensity()
                        update_config_menu_setting("warp_intensity", vis.warp_intensity)
                    elif event.key == pygame.K_PAGEUP:  # Page Up - Increase warp intensity
                        vis.increase_warp_intensity()
                        update_config_menu_setting("warp_intensity", vis.warp_intensity)

                    # Bounce Controls
                    elif event.key == pygame.K_KP2:  # Numpad 2
                        vis.decrease_bounce_intensity()
                        update_config_menu_setting("bounce_intensity", vis.bounce_intensity_multiplier)
                    elif event.key == pygame.K_KP8:  # Numpad 8
                        vis.increase_bounce_intensity()
                        update_config_menu_setting("bounce_intensity", vis.bounce_intensity_multiplier)

                    # Performance Monitoring Controls
                    elif event.key == pygame.K_F1:  # F1 - Print performance stats
                        vis.print_performance_stats()
                    elif event.key == pygame.K_F2:  # F2 - Clear performance stats
                        vis.clear_performance_stats()
                    elif event.key == pygame.K_F3:  # F3 - Toggle performance monitoring
                        vis.toggle_performance_monitoring()
                    elif (
                        event.key == pygame.K_F4
                    ):  # F4 - Show shader compilation status
                        status = vis.shader_manager.get_shader_compilation_status()
                        log_debug("\n" + "=" * 50)
                        log_debug("SHADER COMPILATION STATUS")
                        log_debug("=" * 50)
                        for purpose, status_val in status.items():
                            if not purpose.startswith("_"):
                                log_debug(f"{purpose}: {status_val}")
                        log_debug(f"Queue size: {status.get('_queue_size', '0')}")
                        log_debug(f"Active compilations: {status.get('_active_compilations', '0')}")
                        log_debug("=" * 50)
                    
                    elif event.key == pygame.K_b:  # B - Show benchmark report
                        from modules.benchmark import print_bottleneck_report, get_benchmark_coverage_report
                        print("\n" + get_benchmark_coverage_report())
                        print_bottleneck_report(threshold_ms=2.0)  # Lower threshold for more detailed analysis

                # Pass event to config menu if visible
                if config_menu.visible and config_menu.handle_event(event):
                    continue

            # Get audio data and process it
            audio_data = audio_processor.get_audio_data()
            if audio_data:
                # Update mouse position if interaction is enabled
                if vis.mouse_interaction_enabled:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    vis.mouse_position = [float(mouse_x), float(mouse_y)]
                    vis.program["mouse_position"] = vis.mouse_position  # type: ignore

                # Draw waveform and update state
                vis.draw_waveform((255, 100, 100))
                vis.current_amplitude = audio_data.amplitude
                vis.beat_detected = audio_data.beat_detected
                vis.current_mood = {
                    "energy": audio_data.energy,
                    "warmth": audio_data.warmth,
                }

                # Check >and apply completed shader compilations from preset loading
                if preset_manager:
                    applied_count = preset_manager.check_and_apply_shader_compilations(vis)
                    if applied_count > 0:
                        log_debug(f"Applied {applied_count} completed shader compilation(s) from preset loading")

                # Render visualization
                vis.render()

                # Draw config menu on top if visible
                if config_menu.visible:
                    config_menu.render()  # This is a dummy method for PyQt5 menu
                    pygame.mouse.set_visible(
                        True
                    )  # Ensure mouse is visible when menu is shown

                # Update display
                pygame.display.flip()

            # Periodic performance reporting
            current_time = time()
            if current_time - last_performance_report >= performance_report_interval:
                monitor = get_performance_monitor()
                if monitor.is_enabled():
                    vis.print_performance_stats()
                last_performance_report = current_time

    except SystemExit:
        log_info("Exiting gracefully...")
    except Exception as e:
        log_error(f"Error in main loop: {e}")

    finally:
        log_info("Cleaning up main loop...")
        # Note: cleanup is handled by the calling main() function


def main():
    """Main application entry point"""
    global fps, selected_fullscreen_res_str, frame_interval

    # Parse command line arguments
    args = parse_arguments()
    
    
    # Initialize logging based on debug flag
    init_logging(debug=args.debug)
    
    # Create Qt application early
    qt_app = QApplication(sys.argv)

    audio_processor = None
    ctx = None
    compiled_programs = None

    try:
        # Set SDL window to be resizable
        os.environ["SDL_VIDEO_WINDOW_RESIZABLE"] = "1"

        # Initialize Pygame
        pygame.init()

        # Set window icon for taskbar display
        try:
            icon_path = os.path.join(os.path.dirname(__file__), 'karmaviz_icon.png')
            if os.path.exists(icon_path):
                icon = pygame.image.load(icon_path)
                pygame.display.set_icon(icon)
                log_debug(f"Window icon set from: {icon_path}")
            else:
                log_debug(f"Icon file not found at: {icon_path}")
        except Exception as e:
            log_debug(f"Could not set window icon: {e}")

        # Set up OpenGL attributes before creating any windows
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
        )
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 16)

        # Try to enable multisampling for anti-aliasing
        try:
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
            log_debug("Multisampling attributes set in pygame")
        except Exception as e:
            log_error(f"Could not set multisampling attributes in pygame: {e}")

        # Create the main OpenGL window (this will be used for both splash and main visualization)
        try:
            pygame.display.set_mode(
                (WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
            )
            log_debug("Main OpenGL window created")
        except Exception as e:
            log_error(f"Error creating window with multisampling: {e}")
            # Fall back to standard window without multisampling
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 0)
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 0)
            pygame.display.set_mode(
                (WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
            )
            log_debug("Main window created with fallback settings (no multisampling)")

        # Create OpenGL context
        import moderngl

        try:
            ctx = moderngl.create_context()
            log_debug("OpenGL context created")
        except Exception as e:
            log_critical(f"Failed to create OpenGL context: {e}")
            pygame.quit()
            return

        # Load logo for initial display
        logo_surface = None
        try:
            logo_surface = pygame.image.load("karmaviz_logo.png")
            log_debug("Logo loaded for initial display")
        except Exception as e:
            log_debug(f"Could not load logo: {e}")

        # We'll compile shaders in the visualizer constructor
        compiled_programs = None

        # Start audio processor (lazy import to avoid early sounddevice issues)
        try:
            from modules.audio_handler import AudioProcessor
            audio_processor = AudioProcessor()
            audio_processor.start()
            log_debug("Audio processor started successfully")
        except Exception as e:
            log_error(f"Failed to start audio processor: {e}")
            log_error("This might be due to audio system compatibility issues.")
            log_error("Creating dummy audio processor for visual-only mode...")

            # Use the DummyAudioProcessor from audio_handler module
            from modules.audio_handler import DummyAudioProcessor
            audio_processor = DummyAudioProcessor()
            audio_processor.start()
            log_error("Dummy audio processor created - visual-only mode active")

        # Set window caption now that we're transitioning to main visualization
        pygame.display.set_caption("KarmaViz - Audio Visualizer")
        window_size = (WIDTH, HEIGHT)

        # Create visualizer with logo for initial display
        vis = KarmaVisualizer(
            window_size,
            audio_processor,
            ctx,
            compiled_programs,
            logo_surface=logo_surface,
        )

        # Create preset manager
        preset_manager = PresetManager("presets")
        log_debug("Preset manager initialized")

        # Create config menu
        config_path = os.path.expanduser("~/.config/KarmaViz/settings.json")
        config_menu = ConfigMenuQt(config_path)

        # Set managers properly
        config_menu.set_palette_manager(vis.palette_manager)
        config_menu.set_shader_manager(vis.shader_manager)
        config_menu.set_shader_compiler(vis.shader_compiler)
        config_menu.set_preset_manager(preset_manager)
        config_menu.set_visualizer(vis)

        # Force dialog rebuild if it already exists
        config_menu.rebuild_tabs()

        # Set up config menu connections
        vis._config_menu_ref = config_menu

        # Set up palette preview callback
        def palette_preview_callback(palette_name, mode=None):
            """Handle palette preview from config menu"""
            try:
                vis.set_selected_palette(palette_name)
                log_debug(f"Preview palette: {palette_name}")
            except Exception as e:
                log_error(f"Error applying palette preview: {e}")

        config_menu.set_palette_preview_callback(palette_preview_callback)

        # Set up warp map preview callback with shader compilation
        def warp_map_preview_callback(warp_map_name, persistent=False):
            """Handle warp map preview from config menu with integrated shader compilation"""
            try:
                if warp_map_name == "None" or warp_map_name == "" or warp_map_name is None:
                    # Clear warp map - recompile with no warp maps
                    vis.active_warp_map_name = None
                    vis.active_warp_map_index = -1

                    # If persistent, clear the state and unlock automatic changes
                    if persistent:
                        vis.persistent_warp_map = None
                        vis.warp_map_locked = False
                        log_debug("🚫 Cleared persistent warp map selection - automatic changes resumed")

                    # Recompile main shader without warp maps
                    new_program = vis.shader_compiler.compile_main_shader_with_warp([], vis.shader_manager, vis.current_waveform_name)
                    if new_program:
                        old_program = vis.program

                        # Copy over existing uniform values to preserve effects
                        uniforms_to_preserve = [
                            'time', 'animation_speed', 'rotation', 'trail_intensity',
                            'glow_intensity', 'symmetry_mode', 'kaleidoscope_sections',
                            'smoke_intensity', 'pulse_scale', 'mouse_position', 'resolution',
                            'mouse_enabled', 'warp_first', 'bounce_enabled', 'bounce_height',
                            'waveform_data', 'waveform_length', 'waveform_scale',
                            'waveform_enabled', 'waveform_color'
                        ]

                        for uniform_name in uniforms_to_preserve:
                            if uniform_name in old_program and uniform_name in new_program:
                                try:
                                    old_uniform = old_program[uniform_name]
                                    new_uniform = new_program[uniform_name]
                                    new_uniform.value = old_uniform.value
                                except Exception as e:
                                    log_debug(f"[DEBUG] Failed to copy {uniform_name}: {e}")

                        vis.program = new_program
                        vis.vao = vis.ctx.vertex_array(
                            vis.program, [(vis.vbo, "2f 2f", "in_position", "in_texcoord")]
                        )
                        # Clear uniform cache so all uniforms are re-applied to the new program
                        if hasattr(vis, '_last_uniforms'):
                            vis._last_uniforms.clear()
                        if old_program:
                            old_program.release()
                        log_debug("🚫 Cleared warp map and recompiled shader")
                    else:
                        log_debug("[DEBUG] Failed to recompile shader without warp map")
                else:
                    # Apply warp map by recompiling the main shader
                    vis.active_warp_map_name = warp_map_name
                    vis.active_warp_map_index = 0  # Use first warp map slot

                    # If persistent, store the warp map and lock automatic changes
                    if persistent:
                        vis.persistent_warp_map = warp_map_name
                        vis.warp_map_locked = True  # Prevent automatic warp map changes
                        log_debug(f"🔒 Set persistent warp map: {warp_map_name} - automatic changes paused")

                    new_program = vis.shader_compiler.compile_main_shader_with_warp([warp_map_name], vis.shader_manager, vis.current_waveform_name)
                    if new_program:
                        old_program = vis.program

                        # List of uniforms to preserve
                        uniforms_to_preserve = [
                            'time', 'animation_speed', 'rotation', 'trail_intensity',
                            'glow_intensity', 'symmetry_mode', 'kaleidoscope_sections',
                            'smoke_intensity', 'pulse_scale', 'mouse_position', 'resolution',
                            'mouse_enabled', 'warp_first', 'bounce_enabled', 'bounce_height',
                            'waveform_data', 'waveform_length', 'waveform_scale',
                            'waveform_enabled', 'waveform_color'
                        ]

                        # Copy values from old program if they exist
                        for uniform_name in uniforms_to_preserve:
                            if uniform_name in old_program and uniform_name in new_program:
                                try:
                                    # Get the value from old program and set it in new program
                                    old_uniform = old_program[uniform_name]
                                    new_uniform = new_program[uniform_name]
                                    new_uniform.value = old_uniform.value
                                except Exception as e:
                                    log_debug(f"[DEBUG] Failed to copy {uniform_name}: {e}")

                        vis.program = new_program
                        vis.vao = vis.ctx.vertex_array(
                            vis.program, [(vis.vbo, "2f 2f", "in_position", "in_texcoord")]
                        )
                        # Clear uniform cache so all uniforms are re-applied to the new program
                        if hasattr(vis, '_last_uniforms'):
                            vis._last_uniforms.clear()
                        if old_program:
                            old_program.release()
                        log_debug(f"🌀 Applied warp map: {warp_map_name}")
                    else:
                        log_debug(f"[DEBUG] Failed to compile shader with warp map '{warp_map_name}'")
            except Exception as e:
                log_debug(f"[DEBUG] Error applying warp map preview: {e}")

        config_menu.set_warp_map_preview_callback(warp_map_preview_callback)

        # Set up waveform preview callback with shader compilation
        def waveform_preview_callback(waveform_info, persistent=False):
            """Handle waveform preview from config menu with integrated shader compilation"""
            try:
                if waveform_info is None:
                    # Clear waveform - revert to default
                    vis.current_waveform_name = "normal"

                    # If persistent, clear the state and unlock automatic changes
                    if persistent:
                        vis.persistent_waveform = None
                        vis.waveform_locked = False
                        log_debug("🚫 Cleared persistent waveform selection - automatic changes resumed")

                    # Recompile main shader with default waveform
                    warp_maps = (
                        [vis.active_warp_map_name] if vis.active_warp_map_name else []
                    )
                    new_program = vis.shader_compiler.compile_main_shader_with_warp(
                        warp_maps, vis.shader_manager, "normal"
                    )
                    if new_program:
                        old_program = vis.program

                        # Copy over existing uniform values to preserve effects
                        uniforms_to_preserve = [
                            "time",
                            "animation_speed",
                            "rotation",
                            "trail_intensity",
                            "glow_intensity",
                            "symmetry_mode",
                            "kaleidoscope_sections",
                            "smoke_intensity",
                            "pulse_scale",
                            "mouse_position",
                            "resolution",
                            "mouse_enabled",
                            "warp_first",
                            "bounce_enabled",
                            "bounce_height",
                            "waveform_data",
                            "waveform_length",
                            "waveform_scale",
                            "waveform_style",
                            "waveform_enabled",
                            "waveform_color",
                        ]

                        for uniform_name in uniforms_to_preserve:
                            if (
                                uniform_name in old_program
                                and uniform_name in new_program
                            ):
                                try:
                                    old_uniform = old_program[uniform_name]
                                    new_uniform = new_program[uniform_name]
                                    new_uniform.value = old_uniform.value
                                except Exception as e:
                                    log_debug(f"[DEBUG] Failed to copy {uniform_name}: {e}")

                        vis.program = new_program
                        vis.vao = vis.ctx.vertex_array(
                            vis.program,
                            [(vis.vbo, "2f 2f", "in_position", "in_texcoord")],
                        )
                        # Clear uniform cache so all uniforms are re-applied to the new program
                        if hasattr(vis, "_last_uniforms"):
                            vis._last_uniforms.clear()
                        if old_program:
                            old_program.release()
                        log_debug("🚫 Cleared waveform and recompiled shader with default")
                    else:
                        log_debug("[DEBUG] Failed to recompile shader with default waveform")
                else:
                    # Apply waveform by temporarily updating the waveform manager and recompiling
                    waveform_name = waveform_info.name
                    vis.current_waveform_name = waveform_name

                    # If persistent, store the waveform and lock automatic changes
                    if persistent:
                        vis.persistent_waveform = waveform_name
                        vis.waveform_locked = True  # Prevent automatic waveform changes
                        log_debug(f"🔒 Set persistent waveform: {waveform_name} - automatic changes paused")

                    # Temporarily update the waveform in the shader manager
                    original_waveform = None
                    if waveform_name in vis.shader_manager.waveform_manager.waveforms:
                        original_waveform = (
                            vis.shader_manager.waveform_manager.waveforms[waveform_name]
                        )

                    # Update with the new waveform info (for live editing)
                    vis.shader_manager.waveform_manager.waveforms[waveform_name] = (
                        waveform_info
                    )

                    # Recompile main shader with the updated waveform
                    warp_maps = (
                        [vis.active_warp_map_name] if vis.active_warp_map_name else []
                    )
                    new_program = vis.shader_compiler.compile_main_shader_with_warp(
                        warp_maps, vis.shader_manager, waveform_name
                    )
                    if new_program:
                        old_program = vis.program

                        # List of uniforms to preserve
                        uniforms_to_preserve = [
                            "time",
                            "animation_speed",
                            "rotation",
                            "trail_intensity",
                            "glow_intensity",
                            "symmetry_mode",
                            "kaleidoscope_sections",
                            "smoke_intensity",
                            "pulse_scale",
                            "mouse_position",
                            "resolution",
                            "mouse_enabled",
                            "warp_first",
                            "bounce_enabled",
                            "bounce_height",
                            "waveform_data",
                            "waveform_length",
                            "waveform_scale",
                            "waveform_style",
                            "waveform_enabled",
                            "waveform_color",
                        ]

                        for uniform_name in uniforms_to_preserve:
                            if (
                                uniform_name in old_program
                                and uniform_name in new_program
                            ):
                                try:
                                    old_uniform = old_program[uniform_name]
                                    new_uniform = new_program[uniform_name]
                                    new_uniform.value = old_uniform.value
                                except Exception as e:
                                    log_debug(f"[DEBUG] Failed to copy {uniform_name}: {e}")

                        vis.program = new_program
                        vis.vao = vis.ctx.vertex_array(
                            vis.program,
                            [(vis.vbo, "2f 2f", "in_position", "in_texcoord")],
                        )
                        # Clear uniform cache so all uniforms are re-applied to the new program
                        if hasattr(vis, "_last_uniforms"):
                            vis._last_uniforms.clear()
                        if old_program:
                            old_program.release()
                        log_debug(f"Applied waveform '{waveform_name}' and recompiled shader")
                    else:
                        log_debug(f"[DEBUG] Failed to compile shader with waveform '{waveform_name}'")
                        # Restore original waveform if compilation failed
                        if original_waveform:
                            vis.shader_manager.waveform_manager.waveforms[
                                waveform_name
                            ] = original_waveform
            except Exception as e:
                log_debug(f"[DEBUG] Error applying waveform preview: {e}")

        config_menu.set_waveform_preview_callback(waveform_preview_callback)

        # Set up persistent warp map clearing callback
        def clear_persistent_warp_map():
            """Clear persistent warp map when config menu is closed"""
            if hasattr(vis, 'persistent_warp_map'):
                try:
                    vis.persistent_warp_map = None
                    vis.warp_map_locked = False
                    log_debug("🔓 Cleared persistent warp map - automatic changes resumed")
                except Exception as e:
                    log_error(f"Error clearing persistent warp map: {e}")

        config_menu.set_persistent_warp_restore_callback(clear_persistent_warp_map)

        # Set up persistent waveform clearing callback
        def clear_persistent_waveform():
            """Clear persistent waveform when config menu is closed"""
            if hasattr(vis, "persistent_waveform"):
                try:
                    vis.persistent_waveform = None
                    vis.waveform_locked = False
                    log_debug("🔓 Cleared persistent waveform - automatic changes resumed")
                except Exception as e:
                    log_error(f"Error clearing persistent waveform: {e}")

        config_menu.set_persistent_waveform_restore_callback(clear_persistent_waveform)

        # Set up palette preview callback
        def palette_preview_callback(palette_name, mode=None):
            """Handle palette preview from config menu"""
            try:
                vis.set_selected_palette(palette_name)
                log_debug(f"Preview palette: {palette_name}")
            except Exception as e:
                log_error(f"Error applying palette preview: {e}")

        config_menu.set_palette_preview_callback(palette_preview_callback)

        # Register all visualizer callbacks for config menu settings
        def register_visualizer_callbacks():
            """Register callbacks to connect config menu settings to visualizer"""
            global fps, frame_interval

            def update_fps(new_fps):
                global fps, frame_interval
                fps = new_fps
                frame_interval = 1.0 / fps
                # Update visualizer FPS attributes (now guaranteed to exist)
                vis.fps = new_fps
                vis.frame_interval = 1.0 / new_fps
                log_debug(f"FPS updated to: {fps} (frame_interval: {frame_interval:.4f}s)")

            callbacks = {
                "width": lambda v: setattr(vis, "width", v),
                "height": lambda v: setattr(vis, "height", v),
                "fps": update_fps,
                "rotation_mode": lambda v: vis.set_rotation_mode(v),
                "rotation_speed": lambda v: vis.set_rotation_speed(v),
                "rotation_amplitude": lambda v: vis.set_rotation_amplitude(v),
                "pulse_enabled": lambda v: setattr(vis, "pulse_enabled", v),
                # Ensure pulse_intensity updates both multiplier and base value for compatibility
                "pulse_intensity": lambda v: [
                    setattr(vis, "pulse_intensity_multiplier", v),
                    setattr(vis, "pulse_intensity", v),
                ],
                "trail_intensity": lambda v: setattr(vis, "trail_intensity", v),
                "glow_intensity": lambda v: setattr(vis, "glow_intensity", v),
                "palette_speed": lambda v: setattr(vis, "palette_rotation_speed", v),
                "gpu_waveform_random": lambda v: setattr(vis, "gpu_waveform_random", v),
                "animation_speed": lambda v: setattr(vis, "animation_speed", v),
                "audio_speed_boost": lambda v: setattr(vis, "audio_speed_boost", v),
                "symmetry_mode": lambda v: setattr(vis, "symmetry_mode", v),
                "smoke_intensity": lambda v: setattr(vis, "smoke_intensity", v),
                "warp_intensity": lambda v: setattr(vis, "warp_intensity", v),
                "beat_sensitivity": lambda v: (
                    audio_processor.set_beat_sensitivity(v)
                    if hasattr(audio_processor, "set_beat_sensitivity")
                    else None
                ),
                "chunk_size": lambda v: (
                    audio_processor.set_chunk_size(v)
                    if hasattr(audio_processor, "set_chunk_size")
                    else None
                ),
                "sample_rate": lambda v: (
                    audio_processor.set_sample_rate(v)
                    if hasattr(audio_processor, "set_sample_rate")
                    else None
                ),
                "color_cycle_speed": lambda v: setattr(
                    vis, "color_cycle_speed_multiplier", v
                ),
                "palette_transition_speed": lambda v: setattr(
                    vis, "palette_transition_speed", v
                ),
                "color_transition_smoothness": lambda v: setattr(
                    vis, "color_transition_smoothness", v
                ),
                "transitions_paused": lambda v: setattr(vis, "transitions_paused", v),
                "beats_per_change": lambda v: setattr(vis, "beats_per_change", v),
                "waveform_scale": lambda v: setattr(vis, "waveform_scale", v),
                "gpu_waveform_enabled": lambda v: setattr(
                    vis, "gpu_waveform_enabled", v
                ),
                "warp_first": lambda v: setattr(vis, "warp_first_enabled", v),
                "bounce_enabled": lambda v: setattr(vis, "bounce_enabled", v),
                "bounce_intensity": lambda v: setattr(
                    vis, "bounce_intensity_multiplier", v
                ),
                "fullscreen_resolution": lambda v: setattr(
                    vis, "selected_fullscreen_resolution", v
                ),
                "anti_aliasing": lambda v: vis.update_anti_aliasing(v),
                "selected_palette": lambda v: vis.set_selected_palette(v),

            }

            for setting, callback in callbacks.items():
                config_menu.register_callback(setting, callback)
            log_debug(f"[DEBUG] Registered config menu callbacks: {list(config_menu.callbacks.keys())}")

        register_visualizer_callbacks()

        # Apply initial settings from config menu to visualizer
        def apply_initial_settings():
            """Apply all current config menu settings to the visualizer"""
            log_debug("Applying initial settings from config menu...")

            # Get current settings from config menu
            settings = config_menu.settings

            # Apply each setting that has a callback
            applied_count = 0
            for setting_name, value in settings.items():
                if setting_name in config_menu.callbacks:
                    try:
                        config_menu.callbacks[setting_name](value)
                        applied_count += 1
                    except Exception as e:
                        log_error(f"Warning: Failed to apply setting {setting_name}={value}: {e}")

            log_debug(f"Applied {applied_count} initial settings to visualizer")

        apply_initial_settings()

        print("KarmaViz initialized successfully!")
        print("Press 'TAB' to open the configuration menu")
        print("Press 'F11' to toggle fullscreen")
        print("Press 'q' or ESC to quit")

        # Main application loop would go here
        # For now, we'll import and call the existing main loop from karmaviz.py
        # This is a temporary solution until we fully refactor the main loop

        # Import the main loop function from karmaviz


        # Run the main loop
        run_main_loop(vis, config_menu, audio_processor, ctx, preset_manager)

    except KeyboardInterrupt:
        log_error("\nReceived keyboard interrupt. Shutting down gracefully...")
   
    except Exception as e:
        log_critical(f"Fatal error in main: {e}")
   
    finally:
        # Cleanup
        log_debug("Cleaning up...")

        # Clean up config menu first (to stop Qt timers)
        try:
            if 'config_menu' in locals() and config_menu:
                config_menu.cleanup()
        except Exception as e:
            log_error(f"Error cleaning up config menu: {e}")

        try:
            if audio_processor:
                audio_processor.stop()
                # Import _terminate only if we successfully imported audio modules
                try:
                    from sounddevice import _terminate
                    _terminate()
                except ImportError:
                    pass  # sounddevice wasn't imported successfully
        except Exception as e:
            log_error(f"Error stopping audio processor: {e}")

        try:
            if ctx:
                ctx.release()
        except Exception as e:
            log_error(f"Error releasing OpenGL context: {e}")

        try:
            pygame.quit()
        except Exception as e:
            log_error(f"Error quitting pygame: {e}")

        # Process any remaining Qt events
        try:
            if 'qt_app' in locals() and qt_app:
                qt_app.processEvents()
        except Exception as e:
            log_error(f"Error processing final Qt events: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_error(f"Error occurred: {e}")
        sys.exit(1)

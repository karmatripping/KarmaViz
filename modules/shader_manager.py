import os
import json
from pathlib import Path
import re
from typing import Dict, Callable, Optional
from .waveform_manager import WaveformManager
from .benchmark import benchmark
from .logging_config import get_logger

class ShaderManager:
    """ Manages shader-related operations for
    warp maps and waveforms"""    
    def __init__(self, waveforms_dir="waveforms", warp_maps_dir="warp_maps", warp_map_manager=None, threaded_shader_compiler=None):
        self.waveforms_dir = Path(waveforms_dir)
        self.warp_maps_dir = Path(warp_maps_dir)
        self.warp_map_shaders = {}
        self.warp_map_manager = warp_map_manager  # Reference to WarpMapManager for .glsl files

        # Use WaveformManager for waveform handling
        self.waveform_manager = WaveformManager(waveforms_dir)

        # Shader compilation management
        self.threaded_shader_compiler = threaded_shader_compiler
        self.logger = get_logger('shader_manager')
        self.pending_shader_requests: Dict[str, str] = {}  # Maps request_id to purpose

        self.discover_warp_maps()

    def discover_warp_maps(self):
        self.warp_map_shaders = {}
        if not self.warp_maps_dir.exists():
            return
        for root, dirs, files in os.walk(self.warp_maps_dir): # type:ignore
            for file in files:
                if file.endswith(".json"):
                    name = os.path.splitext(file)[0]
                    self.warp_map_shaders[name] = Path(root) / file

    def list_waveforms(self):
        return self.waveform_manager.list_waveforms()

    def list_warp_maps(self):
        return list(self.warp_map_shaders.keys())

    def load_waveform_code(self, name):
        return self.waveform_manager.load_shader_code(name)

    def load_warp_map_code(self, name):
        # First try to load from JSON files (new format)
        path = self.warp_map_shaders.get(name)
        if path and path.exists():
            with open(path, "r") as f:
                data = json.load(f)
                glsl_code = data.get("glsl_code", "")
                if glsl_code:
                    return glsl_code

        # If not found in JSON files, try to get from WarpMapManager (for .glsl files)
        if hasattr(self, 'warp_map_manager') and self.warp_map_manager:
            warp_map = self.warp_map_manager.get_warp_map(name)
            if warp_map and warp_map.glsl_code:
                return warp_map.glsl_code

        raise FileNotFoundError(f"Warp map descriptor '{name}' not found.")

    def build_full_fragment_shader(self, waveform_name, warp_map_name=None, warp_dispatcher_code=None, return_injection_lines=False):
        base_shader_path = Path(__file__).parent.parent / "shaders" / "shaders.py"
        with open(base_shader_path, "r") as f:
            base_shader_code = f.read()
        match = re.search(r'FRAGMENT_SHADER = """//glsl\n(.*?)"""', base_shader_code, re.DOTALL)
        if not match:
            raise RuntimeError("Could not find FRAGMENT_SHADER in shaders.py")
        fragment_shader = match.group(1)

        injection_lines = {}  # Maps source_name -> injection_line_number
        
        # First, inject waveform code (this comes first in the shader)
        waveform_code = self.load_waveform_code(waveform_name)

        # All waveforms now use compute_waveform_intensity_at_xy() - no conditional logic needed

        # Find where waveform code will be injected
        lines_before_waveform = fragment_shader[:fragment_shader.find("// WAVEFORM_RENDER_PLACEHOLDER")].count('\n')
        waveform_start_line = lines_before_waveform + 1
        injection_lines[f"waveform_{waveform_name}"] = waveform_start_line

        # Inject waveform code
        fragment_shader = fragment_shader.replace("// WAVEFORM_RENDER_PLACEHOLDER", waveform_code)

        # Now inject warp map code if provided (this comes after waveform)
        if warp_map_name:
            warp_map_code = self.load_warp_map_code(warp_map_name)
            
            # Find where warp map code will be injected (after waveform injection)
            lines_before_warp = fragment_shader[:fragment_shader.find("// WARP_MAP_FUNCTIONS_PLACEHOLDER")].count('\n')
            warp_map_start_line = lines_before_warp + 1
            injection_lines[warp_map_name] = warp_map_start_line
            
            fragment_shader = fragment_shader.replace("// WARP_MAP_FUNCTIONS_PLACEHOLDER", warp_map_code)

            # Also inject the dispatcher code that calls the warp map function
            dispatcher_code = "return get_pattern(pos, t);"
            fragment_shader = fragment_shader.replace("// WARP_DISPATCHER_PLACEHOLDER", dispatcher_code)
        if warp_dispatcher_code:
            fragment_shader = fragment_shader.replace("// WARP_DISPATCHER_PLACEHOLDER", warp_dispatcher_code)

        if return_injection_lines:
            self.logger.debug(f"Injection lines: {injection_lines}")
            return fragment_shader, injection_lines
        return fragment_shader

    def get_injection_lines(self, waveform_name, warp_map_name=None):
        """Get injection line numbers without compiling the shader"""
        _, injection_lines = self.build_full_fragment_shader(
            waveform_name=waveform_name,
            warp_map_name=warp_map_name,
            return_injection_lines=True
        )
        return injection_lines

    def compile_shader(
        self,
        ctx,
        waveform_name,
        warp_map_name=None,
        warp_dispatcher_code=None,
        vertex_shader=None,
    ):
        fragment_shader = self.build_full_fragment_shader(
            waveform_name, warp_map_name, warp_dispatcher_code
        )
        if vertex_shader is None:
            vertex_shader = """
            #version 330
            in vec2 in_position;
            in vec2 in_texcoord;
            out vec2 uv;
            void main() {
                gl_Position = vec4(in_position, 0.0, 1.0);
                uv = in_texcoord;
            }
            """
        try:
            program = ctx.program(
                vertex_shader=vertex_shader, fragment_shader=fragment_shader
            )
            return program
        except Exception as e:
            raise RuntimeError(
                f"Failed to compile shader (waveform: {waveform_name}, warp_map: {warp_map_name}): {e}"
            )

    @benchmark("process_shader_results")
    def process_shader_compilation_results(self, program_update_callback: Optional[Callable] = None) -> None:
        """Process completed shader compilation results from background threads.
        
        Args:
            program_update_callback: Optional callback to handle program updates.
                                   Should accept (purpose, old_program, new_program) parameters.
        """
        if not self.threaded_shader_compiler:
            return
            
        results = self.threaded_shader_compiler.process_completed_results()

        for result in results:
            purpose = self.pending_shader_requests.get(result.request_id, "unknown")

            if result.status.value == "completed":

                # Compile the OpenGL program on the main thread
                program = self.threaded_shader_compiler.compile_on_main_thread(result)

                if program:
                    # Call the callback to handle program updates
                    if program_update_callback:
                        program_update_callback(purpose, program)
                else:
                    self.logger.error(
                        f"Failed to compile OpenGL program for {purpose}"
                    )

            elif result.status.value == "failed":
                self.logger.error(
                    f"Failed {purpose} shader source preparation: {result.error_message}"
                )

                # If main shader compilation failed, keep using current program
                if purpose.startswith("main_shader"):
                    self.logger.warning(
                        f"Keeping current shader program due to source preparation failure"
                    )

            # Clean up tracking
            if result.request_id in self.pending_shader_requests:
                del self.pending_shader_requests[result.request_id]

    def compile_shader_async(
        self, 
        warp_maps: list, 
        current_waveform_name: str,
        purpose: str = "main_shader", 
        priority: int = 0
    ) -> str:
        """Compile shader asynchronously using background threads.

        Args:
            warp_maps: List of warp map names
            current_waveform_name: Name of the current waveform
            purpose: Purpose of the compilation (for tracking)
            priority: Priority of the request (higher = more urgent)

        Returns:
            Request ID for tracking the compilation
        """
        if not self.threaded_shader_compiler:
            raise RuntimeError("Threaded shader compiler not available")

        def shader_ready_callback(program):
            """Callback called when shader compilation is complete."""
            if program:
                self.logger.debug(
                    f"Async {purpose} compilation completed successfully"
                )
            else:
                self.logger.error(f"Async {purpose} compilation failed")

        request_id = self.threaded_shader_compiler.compile_async(
            warp_maps=warp_maps,
            shader_manager=self,
            current_waveform_name=current_waveform_name,
            callback=shader_ready_callback,
            priority=priority,
        )

        # Track the request
        self.pending_shader_requests[request_id] = purpose

        return request_id

    def get_shader_compilation_status(self) -> Dict[str, str]:
        """Get status of all pending shader compilations.

        Returns:
            Dictionary mapping request purposes to their status
        """
        if not self.threaded_shader_compiler:
            return {}
            
        status_info = {}
        for request_id, purpose in self.pending_shader_requests.items():
            status = self.threaded_shader_compiler.get_status(request_id)
            status_info[purpose] = status.value

        # Add queue information
        queue_size = self.threaded_shader_compiler.get_queue_size()
        active_count = self.threaded_shader_compiler.get_active_count()

        status_info["_queue_size"] = str(queue_size)
        status_info["_active_compilations"] = str(active_count)

        return status_info

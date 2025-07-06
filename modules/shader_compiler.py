"""
Dynamic Shader Compiler for KarmaViz

This module handles dynamic compilation of shaders with warp maps loaded from files.
It replaces the monolithic shader approach with a modular system.
"""

import moderngl
from typing import List, Dict, Optional, Tuple, Callable
from modules.warp_map_manager import WarpMapManager, WarpMapInfo
from modules.benchmark import benchmark
from modules.shader_error_parser import ShaderErrorParser, UserCodeSection
from modules.logging_config import get_logger
import threading
import queue
from dataclasses import dataclass
from enum import Enum
import pygame
import weakref

# Import shader constants
try:
    from shaders.shaders import VERTEX_SHADER
except ImportError:
    # Fallback vertex shader if import fails
    VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;
void main() {
    gl_Position = vec4(aPos, 1.0);
}
"""


class CompilationStatus(Enum):
    """Status of shader compilation."""

    PENDING = "pending"
    COMPILING = "compiling"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class CompilationRequest:
    """Request for shader compilation."""

    request_id: str
    warp_maps: List[str]
    shader_manager: Optional[object]
    current_waveform_name: str
    callback: Optional[Callable[[Optional[moderngl.Program]], None]]
    priority: int = 0  # Higher numbers = higher priority


@dataclass
class CompilationResult:
    """Result of shader compilation."""

    request_id: str
    program: Optional[moderngl.Program]
    status: CompilationStatus
    error_message: Optional[str] = None
    compilation_time: float = 0.0
    # For deferred compilation on main thread
    vertex_shader_source: Optional[str] = None
    fragment_shader_source: Optional[str] = None
    # Enhanced error information
    detailed_errors: Optional[List] = None  # List of ShaderError objects
    user_code_sections: Optional[List] = None  # List of UserCodeSection objects


class ThreadedShaderCompiler:
    """Thread-safe shader compiler with multithreaded validation and compilation.

    This version can operate in multiple modes:
    1. Independent contexts: Creates separate OpenGL contexts in worker threads
       to validate shaders in parallel, then compiles final programs in main thread
    2. Main context sharing: Attempts to use main context across threads (UNSAFE - can cause crashes)
    3. Fallback mode: Only prepares shader source in worker threads
    
    The independent context mode provides the best balance of performance and safety.
    """

    def __init__(
        self,
        ctx: moderngl.Context,
        warp_map_manager: WarpMapManager,
        max_workers: int = 2,
        enable_shared_contexts: bool = True,
    ):
        """Initialize the threaded shader compiler.

        Args:
            ctx: ModernGL context from main thread
            warp_map_manager: Warp map manager instance
            max_workers: Maximum number of worker threads
            enable_shared_contexts: If True, use independent contexts for validation
        """
        self.main_ctx = ctx
        self.warp_map_manager = warp_map_manager
        self.max_workers = max_workers
        self.enable_shared_contexts = enable_shared_contexts
        self.logger = get_logger('shader_compiler')

        # Thread-safe queues
        self.request_queue = queue.PriorityQueue()
        self.result_queue = queue.Queue()

        # Thread management
        self.workers: List[threading.Thread] = []
        self.shutdown_event = threading.Event()
        self.lock = threading.Lock()

        # Compilation tracking
        self.active_requests: Dict[str, CompilationRequest] = {}
        self.completed_results: Dict[str, CompilationResult] = {}

        # Start worker threads
        self._start_workers()

    def _start_workers(self) -> None:
        """Start worker threads for shader compilation."""
        compilation_type = "compilation" if self.enable_shared_contexts else "source preparation"
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_thread, 
                name=f"ShaderCompiler-{i}", 
                daemon=True,
                args=(i,)  # Pass worker ID for context creation
            )
            worker.start()
            self.workers.append(worker)
        self.logger.debug(f"Started {self.max_workers} shader {compilation_type} worker threads")

    def _worker_thread(self, worker_id: int) -> None:
        """Worker thread that processes shader compilation requests."""
        thread_ctx = None
        # Create independent context
        try:
            thread_ctx = self._create_shared_context(worker_id)
            if thread_ctx:
                self.logger.debug(f"Worker {worker_id}: Created independent OpenGL context")
            else:
                self.logger.debug(f"Worker {worker_id}: Failed to create independent context, falling back to source preparation only")
        except Exception as e:
            self.logger.debug(f"Worker {worker_id}: Error creating independent context: {e}, falling back to source preparation only")
            thread_ctx = None

        while not self.shutdown_event.is_set():
            request = None
            try:
                # Get next request with timeout
                try:
                    priority, request = self.request_queue.get(timeout=1.0) # type:ignore
                except queue.Empty:
                    continue

                # Update status
                with self.lock:
                    if request.request_id in self.active_requests:
                        self.active_requests[request.request_id] = request

                # Process request based on available context
                if thread_ctx and self.enable_shared_contexts:
                        # Use independent context (validation only)
                        result = self._compile_shader_in_thread(request, thread_ctx, worker_id)
                else:
                    # Fallback to source preparation only
                    result = self._prepare_shader_source(request)

                # Store result
                with self.lock:
                    self.completed_results[request.request_id] = result
                    if request.request_id in self.active_requests:
                        del self.active_requests[request.request_id]

                # Put result in queue for main thread
                self.result_queue.put(result)

            except Exception as e:
                self.logger.error(f"Error in shader compilation worker {worker_id}: {e}")

                # Create error result if we have a request
                if request:
                    error_result = CompilationResult(
                        request_id=request.request_id,
                        program=None,
                        status=CompilationStatus.FAILED,
                        error_message=str(e),
                        compilation_time=0.0
                    )

                    # Store error result
                    with self.lock:
                        self.completed_results[request.request_id] = error_result
                        if request.request_id in self.active_requests:
                            del self.active_requests[request.request_id]

                    # Put error result in queue
                    self.result_queue.put(error_result)

            finally:
                # ALWAYS call task_done() if we successfully got a request from the queue
                if request is not None:
                    self.request_queue.task_done()


        
        # Cleanup thread context (only for independent contexts)
        if thread_ctx:
            try:
                # ModernGL contexts are automatically cleaned up when they go out of scope
                self.logger.debug(f"🧹 Worker {worker_id}: Cleaning up independent context")
            except Exception as e:
                self.logger.debug(f"Worker {worker_id}: Error cleaning up context: {e}")

    def _create_shared_context(self, worker_id: int) -> Optional[moderngl.Context]:
        """Create a shared OpenGL context for a worker thread.
        
        Args:
            worker_id: ID of the worker thread
            
        Returns:
            ModernGL context for shader compilation in this thread, or None if failed
        """
        try:        
            self.logger.debug(f"Worker {worker_id}: Creating shared OpenGL context...")
            
            # For true context sharing, we need to create a context that shares resources
            # with the main context. ModernGL's standalone contexts are independent,
            # so we'll create a new context that can be used for compilation.
            
            try:
                # Create a new ModernGL context in this thread
                # Note: This creates an independent context, not a shared one
                # But it allows us to compile shaders in parallel
                shared_ctx = moderngl.create_context(standalone=True, require=330)
                
                if not shared_ctx:
                    self.logger.debug(f"Worker {worker_id}: Failed to create standalone context")
                    return None
                
                self.logger.debug(f"Worker {worker_id}: Created context - OpenGL {shared_ctx.version_code}")
                
                return shared_ctx
   
            except Exception as ctx_e:
                self.logger.debug(f"Worker {worker_id}: Error creating context: {ctx_e}")
                return None
                
        except Exception as e:
            self.logger.debug(f"Worker {worker_id}: Error in _create_shared_context: {e}")
            
            
            return None
    @benchmark("threaded_shader_compilation")
    def _compile_shader_in_thread(
        self, request: CompilationRequest, ctx: moderngl.Context, worker_id: int
    ) -> CompilationResult:
        """Compile shader program in worker thread using shared context.
        
        Args:
            request: Compilation request
            ctx: Shared ModernGL context for this thread
            worker_id: ID of the worker thread
            
        Returns:
            CompilationResult with compiled program or error information
        """
        import time
        start_time = time.perf_counter()
        
        try:
            self.logger.debug(f"[Worker {worker_id}] Compiling shader for request {request.request_id}")
            
            # Get vertex shader source
            from shaders.shaders import VERTEX_SHADER
            vertex_shader_source = VERTEX_SHADER
            
            # Build fragment shader source using shader manager
            if request.shader_manager:
                fragment_shader_source = (
                    request.shader_manager.build_full_fragment_shader(
                        waveform_name=request.current_waveform_name,
                        warp_map_name=(
                            request.warp_maps[0] if request.warp_maps else None
                        ),
                    )
                )
            else:
                # Fallback to basic shader
                from shaders.shaders import FRAGMENT_SHADER
                fragment_shader_source = FRAGMENT_SHADER
            
            # Compile the program using the shared context
            program = ctx.program(
                vertex_shader=vertex_shader_source,
                fragment_shader=fragment_shader_source,
            )
            
            end_time = time.perf_counter()
            compilation_time = end_time - start_time
            
            self.logger.debug(f"[Worker {worker_id}] Successfully compiled shader for {request.request_id} in {compilation_time*1000:.2f}ms")
            
            return CompilationResult(
                request_id=request.request_id,
                program=program,
                status=CompilationStatus.COMPLETED,
                compilation_time=compilation_time,
                vertex_shader_source=vertex_shader_source,
                fragment_shader_source=fragment_shader_source,
            )
            
        except Exception as e:
            end_time = time.perf_counter()
            compilation_time = end_time - start_time
            error_msg = str(e)
            self.logger.error(f"[Worker {worker_id}] Failed to compile shader for {request.request_id}: {error_msg}")
            
            return CompilationResult(
                request_id=request.request_id,
                program=None,
                status=CompilationStatus.FAILED,
                error_message=error_msg,
                compilation_time=compilation_time,
                vertex_shader_source=None,
                fragment_shader_source=None,
            )

    @benchmark("threaded_shader_source_prep")
    def _prepare_shader_source(self, request: CompilationRequest) -> CompilationResult:
        """Prepare shader source code without OpenGL compilation."""
        import time

        start_time = time.perf_counter()

        try:
            self.logger.debug(
                f"[ThreadedCompiler] Preparing shader source for request {request.request_id}"
            )

            # Get vertex shader source
            from shaders.shaders import VERTEX_SHADER

            vertex_shader_source = VERTEX_SHADER

            # Build fragment shader source using shader manager
            if request.shader_manager:
                fragment_shader_source = (
                    request.shader_manager.build_full_fragment_shader(
                        waveform_name=request.current_waveform_name,
                        warp_map_name=(
                            request.warp_maps[0] if request.warp_maps else None
                        ),
                    )
                )
            else:
                # Fallback to basic shader
                from shaders.shaders import FRAGMENT_SHADER

                fragment_shader_source = FRAGMENT_SHADER

            end_time = time.perf_counter()
            compilation_time = end_time - start_time

            self.logger.debug(
                f"[ThreadedCompiler] Prepared shader source for {request.request_id} in {compilation_time*1000:.2f}ms"
            )
            return CompilationResult(
                request_id=request.request_id,
                program=None,  # Will be compiled on main thread
                status=CompilationStatus.COMPLETED,
                compilation_time=compilation_time,
                vertex_shader_source=vertex_shader_source,
                fragment_shader_source=fragment_shader_source,
            )

        except Exception as e:
            end_time = time.perf_counter()
            compilation_time = end_time - start_time
            error_msg = str(e)
            self.logger.error(
                f"[ThreadedCompiler] Exception during source preparation for {request.request_id}: {error_msg}"
            )

            return CompilationResult(
                request_id=request.request_id,
                program=None,
                status=CompilationStatus.FAILED,
                error_message=error_msg,
                compilation_time=compilation_time,
            )

    def compile_async(
        self,
        warp_maps: List[str],
        shader_manager=None,
        current_waveform_name: str = "normal",
        callback: Optional[Callable[[Optional[moderngl.Program]], None]] = None,
        priority: int = 0,
    ) -> str:
        """Submit a shader source preparation request asynchronously.

        Args:
            warp_maps: List of warp map names
            shader_manager: Shader manager instance
            current_waveform_name: Current waveform name
            callback: Optional callback function called when compilation completes
            priority: Priority of the request (higher = more urgent)

        Returns:
            Request ID for tracking the compilation
        """
        import uuid

        request_id = str(uuid.uuid4())

        request = CompilationRequest(
            request_id=request_id,
            warp_maps=warp_maps,
            shader_manager=shader_manager,
            current_waveform_name=current_waveform_name,
            callback=callback,
            priority=priority,
        )

        with self.lock:
            self.active_requests[request_id] = request

        # Submit to queue (negative priority for max-heap behavior)
        self.request_queue.put((-priority, request))

        self.logger.debug(
            f"[ThreadedCompiler] Submitted shader source preparation request {request_id} with priority {priority}"
        )
        return request_id

    def get_result(self, request_id: str) -> Optional[CompilationResult]:
        """Get the result of a compilation request.

        Args:
            request_id: ID of the compilation request

        Returns:
            CompilationResult if available, None otherwise
        """
        with self.lock:
            return self.completed_results.get(request_id)

    def is_ready(self, request_id: str) -> bool:
        """Check if a compilation request is ready.

        Args:
            request_id: ID of the compilation request

        Returns:
            True if the compilation is complete
        """
        with self.lock:
            return request_id in self.completed_results

    def get_status(self, request_id: str) -> CompilationStatus:
        """Get the status of a compilation request.

        Args:
            request_id: ID of the compilation request

        Returns:
            Current status of the compilation
        """
        with self.lock:
            if request_id in self.completed_results:
                return self.completed_results[request_id].status
            elif request_id in self.active_requests:
                return CompilationStatus.COMPILING
            else:
                return CompilationStatus.PENDING

    def process_completed_results(self) -> List[CompilationResult]:
        """Process all completed results from the result queue.

        Returns:
            List of completed compilation results
        """
        results = []
        while True:
            try:
                result = self.result_queue.get_nowait()
                results.append(result)
            except queue.Empty:
                # No more results available - this is normal
                break
        return results

    @benchmark("compile_on_main_thread")
    def compile_on_main_thread(
        self, result: CompilationResult
    ) -> Optional[moderngl.Program]:
        """Compile the final program on main thread using validated shader source.

        Args:
            result: CompilationResult with validated shader source from worker thread

        Returns:
            Compiled ModernGL program or None if compilation failed
        """        
        # For independent contexts or fallback mode, always compile on main thread
        if not result.vertex_shader_source or not result.fragment_shader_source:
            self.logger.error(f"[ThreadedCompiler] No shader source available for {result.request_id}")
            return None

        try:
            import time
            start_time = time.perf_counter()
            
            program = self.main_ctx.program(
                vertex_shader=result.vertex_shader_source,
                fragment_shader=result.fragment_shader_source,
            )
            
            end_time = time.perf_counter()
            main_thread_time = end_time - start_time
            
            if self.enable_shared_contexts:
                self.logger.debug(
                    f"[ThreadedCompiler] Compiled validated program on main thread for {result.request_id} "
                    f"(validation: {result.compilation_time*1000:.2f}ms, main: {main_thread_time*1000:.2f}ms)"
                )
            else:
                self.logger.debug(
                    f"[ThreadedCompiler] Successfully compiled program on main thread for {result.request_id} "
                    f"({main_thread_time*1000:.2f}ms)"
                )
            return program
        except Exception as e:
            self.logger.error(
                f"[ThreadedCompiler] Failed to compile program on main thread for {result.request_id}: {e}"
            )
            return None

    def get_queue_size(self) -> int:
        """Get the number of pending compilation requests."""
        return self.request_queue.qsize()

    def get_active_count(self) -> int:
        """Get the number of active compilation requests."""
        with self.lock:
            return len(self.active_requests)

    def is_using_shared_contexts(self) -> bool:
        """Check if the compiler is using shared contexts for true multithreaded compilation."""
        return self.enable_shared_contexts

    def shutdown(self) -> None:
        """Shutdown the threaded compiler and wait for workers to finish."""
        self.logger.debug("Shutting down threaded shader compiler...")
        self.shutdown_event.set()

        # Wait for all workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)
            if worker.is_alive():
                self.logger.warning(f"Warning: Worker {worker.name} did not shutdown cleanly")

        self.logger.debug("Threaded shader compiler shutdown complete")

    def __del__(self):
        """Ensure cleanup on deletion."""
        if hasattr(self, "shutdown_event") and not self.shutdown_event.is_set():
            self.shutdown()


class ShaderCompiler:
    """Compiles stackable post-processing shaders for warp maps"""

    def __init__(self, ctx: moderngl.Context, warp_map_manager: WarpMapManager):
        self.ctx = ctx
        self.warp_map_manager = warp_map_manager
        self.compiled_programs: Dict[str, moderngl.Program] = {}
        self.logger = get_logger('shader_compiler')

        # Stackable shader templates for post-processing
        self.vertex_shader_template = self._get_vertex_shader_template()
        self.warp_fragment_template = self._get_warp_fragment_template()
        
        # Store latest GLSL errors for user code sections
        self.latest_glsl_errors = {}
        
        # Store injection lines for error translation
        self.current_injection_lines = {}
        
        # Shader manager reference (set later)
        self.shader_manager = None
        
        # Current waveform name for injection line calculation
        self.current_waveform_name = 'normal'

    def set_shader_manager(self, shader_manager):
        """Set the shader manager reference"""
        self.shader_manager = shader_manager

    def _get_vertex_shader_template(self) -> str:
        """Get the vertex shader template"""
        return """//glsl
#version 330
in vec2 in_position;
in vec2 in_texcoord;
out vec2 v_texcoord;

void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    v_texcoord = in_texcoord;
}"""

    def _get_warp_fragment_template(self) -> str:
        """Get the fragment shader template for warp effects"""
        return """//glsl
#version 330

uniform sampler2D u_texture;
uniform float time;
uniform vec2 resolution;
uniform float warp_intensity;

in vec2 v_texcoord;
out vec4 fragColor;

// WARP_MAP_PLACEHOLDER - This will be replaced with actual warp map code

void main() {
    vec2 uv = v_texcoord;
    
    // Apply warp map transformation
    vec2 warped_uv = apply_warp_map(uv, time);
    
    // Sample the texture with warped coordinates
    vec4 color = texture(u_texture, warped_uv);
    
    fragColor = color;
}"""

    def get_warp_maps_for_stacking(
        self, warp_map_names: List[str]
    ) -> List[WarpMapInfo]:
        """Get warp map info objects for stacking multiple effects"""
        warp_maps = []
        for name in warp_map_names:
            warp_map = self.warp_map_manager.get_warp_map(name)
            if warp_map:
                warp_maps.append(warp_map)
            else:
                self.logger.warning(f"Warning: Warp map '{name}' not found")
        return warp_maps

    def build_stacked_warp_shader(
        self, warp_maps: List[WarpMapInfo]
    ) -> Tuple[str, str]:
        """Build a shader that applies multiple warp maps in sequence"""
        vertex = self.vertex_shader_template

        # Start with base fragment shader
        fragment = self.warp_fragment_template

        if not warp_maps:
            # No warp maps - just pass through
            fragment = fragment.replace(
                "// WARP_MAP_PLACEHOLDER - This will be replaced with actual warp map code",
                """
vec2 apply_warp_map(vec2 uv, float time) {
    return uv; // No warping
}""",
            )
        else:
            # Build combined warp function
            warp_functions = []
            for i, warp_map in enumerate(warp_maps):
                func_name = f"warp_map_{i}"
                warp_functions.append(
                    f"""
// Warp map: {warp_map.name}
vec2 {func_name}(vec2 uv, float time) {{
{warp_map.code}
}}"""
                )

            # Build the main apply_warp_map function that chains all warp maps
            apply_function = """
vec2 apply_warp_map(vec2 uv, float time) {
    vec2 result = uv;
"""
            for i in range(len(warp_maps)):
                apply_function += f"    result = warp_map_{i}(result, time);\n"

            apply_function += """    return result;
}"""

            # Combine all functions
            all_warp_code = "\n".join(warp_functions) + "\n" + apply_function

            fragment = fragment.replace(
                "// WARP_MAP_PLACEHOLDER - This will be replaced with actual warp map code",
                all_warp_code,
            )

        return vertex, fragment

    @benchmark("shader_compile_main")
    def compile_main_shader_with_warp(
        self, warp_maps: List[str], shader_manager=None, current_waveform_name="normal"
    ) -> Optional[moderngl.Program]:
        """Compile the main shader with integrated warp map functionality"""
        try:
            from shaders.shaders import VERTEX_SHADER

            # Use shader manager to build full fragment shader with waveform support
            if shader_manager:
                # Build fragment shader with waveform and warp map support
                fragment_shader, injection_lines = shader_manager.build_full_fragment_shader(
                    waveform_name=current_waveform_name,
                    warp_map_name=warp_maps[0] if warp_maps else None,
                    return_injection_lines=True,
                )
                # Store injection lines for error translation
                self.current_injection_lines = injection_lines
                # Store current waveform name
                self.current_waveform_name = current_waveform_name
            else:
                # Fallback to basic shader with minimal waveform support
                from shaders.shaders import FRAGMENT_SHADER
                
                # Clear injection lines for fallback case
                self.current_injection_lines = {}

                # Use basic fragment shader without waveform support
                fragment_shader = FRAGMENT_SHADER

            # Compile the program
            program = self.ctx.program(
                vertex_shader=VERTEX_SHADER,
                fragment_shader=fragment_shader
            )

            # Clear errors on successful compilation
            if warp_maps:
                for warp_name in warp_maps:
                    self.clear_errors_for_warp(warp_name)
            if current_waveform_name:
                self.clear_errors_for_waveform(current_waveform_name)

            return program

        except Exception as e:
            # Check if this is a GLSL compilation error
            error_str = str(e)
            if "GLSL Compiler failed" in error_str:
                # Extract and parse GLSL errors
                self._handle_glsl_compilation_error(error_str, warp_maps, current_waveform_name)
            else:
                self.logger.error(f"Error compiling main shader with warp maps {warp_maps}: {e}")
                
                
            return None

    def _handle_glsl_compilation_error(self, error_str: str, warp_maps: List[str] = None, waveform_name: str = None):
        """Handle GLSL compilation errors and extract user-relevant information"""
        self.logger.error(f"GLSL Compilation Error: {error_str}")
        
        # Parse the error to extract line numbers and messages
        lines = error_str.split('\n')
        for line in lines:
            if '(' in line and ')' in line and 'error' in line.lower():
                # Extract line number and error message
                # Format: "0(274) : error C0000: syntax error, unexpected '}', expecting ',' or ';' at token "}""
                try:
                    # Find the line number in parentheses
                    start = line.find('(')
                    end = line.find(')')
                    if start != -1 and end != -1:
                        line_num = int(line[start+1:end])
                        
                        # Extract error message after the colon
                        colon_pos = line.find(':', end)
                        if colon_pos != -1:
                            error_msg = line[colon_pos+1:].strip()
                            
                            # Store errors with original line numbers - editors will handle display
                            if warp_maps:
                                for warp_name in warp_maps:
                                    if warp_name not in self.latest_glsl_errors:
                                        self.latest_glsl_errors[warp_name] = []
                                    self.latest_glsl_errors[warp_name].append((line_num, error_msg))
                            
                            if waveform_name:
                                key = f"waveform_{waveform_name}"
                                if key not in self.latest_glsl_errors:
                                    self.latest_glsl_errors[key] = []
                                self.latest_glsl_errors[key].append((line_num, error_msg))
                                
                except (ValueError, IndexError):
                    continue

    def get_latest_errors_for_warp(self, warp_name: str) -> List[Tuple[int, str]]:
        """Get the latest GLSL errors for a specific warp map"""
        return self.latest_glsl_errors.get(warp_name, [])

    def clear_errors_for_warp(self, warp_name: str):
        """Clear errors for a specific warp map"""
        if warp_name in self.latest_glsl_errors:
            del self.latest_glsl_errors[warp_name]

    def get_latest_errors_for_waveform(self, waveform_name: str) -> List[Tuple[int, str]]:
        """Get the latest GLSL errors for a specific waveform"""
        return self.latest_glsl_errors.get(f"waveform_{waveform_name}", [])

    def clear_errors_for_waveform(self, waveform_name: str):
        """Clear errors for a specific waveform"""
        key = f"waveform_{waveform_name}"
        if key in self.latest_glsl_errors:
            del self.latest_glsl_errors[key]

    def get_injection_line_for_warp(self, warp_name: str) -> int:
        """Get the injection line number for a warp map"""
        # Try to get from current injection lines first
        if warp_name in self.current_injection_lines:
            return self.current_injection_lines[warp_name]
        
        # If not available, calculate it using shader manager
        if self.shader_manager:
            try:
                injection_lines = self.shader_manager.get_injection_lines(self.current_waveform_name, warp_name)
                return injection_lines.get(warp_name, 1)
            except Exception as e:
                self.logger.debug(f"Error getting injection line for warp {warp_name}: {e}")
                pass
        return 1

    def get_injection_line_for_waveform(self, waveform_name: str) -> int:
        """Get the injection line number for a waveform"""
        key = f"waveform_{waveform_name}"
        # Try to get from current injection lines first
        if key in self.current_injection_lines:
            return self.current_injection_lines[key]
        
        # If not available, calculate it using shader manager
        if self.shader_manager:
            try:
                injection_lines = self.shader_manager.get_injection_lines(waveform_name)
                return injection_lines.get(key, 1)
            except Exception as e:
                self.logger.debug(f"Error getting injection line for waveform {waveform_name}: {e}")
                pass
        return 1

    @benchmark("shader_compile_spectrogram")
    def compile_spectrogram_shader(self) -> Optional[moderngl.Program]:
        """Compile the spectrogram shader"""
        try:
            vertex, fragment = self.spectrogram_shaders
            return self.ctx.program(vertex_shader=vertex, fragment_shader=fragment)
        except Exception as e:
            self.logger.error(f"Error compiling spectrogram shader: {e}")
            return None

    @property
    def spectrogram_shaders(self) -> Tuple[str, str]:
        """Get spectrogram vertex and fragment shaders"""
        from shaders.shaders import (
            SPECTROGRAM_VERTEX_SHADER,
            SPECTROGRAM_FRAGMENT_SHADER,
        )

        return SPECTROGRAM_VERTEX_SHADER, SPECTROGRAM_FRAGMENT_SHADER

    def compile_warp_stack(
        self, warp_map_names: List[str]
    ) -> Optional[moderngl.Program]:
        """Compile a shader that applies multiple warp maps in sequence"""
        try:
            # Get warp map info objects
            warp_maps = self.get_warp_maps_for_stacking(warp_map_names)

            if not warp_maps:
                self.logger.warning("No valid warp maps found for stacking")
                return None

            # Build stacked shader
            vertex, fragment = self.build_stacked_warp_shader(warp_maps)

            # Compile program
            program = self.ctx.program(vertex_shader=vertex, fragment_shader=fragment)

            # Cache the compiled program
            cache_key = "|".join(warp_map_names)
            self.compiled_programs[cache_key] = program

            self.logger.debug(
                f"Compiled stacked warp shader with {len(warp_maps)} effects: {', '.join([w.name for w in warp_maps])}"
            )
            return program

        except Exception as e:
            self.logger.error(f"Error compiling stacked warp shader: {e}")
            

            
            return None

    def get_cached_program(
        self, warp_map_names: List[str]
    ) -> Optional[moderngl.Program]:
        """Get a cached compiled program for the given warp map combination"""
        cache_key = "|".join(warp_map_names)
        return self.compiled_programs.get(cache_key)

    def clear_cache(self):
        """Clear all cached compiled programs"""
        for program in self.compiled_programs.values():
            try:
                program.release()
            except:
                pass  # Ignore errors during cleanup
        self.compiled_programs.clear()
        self.logger.debug("Cleared shader program cache")
    
    def test_waveform_code(self, waveform_code: str, waveform_name: str = "test") -> Tuple[bool, List[Tuple[int, str]]]:
        """
        Test waveform code for compilation errors
        
        Returns:
            Tuple of (success, list of (line_number, error_message))
        """
        try:
            # Create a test shader with the waveform code
            fragment_shader = self._build_fragment_shader_with_waveform(waveform_code, waveform_name)
            
            # Try to compile it
            test_program = self.ctx.program(
                vertex_shader=VERTEX_SHADER,
                fragment_shader=fragment_shader
            )
            
            # If we get here, compilation succeeded
            test_program.release()
            return True, []
            
        except Exception as e:
            # Parse the error and extract line numbers
            error_parser = ShaderErrorParser()
            errors = error_parser.parse_shader_errors(str(e))
            
            # Find where the user's waveform code appears in the full shader
            user_section = error_parser._find_user_code_section(waveform_code, fragment_shader, 'waveform')
            
            if user_section:
                # Translate errors to user code line numbers
                translated = error_parser.translate_errors_to_user_code(errors, [user_section])
                return False, translated.get('waveform', [])
            else:
                # Fallback: return generic error
                return False, [(1, f"Compilation error: {str(e)}")]
    
    def test_warp_map_code(self, warp_code: str, warp_name: str = "test") -> Tuple[bool, List[Tuple[int, str]]]:
        """
        Test warp map code for compilation errors
        
        Returns:
            Tuple of (success, list of (line_number, error_message))
        """
        try:
            # Create a test warp map info
            from modules.warp_map_manager import WarpMapInfo
            test_warp = WarpMapInfo(
                name=warp_name,
                category="test",
                description="Test warp map",
                glsl_code=warp_code,
                complexity="Simple",
                author="Test",
                version="1.0"
            )
            
            # Create a test shader with the warp map
            fragment_shader = self._build_fragment_shader_with_warp_maps([test_warp], "normal")
            
            # Try to compile it
            test_program = self.ctx.program(
                vertex_shader=VERTEX_SHADER,
                fragment_shader=fragment_shader
            )
            
            # If we get here, compilation succeeded
            test_program.release()
            return True, []
            
        except Exception as e:
            # Parse the error and extract line numbers
            error_parser = ShaderErrorParser()
            errors = error_parser.parse_shader_errors(str(e))
            
            # Find where the user's warp code appears in the full shader
            user_section = error_parser._find_user_code_section(warp_code, fragment_shader, 'warp_map')
            
            if user_section:
                # Translate errors to user code line numbers
                translated = error_parser.translate_errors_to_user_code(errors, [user_section])
                return False, translated.get('warp_map', [])
            else:
                # Fallback: return generic error
                return False, [(1, f"Compilation error: {str(e)}")]
    
    def _build_fragment_shader_with_waveform(self, waveform_code: str, waveform_name: str) -> str:
        """Build a complete fragment shader with the given waveform code for testing"""
        # Get the base fragment shader template
        if hasattr(self, 'shader_manager') and self.shader_manager:
            base_shader = self.shader_manager.get_fragment_shader()
        else:
            # Fallback minimal shader template
            base_shader = """
#version 330 core
out vec4 fragColor;
uniform float u_time;
uniform vec2 u_resolution;
uniform float u_audio_data[512];

// User waveform code will be inserted here
{waveform_code}

void main() {{
    vec2 pos = gl_FragCoord.xy / u_resolution.xy;
    vec3 color = render_waveform(pos, u_time, u_audio_data);
    fragColor = vec4(color, 1.0);
}}
"""
        
        # Insert the waveform code
        return base_shader.replace("{waveform_code}", waveform_code)

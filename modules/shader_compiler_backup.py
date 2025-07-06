"""
Dynamic Shader Compiler for KarmaViz

This module handles dynamic compilation of shaders with warp maps loaded from files.
It replaces the monolithic shader approach with a modular system.
"""

import moderngl
from typing import List, Dict, Optional, Tuple, Callable, Callable
from modules.warp_map_manager import WarpMapManager, WarpMapInfo
from modules.benchmark import benchmark
import threading
import queue
from dataclasses import dataclass
from enum import Enum
import threading
import queue
from dataclasses import dataclass
from enum import Enum


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


class ThreadedShaderCompiler:
    """Thread-safe shader compiler that runs compilation in background threads."""
    
    def __init__(self, ctx: moderngl.Context, warp_map_manager: WarpMapManager, max_workers: int = 2):
        """Initialize the threaded shader compiler.
        
        Args:
            ctx: ModernGL context
            warp_map_manager: Warp map manager instance
            max_workers: Maximum number of worker threads
        """
        self.ctx = ctx
        self.warp_map_manager = warp_map_manager
        self.max_workers = max_workers
        
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
        
        # Create the base shader compiler for actual compilation work
        self.base_compiler = ShaderCompiler(ctx, warp_map_manager)
        
        # Start worker threads
        self._start_workers()
    
    def _start_workers(self) -> None:
        """Start worker threads for shader compilation."""
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_thread,
                name=f"ShaderCompiler-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        print(f"Started {self.max_workers} shader compilation worker threads")
    
    def _worker_thread(self) -> None:
        """Worker thread that processes compilation requests."""
        while not self.shutdown_event.is_set():
            request = None
            try:
                # Get next request with timeout
                try:
                    priority, request = self.request_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # Update status
                with self.lock:
                    if request.request_id in self.active_requests:
                        self.active_requests[request.request_id] = request

                # Perform compilation
                result = self._compile_shader(request)

                # Store result
                with self.lock:
                    self.completed_results[request.request_id] = result
                    if request.request_id in self.active_requests:
                        del self.active_requests[request.request_id]

                # Put result in queue for main thread
                self.result_queue.put(result)

                # Call callback if provided
                if request.callback:
                    try:
                        request.callback(result.program)
                    except Exception as e:
                        print(f"Error in shader compilation callback: {e}")

            except Exception as e:
                print(f"Error in shader compilation worker: {e}")

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


    
    @benchmark("threaded_shader_compile")
    def _compile_shader(self, request: CompilationRequest) -> CompilationResult:
        """Compile shader using the base compiler."""
        import time
        start_time = time.perf_counter()
        
        try:
            program = self.base_compiler.compile_main_shader_with_warp(
                request.warp_maps,
                request.shader_manager,
                request.current_waveform_name
            )
            
            end_time = time.perf_counter()
            compilation_time = end_time - start_time
            
            if program:
                print(f"[ThreadedCompiler] Completed compilation for {request.request_id} in {compilation_time*1000:.2f}ms")
                return CompilationResult(
                    request_id=request.request_id,
                    program=program,
                    status=CompilationStatus.COMPLETED,
                    compilation_time=compilation_time
                )
            else:
                print(f"[ThreadedCompiler] Failed compilation for {request.request_id}")
                return CompilationResult(
                    request_id=request.request_id,
                    program=None,
                    status=CompilationStatus.FAILED,
                    error_message="Compilation returned None",
                    compilation_time=compilation_time
                )
                
        except Exception as e:
            end_time = time.perf_counter()
            compilation_time = end_time - start_time
            error_msg = str(e)
            print(f"[ThreadedCompiler] Exception during compilation for {request.request_id}: {error_msg}")
            
            return CompilationResult(
                request_id=request.request_id,
                program=None,
                status=CompilationStatus.FAILED,
                error_message=error_msg,
                compilation_time=compilation_time
            )
    
    def compile_async(
        self,
        warp_maps: List[str],
        shader_manager=None,
        current_waveform_name: str = "normal",
        callback: Optional[Callable[[Optional[moderngl.Program]], None]] = None,
        priority: int = 0
    ) -> str:
        """Submit a shader compilation request asynchronously.
        
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
            priority=priority
        )
        
        with self.lock:
            self.active_requests[request_id] = request
        
        # Submit to queue (negative priority for max-heap behavior)
        self.request_queue.put((-priority, request))
        
        print(f"[ThreadedCompiler] Submitted compilation request {request_id} with priority {priority}")
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
                break
        return results
    
    def get_queue_size(self) -> int:
        """Get the number of pending compilation requests."""
        return self.request_queue.qsize()
    
    def get_active_count(self) -> int:
        """Get the number of active compilation requests."""
        with self.lock:
            return len(self.active_requests)
    
    def shutdown(self) -> None:
        """Shutdown the threaded compiler and wait for workers to finish."""
        print("Shutting down threaded shader compiler...")
        self.shutdown_event.set()
        
        # Wait for all workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)
            if worker.is_alive():
                print(f"Warning: Worker {worker.name} did not shutdown cleanly")
        
        print("Threaded shader compiler shutdown complete")
    
    def __del__(self):
        """Ensure cleanup on deletion."""
        if hasattr(self, 'shutdown_event') and not self.shutdown_event.is_set():
            self.shutdown()


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


class ThreadedShaderCompiler:
    """Thread-safe shader compiler that runs compilation in background threads."""
    
    def __init__(self, ctx: moderngl.Context, warp_map_manager: WarpMapManager, max_workers: int = 2):
        """Initialize the threaded shader compiler.
        
        Args:
            ctx: ModernGL context
            warp_map_manager: Warp map manager instance
            max_workers: Maximum number of worker threads
        """
        self.ctx = ctx
        self.warp_map_manager = warp_map_manager
        self.max_workers = max_workers
        
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
        
        # Create the base shader compiler for actual compilation work
        self.base_compiler = ShaderCompiler(ctx, warp_map_manager)
        
        # Start worker threads
        self._start_workers()
    
    def _start_workers(self) -> None:
        """Start worker threads for shader compilation."""
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_thread,
                name=f"ShaderCompiler-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        print(f"Started {self.max_workers} shader compilation worker threads")
    
    def _worker_thread(self) -> None:
        """Worker thread that processes compilation requests."""
        while not self.shutdown_event.is_set():
            request = None
            try:
                # Get next request with timeout
                try:
                    priority, request = self.request_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # Update status
                with self.lock:
                    if request.request_id in self.active_requests:
                        self.active_requests[request.request_id] = request

                # Perform compilation
                result = self._compile_shader(request)

                # Store result
                with self.lock:
                    self.completed_results[request.request_id] = result
                    if request.request_id in self.active_requests:
                        del self.active_requests[request.request_id]

                # Put result in queue for main thread
                self.result_queue.put(result)

                # Call callback if provided
                if request.callback:
                    try:
                        request.callback(result.program)
                    except Exception as e:
                        print(f"Error in shader compilation callback: {e}")

            except Exception as e:
                print(f"Error in shader compilation worker: {e}")

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


    
    @benchmark("threaded_shader_compile")
    def _compile_shader(self, request: CompilationRequest) -> CompilationResult:
        """Compile shader using the base compiler."""
        import time
        start_time = time.perf_counter()
        
        try:
            print(f"[ThreadedCompiler] Starting compilation for request {request.request_id}")
            program = self.base_compiler.compile_main_shader_with_warp(
                request.warp_maps,
                request.shader_manager,
                request.current_waveform_name
            )
            
            end_time = time.perf_counter()
            compilation_time = end_time - start_time
            
            if program:
                print(f"[ThreadedCompiler] Completed compilation for {request.request_id} in {compilation_time*1000:.2f}ms")
                return CompilationResult(
                    request_id=request.request_id,
                    program=program,
                    status=CompilationStatus.COMPLETED,
                    compilation_time=compilation_time
                )
            else:
                print(f"[ThreadedCompiler] Failed compilation for {request.request_id}")
                return CompilationResult(
                    request_id=request.request_id,
                    program=None,
                    status=CompilationStatus.FAILED,
                    error_message="Compilation returned None",
                    compilation_time=compilation_time
                )
                
        except Exception as e:
            end_time = time.perf_counter()
            compilation_time = end_time - start_time
            error_msg = str(e)
            print(f"[ThreadedCompiler] Exception during compilation for {request.request_id}: {error_msg}")
            
            return CompilationResult(
                request_id=request.request_id,
                program=None,
                status=CompilationStatus.FAILED,
                error_message=error_msg,
                compilation_time=compilation_time
            )
    
    def compile_async(
        self,
        warp_maps: List[str],
        shader_manager=None,
        current_waveform_name: str = "normal",
        callback: Optional[Callable[[Optional[moderngl.Program]], None]] = None,
        priority: int = 0
    ) -> str:
        """Submit a shader compilation request asynchronously.
        
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
            priority=priority
        )
        
        with self.lock:
            self.active_requests[request_id] = request
        
        # Submit to queue (negative priority for max-heap behavior)
        self.request_queue.put((-priority, request))
        
        print(f"[ThreadedCompiler] Submitted compilation request {request_id} with priority {priority}")
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
                break
        return results
    
    def get_queue_size(self) -> int:
        """Get the number of pending compilation requests."""
        return self.request_queue.qsize()
    
    def get_active_count(self) -> int:
        """Get the number of active compilation requests."""
        with self.lock:
            return len(self.active_requests)
    
    def shutdown(self) -> None:
        """Shutdown the threaded compiler and wait for workers to finish."""
        print("Shutting down threaded shader compiler...")
        self.shutdown_event.set()
        
        # Wait for all workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)
            if worker.is_alive():
                print(f"Warning: Worker {worker.name} did not shutdown cleanly")
        
        print("Threaded shader compiler shutdown complete")
    
    def __del__(self):
        """Ensure cleanup on deletion."""
        if hasattr(self, 'shutdown_event') and not self.shutdown_event.is_set():
            self.shutdown()


class ShaderCompiler:
    """Compiles stackable post-processing shaders for warp maps"""

    def __init__(self, ctx: moderngl.Context, warp_map_manager: WarpMapManager):
        self.ctx = ctx
        self.warp_map_manager = warp_map_manager
        self.compiled_programs: Dict[str, moderngl.Program] = {}

        # Stackable shader templates for post-processing
        self.vertex_shader_template = self._get_vertex_shader_template()
        self.warp_fragment_template = self._get_warp_fragment_template()

    def _get_vertex_shader_template(self) -> str:
        """Get the vertex shader template"""
        return """//glsl
#version 330
in vec2 in_position;
in vec2 in_texcoord;
out vec2 uv;
void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    uv = in_texcoord;
}"""

    def _get_warp_fragment_template(self) -> str:
        """Get the simple warp post-processing fragment shader template"""
        return """//glsl
#version 330
uniform sampler2D input_texture;  // The rendered frame from main visualizer
uniform float time;
uniform float animation_speed;
uniform float warp_intensity;    // Control intensity of warp effect
uniform int active_warp_map;     // Which warp map to use (0-based index)
uniform int warp_blend_mode;     // Blending mode: 0=Replace, 1=Add, 2=Multiply, 3=Screen, 4=Overlay


in vec2 uv;
out vec4 fragColor;

// WARP_MAP_FUNCTIONS_PLACEHOLDER

vec2 apply_warp(vec2 pos, float t, int warp_index) {
    // PATTERN_DISPATCHER_PLACEHOLDER
    return vec2(0.0); // Default: no warp offset
}

// Blending functions
vec3 blend_add(vec3 base, vec3 blend) {
    return base + blend;
}

vec3 blend_multiply(vec3 base, vec3 blend) {
    return base * blend;
}

vec3 blend_screen(vec3 base, vec3 blend) {
    return 1.0 - (1.0 - base) * (1.0 - blend);
}

vec3 blend_overlay(vec3 base, vec3 blend) {
    return mix(
        2.0 * base * blend,
        1.0 - 2.0 * (1.0 - base) * (1.0 - blend),print(f"[DEBUG] Generated functions result length: {len(result)}")
        step(0.5, base)
    );
}

vec4 apply_blend_mode(vec4 original, vec4 warped, int blend_mode, float intensity) {
    if (blend_mode == 0) {
        // Replace mode - just use warped with intensity
        return mix(original, warped, intensity);
    } else if (blend_mode == 1) {
        // Additive mode
        vec3 blended = blend_add(original.rgb, warped.rgb * intensity);
        return vec4(blended, original.a);
    } else if (blend_mode == 2) {
        // Multiply mode
        vec3 blended = mix(original.rgb, blend_multiply(original.rgb, warped.rgb), intensity);
        return vec4(blended, original.a);
    } else if (blend_mode == 3) {
        // Screen mode
        vec3 blended = mix(original.rgb, blend_screen(original.rgb, warped.rgb), intensity);
        return vec4(blended, original.a);
    } else if (blend_mode == 4) {
        // Overlay mode
        vec3 blended = mix(original.rgb, blend_overlay(original.rgb, warped.rgb), intensity);
        return vec4(blended, original.a);
    }

    // Fallback to replace mode
    return mix(original, warped, intensity);
}

void main() {
    vec2 pos = uv;

    float t = time * animation_speed;

    // Sample the original texture

    vec4 original_color = texture(input_texture, pos);

    // Apply the selected warp map
    vec2 warp_offset = apply_warp(pos, t, active_warp_map);

    // Amplify the warp effect
    // The base warp maps are designed to be subtle, so we amplify them
    vec2 amplified_offset = warp_offset * 5.0;
    vec2 warped_pos = pos + amplified_offset;

    // Ensure coordinates stay in valid range with wrapping
    warped_pos = fract(warped_pos);

    // Sample the warped texture
    vec4 warped_color = texture(input_texture, warped_pos);

    // Apply blending mode
    vec4 final_color = apply_blend_mode(original_color, warped_color, warp_blend_mode, warp_intensity);

    fragColor = final_color;
}"""

    def _get_spectrogram_shaders(self) -> Tuple[str, str]:
        """Get the spectrogram vertex and fragment shaders"""
        vertex = """//glsl
#version 330
in vec2 in_position;
in vec2 in_texcoord;
out vec2 uv;
void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    uv = in_texcoord;
}"""

        fragment = """//glsl
#version 330
uniform sampler2D frequency_data;
uniform float opacity;
in vec2 uv;
out vec4 fragColor;

void main() {
    float center = 0.5;
    float dist = abs(uv.y - center);

    float freq;
    if (uv.x < 0.5) {
        freq = texture(frequency_data, vec2(uv.x * 2.0, 0.5)).r;
    } else {
        freq = texture(frequency_data, vec2((1.0 - uv.x) * 2.0, 0.5)).r;
    }

    float height = freq * 0.4;
    float intensity = 1.0 - smoothstep(0.0, height, dist);

    vec3 color = mix(vec3(0.0, 0.5, 1.0), vec3(1.0, 0.5, 0.0), freq);
    fragColor = vec4(color * intensity, intensity * opacity);
}"""

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
                warp_map_name = warp_maps[0] if warp_maps else None
                fragment_shader = shader_manager.build_full_fragment_shader(
                    waveform_name=current_waveform_name, warp_map_name=warp_map_name
                )
            else:
                # Fallback to basic shader with minimal waveform support
                from shaders.shaders import FRAGMENT_SHADER

                # No longer need basic waveform function - all waveforms use XY approach
                basic_waveform_function = """
// All waveforms now use compute_waveform_intensity_at_xy()
float compute_waveform_intensity_at_xy(float x_coord, float y_coord) {
    return 0.0; // Placeholder - no waveform data
}
"""

                # Generate warp map functions (empty if no warp maps)
                if warp_maps:
                    warp_functions = basic_waveform_function + "\n" + self._generate_warp_functions(warp_maps)
                    warp_dispatcher = self._generate_warp_dispatcher(warp_maps)
                else:
                    # No warp maps - provide empty functions
                    warp_functions = basic_waveform_function + "\n// No warp maps active"
                    warp_dispatcher = "return vec2(0.0); // No warp maps active"

                # Replace placeholders in main fragment shader
                fragment_shader = FRAGMENT_SHADER
                fragment_shader = fragment_shader.replace(
                    "// WARP_MAP_FUNCTIONS_PLACEHOLDER", warp_functions
                )
                fragment_shader = fragment_shader.replace(
                    "// WARP_DISPATCHER_PLACEHOLDER", warp_dispatcher
                )

            # Compile the program
            program = self.ctx.program(
                vertex_shader=VERTEX_SHADER,
                fragment_shader=fragment_shader
            )

            print(f"[DEBUG] Successfully compiled shader program")

            # Debug: List all uniforms in the compiled program
            uniform_names = [name for name in program._members.keys()]
            print(f"[DEBUG] Available uniforms: {uniform_names}")

            return program

        except Exception as e:
            print(f"[DEBUG] Error compiling main shader with warp: {e}")
            
            
            return None

    @benchmark("shader_compile_spectrogram")
    def compile_spectrogram_shader(self) -> Optional[moderngl.Program]:
        """Compile the spectrogram shader"""
        try:
            vertex, fragment = self.spectrogram_shaders
            return self.ctx.program(vertex_shader=vertex, fragment_shader=fragment)
        except Exception as e:
            print(f"Error compiling spectrogram shader: {e}")
            return None

    def _generate_warp_functions(self, warp_map_names: List[str]) -> str:
        """Generate GLSL functions for the specified warp maps"""
        functions = []

        for i, name in enumerate(warp_map_names):
            print(f"[DEBUG] Processing warp map {i}: {name}")
            warp_map = self.warp_map_manager.get_warp_map(name)
            if warp_map:
                print(f"[DEBUG] Found warp map: {name}, GLSL length: {len(warp_map.glsl_code)}")
                # Rename the function to include pattern number
                glsl_code = warp_map.glsl_code
                # Replace get_pattern with get_warp{i} for clarity
                glsl_code = glsl_code.replace("get_pattern", f"get_warp{i}")
                functions.append(f"// Warp {i}: {name}")
                functions.append(glsl_code)
                functions.append("")
            else:
                print(f"[DEBUG] WARNING: Warp map '{name}' not found!")

        result = "\n".join(functions)
        return result

    def _generate_warp_dispatcher(self, warp_map_names: List[str]) -> str:
        """Generate the warp dispatcher switch statement"""
        cases = []

        for i, _ in enumerate(warp_map_names):
            cases.append(f"    if (warp_index == {i}) return get_warp{i}(pos, t);")

        return "\n".join(cases)

    def get_available_warp_maps(self) -> List[str]:
        """Get list of available warp map names"""
        return list(self.warp_map_manager.warp_maps.keys())

    def reload_warp_maps(self):
        """Reload warp maps from disk"""
        self.warp_map_manager.load_all_warp_maps()
        # Clear compiled programs cache
        self.compiled_programs.clear()

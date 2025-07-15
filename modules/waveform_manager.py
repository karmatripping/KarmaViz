import os
import json
import struct
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass
from modules.logging_config import get_logger
from modules.input_sanitizer import input_sanitizer

logger = get_logger("waveform_manager")

@dataclass
class WaveformInfo:
    """Information about a waveform shader"""
    name: str
    category: str
    description: str
    complexity: str
    author: str
    glsl_code: str
    is_builtin: bool = True
    version: str = "1.0"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization (legacy support)"""
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "complexity": self.complexity,
            "author": self.author,
            "glsl_code": self.glsl_code,
            "is_builtin": self.is_builtin,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'WaveformInfo':
        """Create from dictionary (legacy JSON support)"""
        return cls(
            name=data.get("name", "Unknown"),
            category=data.get("category", "Basic"),
            description=data.get("description", ""),
            complexity=data.get("complexity", "medium"),
            author=data.get("author", "Unknown"),
            glsl_code=data.get("glsl_code", ""),
            is_builtin=data.get("is_builtin", True),
            version=data.get("version", "1.0"),
        )

    def to_binary(self) -> bytes:
        """Convert to binary format"""
        # Binary format specification:
        # - Magic header: b'KVWF' (4 bytes)
        # - Format version: uint32 (4 bytes)
        # - Is builtin: uint8 (1 byte, 0 or 1)
        # - Name length: uint32 (4 bytes)
        # - Name: UTF-8 string
        # - Category length: uint32 (4 bytes)
        # - Category: UTF-8 string
        # - Description length: uint32 (4 bytes)
        # - Description: UTF-8 string
        # - Complexity length: uint32 (4 bytes)
        # - Complexity: UTF-8 string
        # - Author length: uint32 (4 bytes)
        # - Author: UTF-8 string
        # - Version length: uint32 (4 bytes)
        # - Version: UTF-8 string
        # - GLSL code length: uint32 (4 bytes)
        # - GLSL code: UTF-8 string

        data = bytearray()

        # Magic header and format version
        data.extend(b"KVWF")
        data.extend(struct.pack("<I", 1))  # Format version 1

        # Builtin flag
        data.extend(struct.pack("<B", 1 if self.is_builtin else 0))

        # String fields with length prefixes
        for field in [
            self.name,
            self.category,
            self.description,
            self.complexity,
            self.author,
            self.version,
            self.glsl_code,
        ]:
            field_bytes = field.encode("utf-8")
            data.extend(struct.pack("<I", len(field_bytes)))
            data.extend(field_bytes)

        return bytes(data)

    @classmethod
    def from_binary(cls, data: bytes) -> "WaveformInfo":
        """Create from binary format"""
        if len(data) < 8:
            raise ValueError("Invalid binary data: too short")

        # Check magic header
        if data[:4] != b"KVWF":
            raise ValueError("Invalid binary data: wrong magic header")

        # Check format version
        format_version = struct.unpack("<I", data[4:8])[0]
        if format_version != 1:
            raise ValueError(f"Unsupported format version: {format_version}")

        offset = 8

        # Read builtin flag
        if offset + 1 > len(data):
            raise ValueError("Invalid binary data: incomplete header")

        is_builtin = bool(struct.unpack("<B", data[offset : offset + 1])[0])
        offset += 1

        # Read string fields
        fields = []
        for _ in range(
            7
        ):  # name, category, description, complexity, author, version, glsl_code
            if offset + 4 > len(data):
                raise ValueError("Invalid binary data: incomplete string length")

            str_len = struct.unpack("<I", data[offset : offset + 4])[0]
            offset += 4

            if offset + str_len > len(data):
                raise ValueError("Invalid binary data: incomplete string data")

            field_str = data[offset : offset + str_len].decode("utf-8")
            fields.append(field_str)
            offset += str_len

        # Sanitize the loaded data
        metadata = {
            'name': fields[0],
            'category': fields[1],
            'description': fields[2],
            'author': fields[4],
            'version': fields[5],
            'glsl_code': fields[6]
        }
        
        sanitized_metadata, warnings, errors = input_sanitizer.sanitize_all_metadata(metadata)
        
        # Log warnings and errors but don't fail loading
        if warnings:
            logger.warning(f"Sanitization warnings for loaded waveform: {warnings}")
        if errors:
            logger.error(f"Sanitization errors for loaded waveform: {errors}")
            # For critical errors, use safe defaults
            if 'name' in [e.split(':')[0] for e in errors]:
                sanitized_metadata['name'] = "invalid_waveform"
            if 'glsl_code' in [e.split(':')[0] for e in errors]:
                sanitized_metadata['glsl_code'] = "// Invalid GLSL code was sanitized\nvoid main() {}"

        return cls(
            name=sanitized_metadata['name'],
            category=sanitized_metadata['category'],
            description=sanitized_metadata['description'],
            complexity=fields[3],  # Complexity is validated by UI, not sanitizer
            author=sanitized_metadata['author'],
            version=sanitized_metadata['version'],
            glsl_code=sanitized_metadata['glsl_code'],
            is_builtin=is_builtin,
        )


class WaveformManager:
    """Manages discovery and compilation of modular waveform shaders."""
    def __init__(self, waveforms_dir="waveforms"):
        self.waveforms_dir = Path(waveforms_dir)
        self.waveform_shaders = {}
        self.waveforms: Dict[str, WaveformInfo] = {}
        self.discover_waveforms()

    def discover_waveforms(self):
        """Find all .kvwf (binary) and .json (legacy) waveform files in the directory and subdirectories."""
        self.waveform_shaders = {}
        self.waveforms = {}
        if not self.waveforms_dir.exists():
            self.waveforms_dir.mkdir(parents=True, exist_ok=True)
            return

        # First, load binary waveforms (.kvwf files) recursively
        for file in self.waveforms_dir.rglob("*.kvwf"):
            name = file.stem
            self.waveform_shaders[name] = file

            # Load waveform info from binary format
            try:
                with open(file, "rb") as f:
                    data = f.read()
                    waveform_info = WaveformInfo.from_binary(data)
                    waveform_info.name = name  # Ensure name matches filename
                    self.waveforms[name] = waveform_info
            except Exception as e:
                print(f"Error loading binary waveform {name}: {e}")

        # Then, load legacy JSON waveforms (only if no binary version exists) recursively
        for file in self.waveforms_dir.rglob("*.json"):
            name = file.stem
            if name not in self.waveforms:  # Only load if binary version doesn't exist
                self.waveform_shaders[name] = file

                # Load waveform info from JSON format
                try:
                    with open(file, "r") as f:
                        data = json.load(f)
                        waveform_info = WaveformInfo.from_dict(data)
                        waveform_info.name = name  # Ensure name matches filename
                        self.waveforms[name] = waveform_info
                except Exception as e:
                    print(f"Error loading JSON waveform {name}: {e}")

    def list_waveforms(self):
        """Return a list of available waveform names."""
        return list(self.waveform_shaders.keys())

    def list_waveforms_by_category(self) -> Dict[str, List[str]]:
        """Return waveforms organized by their subdirectory/category"""
        categories = {}

        for name in self.waveform_shaders.keys():
            subdirectory = self.get_waveform_subdirectory(name)
            category = subdirectory if subdirectory else "root"

            if category not in categories:
                categories[category] = []
            categories[category].append(name)

        # Sort waveforms within each category
        for category in categories:
            categories[category].sort()

        return categories

    def get_available_categories(self) -> List[str]:
        """Get a list of available waveform categories/subdirectories"""
        categories = set()

        for name in self.waveform_shaders.keys():
            subdirectory = self.get_waveform_subdirectory(name)
            if subdirectory:
                categories.add(subdirectory)

        return sorted(list(categories))

    def get_all_waveforms(self) -> List[WaveformInfo]:
        """Get all waveform info objects"""
        return list(self.waveforms.values())

    def get_waveform(self, name: str) -> Optional[WaveformInfo]:
        """Get a specific waveform by name"""
        return self.waveforms.get(name)

    def get_waveform_subdirectory(self, name: str) -> Optional[str]:
        """Get the subdirectory where a waveform is located"""
        file_path = self.waveform_shaders.get(name)
        if not file_path:
            return None

        # Get the relative path from the waveforms directory
        try:
            relative_path = file_path.relative_to(self.waveforms_dir)
            # If the file is in a subdirectory, return the first part of the path
            if len(relative_path.parts) > 1:
                return relative_path.parts[0]
            else:
                return None  # File is in the root waveforms directory
        except ValueError:
            return None

    def save_waveform(
        self,
        waveform_info: WaveformInfo,
        overwrite: bool = False,
        subdirectory: str = None,
    ) -> bool:
        """Save a waveform to disk in binary format

        Args:
            waveform_info: The waveform information to save
            overwrite: Whether to overwrite existing waveforms
            subdirectory: Optional subdirectory to save the waveform in (e.g., 'basic', 'experimental')
        """
        try:
            # Check if waveform already exists and we're not overwriting
            if not overwrite and waveform_info.name in self.waveforms:
                return False

            # Create the file path for binary format
            if subdirectory:
                # Ensure subdirectory exists
                subdir_path = self.waveforms_dir / subdirectory
                subdir_path.mkdir(parents=True, exist_ok=True)
                file_path = subdir_path / f"{waveform_info.name}.kvwf"
                old_json_path = subdir_path / f"{waveform_info.name}.json"
            else:
                file_path = self.waveforms_dir / f"{waveform_info.name}.kvwf"
                old_json_path = self.waveforms_dir / f"{waveform_info.name}.json"

            # Save to binary file
            with open(file_path, "wb") as f:
                f.write(waveform_info.to_binary())

            # Remove old JSON file if it exists (check both root and subdirectory)
            if old_json_path.exists():
                try:
                    old_json_path.unlink()
                    print(f"Removed legacy JSON file: {old_json_path}")
                except Exception as e:
                    print(
                        f"Warning: Could not remove old JSON file {old_json_path}: {e}"
                    )

            # Also check for JSON file in root directory if we're saving to subdirectory
            if subdirectory:
                root_json_path = self.waveforms_dir / f"{waveform_info.name}.json"
                if root_json_path.exists():
                    try:
                        root_json_path.unlink()
                        print(f"Removed legacy JSON file from root: {root_json_path}")
                    except Exception as e:
                        print(
                            f"Warning: Could not remove old JSON file {root_json_path}: {e}"
                        )

            # Update internal storage
            self.waveforms[waveform_info.name] = waveform_info
            self.waveform_shaders[waveform_info.name] = file_path

            return True
        except Exception as e:
            print(f"Error saving waveform {waveform_info.name}: {e}")
            return False

    def delete_waveform(self, name: str) -> bool:
        """Delete a waveform"""
        try:
            # Don't delete built-in waveforms
            waveform = self.waveforms.get(name)
            if waveform and waveform.is_builtin:
                return False

            # Get the actual file path from our internal storage
            file_path = self.waveform_shaders.get(name)
            if file_path and file_path.exists():
                file_path.unlink()
                print(f"Deleted waveform file: {file_path}")

            # Also check for legacy JSON files in the same directory as the binary file
            if file_path:
                json_path = file_path.with_suffix(".json")
                if json_path.exists():
                    json_path.unlink()
                    print(f"Deleted legacy JSON file: {json_path}")

            # Remove from internal storage
            self.waveforms.pop(name, None)
            self.waveform_shaders.pop(name, None)

            return True
        except Exception as e:
            print(f"Error deleting waveform {name}: {e}")
            return False

    def load_shader_code(self, name):
        """Load the GLSL code for a given waveform by name."""
        
        waveform = self.waveforms.get(name)
        if not waveform:
            raise FileNotFoundError(f"Waveform '{name}' not found.")

        glsl_code = waveform.glsl_code
        if not glsl_code:
            raise ValueError(f"No GLSL code found in waveform '{name}'.")

        logger.debug(f"Waveform {name} shader code loaded...")
        return glsl_code

    def compile_shader(self, name, ctx, vertex_shader=None):
        """Compile the shader code for the given waveform name using the provided ModernGL context.
        Injects the waveform code into the base fragment shader at the placeholder.
        """
        import pathlib
        base_shader_path = pathlib.Path(__file__).parent.parent / "shaders" / "shaders.py"
        # Extract the FRAGMENT_SHADER string from shaders.py
        with open(base_shader_path, "r") as f:
            base_shader_code = f.read()
        # Find the FRAGMENT_SHADER string
        import re
        match = re.search(r'FRAGMENT_SHADER = """//glsl\n(.*?)"""', base_shader_code, re.DOTALL)
        if not match:
            raise RuntimeError("Could not find FRAGMENT_SHADER in shaders.py")
        fragment_shader = match.group(1)
        # Load the waveform code from JSON
        waveform_code = self.load_shader_code(name)
        # Inject the waveform code
        fragment_shader = fragment_shader.replace("// WAVEFORM_RENDER_PLACEHOLDER", waveform_code)
        if vertex_shader is None:
            # Default passthrough vertex shader (GLSL 330)
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
            program = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
            print("Shader compiled succefully.")
            return program
        except Exception as e:
            raise RuntimeError(f"Failed to compile shader '{name}': {e}")

"""
Preset Manager for KarmaViz

This module handles saving and loading complete visualizer presets that capture:
- All visualizer settings and parameters
- Current palette information
- Current warp map with shader code
- Audio processing settings
- Effect configurations

Presets are stored as compressed .kviz files using a compact binary format.
"""

import os
import time
import gzip
import pickle
import struct
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

from modules.logging_config import get_logger
logger = get_logger("preset_manager")
# Import shader compilation status for threaded compilation
try:
    from modules.shader_compiler import CompilationStatus
except ImportError:
    # Fallback if import fails
    from enum import Enum
    class CompilationStatus(Enum):
        ''' Status of shader compilation'''
        COMPLETED = "completed"

# KarmaViz preset format constants
KVIZ_MAGIC = b'KVIZ'  # File magic number
KVIZ_VERSION = 1      # Format version
KVIZ_EXTENSION = '.kviz'

# Compression levels
COMPRESSION_NONE = 0
COMPRESSION_GZIP = 1
COMPRESSION_PICKLE = 2

# Data type identifiers for compact encoding
TYPE_FLOAT = 0x01
TYPE_INT = 0x02
TYPE_BOOL = 0x03
TYPE_STRING = 0x04
TYPE_LIST = 0x05
TYPE_DICT = 0x06


@dataclass
class PresetInfo:
    """Complete preset information"""
    name: str
    description: str
    created_date: str
    author: str
    version: str

    # Visualizer state
    visualizer_settings: Dict[str, Any]

    # Palette information
    palette_info: Dict[str, Any]

    # Warp map information
    warp_map_info: Dict[str, Any]

    # Waveform information
    waveform_info: Dict[str, Any]

    # Audio settings
    audio_settings: Dict[str, Any]

    # Effect settings
    effect_settings: Dict[str, Any]

    # Metadata
    tags: List[str] = None
    thumbnail_path: str = ""
    file_path: str = ""


class PresetManager:
    """Manages saving and loading of KarmaViz presets"""

    def __init__(self, presets_dir: str = "presets"):
        self.presets_dir = Path(presets_dir)
        self.presets_dir.mkdir(exist_ok=True)

        # Create subdirectories
        self.quick_presets_dir = self.presets_dir / "quick"
        self.quick_presets_dir.mkdir(exist_ok=True)
        self.user_presets_dir = self.presets_dir / "user"
        self.user_presets_dir.mkdir(exist_ok=True)

        self.presets: Dict[str, PresetInfo] = {}
        self.compression_level = COMPRESSION_GZIP  # Default compression
        self.load_all_presets()

    def _encode_compact_data(self, data: Any) -> bytes:
        """Encode data in a compact binary format"""
        if isinstance(data, float):
            return struct.pack('Bf', TYPE_FLOAT, data)
        elif isinstance(data, int):
            return struct.pack('Bi', TYPE_INT, data)
        elif isinstance(data, bool):
            return struct.pack('B?', TYPE_BOOL, data)
        elif isinstance(data, str):
            encoded_str = data.encode('utf-8')
            return struct.pack('BI', TYPE_STRING, len(encoded_str)) + encoded_str
        elif isinstance(data, list):
            result = struct.pack('BI', TYPE_LIST, len(data))
            for item in data:
                result += self._encode_compact_data(item)
            return result
        elif isinstance(data, dict):
            result = struct.pack('BI', TYPE_DICT, len(data))
            for key, value in data.items():
                result += self._encode_compact_data(key)
                result += self._encode_compact_data(value)
            return result
        else:
            # Fallback to string representation
            return self._encode_compact_data(str(data))

    def _decode_compact_data(self, data: bytes, offset: int = 0) -> tuple[Any, int]:
        """Decode compact binary data, returns (value, new_offset)"""
        if offset >= len(data):
            raise ValueError("Unexpected end of data")

        data_type = data[offset]
        offset += 1

        if data_type == TYPE_FLOAT:
            value = struct.unpack_from('f', data, offset)[0]
            return value, offset + 4
        elif data_type == TYPE_INT:
            value = struct.unpack_from('i', data, offset)[0]
            return value, offset + 4
        elif data_type == TYPE_BOOL:
            value = struct.unpack_from('?', data, offset)[0]
            return value, offset + 1
        elif data_type == TYPE_STRING:
            length = struct.unpack_from('I', data, offset)[0]
            offset += 4
            value = data[offset:offset + length].decode('utf-8')
            return value, offset + length
        elif data_type == TYPE_LIST:
            length = struct.unpack_from('I', data, offset)[0]
            offset += 4
            result = []
            for _ in range(length):
                item, offset = self._decode_compact_data(data, offset)
                result.append(item)
            return result, offset
        elif data_type == TYPE_DICT:
            length = struct.unpack_from('I', data, offset)[0]
            offset += 4
            result = {}
            for _ in range(length):
                key, offset = self._decode_compact_data(data, offset)
                value, offset = self._decode_compact_data(data, offset)
                result[key] = value
            return result, offset
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    def _save_kviz_format(self, preset_info: PresetInfo, filepath: Path) -> bool:
        """Save preset in compact .kviz format"""
        try:
            logger.debug(f"Saving preset to {filepath}")
            # Convert preset to dictionary
            preset_dict = asdict(preset_info)
            logger.debug(f"Preset dict keys: {list(preset_dict.keys())}")
            logger.debug(f"Visualizer settings keys: {list(preset_dict.get('visualizer_settings', {}).keys())}")
            logger.debug(f"Waveform name in preset: {preset_dict.get('visualizer_settings', {}).get('current_waveform_name', 'NOT_SET')}")

            # Create header
            header = struct.pack('4sII', KVIZ_MAGIC, KVIZ_VERSION, self.compression_level)
            logger.debug(f"Using compression level: {self.compression_level}")

            # Serialize data based on compression level
            if self.compression_level == COMPRESSION_NONE:
                # Use compact binary encoding
                data = self._encode_compact_data(preset_dict)
            elif self.compression_level == COMPRESSION_GZIP:
                # Use gzip compression on JSON
                json_data = json.dumps(preset_dict, separators=(',', ':')).encode('utf-8')
                logger.debug(f"JSON data length: {len(json_data)}")
                logger.debug(f"JSON preview: {json_data[:200]}...")
                data = gzip.compress(json_data, compresslevel=9)
                logger.debug(f"Compressed data length: {len(data)}")
            elif self.compression_level == COMPRESSION_PICKLE:
                # Use pickle with gzip compression
                pickle_data = pickle.dumps(preset_dict, protocol=pickle.HIGHEST_PROTOCOL)
                data = gzip.compress(pickle_data, compresslevel=9)
            else:
                raise ValueError(f"Unknown compression level: {self.compression_level}")

            # Write file
            with open(filepath, 'wb') as f:
                f.write(header)
                f.write(struct.pack('I', len(data)))  # Data length
                f.write(data)

            logger.debug(f"File written successfully, total size: {filepath.stat().st_size} bytes")
            return True

        except Exception as e:
            print(f"Error saving .kviz format: {e}")
            return False

    def _load_kviz_format(self, filepath: Path) -> Optional[PresetInfo]:
        """Load preset from .kviz format"""
        try:
            logger.debug(f"Loading .kviz file: {filepath}")
            
            with open(filepath, 'rb') as f:
                # Read header
                header = f.read(12)  # 4 bytes magic + 4 bytes version + 4 bytes compression
                if len(header) != 12:
                    raise ValueError("Invalid file header")

                magic, version, compression = struct.unpack('4sII', header)
                logger.debug(f"File magic: {magic}, version: {version}, compression: {compression}")

                if magic != KVIZ_MAGIC:
                    raise ValueError(f"Invalid magic number: {magic}")

                if version != KVIZ_VERSION:
                    print(f"Warning: Preset version {version} may not be fully compatible")

                # Read data length
                data_length = struct.unpack('I', f.read(4))[0]
                logger.debug(f"Data length: {data_length} bytes")

                # Read data
                data = f.read(data_length)
                if len(data) != data_length:
                    raise ValueError("Incomplete data")

                logger.debug(f"Successfully read {len(data)} bytes of data")

                # Deserialize based on compression
                if compression == COMPRESSION_NONE:
                    logger.debug(f"Using compact binary decompression")
                    preset_dict, _ = self._decode_compact_data(data)
                elif compression == COMPRESSION_GZIP:
                    logger.debug(f"Using gzip decompression")
                    json_data = gzip.decompress(data).decode('utf-8')
                    logger.debug(f"Decompressed JSON length: {len(json_data)}")
                    preset_dict = json.loads(json_data)
                elif compression == COMPRESSION_PICKLE:
                    logger.debug(f"Using pickle decompression")
                    pickle_data = gzip.decompress(data)
                    preset_dict = pickle.loads(pickle_data)
                else:
                    raise ValueError(f"Unknown compression level: {compression}")

                logger.debug(f"Deserialized preset dict with keys: {list(preset_dict.keys())}")
                
                # Check for new palette format
                if 'palette_info' in preset_dict:
                    palette_info = preset_dict['palette_info']
                    if 'fixed_palette_json' in palette_info:
                        logger.debug(f"Found fixed palette JSON in loaded preset")
                        logger.debug(f"Fixed palette name: {palette_info.get('fixed_palette_name', 'UNKNOWN')}")

                # Convert back to PresetInfo
                preset_info = PresetInfo(**preset_dict)
                preset_info.file_path = str(filepath)

                logger.debug(f"Successfully loaded preset: {preset_info.name}")
                return preset_info

        except Exception as e:
            print(f"Error loading .kviz format from {filepath}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def capture_visualizer_state(self, visualizer) -> Dict[str, Any]:
        """Capture complete visualizer state"""
        try:
            logger.debug(f"Capturing visualizer state...")
            current_waveform = getattr(visualizer, "current_waveform_name", None)
            logger.debug(f"Current waveform name: {current_waveform}")

            # Only capture attributes that actually exist on the visualizer
            state = {}
            
            # Define the attributes we want to capture with their expected defaults
            # Only include attributes that should be part of presets (exclude runtime state)
            attributes_to_capture = {
                # Animation settings
                "animation_speed": 1.0,
                "audio_speed_boost": 1.0,
                # Waveform settings - CRITICAL: Include current waveform name
                "current_waveform_name": None,
                "waveform_style": 0,
                "current_waveform_style": 0,
                "waveform_scale": 1.0,
                # Effect settings
                "rotation_mode": 0,
                "pulse_enabled": True,
                "pulse_intensity": 1.0,
                "trail_intensity": 0.8,
                "glow_intensity": 0.9,
                "smoke_intensity": 0.5,
                # Symmetry settings
                "symmetry_mode": 0,
                "kaleidoscope_sections": 6,
                # Bounce settings
                "bounce_enabled": False,
                # Beat detection
                "beats_per_change": 16,
                # Warp settings
                "warp_intensity": 1.0,
                # Note: transitions_paused excluded - it's runtime state, not preset data
            }
            
            # Handle special cases for attributes with different names
            attribute_mappings = {
                "warp_first": "warp_first_enabled",
                "invert_rotation": "invert_rotation_direction", 
                "bounce_intensity": "bounce_intensity_multiplier",
            }
            
            # Capture attributes that exist
            for attr_name, default_value in attributes_to_capture.items():
                if hasattr(visualizer, attr_name):
                    state[attr_name] = getattr(visualizer, attr_name, default_value)
                    logger.debug(f"Captured {attr_name} = {state[attr_name]}")
                else:
                    logger.debug(f"Visualizer missing attribute: {attr_name}")
            
            # Handle mapped attributes
            for preset_name, visualizer_name in attribute_mappings.items():
                if hasattr(visualizer, visualizer_name):
                    state[preset_name] = getattr(visualizer, visualizer_name)
                    logger.debug(f"Captured {preset_name} (mapped from {visualizer_name}) = {state[preset_name]}")
                else:
                    logger.debug(f"Visualizer missing mapped attribute: {visualizer_name}")

            return state

        except Exception as e:
            print(f"Error capturing visualizer state: {e}")
            return {}

    def capture_palette_info(self, visualizer) -> Dict[str, Any]:
        """Capture current palette information"""
        try:
            logger.debug(f"Starting palette capture...")
            
            palette_info = {
                "palette_mode": getattr(visualizer, 'palette_mode', 'auto'),
                "selected_palette_name": getattr(visualizer, 'selected_palette_name', None),
                "palette_speed": getattr(visualizer, 'palette_speed', 1.0),
                "color_cycle_speed": getattr(visualizer, 'color_cycle_speed', 1.0),
                "palette_transition_speed": getattr(visualizer, 'palette_transition_speed', 0.02),
                "color_transition_smoothness": getattr(visualizer, 'color_transition_smoothness', 0.1),
                "color_index": getattr(visualizer, 'color_index', 0),
                "color_time": getattr(visualizer, 'color_time', 0),
            }
            
            logger.debug(f"Palette mode: {palette_info['palette_mode']}")
            logger.debug(f"Selected palette name: {palette_info['selected_palette_name']}")

            # Capture current palette colors if available
            if hasattr(visualizer, 'current_palette') and visualizer.current_palette:
                palette_info["current_palette_colors"] = visualizer.current_palette
                logger.debug(f"Captured current palette colors: {len(visualizer.current_palette)} colors")

            # Capture palette manager state if available
            if hasattr(visualizer, 'palette_manager') and visualizer.palette_manager:
                pm = visualizer.palette_manager
                logger.debug(f"Palette manager found")
                
                if hasattr(pm, 'current_palette_info') and pm.current_palette_info:
                    logger.debug(f"Current palette info: {pm.current_palette_info.name}")
                    
                    # If a fixed palette is selected, save the complete palette in JSON format
                    if (palette_info["palette_mode"] == "fixed" and 
                        palette_info["selected_palette_name"] and 
                        pm.current_palette_info):
                        
                        logger.debug(f"Fixed palette selected - saving complete palette JSON")
                        
                        # Create complete palette JSON structure
                        palette_json = {
                            "name": pm.current_palette_info.name,
                            "energy_level": pm.current_palette_info.energy_level,
                            "warmth": pm.current_palette_info.warmth,
                            "colors": pm.current_palette_info.colors,
                            "description": pm.current_palette_info.description
                        }
                        
                        palette_info["fixed_palette_json"] = palette_json
                        palette_info["fixed_palette_name"] = pm.current_palette_info.name
                        
                        logger.debug(f"Saved complete palette JSON for '{pm.current_palette_info.name}'")
                        logger.debug(f"Palette JSON: {palette_json}")
                else:
                    logger.debug(f"No current palette info in palette manager")
            else:
                logger.debug(f"No palette manager found")

            logger.debug(f"Palette capture completed with keys: {list(palette_info.keys())}")
            return palette_info

        except Exception as e:
            print(f"Error capturing palette info: {e}")
            return {}

    def capture_warp_map_info(self, visualizer) -> Dict[str, Any]:
        """Capture current warp map information including complete shader code"""
        try:
            logger.debug(f"Starting warp map capture...")
            logger.debug(f"Visualizer active_warp_map: {getattr(visualizer, 'active_warp_map', 'NOT_SET')}")
            logger.debug(f"Visualizer active_warp_map_name: {getattr(visualizer, 'active_warp_map_name', 'NOT_SET')}")
            logger.debug(f"Visualizer active_warp_map_index: {getattr(visualizer, 'active_warp_map_index', 'NOT_SET')}")

            warp_info = {
                "active_warp_map": getattr(visualizer, 'active_warp_map', 0),
                "active_warp_map_name": getattr(visualizer, 'active_warp_map_name', None),
                "active_warp_map_index": getattr(visualizer, 'active_warp_map_index', -1),
                "warp_intensity": getattr(visualizer, 'warp_intensity', 1.0),
                "warp_first_enabled": getattr(visualizer, 'warp_first_enabled', False),
            }

            # Capture warp map manager state if available
            if hasattr(visualizer, 'warp_map_manager') and visualizer.warp_map_manager:
                wmm = visualizer.warp_map_manager
                logger.debug(f"Warp map manager found")
                logger.debug(f"WMM current_warp_map: {wmm.current_warp_map.name if hasattr(wmm, 'current_warp_map') and wmm.current_warp_map else 'NOT_SET'}")

                # Priority 1: Try to get warp map by visualizer's active_warp_map_name first
                if hasattr(visualizer, 'active_warp_map_name') and visualizer.active_warp_map_name:
                    warp_map_name = visualizer.active_warp_map_name
                    logger.debug(f"Trying to find warp map by name: {warp_map_name}")
                    if hasattr(wmm, 'warp_maps') and warp_map_name in wmm.warp_maps:
                        warp_map = wmm.warp_maps[warp_map_name]
                        logger.debug(f"Found warp map in wmm.warp_maps")
                        logger.debug(f"Shader code length: {len(warp_map.glsl_code) if warp_map.glsl_code else 0}")
                        logger.debug(f"Shader code preview: {warp_map.glsl_code[:200] if warp_map.glsl_code else 'EMPTY'}...")
                        logger.debug(f"COMPLETE WARP MAP SHADER CODE FOR CAPTURE (BY NAME):")
                        print(f"{'='*80}")
                        print(warp_map.glsl_code if warp_map.glsl_code else "NO SHADER CODE")
                        print(f"{'='*80}")

                        warp_info["current_warp_map"] = {
                            "name": warp_map.name,
                            "category": warp_map.category,
                            "description": warp_map.description,
                            "glsl_code": warp_map.glsl_code,  # Complete shader code
                            "complexity": warp_map.complexity,
                            "author": warp_map.author,
                            "version": warp_map.version,
                            "is_builtin": warp_map.is_builtin,
                        }
                        print(f"📝 Captured shader code for '{warp_map.name}' by name ({len(warp_map.glsl_code)} characters)")

                        # Also update the warp map manager's current_warp_map for consistency
                        logger.debug(f"Updating wmm.current_warp_map for consistency")
                        wmm.current_warp_map = warp_map
                    else:
                        logger.debug(f"Warp map '{warp_map_name}' not found in wmm.warp_maps")
                
                # Priority 2: Fallback to passthrough shader if no valid warp map found
                if "current_warp_map" not in warp_info:
                    logger.debug(f"No valid warp map found - creating passthrough shader")
                    passthrough_shader = """// Basic passthrough warp map for preset compatibility
// This shader applies no transformation to the coordinates

// Required function for warp map compatibility - correct signature
vec2 get_pattern(vec2 pos, float t) {
    return vec2(0.0, 0.0);  // No warp displacement - passthrough
}"""
                    
                    warp_info["current_warp_map"] = {
                        "name": "passthrough",
                        "category": "basic",
                        "description": "Passthrough warp map - no transformation",
                        "glsl_code": passthrough_shader,
                        "complexity": 1,
                        "author": "KarmaViz",
                        "version": "1.0",
                        "is_builtin": True,
                    }
                    print(f"📝 Created passthrough shader for preset compatibility")

                # Get list of available warp maps for reference
                if hasattr(wmm, 'warp_map_names'):
                    warp_info["available_warp_maps"] = wmm.warp_map_names
                elif hasattr(wmm, 'warp_maps'):
                    warp_info["available_warp_maps"] = list(wmm.warp_maps.keys())

            # Capture shader compiler state if available
            if hasattr(visualizer, 'shader_compiler') and visualizer.shader_compiler:
                sc = visualizer.shader_compiler
                if hasattr(sc, 'active_warp_maps'):
                    warp_info["active_warp_maps"] = sc.active_warp_maps

                # Capture compiled shader information
                if hasattr(sc, 'current_shader_source'):
                    warp_info["compiled_shader_info"] = {
                        "has_compiled_shader": True,
                        "shader_length": len(sc.current_shader_source) if sc.current_shader_source else 0
                    }

            # Final verification
            if "current_warp_map" in warp_info:
                shader_code = warp_info["current_warp_map"].get("glsl_code", "")
                if shader_code and len(shader_code.strip()) > 0:
                    print(f"Successfully captured warp map with {len(shader_code)} character shader code")
                else:
                    print(f"Warning: Captured warp map but shader code is empty or missing")

            return warp_info

        except Exception as e:
            print(f"Error capturing warp map info: {e}")
            return {}

    def capture_waveform_info(self, visualizer) -> Dict[str, Any]:
        """Capture current waveform information including complete shader code"""
        try:
            logger.debug(f"Starting waveform capture...")
            current_waveform_name = getattr(visualizer, 'current_waveform_name', None)
            logger.debug(f"Current waveform name: {current_waveform_name}")

            waveform_info = {
                "current_waveform_name": current_waveform_name,
                "waveform_index": getattr(visualizer, 'waveform_index', -1),
            }

            # Capture waveform manager state if available
            wfm = None
            if hasattr(visualizer, 'waveform_manager') and visualizer.waveform_manager:
                wfm = visualizer.waveform_manager
                logger.debug(f"Waveform manager found via visualizer.waveform_manager")
            elif hasattr(visualizer, 'shader_manager') and visualizer.shader_manager and hasattr(visualizer.shader_manager, 'waveform_manager'):
                wfm = visualizer.shader_manager.waveform_manager
                logger.debug(f"Waveform manager found via visualizer.shader_manager.waveform_manager")

            if wfm:

                # Get current waveform info with complete shader code
                if current_waveform_name and hasattr(wfm, 'waveforms') and current_waveform_name in wfm.waveforms:
                    current_waveform = wfm.waveforms[current_waveform_name]
                    logger.debug(f"Capturing waveform: {current_waveform.name}")
                    logger.debug(f"Shader code length: {len(current_waveform.glsl_code) if current_waveform.glsl_code else 0}")
                    logger.debug(f"Shader code preview: {current_waveform.glsl_code[:200] if current_waveform.glsl_code else 'EMPTY'}...")
                    logger.debug(f"COMPLETE SHADER CODE FOR CAPTURE:")
                    print(f"{'='*80}")
                    print(current_waveform.glsl_code if current_waveform.glsl_code else "NO SHADER CODE")
                    print(f"{'='*80}")

                    waveform_info["current_waveform"] = {
                        "name": current_waveform.name,
                        "category": current_waveform.category,
                        "description": current_waveform.description,
                        "glsl_code": current_waveform.glsl_code,  # Complete shader code
                        "complexity": current_waveform.complexity,
                        "author": current_waveform.author,
                        "version": current_waveform.version,
                        "is_builtin": current_waveform.is_builtin,
                    }

                    # Verify shader code is not empty
                    if not current_waveform.glsl_code or len(current_waveform.glsl_code.strip()) == 0:
                        print(f"Warning: Waveform '{current_waveform.name}' has empty shader code")
                    else:
                        print(f"📝 Captured waveform shader code for '{current_waveform.name}' ({len(current_waveform.glsl_code)} characters)")
                else:
                    logger.debug(f"Waveform '{current_waveform_name}' not found in waveform manager")
                    if hasattr(wfm, 'waveforms'):
                        logger.debug(f"Available waveforms: {list(wfm.waveforms.keys())}")

                # Get list of available waveforms for reference
                if hasattr(wfm, 'waveform_names'):
                    waveform_info["available_waveforms"] = wfm.waveform_names
            else:
                logger.debug(f"No waveform manager found")

            return waveform_info

        except Exception as e:
            print(f"Error capturing waveform info: {e}")
            return {}

    def capture_audio_settings(self, visualizer) -> Dict[str, Any]:
        """Capture audio processing settings"""
        try:
            logger.debug(f"Starting audio settings capture...")
            
            audio_settings = {
                "chunk_size": getattr(visualizer, 'chunk_size', 512),
                "sample_rate": getattr(visualizer, 'sample_rate', 80000),
                "channels": getattr(visualizer, 'channels', 1),
                "audio_format": getattr(visualizer, 'audio_format', 'float32'),
            }
            
            logger.debug(f"Audio settings captured:")
            for key, value in audio_settings.items():
                logger.debug(f"  {key}: {value}")

            return audio_settings

        except Exception as e:
            print(f"Error capturing audio settings: {e}")
            return {}

    def capture_effect_settings(self, visualizer) -> Dict[str, Any]:
        """Capture effect-specific settings"""
        try:
            logger.debug(f"Starting effect settings capture...")
            
            effect_settings = {
                # Anti-aliasing
                "msaa_samples": getattr(visualizer, 'msaa_samples', 0),
                "fxaa_enabled": getattr(visualizer, 'fxaa_enabled', False),

                # Performance settings
                "vsync_enabled": getattr(visualizer, 'vsync_enabled', True),
                "frame_limit": getattr(visualizer, 'frame_limit', None),

                # Rendering settings
                "render_quality": getattr(visualizer, 'render_quality', 'high'),
                "texture_filtering": getattr(visualizer, 'texture_filtering', True),
            }
            
            logger.debug(f"Effect settings captured:")
            for key, value in effect_settings.items():
                logger.debug(f"  {key}: {value}")

            return effect_settings

        except Exception as e:
            print(f"Error capturing effect settings: {e}")
            return {}

    def save_preset(self, visualizer, name: str, description: str = "",
                   author: str = "User", tags: List[str] = None) -> bool:
        """Save current visualizer state as a preset"""
        try:
            logger.debug(f"========== STARTING PRESET SAVE: '{name}' ==========")
            
            # Sanitize filename
            safe_name = self._sanitize_filename(name)
            logger.debug(f"Sanitized filename: {safe_name}")

            # Capture all state information with detailed logging
            logger.debug(f"=== CAPTURING VISUALIZER STATE ===")
            visualizer_settings = self.capture_visualizer_state(visualizer)
            logger.debug(f"Visualizer settings captured: {list(visualizer_settings.keys()) if visualizer_settings else 'EMPTY'}")
            
            logger.debug(f"=== CAPTURING PALETTE INFO ===")
            palette_info = self.capture_palette_info(visualizer)
            logger.debug(f"Palette info captured: {list(palette_info.keys()) if palette_info else 'EMPTY'}")
            
            logger.debug(f"=== CAPTURING WARP MAP INFO ===")
            warp_map_info = self.capture_warp_map_info(visualizer)
            logger.debug(f"Warp map info captured: {list(warp_map_info.keys()) if warp_map_info else 'EMPTY'}")
            
            logger.debug(f"=== CAPTURING WAVEFORM INFO ===")
            waveform_info = self.capture_waveform_info(visualizer)
            logger.debug(f"Waveform info captured: {list(waveform_info.keys()) if waveform_info else 'EMPTY'}")
            
            logger.debug(f"=== CAPTURING AUDIO SETTINGS ===")
            audio_settings = self.capture_audio_settings(visualizer)
            logger.debug(f"Audio settings captured: {list(audio_settings.keys()) if audio_settings else 'EMPTY'}")
            
            logger.debug(f"=== CAPTURING EFFECT SETTINGS ===")
            effect_settings = self.capture_effect_settings(visualizer)
            logger.debug(f"Effect settings captured: {list(effect_settings.keys()) if effect_settings else 'EMPTY'}")

            # Create preset info structure
            logger.debug(f"=== CREATING PRESET INFO STRUCTURE ===")
            preset_info = PresetInfo(
                name=name,
                description=description,
                created_date=datetime.now().isoformat(),
                author=author,
                version="1.0",
                visualizer_settings=visualizer_settings,
                palette_info=palette_info,
                warp_map_info=warp_map_info,
                waveform_info=waveform_info,
                audio_settings=audio_settings,
                effect_settings=effect_settings,
                tags=tags or [],
            )
            
            logger.debug(f"Preset info structure created successfully")
            logger.debug(f"Preset contains:")
            logger.debug(f"  - Visualizer settings: {len(visualizer_settings)} items")
            logger.debug(f"  - Palette info: {len(palette_info)} items")
            logger.debug(f"  - Warp map info: {len(warp_map_info)} items")
            logger.debug(f"  - Waveform info: {len(waveform_info)} items")
            logger.debug(f"  - Audio settings: {len(audio_settings)} items")
            logger.debug(f"  - Effect settings: {len(effect_settings)} items")

            # Save to file in .kviz format
            filepath = self.presets_dir / f"{safe_name}{KVIZ_EXTENSION}"
            preset_info.file_path = str(filepath)
            
            logger.debug(f"=== SAVING TO FILE ===")
            logger.debug(f"Target filepath: {filepath}")

            # Save in compact .kviz format
            success = self._save_kviz_format(preset_info, filepath)

            if success:
                # Add to internal storage
                self.presets[name] = preset_info

                # Get file size for comparison
                file_size = filepath.stat().st_size
                logger.debug(f"========== PRESET SAVE COMPLETED ==========")
                print(f"Preset '{name}' saved successfully to {filepath} ({file_size} bytes)")
                return True
            else:
                logger.debug(f"========== PRESET SAVE FAILED ==========")
                print(f"Failed to save preset '{name}' in .kviz format")
                return False

        except Exception as e:
            logger.debug(f"========== PRESET SAVE ERROR ==========")
            print(f"Error saving preset '{name}': {e}")
            import traceback
            traceback.print_exc()
            return False

    def save_quick_preset(self, visualizer, slot: int) -> bool:
        """Save current state as a quick preset (Ctrl+0-9)
        
        Args:
            visualizer: The visualizer instance
            slot: Quick preset slot (0-9)
            
        Returns:
            True if successful
        """
        if not (0 <= slot <= 9):
            print(f"Invalid quick preset slot: {slot}")
            return False

        try:
            name = f"Quick Preset {slot}"
            description = f"Quick preset saved to slot {slot} on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

            # Capture all state information
            logger.debug(f"About to capture waveform info for quick preset...")
            waveform_info = self.capture_waveform_info(visualizer)
            logger.debug(f"Quick preset waveform info captured: {list(waveform_info.keys()) if waveform_info else 'EMPTY'}")

            preset_info = PresetInfo(
                name=name,
                description=description,
                created_date=datetime.now().isoformat(),
                author="User",
                version="1.0",
                visualizer_settings=self.capture_visualizer_state(visualizer),
                palette_info=self.capture_palette_info(visualizer),
                warp_map_info=self.capture_warp_map_info(visualizer),
                waveform_info=waveform_info,  # Add missing waveform_info
                audio_settings=self.capture_audio_settings(visualizer),
                effect_settings=self.capture_effect_settings(visualizer),
                tags=["quick"],
            )

            # Save to quick presets directory
            filepath = self.quick_presets_dir / f"quick_{slot}{KVIZ_EXTENSION}"
            preset_info.file_path = str(filepath)

            # Save in compact .kviz format
            success = self._save_kviz_format(preset_info, filepath)

            if success:
                print(f"Quick preset {slot} saved successfully")
                return True
            else:
                print(f"Failed to save quick preset {slot}")
                return False

        except Exception as e:
            print(f"Error saving quick preset {slot}: {e}")
            return False

    def load_quick_preset(self, visualizer, slot: int) -> bool:
        """Load a quick preset (0-9)
        
        Args:
            visualizer: The visualizer instance
            slot: Quick preset slot (0-9)
            
        Returns:
            True if successful
        """
        if not (0 <= slot <= 9):
            print(f"Invalid quick preset slot: {slot}")
            return False

        try:
            filepath = self.quick_presets_dir / f"quick_{slot}{KVIZ_EXTENSION}"

            if not filepath.exists():
                print(f"Quick preset {slot} not found")
                return False

            # Load the preset
            preset_info = self._load_kviz_format(filepath)
            if preset_info is None:
                print(f"Failed to load quick preset {slot}")
                return False

            # Apply the preset to the visualizer (shader-only mode)
            success = self.apply_preset(visualizer, preset_info)

            if success:
                print(f"Quick preset {slot} loaded successfully")
                return True
            else:
                print(f"Failed to apply quick preset {slot}")
                return False

        except Exception as e:
            print(f"Error loading quick preset {slot}: {e}")
            return False

    def quick_preset_exists(self, slot: int) -> bool:
        """Check if a quick preset exists
        
        Args:
            slot: Quick preset slot (0-9)
            
        Returns:
            True if the preset exists
        """
        if not (0 <= slot <= 9):
            return False

        filepath = self.quick_presets_dir / f"quick_{slot}{KVIZ_EXTENSION}"
        return filepath.exists()

    def _sanitize_filename(self, name: str) -> str:
        """Convert a preset name to a safe filename"""
        # Replace spaces with underscores, convert to lowercase, and remove unsafe characters
        safe_name = name.lower().replace(' ', '_')

        # Remove or replace unsafe characters
        unsafe_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        for char in unsafe_chars:
            safe_name = safe_name.replace(char, '_')

        # Remove multiple consecutive underscores
        while '__' in safe_name:
            safe_name = safe_name.replace('__', '_')

        # Remove leading/trailing underscores
        safe_name = safe_name.strip('_')

        # Ensure it's not empty
        if not safe_name:
            safe_name = f'preset_{int(time.time())}'

        return safe_name

    def load_preset(self, name: str, visualizer) -> bool:
        """Load a preset and apply it to the visualizer"""
        try:
            logger.debug(f"========== STARTING PRESET LOAD: '{name}' ==========")
            
            if name not in self.presets:
                logger.debug(f"Preset '{name}' not found in loaded presets")
                logger.debug(f"Available presets: {list(self.presets.keys())}")
                return False

            preset = self.presets[name]
            logger.debug(f"Found preset '{name}' in memory")
            logger.debug(f"Preset file path: {preset.file_path}")
            logger.debug(f"Preset created: {preset.created_date}")
            logger.debug(f"Preset author: {preset.author}")

            logger.debug(f"=== APPLYING PRESET ===")
            print(f"Loading preset '{name}' (full restoration mode)...")

            # Apply the preset to the visualizer
            success = self.apply_preset(visualizer, preset)
            if not success:
                logger.debug(f"========== PRESET LOAD FAILED ==========")
                return False

            logger.debug(f"========== PRESET LOAD COMPLETED ==========")
            print(f"Preset '{name}' loaded successfully!")
            return True

        except Exception as e:
            logger.debug(f"========== PRESET LOAD ERROR ==========")
            print(f"Error loading preset '{name}': {e}")
            import traceback
            traceback.print_exc()
            return False

    def apply_preset(self, visualizer, preset_info: PresetInfo) -> bool:
        """Apply complete preset restoration including palette JSON format
        
        This method restores all aspects of the visualizer state from the preset,
        including the new fixed palette JSON format.
        """
        try:
            logger.debug(f"=== STARTING PRESET APPLICATION ===")
            logger.debug(f"Applying preset '{preset_info.name}'...")
            logger.debug(f"Preset contains:")
            logger.debug(f"  - Visualizer settings: {len(preset_info.visualizer_settings)} items")
            logger.debug(f"  - Palette info: {len(preset_info.palette_info)} items")
            logger.debug(f"  - Warp map info: {len(preset_info.warp_map_info)} items")
            logger.debug(f"  - Waveform info: {len(preset_info.waveform_info)} items")
            logger.debug(f"  - Audio settings: {len(preset_info.audio_settings)} items")
            logger.debug(f"  - Effect settings: {len(preset_info.effect_settings)} items")

            # Apply visualizer settings first
            logger.debug(f"=== APPLYING VISUALIZER SETTINGS ===")
            if preset_info.visualizer_settings:
                self._apply_visualizer_settings(visualizer, preset_info.visualizer_settings)
            else:
                logger.debug(f"No visualizer settings to apply")

            # Apply palette settings (including new fixed palette JSON format)
            logger.debug(f"=== APPLYING PALETTE SETTINGS ===")
            if preset_info.palette_info:
                self._apply_palette_settings(visualizer, preset_info.palette_info)
            else:
                logger.debug(f"No palette settings to apply")

            # Apply warp map shader code
            logger.debug(f"=== APPLYING WARP MAP SETTINGS ===")
            if preset_info.warp_map_info and 'current_warp_map' in preset_info.warp_map_info:
                self._apply_warp_map_settings(visualizer, preset_info.warp_map_info)
            else:
                logger.debug(f"No warp map settings to apply")

            # Apply waveform shader code
            logger.debug(f"=== APPLYING WAVEFORM SETTINGS ===")
            if preset_info.waveform_info and 'current_waveform' in preset_info.waveform_info:
                self._apply_waveform_settings(visualizer, preset_info.waveform_info)
            else:
                logger.debug(f"No waveform settings to apply")

            # Apply audio settings
            logger.debug(f"=== APPLYING AUDIO SETTINGS ===")
            if preset_info.audio_settings:
                self._apply_audio_settings(visualizer, preset_info.audio_settings)
            else:
                logger.debug(f"No audio settings to apply")

            # Apply effect settings
            logger.debug(f"=== APPLYING EFFECT SETTINGS ===")
            if preset_info.effect_settings:
                self._apply_effect_settings(visualizer, preset_info.effect_settings)
            else:
                logger.debug(f"No effect settings to apply")

            logger.debug(f"=== PRESET APPLICATION COMPLETED ===")
            print(f"Preset '{preset_info.name}' applied successfully!")
            return True

        except Exception as e:
            logger.debug(f"=== PRESET APPLICATION ERROR ===")
            print(f"Error applying preset '{preset_info.name}': {e}")
            import traceback
            traceback.print_exc()
            return False

    def check_and_apply_shader_compilations(self, visualizer) -> int:
        """Check for completed shader compilations from preset loading and apply them.
        
        Args:
            visualizer: The visualizer instance
            
        Returns:
            Number of shader compilations applied
        """
        if not hasattr(visualizer, '_preset_shader_requests') or not visualizer._preset_shader_requests:
            return 0

        if not hasattr(visualizer, 'threaded_shader_compiler') or not visualizer.threaded_shader_compiler:
            return 0

        logger.debug(f"Checking {len(visualizer._preset_shader_requests)} preset shader requests...")
        applied_count = 0
        completed_requests = []

        try:
            # Collect all ready requests first
            ready_requests = []
            for request_id in visualizer._preset_shader_requests:
                logger.debug(f"Checking request {request_id[:8]}...")
                if visualizer.threaded_shader_compiler.is_ready(request_id):
                    logger.debug(f"Request {request_id[:8]} is ready")
                    result = visualizer.threaded_shader_compiler.get_result(request_id)
                    if result and result.status == CompilationStatus.COMPLETED:
                        logger.debug(f"Request {request_id[:8]} completed successfully")
                        ready_requests.append((request_id, result))
                    else:
                        logger.debug(f"Request {request_id[:8]} failed")
                        completed_requests.append(request_id)
                else:
                    logger.debug(f"Request {request_id[:8]} not ready yet")

            # Apply only the most recent successful compilation to avoid conflicts
            if ready_requests:
                # Use the last (most recent) successful compilation
                request_id, result = ready_requests[-1]
                logger.debug(f"Applying most recent compilation: {request_id[:8]} (skipping {len(ready_requests)-1} others)")

                # Compile the final program on main thread
                new_program = visualizer.threaded_shader_compiler.compile_on_main_thread(result)
                if new_program:
                    logger.debug(f"New program compiled successfully for {request_id[:8]}")
                    # Apply the new program to the visualizer
                    if hasattr(visualizer, 'program'):
                        old_program = visualizer.program
                        logger.debug(f"Replacing old program with new one")
                        visualizer.program = new_program

                        # Update shader manager state if available
                        if hasattr(visualizer, 'shader_manager') and visualizer.shader_manager:
                            try:
                                # Update the shader manager's current program reference
                                visualizer.shader_manager.current_program = new_program
                                print(f"   Updated shader manager with new program")
                            except Exception as sm_error:
                                print(f"   Warning: Could not update shader manager: {sm_error}")

                        # Clean up old program
                        if old_program:
                            try:
                                old_program.release()
                            except:
                                pass
                        applied_count += 1
                        print(f"   Applied threaded shader compilation (ID: {request_id[:8]}...)")
                    else:
                        print(f"   Warning: Visualizer has no program attribute to update")
                else:
                    print(f"   Failed to compile final program for request {request_id[:8]}...")
                    logger.debug(f"Compilation result: {result}")

                # Mark all ready requests as completed (since we only applied the most recent)
                for req_id, _ in ready_requests:
                    completed_requests.append(req_id)

            # Remove completed requests from the list
            for request_id in completed_requests:
                visualizer._preset_shader_requests.remove(request_id)

        except Exception as e:
            print(f"   Error checking shader compilations: {e}")

        return applied_count

    def _apply_visualizer_settings(self, visualizer, settings: Dict[str, Any]):
        """Apply visualizer settings to the visualizer instance"""
        try:
            logger.debug(f"Applying visualizer settings...")
            logger.debug(f"Settings keys: {list(settings.keys())}")
            waveform_name = settings.get('current_waveform_name')
            logger.debug(f"Waveform name in settings: {waveform_name}")

            # Handle mapped attributes (preset name -> visualizer attribute name)
            attribute_mappings = {
                "warp_first": "warp_first_enabled",
                "invert_rotation": "invert_rotation_direction",
                "bounce_intensity": "bounce_intensity_multiplier",
            }

            # Skip runtime-only attributes that shouldn't be restored from presets
            runtime_only_attributes = {
                'transitions_paused',  # User-specific runtime state
                'fps',                 # System-specific setting
                'width',               # Window-specific setting
                'height',              # Window-specific setting
                'fullscreen',          # User-specific runtime state
                'selected_fullscreen_res',  # System-specific setting
                'mouse_enabled',       # User-specific runtime state
                'deformation_time',    # Internal timing state
                'waveform_length',     # Internal buffer size
                'beat_sensitivity',    # May not exist on all versions
                'gpu_waveform_enabled', # Internal rendering state
            }

            for key, value in settings.items():
                # Skip runtime-only attributes
                if key in runtime_only_attributes:
                    logger.debug(f"Skipping runtime-only attribute: {key}")
                    continue
                    
                # Check if this is a mapped attribute
                actual_attr_name = attribute_mappings.get(key, key)
                
                if hasattr(visualizer, actual_attr_name):
                    logger.debug(f"Setting {actual_attr_name} = {value}")
                    setattr(visualizer, actual_attr_name, value)
                else:
                    logger.debug(f"Visualizer has no attribute: {actual_attr_name} (from {key})")

            # Special handling for waveform switching
            if waveform_name and hasattr(visualizer, 'select_waveform'):
                logger.debug(f"Switching to waveform: {waveform_name}")
                try:
                    visualizer.select_waveform(waveform_name)
                    logger.debug(f"Successfully switched to waveform: {waveform_name}")
                except Exception as wf_error:
                    logger.debug(f"Failed to switch waveform: {wf_error}")

            # Force update shader uniforms for visual settings
            if hasattr(visualizer, 'program') and visualizer.program:
                # Update common shader uniforms
                uniform_mappings = {
                    'symmetry_mode': 'symmetry_mode',
                    'rotation_mode': 'rotation_mode',
                    'brightness': 'brightness',
                    'contrast': 'contrast',
                    'saturation': 'saturation',
                    'hue_shift': 'hue_shift',
                    'pulse_intensity': 'pulse_intensity',
                    'trail_intensity': 'trail_intensity',
                    'glow_intensity': 'glow_intensity',
                    'glow_radius': 'glow_radius',
                    'smoke_intensity': 'smoke_intensity',
                    'waveform_scale': 'waveform_scale',
                    'rotation_speed': 'rotation_speed',
                    'rotation_amplitude': 'rotation_amplitude',
                    'warp_intensity': 'warp_intensity'
                }

                for setting_key, uniform_name in uniform_mappings.items():
                    if setting_key in settings:
                        try:
                            if uniform_name in visualizer.program:
                                visualizer.program[uniform_name].value = settings[setting_key]
                        except Exception:
                            # Some uniforms might not exist in all shaders
                            pass

            # Force palette update if colors changed
            if 'current_palette' in settings and hasattr(visualizer, 'update_palette_texture'):
                try:
                    visualizer.update_palette_texture()
                except Exception:
                    pass

            print(f"   Applied visualizer settings with shader updates")

        except Exception as e:
            print(f"   Error applying visualizer settings: {e}")

    def _apply_palette_settings(self, visualizer, palette_info: Dict[str, Any]):
        """Apply palette settings to the visualizer instance"""
        try:
            logger.debug(f"Starting palette restoration...")
            logger.debug(f"Palette info keys: {list(palette_info.keys())}")
            
            # Apply basic palette settings
            for key, value in palette_info.items():
                if hasattr(visualizer, key) and key not in ['current_palette_colors', 'fixed_palette_json', 'fixed_palette_name']:
                    logger.debug(f"Setting visualizer.{key} = {value}")
                    setattr(visualizer, key, value)

            # Apply current palette colors if available
            if 'current_palette_colors' in palette_info:
                visualizer.current_palette = palette_info['current_palette_colors']
                logger.debug(f"Applied current palette colors: {len(palette_info['current_palette_colors'])} colors")

                # Force palette texture update
                if hasattr(visualizer, 'update_palette_texture'):
                    try:
                        visualizer.update_palette_texture()
                        logger.debug(f"Updated palette texture")
                    except Exception as e:
                        print(f"   Warning: Could not update palette texture: {e}")

            # Handle fixed palette JSON format (new feature)
            if 'fixed_palette_json' in palette_info and hasattr(visualizer, 'palette_manager'):
                logger.debug(f"Found fixed palette JSON - restoring complete palette")
                
                palette_json = palette_info['fixed_palette_json']
                palette_name = palette_json['name']
                
                logger.debug(f"Restoring fixed palette: {palette_name}")
                logger.debug(f"Palette JSON: {palette_json}")
                
                # Create or update the palette in the palette manager
                from modules.palette_manager import PaletteInfo
                restored_palette = PaletteInfo(
                    name=palette_json['name'],
                    colors=[tuple(color) if isinstance(color, list) else color for color in palette_json['colors']],
                    energy_level=palette_json['energy_level'],
                    warmth=palette_json['warmth'],
                    description=palette_json['description'],
                    is_builtin=False  # Mark as non-builtin since it's from preset
                )
                
                # Add to palette manager
                visualizer.palette_manager.palettes[palette_name] = restored_palette
                
                # Set as current palette
                if hasattr(visualizer.palette_manager, 'set_current_palette'):
                    visualizer.palette_manager.set_current_palette(palette_name)
                    logger.debug(f"Set current palette to: {palette_name}")
                else:
                    visualizer.palette_manager.current_palette_info = restored_palette
                    logger.debug(f"Set current palette info to: {palette_name}")
                
                # Update visualizer state
                visualizer.palette_mode = "fixed"
                visualizer.selected_palette_name = palette_name
                
                logger.debug(f"Successfully restored fixed palette: {palette_name}")
                


            logger.debug(f"Palette restoration completed")
            print(f"   Applied palette settings")

        except Exception as e:
            print(f"   Error applying palette settings: {e}")

    def _apply_warp_map_settings(self, visualizer, warp_info: Dict[str, Any]):
        """Apply warp map settings to the visualizer instance"""
        try:
            logger.debug(f"Starting warp map restoration...")
            logger.debug(f"Warp info keys: {list(warp_info.keys())}")
            logger.debug(f"Current warp map in info: {'current_warp_map' in warp_info}")

            # Apply basic warp settings
            for key, value in warp_info.items():
                if hasattr(visualizer, key) and key not in ['current_warp_map', 'available_warp_maps', 'active_warp_maps', 'compiled_shader_info']:
                    logger.debug(f"Setting visualizer.{key} = {value}")
                    setattr(visualizer, key, value)

            # Apply specific warp map if available
            if 'current_warp_map' in warp_info and hasattr(visualizer, 'warp_map_manager'):
                warp_details = warp_info['current_warp_map']
                logger.debug(f"Restoring warp map: {warp_details.get('name', 'UNKNOWN')}")

            if "current_warp_map" not in warp_info:
                print(f"   No current_warp_map in preset data")
                return

            warp_details = warp_info["current_warp_map"]
            warp_name = warp_details.get("name", "UNKNOWN")
            shader_code = warp_details.get("glsl_code", "")

            logger.debug(f"Restoring warp map shader: {warp_name}")
            print(
                f"🔍 [DEBUG] Shader code length: {len(shader_code) if shader_code else 0}"
            )

            if not shader_code or len(shader_code.strip()) == 0:
                print(f"   Warning: No shader code for warp map '{warp_name}'")
                return

            if (
                not hasattr(visualizer, "warp_map_manager")
                or not visualizer.warp_map_manager
            ):
                print(f"   Warning: No warp map manager available")
                return

            # Update or create the warp map with the shader code
            if warp_name in visualizer.warp_map_manager.warp_maps:
                # Update existing warp map
                existing_warp = visualizer.warp_map_manager.warp_maps[warp_name]
                existing_warp.glsl_code = shader_code
                print(
                    f"   📝 Updated existing warp map '{warp_name}' with preset shader code"
                )
            else:
                # Create new warp map (this shouldn't normally happen, but handle it)
                from modules.warp_map_manager import WarpMapInfo

                new_warp = WarpMapInfo(
                    name=warp_name,
                    category=warp_details.get("category", "basic"),
                    glsl_code=shader_code,
                    description=warp_details.get(
                        "description", f"Restored from preset"
                    ),
                    complexity=warp_details.get("complexity", "medium"),
                )
                visualizer.warp_map_manager.warp_maps[warp_name] = new_warp
                print(
                    f"   📝 Created new warp map '{warp_name}' from preset shader code"
                )

            # Set as current warp map
            if hasattr(visualizer.warp_map_manager, "set_current_warp_map"):
                visualizer.warp_map_manager.set_current_warp_map(warp_name)
            else:
                visualizer.warp_map_manager.current_warp_map = (
                    visualizer.warp_map_manager.warp_maps[warp_name]
                )

            # Update visualizer state
            visualizer.active_warp_map_name = warp_name

            # Don't trigger shader compilation here - will be done after all restoration is complete

            print(f"   Applied warp map shader-only restoration for '{warp_name}'")

        except Exception as e:
            print(f"   Error applying warp map shader-only: {e}")
            

            

    def _apply_audio_settings(self, visualizer, audio_settings: Dict[str, Any]):
        """Apply audio settings to the visualizer instance"""
        try:
            logger.debug(f"Starting audio settings restoration...")
            logger.debug(f"Audio settings keys: {list(audio_settings.keys())}")
            
            for key, value in audio_settings.items():
                if hasattr(visualizer, key):
                    logger.debug(f"Setting visualizer.{key} = {value}")
                    setattr(visualizer, key, value)
                else:
                    logger.debug(f"Visualizer has no attribute: {key}")

            logger.debug(f"Audio settings restoration completed")
            print(f"   Applied audio settings")

        except Exception as e:
            print(f"   Error applying audio settings: {e}")

    def _apply_effect_settings(self, visualizer, effect_settings: Dict[str, Any]):
        """Apply effect settings to the visualizer instance"""
        try:
            logger.debug(f"Starting effect settings restoration...")
            logger.debug(f"Effect settings keys: {list(effect_settings.keys())}")
            
            for key, value in effect_settings.items():
                if hasattr(visualizer, key):
                    logger.debug(f"Setting visualizer.{key} = {value}")
                    setattr(visualizer, key, value)
                else:
                    logger.debug(f"Visualizer has no attribute: {key}")

            logger.debug(f"Effect settings restoration completed")
            print(f"   Applied effect settings")

        except Exception as e:
            print(f"   Error applying effect settings: {e}")


    def _apply_waveform_settings(self, visualizer, waveform_info: Dict[str, Any]):
        """Apply only the waveform shader code from preset data"""
        try:
            logger.debug(f"Starting waveform shader-only restoration...")

            if 'current_waveform' not in waveform_info:
                print(f"   No current_waveform in preset data")
                return

            waveform_details = waveform_info['current_waveform']
            waveform_name = waveform_details.get('name', 'UNKNOWN')
            shader_code = waveform_details.get('glsl_code', '')

            logger.debug(f"Restoring waveform shader: {waveform_name}")
            logger.debug(f"Shader code length: {len(shader_code) if shader_code else 0}")

            if not shader_code or len(shader_code.strip()) == 0:
                print(f"   Warning: No shader code for waveform '{waveform_name}'")
                return

            # Find waveform manager
            wfm = None
            if hasattr(visualizer, 'waveform_manager') and visualizer.waveform_manager:
                wfm = visualizer.waveform_manager
            elif hasattr(visualizer, 'shader_manager') and visualizer.shader_manager and hasattr(visualizer.shader_manager, 'waveform_manager'):
                wfm = visualizer.shader_manager.waveform_manager

            if not wfm:
                print(f"   Warning: No waveform manager available")
                return

            # Update or create the waveform with the shader code
            if waveform_name in wfm.waveforms:
                # Update existing waveform
                existing_waveform = wfm.waveforms[waveform_name]
                existing_waveform.glsl_code = shader_code
                print(f"   📝 Updated existing waveform '{waveform_name}' with preset shader code")
            else:
                # Create new waveform (this shouldn't normally happen, but handle it)
                from modules.waveform_manager import WaveformInfo
                new_waveform = WaveformInfo(
                    name=waveform_name,
                    glsl_code=shader_code,
                    description=waveform_details.get('description', f'Restored from preset'),
                    complexity=waveform_details.get('complexity', 'medium')
                )
                wfm.waveforms[waveform_name] = new_waveform
                print(f"   📝 Created new waveform '{waveform_name}' from preset shader code")

            # Set as current waveform
            if hasattr(wfm, 'set_current_waveform'):
                wfm.set_current_waveform(waveform_name)
            else:
                wfm.current_waveform = wfm.waveforms[waveform_name]

            # Update visualizer state
            visualizer.current_waveform_name = waveform_name

            # Trigger shader compilation
            if hasattr(visualizer, 'shader_manager') and visualizer.shader_manager:
                print(f"   Triggering shader compilation for waveform '{waveform_name}'")
                # Get current warp maps for compilation (names, not objects)
                warp_maps = []
                if hasattr(visualizer, 'warp_map_manager') and visualizer.warp_map_manager:
                    if hasattr(visualizer.warp_map_manager, 'current_warp_map') and visualizer.warp_map_manager.current_warp_map:
                        warp_maps = [visualizer.warp_map_manager.current_warp_map.name]
                    elif hasattr(visualizer, 'active_warp_map_name') and visualizer.active_warp_map_name:
                        active_warp_name = visualizer.active_warp_map_name
                        if active_warp_name in visualizer.warp_map_manager.warp_maps:
                            warp_maps = [active_warp_name]

                visualizer.shader_manager.compile_shader_async(
                    warp_maps=warp_maps,
                    current_waveform_name=waveform_name,
                    purpose="main_shader_waveform_preset_restore"
                )

            print(f"   Applied waveform shader-only restoration for '{waveform_name}'")

        except Exception as e:
            print(f"   Error applying waveform shader-only: {e}")
            
            

    def load_all_presets(self):
        """Load all presets from the presets directory (.kviz format)"""
        try:
            self.presets.clear()
            kviz_count = 0

            # Load .kviz files (new format)
            for preset_file in self.presets_dir.glob("*.kviz"):
                preset_info = self._load_kviz_format(preset_file)
                if preset_info:
                    self.presets[preset_info.name] = preset_info
                    kviz_count += 1

            # Report loading results
            total_presets = len(self.presets)
            if kviz_count > 0:
                print(f" Loaded {total_presets} .kviz presets from {self.presets_dir}")
            else:
                print(f" Loaded {total_presets} presets from {self.presets_dir}")

        except Exception as e:
            print(f"Error loading presets: {e}")

    def get_preset_names(self) -> List[str]:
        """Get list of all preset names"""
        return list(self.presets.keys())

    def list_presets(self, directory=None) -> List[tuple]:
        """Get list of all presets as (filepath, preset_info) tuples"""
        result = []
        for preset_info in self.presets.values():
            if preset_info.file_path:
                result.append((preset_info.file_path, preset_info))
        return result

    def get_preset_info(self, name: str) -> Optional[PresetInfo]:
        """Get preset information by name"""
        return self.presets.get(name)

    def delete_preset(self, name: str) -> bool:
        """Delete a preset"""
        try:
            if name not in self.presets:
                print(f"Preset '{name}' not found")
                return False

            preset = self.presets[name]

            # Remove file
            if preset.file_path and os.path.exists(preset.file_path):
                os.remove(preset.file_path)

            # Remove from internal storage
            del self.presets[name]

            print(f"Preset '{name}' deleted successfully")
            return True

        except Exception as e:
            print(f"Error deleting preset '{name}': {e}")
            return False

    def export_preset(self, name: str, export_path: str) -> bool:
        """Export a preset to a specific file path (.kviz format)"""
        try:
            if name not in self.presets:
                print(f"Preset '{name}' not found")
                return False

            preset = self.presets[name]
            export_path_obj = Path(export_path)

            # Ensure .kviz extension
            if export_path_obj.suffix.lower() != KVIZ_EXTENSION:
                export_path_obj = export_path_obj.with_suffix(KVIZ_EXTENSION)

            # Export in .kviz format
            success = self._save_kviz_format(preset, export_path_obj)

            if success:
                file_size = export_path_obj.stat().st_size
                print(f"Preset '{name}' exported to {export_path_obj} ({file_size} bytes)")
                return True
            else:
                print(f"Failed to export preset '{name}' to {export_path_obj}")
                return False

        except Exception as e:
            print(f"Error exporting preset '{name}': {e}")
            return False

    def import_preset(self, import_path: str) -> bool:
        """Import a preset from a .kviz file"""
        try:
            import_path_obj = Path(import_path)

            # Only support .kviz format
            if import_path_obj.suffix.lower() != KVIZ_EXTENSION:
                print(f"Only .kviz preset files are supported. Got: {import_path_obj.suffix}")
                return False

            # Load .kviz format
            preset_info = self._load_kviz_format(import_path_obj)
            if not preset_info:
                print(f"Failed to load .kviz preset from {import_path}")
                return False

            # Save to presets directory in .kviz format
            safe_name = self._sanitize_filename(preset_info.name)
            new_filepath = self.presets_dir / f"{safe_name}{KVIZ_EXTENSION}"
            preset_info.file_path = str(new_filepath)

            # Save in .kviz format
            success = self._save_kviz_format(preset_info, new_filepath)

            if success:
                # Add to internal storage
                self.presets[preset_info.name] = preset_info

                file_size = new_filepath.stat().st_size
                print(f"Preset '{preset_info.name}' imported successfully ({file_size} bytes)")
                return True
            else:
                print(f"Failed to save imported preset '{preset_info.name}'")
                return False

        except Exception as e:
            print(f"Error importing preset from {import_path}: {e}")
            return False

    def _force_visual_refresh(self, visualizer):
        """Force a complete visual refresh after applying preset"""
        try:
            logger.debug(f"Starting forced visual refresh...")
            logger.debug(f"Current program: {getattr(visualizer, 'program', 'NOT_SET')}")
            logger.debug(f"Current warp map: {getattr(visualizer, 'active_warp_map_name', 'NOT_SET')}")
            logger.debug(f"Current waveform: {getattr(visualizer, 'current_waveform_name', 'NOT_SET')}")
            # Force shader recompilation if needed using threaded compiler
            if hasattr(visualizer, 'threaded_shader_compiler') and visualizer.threaded_shader_compiler:
                try:
                    # Trigger a complete shader recompilation with current settings
                    warp_maps = []
                    if hasattr(visualizer, 'active_warp_map_name') and visualizer.active_warp_map_name:
                        warp_maps = [visualizer.active_warp_map_name]

                    request_id = visualizer.threaded_shader_compiler.compile_async(
                        warp_maps, 
                        visualizer.shader_manager, 
                        getattr(visualizer, 'current_waveform_name', 'default'),
                        priority=10  # High priority for preset loading
                    )
                    print(f"   Submitted forced shader recompilation request (ID: {request_id[:8]}...)")

                    # Store request ID for later retrieval if needed
                    if not hasattr(visualizer, '_preset_shader_requests'):
                        visualizer._preset_shader_requests = []
                    visualizer._preset_shader_requests.append(request_id)

                except Exception as shader_error:
                    print(f"   Warning: Could not submit forced shader recompilation: {shader_error}")
                    # Fallback to regular shader compiler
                    if hasattr(visualizer, 'shader_compiler') and visualizer.shader_compiler:
                        try:
                            if hasattr(visualizer.shader_compiler, 'compile_main_shader_with_warp'):
                                warp_maps = []
                                if hasattr(visualizer, 'active_warp_map_name') and visualizer.active_warp_map_name:
                                    warp_maps = [visualizer.active_warp_map_name]

                                visualizer.shader_compiler.compile_main_shader_with_warp(
                                    warp_maps, 
                                    visualizer.shader_manager, 
                                    getattr(visualizer, 'current_waveform_name', 'default')
                                )
                                print(f"   Fallback: Used regular shader compiler for forced recompilation")
                        except Exception as fallback_error:
                            print(f"   Fallback forced shader compilation also failed: {fallback_error}")
            elif hasattr(visualizer, 'shader_compiler') and visualizer.shader_compiler:
                try:
                    # Trigger a complete shader recompilation with current settings
                    if hasattr(visualizer.shader_compiler, 'compile_main_shader_with_warp'):
                        warp_maps = []
                        if hasattr(visualizer, 'active_warp_map_name') and visualizer.active_warp_map_name:
                            warp_maps = [visualizer.active_warp_map_name]

                        visualizer.shader_compiler.compile_main_shader_with_warp(
                            warp_maps, 
                            visualizer.shader_manager, 
                            getattr(visualizer, 'current_waveform_name', 'default')
                        )
                        print(f"   Forced shader recompilation")
                except Exception as shader_error:
                    print(f"   Warning: Could not force shader recompilation: {shader_error}")

            # Force palette texture update
            if hasattr(visualizer, 'update_palette_texture'):
                try:
                    visualizer.update_palette_texture()
                    print(f"   Forced palette texture update")
                except Exception as palette_error:
                    print(f"   Warning: Could not update palette texture: {palette_error}")

            # Force waveform update if needed
            if hasattr(visualizer, 'current_waveform_name') and hasattr(visualizer, 'waveform_manager'):
                try:
                    if visualizer.current_waveform_name in visualizer.waveform_manager.waveforms:
                        # Force waveform recompilation
                        if hasattr(visualizer, 'select_waveform'):
                            visualizer.select_waveform(visualizer.current_waveform_name)
                        print(f"   Forced waveform update")
                except Exception as waveform_error:
                    print(f"   Warning: Could not update waveform: {waveform_error}")

            # Update all shader uniforms with current values
            if hasattr(visualizer, 'program') and visualizer.program:
                try:
                    # Common uniforms that need updating
                    uniform_updates = {
                        'brightness': getattr(visualizer, 'brightness', 1.0),
                        'contrast': getattr(visualizer, 'contrast', 1.0),
                        'saturation': getattr(visualizer, 'saturation', 1.0),
                        'hue_shift': getattr(visualizer, 'hue_shift', 0.0),
                        'pulse_intensity': getattr(visualizer, 'pulse_intensity', 1.0),
                        'trail_intensity': getattr(visualizer, 'trail_intensity', 0.5),
                        'glow_intensity': getattr(visualizer, 'glow_intensity', 0.5),
                        'glow_radius': getattr(visualizer, 'glow_radius', 2.0),
                        'smoke_intensity': getattr(visualizer, 'smoke_intensity', 0.0),
                        'waveform_scale': getattr(visualizer, 'waveform_scale', 1.0),
                        'waveform_enabled': getattr(visualizer, 'gpu_waveform_enabled', True),  # CRITICAL: Map gpu_waveform_enabled to waveform_enabled uniform
                        'waveform_length': getattr(visualizer, 'waveform_length', 1024),
                        'rotation_speed': getattr(visualizer, 'rotation_speed', 1.0),
                        'rotation_amplitude': getattr(visualizer, 'rotation_amplitude', 1.0),
                        'warp_intensity': getattr(visualizer, 'warp_intensity', 1.0),
                        'symmetry_mode': getattr(visualizer, 'symmetry_mode', 0),
                        'rotation_mode': getattr(visualizer, 'rotation_mode', 0)
                    }

                    updated_count = 0
                    for uniform_name, value in uniform_updates.items():
                        try:
                            if uniform_name in visualizer.program:
                                visualizer.program[uniform_name].value = value
                                if uniform_name == 'waveform_enabled':
                                    print(f"   🔍 [DEBUG] ⭐ CRITICAL: Updated shader uniform {uniform_name} = {value}")
                                updated_count += 1
                        except Exception:
                            pass  # Some uniforms might not exist in current shader

                    if updated_count > 0:
                        print(f"   Updated {updated_count} shader uniforms")

                except Exception as uniform_error:
                    print(f"   Warning: Could not update shader uniforms: {uniform_error}")

            # Force immediate visual update by clearing any cached states
            if hasattr(visualizer, 'force_update'):
                try:
                    visualizer.force_update()
                except Exception:
                    pass

            print(f"   ✨ Forced complete visual refresh")

        except Exception as e:
            print(f"   Error during visual refresh: {e}")

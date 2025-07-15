"""
Warp Map Manager for KarmaViz

This module handles loading, saving, and managing warp maps for the visualizer.
Warp maps are stored as GLSL files with accompanying JSON metadata in the warp_maps/ directory.
"""

import json
import os
import re
import struct
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from modules.input_sanitizer import input_sanitizer


@dataclass
class WarpMapInfo:
    """Information about a warp map"""
    name: str
    category: str  # "basic", "geometric", "organic", "experimental", etc.
    description: str
    glsl_code: str
    complexity: str  # "low", "medium", "high"
    author: str = "Unknown"
    version: str = "1.0"
    is_builtin: bool = True
    file_path: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization (legacy support)"""
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "complexity": self.complexity,
            "author": self.author,
            "version": self.version,
            "is_builtin": self.is_builtin,
            "glsl_code": self.glsl_code,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'WarpMapInfo':
        """Create from dictionary (legacy JSON support)"""
        return cls(
            name=data.get("name", "Unknown"),
            category=data.get("category", "basic"),
            description=data.get("description", ""),
            complexity=data.get("complexity", "medium"),
            author=data.get("author", "Unknown"),
            version=data.get("version", "1.0"),
            is_builtin=data.get("is_builtin", True),
            glsl_code=data.get("glsl_code", ""),
        )

    def to_binary(self) -> bytes:
        """Convert to binary format"""
        # Binary format specification:
        # - Magic header: b'KVWM' (4 bytes)
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
        data.extend(b"KVWM")
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
    def from_binary(cls, data: bytes) -> "WarpMapInfo":
        """Create from binary format"""
        if len(data) < 8:
            raise ValueError("Invalid binary data: too short")

        # Check magic header
        if data[:4] != b"KVWM":
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
            print(f"Sanitization warnings for loaded warp map: {warnings}")
        if errors:
            print(f"Sanitization errors for loaded warp map: {errors}")
            # For critical errors, use safe defaults
            if 'name' in [e.split(':')[0] for e in errors]:
                sanitized_metadata['name'] = "invalid_warp_map"
            if 'glsl_code' in [e.split(':')[0] for e in errors]:
                sanitized_metadata['glsl_code'] = "// Invalid GLSL code was sanitized\nvec2 warp(vec2 uv) { return uv; }"

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


class WarpMapManager:
    """Manages warp maps for the visualizer"""

    def __init__(self, warp_maps_dir: str = "warp_maps"):
        self.warp_maps_dir = Path(warp_maps_dir)
        self.warp_maps: Dict[str, WarpMapInfo] = {}
        self.categories: Dict[str, List[str]] = {}

        # Create warp maps directory if it doesn't exist
        self.warp_maps_dir.mkdir(exist_ok=True)

        # Load all warp maps
        self.load_all_warp_maps()

    def load_all_warp_maps(self):
        """Load all warp maps from the warp_maps directory"""
        self.warp_maps.clear()
        self.categories.clear()

        if not self.warp_maps_dir.exists():
            print(f"Warp maps directory {self.warp_maps_dir} not found, creating...")
            self.warp_maps_dir.mkdir(exist_ok=True)
            self._create_default_warp_maps()
            return

        # First, load binary warp maps (.kvwm files) recursively
        for file in self.warp_maps_dir.rglob("*.kvwm"):
            filename_key = file.stem
            self.warp_maps[filename_key] = file

            # Load warp map info from binary format
            try:
                with open(file, "rb") as f:
                    data = f.read()
                    warp_map_info = WarpMapInfo.from_binary(data)
                    warp_map_info.name = filename_key  # Ensure name matches filename
                    warp_map_info.file_path = str(file)
                    self.warp_maps[filename_key] = warp_map_info

                    # Organize by category using filename key
                    if warp_map_info.category not in self.categories:
                        self.categories[warp_map_info.category] = []
                    self.categories[warp_map_info.category].append(filename_key)
            except Exception as e:
                print(f"Error loading binary warp map {filename_key}: {e}")

        # Then, load legacy formats (only if no binary version exists)
        # Scan for .glsl files (old format)
        for glsl_file in self.warp_maps_dir.rglob("*.glsl"):
            filename_key = glsl_file.stem
            if filename_key not in self.warp_maps:  # Only load if binary version doesn't exist
                try:
                    warp_map = self._load_warp_map_old_format(glsl_file)
                    if warp_map:
                        self.warp_maps[filename_key] = warp_map

                        # Organize by category using filename key
                        if warp_map.category not in self.categories:
                            self.categories[warp_map.category] = []
                        self.categories[warp_map.category].append(filename_key)

                except Exception as e:
                    print(f"Error loading warp map {glsl_file}: {e}")

        # Scan for .json files (new self-contained format)
        for json_file in self.warp_maps_dir.rglob("*.json"):
            filename_key = json_file.stem
            # Skip if there's a corresponding .glsl file (old format) or binary version exists
            glsl_file = json_file.with_suffix('.glsl')
            if glsl_file.exists() or filename_key in self.warp_maps:
                continue

            try:
                warp_map = self._load_warp_map_new_format(json_file)
                if warp_map:
                    self.warp_maps[filename_key] = warp_map

                    # Organize by category using filename key
                    if warp_map.category not in self.categories:
                        self.categories[warp_map.category] = []
                    self.categories[warp_map.category].append(filename_key)

            except Exception as e:
                print(f"Error loading warp map {json_file}: {e}")

        print(f"Loaded {len(self.warp_maps)} warp maps in {len(self.categories)} categories")
        
    def _load_warp_map_old_format(self, glsl_file: Path) -> Optional[WarpMapInfo]:
        """Load a single warp map from GLSL and JSON files (old format)"""
        json_file = glsl_file.with_suffix('.json')

        # Read GLSL code
        try:
            with open(glsl_file, 'r', encoding='utf-8') as f:
                glsl_code = f.read()
        except Exception as e:
            print(f"Error reading GLSL file {glsl_file}: {e}")
            return None

        # Read metadata
        metadata = {}
        if json_file.exists():
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except Exception as e:
                print(f"Error reading metadata file {json_file}: {e}")

        # Extract name from filename if not in metadata
        name = metadata.get('name', glsl_file.stem)

        # Determine category from directory structure
        relative_path = glsl_file.relative_to(self.warp_maps_dir)
        category = str(relative_path.parent) if relative_path.parent != Path('.') else 'uncategorized'

        return WarpMapInfo(
            name=name,
            category=metadata.get('category', category),
            description=metadata.get('description', f"Warp map: {name}"),
            glsl_code=glsl_code,
            complexity=metadata.get('complexity', 'medium'),
            author=metadata.get('author', 'Unknown'),
            version=metadata.get('version', '1.0'),
            is_builtin=metadata.get('is_builtin', True),
            file_path=str(glsl_file)
        )

    def _load_warp_map_new_format(self, json_file: Path) -> Optional[WarpMapInfo]:
        """Load a single warp map from self-contained JSON file (new format)"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading JSON file {json_file}: {e}")
            return None

        # Validate required fields
        if 'glsl_code' not in data:
            print(f"JSON file {json_file} missing required 'glsl_code' field")
            return None

        # Extract name from filename if not in data
        name = data.get('name', json_file.stem)

        # Determine category from directory structure
        relative_path = json_file.relative_to(self.warp_maps_dir)
        category = str(relative_path.parent) if relative_path.parent != Path('.') else 'uncategorized'

        return WarpMapInfo(
            name=name,
            category=data.get('category', category),
            description=data.get('description', f"Warp map: {name}"),
            glsl_code=data.get('glsl_code', ''),
            complexity=data.get('complexity', 'medium'),
            author=data.get('author', 'Unknown'),
            version=data.get('version', '1.0'),
            is_builtin=data.get('is_builtin', True),
            file_path=str(json_file)
        )

    def get_warp_map(self, name: str) -> Optional[WarpMapInfo]:
        """Get a warp map by name"""
        return self.warp_maps.get(name)

    def get_all_warp_maps(self) -> List[WarpMapInfo]:
        """Get all loaded warp maps"""
        return list(self.warp_maps.values())

    def get_warp_maps_by_category(self, category: str) -> List[WarpMapInfo]:
        """Get all warp maps in a specific category"""
        if category not in self.categories:
            return []
        return [self.warp_maps[filename_key] for filename_key in self.categories[category]]

    def get_categories(self) -> List[str]:
        """Get all available categories"""
        return list(self.categories.keys())

    def get_warp_map_key(self, warp_map: WarpMapInfo) -> Optional[str]:
        """Get the filename key for a warp map"""
        for key, stored_warp_map in self.warp_maps.items():
            if stored_warp_map == warp_map:
                return key
        return None

    def get_warp_map_key_by_name(self, name: str) -> Optional[str]:
        """Get the filename key for a warp map by its display name"""
        for key, warp_map in self.warp_maps.items():
            if warp_map.name == name:
                return key
        return None

    def save_warp_map(self, warp_map: WarpMapInfo, overwrite: bool = False, use_binary_format: bool = True, subdirectory: str = None) -> bool:
        """Save a warp map to disk in binary format
        
        Args:
            warp_map: The warp map information to save
            overwrite: Whether to overwrite existing warp maps
            use_binary_format: Whether to use binary format (.kvwm) or legacy formats
            subdirectory: Optional subdirectory to save the warp map in (e.g., 'basic', 'experimental')
        """
        try:
            # Check if warp map already exists and we're not overwriting
            if not overwrite and warp_map.name in self.warp_maps:
                return False

            # Create the file path for binary format
            if subdirectory:
                # Ensure subdirectory exists
                subdir_path = self.warp_maps_dir / subdirectory
                subdir_path.mkdir(parents=True, exist_ok=True)
                category_dir = subdir_path
            else:
                # Create category directory if needed
                category_dir = self.warp_maps_dir / warp_map.category.lower()
                category_dir.mkdir(exist_ok=True)

            # Use lowercase filename
            filename = warp_map.name.lower().replace(' ', '_').replace('-', '_')

            if use_binary_format:
                # Binary format: .kvwm file
                kvwm_file = category_dir / f"{filename}.kvwm"
                old_json_path = category_dir / f"{filename}.json"
                old_glsl_path = category_dir / f"{filename}.glsl"

                # Check if file exists
                if kvwm_file.exists() and not overwrite:
                    print(f"Warp map {warp_map.name} already exists. Use overwrite=True to replace.")
                    return False

                # Save to binary file
                with open(kvwm_file, "wb") as f:
                    f.write(warp_map.to_binary())

                # Remove old JSON/GLSL files if they exist
                for old_path in [old_json_path, old_glsl_path]:
                    if old_path.exists():
                        try:
                            old_path.unlink()
                            print(f"Removed legacy file: {old_path}")
                        except Exception as e:
                            print(f"Warning: Could not remove old file {old_path}: {e}")

                # Update internal storage
                warp_map.file_path = str(kvwm_file)

            else:
                # Legacy formats
                if True:  # JSON format (new legacy format)
                    # New format: single JSON file with embedded GLSL
                    json_file = category_dir / f"{filename}.json"

                    # Check if file exists
                    if json_file.exists() and not overwrite:
                        print(f"Warp map {warp_map.name} already exists. Use overwrite=True to replace.")
                        return False

                    # Save self-contained JSON
                    data = warp_map.to_dict()

                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2)

                    # Update internal storage
                    warp_map.file_path = str(json_file)

                else:
                    # Old format: separate GLSL and JSON files
                    glsl_file = category_dir / f"{filename}.glsl"
                    json_file = category_dir / f"{filename}.json"

                    # Check if files exist
                    if (glsl_file.exists() or json_file.exists()) and not overwrite:
                        print(f"Warp map {warp_map.name} already exists. Use overwrite=True to replace.")
                        return False

                    # Save GLSL code
                    with open(glsl_file, 'w', encoding='utf-8') as f:
                        f.write(warp_map.glsl_code)

                    # Save metadata
                    metadata = {
                        'name': warp_map.name,
                        'category': warp_map.category,
                        'description': warp_map.description,
                        'complexity': warp_map.complexity,
                        'author': warp_map.author,
                        'version': warp_map.version,
                        'is_builtin': warp_map.is_builtin
                    }

                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2)

                    # Update internal storage
                    warp_map.file_path = str(glsl_file)

            # Use filename as key instead of display name
            filename_key = filename  # filename is already calculated above
            self.warp_maps[filename_key] = warp_map

            # Update categories using filename key
            if warp_map.category not in self.categories:
                self.categories[warp_map.category] = []
            if filename_key not in self.categories[warp_map.category]:
                self.categories[warp_map.category].append(filename_key)

            return True

        except Exception as e:
            print(f"Error saving warp map {warp_map.name}: {e}")
            return False

    def delete_warp_map(self, name_or_key: str) -> bool:
        """Delete a warp map by name or filename key"""
        # First try to find by filename key (direct lookup)
        warp_map = None
        key_to_delete = None
        
        if name_or_key in self.warp_maps:
            # Direct key lookup
            warp_map = self.warp_maps[name_or_key]
            key_to_delete = name_or_key
        else:
            # Try to find by display name
            for key, stored_warp_map in self.warp_maps.items():
                if stored_warp_map.name == name_or_key:
                    warp_map = stored_warp_map
                    key_to_delete = key
                    break
        
        if not warp_map or not key_to_delete:
            print(f"Warp map '{name_or_key}' not found")
            return False

        # Don't delete built-in warp maps
        if warp_map.is_builtin:
            print(f"Cannot delete built-in warp map '{warp_map.name}'")
            return False

        try:
            # Delete files
            file_path = Path(warp_map.file_path)
            
            if file_path.suffix == '.kvwm':
                # Binary format - delete .kvwm file
                if file_path.exists():
                    file_path.unlink()
                    print(f"Deleted binary warp map file: {file_path}")
            elif file_path.suffix == '.glsl':
                # Old format - delete both .glsl and .json files
                json_file = file_path.with_suffix('.json')
                if file_path.exists():
                    file_path.unlink()
                if json_file.exists():
                    json_file.unlink()
                print(f"Deleted old format warp map files: {file_path}, {json_file}")
            elif file_path.suffix == '.json':
                # New JSON format - delete .json file
                if file_path.exists():
                    file_path.unlink()
                print(f"Deleted JSON warp map file: {file_path}")

            # Also check for legacy files in the same directory as the main file
            if file_path.suffix == '.kvwm':
                # Check for old JSON/GLSL files with same name
                json_path = file_path.with_suffix('.json')
                glsl_path = file_path.with_suffix('.glsl')
                for legacy_path in [json_path, glsl_path]:
                    if legacy_path.exists():
                        legacy_path.unlink()
                        print(f"Deleted legacy file: {legacy_path}")

            # Remove from internal storage using the correct key
            del self.warp_maps[key_to_delete]

            # Remove from categories using filename key
            if warp_map.category in self.categories:
                if key_to_delete in self.categories[warp_map.category]:
                    self.categories[warp_map.category].remove(key_to_delete)
                # Remove empty categories
                if not self.categories[warp_map.category]:
                    del self.categories[warp_map.category]

            print(f"Successfully deleted warp map '{warp_map.name}' (key: {key_to_delete})")
            return True

        except Exception as e:
            print(f"Error deleting warp map {name_or_key}: {e}")
            return False

    def _create_default_warp_maps(self):
        """Create some default warp maps if none exist"""
        print("Creating default warp maps...")

        # We'll implement this after creating the directory structure
        pass

    def validate_glsl_code(self, glsl_code: str) -> Tuple[bool, str]:
        """Validate GLSL code for basic syntax and required function signature"""
        # Check for required function signature
        pattern = r'vec2\s+get_pattern\d*\s*\(\s*vec2\s+pos\s*,\s*float\s+t\s*\)'
        if not re.search(pattern, glsl_code):
            return False, "GLSL code must contain a function with signature: vec2 get_pattern(vec2 pos, float t)"

        # Basic syntax checks
        if glsl_code.count('{') != glsl_code.count('}'):
            return False, "Mismatched braces in GLSL code"

        if glsl_code.count('(') != glsl_code.count(')'):
            return False, "Mismatched parentheses in GLSL code"

        return True, "GLSL code appears valid"

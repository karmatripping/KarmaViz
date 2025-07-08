"""
Palette Manager for KarmaViz

This module handles loading, saving, and managing color palettes for the visualizer.
Palettes are stored as JSON files in the palettes/ directory.
"""

import json
import os
from typing import List, Tuple, Dict, Optional
from random import choice, choices
from dataclasses import dataclass
from modules.benchmark import benchmark


@dataclass
class PaletteInfo:
    """Information about a palette"""
    name: str
    energy_level: str  # "high", "low", "moderate"
    warmth: str  # "warm", "cool", "neutral"
    colors: List[Tuple[int, int, int]]
    description: str = ""
    is_builtin: bool = True

    @property
    def category(self) -> str:
        """Derive category from energy_level and warmth"""
        if self.energy_level == "high" and self.warmth == "warm":
            return "High Energy Warm"
        elif self.energy_level == "high" and self.warmth == "cool":
            return "High Energy Cool"
        elif self.energy_level == "low" and self.warmth == "warm":
            return "Low Energy Warm"
        elif self.energy_level == "low" and self.warmth == "cool":
            return "Low Energy Cool"
        elif self.energy_level == "moderate" and self.warmth == "warm":
            return "Moderate Energy Warm"
        elif self.energy_level == "moderate" and self.warmth == "cool":
            return "Moderate Energy Cool"
        else:
            return "Special"  # For neutral warmth or other combinations


class PaletteManager:
    """Manages color palettes for KarmaViz"""

    def __init__(self, palettes_dir: str = "palettes"):
        self.palettes_dir = palettes_dir
        self.palettes: Dict[str, PaletteInfo] = {}
        self.categories = {
            "high_energy_warm": [],
            "high_energy_cool": [],
            "low_energy_warm": [],
            "low_energy_cool": [],
            "moderate_energy_warm": [],
            "moderate_energy_cool": [],
            "special": []  # For rainbow and other special palettes
        }

        # Ensure palettes directory exists
        os.makedirs(self.palettes_dir, exist_ok=True)

        # Load all palettes
        self._create_builtin_palette_files()
        self._load_all_palettes()

    def _create_builtin_palette_files(self):
        """Create JSON files for built-in palettes if they don't exist"""
        # The palette files should already be created by extract_palettes.py
        # This method is now just a placeholder for future built-in palette creation
        pass

    def _get_builtin_palette_names(self) -> List[str]:
        """Returns the built-in palette names for checking if a palette is built-in"""
        # Return the new user-friendly names
        return [
            "Bold Red Orange Yellow", "Hot Pink Orange Magenta", "Fiery Orange Yellow",
            "Rainbow", "Sunset Red Orange", "Volcanic Red Deep Orange", "Cyan Blue Green",
            "Azure Mint Purple", "Electric Blue Cyan", "Neon Lime Aqua Purple",
            "Bright Arctic Blue", "Vibrant Earth Tones", "Saturated Pink Coral",
            "Vibrant Desert", "Rich Earth", "Saturated Clay", "Deep Blue Slate",
            "Rich Sea Tones", "Rich Night Sky", "Rich Water", "Rich Storm Blue",
            "Vibrant Coral", "Rich Violet Pink", "Rich Autumn", "Rich Rose Gold",
            "Rich Terra", "Rich Forest", "Rich Turquoise", "Rich Ocean",
            "Rich Lavender", "Rich Forest Green"
        ]

    def _load_all_palettes(self):
        """Load all palettes from JSON files"""
        if not os.path.exists(self.palettes_dir):
            return

        for filename in os.listdir(self.palettes_dir):
            if filename.endswith('.json'):
                try:
                    filepath = os.path.join(self.palettes_dir, filename)
                    with open(filepath, 'r') as f:
                        palette_data = json.load(f)

                    # Convert colors from list to tuples
                    palette_data['colors'] = [tuple(color) for color in palette_data['colors']]

                    # Set is_builtin based on whether it's in our builtin list
                    builtin_names = self._get_builtin_palette_names()
                    palette_data['is_builtin'] = palette_data['name'] in builtin_names

                    palette_info = PaletteInfo(**palette_data)
                    self.palettes[palette_info.name] = palette_info

                    # Categorize using derived category
                    category_key = f"{palette_info.energy_level}_energy_{palette_info.warmth}"
                    if category_key in self.categories:
                        self.categories[category_key].append(palette_info.name)
                    else:
                        self.categories["special"].append(palette_info.name)

                except Exception as e:
                    print(f"Error loading palette {filename}: {e}")

    def get_palette_by_name(self, name: str) -> Optional[PaletteInfo]:
        """Get a palette by name"""
        return self.palettes.get(name)

    def get_palettes_by_category(self, category: str) -> List[PaletteInfo]:
        """Get all palettes in a category"""
        if category in self.categories:
            return [self.palettes[name] for name in self.categories[category] if name in self.palettes]
        return []

    def get_all_palettes(self) -> List[PaletteInfo]:
        """Get all available palettes"""
        return list(self.palettes.values())

    @benchmark("get_mood_palette")
    def get_mood_palette(self, mood: Dict[str, float]) -> List[Tuple[int, int, int]]:
        """Get a palette based on mood analysis (maintains original logic)"""
        # Convert all palettes to the original format for mood selection
        all_palette_colors = []
        palette_names = []

        # Build lists in the original order for compatibility
        for category in ["high_energy_warm", "high_energy_cool", "low_energy_warm",
                        "low_energy_cool", "moderate_energy_warm", "moderate_energy_cool"]:
            for palette_name in self.categories.get(category, []):
                if palette_name in self.palettes:
                    all_palette_colors.append(self.palettes[palette_name].colors)
                    palette_names.append(palette_name)

        # Add special palettes (like rainbow)
        for palette_name in self.categories.get("special", []):
            if palette_name in self.palettes:
                all_palette_colors.append(self.palettes[palette_name].colors)
                palette_names.append(palette_name)

        if not all_palette_colors:
            # Fallback to a default palette
            return [(255, 0, 0), (255, 100, 0), (255, 255, 0), (0, 255, 0),
                   (0, 100, 255), (150, 0, 255), (255, 50, 0)]

        # Use original mood-based selection logic
        weights = self._calculate_mood_weights(mood, len(all_palette_colors))
        chosen_palette_colors = choices(all_palette_colors, weights=weights, k=1)[0]

        return chosen_palette_colors

    def _calculate_mood_weights(self, mood: Dict[str, float], num_palettes: int) -> List[float]:
        """Calculate weights for palette selection based on mood (original logic)"""
        weights = []
        palettes_per_category = 5  # Original logic assumption

        for i in range(num_palettes):
            weight = 1.0

            # Special handling for rainbow palette (assuming it's at index 3 in original)
            if i == 3 and num_palettes > 3:  # Rainbow palette
                weight = 1.5
                weight *= 1.0 + mood["energy"]
                weight *= 1.0 + abs(mood["warmth"] - 0.5)
            else:
                # Determine category based on index (original logic)
                category_index = i // palettes_per_category

                # High energy categories
                if category_index < 2:
                    weight *= 1.0 + mood["energy"] * 2.0
                    if category_index == 0:  # Warm high energy
                        weight *= 1.0 + mood["warmth"]
                    else:  # Cool high energy
                        weight *= 2.0 - mood["warmth"]

                # Low energy categories
                elif category_index < 4:
                    weight *= 2.0 - mood["energy"]
                    if category_index == 2:  # Warm low energy
                        weight *= 1.0 + mood["warmth"]
                    else:  # Cool low energy
                        weight *= 2.0 - mood["warmth"]

                # Moderate energy categories
                else:
                    weight *= 1.0 + (1.0 - abs(mood["energy"] - 0.5) * 2.0)
                    if category_index == 4:  # Warm moderate
                        weight *= 1.0 + mood["warmth"]
                    else:  # Cool moderate
                        weight *= 2.0 - mood["warmth"]

            weights.append(max(0.1, weight))

        return weights

    def _sanitize_filename(self, name: str) -> str:
        """Convert a palette name to a safe filename"""
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
            safe_name = 'unnamed_palette'

        return safe_name

    def save_palette(self, palette_info: PaletteInfo) -> bool:
        """Save a palette to a JSON file"""
        # Validate color count (2-10 colors)
        if len(palette_info.colors) < 2 or len(palette_info.colors) > 10:
            print(f"Error: Palette must have between 2 and 10 colors. Got {len(palette_info.colors)} colors.")
            return False

        try:
            # Create a safe filename from the palette name
            filename = self._sanitize_filename(palette_info.name) + ".json"
            filepath = os.path.join(self.palettes_dir, filename)

            # Convert to dict for JSON serialization (no category field - it's derived)
            palette_dict = {
                "name": palette_info.name,
                "energy_level": palette_info.energy_level,
                "warmth": palette_info.warmth,
                "colors": list(palette_info.colors),  # Convert tuples to lists
                "description": palette_info.description,
                "is_builtin": False
            }

            with open(filepath, 'w') as f:
                json.dump(palette_dict, f, indent=2)

            # Add to our internal storage
            self.palettes[palette_info.name] = palette_info

            # Categorize
            category_key = f"{palette_info.energy_level}_energy_{palette_info.warmth}"
            if category_key in self.categories:
                if palette_info.name not in self.categories[category_key]:
                    self.categories[category_key].append(palette_info.name)
            else:
                if palette_info.name not in self.categories["special"]:
                    self.categories["special"].append(palette_info.name)

            return True

        except Exception as e:
            print(f"Error saving palette {palette_info.name}: {e}")
            return False

    def delete_palette(self, name: str) -> bool:
        """Delete a custom palette (built-in palettes cannot be deleted)"""
        if name not in self.palettes:
            return False

        palette_info = self.palettes[name]
        if palette_info.is_builtin:
            return False  # Cannot delete built-in palettes

        try:
            # Remove file using the same sanitization as save_palette
            filename = self._sanitize_filename(name) + ".json"
            filepath = os.path.join(self.palettes_dir, filename)
            if os.path.exists(filepath):
                os.remove(filepath)

            # Remove from internal storage
            del self.palettes[name]

            # Remove from categories
            for category_list in self.categories.values():
                if name in category_list:
                    category_list.remove(name)

            return True

        except Exception as e:
            print(f"Error deleting palette {name}: {e}")
            return False

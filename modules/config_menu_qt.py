import os
import json
import re
import subprocess
from typing import List, Callable
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QCheckBox,
    QPushButton,
    QComboBox,
    QScrollArea,
    QGroupBox,
    QDialog,
    QApplication,
    QTabWidget,
    QFrame,
    QColorDialog,
    QLineEdit,
    QTextEdit,
    QMessageBox,
    QListWidget,
    QListWidgetItem,
    QGridLayout,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QStyle,
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QRect, QEvent
from PyQt5.QtGui import QColor, QPainter, QLinearGradient, QBrush, QPen
from PyQt5.QtWidgets import QDesktopWidget
from modules.logging_config import get_logger

logger = get_logger("config_menu")
# Import warp map system (with lazy loading to avoid circular imports)
try:
    from modules.warp_map_manager import WarpMapManager
    from modules.warp_map_editor import WarpMapEditor
    WARP_MAP_AVAILABLE = True
except ImportError:
    WarpMapManager = None
    WarpMapEditor = None
    WARP_MAP_AVAILABLE = False

# Import waveform system (with lazy loading to avoid circular imports)
try:
    from modules.waveform_manager import WaveformManager, WaveformInfo
    from modules.waveform_editor import WaveformEditor
    WAVEFORM_AVAILABLE = True
except ImportError:
    WaveformManager = None
    WaveformInfo = None
    WaveformEditor = None
    WAVEFORM_AVAILABLE = False

def get_available_resolutions() -> List[str]:
    """Get available screen resolutions using xrandr."""
    resolutions = ["Native"]  # Default option
    try:
        result = subprocess.run(["xrandr"], capture_output=True, text=True, check=True)
        output = result.stdout
        # Regex to find lines like '   1920x1080     60.00*+  59.94    50.00  '
        matches = re.findall(r"^\s*(\d+x\d+)\s", output, re.MULTILINE)
        if matches:
            unique_resolutions = sorted(
                list(set(matches)),
                key=lambda res: int(res.split("x")[0]) * int(res.split("x")[1]),
                reverse=True,
            )
            resolutions.extend(unique_resolutions)
    except (FileNotFoundError, subprocess.CalledProcessError, Exception) as e:
        logger.debug(f"Warning: Could not get resolutions via xrandr: {e}. Using default.")
        # Fallback to some common resolutions
        common_resolutions = ["1920x1080", "1280x720", "800x600"]
        for res in common_resolutions:
            if res not in resolutions:
                resolutions.append(res)

    return resolutions


class NonFocusStealingWidget(QWidget):
    """A custom QWidget that doesn't steal focus when shown"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
    def showEvent(self, event):
        """Override show event to prevent focus stealing"""
        # Call parent showEvent but don't activate
        super().showEvent(event)
        
        # Try to restore focus to the previous window immediately
        try:
            import subprocess
            # Get the previously active window and reactivate it
            result = subprocess.run(['xdotool', 'getactivewindow'], 
                                  capture_output=True, text=True, timeout=0.5)
            if result.returncode == 0 and result.stdout.strip():
                prev_window = result.stdout.strip()
                # If the active window is this dialog, find the pygame window
                if prev_window != str(int(self.winId())):
                    subprocess.run(['xdotool', 'windowactivate', prev_window], timeout=0.5)
        except:
            pass
    
    def focusInEvent(self, event):
        """Override focus in event to redirect focus back to pygame"""
        # Accept the event but immediately try to restore pygame focus
        super().focusInEvent(event)
        
        # Try to find and activate pygame window
        try:
            import subprocess
            # Search for pygame window
            result = subprocess.run(['xdotool', 'search', '--name', '.*pygame.*'], 
                                  capture_output=True, text=True, timeout=0.5)
            if result.returncode == 0 and result.stdout.strip():
                pygame_window = result.stdout.strip().split('\n')[0]
                subprocess.run(['xdotool', 'windowactivate', pygame_window], timeout=0.5)
        except:
            pass


class SettingWidget(QWidget):
    """Base class for all setting widgets"""
    valueChanged = pyqtSignal(str, object)  # Setting name, new value

    def __init__(self, name: str, label: str, parent=None):
        super().__init__(parent)
        self.name = name
        self.label_text = label
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.label = QLabel(label)
        self.label.setMinimumWidth(180)
        self.layout.addWidget(self.label)

    def set_value(self, value):
        """Set the widget value"""
        pass

    def get_value(self):
        """Get the widget value"""
        pass


class SliderSetting(SettingWidget):
    """Slider widget for numeric settings"""

    def __init__(self, name: str, label: str, min_val: float, max_val: float,
                 step: float, parent=None):
        super().__init__(name, label, parent)

        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.decimals = len(str(step).split('.')[-1]) if '.' in str(step) else 0
        self.scale = 10 ** self.decimals

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(int(min_val * self.scale))
        self.slider.setMaximum(int(max_val * self.scale))
        self.slider.setSingleStep(int(step * self.scale))
        self.slider.setPageStep(int(step * self.scale * 10))

        self.value_label = QLabel()
        self.value_label.setMinimumWidth(60)
        self.value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.layout.addWidget(self.slider)
        self.layout.addWidget(self.value_label)

        self.slider.valueChanged.connect(self._on_slider_changed)

    def _on_slider_changed(self, value):
        actual_value = value / self.scale
        self.value_label.setText(f"{actual_value:.{self.decimals}f}")
        self.valueChanged.emit(self.name, actual_value)

    def set_value(self, value):
        self.slider.setValue(int(value * self.scale))
        self.value_label.setText(f"{value:.{self.decimals}f}")

    def get_value(self):
        return self.slider.value() / self.scale


class ToggleSetting(SettingWidget):
    """Toggle widget for boolean settings"""

    def __init__(self, name: str, label: str, parent=None):
        super().__init__(name, label, parent)

        self.checkbox = QCheckBox()
        self.layout.addWidget(self.checkbox)
        self.layout.addStretch()

        self.checkbox.stateChanged.connect(self._on_state_changed)

    def _on_state_changed(self, state):
        self.valueChanged.emit(self.name, state == Qt.Checked)

    def set_value(self, value):
        self.checkbox.setChecked(bool(value))

    def get_value(self):
        return self.checkbox.isChecked()


class CycleSetting(SettingWidget):
    """Cycle widget for options settings"""

    def __init__(self, name: str, label: str, options: List, option_labels: List = None, parent=None):
        super().__init__(name, label, parent)

        self.options = options
        self.option_labels = option_labels or [str(option) for option in options]
        self.combo = QComboBox()

        # Use option labels for display if provided, otherwise convert options to strings
        for i, option in enumerate(options):
            display_text = self.option_labels[i] if i < len(self.option_labels) else str(option)
            self.combo.addItem(display_text)

        self.layout.addWidget(self.combo)

        self.combo.currentIndexChanged.connect(self._on_selection_changed)

    def _on_selection_changed(self, index):
        # Return the original type, not just the string representation
        self.valueChanged.emit(self.name, self.options[index])

    def set_value(self, value):
        # Find the index of the value in options
        try:
            index = self.options.index(value)
            self.combo.setCurrentIndex(index)
        except ValueError:
            # If value not in options, try to find it by string representation
            for i, option in enumerate(self.options):
                if str(option) == str(value):
                    self.combo.setCurrentIndex(i)
                    break

    def get_value(self):
        return self.options[self.combo.currentIndex()]


class ColorButton(QPushButton):
    """A button that displays a color and opens a color picker when clicked"""
    colorChanged = pyqtSignal(tuple)  # Emits (r, g, b) tuple

    def __init__(self, color=(255, 255, 255), parent=None):
        super().__init__(parent)
        self.color = color
        self.setFixedSize(40, 30)
        self.update_color_display()
        self.clicked.connect(self.pick_color)

    def update_color_display(self):
        """Update the button's background color"""
        r, g, b = self.color
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: rgb({r}, {g}, {b});
                border: 2px solid #555;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                border: 2px solid #777;
            }}
        """)

    def pick_color(self):
        """Open color picker dialog"""
        color = QColorDialog.getColor(QColor(*self.color), self)
        if color.isValid():
            self.color = (color.red(), color.green(), color.blue())
            self.update_color_display()
            self.colorChanged.emit(self.color)

    def set_color(self, color):
        """Set the color programmatically"""
        self.color = color
        self.update_color_display()


class PaletteColorBarDelegate(QStyledItemDelegate):
    """Custom delegate for rendering palette color bars in list widgets"""

    def __init__(self, palette_manager, parent=None):
        super().__init__(parent)
        self.palette_manager = palette_manager

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index):
        """Paint the palette color bar"""
        painter.save()

        # Get palette name from the model
        palette_name = index.data(Qt.DisplayRole)
        if not palette_name or not self.palette_manager:
            # Fallback to default rendering
            super().paint(painter, option, index)
            painter.restore()
            return

        # Handle special options (Auto, Random)
        if palette_name in ["Auto (Mood-based)", "Random"]:
            # Use default text rendering for special options
            super().paint(painter, option, index)
            painter.restore()
            return

        # Get palette colors
        palette_colors = None
        if palette_name in self.palette_manager.palettes:
            palette_colors = self.palette_manager.palettes[palette_name].colors

        if not palette_colors:
            # Fallback to default rendering if no colors found
            super().paint(painter, option, index)
            painter.restore()
            return

        # Set up drawing area
        rect = option.rect
        color_bar_height = 20
        text_height = 20
        margin = 4

        # Calculate color bar rectangle
        color_bar_rect = QRect(
            rect.left() + margin,
            rect.top() + margin,
            rect.width() - 2 * margin,
            color_bar_height
        )

        # Create gradient with palette colors
        gradient = QLinearGradient(color_bar_rect.left(), 0, color_bar_rect.right(), 0)
        num_colors = len(palette_colors)

        for i, color in enumerate(palette_colors):
            # Calculate position (0.0 to 1.0)
            if num_colors == 1:
                position = 0.0
            else:
                position = i / max(1, num_colors - 1)
                if i == num_colors - 1:  # Ensure last color is at 1.0
                    position = 1.0

            qcolor = QColor(color[0], color[1], color[2])
            gradient.setColorAt(position, qcolor)

        # Draw selection background if selected
        if option.state & QStyle.State_Selected:
            selection_color = option.palette.highlight().color()
            painter.fillRect(rect, selection_color)

        # Draw color bar with rounded corners
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(gradient))
        painter.setPen(QPen(QColor(60, 60, 60), 1))  # Dark border
        painter.drawRoundedRect(color_bar_rect, 3, 3)

        # Draw palette name below color bar
        text_rect = QRect(
            rect.left() + margin,
            rect.top() + margin + color_bar_height + 2,
            rect.width() - 2 * margin,
            text_height
        )

        # Set text color based on selection state
        if option.state & QStyle.State_Selected:
            text_color = option.palette.highlightedText().color()
        else:
            text_color = QColor(220, 220, 220)  # Light gray

        painter.setPen(text_color)
        painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, palette_name)

        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index):
        """Return the size hint for palette items"""
        return QSize(200, 44)  # Width, Height


class PaletteComboBoxDelegate(QStyledItemDelegate):
    """Custom delegate for rendering compact palette color bars in combo boxes"""

    def __init__(self, palette_manager, parent=None):
        super().__init__(parent)
        self.palette_manager = palette_manager

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index):
        """Paint the compact palette color bar for combo box"""
        painter.save()

        # Get palette name from the model
        palette_name = index.data(Qt.DisplayRole)
        if not palette_name or not self.palette_manager:
            # Fallback to default rendering
            super().paint(painter, option, index)
            painter.restore()
            return

        # Handle special options (Auto, Random)
        if palette_name in ["Auto (Mood-based)", "Random"]:
            # Use default text rendering for special options
            super().paint(painter, option, index)
            painter.restore()
            return

        # Get palette colors
        palette_colors = None
        if palette_name in self.palette_manager.palettes:
            palette_colors = self.palette_manager.palettes[palette_name].colors

        if not palette_colors:
            # Fallback to default rendering if no colors found
            super().paint(painter, option, index)
            painter.restore()
            return

        # Set up drawing area for compact display
        rect = option.rect
        color_bar_width = 80
        color_bar_height = 16
        margin = 2

        # Calculate color bar rectangle (left side)
        color_bar_rect = QRect(
            rect.left() + margin,
            rect.top() + (rect.height() - color_bar_height) // 2,
            color_bar_width,
            color_bar_height
        )

        # Create gradient with palette colors
        gradient = QLinearGradient(color_bar_rect.left(), 0, color_bar_rect.right(), 0)
        num_colors = len(palette_colors)

        for i, color in enumerate(palette_colors):
            # Calculate position (0.0 to 1.0)
            if num_colors == 1:
                position = 0.0
            else:
                position = i / max(1, num_colors - 1)
                if i == num_colors - 1:  # Ensure last color is at 1.0
                    position = 1.0

            qcolor = QColor(color[0], color[1], color[2])
            gradient.setColorAt(position, qcolor)

        # Draw selection background if selected
        if option.state & QStyle.State_Selected:
            selection_color = option.palette.highlight().color()
            painter.fillRect(rect, selection_color)

        # Draw compact color bar with rounded corners
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(gradient))
        painter.setPen(QPen(QColor(60, 60, 60), 1))  # Dark border
        painter.drawRoundedRect(color_bar_rect, 2, 2)

        # Draw palette name to the right of color bar
        text_rect = QRect(
            color_bar_rect.right() + margin * 2,
            rect.top(),
            rect.width() - color_bar_width - margin * 4,
            rect.height()
        )

        # Set text color based on selection state
        if option.state & QStyle.State_Selected:
            text_color = option.palette.highlightedText().color()
        else:
            text_color = QColor(220, 220, 220)  # Light gray

        painter.setPen(text_color)
        painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, palette_name)

        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index):
        """Return the size hint for compact palette items"""
        return QSize(200, 24)  # Width, Height


class PaletteCycleSetting(CycleSetting):
    """Specialized cycle widget for palette selection with color bar display"""

    def __init__(self, name: str, label: str, options: List, palette_manager, parent=None):
        # Initialize the base CycleSetting first
        super().__init__(name, label, options, parent)

        # Store palette manager reference
        self.palette_manager = palette_manager

        # Create and set the custom delegate for color bars
        if self.palette_manager:
            self.delegate = PaletteComboBoxDelegate(self.palette_manager)
            self.combo.setItemDelegate(self.delegate)


class PaletteEditor(QWidget):
    """Widget for editing color palettes"""
    paletteChanged = pyqtSignal()  # Emitted when palette is modified

    def __init__(self, palette_manager, parent=None):
        super().__init__(parent)
        self.palette_manager = palette_manager
        self.current_palette = None
        self.color_buttons = []
        self.preview_callback = None  # Callback for immediate palette preview
        self.original_palette_mode = None  # Store original mode for restoration
        self.setup_ui()
        self.refresh_palette_list()

    def setup_ui(self):
        """Set up the palette editor UI"""
        layout = QHBoxLayout(self)

        # Left side: Palette list and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Palette list
        left_layout.addWidget(QLabel("Palettes:"))
        self.palette_list = QListWidget()
        self.palette_list.currentItemChanged.connect(self.on_palette_selected)

        # Set up color bar delegate for visual palette display
        if self.palette_manager:
            self.color_bar_delegate = PaletteColorBarDelegate(self.palette_manager)
            self.palette_list.setItemDelegate(self.color_bar_delegate)

        left_layout.addWidget(self.palette_list)

        # Palette controls
        controls_layout = QHBoxLayout()
        self.new_button = QPushButton("New")
        self.new_button.clicked.connect(self.create_new_palette)
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self.delete_palette)
        self.delete_button.setEnabled(False)

        controls_layout.addWidget(self.new_button)
        controls_layout.addWidget(self.delete_button)
        left_layout.addLayout(controls_layout)

        left_panel.setMaximumWidth(300)
        layout.addWidget(left_panel)

        # Right side: Palette editor
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Palette info
        info_group = QGroupBox("Palette Information")
        info_layout = QGridLayout(info_group)

        info_layout.addWidget(QLabel("Name:"), 0, 0)
        self.name_edit = QLineEdit()
        self.name_edit.setFocusPolicy(Qt.ClickFocus)  # Only focus on click, not on show
        self.name_edit.textChanged.connect(self.on_palette_modified)
        info_layout.addWidget(self.name_edit, 0, 1)

        info_layout.addWidget(QLabel("Category:"), 1, 0)
        self.category_combo = QComboBox()
        self.category_combo.addItems(["High Energy Warm", "High Energy Cool",
                                     "Low Energy Warm", "Low Energy Cool",
                                     "Moderate Energy Warm", "Moderate Energy Cool", "Special"])
        self.category_combo.currentTextChanged.connect(self.on_palette_modified)
        info_layout.addWidget(self.category_combo, 1, 1)

        info_layout.addWidget(QLabel("Energy Level:"), 2, 0)
        self.energy_combo = QComboBox()
        self.energy_combo.addItems(["high", "moderate", "low"])
        self.energy_combo.currentTextChanged.connect(self.on_palette_modified)
        info_layout.addWidget(self.energy_combo, 2, 1)

        info_layout.addWidget(QLabel("Warmth:"), 3, 0)
        self.warmth_combo = QComboBox()
        self.warmth_combo.addItems(["warm", "cool", "neutral"])
        self.warmth_combo.currentTextChanged.connect(self.on_palette_modified)
        info_layout.addWidget(self.warmth_combo, 3, 1)

        info_layout.addWidget(QLabel("Description:"), 4, 0)
        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(60)
        self.description_edit.setFocusPolicy(Qt.ClickFocus)  # Only focus on click, not on show
        self.description_edit.textChanged.connect(self.on_palette_modified)
        info_layout.addWidget(self.description_edit, 4, 1)

        right_layout.addWidget(info_group)

        # Colors section
        colors_group = QGroupBox("Colors (2-10 colors)")
        colors_layout = QVBoxLayout(colors_group)

        # Add/Remove color buttons
        color_controls_layout = QHBoxLayout()

        self.add_color_btn = QPushButton("+ Add Color")
        self.add_color_btn.clicked.connect(self.add_color)
        color_controls_layout.addWidget(self.add_color_btn)

        self.remove_color_btn = QPushButton("- Remove Color")
        self.remove_color_btn.clicked.connect(self.remove_color)
        color_controls_layout.addWidget(self.remove_color_btn)

        # Color count display
        self.color_count_label = QLabel("Colors: 7")
        color_controls_layout.addWidget(self.color_count_label)

        color_controls_layout.addStretch()
        colors_layout.addLayout(color_controls_layout)

        # Color buttons container
        self.colors_widget = QWidget()
        self.colors_layout = QGridLayout(self.colors_widget)
        colors_layout.addWidget(self.colors_widget)

        right_layout.addWidget(colors_group)

        # Save button
        self.save_button = QPushButton("Save Palette")
        self.save_button.clicked.connect(self.save_palette)
        self.save_button.setEnabled(False)
        right_layout.addWidget(self.save_button)

        right_layout.addStretch()
        layout.addWidget(right_panel)

        # Initialize with default colors
        self.create_color_buttons(7)

    def refresh_palette_list(self):
        """Refresh the list of available palettes"""
        self.palette_list.clear()
        if not self.palette_manager:
            return

        all_palettes = self.palette_manager.get_all_palettes()
        for palette in all_palettes:
            item = QListWidgetItem(palette.name)
            # Mark built-in palettes differently
            if palette.is_builtin:
                item.setToolTip("Built-in palette (read-only)")
                item.setData(Qt.UserRole, "builtin")
            else:
                item.setToolTip("Custom palette")
                item.setData(Qt.UserRole, "custom")
            self.palette_list.addItem(item)

    def on_palette_selected(self, current, previous):
        """Handle palette selection"""
        if not current:
            self.current_palette = None
            self.clear_editor()
            return

        palette_name = current.text()
        is_builtin = current.data(Qt.UserRole) == "builtin"

        if palette_name in self.palette_manager.palettes:
            self.current_palette = self.palette_manager.palettes[palette_name]
            self.load_palette_to_editor(self.current_palette)

            # Apply palette immediately for preview
            self.apply_palette_preview(self.current_palette)

            # Enable/disable controls based on whether it's built-in
            self.delete_button.setEnabled(not is_builtin)
            self.save_button.setEnabled(not is_builtin)

            # Make fields read-only for built-in palettes
            readonly = is_builtin
            self.name_edit.setReadOnly(readonly)
            self.category_combo.setEnabled(not readonly)
            self.energy_combo.setEnabled(not readonly)
            self.warmth_combo.setEnabled(not readonly)
            self.description_edit.setReadOnly(readonly)
            # Update button states based on readonly status and color count
            if readonly:
                self.add_color_btn.setEnabled(False)
                self.remove_color_btn.setEnabled(False)
            else:
                self.update_button_states()

            for button in self.color_buttons:
                button.setEnabled(not readonly)

    def load_palette_to_editor(self, palette_info):
        """Load a palette into the editor"""
        self.name_edit.setText(palette_info.name)
        self.description_edit.setPlainText(palette_info.description)

        # Set category combo (derived from energy_level and warmth)
        category_display = palette_info.category  # Use the derived category property
        index = self.category_combo.findText(category_display)
        if index >= 0:
            self.category_combo.setCurrentIndex(index)

        # Set energy and warmth
        energy_index = self.energy_combo.findText(palette_info.energy_level)
        if energy_index >= 0:
            self.energy_combo.setCurrentIndex(energy_index)

        warmth_index = self.warmth_combo.findText(palette_info.warmth)
        if warmth_index >= 0:
            self.warmth_combo.setCurrentIndex(warmth_index)

        # Set colors
        self.create_color_buttons(len(palette_info.colors))

        for i, color in enumerate(palette_info.colors):
            if i < len(self.color_buttons):
                self.color_buttons[i].set_color(color)

    def clear_editor(self):
        """Clear the editor fields"""
        self.name_edit.clear()
        self.description_edit.clear()
        self.category_combo.setCurrentIndex(0)
        self.energy_combo.setCurrentIndex(0)
        self.warmth_combo.setCurrentIndex(0)
        self.create_color_buttons(7)
        self.save_button.setEnabled(False)
        self.delete_button.setEnabled(False)

    def create_color_buttons(self, count):
        """Create the initial set of color buttons"""
        # Clear existing buttons
        for button in self.color_buttons:
            button.deleteLater()
        self.color_buttons.clear()

        # Create new buttons
        for i in range(count):
            color_button = ColorButton()
            color_button.colorChanged.connect(self.on_palette_modified)
            self.color_buttons.append(color_button)

            # Arrange in a grid (5 columns max)
            row = i // 5
            col = i % 5
            self.colors_layout.addWidget(color_button, row, col)

        self.update_color_count_display()
        self.update_button_states()
        self.on_palette_modified()

    def add_color(self):
        """Add a new color to the palette"""
        if len(self.color_buttons) >= 10:
            return  # Maximum 10 colors

        # Create new color button
        color_button = ColorButton()
        color_button.colorChanged.connect(self.on_palette_modified)
        self.color_buttons.append(color_button)

        # Add to grid layout
        index = len(self.color_buttons) - 1
        row = index // 5
        col = index % 5
        self.colors_layout.addWidget(color_button, row, col)

        self.update_color_count_display()
        self.update_button_states()
        self.on_palette_modified()

    def remove_color(self):
        """Remove the last color from the palette"""
        if len(self.color_buttons) <= 2:
            return  # Minimum 2 colors

        # Remove the last button
        button = self.color_buttons.pop()
        button.deleteLater()

        self.update_color_count_display()
        self.update_button_states()
        self.on_palette_modified()

    def update_color_count_display(self):
        """Update the color count label"""
        count = len(self.color_buttons)
        self.color_count_label.setText(f"Colors: {count}")

    def update_button_states(self):
        """Update the enabled state of add/remove buttons"""
        count = len(self.color_buttons)
        self.add_color_btn.setEnabled(count < 10)
        self.remove_color_btn.setEnabled(count > 2)

    def on_palette_modified(self):
        """Handle palette modifications"""
        # Enable save button for custom palettes
        if self.current_palette and not self.current_palette.is_builtin:
            self.save_button.setEnabled(True)
        elif not self.current_palette:  # New palette
            self.save_button.setEnabled(True)

        self.paletteChanged.emit()

    def create_new_palette(self):
        """Create a new palette"""
        self.current_palette = None
        self.clear_editor()
        self.name_edit.setText("New Palette")
        self.name_edit.setReadOnly(False)
        self.category_combo.setEnabled(True)
        self.energy_combo.setEnabled(True)
        self.warmth_combo.setEnabled(True)
        self.description_edit.setReadOnly(False)
        # Update button states properly
        self.update_button_states()

        for button in self.color_buttons:
            button.setEnabled(True)

        self.save_button.setEnabled(True)
        self.delete_button.setEnabled(False)

        # Clear selection in list
        self.palette_list.clearSelection()

    def delete_palette(self):
        """Delete the selected custom palette"""
        if not self.current_palette or self.current_palette.is_builtin:
            return

        # Confirm deletion
        reply = QMessageBox.question(
            self,
            "Delete Palette",
            f"Are you sure you want to delete the palette '{self.current_palette.name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            if self.palette_manager.delete_palette(self.current_palette.name):
                self.refresh_palette_list()
                self.clear_editor()
                QMessageBox.information(self, "Success", "Palette deleted successfully!")
            else:
                QMessageBox.warning(self, "Error", "Failed to delete palette.")

    def save_palette(self):
        """Save the current palette"""
        # Validate input
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Error", "Please enter a palette name.")
            return

        # Check if name already exists (for new palettes)
        if not self.current_palette and name in self.palette_manager.palettes:
            QMessageBox.warning(self, "Error", "A palette with this name already exists.")
            return

        # Get colors
        colors = []
        for button in self.color_buttons:
            colors.append(button.color)

        if len(colors) < 2:
            QMessageBox.warning(self, "Error", "Palette must have at least 2 colors.")
            return

        # Create palette info (category is derived automatically)
        from modules.palette_manager import PaletteInfo

        palette_info = PaletteInfo(
            name=name,
            energy_level=self.energy_combo.currentText(),
            warmth=self.warmth_combo.currentText(),
            colors=colors,
            description=self.description_edit.toPlainText(),
            is_builtin=False
        )

        # Save palette
        if self.palette_manager.save_palette(palette_info):
            self.current_palette = palette_info
            self.refresh_palette_list()
            self.save_button.setEnabled(False)
            QMessageBox.information(self, "Success", "Palette saved successfully!")
        else:
            QMessageBox.warning(self, "Error", "Failed to save palette.")

    def set_preview_callback(self, callback):
        """Set the callback function for palette preview"""
        self.preview_callback = callback

    def apply_palette_preview(self, palette_info):
        """Apply a palette for immediate preview"""
        if self.preview_callback:
            # Store original palette selection if not already stored
            if self.original_palette_mode is None:
                # Get current palette selection from the config menu
                parent_config = self.parent()
                while parent_config and not hasattr(parent_config, 'settings'):
                    parent_config = parent_config.parent()
                if parent_config and hasattr(parent_config, 'settings'):
                    selected_palette = parent_config.settings.get('selected_palette', 'Auto (Mood-based)')
                    if selected_palette == 'Auto (Mood-based)':
                        self.original_palette_mode = 'Mood-based'
                    elif selected_palette == 'Random':
                        self.original_palette_mode = 'Random'
                    else:
                        self.original_palette_mode = 'Fixed'
                else:
                    self.original_palette_mode = 'Mood-based'

            # Apply the palette for preview
            self.preview_callback(palette_info.name)

    def restore_original_palette_mode(self):
        """Restore the original palette mode"""
        if self.preview_callback and self.original_palette_mode is not None:
            # Restore original mode
            if self.original_palette_mode == 'Mood-based':
                self.preview_callback('Auto (Mood-based)')
            elif self.original_palette_mode == 'Random':
                self.preview_callback('Random')

            # Clear the stored mode
            self.original_palette_mode = None


class ConfigMenuQt:
    """PyQt5-based configuration menu for KarmaViz"""

    def inherit_from_visualizer(self, vis):
        """Update config menu settings from the current visualizer state."""
        # Map config menu setting names to visualizer attributes
        mapping = {
            "fps": getattr(vis, "fps", 60) if hasattr(vis, "fps") else 60,
            "fullscreen_resolution": getattr(
                vis, "selected_fullscreen_resolution", "Native"
            ),
            "anti_aliasing": getattr(vis, "anti_aliasing_samples", 4),
            "rotation_mode": getattr(vis, "rotation_mode", 1),
            "rotation_speed": getattr(vis, "rotation_speed", 1.0),
            "rotation_amplitude": getattr(vis, "rotation_amplitude", 1.0),
            "pulse_enabled": getattr(vis, "pulse_enabled", True),
            "pulse_intensity": getattr(vis, "pulse_intensity", 1.0),
            "trail_intensity": getattr(vis, "trail_intensity", 0.8),
            "glow_intensity": getattr(vis, "glow_intensity", 0.9),
            "symmetry_mode": getattr(vis, "symmetry_mode", -1),
            "smoke_intensity": getattr(vis, "smoke_intensity", 0.0),
            "bounce_enabled": getattr(vis, "bounce_enabled", False),
            "bounce_intensity": getattr(vis, "bounce_intensity_multiplier", 1.0),
            "animation_speed": getattr(vis, "animation_speed", 1.0),
            "audio_speed_boost": getattr(vis, "audio_speed_boost", 1.0),
            "gpu_waveform_random": getattr(vis, "gpu_waveform_random", True),
            "waveform_scale": getattr(vis, "waveform_scale", 1.0),
            "beats_per_change": getattr(vis, "beats_per_change", 16),
            "beat_sensitivity": (
                getattr(vis.audio_processor, "beat_sensitivity", 1.0)
                if hasattr(vis, "audio_processor")
                else 1.0
            ),
            "transitions_paused": getattr(vis, "transitions_paused", False),
            "chunk_size": (
                getattr(vis.audio_processor, "chunk_size", 256)
                if hasattr(vis, "audio_processor")
                else 256
            ),
            "sample_rate": (
                getattr(vis.audio_processor, "sample_rate", 44100)
                if hasattr(vis, "audio_processor")
                else 44100
            ),
            "selected_palette": getattr(
                vis, "selected_palette_name", "Auto (Mood-based)"
            ),
            "palette_speed": getattr(vis, "palette_rotation_speed", 1.0),
            "color_cycle_speed": getattr(vis, "color_cycle_speed_multiplier", 1.0),
            "palette_transition_speed": getattr(vis, "palette_transition_speed", 0.02),
            "color_transition_smoothness": getattr(
                vis, "color_transition_smoothness", 0.1
            ),
            "warp_intesity": getattr(vis, "warp_intensity", 0.3),
            "warp_first": getattr(vis, "warp_first_enabled", False),
        }
        for key, value in mapping.items():
            self.settings[key] = value
        # Update widgets if dialog exists
        if self.dialog:
            for setting_name, value in self.settings.items():
                if setting_name in self.widgets:
                    self.widgets[setting_name].set_value(value)

    def update_setting_from_visualizer(self, setting_name, value):
        """Update a single config menu setting and widget from the visualizer."""
        self.settings[setting_name] = value
        if self.dialog and setting_name in self.widgets:
            self.widgets[setting_name].set_value(value)

    def __init__(self, settings_path: str, parent=None):
        self.settings_path = settings_path
        self.settings = {}
        self.callbacks = {}
        self.base_title = "KarmaViz Configuration"
        self.widgets = {}
        self.app = None
        self.dialog = None
        self._visible = False
        self.palette_manager = None  # Will be set by the main application
        self.palette_preview_callback = None  # Callback for palette preview
        self.warp_map_manager = None  # Will be set by the main application
        self.warp_map_editor = None  # Warp map editor instance
        self.warp_map_preview_callback = None  # Callback for warp map preview
        self.waveform_manager = None  # Will be set by the main application
        self.waveform_editor = None  # Waveform editor instance
        self.waveform_preview_callback = None  # Callback for waveform preview
        self.preset_manager = None  # Will be set by the main application
        self.visualizer = None  # Reference to the visualizer for syncing settings
        self.shader_compiler = None  # Shader compiler for syntax validation
        self.preferred_monitor = 1  # 0 = primary, 1 = secondary (if available)

        # Mapping of settings to their corresponding hotkeys
        self.hotkey_map = {
            "pulse_enabled": "P",
            "bounce_enabled": "Numpad 5",
            "warp_first": "L",
            "rotation_mode": "R (cycles through modes)",
            "symmetry_mode": "M (cycles through modes)",
            "gpu_waveform_random": "W (toggle random GPU waveforms)",
            "pulse_intensity": "[ (decrease) / ] (increase)",
            "trail_intensity": "Shift+T (decrease) / T (increase)",
            "glow_intensity": "Shift+G (decrease) / G (increase)",
            "glow_radius": "Shift+Down Arrow (decrease) / Shift+Up Arrow (increase)",
            "smoke_intensity": "Shift+F (decrease) / F (increase)",
            "waveform_scale": "Down Arrow (decrease) / Up Arrow (increase)",
            "animation_speed": "Numpad - (decrease) / Numpad + (increase)",
            "audio_speed_boost": "Numpad / (decrease) / Numpad * (increase)",
            "palette_speed": "Numpad 1 (decrease) / Numpad 3 (increase)",
            "color_cycle_speed": "Numpad 4 (decrease) / Numpad 6 (increase)",
            "beats_per_change": ", (decrease) / . (increase)",
            "beat_sensitivity": "Numpad 7 (decrease) / Numpad 9 (increase)",
            "transitions_paused": "/ (toggle)",
            "bounce_intensity": "Numpad 2 (decrease) / Numpad 8 (increase)",
        }

        # We'll create the actual dialog when needed
        # This allows us to avoid QApplication issues

        # Create tabs for each section - organized for better UX
        self.sections = {
            "Display": {
                "settings": [
                    "fps",
                    "fullscreen_resolution",
                    "anti_aliasing",
                ],
                "groups": {
                    "Display Settings": [
                        "fps",
                        "fullscreen_resolution",
                        "anti_aliasing",
                    ],
                },
            },
            "Visual Effects": {
                "settings": [
                    "rotation_mode",
                    "rotation_speed",
                    "rotation_amplitude",
                    "pulse_enabled",
                    "pulse_intensity",
                    "trail_intensity",
                    "glow_intensity",
                    "glow_radius",
                    "symmetry_mode",
                    "smoke_intensity",
                    "bounce_enabled",
                    "bounce_intensity",
                ],
                "groups": {
                    "Rotation Effects": [
                        "rotation_mode",
                        "rotation_speed",
                        "rotation_amplitude",
                    ],
                    "Visual Enhancements": [
                        "pulse_enabled",
                        "pulse_intensity",
                        "trail_intensity",
                        "glow_intensity",
                        "glow_radius",
                    ],
                    "Transformation Effects": ["symmetry_mode", "smoke_intensity"],
                    "Motion Effects": ["bounce_enabled", "bounce_intensity"],
                },
            },
            "Audio Animation": {
                "settings": [
                    "animation_speed",
                    "audio_speed_boost",
                    "beats_per_change",
                    "beat_sensitivity",
                    "transitions_paused",
                    "chunk_size",
                    "sample_rate",
                ],
                "groups": {
                    "Animation Control": ["animation_speed", "audio_speed_boost"],
                    "Beat Detection": [
                        "beats_per_change",
                        "beat_sensitivity",
                        "transitions_paused",
                    ],
                    "Audio Processing": [
                        "chunk_size",
                        "sample_rate",
                    ],
                },
            },
            "Waveforms": {
                "settings": [
                    "gpu_waveform_random",
                    "waveform_scale",
                ],
                "groups": {
                    "GPU Waveform Settings": ["gpu_waveform_random", "waveform_scale"],
                },
            },
            "Palettes": {
                "settings": [
                    "selected_palette",
                    "palette_mode",
                    "palette_speed",
                    "color_cycle_speed",
                    "palette_transition_speed",
                    "color_transition_smoothness",
                ],
                "groups": {
                    "Palette Selection": ["selected_palette"],
                    "Color Animation": [
                        "palette_speed",
                        "color_cycle_speed",
                        "palette_transition_speed",
                        "color_transition_smoothness",
                    ],
                },
            },
            "Warp Maps": {
                "settings": [
                    "warp_intensity",
                    "warp_first",
                ],  # Removed deprecated warp_blend_mode
                "groups": {"Warp Settings": ["warp_intensity", "warp_first"]},
            },
            "Presets": {
                "settings": [],  # No regular settings, just the preset manager
                "groups": {},
            },
        }

        # Setting ranges and types
        self.setting_info = {
            "fps": {"type": "slider", "range": (20, 120), "step": 1, "label": "FPS"},
            "fullscreen_resolution": {
                "type": "cycle",
                "options": get_available_resolutions(),
                "label": "Fullscreen Resolution",
            },
            "anti_aliasing": {
                "type": "cycle",
                "options": [0, 2, 4, 8, 16],
                "option_labels": ["Off", "2x MSAA", "4x MSAA", "8x MSAA", "16x MSAA"],
                "label": "Anti-Aliasing",
            },
            "rotation_mode": {
                "type": "cycle",
                "options": [0, 1, 2, 3],
                "option_labels": [
                    "None",
                    "Clockwise",
                    "Counter-clockwise",
                    "Beat Driven",
                ],
                "label": "Rotation Mode",
            },
            "rotation_speed": {
                "type": "slider",
                "range": (0.1, 5.0),
                "step": 0.1,
                "label": "Rotation Speed",
            },
            "rotation_amplitude": {
                "type": "slider",
                "range": (0.1, 3.0),
                "step": 0.1,
                "label": "Rotation Amplitude",
            },
            "pulse_enabled": {"type": "toggle", "label": "Enable Pulse"},
            "pulse_intensity": {
                "type": "slider",
                "range": (0.0, 2.0),
                "step": 0.01,
                "label": "Pulse Intensity",
            },
            "trail_intensity": {
                "type": "slider",
                "range": (0.0, 5.0),
                "step": 0.05,
                "label": "Trail Intensity",
            },
            "glow_intensity": {
                "type": "slider",
                "range": (0.5, 1.0),
                "step": 0.01,
                "label": "Glow Intensity",
            },
            "glow_radius": {
                "type": "slider",
                "range": (0.01, 0.2),
                "step": 0.005,
                "label": "Glow Radius",
            },
            "symmetry_mode": {
                "type": "cycle",
                "options": [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                "option_labels": [
                    "Random",
                    "None",
                    "Mirror",
                    "Quad",
                    "Kaleidoscope",
                    "Grid",
                    "Spiral",
                    "Diamond",
                    "Fractal",
                    "Radial",
                    "Cross",
                    "Star",
                    "Hexagon",
                ],
                "label": "Symmetry Mode",
            },
            "animation_speed": {
                "type": "slider",
                "range": (0.0, 2.0),
                "step": 0.2,
                "label": "Base Acceleration",
            },
            "audio_speed_boost": {
                "type": "slider",
                "range": (0.0, 1.0),
                "step": 0.05,
                "label": "Audio Speed Multiplier",
            },
            "palette_speed": {
                "type": "slider",
                "range": (0.1, 15.0),
                "step": 0.1,
                "label": "Palette Speed",
            },
            "gpu_waveform_random": {
                "type": "toggle",
                "label": "Random GPU Waveforms",
            },
            "beats_per_change": {
                "type": "slider",
                "range": (1, 64),
                "step": 1,
                "label": "Beats Per Change",
            },
            "smoke_intensity": {
                "type": "slider",
                "range": (0.0, 1.0),
                "step": 0.1,
                "label": "Smoke Intensity",
            },
            "beat_sensitivity": {
                "type": "slider",
                "range": (0.1, 2.0),
                "step": 0.02,
                "label": "Beat Sensitivity",
            },
            "chunk_size": {
                "type": "cycle",
                "options": [128, 256, 512, 1024, 2048, 4096],
                "option_labels": ["128", "256", "512", "1024", "2048", "4096"],
                "label": "Audio Buffer Size",
            },
            "sample_rate": {
                "type": "cycle",
                "options": [22050, 44100, 48000, 88200, 96000],
                "option_labels": ["22.05 kHz", "44.1 kHz", "48 kHz", "88.2 kHz", "96 kHz"],
                "label": "Sample Rate",
            },
            "color_cycle_speed": {
                "type": "slider",
                "range": (0.1, 15.0),
                "step": 0.1,
                "label": "Color Cycle Speed",
            },
            "palette_transition_speed": {
                "type": "slider",
                "range": (0.005, 0.1),
                "step": 0.005,
                "label": "Palette Transition Speed",
            },
            "color_transition_smoothness": {
                "type": "slider",
                "range": (0.05, 1.0),
                "step": 0.05,
                "label": "Color Smoothness",
            },
            "transitions_paused": {"type": "toggle", "label": "Pause Transitions"},
            "waveform_scale": {
                "type": "slider",
                "range": (0.1, 5.0),
                "step": 0.05,
                "label": "Waveform Scale",
            },
            "warp_first": {
                "type": "toggle",
                "label": "Warp Before Symmetry Transformations",
            },
            "bounce_enabled": {"type": "toggle", "label": "Enable Bounce"},
            "bounce_intensity": {
                "type": "slider",
                "range": (0.1, 1.0),
                "step": 0.05,
                "label": "Bounce Intensity",
            },
            "warp_intensity": {
                "type": "slider",
                "range": (0.0, 5.0),    
                "step": 0.10,
                "label": "Warp Intensity",
            },
            "selected_palette": {
                "type": "cycle",
                "options": [
                    "Auto (Mood-based)",
                    "Random",
                ],  # Will be populated with actual palette names
                "label": "Selected Palette",
            },

        }

        # Load settings immediately (without updating widgets since they don't exist yet)
        self.load_settings()

    def _on_setting_changed(self, setting_name, value):
        """Handle setting changes"""
        self.settings[setting_name] = value
        self.trigger_callback(setting_name)

    def register_callback(self, setting_name: str, callback: Callable):
        """Register a callback for a setting"""
        self.callbacks[setting_name] = callback

        # If we already have a setting value, trigger the callback immediately
        if setting_name in self.settings:
            self.trigger_callback(setting_name)

    def trigger_callback(self, setting_name: str):
        """Trigger the callback for a setting"""
        if setting_name in self.callbacks and setting_name in self.settings:
            try:
                self.callbacks[setting_name](self.settings[setting_name])
            except Exception as e:
                logger.debug(f"Error in callback for {setting_name}: {e}")

    def set_palette_manager(self, palette_manager):
        """Set the palette manager and update palette options"""
        self.palette_manager = palette_manager
        self._update_palette_options()

        # Update palette editor if it exists
        if hasattr(self, 'palette_editor') and self.palette_editor:
            self.palette_editor.palette_manager = palette_manager
            self.palette_editor.refresh_palette_list()

    def set_palette_preview_callback(self, callback):
        """Set the callback for palette preview"""
        self.palette_preview_callback = callback

        # Set the callback on the palette editor if it exists
        if hasattr(self, 'palette_editor') and self.palette_editor:
            self.palette_editor.set_preview_callback(callback)

    def set_warp_map_manager(self, warp_map_manager):
        """Set the warp map manager for the config menu"""
        self.warp_map_manager = warp_map_manager

        # If dialog is already created and warp map editor exists, update it
        if hasattr(self, 'warp_map_editor') and self.warp_map_editor:
            self.warp_map_editor.warp_map_manager = warp_map_manager
            self.warp_map_editor.load_warp_map_list()

    def set_warp_map_preview_callback(self, callback):
        """Set the callback for warp map preview"""
        self.warp_map_preview_callback = callback

        # Set the callback on the warp map editor if it exists
        if hasattr(self, 'warp_map_editor') and self.warp_map_editor:
            self.warp_map_editor.set_preview_callback(callback)
            logger.debug(f"Warp map preview callback set on existing editor")

    def set_waveform_manager(self, waveform_manager):
        """Set the waveform manager for the config menu"""
        self.waveform_manager = waveform_manager

        # If dialog is already created and waveform editor exists, update it
        if hasattr(self, 'waveform_editor') and self.waveform_editor:
            self.waveform_editor.waveform_manager = waveform_manager
            self.waveform_editor.load_waveform_list()

    def set_waveform_preview_callback(self, callback):
        """Set the callback for waveform preview"""
        self.waveform_preview_callback = callback

        # Set the callback on the waveform editor if it exists
        if hasattr(self, 'waveform_editor') and self.waveform_editor:
            self.waveform_editor.set_preview_callback(callback)

    def set_shader_manager(self, shader_manager):
        """Set the shader manager for the config menu (handles both waveforms and warp maps)"""
        self.shader_manager = shader_manager

        # Extract warp map manager if available
        if hasattr(shader_manager, 'warp_map_manager') and shader_manager.warp_map_manager:
            self.warp_map_manager = shader_manager.warp_map_manager

            # If dialog is already created and warp map editor exists, update it
            if hasattr(self, 'warp_map_editor') and self.warp_map_editor:
                self.warp_map_editor.warp_map_manager = self.warp_map_manager
                self.warp_map_editor.load_warp_map_list()

        # Use the WaveformManager directly from shader_manager
        if hasattr(shader_manager, "waveform_manager"):
            self.waveform_manager = shader_manager.waveform_manager

            # If dialog is already created and waveform editor exists, update it
            if hasattr(self, 'waveform_editor') and self.waveform_editor:
                self.waveform_editor.waveform_manager = self.waveform_manager
                self.waveform_editor.load_waveform_list()

    def set_visualizer(self, visualizer):
        """Set the visualizer reference for syncing settings"""
        self.visualizer = visualizer
        
        # Also set visualizer reference on preset manager widget if it exists
        if hasattr(self, 'preset_manager_widget') and self.preset_manager_widget:
            self.preset_manager_widget.set_visualizer(visualizer)
        
    def set_shader_compiler(self, shader_compiler):
        """Set the shader compiler for syntax validation"""
        self.shader_compiler = shader_compiler

    def set_preset_manager(self, preset_manager):
        """Set the preset manager for the config menu"""
        self.preset_manager = preset_manager
        
        # If dialog is already created and preset manager widget exists, update it
        if hasattr(self, 'preset_manager_widget') and self.preset_manager_widget:
            self.preset_manager_widget.preset_manager = self.preset_manager
            self.preset_manager_widget.refresh_preset_list()
            # Also ensure visualizer reference is set
            if hasattr(self, 'visualizer') and self.visualizer:
                self.preset_manager_widget.set_visualizer(self.visualizer)

    def rebuild_tabs(self):
        """Force a full rebuild of the dialog and tabs."""
        # Ensure we have the managers before rebuilding
        if not self.palette_manager and hasattr(self, '_pending_palette_manager'):
            self.set_palette_manager(self._pending_palette_manager)
        if not self.warp_map_manager and hasattr(self, '_pending_warp_map_manager'):
            self.set_warp_map_manager(self._pending_warp_map_manager)

        # Rebuild the dialog
        if self.dialog:
            self.dialog = None
        self._create_dialog_if_needed()

    def set_persistent_warp_restore_callback(self, callback):
        """Set the callback for restoring persistent warp maps"""
        self.persistent_warp_restore_callback = callback

    def set_persistent_waveform_restore_callback(self, callback):
        """Set the callback for restoring persistent waveforms"""
        self.persistent_waveform_restore_callback = callback

    def _update_palette_options(self):
        """Update the palette options based on available palettes"""
        if not self.palette_manager:
            return

        # Get all available palettes
        all_palettes = self.palette_manager.get_all_palettes()
        palette_names = ["Auto (Mood-based)", "Random"] + [p.name for p in all_palettes]

        # Update the setting info
        self.setting_info["selected_palette"]["options"] = palette_names

        # Update the widget if it exists
        if "selected_palette" in self.widgets:
            widget = self.widgets["selected_palette"]
            widget.options = palette_names
            widget.combo.clear()
            for option in palette_names:
                widget.combo.addItem(str(option))

            # Update the delegate if it's a PaletteCycleSetting
            if hasattr(widget, 'delegate') and hasattr(widget.delegate, 'palette_manager'):
                widget.delegate.palette_manager = self.palette_manager

    def _on_palette_editor_changed(self):
        """Handle changes from the palette editor"""
        # Refresh the palette options when palettes are modified
        self._update_palette_options()

    def _on_warp_map_changed(self, warp_map_name):
        """Handle warp map changes"""
        logger.debug(f"Warp map '{warp_map_name}' was modified")
        # Trigger shader recompilation with the new warp map
        if hasattr(self, 'warp_map_preview_callback') and self.warp_map_preview_callback:
            self.warp_map_preview_callback(warp_map_name)

    def _on_waveform_changed(self, waveform_name):
        """Handle waveform changes"""
        logger.debug(f"Waveform '{waveform_name}' was modified")
        # Trigger shader recompilation with the new waveform
        if hasattr(self, 'waveform_preview_callback') and self.waveform_preview_callback:
            # Get the waveform info object instead of just the name
            waveform_info = self.waveform_manager.get_waveform(waveform_name)
            if waveform_info:
                self.waveform_preview_callback(waveform_info)

    def _create_setting_widget(self, setting_name: str):
        """Create a widget for a specific setting"""
        if setting_name not in self.setting_info:
            return None

        info = self.setting_info[setting_name]
        label = info.get("label", setting_name.replace("_", " ").title())

        # Create the widget based on the setting type

        if info["type"] == "slider":
            widget = SliderSetting(
                setting_name, label, info["range"][0], info["range"][1], info["step"]
            )
        elif info["type"] == "toggle":
            widget = ToggleSetting(setting_name, label)
        elif info["type"] == "cycle":
            # Use special palette cycle widget for palette selection
            if setting_name == "selected_palette" and self.palette_manager:
                widget = PaletteCycleSetting(
                    setting_name, label, info["options"], self.palette_manager
                )
            else:
                option_labels = info.get("option_labels", None)
                widget = CycleSetting(
                    setting_name, label, info["options"], option_labels
                )

        # Add tooltip with hotkey information if available
        if setting_name in self.hotkey_map:
            hotkey_info = self.hotkey_map[setting_name]
            tooltip = f"Hotkey: {hotkey_info}"
            widget.label.setToolTip(tooltip)
            # Add a small indicator to show there's a tooltip
            widget.label.setText(f"{label} [K]")

        widget.valueChanged.connect(self._on_setting_changed)
        self.widgets[setting_name] = widget
        return widget

    def _launch_standalone_warp_editor(self):
        """Launch the standalone warp map editor"""
        try:
            import subprocess
            import sys
            import os

            # Get the path to the launcher script
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            launcher_path = os.path.join(script_dir, "warp_map_editor_launcher.py")

            if os.path.exists(launcher_path):
                # Launch the standalone editor
                subprocess.Popen([sys.executable, launcher_path])
                QMessageBox.information(
                    self.dialog, "Launched",
                    "Standalone warp map editor launched in a separate window."
                )
            else:
                QMessageBox.warning(
                    self.dialog, "Error",
                    f"Warp map editor launcher not found at: {launcher_path}"
                )
        except Exception as e:
            QMessageBox.critical(
                self.dialog, "Error",
                f"Failed to launch standalone warp map editor:\n{str(e)}"
            )

    def _launch_standalone_waveform_editor(self):
        """Launch the standalone waveform editor"""
        try:
            import subprocess
            import sys
            import os

            # Get the path to the launcher script
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            launcher_path = os.path.join(script_dir, "waveform_editor_launcher.py")

            if os.path.exists(launcher_path):
                # Launch the standalone editor
                subprocess.Popen([sys.executable, launcher_path])
                QMessageBox.information(
                    self.dialog, "Launched",
                    "Standalone waveform editor launched in a separate window."
                )
            else:
                QMessageBox.warning(
                    self.dialog, "Error",
                    f"Waveform editor launcher not found at: {launcher_path}"
                )
        except Exception as e:
            QMessageBox.critical(
                self.dialog, "Error",
                f"Failed to launch standalone waveform editor:\n{str(e)}"
            )

    def _get_default_settings(self):
        """Get the default settings dictionary"""
        return {
            "fps": 30,
            # Display & Performance
            "fullscreen_resolution": "1024x576",
            "anti_aliasing": 16,  # 4x MSAA by default
            "rotation_mode": 1,  # Clockwise by default
            "rotation_speed": 0.5,
            "rotation_amplitude": 0.5,
            "pulse_enabled": False,
            "pulse_intensity": 0.5,
            "trail_intensity": 5.0,
            "glow_intensity": 0.85,
            "glow_radius": 0.20,
            "symmetry_mode": 1,  # Random
            "smoke_intensity": 0.0,
            "bounce_enabled": False,
            "bounce_intensity": 0.5,
            # Audio & Animation
            "animation_speed": 0.5,  # Base acceleration (matches hotkey default)
            "audio_speed_boost": 1.0,  # Audio speed multiplier (matches hotkey default)
            "gpu_waveform_random": True,  # Random GPU waveforms enabled
            "waveform_scale": 1.0,
            "beats_per_change": 8,
            "beat_sensitivity": 1.5,
            "transitions_paused": False,
            "chunk_size": 512,
            "sample_rate": 44100,
            # Palettes
            "selected_palette": "Auto (Mood-based)",
            "palette_speed": 3.0,
            "color_cycle_speed": 3.0,
            "palette_transition_speed": 0.05,
            "color_transition_smoothness": 0.5,
            # Warp Maps
            "warp_intensity": 1.0,
            "warp_first": False,
        }

    def load_settings(self):
        """Load settings from file"""
        default_settings = self._get_default_settings()

        try:
            if os.path.exists(self.settings_path):
                with open(self.settings_path, "r") as f:
                    loaded = json.load(f)
                    # Merge loaded settings with defaults, ensuring all keys exist
                    self.settings = {**default_settings, **loaded}
            else:
                self.settings = default_settings.copy()
        except (json.JSONDecodeError, IOError) as e:
            logger.debug(f"Error loading settings: {e}. Using default settings.")
            self.settings = default_settings.copy()

        # Update widgets with loaded settings if dialog exists
        if self.dialog:
            for setting_name, value in self.settings.items():
                if setting_name in self.widgets:
                    self.widgets[setting_name].set_value(value)

        # Apply loaded settings through callbacks
        for setting_name in self.settings:
            self.trigger_callback(setting_name)

    def save_settings(self):
        """Save settings to file"""
        try:
            os.makedirs(os.path.dirname(self.settings_path), exist_ok=True)
            with open(self.settings_path, "w") as f:
                json.dump(self.settings, f, indent=4)
            logger.debug(f"Settings saved to {self.settings_path}")
        except IOError as e:
            logger.debug(f"Error saving settings: {e}")

    def reset_to_defaults(self):
        """Reset settings to defaults"""
        default_settings = self._get_default_settings()

        # Update settings and widgets
        self.settings = default_settings.copy()

        # Update widgets if dialog exists
        if self.dialog:
            for setting_name, value in self.settings.items():
                if setting_name in self.widgets:
                    self.widgets[setting_name].set_value(value)

        # Apply settings through callbacks
        for setting_name in self.settings:
            self.trigger_callback(setting_name)

    def _create_dialog_if_needed(self):
        """Create the dialog if it doesn't exist yet"""
        if self.dialog is None:
            # Use existing QApplication
            self.app = QApplication.instance()

            # Create the dialog
            # Use QDialog but with careful focus handling
            self.dialog = QDialog()
            self.dialog.setWindowTitle(self.base_title)
            self.dialog.resize(800, 1000)
            
            # Set window flags to ensure it doesn't interfere with fullscreen
            # Qt.Tool makes it a utility window
            self.dialog.setWindowFlags(Qt.Tool)
            
            # Set attributes to prevent focus stealing
            self.dialog.setAttribute(Qt.WA_ShowWithoutActivating, True)
            
            # Set modal to false to allow interaction with other windows
            self.dialog.setModal(False)
            
            # Set focus policy to prevent automatic focusing
            self.dialog.setFocusPolicy(Qt.ClickFocus)
            
            # Store the original pygame window for focus restoration
            self._pygame_window_id = None
            self._store_pygame_window_id()
            
            # Don't set NoFocus policy - we need the dialog to be interactive
            # Instead, we'll handle focus in the show method
            
            # Position dialog on secondary monitor if available
            self._position_on_secondary_monitor()

            # Add a tooltip to the dialog title bar
            self.dialog.setToolTip("Settings with [K] indicator have keyboard shortcuts. Hover over them to see the hotkey.")

            # Apply dark theme
            self._apply_dark_theme()

            # Main layout
            main_layout = QVBoxLayout(self.dialog)

            # Add a help label at the top
            help_label = QLabel("Settings with [K] indicator have keyboard shortcuts. Hover over them to see the hotkey.")
            help_label.setStyleSheet("color: #aaaaaa; font-style: italic; padding: 5px;")
            help_label.setAlignment(Qt.AlignCenter)
            main_layout.addWidget(help_label)

            # Create tab widget
            self.tab_widget = QTabWidget()
            self.tab_widget.currentChanged.connect(self._on_tab_changed)  # Detect tab changes
            main_layout.addWidget(self.tab_widget)

            # Create tabs and widgets using new grouped structure
            for section_name, section_data in self.sections.items():
                tab = QWidget()
                tab_layout = QVBoxLayout(tab)

                if section_name == "Palettes":
                    # Special handling for Palettes tab - add palette editor
                    self._create_palette_tab(tab_layout, section_data)
                elif section_name == "Warp Maps":
                    # Special handling for Warp Maps tab - add warp map editor
                    self._create_warp_maps_tab(tab_layout, section_data)
                elif section_name == "Waveforms":
                    # Special handling for Waveforms tab - add waveform editor
                    self._create_waveforms_tab(tab_layout, section_data)
                elif section_name == "Presets":
                    # Special handling for Presets tab - add preset manager
                    self._create_presets_tab(tab_layout, section_data)
                else:
                    # Regular tabs with grouped settings
                    self._create_regular_tab(tab_layout, section_data)

                # Add the tab
                self.tab_widget.addTab(tab, section_name)

            # Add buttons at the bottom
            button_layout = QHBoxLayout()

            self.save_button = QPushButton("Save and Close")
            self.save_button.clicked.connect(self.save_and_close)

            self.reset_button = QPushButton("Reset to Defaults")
            self.reset_button.clicked.connect(self.reset_to_defaults)

            button_layout.addWidget(self.reset_button)
            button_layout.addStretch()
            button_layout.addWidget(self.save_button)

            main_layout.addLayout(button_layout)

            # Update widgets with current settings
            for setting_name, value in self.settings.items():
                if setting_name in self.widgets:
                    self.widgets[setting_name].set_value(value)

            # Connect dialog close event to update visibility
            self.dialog.finished.connect(self._on_dialog_closed)
            
            # Set focus policy on all child widgets to prevent focus stealing
            self._set_no_auto_focus_on_children(self.dialog)
            
            # Override the dialog's showEvent to handle focus restoration
            original_show_event = self.dialog.showEvent
            def custom_show_event(event):
                # Call the original showEvent
                original_show_event(event)
                # Immediately try to restore pygame focus
                from PyQt5.QtCore import QTimer
                QTimer.singleShot(1, self._restore_pygame_focus)
                
            self.dialog.showEvent = custom_show_event

    def _position_on_secondary_monitor(self):
        """Position the dialog on the preferred monitor"""
        if not self.dialog:
            return
            
        desktop = QDesktopWidget()
        screen_count = desktop.screenCount()
        
        # Determine which monitor to use
        target_monitor = min(self.preferred_monitor, screen_count - 1)
        
        if screen_count > 1 and self.preferred_monitor > 0:
            # Use secondary monitor if available and preferred
            target_screen = desktop.screenGeometry(target_monitor)
            monitor_name = f"secondary monitor (#{target_monitor})"
        else:
            # Use primary monitor
            target_screen = desktop.screenGeometry(0)
            target_monitor = 0
            monitor_name = "primary monitor"
        
        # Position the dialog in the top-right corner with some margin
        margin = 50
        dialog_width = self.dialog.width()
        
        x = target_screen.x() + target_screen.width() - dialog_width - margin
        y = target_screen.y() + margin
        
        self.dialog.move(x, y)
        logger.debug(f"Positioned config menu on {monitor_name} at ({x}, {y})")
        
        if screen_count > 1:
            logger.debug(f"Detected {screen_count} monitors - config menu will not interfere with fullscreen visualization")

    def set_preferred_monitor(self, monitor_index: int):
        """Set the preferred monitor for the config menu
        
        Args:
            monitor_index: 0 for primary monitor, 1+ for secondary monitors
        """
        self.preferred_monitor = monitor_index
        # If dialog exists, reposition it
        if self.dialog:
            self._position_on_secondary_monitor()

    def get_monitor_info(self):
        """Get information about available monitors
        
        Returns:
            dict: Monitor information including count and geometries
        """
        if not self.app:
            self.app = QApplication.instance()
            
        desktop = QDesktopWidget()
        screen_count = desktop.screenCount()
        
        monitors = []
        for i in range(screen_count):
            geometry = desktop.screenGeometry(i)
            monitors.append({
                'index': i,
                'width': geometry.width(),
                'height': geometry.height(),
                'x': geometry.x(),
                'y': geometry.y(),
                'name': f"Monitor {i}" + (" (Primary)" if i == 0 else "")
            })
        
        return {
            'count': screen_count,
            'monitors': monitors,
            'supports_multimonitor': screen_count > 1
        }

    def _restore_pygame_focus(self):
        """Attempt to restore focus to the pygame window after showing the config menu"""
        try:
            import subprocess
            import time
            
            # Small delay to let the Qt dialog finish showing
            time.sleep(0.05)
            
            # Method 1: Use stored window ID if available
            if hasattr(self, '_pygame_window_id') and self._pygame_window_id:
                try:
                    subprocess.run(['xdotool', 'windowactivate', self._pygame_window_id], timeout=1)
                    logger.debug(f"Restored focus using stored window ID: {self._pygame_window_id}")
                    return True
                except:
                    logger.error(f"Failed to activate stored window {self._pygame_window_id}")
            
            # Method 2: Search by window title containing "pygame"
            window_id = None
            try:
                result = subprocess.run(['xdotool', 'search', '--name', '.*pygame.*'], 
                                      capture_output=True, text=True, timeout=1)
                if result.returncode == 0 and result.stdout.strip():
                    window_id = result.stdout.strip().split('\n')[0]
                    logger.debug(f"Found pygame window by name (ID: {window_id})")
            except:
                pass
            
            # Method 3: Search by window class
            if not window_id:
                try:
                    result = subprocess.run(['xdotool', 'search', '--class', 'pygame'], 
                                          capture_output=True, text=True, timeout=1)
                    if result.returncode == 0 and result.stdout.strip():
                        window_id = result.stdout.strip().split('\n')[0]
                        logger.debug(f"🎮 Found pygame window by class (ID: {window_id})")
                except:
                    pass
            
            # If we found a window, activate it
            if window_id:
                try:
                    subprocess.run(['xdotool', 'windowactivate', window_id], timeout=1)
                    logger.debug(f"🎮 Restored focus to pygame window (ID: {window_id})")
                    return True
                except:
                    logger.error(f" Failed to activate window {window_id}")
            
            logger.warning(" Could not find pygame window to restore focus")
            return False
                    
        except Exception as e:
            logger.warning(f" Could not restore pygame focus: {e}")
            return False

    def print_debug_monitor_info(self):
        """print debug information about available monitors for debugging"""
        monitor_info = self.get_monitor_info()
        logger.debug(f"Total monitors: {monitor_info['count']}")
        logger.debug(f"Multi-monitor support: {monitor_info['supports_multimonitor']}")
        
        for monitor in monitor_info['monitors']:
            logger.debug(f"   {monitor['name']}: {monitor['width']}x{monitor['height']} at ({monitor['x']}, {monitor['y']})")
        
        logger.debug(f"Config menu preferred monitor: {self.preferred_monitor}")
        
        if monitor_info['supports_multimonitor']:
            logger.debug("Multi-monitor setup detected - config menu can be positioned on secondary monitor")
        else:
            logger.debug(" Single monitor setup - config menu will be positioned on primary monitor")

    def _store_pygame_window_id(self):
        """Store the current pygame window ID for later focus restoration"""
        try:
            import subprocess
            
            # Try to find the currently active window (likely pygame)
            result = subprocess.run(['xdotool', 'getactivewindow'], 
                                  capture_output=True, text=True, timeout=1)
            if result.returncode == 0 and result.stdout.strip():
                self._pygame_window_id = result.stdout.strip()
                logger.debug(f"Stored pygame window ID: {self._pygame_window_id}")
                return True
        except Exception as e:
            logger.error(f" Could not store pygame window ID: {e}")
        
        return False

    def _set_no_auto_focus_on_children(self, widget):
        """Recursively set focus policy on all child widgets to prevent auto-focus"""
        try:
            from PyQt5.QtWidgets import QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox
            
            # Set focus policy on the widget itself
            if hasattr(widget, 'setFocusPolicy'):
                # Only change focus policy for input widgets that might auto-focus
                if isinstance(widget, (QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox)):
                    widget.setFocusPolicy(Qt.ClickFocus)
            
            # Recursively apply to all children
            for child in widget.findChildren(QWidget):
                if isinstance(child, (QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox)):
                    child.setFocusPolicy(Qt.ClickFocus)
                    
        except Exception as e:
            logger.warning(f" Could not set focus policy on child widgets: {e}")

    def _immediate_focus_restore(self):
        """Immediately try to restore focus without any delays"""
        try:
            import subprocess
            
            # Method 1: Use stored window ID if available
            if hasattr(self, '_pygame_window_id') and self._pygame_window_id:
                try:
                    subprocess.run(['xdotool', 'windowactivate', self._pygame_window_id], 
                                 timeout=0.5, check=True)
                    logger.debug(f"🎮 Immediate focus restore using stored window ID: {self._pygame_window_id}")
                    return True
                except:
                    pass
            
            # Method 2: Search for pygame window immediately
            try:
                result = subprocess.run(['xdotool', 'search', '--name', '.*pygame.*'], 
                                      capture_output=True, text=True, timeout=0.5)
                if result.returncode == 0 and result.stdout.strip():
                    window_id = result.stdout.strip().split('\n')[0]
                    subprocess.run(['xdotool', 'windowactivate', window_id], timeout=0.5)
                    logger.debug(f"🎮 Immediate focus restore found pygame window: {window_id}")
                    return True
            except:
                pass
                
            return False
        except Exception as e:
            logger.debug(f" Immediate focus restore failed: {e}")
            return False

    def _create_palette_tab(self, tab_layout, section_data):
        """Create the Palettes tab with grouped settings and palette editor"""
        # Create a scroll area for regular settings
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setFrameShape(QFrame.NoFrame)
        settings_scroll.setMaximumHeight(
            400
        )  # Increased height for settings to show all options

        # Create a widget to hold palette settings
        settings_content = QWidget()
        settings_layout = QVBoxLayout(settings_content)

        # Add grouped palette settings
        for group_name, setting_names in section_data["groups"].items():
            group_box = QGroupBox(group_name)
            group_layout = QVBoxLayout(group_box)

            for setting_name in setting_names:
                widget = self._create_setting_widget(setting_name)
                if widget:
                    group_layout.addWidget(widget)

            settings_layout.addWidget(group_box)

        settings_layout.addStretch()
        settings_scroll.setWidget(settings_content)
        tab_layout.addWidget(settings_scroll)

        # Add palette editor
        if self.palette_manager:
            logger.debug("Creating palette editor with palette manager")
            self.palette_editor = PaletteEditor(self.palette_manager)
            self.palette_editor.paletteChanged.connect(self._on_palette_editor_changed)
            # Set up preview callback if available
            if self.palette_preview_callback:
                self.palette_editor.set_preview_callback(self.palette_preview_callback)
            tab_layout.addWidget(self.palette_editor)
        else:
            # Placeholder if palette manager not available yet
            placeholder = QLabel(
                "Palette editor will be available once palette manager is loaded."
            )
            logger.warning("Palette manager not available for palette editor")
            tab_layout.addWidget(placeholder)

    def _create_warp_maps_tab(self, tab_layout, section_data):
        """Create the Warp Maps tab with grouped settings and warp map editor"""
        # Add grouped warp settings
        for group_name, setting_names in section_data["groups"].items():
            group_box = QGroupBox(group_name)
            group_layout = QVBoxLayout(group_box)

            for setting_name in setting_names:
                widget = self._create_setting_widget(setting_name)
                if widget:
                    group_layout.addWidget(widget)

            tab_layout.addWidget(group_box)

        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        tab_layout.addWidget(separator)

        # Add warp map editor (without preview section when launched from config menu)
        if WARP_MAP_AVAILABLE and self.warp_map_manager:
            logger.debug("Creating warp map editor with warp map manager")
            self.warp_map_editor = WarpMapEditor(
                self.warp_map_manager, show_preview=False, shader_compiler=self.shader_compiler
            )
            self.warp_map_editor.warp_map_changed.connect(self._on_warp_map_changed)
            # Set up preview callback if available
            if self.warp_map_preview_callback:
                self.warp_map_editor.set_preview_callback(
                    self.warp_map_preview_callback
                )
            tab_layout.addWidget(self.warp_map_editor)
        else:
            # Placeholder if warp map system not available
            placeholder_layout = QVBoxLayout()
            if not WARP_MAP_AVAILABLE:
                placeholder = QLabel(
                    "Warp map editor not available (missing dependencies)."
                )
                logger.warning("Warp map editor not available (missing dependencies)")
            else:
                placeholder = QLabel(
                    "Warp map editor will be available once warp map manager is loaded."
                )
                logger.warning("Warp map manager not available for warp map editor")
            placeholder_layout.addWidget(placeholder)

            # Add a button to launch standalone editor
            launch_button = QPushButton("Launch Standalone Warp Map Editor")
            launch_button.clicked.connect(self._launch_standalone_warp_editor)
            placeholder_layout.addWidget(launch_button)
            placeholder_layout.addStretch()

            placeholder_widget = QWidget()
            placeholder_widget.setLayout(placeholder_layout)
            tab_layout.addWidget(placeholder_widget)

    def _create_waveforms_tab(self, tab_layout, section_data):
        """Create the Waveforms tab with grouped settings and waveform editor"""
        # Add grouped waveform settings
        for group_name, setting_names in section_data["groups"].items():
            group_box = QGroupBox(group_name)
            group_layout = QVBoxLayout(group_box)

            for setting_name in setting_names:
                widget = self._create_setting_widget(setting_name)
                if widget:
                    group_layout.addWidget(widget)

            tab_layout.addWidget(group_box)

        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        tab_layout.addWidget(separator)

        # Add waveform editor (without preview section when launched from config menu)
        if WAVEFORM_AVAILABLE and self.waveform_manager:
            logger.debug("Creating waveform editor with waveform manager")
            from modules.waveform_editor import WaveformEditorWidget
            self.waveform_editor = WaveformEditorWidget(
                self.waveform_manager, show_preview=False, shader_compiler=self.shader_compiler
            )
            self.waveform_editor.waveform_saved.connect(self._on_waveform_changed)
            # Set up preview callback if available
            if self.waveform_preview_callback:
                self.waveform_editor.set_preview_callback(
                    self.waveform_preview_callback
                )
            tab_layout.addWidget(self.waveform_editor)
        else:
            # Placeholder if waveform system not available
            placeholder_layout = QVBoxLayout()
            if not WAVEFORM_AVAILABLE:
                placeholder = QLabel(
                    "Waveform editor not available (missing dependencies)."
                )
                logger.warning("Waveform editor not available (missing dependencies)")
            else:
                placeholder = QLabel(
                    "Waveform editor will be available once waveform manager is loaded."
                )
                logger.warning("Waveform manager not available for waveform editor")
            placeholder_layout.addWidget(placeholder)

            # Add a button to launch standalone editor
            launch_button = QPushButton("Launch Standalone Waveform Editor")
            launch_button.clicked.connect(self._launch_standalone_waveform_editor)
            placeholder_layout.addWidget(launch_button)
            placeholder_layout.addStretch()

            placeholder_widget = QWidget()
            placeholder_widget.setLayout(placeholder_layout)
            tab_layout.addWidget(placeholder_widget)

    def _create_presets_tab(self, tab_layout, section_data):
        """Create the presets management tab"""
        # Create preset manager widget
        from modules.preset_manager_widget import PresetManagerWidget
        
        try:
            # Create the preset manager widget with config menu reference
            self.preset_manager_widget = PresetManagerWidget(self.preset_manager, config_menu=self)
            # Set visualizer reference if available
            if hasattr(self, 'visualizer') and self.visualizer:
                self.preset_manager_widget.set_visualizer(self.visualizer)
            tab_layout.addWidget(self.preset_manager_widget)
        except ImportError:
            # Fallback if preset manager widget doesn't exist yet
            placeholder_layout = QVBoxLayout()
            placeholder = QLabel("Preset Manager")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("font-size: 16px; font-weight: bold; color: #888;")
            placeholder_layout.addWidget(placeholder)
            
            info_label = QLabel(
                "WARNING: Unable to load preset management system"
            )
            info_label.setAlignment(Qt.AlignCenter)
            info_label.setStyleSheet("color: #666; font-size: 12px;")
            placeholder_layout.addWidget(info_label)
            
            placeholder_layout.addStretch()
            
            placeholder_widget = QWidget()
            placeholder_widget.setLayout(placeholder_layout)
            tab_layout.addWidget(placeholder_widget)

    def _create_regular_tab(self, tab_layout, section_data):
        """Create a regular tab with grouped settings"""
        # Create a scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        # Create a widget to hold all settings
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Add grouped settings
        for group_name, setting_names in section_data["groups"].items():
            group_box = QGroupBox(group_name)
            group_layout = QVBoxLayout(group_box)

            for setting_name in setting_names:
                widget = self._create_setting_widget(setting_name)
                if widget:
                    group_layout.addWidget(widget)

            scroll_layout.addWidget(group_box)

        # Add stretch to push settings to the top
        scroll_layout.addStretch()

        # Set the scroll content
        scroll.setWidget(scroll_content)
        tab_layout.addWidget(scroll)

    def _on_dialog_closed(self):
        """Handle dialog closed event"""
        self._visible = False
        # Restore original palette mode when dialog is closed
        if hasattr(self, 'palette_editor') and self.palette_editor:
            self.palette_editor.restore_original_palette_mode()

    def _on_tab_changed(self, index):
        """Handle tab change event"""
        current_tab_name = self.tab_widget.tabText(index)

        # Restore original palette mode when switching away from Palettes tab
        if hasattr(self, 'palette_editor') and self.palette_editor:
            if current_tab_name != "Palettes":
                self.palette_editor.restore_original_palette_mode()

        # Restore persistent warp map when switching away from Warp Maps tab
        if current_tab_name != "Warp Maps":
            if hasattr(self, 'persistent_warp_restore_callback') and self.persistent_warp_restore_callback:
                self.persistent_warp_restore_callback()

        # Restore persistent waveform when switching away from Waveforms tab
        if current_tab_name != "Waveforms":
            if hasattr(self, 'persistent_waveform_restore_callback') and self.persistent_waveform_restore_callback:
                self.persistent_waveform_restore_callback()

    def _apply_dark_theme(self):
        """Apply a dark theme to the dialog"""
        if self.dialog is None:
            return

        # Use pure stylesheet approach - no palette conflicts

        # Set comprehensive dark stylesheet
        self.dialog.setStyleSheet("""
            /* Main dialog background */
            QDialog {
                background-color: #2b2b2b;
                color: #ffffff;
            }

            /* All widgets default styling */
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            /* Tab widget styling */
            QTabWidget::pane {
                border: 1px solid #555555;
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QTabBar::tab {
                background-color: #3c3c3c;
                color: #ffffff;
                padding: 8px 16px;
                border: 1px solid #555555;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #2b2b2b;
                border-bottom: 1px solid #2b2b2b;
            }
            QTabBar::tab:hover {
                background-color: #4c4c4c;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #2A2A2A;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #5A5A5A;
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #6A6A6A;
            }
            QPushButton {
                background-color: #2A82DA;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #3A92EA;
            }
            QPushButton:pressed {
                background-color: #1A72CA;
            }
            /* ComboBox styling */
            QComboBox {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 6px 8px;
                border-radius: 4px;
                min-height: 20px;
            }
            QComboBox:hover {
                border: 1px solid #777777;
                background-color: #4c4c4c;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #555555;
                background-color: #4c4c4c;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #ffffff;
                width: 0px;
                height: 0px;
            }
            QComboBox QAbstractItemView {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555555;
                selection-background-color: #0078d4;
                selection-color: #ffffff;
                outline: none;
            }
            QComboBox QAbstractItemView::item {
                padding: 8px;
                border: none;
                color: #ffffff;
                background-color: #3c3c3c;
            }
            QComboBox QAbstractItemView::item:hover {
                background-color: #4c4c4c;
                color: #ffffff;
            }
            QComboBox QAbstractItemView::item:selected {
                background-color: #0078d4;
                color: #ffffff;
            }
            QScrollArea {
                background-color: transparent;
                border: none;
            }
            QCheckBox {
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:unchecked {
                border: 2px solid #999;
                background-color: #2A2A2A;
                border-radius: 4px;
            }
            QCheckBox::indicator:checked {
                border: 2px solid #2A82DA;
                background-color: #2A82DA;
                border-radius: 4px;
            }
            /* Text input styling */
            QLineEdit {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 6px 8px;
                border-radius: 4px;
                selection-background-color: #0078d4;
            }
            QLineEdit:hover {
                border: 1px solid #777777;
            }
            QLineEdit:focus {
                border: 2px solid #0078d4;
            }
            QTextEdit {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 4px;
                selection-background-color: #0078d4;
            }
            QTextEdit:hover {
                border: 1px solid #777777;
            }
            QTextEdit:focus {
                border: 2px solid #0078d4;
            }
            QSpinBox {
                background-color: #2A2A2A;
                color: white;
                border: 1px solid #555;
                padding: 4px 8px;
                border-radius: 4px;
                min-height: 20px;
            }
            QSpinBox:hover {
                border: 1px solid #777;
            }
            QSpinBox:focus {
                border: 2px solid #2A82DA;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: #3A3A3A;
                border: 1px solid #555;
                width: 16px;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: #4A4A4A;
            }
            QSpinBox::up-arrow, QSpinBox::down-arrow {
                width: 0px;
                height: 0px;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
            }
            QSpinBox::up-arrow {
                border-bottom: 6px solid #CCCCCC;
            }
            QSpinBox::down-arrow {
                border-top: 6px solid #CCCCCC;
            }
            QListWidget {
                background-color: #2A2A2A;
                color: white;
                border: 1px solid #555;
                border-radius: 4px;
                selection-background-color: #2A82DA;
                outline: none;
            }
            QListWidget::item {
                padding: 6px 8px;
                border: none;
                background-color: transparent;
            }
            QListWidget::item:hover {
                background-color: #3A3A3A;
            }
            QListWidget::item:selected {
                background-color: #2A82DA;
                color: white;
            }
            QGroupBox {
                color: white;
                border: 1px solid #555;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                background-color: #353535;
            }
            /* Labels and other elements */
            QLabel {
                color: #ffffff;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background-color: #2A2A2A;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #555;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #666;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

    def resize(self, _window_size):
        """Handle window resize events - not needed for Qt dialog but kept for compatibility"""
        # Qt dialogs handle their own sizing, so this is just a stub for compatibility
        pass

    def toggle(self):
        """Toggle the visibility of the menu"""
        # Ensure managers are set before dialog creation
        if not self.palette_manager and hasattr(self, '_pending_palette_manager'):
            self.set_palette_manager(self._pending_palette_manager)
        if not self.warp_map_manager and hasattr(self, '_pending_warp_map_manager'):
            self.set_warp_map_manager(self._pending_warp_map_manager)
        if self._visible:
            # Ensure all pending PyQt events are processed before hiding
            if self.app:
                self.app.processEvents()
            if self.dialog:
                self.dialog.hide()

            # Clean up timers when hiding
            if hasattr(self, "warp_map_editor") and self.warp_map_editor:
                self.warp_map_editor.cleanup()

            self._visible = False

            # Restore persistent warp map when menu is closed
            if hasattr(self, 'persistent_warp_restore_callback') and self.persistent_warp_restore_callback:
                self.persistent_warp_restore_callback()

            # Restore persistent waveform when menu is closed
            if hasattr(self, 'persistent_waveform_restore_callback') and self.persistent_waveform_restore_callback:
                self.persistent_waveform_restore_callback()
        else:
            # Create dialog if needed
            self._create_dialog_if_needed()
            
            # Update all settings from visualizer before showing the menu
            if hasattr(self, 'visualizer') and self.visualizer:
                self.inherit_from_visualizer(self.visualizer)

            if self.dialog:
                # Store the current active window (should be pygame) before showing dialog
                self._store_pygame_window_id()
                
                # Ensure it's positioned correctly on the preferred monitor
                self._position_on_secondary_monitor()
                
                # Show the dialog without activating it
                self.dialog.show()
                
                # Force immediate focus restoration using direct xdotool call
                self._immediate_focus_restore()
                
                # Also use timers as backup
                from PyQt5.QtCore import QTimer
                QTimer.singleShot(10, self._restore_pygame_focus)   # 10ms delay
                QTimer.singleShot(50, self._restore_pygame_focus)   # 50ms delay
                QTimer.singleShot(100, self._restore_pygame_focus)  # 100ms delay as backup
            self._visible = True

    @property
    def visible(self):
        """Get the visibility state of the menu"""
        return self._visible

    def handle_event(self, _event):
        """Handle pygame events - not needed for Qt dialog but kept for compatibility"""
        # Qt handles its own events, so this is just a stub for compatibility
        return False

    def _cleanup_dialog(self):
        """Clean up dialog resources, especially timers in child widgets"""
        try:
            if hasattr(self, "warp_map_editor") and self.warp_map_editor:
                self.warp_map_editor.cleanup()
                self.warp_map_editor = None

            if self.dialog:
                # Process any pending events before cleanup
                if self.app:
                    self.app.processEvents()

                # Clean up the dialog
                self.dialog.deleteLater()
        except Exception as e:
            logger.warning(f"Error cleaning up config dialog: {e}")

    def save_and_close(self):
        """Save settings and close the dialog"""
        self.save_settings()
        if self.dialog:
            self.dialog.close()

    def render(self):
        """Render the menu - not needed for Qt dialog but kept for compatibility"""
        # Qt handles its own rendering, so this is just a stub for compatibility
        # Process Qt events to keep the UI responsive
        if self.app and self._visible:
            self.app.processEvents()
        pass

    def cleanup(self):
        """Clean up all resources"""
        try:
            self._cleanup_dialog()
            if self.app:
                self.app.processEvents()
        except Exception as e:
            logger.warning(f"Error during ConfigMenuQt cleanup: {e}")

    def __del__(self):
        """Destructor - ensure cleanup"""
        try:
            self.cleanup()
        except:
            pass

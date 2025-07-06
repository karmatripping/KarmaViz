"""
Warp Map Editor for KarmaViz

This module provides a GUI for creating, editing, and managing warp maps.
Similar to the palette editor but for GLSL warp map functions.
"""

import sys
import json
import time
import math
import numpy as np
from typing import Optional, List, Dict
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLabel,
    QLineEdit, QComboBox, QPushButton, QListWidget, QSplitter,
    QGroupBox, QFormLayout, QMessageBox, QFileDialog, QTabWidget,
    QListWidgetItem, QDialog, QDialogButtonBox, QSpinBox, QSlider,
    QCheckBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QTextDocument

# Try to import pygame for OpenGL preview
try:
    import pygame
    import pygame.locals
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    import moderngl
    MODERNGL_AVAILABLE = True
except ImportError:
    MODERNGL_AVAILABLE = False


from modules.warp_map_manager import WarpMapManager, WarpMapInfo
from modules.glsl_syntax_highlighter import GLSLSyntaxHighlighter
from modules.line_numbered_editor import LineNumberedCodeEditor
from modules.logging_config import get_logger

logger = get_logger("warp_map_editor")

def apply_dark_theme(app):
    """Apply a comprehensive dark theme to the application"""
    dark_stylesheet = """
    /* Main application styling */
    QWidget {
        background-color: #2b2b2b;
        color: #ffffff;
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 9pt;
    }

    /* Main window and panels */
    QMainWindow {
        background-color: #1e1e1e;
    }

    /* Group boxes */
    QGroupBox {
        font-weight: bold;
        border: 2px solid #555555;
        border-radius: 5px;
        margin-top: 1ex;
        padding-top: 10px;
        background-color: #333333;
    }

    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px 0 5px;
        color: #ffffff;
    }

    /* Buttons */
    QPushButton {
        background-color: #404040;
        border: 1px solid #555555;
        border-radius: 4px;
        padding: 6px 12px;
        color: #ffffff;
        font-weight: bold;
    }

    QPushButton:hover {
        background-color: #4a4a4a;
        border-color: #777777;
    }

    QPushButton:pressed {
        background-color: #353535;
        border-color: #999999;
    }

    QPushButton:disabled {
        background-color: #2a2a2a;
        color: #666666;
        border-color: #444444;
    }

    /* Text inputs */
    QLineEdit, QTextEdit {
        background-color: #404040;
        border: 1px solid #555555;
        border-radius: 3px;
        padding: 4px;
        color: #ffffff;
        selection-background-color: #0078d4;
    }

    QLineEdit:focus, QTextEdit:focus {
        border-color: #0078d4;
        background-color: #454545;
    }

    /* Code editor specific styling */
    QTextEdit[objectName="code_editor"] {
        background-color: #1e1e1e;
        color: #d4d4d4;
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
        font-size: 10pt;
        border: 1px solid #3c3c3c;
        line-height: 1.4;
    }

    /* Combo boxes */
    QComboBox {
        background-color: #404040;
        border: 1px solid #555555;
        border-radius: 3px;
        padding: 4px 8px;
        color: #ffffff;
        min-width: 100px;
    }

    QComboBox:hover {
        border-color: #777777;
    }

    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 20px;
        border-left: 1px solid #555555;
        background-color: #505050;
    }

    QComboBox::down-arrow {
        image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 4px solid #ffffff;
        margin: 0px;
    }

    QComboBox QAbstractItemView {
        background-color: #404040;
        border: 1px solid #555555;
        selection-background-color: #0078d4;
        color: #ffffff;
    }

    /* List widgets */
    QListWidget {
        background-color: #353535;
        border: 1px solid #555555;
        border-radius: 3px;
        color: #ffffff;
        alternate-background-color: #3a3a3a;
    }

    QListWidget::item {
        padding: 6px;
        border-bottom: 1px solid #444444;
    }

    QListWidget::item:selected {
        background-color: #0078d4;
        color: #ffffff;
    }

    QListWidget::item:hover {
        background-color: #404040;
    }

    /* Sliders */
    QSlider::groove:horizontal {
        border: 1px solid #555555;
        height: 6px;
        background: #404040;
        border-radius: 3px;
    }

    QSlider::handle:horizontal {
        background: #0078d4;
        border: 1px solid #0078d4;
        width: 16px;
        margin: -6px 0;
        border-radius: 8px;
    }

    QSlider::handle:horizontal:hover {
        background: #106ebe;
    }

    /* Checkboxes */
    QCheckBox {
        color: #ffffff;
        spacing: 8px;
    }

    QCheckBox::indicator {
        width: 16px;
        height: 16px;
        border: 1px solid #555555;
        border-radius: 3px;
        background-color: #404040;
    }

    QCheckBox::indicator:checked {
        background-color: #0078d4;
        border-color: #0078d4;
    }

    QCheckBox::indicator:checked:hover {
        background-color: #106ebe;
    }

    /* Labels */
    QLabel {
        color: #ffffff;
        background: transparent;
    }

    /* Splitters */
    QSplitter::handle {
        background-color: #555555;
    }

    QSplitter::handle:horizontal {
        width: 3px;
    }

    QSplitter::handle:vertical {
        height: 3px;
    }

    /* Tab widget */
    QTabWidget::pane {
        border: 1px solid #555555;
        background-color: #333333;
    }

    QTabBar::tab {
        background-color: #404040;
        border: 1px solid #555555;
        padding: 6px 12px;
        margin-right: 2px;
        color: #ffffff;
    }

    QTabBar::tab:selected {
        background-color: #0078d4;
        border-bottom-color: #0078d4;
    }

    QTabBar::tab:hover {
        background-color: #4a4a4a;
    }

    /* Scrollbars */
    QScrollBar:vertical {
        background-color: #2b2b2b;
        width: 12px;
        border: none;
    }

    QScrollBar::handle:vertical {
        background-color: #555555;
        border-radius: 6px;
        min-height: 20px;
    }

    QScrollBar::handle:vertical:hover {
        background-color: #666666;
    }

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }

    QScrollBar:horizontal {
        background-color: #2b2b2b;
        height: 12px;
        border: none;
    }

    QScrollBar::handle:horizontal {
        background-color: #555555;
        border-radius: 6px;
        min-width: 20px;
    }

    QScrollBar::handle:horizontal:hover {
        background-color: #666666;
    }

    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        width: 0px;
    }

    /* Status labels */
    QLabel[objectName="preview_status_label"] {
        padding: 4px;
        border-radius: 3px;
        font-weight: bold;
    }
    
    QLabel[objectName="error_status_label"] {
        padding: 6px;
        border-radius: 3px;
        font-weight: bold;
        background-color: #333333;
        border: 1px solid #555555;
    }

    /* Menu bar (if present) */
    QMenuBar {
        background-color: #2b2b2b;
        color: #ffffff;
        border-bottom: 1px solid #555555;
    }

    QMenuBar::item {
        background: transparent;
        padding: 4px 8px;
    }

    QMenuBar::item:selected {
        background-color: #404040;
    }

    QMenu {
        background-color: #404040;
        border: 1px solid #555555;
        color: #ffffff;
    }

    QMenu::item {
        padding: 4px 20px;
    }

    QMenu::item:selected {
        background-color: #0078d4;
    }
    """

    app.setStyleSheet(dark_stylesheet)


class WarpMapPreviewWidget(QWidget):
    """Preview widget for warp map effects - shows helpful information"""

    def __init__(self):
        super().__init__()

        # Set minimum size and dark background
        self.setMinimumSize(300, 300)
        self.setStyleSheet("""
            WarpMapPreviewWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 2px solid #555555;
                border-radius: 5px;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
        """)

    def paintEvent(self, event):
        """Paint the preview widget with helpful information"""
        from PyQt5.QtGui import QPainter, QPen, QColor, QFont
        from PyQt5.QtCore import Qt

        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(43, 43, 43))

        # Set up text styling
        painter.setPen(QPen(QColor(255, 255, 255)))

        # Title font
        title_font = QFont("Segoe UI", 14, QFont.Bold)
        painter.setFont(title_font)

        # Draw title
        title_rect = self.rect()
        title_rect.setHeight(40)
        painter.drawText(title_rect, Qt.AlignCenter, "Warp Map Preview")

        # Body font
        body_font = QFont("Segoe UI", 10)
        painter.setFont(body_font)

        # Draw main message
        body_rect = self.rect()
        body_rect.setTop(50)
        body_rect.setBottom(body_rect.bottom() - 60)

        message = """To see your warp map in action:

1. Click '🚀 Launch Preview' to open the preview window
2. The preview shows a pink & blue checkerboard pattern
3. Animated yellow particles follow a sine wave pattern
4. Adjust Grid Density and Warp Intensity sliders
5. Watch your warp map distort both background and particles!

The editor validates your GLSL code and
provides real-time syntax checking.

Edit the code on the right to create
amazing visual distortion effects!"""

        painter.drawText(body_rect, Qt.AlignCenter | Qt.TextWordWrap, message)

        # Footer
        footer_font = QFont("Segoe UI", 9)
        painter.setFont(footer_font)
        painter.setPen(QPen(QColor(150, 150, 150)))

        footer_rect = self.rect()
        footer_rect.setTop(footer_rect.bottom() - 40)
        painter.drawText(footer_rect, Qt.AlignCenter, "✨ Tip: Yellow particles and checkerboard show warp distortion beautifully!")

    def update_warp_code(self, warp_code: str):
        """Store the warp map code (no preview rendering)"""
        # Just store the code - no OpenGL rendering needed
        pass

    def set_warp_intensity(self, intensity: float):
        """Set warp effect intensity (placeholder)"""
        pass

    def set_animation_enabled(self, enabled: bool):
        """Enable or disable animation (placeholder)"""
        pass

    def set_grid_density(self, density: int):
        """Set grid density (placeholder)"""
        pass


# All OpenGL methods removed - using simple preview widget now


class WarpMapEditor(QWidget):
    """Warp map editor widget"""

    warp_map_changed = pyqtSignal(str)  # Emitted when a warp map is modified

    def __init__(self, warp_map_manager: WarpMapManager, show_preview: bool = True, shader_compiler=None):
        super().__init__()
        self.warp_map_manager = warp_map_manager
        self.current_warp_map: Optional[WarpMapInfo] = None
        self.current_warp_map_key: Optional[str] = None  # Store the filename key
        self.unsaved_changes = False
        self.preview_callback = None  # Callback for live preview
        self.show_preview = show_preview  # Control whether to show preview section
        self.shader_compiler = shader_compiler  # For syntax validation

        # Timer for debouncing live updates
        self.live_update_timer = QTimer()
        self.live_update_timer.setSingleShot(True)
        self.live_update_timer.timeout.connect(self.apply_pending_live_changes)
        self.pending_temp_warp_map = None

        # Flag to track if widget is being destroyed
        self._is_destroying = False

        self.init_ui()
        self.load_warp_map_list()

        # Load template code by default for immediate preview (delayed)
        if self.show_preview:
            QTimer.singleShot(
                100,
                lambda: (
                    self.load_default_template() if not self._is_destroying else None
                ),
            )

    def load_default_template(self):
        """Load the default template code for immediate preview"""
        # Temporarily disconnect change handlers to avoid triggering unsaved changes
        self.code_editor.textChanged.disconnect(self.on_code_changed)
        
        self.code_editor.setPlainText(self.get_template_code())
        self.update_preview()
        
        # Reconnect change handlers
        self.code_editor.textChanged.connect(self.on_code_changed)
        
        # Ensure unsaved_changes is False after loading template
        self.unsaved_changes = False
        self.update_ui_state()

    def cleanup(self):
        """Clean up resources, especially timers"""
        self._is_destroying = True
        try:
            if hasattr(self, "live_update_timer") and self.live_update_timer is not None:
                try:
                    # Check if the timer is still valid before accessing it
                    self.live_update_timer.stop()
                    self.live_update_timer.deleteLater()
                except RuntimeError:
                    # Timer was already deleted by Qt's cleanup
                    pass
                finally:
                    self.live_update_timer = None
        except Exception as e:
            logger.warning("Error cleaning up WarpMapEditor timer: {e}")

    def closeEvent(self, event):
        """Handle close event"""
        self.cleanup()
        super().closeEvent(event)

    def __del__(self):
        """Destructor - ensure cleanup"""
        try:
            self.cleanup()
        except:
            pass

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Warp Map Editor")
        self.setGeometry(100, 100, 1200, 800)

        # Main layout
        main_layout = QHBoxLayout(self)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel - Warp map list and controls
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)

        # Right panel - Editor and preview
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)

        # Set splitter proportions
        splitter.setSizes([300, 900])

    def create_left_panel(self) -> QWidget:
        """Create the left panel with warp map list and controls"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Warp map list
        list_group = QGroupBox("Warp Maps")
        list_layout = QVBoxLayout(list_group)

        self.warp_map_list = QListWidget()
        self.warp_map_list.itemClicked.connect(self.on_warp_map_selected)
        list_layout.addWidget(self.warp_map_list)

        # Control buttons
        button_layout = QHBoxLayout()

        self.new_button = QPushButton("New")
        self.new_button.clicked.connect(self.new_warp_map)
        button_layout.addWidget(self.new_button)

        self.duplicate_button = QPushButton("Duplicate")
        self.duplicate_button.clicked.connect(self.duplicate_warp_map)
        button_layout.addWidget(self.duplicate_button)

        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self.delete_warp_map)
        button_layout.addWidget(self.delete_button)

        list_layout.addLayout(button_layout)
        layout.addWidget(list_group)

        # Import/Export buttons
        io_layout = QHBoxLayout()

        self.import_button = QPushButton("Import")
        self.import_button.clicked.connect(self.import_warp_map)
        io_layout.addWidget(self.import_button)

        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self.export_warp_map)
        io_layout.addWidget(self.export_button)

        layout.addLayout(io_layout)

        # Filter controls
        filter_group = QGroupBox("Filters")
        filter_layout = QVBoxLayout(filter_group)

        # Search filter
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search:"))
        self.search_filter = QLineEdit()
        self.search_filter.setPlaceholderText("Filter by name, category, or description...")
        self.search_filter.textChanged.connect(self.apply_filters)
        search_layout.addWidget(self.search_filter)
        filter_layout.addLayout(search_layout)

        # Category filter
        category_layout = QHBoxLayout()
        category_layout.addWidget(QLabel("Category:"))
        self.category_filter = QComboBox()
        self.category_filter.addItem("All Categories")
        self.category_filter.currentTextChanged.connect(self.apply_filters)
        category_layout.addWidget(self.category_filter)
        filter_layout.addLayout(category_layout)

        layout.addWidget(filter_group)

        return panel

    def create_right_panel(self) -> QWidget:
        """Create the right panel with editor, preview, and metadata"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        if self.show_preview:
            # Create horizontal splitter for editor and preview
            editor_splitter = QSplitter(Qt.Horizontal)
            layout.addWidget(editor_splitter)

            # Left side - Tab widget for editor and metadata
            self.tab_widget = QTabWidget()
            editor_splitter.addWidget(self.tab_widget)

            # Editor tab
            editor_tab = self.create_editor_tab()
            self.tab_widget.addTab(editor_tab, "Editor")

            # Metadata tab
            metadata_tab = self.create_metadata_tab()
            self.tab_widget.addTab(metadata_tab, "Properties")

            # Right side - Preview pane
            preview_panel = self.create_preview_panel()
            editor_splitter.addWidget(preview_panel)

            # Set splitter proportions (60% editor, 40% preview)
            editor_splitter.setSizes([600, 400])
        else:
            # No preview - just tab widget for editor and metadata
            self.tab_widget = QTabWidget()
            layout.addWidget(self.tab_widget)

            # Editor tab
            editor_tab = self.create_editor_tab()
            self.tab_widget.addTab(editor_tab, "Editor")

            # Metadata tab
            metadata_tab = self.create_metadata_tab()
            self.tab_widget.addTab(metadata_tab, "Properties")

        # Save/Revert buttons
        button_layout = QHBoxLayout()

        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_current_warp_map)
        self.save_button.setEnabled(False)
        button_layout.addWidget(self.save_button)

        self.revert_button = QPushButton("Revert")
        self.revert_button.clicked.connect(self.revert_changes)
        self.revert_button.setEnabled(False)
        button_layout.addWidget(self.revert_button)

        # Test compilation button
        self.test_button = QPushButton("Test Compilation")
        self.test_button.setObjectName("test_button")
        self.test_button.clicked.connect(self.test_compilation)
        button_layout.addWidget(self.test_button)

        # Error status label (no manual validate button - auto-check on save/update)
        self.error_status_label = QLabel("")
        self.error_status_label.setObjectName("error_status_label")
        self.error_status_label.setWordWrap(True)
        self.error_status_label.setMaximumHeight(60)
        button_layout.addWidget(self.error_status_label)

        # Add auto-update checkbox for integrated mode
        if not self.show_preview:
            self.auto_update_checkbox = QCheckBox("Auto-apply changes")
            self.auto_update_checkbox.setChecked(True)
            self.auto_update_checkbox.setToolTip("Automatically apply warp map changes to the visualizer as you type")
            button_layout.addWidget(self.auto_update_checkbox)

        # Only add preview button if preview is enabled
        if self.show_preview:
            self.preview_button = QPushButton("Preview")
            self.preview_button.clicked.connect(self.apply_current_warp_map_preview)
            self.preview_button.setEnabled(False)
            button_layout.addWidget(self.preview_button)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        return panel

    def create_editor_tab(self) -> QWidget:
        """Create the GLSL editor tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # GLSL code editor
        editor_group = QGroupBox("GLSL Code")
        editor_layout = QVBoxLayout(editor_group)

        self.code_editor = LineNumberedCodeEditor()
        self.code_editor.setObjectName("code_editor")  # For dark theme styling
        self.code_editor.textChanged.connect(self.on_code_changed)
        self.code_editor.error_line_changed.connect(self.on_error_line_changed)

        # Add syntax highlighting
        self.highlighter = GLSLSyntaxHighlighter(self.code_editor.document())

        editor_layout.addWidget(self.code_editor)

        # Template and help
        help_layout = QHBoxLayout()

        template_button = QPushButton("Insert Template")
        template_button.clicked.connect(self.insert_template)
        help_layout.addWidget(template_button)

        help_button = QPushButton("GLSL Help")
        help_button.clicked.connect(self.show_glsl_help)
        help_layout.addWidget(help_button)

        help_layout.addStretch()
        editor_layout.addLayout(help_layout)

        layout.addWidget(editor_group)

        return tab

    def create_metadata_tab(self) -> QWidget:
        """Create the metadata editing tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Metadata form
        metadata_group = QGroupBox("Warp Map Properties")
        form_layout = QFormLayout(metadata_group)

        self.name_edit = QLineEdit()
        self.name_edit.textChanged.connect(self.on_metadata_changed)
        form_layout.addRow("Name:", self.name_edit)

        self.category_edit = QComboBox()
        self.category_edit.setEditable(True)
        self.category_edit.currentTextChanged.connect(self.on_metadata_changed)
        form_layout.addRow("Category:", self.category_edit)

        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(100)
        self.description_edit.textChanged.connect(self.on_metadata_changed)
        form_layout.addRow("Description:", self.description_edit)

        self.complexity_edit = QComboBox()
        self.complexity_edit.addItems(["low", "medium", "high"])
        self.complexity_edit.currentTextChanged.connect(self.on_metadata_changed)
        form_layout.addRow("Complexity:", self.complexity_edit)

        self.author_edit = QLineEdit()
        self.author_edit.textChanged.connect(self.on_metadata_changed)
        form_layout.addRow("Author:", self.author_edit)

        self.version_edit = QLineEdit()
        self.version_edit.textChanged.connect(self.on_metadata_changed)
        form_layout.addRow("Version:", self.version_edit)

        layout.addWidget(metadata_group)
        layout.addStretch()

        return tab

    def create_preview_panel(self) -> QWidget:
        """Create the preview panel with OpenGL widget and controls"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Preview group
        preview_group = QGroupBox("Live Preview")
        preview_layout = QVBoxLayout(preview_group)

        # OpenGL preview widget
        self.preview_widget = WarpMapPreviewWidget()
        self.preview_widget.setMinimumSize(300, 300)
        preview_layout.addWidget(self.preview_widget)

        # Preview controls
        controls_layout = QVBoxLayout()

        # Animation toggle
        self.animation_checkbox = QCheckBox("Enable Animation")
        self.animation_checkbox.setChecked(True)
        self.animation_checkbox.toggled.connect(self.preview_widget.set_animation_enabled)
        controls_layout.addWidget(self.animation_checkbox)

        # Warp intensity slider
        intensity_layout = QHBoxLayout()
        intensity_layout.addWidget(QLabel("Intensity:"))
        self.intensity_slider = QSlider(Qt.Horizontal)
        self.intensity_slider.setRange(0, 100)
        self.intensity_slider.setValue(50)
        self.intensity_slider.valueChanged.connect(self.on_intensity_changed)
        intensity_layout.addWidget(self.intensity_slider)
        self.intensity_label = QLabel("0.50")
        intensity_layout.addWidget(self.intensity_label)
        controls_layout.addLayout(intensity_layout)

        # Grid density slider
        density_layout = QHBoxLayout()
        density_layout.addWidget(QLabel("Grid Density:"))
        self.density_slider = QSlider(Qt.Horizontal)
        self.density_slider.setRange(5, 50)
        self.density_slider.setValue(20)
        self.density_slider.valueChanged.connect(self.on_density_changed)
        density_layout.addWidget(self.density_slider)
        self.density_label = QLabel("20")
        density_layout.addWidget(self.density_label)
        controls_layout.addLayout(density_layout)

        # Auto-update toggle
        self.auto_update_checkbox = QCheckBox("Auto-update on code change")
        self.auto_update_checkbox.setChecked(True)
        controls_layout.addWidget(self.auto_update_checkbox)

        # Preview buttons
        preview_buttons_layout = QHBoxLayout()

        self.update_preview_button = QPushButton("Update Preview")
        self.update_preview_button.clicked.connect(self.update_preview)
        preview_buttons_layout.addWidget(self.update_preview_button)

        self.launch_preview_button = QPushButton("🚀 Launch Preview")
        self.launch_preview_button.clicked.connect(self.launch_preview_window)
        self.launch_preview_button.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                font-weight: bold;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
        """)
        preview_buttons_layout.addWidget(self.launch_preview_button)

        self.clear_selection_button = QPushButton("🚫 Clear Selection")
        self.clear_selection_button.clicked.connect(self.clear_warp_map_selection)
        self.clear_selection_button.setStyleSheet("""
            QPushButton {
                background-color: #d13438;
                font-weight: bold;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #b71c1c;
            }
        """)
        preview_buttons_layout.addWidget(self.clear_selection_button)

        controls_layout.addLayout(preview_buttons_layout)

        preview_layout.addLayout(controls_layout)
        layout.addWidget(preview_group)

        # Status label
        self.preview_status_label = QLabel("Ready")
        self.preview_status_label.setObjectName("preview_status_label")
        self.preview_status_label.setStyleSheet("color: #00ff00;")  # Green for ready state
        layout.addWidget(self.preview_status_label)

        layout.addStretch()
        return panel

    def on_intensity_changed(self, value: int):
        """Handle intensity slider change"""
        intensity = value / 100.0
        self.intensity_label.setText(f"{intensity:.2f}")
        self.preview_widget.set_warp_intensity(intensity)
        # Update preview window controls
        self.update_preview_controls()

    def on_density_changed(self, value: int):
        """Handle grid density slider change"""
        self.density_label.setText(str(value))
        self.preview_widget.set_grid_density(value)
        # Update preview window controls
        self.update_preview_controls()

    def update_preview_controls(self):
        """Update the preview window controls via temp file"""
        try:
            import json
            controls = {
                'grid_density': self.density_slider.value(),
                'warp_intensity': self.intensity_slider.value() / 100.0
            }
            with open('temp_warp_controls.json', 'w') as f:
                json.dump(controls, f)
        except Exception:
            # Silently ignore errors
            pass

    def update_preview(self):
        """Update the preview with current code"""
        if self.show_preview and hasattr(self, 'preview_widget'):
            code = self.code_editor.toPlainText()
            try:
                self.preview_widget.update_warp_code(code)
                if hasattr(self, 'preview_status_label'):
                    self.preview_status_label.setText("Preview updated")
                    self.preview_status_label.setStyleSheet("color: #00ff00;")  # Green
            except Exception as e:
                if hasattr(self, 'preview_status_label'):
                    self.preview_status_label.setText(f"Error: {str(e)}")
                    self.preview_status_label.setStyleSheet("color: #ff4444;")  # Red

        # Also update the temp file if preview window is running
        if self.show_preview:
            self.update_preview_temp_file()

    def load_warp_map_list(self):
        """Load the list of available warp maps"""
        self.warp_map_list.clear()

        # Update category filter
        sorted_categories = sorted(self.warp_map_manager.get_categories())
        categories = ["All Categories"] + sorted_categories
        current_category = self.category_filter.currentText()
        self.category_filter.clear()
        self.category_filter.addItems(categories)
        if current_category in categories:
            self.category_filter.setCurrentText(current_category)

        # Update category combo in metadata tab
        self.category_edit.clear()
        self.category_edit.addItems(sorted_categories)

        # Load warp maps and sort them alphabetically by name
        warp_maps = self.warp_map_manager.get_all_warp_maps()
        sorted_warp_maps = sorted(warp_maps, key=lambda wm: wm.name.lower())

        for warp_map in sorted_warp_maps:
            item = QListWidgetItem(f"{warp_map.name} ({warp_map.category})")
            # Store the filename key instead of display name for proper lookup
            filename_key = self.warp_map_manager.get_warp_map_key(warp_map)
            item.setData(Qt.UserRole, filename_key)
            # Store additional data for filtering
            item.setData(Qt.UserRole + 1, warp_map.category)
            item.setData(Qt.UserRole + 2, warp_map.description)
            item.setData(Qt.UserRole + 3, warp_map.name)
            self.warp_map_list.addItem(item)

    def apply_filters(self):
        """Apply both search and category filters to the warp map list"""
        search_text = self.search_filter.text().lower()
        category_filter = self.category_filter.currentText()
        
        for i in range(self.warp_map_list.count()):
            item = self.warp_map_list.item(i)
            if not item:
                continue
                
            try:
                # Get stored data
                warp_map_name = item.data(Qt.UserRole + 3) or ""
                category = item.data(Qt.UserRole + 1) or ""
                description = item.data(Qt.UserRole + 2) or ""
                
                # Apply search filter
                search_match = True
                if search_text:
                    search_match = (
                        search_text in warp_map_name.lower() or
                        search_text in category.lower() or
                        search_text in description.lower()
                    )
                
                # Apply category filter
                category_match = (
                    category_filter == "All Categories" or
                    category == category_filter
                )
                
                # Show item only if both filters match
                item.setHidden(not (search_match and category_match))
                
            except RuntimeError:
                # Item has been deleted, skip it
                continue

    def filter_by_category(self, category: str):
        """Legacy method for backward compatibility - now calls apply_filters"""
        self.apply_filters()

    def on_warp_map_selected(self, item: QListWidgetItem):
        """Handle warp map selection"""
        # Safety check: ensure the item is still valid
        if not item or not hasattr(item, 'data'):
            return
            
        # Get the warp map key BEFORE potentially saving (which clears the list)
        try:
            warp_map_key = item.data(Qt.UserRole)
            if not warp_map_key:
                return
        except RuntimeError:
            # Item has been deleted, ignore the selection
            return
            
        if self.unsaved_changes:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Do you want to save them?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )

            if reply == QMessageBox.Yes:
                if not self.save_current_warp_map():
                    return
            elif reply == QMessageBox.Cancel:
                return

        # Use the warp map key we retrieved earlier
        self.load_warp_map(warp_map_key)

    def load_warp_map(self, key: str):
        """Load a warp map into the editor by filename key"""
        warp_map = self.warp_map_manager.get_warp_map(key)
        if not warp_map:
            return

        self.current_warp_map = warp_map
        self.current_warp_map_key = key  # Store the filename key

        # Temporarily disconnect change handlers to avoid triggering unsaved changes
        self.code_editor.textChanged.disconnect(self.on_code_changed)
        self.name_edit.textChanged.disconnect(self.on_metadata_changed)
        self.category_edit.currentTextChanged.disconnect(self.on_metadata_changed)
        self.description_edit.textChanged.disconnect(self.on_metadata_changed)
        self.complexity_edit.currentTextChanged.disconnect(self.on_metadata_changed)
        self.author_edit.textChanged.disconnect(self.on_metadata_changed)
        self.version_edit.textChanged.disconnect(self.on_metadata_changed)

        # Load into editor
        self.code_editor.setPlainText(warp_map.glsl_code)
        
        # Set line number offset based on injection line in main shader
        if self.shader_compiler:
            injection_line = self.shader_compiler.get_injection_line_for_warp(key)
            self.code_editor.set_line_number_offset(injection_line - 1)

        # Load metadata
        self.name_edit.setText(warp_map.name)
        self.category_edit.setCurrentText(warp_map.category)
        self.description_edit.setPlainText(warp_map.description)
        self.complexity_edit.setCurrentText(warp_map.complexity)
        self.author_edit.setText(warp_map.author)
        self.version_edit.setText(warp_map.version)

        # Reconnect change handlers
        self.code_editor.textChanged.connect(self.on_code_changed)
        self.name_edit.textChanged.connect(self.on_metadata_changed)
        self.category_edit.currentTextChanged.connect(self.on_metadata_changed)
        self.description_edit.textChanged.connect(self.on_metadata_changed)
        self.complexity_edit.currentTextChanged.connect(self.on_metadata_changed)
        self.author_edit.textChanged.connect(self.on_metadata_changed)
        self.version_edit.textChanged.connect(self.on_metadata_changed)

        self.unsaved_changes = False
        self.update_ui_state()

        # Check for compilation errors after loading
        self.check_compilation_errors()

        # Update the preview with the loaded warp map
        self.update_preview()

        # Apply warp map for live preview (always apply if preview callback exists)
        if self.preview_callback:
            self.apply_warp_map_preview(warp_map)

        # Update status to show locked application (only if preview UI exists)
        if self.show_preview and hasattr(self, 'preview_status_label'):
            self.preview_status_label.setText(f"🔒 Applied '{warp_map.name}' - automatic changes paused")
            self.preview_status_label.setStyleSheet("color: #00ff00;")  # Green

            # Reset status after 5 seconds (only if not destroying)
            if not self._is_destroying:
                QTimer.singleShot(
                    5000,
                    lambda: (
                        self.preview_status_label.setText("Ready")
                        if not self._is_destroying
                        else None
                    ),
                )

    def on_code_changed(self):
        """Handle code editor changes"""
        # Only set unsaved_changes if we have a current warp map loaded
        if self.current_warp_map is not None:
            self.unsaved_changes = True
        self.update_ui_state()

        # Auto-update preview if enabled and preview is shown
        if self.show_preview and hasattr(self, 'auto_update_checkbox') and self.auto_update_checkbox.isChecked():
            self.update_preview()

        # Auto-apply changes to main visualizer if in integrated mode and auto-update is enabled
        elif not self.show_preview and hasattr(self, 'auto_update_checkbox') and self.auto_update_checkbox.isChecked():
            if self.current_warp_map and self.preview_callback and not self._is_destroying:
                # Create a temporary warp map with the current code
                self.pending_temp_warp_map = WarpMapInfo(
                    name=self.current_warp_map.name,
                    category=self.current_warp_map.category,
                    description=self.current_warp_map.description,
                    glsl_code=self.code_editor.toPlainText(),
                    complexity=self.current_warp_map.complexity,
                    author=self.current_warp_map.author,
                    version=self.current_warp_map.version,
                    is_builtin=self.current_warp_map.is_builtin,
                    file_path=self.current_warp_map.file_path
                )

                # Debounce the live updates (wait 500ms after last change) - check if timer is still valid
                try:
                    if hasattr(self, "live_update_timer") and self.live_update_timer is not None:
                        self.live_update_timer.stop()
                        self.live_update_timer.start(500)
                except RuntimeError:
                    # Timer was already deleted, ignore
                    pass
        else:
            # Check for compilation errors even when not in live update mode
            self.check_compilation_errors()

    def on_metadata_changed(self):
        """Handle metadata changes"""
        # Only set unsaved_changes if we have a current warp map loaded
        if self.current_warp_map is not None:
            self.unsaved_changes = True
        self.update_ui_state()

    def update_ui_state(self):
        """Update UI state based on current conditions"""
        has_warp_map = self.current_warp_map is not None

        self.save_button.setEnabled(has_warp_map and self.unsaved_changes)
        self.revert_button.setEnabled(has_warp_map and self.unsaved_changes)
        self.duplicate_button.setEnabled(has_warp_map)
        self.delete_button.setEnabled(has_warp_map and not self.current_warp_map.is_builtin)
        self.export_button.setEnabled(has_warp_map)

        # Only update preview button if it exists
        if hasattr(self, 'preview_button'):
            self.preview_button.setEnabled(has_warp_map and self.preview_callback is not None)

        # Update window title
        title = "Warp Map Editor"
        if self.current_warp_map:
            title += f" - {self.current_warp_map.name}"
            if self.unsaved_changes:
                title += " *"
        self.setWindowTitle(title)

    def save_current_warp_map(self) -> bool:
        """Save the current warp map"""
        if not self.current_warp_map:
            return False

        # Check for any recent GLSL compilation errors
        if hasattr(self, 'shader_compiler') and self.shader_compiler and self.current_warp_map_key:
            errors = self.shader_compiler.get_latest_errors_for_warp(self.current_warp_map_key)
            if errors:
                # Set errors in editor for highlighting
                self.code_editor.clear_errors()
                self.code_editor.set_errors_from_main_shader(errors)
                
                error_msg = f"GLSL compilation errors found:\n" + "\n".join([f"Line {line}: {msg}" for line, msg in errors[:3]])
                reply = QMessageBox.question(
                    self, 
                    "Syntax Errors", 
                    f"{error_msg}\n\nDo you want to save anyway?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return False

        # Update warp map with current data
        self.current_warp_map.name = self.name_edit.text().strip()
        self.current_warp_map.category = self.category_edit.currentText().strip()
        self.current_warp_map.description = self.description_edit.toPlainText().strip()
        self.current_warp_map.complexity = self.complexity_edit.currentText()
        self.current_warp_map.author = self.author_edit.text().strip()
        self.current_warp_map.version = self.version_edit.text().strip()
        self.current_warp_map.glsl_code = self.code_editor.toPlainText()

        # Validate required fields
        if not self.current_warp_map.name:
            QMessageBox.warning(self, "Validation Error", "Name is required.")
            return False

        # Save to disk
        if self.warp_map_manager.save_warp_map(self.current_warp_map, overwrite=True):
            self.unsaved_changes = False
            self.update_ui_state()
            self.load_warp_map_list()
            self.warp_map_changed.emit(self.current_warp_map.name)
            QMessageBox.information(self, "Success", "Warp map saved successfully.")
            return True
        else:
            QMessageBox.critical(self, "Error", "Failed to save warp map.")
            return False

    def revert_changes(self):
        """Revert changes to the current warp map"""
        if self.current_warp_map and self.current_warp_map_key:
            # Reload the original warp map from disk
            self.warp_map_manager.load_all_warp_maps()  # Refresh from disk
            self.load_warp_map(self.current_warp_map_key)
        elif not self.current_warp_map_key:
            logger.debug(f"No filename key available for current warp map")

    def test_compilation(self):
        """Test compile the current warp map to generate error information"""
        if not self.current_warp_map or not self.shader_compiler:
            self.error_status_label.setText("No warp map loaded or shader compiler not available")
            self.error_status_label.setStyleSheet("color: #ff8800;")  # Orange
            return
        
        # Create a temporary warp map with current editor content
        temp_warp_map = WarpMapInfo(
            name=self.current_warp_map.name,
            category=self.current_warp_map.category,
            description=self.current_warp_map.description,
            glsl_code=self.code_editor.toPlainText(),
            complexity=self.current_warp_map.complexity,
            author=self.current_warp_map.author,
            version=self.current_warp_map.version,
            is_builtin=self.current_warp_map.is_builtin,
            file_path=self.current_warp_map.file_path
        )
        
        # Force a test compilation
        if self.preview_callback:
            try:
                self.error_status_label.setText("Testing compilation...")
                self.error_status_label.setStyleSheet("color: #ffff00;")  # Yellow
                self.preview_callback(temp_warp_map, persistent=False)  # Non-persistent test
                # Check for errors after compilation
                self.check_compilation_errors()
            except Exception as e:
                self.error_status_label.setText(f"Test compilation failed: {str(e)}")
                self.error_status_label.setStyleSheet("color: #ff4444;")  # Red
        else:
            self.error_status_label.setText("No preview callback available for testing")
            self.error_status_label.setStyleSheet("color: #ff8800;")  # Orange

    def check_compilation_errors(self):
        """Check for recent GLSL compilation errors and display them"""
        if hasattr(self, 'shader_compiler') and self.shader_compiler and self.current_warp_map_key:
            errors = self.shader_compiler.get_latest_errors_for_warp(self.current_warp_map_key)
            if errors:
                # Clear previous errors in editor
                self.code_editor.clear_errors()
                
                # Set errors using main shader line numbers (will be converted to editor line numbers)
                self.code_editor.set_errors_from_main_shader(errors)
                # Show first error in status
                first_error = errors[0]
                self.error_status_label.setText(f"Line {first_error[0]}: {first_error[1]}")
                self.error_status_label.setStyleSheet("color: #ff4444;")  # Red
            else:
                self.error_status_label.setText("No compilation errors")
                self.error_status_label.setStyleSheet("color: #00ff00;")  # Green
                self.code_editor.clear_errors()
        else:
            self.error_status_label.setText("⚪ Ready")
            self.error_status_label.setStyleSheet("color: #cccccc;")  # Gray
    
    def on_error_line_changed(self, line_number: int):
        """Handle when cursor moves to an error line"""
        error_message = self.code_editor.get_error_message(line_number)
        if error_message:
            self.error_status_label.setText(f"Line {line_number}: {error_message}")
            self.error_status_label.setStyleSheet("color: #ff4444;")  # Red

    def get_template_code(self) -> str:
        """Get template GLSL code for new warp maps"""
        return """// Animated wave warp map template
vec2 get_pattern(vec2 pos, float t) {
    // pos: current pixel position (0.0 to 1.0)
    // t: time variable for animation

    // Center the coordinates around (0.5, 0.5)
    vec2 centered = pos - 0.5;

    // Create animated wave distortion
    float wave_freq = 8.0;
    float wave_speed = 2.0;
    float wave_amplitude = 0.05;

    // Horizontal waves based on Y position
    float wave_x = sin(pos.y * wave_freq + t * wave_speed) * wave_amplitude;

    // Vertical waves based on X position
    float wave_y = cos(pos.x * wave_freq + t * wave_speed) * wave_amplitude;

    // Add some circular distortion for more interesting effects
    float dist = length(centered);
    float radial_wave = sin(dist * 15.0 + t * 3.0) * 0.02;

    return vec2(wave_x + radial_wave, wave_y + radial_wave);
}"""

    def insert_template(self):
        """Insert template code into the editor"""
        if self.code_editor.toPlainText().strip():
            reply = QMessageBox.question(
                self, "Insert Template",
                "This will replace the current code. Continue?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        self.code_editor.setPlainText(self.get_template_code())
        self.update_preview()

    def show_glsl_help(self):
        """Show GLSL help dialog"""
        help_text = """
GLSL Warp Map Help

Function Signature:
vec2 get_pattern(vec2 pos, float t)

Parameters:
- pos: Current pixel position (0.0 to 1.0 for both x and y)
- t: Time variable for animation
- Return: vec2 offset to apply to the pixel position

Common GLSL Functions:
- sin(x), cos(x), tan(x): Trigonometric functions
- atan(y, x): Arc tangent of y/x
- length(v): Length of vector v
- normalize(v): Normalize vector v to unit length
- mix(a, b, t): Linear interpolation between a and b
- clamp(x, min, max): Clamp x to [min, max] range
- smoothstep(edge0, edge1, x): Smooth interpolation
- abs(x): Absolute value
- sign(x): Sign of x (-1, 0, or 1)
- floor(x), ceil(x): Floor and ceiling functions
- fract(x): Fractional part of x
- mod(x, y): Modulo operation

Example Patterns:
1. Spiral: Use atan(pos.y-0.5, pos.x-0.5) for angle
2. Waves: Use sin/cos with pos.x or pos.y
3. Radial: Use length(pos - 0.5) for distance from center
4. Grid: Use mod() for repeating patterns

Tips:
- Keep offsets small (typically < 0.1) for subtle effects
- Use time variable 't' for animation
- Combine multiple functions for complex patterns
- Test with different complexity levels
"""

        msg = QMessageBox(self)
        msg.setWindowTitle("GLSL Help")
        msg.setText(help_text)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def import_warp_map(self):
        """Import a warp map from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Warp Map", "", "GLSL Files (*.glsl);;All Files (*)"
        )

        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    glsl_code = f.read()

                # Create new warp map
                import os
                name = os.path.splitext(os.path.basename(file_path))[0]

                new_warp_map = WarpMapInfo(
                    name=name,
                    category="imported",
                    description=f"Imported from {os.path.basename(file_path)}",
                    glsl_code=glsl_code,
                    complexity="medium",
                    author="Imported",
                    version="1.0",
                    is_builtin=False
                )

                self.current_warp_map = new_warp_map
                self.load_warp_map_data(new_warp_map)
                self.unsaved_changes = True
                self.update_ui_state()

                QMessageBox.information(self, "Success", f"Imported warp map '{name}'. Don't forget to save!")

            except Exception as e:
                QMessageBox.critical(self, "Import Error", f"Failed to import warp map:\n{str(e)}")

    def export_warp_map(self):
        """Export the current warp map to file"""
        if not self.current_warp_map:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Warp Map", f"{self.current_warp_map.name}.glsl",
            "GLSL Files (*.glsl);;All Files (*)"
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.current_warp_map.glsl_code)

                QMessageBox.information(self, "Success", f"Exported warp map to {file_path}")

            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export warp map:\n{str(e)}")

    def load_warp_map_data(self, warp_map: WarpMapInfo):
        """Load warp map data into the editor (helper method)"""
        self.code_editor.setPlainText(warp_map.glsl_code)
        self.name_edit.setText(warp_map.name)
        self.category_edit.setCurrentText(warp_map.category)
        self.description_edit.setPlainText(warp_map.description)
        self.complexity_edit.setCurrentText(warp_map.complexity)
        self.author_edit.setText(warp_map.author)
        self.version_edit.setText(warp_map.version)

    def new_warp_map(self):
        """Create a new warp map"""
        if self.unsaved_changes:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Do you want to save them?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )

            if reply == QMessageBox.Yes:
                if not self.save_current_warp_map():
                    return
            elif reply == QMessageBox.Cancel:
                return

        # Create new warp map
        new_warp_map = WarpMapInfo(
            name="New Warp Map",
            category="custom",
            description="A new custom warp map",
            glsl_code=self.get_template_code(),
            complexity="medium",
            author="User",
            version="1.0",
            is_builtin=False
        )

        self.current_warp_map = new_warp_map
        self.load_warp_map_data(new_warp_map)
        self.unsaved_changes = True
        self.update_ui_state()

    def duplicate_warp_map(self):
        """Duplicate the current warp map"""
        if not self.current_warp_map:
            return

        # Create duplicate
        duplicate = WarpMapInfo(
            name=f"{self.current_warp_map.name} Copy",
            category=self.current_warp_map.category,
            description=f"Copy of {self.current_warp_map.description}",
            glsl_code=self.current_warp_map.glsl_code,
            complexity=self.current_warp_map.complexity,
            author=self.current_warp_map.author,
            version=self.current_warp_map.version,
            is_builtin=False
        )

        self.current_warp_map = duplicate
        self.load_warp_map_data(duplicate)
        self.unsaved_changes = True
        self.update_ui_state()

    def delete_warp_map(self):
        """Delete the current warp map"""
        if not self.current_warp_map or self.current_warp_map.is_builtin:
            return

        reply = QMessageBox.question(
            self, "Delete Warp Map",
            f"Are you sure you want to delete '{self.current_warp_map.name}'?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            if self.warp_map_manager.delete_warp_map(self.current_warp_map.name):
                self.current_warp_map = None
                self.code_editor.clear()
                self.name_edit.clear()
                self.description_edit.clear()
                self.unsaved_changes = False
                self.update_ui_state()
                self.load_warp_map_list()
                QMessageBox.information(self, "Success", "Warp map deleted successfully.")
            else:
                QMessageBox.critical(self, "Error", "Failed to delete warp map.")

    def set_preview_callback(self, callback):
        """Set the callback function for warp map preview"""
        self.preview_callback = callback
        logger.debug(f"Warp map editor preview callback {'set' if callback else 'cleared'}")
        # Update UI state to enable/disable preview button
        self.update_ui_state()

    def apply_warp_map_preview(self, warp_map: WarpMapInfo):
        """Apply a warp map for persistent preview in the visualizer"""
        if self.preview_callback and warp_map and self.current_warp_map_key:
            try:
                # Use the stored filename key
                self.preview_callback(self.current_warp_map_key, persistent=True)
                logger.debug(f"Applied warp map '{warp_map.name}' (key: {self.current_warp_map_key}) persistently via callback")
            except Exception as e:
                logger.errror(f"Error applying warp map preview: {e}")
        elif not self.preview_callback:
            logger.debug(f"No preview callback available to apply warp map '{warp_map.name if warp_map else 'None'}'")
        elif not warp_map:
            logger.debug(f"No warp map provided to apply")
        elif not self.current_warp_map_key:
            logger.debug(f"No filename key available for warp map '{warp_map.name if warp_map else 'None'}'")

    def apply_current_warp_map_preview(self):
        """Apply the current warp map for preview"""
        if self.current_warp_map:
            self.apply_warp_map_preview(self.current_warp_map)

    def apply_live_warp_map_changes(self, temp_warp_map: WarpMapInfo):
        """Apply live warp map changes to the visualizer without saving to disk"""
        if self.preview_callback and self.current_warp_map_key:
            try:
                # Temporarily update the warp map in the manager with the new code
                original_warp_map = self.warp_map_manager.warp_maps[self.current_warp_map_key]
                self.warp_map_manager.warp_maps[self.current_warp_map_key] = temp_warp_map

                # Apply the updated warp map to the visualizer
                self.preview_callback(self.current_warp_map_key, persistent=True)

                # Note: We don't restore the original here because we want the live changes
                # The original will be restored when the user reverts or loads a different warp map

            except Exception as e:
                logger.errror(f"Error applying live warp map changes: {e}")
                
                
        elif not self.current_warp_map_key:
            logger.debug(f"No filename key available for current warp map")
        elif not self.preview_callback:
            logger.debug(f"No preview callback available")

    def apply_pending_live_changes(self):
        """Apply pending live changes after debounce timer"""
        if self._is_destroying:
            return
        if self.pending_temp_warp_map:
            # Check for GLSL errors after applying changes
            self.apply_live_warp_map_changes(self.pending_temp_warp_map)
            
            # Display any compilation errors
            self.check_compilation_errors()
            
            self.pending_temp_warp_map = None

    def clear_warp_map_selection(self):
        """Clear the current warp map selection from the visualizer"""
        if self.preview_callback:
            try:
                # Call the preview callback with None to clear the selection
                self.preview_callback(None, persistent=True)
                logger.debug("Cleared warp map selection")

                # Update status
                self.preview_status_label.setText("🔓 Warp map cleared - automatic changes resumed")
                self.preview_status_label.setStyleSheet("color: #ff8800;")  # Orange

                # Reset status after 5 seconds (only if not destroying)
                if not self._is_destroying:
                    QTimer.singleShot(
                        5000,
                        lambda: (
                            self.preview_status_label.setText("Ready")
                            if not self._is_destroying
                            else None
                        ),
                    )

            except Exception as e:
                logger.errror(f"Error clearing warp map selection: {e}")

    def launch_preview_window(self):
        """Launch the standalone Pygame preview window with current code"""
        try:
            import subprocess
            import sys
            import os
            import tempfile

            # Get the path to the pygame preview script
            script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pygame_warp_preview.py")

            if not os.path.exists(script_path):
                QMessageBox.warning(self, "Preview Error",
                                  f"Preview script not found at: {script_path}")
                return

            # Create a temporary file with the current GLSL code
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, "karmaviz_warp_preview.glsl")

            # Write current code to temp file
            current_code = self.code_editor.toPlainText()
            with open(temp_file, 'w') as f:
                f.write(current_code)

            # Store temp file path for updates
            self.preview_temp_file = temp_file

            # Launch the preview window with file watching
            cmd = [
                sys.executable, script_path,
                "--warp-file", temp_file,
                "--watch-file", temp_file
            ]

            subprocess.Popen(cmd, cwd=os.path.dirname(os.path.dirname(__file__)))

            # Show success message
            self.preview_status_label.setText("Preview window launched with live updates!")
            self.preview_status_label.setStyleSheet("color: #00ff00;")

            # Reset status after 3 seconds (only if not destroying)
            if not self._is_destroying:
                QTimer.singleShot(
                    3000,
                    lambda: (
                        self.preview_status_label.setText("Ready")
                        if not self._is_destroying
                        else None
                    ),
                )

            logger.debug(f"Preview launched with temp file: {temp_file}")

        except Exception as e:
            QMessageBox.critical(self, "Preview Error",
                               f"Failed to launch preview window:\n{str(e)}")
            logger.errror(f"Error launching preview: {e}")
            
            

    def update_preview_temp_file(self):
        """Update the temporary file for live preview"""
        if hasattr(self, 'preview_temp_file') and self.preview_temp_file:
            try:
                current_code = self.code_editor.toPlainText()
                with open(self.preview_temp_file, 'w') as f:
                    f.write(current_code)
                # Don't spam the console, just update silently
            except Exception as e:
                logger.errror(f"Error updating preview temp file: {e}")

"""
Waveform Editor for KarmaViz

This module provides a GUI for creating, editing, and managing waveforms.
Similar to the warp map editor but for GLSL waveform functions using binary format.
"""

import sys
import os
import json
import time
import math
import numpy as np
from typing import Optional, List, Dict
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QLabel,
    QLineEdit,
    QComboBox,
    QPushButton,
    QListWidget,
    QSplitter,
    QGroupBox,
    QFormLayout,
    QMessageBox,
    QFileDialog,
    QTabWidget,
    QListWidgetItem,
    QDialog,
    QDialogButtonBox,
    QSpinBox,
    QSlider,
    QCheckBox,
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QPointF
from PyQt5.QtGui import (
    QFont,
    QTextDocument,
    QPainter,
    QPen,
    QBrush,
    QPolygonF,
    QColor,
)

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

from modules.waveform_manager import WaveformManager, WaveformInfo
from modules.glsl_syntax_highlighter import GLSLSyntaxHighlighter
from modules.line_numbered_editor import LineNumberedCodeEditor


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

    /* Text editors */
    QTextEdit {
        background-color: #1e1e1e;
        border: 1px solid #555555;
        color: #ffffff;
        selection-background-color: #0078d4;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 10pt;
        line-height: 1.2;
    }

    QTextEdit#code_editor {
        background-color: #0d1117;
        border: 2px solid #30363d;
        color: #c9d1d9;
        selection-background-color: #264f78;
        padding: 8px;
    }

    /* Line edits */
    QLineEdit {
        background-color: #404040;
        border: 1px solid #555555;
        color: #ffffff;
        padding: 4px 8px;
        border-radius: 3px;
    }

    QLineEdit:focus {
        border-color: #0078d4;
        background-color: #4a4a4a;
    }

    /* Buttons */
    QPushButton {
        background-color: #0078d4;
        border: none;
        color: #ffffff;
        padding: 6px 16px;
        border-radius: 3px;
        font-weight: bold;
    }

    QPushButton:hover {
        background-color: #106ebe;
    }

    QPushButton:pressed {
        background-color: #005a9e;
    }

    QPushButton:disabled {
        background-color: #555555;
        color: #888888;
    }

    /* Combo boxes */
    QComboBox {
        background-color: #404040;
        border: 1px solid #555555;
        color: #ffffff;
        padding: 4px 8px;
        border-radius: 3px;
        min-width: 100px;
    }

    QComboBox:hover {
        border-color: #0078d4;
    }

    QComboBox::drop-down {
        border: none;
        width: 20px;
    }

    QComboBox::down-arrow {
        image: none;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 5px solid #ffffff;
        margin-right: 5px;
    }

    QComboBox QAbstractItemView {
        background-color: #404040;
        border: 1px solid #555555;
        color: #ffffff;
        selection-background-color: #0078d4;
    }

    /* List widgets */
    QListWidget {
        background-color: #333333;
        border: 1px solid #555555;
        color: #ffffff;
        alternate-background-color: #3a3a3a;
    }

    QListWidget::item {
        padding: 4px 8px;
        border-bottom: 1px solid #444444;
    }

    QListWidget::item:selected {
        background-color: #0078d4;
        color: #ffffff;
    }

    QListWidget::item:hover {
        background-color: #4a4a4a;
    }

    /* Group boxes */
    QGroupBox {
        color: #ffffff;
        border: 1px solid #555555;
        border-radius: 5px;
        margin-top: 10px;
        font-weight: bold;
        padding-top: 10px;
    }

    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 8px 0 8px;
        color: #0078d4;
    }

    /* Spin boxes */
    QSpinBox {
        background-color: #404040;
        border: 1px solid #555555;
        color: #ffffff;
        padding: 4px;
        border-radius: 3px;
    }

    QSpinBox:focus {
        border-color: #0078d4;
    }

    QSpinBox::up-button, QSpinBox::down-button {
        background-color: #555555;
        border: none;
        width: 16px;
    }

    QSpinBox::up-button:hover, QSpinBox::down-button:hover {
        background-color: #666666;
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

    /* Message boxes */
    QMessageBox {
        background-color: #2b2b2b;
        color: #ffffff;
    }

    QMessageBox QPushButton {
        min-width: 80px;
        padding: 6px 12px;
    }

    /* File dialogs */
    QFileDialog {
        background-color: #2b2b2b;
        color: #ffffff;
    }

    /* Dialogs */
    QDialog {
        background-color: #2b2b2b;
        color: #ffffff;
    }

    /* Form layouts */
    QFormLayout QLabel {
        color: #cccccc;
        font-weight: normal;
    }
    
    /* Error status labels */
    QLabel[objectName="error_status_label"] {
        padding: 6px;
        border-radius: 3px;
        font-weight: bold;
        background-color: #333333;
        border: 1px solid #555555;
    }
    """
    app.setStyleSheet(dark_stylesheet)



class WaveformEditor(QWidget):
    """Waveform editor widget"""

    waveform_changed = pyqtSignal(str)  # Emitted when a waveform is modified

    def __init__(self, waveform_manager: WaveformManager, show_preview: bool = True, shader_compiler=None):
        super().__init__()
        self.waveform_manager = waveform_manager
        self.current_waveform: Optional[WaveformInfo] = None
        self.current_waveform_key: Optional[str] = None  # Store the waveform name
        self.original_waveform_backup: Optional[WaveformInfo] = None  # Backup of original for revert
        self.unsaved_changes = False
        self.preview_callback = None  # Callback for live preview
        self.show_preview = show_preview  # Control whether to show preview section
        self.shader_compiler = shader_compiler  # For syntax validation

        # Timer for debouncing live updates
        self.live_update_timer = QTimer()
        self.live_update_timer.setSingleShot(True)
        self.live_update_timer.timeout.connect(self.apply_pending_live_changes)
        self.pending_temp_waveform = None

        # Flag to track if widget is being destroyed
        self._is_destroying = False

        # Flag to prevent change detection during loading
        self._loading_waveform = False

        self.init_ui()
        self.load_waveform_list()

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
        self.code_editor.setPlainText(self.get_template_code())
        self.update_preview()

    def cleanup(self):
        """Clean up resources, especially timers and preview process"""
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
            print(f"Warning: Error cleaning up WaveformEditor timer: {e}")

        # Clean up preview process
        try:
            if hasattr(self, "preview_process") and self.preview_process:
                if self.preview_process.poll() is None:  # Process is still running
                    self.preview_process.terminate()
                    self.preview_process.wait()
        except Exception as e:
            print(f"Warning: Error cleaning up preview process: {e}")

        # Clean up temp file
        try:
            if hasattr(self, "temp_waveform_file") and os.path.exists(
                self.temp_waveform_file
            ):
                os.unlink(self.temp_waveform_file)
        except Exception as e:
            print(f"Warning: Error cleaning up temp file: {e}")

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
        self.setWindowTitle("Waveform Editor")
        self.setGeometry(100, 100, 1200, 800)

        # Main layout
        main_layout = QHBoxLayout(self)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel - Waveform list and controls
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)

        # Right panel - Editor and preview
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)

        # Set splitter proportions
        splitter.setSizes([300, 900])

    def create_left_panel(self) -> QWidget:
        """Create the left panel with waveform list and controls"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Waveform list
        list_group = QGroupBox("Waveforms")
        list_layout = QVBoxLayout(list_group)

        self.waveform_list = QListWidget()
        self.waveform_list.itemClicked.connect(self.on_waveform_selected)
        list_layout.addWidget(self.waveform_list)

        # Search filter
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search:"))
        self.search_filter = QLineEdit()
        self.search_filter.setPlaceholderText("Filter by name, category, or description...")
        self.search_filter.textChanged.connect(self.apply_filters)
        search_layout.addWidget(self.search_filter)
        list_layout.addLayout(search_layout)

        # Category filter
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Category:"))
        self.category_filter = QComboBox()
        self.category_filter.addItem("All Categories")
        self.category_filter.currentTextChanged.connect(self.apply_filters)
        filter_layout.addWidget(self.category_filter)
        list_layout.addLayout(filter_layout)

        layout.addWidget(list_group)

        # Waveform management buttons
        button_group = QGroupBox("Waveform Management")
        button_layout = QVBoxLayout(button_group)

        self.new_button = QPushButton("New Waveform")
        self.new_button.clicked.connect(self.new_waveform)
        button_layout.addWidget(self.new_button)

        self.duplicate_button = QPushButton("Duplicate")
        self.duplicate_button.clicked.connect(self.duplicate_waveform)
        self.duplicate_button.setEnabled(False)
        button_layout.addWidget(self.duplicate_button)

        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self.delete_waveform)
        self.delete_button.setEnabled(False)
        button_layout.addWidget(self.delete_button)

        self.export_button = QPushButton("Export...")
        self.export_button.clicked.connect(self.export_waveform)
        self.export_button.setEnabled(False)
        button_layout.addWidget(self.export_button)

        self.import_button = QPushButton("Import...")
        self.import_button.clicked.connect(self.import_waveform)
        button_layout.addWidget(self.import_button)

        layout.addWidget(button_group)

        layout.addStretch()
        return panel

    def create_right_panel(self) -> QWidget:
        """Create the right panel with editor and preview"""
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
        self.save_button.clicked.connect(self.save_current_waveform)
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
            self.auto_update_checkbox.setToolTip(
                "Automatically apply waveform changes to the visualizer as you type"
            )
            button_layout.addWidget(self.auto_update_checkbox)

        # Only add preview button if preview is enabled
        if self.show_preview:
            self.preview_button = QPushButton("Preview")
            self.preview_button.clicked.connect(self.apply_current_waveform_preview)
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
        """Create the metadata/properties tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Basic properties
        basic_group = QGroupBox("Basic Properties")
        basic_form = QFormLayout(basic_group)

        self.name_edit = QLineEdit()
        self.name_edit.textChanged.connect(self.on_metadata_changed)
        basic_form.addRow("Name:", self.name_edit)

        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(80)
        self.description_edit.textChanged.connect(self.on_metadata_changed)
        basic_form.addRow("Description:", self.description_edit)

        self.author_edit = QLineEdit()
        self.author_edit.textChanged.connect(self.on_metadata_changed)
        basic_form.addRow("Author:", self.author_edit)

        self.category_edit = QComboBox()
        self.category_edit.setEditable(True)
        self.category_edit.addItems(
            ["basic", "experimental", "fractal", "mathematical", "motion", "organic"]
        )
        self.category_edit.currentTextChanged.connect(self.on_metadata_changed)
        basic_form.addRow("Category:", self.category_edit)

        layout.addWidget(basic_group)

        # Advanced properties
        advanced_group = QGroupBox("Advanced Properties")
        advanced_form = QFormLayout(advanced_group)

        self.complexity_combo = QComboBox()
        self.complexity_combo.addItems(["low", "medium", "high"])
        self.complexity_combo.currentTextChanged.connect(self.on_metadata_changed)
        advanced_form.addRow("Complexity:", self.complexity_combo)

        self.builtin_checkbox = QCheckBox()
        self.builtin_checkbox.stateChanged.connect(self.on_metadata_changed)
        advanced_form.addRow("Built-in:", self.builtin_checkbox)

        layout.addWidget(advanced_group)

        layout.addStretch()
        return tab

    def create_preview_panel(self) -> QWidget:
        """Create the preview panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)

        # Preview area placeholder
        self.preview_label = QLabel("Preview will be shown here")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(200)
        self.preview_label.setStyleSheet(
            "border: 1px solid #555555; background-color: #1e1e1e;"
        )
        preview_layout.addWidget(self.preview_label)

        # Preview controls
        controls_layout = QHBoxLayout()

        self.launch_preview_button = QPushButton("🚀 Launch Preview")
        self.launch_preview_button.clicked.connect(self.launch_preview_window)
        self.launch_preview_button.setToolTip("Launch standalone preview window")
        controls_layout.addWidget(self.launch_preview_button)

        self.auto_preview_checkbox = QCheckBox("Auto-preview")
        self.auto_preview_checkbox.setChecked(True)
        self.auto_preview_checkbox.setToolTip(
            "Automatically update preview as you type"
        )
        controls_layout.addWidget(self.auto_preview_checkbox)

        controls_layout.addStretch()
        preview_layout.addLayout(controls_layout)

        # Preview status
        self.preview_status_label = QLabel(
            "Click 'Launch Preview' to open preview window"
        )
        self.preview_status_label.setStyleSheet("color: #888888; font-style: italic;")
        preview_layout.addWidget(self.preview_status_label)

        layout.addWidget(preview_group)
        return panel

    def load_waveform_list(self):
        """Load the list of available waveforms"""
        self.waveform_list.clear()

        # Get waveforms organized by category
        categories = self.waveform_manager.list_waveforms_by_category()

        # Update category filter
        self.category_filter.clear()
        self.category_filter.addItem("All Categories")
        for category in sorted(categories.keys()):
            if category != "root":
                self.category_filter.addItem(category)

        # Add all waveforms to the list
        all_waveforms = []
        for category, waveforms in categories.items():
            for waveform in waveforms:
                all_waveforms.append((waveform, category))

        # Sort by name
        all_waveforms.sort(key=lambda x: x[0])

        for waveform_name, category in all_waveforms:
            # Get waveform info for better filtering
            waveform_info = self.waveform_manager.get_waveform(waveform_name)
            description = waveform_info.description if waveform_info else ""
            
            item = QListWidgetItem(f"{waveform_name} ({category})")
            item.setData(Qt.UserRole, waveform_name)
            # Store additional data for filtering
            item.setData(Qt.UserRole + 1, category)
            item.setData(Qt.UserRole + 2, description)
            self.waveform_list.addItem(item)

    def apply_filters(self):
        """Apply both search and category filters to the waveform list"""
        search_text = self.search_filter.text().lower()
        category_filter = self.category_filter.currentText()
        
        for i in range(self.waveform_list.count()):
            item = self.waveform_list.item(i)
            if not item:
                continue
                
            # Get stored data
            waveform_name = item.data(Qt.UserRole)
            category = item.data(Qt.UserRole + 1)
            description = item.data(Qt.UserRole + 2) or ""
            
            # Apply search filter
            search_match = True
            if search_text:
                search_match = (
                    search_text in waveform_name.lower() or
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

    def filter_waveforms(self, category_filter: str):
        """Legacy method for backward compatibility - now calls apply_filters"""
        self.apply_filters()

    def on_waveform_selected(self, item: QListWidgetItem):
        """Handle waveform selection"""
        # Store the waveform name before showing any dialogs
        # to avoid issues with Qt object deletion
        waveform_name = item.data(Qt.UserRole)

        if self.unsaved_changes:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save them before switching waveforms?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Yes:
                if not self.save_current_waveform():
                    return  # Save failed, don't switch
            elif reply == QMessageBox.Cancel:
                return  # Don't switch

        self.load_waveform(waveform_name)

    def load_waveform(self, waveform_name: str):
        """Load a waveform for editing"""
        waveform_info = self.waveform_manager.get_waveform(waveform_name)
        if not waveform_info:
            QMessageBox.warning(self, "Error", f"Waveform '{waveform_name}' not found.")
            return

        self.current_waveform = waveform_info
        self.current_waveform_key = waveform_name
        
        # Create a deep copy backup of the original waveform for revert functionality
        import copy
        self.original_waveform_backup = copy.deepcopy(waveform_info)
        
        self.unsaved_changes = False

        # Set loading flag to prevent change detection
        self._loading_waveform = True

        # Update UI
        self.code_editor.setPlainText(waveform_info.glsl_code)
        
        # Set line number offset based on injection line in main shader
        if self.shader_compiler:
            injection_line = self.shader_compiler.get_injection_line_for_waveform(waveform_name)
            self.code_editor.set_line_number_offset(injection_line - 1)
        
        self.name_edit.setText(waveform_info.name)
        self.description_edit.setPlainText(waveform_info.description)
        self.author_edit.setText(waveform_info.author)
        self.category_edit.setCurrentText(waveform_info.category)
        self.complexity_combo.setCurrentText(waveform_info.complexity)

        self.builtin_checkbox.setChecked(waveform_info.is_builtin)

        # Clear loading flag
        self._loading_waveform = False

        # Update button states
        self.duplicate_button.setEnabled(True)
        self.delete_button.setEnabled(not waveform_info.is_builtin)
        self.export_button.setEnabled(True)
        self.save_button.setEnabled(False)
        self.revert_button.setEnabled(False)
        if self.show_preview:
            self.preview_button.setEnabled(True)

        # Check for compilation errors after loading
        self.check_compilation_errors()

        # Update preview
        if self.show_preview and self.auto_preview_checkbox.isChecked():
            self.update_preview()

        # Apply waveform for live preview (always apply if preview callback exists)
        if self.preview_callback:
            self.apply_waveform_preview(waveform_info)

    def on_code_changed(self):
        """Handle code editor changes"""
        if not self.current_waveform or self._loading_waveform:
            return

        self.unsaved_changes = True
        self.save_button.setEnabled(True)
        self.revert_button.setEnabled(True)

        # Auto-preview if enabled
        if self.show_preview and self.auto_preview_checkbox.isChecked():
            self.update_preview()

        # Auto-apply if in integrated mode or if preview callback exists (live editing)
        if (
            not self.show_preview
            and hasattr(self, "auto_update_checkbox")
            and self.auto_update_checkbox.isChecked()
        ) or (self.preview_callback and self.current_waveform):
            self.schedule_live_update()
        else:
            # Check for compilation errors even when not in live update mode
            self.check_compilation_errors()

    def on_metadata_changed(self):
        """Handle metadata changes"""
        if not self.current_waveform or self._loading_waveform:
            return

        self.unsaved_changes = True
        self.save_button.setEnabled(True)
        self.revert_button.setEnabled(True)

    def schedule_live_update(self):
        """Schedule a live update with debouncing"""
        if not self.current_waveform or self._is_destroying:
            return

        # Create temporary waveform with current editor content
        temp_waveform = WaveformInfo(
            name=self.current_waveform.name,
            description=self.current_waveform.description,
            author=self.current_waveform.author,
            category=self.current_waveform.category,
            complexity=self.current_waveform.complexity,
            is_builtin=self.current_waveform.is_builtin,
            glsl_code=self.code_editor.toPlainText(),
        )

        self.pending_temp_waveform = temp_waveform

        # Restart the timer (debouncing) - check if timer is still valid
        try:
            if hasattr(self, "live_update_timer") and self.live_update_timer is not None:
                self.live_update_timer.stop()
                self.live_update_timer.start(500)  # 500ms delay
        except RuntimeError:
            # Timer was already deleted, ignore
            pass

    def apply_pending_live_changes(self):
        """Apply pending live changes"""
        if self.pending_temp_waveform and self.preview_callback:
            try:
                self.preview_callback(self.pending_temp_waveform, persistent=True)
                
                # Check for compilation errors after applying changes
                self.check_compilation_errors()
                        
            except Exception as e:
                print(f"Error applying live waveform changes: {e}")
                self.error_status_label.setText(f"Error: {str(e)}")
                self.error_status_label.setStyleSheet("color: #ff4444;")  # Red

    def apply_waveform_preview(self, waveform_info):
        """Apply waveform for preview with persistent flag"""
        if self.preview_callback:
            try:
                self.preview_callback(waveform_info, persistent=True)
                print(f"🔒 Applied '{waveform_info.name}' - automatic changes paused")
            except Exception as e:
                print(f"Error applying waveform preview: {e}")
                

                

    def clear_waveform_selection(self):
        """Clear the current waveform selection from the visualizer"""
        if self.preview_callback:
            try:
                # Call the preview callback with None to clear the selection
                self.preview_callback(None, persistent=True)
                print("Cleared waveform selection")
            except Exception as e:
                print(f"Error clearing waveform selection: {e}")

    def update_preview(self):
        """Update the preview"""
        if not self.show_preview:
            return

        # For now, just show a placeholder
        # In a full implementation, this would render the waveform
        self.preview_label.setText(
            f"Preview of: {self.current_waveform.name if self.current_waveform else 'New Waveform'}"
        )

    def apply_current_waveform_preview(self):
        """Apply current waveform for preview"""
        if not self.current_waveform:
            return

        # Create temporary waveform with current editor content
        temp_waveform = WaveformInfo(
            name=self.current_waveform.name,
            description=self.description_edit.toPlainText(),
            author=self.author_edit.text(),
            category=self.category_edit.currentText(),
            complexity=self.complexity_combo.currentText(),
            is_builtin=self.builtin_checkbox.isChecked(),
            glsl_code=self.code_editor.toPlainText(),
        )

        if self.preview_callback:
            try:
                self.preview_callback(temp_waveform)
            except Exception as e:
                QMessageBox.warning(
                    self, "Preview Error", f"Error applying waveform preview: {e}"
                )

    def new_waveform(self):
        """Create a new waveform"""
        if self.unsaved_changes:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save them before creating a new waveform?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Yes:
                if not self.save_current_waveform():
                    return  # Save failed, don't create new
            elif reply == QMessageBox.Cancel:
                return  # Don't create new

        # Create new waveform with template
        new_waveform = WaveformInfo(
            name="new_waveform",
            description="A new custom waveform",
            author="User",
            category="basic",
            complexity="low",
            is_builtin=False,
            glsl_code=self.get_template_code(),
        )

        self.current_waveform = new_waveform
        self.current_waveform_key = None  # New waveform, not saved yet
        self.original_waveform_backup = None  # Clear backup for new waveforms

        # Set loading flag to prevent change detection during UI updates
        self._loading_waveform = True

        # Update UI
        self.code_editor.setPlainText(new_waveform.glsl_code)
        self.name_edit.setText(new_waveform.name)
        self.description_edit.setPlainText(new_waveform.description)
        self.author_edit.setText(new_waveform.author)
        self.category_edit.setCurrentText(new_waveform.category)
        self.complexity_combo.setCurrentText(new_waveform.complexity)

        self.builtin_checkbox.setChecked(new_waveform.is_builtin)

        # Clear loading flag and set unsaved changes
        self._loading_waveform = False
        self.unsaved_changes = True

        # Update button states
        self.duplicate_button.setEnabled(False)
        self.delete_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.save_button.setEnabled(True)
        self.revert_button.setEnabled(False)
        if self.show_preview:
            self.preview_button.setEnabled(True)

        # Clear selection in list
        self.waveform_list.clearSelection()

    def duplicate_waveform(self):
        """Duplicate the current waveform"""
        if not self.current_waveform:
            return

        # Create duplicate with modified name
        duplicate_name = f"{self.current_waveform.name}_copy"
        counter = 1
        while self.waveform_manager.get_waveform(duplicate_name):
            duplicate_name = f"{self.current_waveform.name}_copy_{counter}"
            counter += 1

        duplicate_waveform = WaveformInfo(
            name=duplicate_name,
            description=f"Copy of {self.current_waveform.description}",
            author=self.current_waveform.author,
            category=self.current_waveform.category,
            complexity=self.current_waveform.complexity,
            is_builtin=False,  # Duplicates are never built-in
            glsl_code=self.current_waveform.glsl_code,
        )

        # Save the duplicate
        if self.waveform_manager.save_waveform(
            duplicate_waveform, subdirectory=duplicate_waveform.category
        ):
            # Load the duplicated waveform into the editor
            self.current_waveform = duplicate_waveform
            self.current_waveform_key = duplicate_name
            self.original_waveform_backup = None  # Clear backup for duplicated waveforms
            
            # Set loading flag to prevent change detection
            self._loading_waveform = True
            
            # Update UI with duplicated waveform
            self.code_editor.setPlainText(duplicate_waveform.glsl_code)
            self.name_edit.setText(duplicate_waveform.name)
            self.description_edit.setPlainText(duplicate_waveform.description)
            self.author_edit.setText(duplicate_waveform.author)
            self.category_edit.setCurrentText(duplicate_waveform.category)
            self.complexity_combo.setCurrentText(duplicate_waveform.complexity)
            self.builtin_checkbox.setChecked(duplicate_waveform.is_builtin)
            
            # Clear loading flag and set unsaved changes
            self._loading_waveform = False
            self.unsaved_changes = True
            self.save_button.setEnabled(True)
            self.revert_button.setEnabled(False)  # No backup yet for duplicated waveform
            
            self.load_waveform_list()
            QMessageBox.information(
                self, "Success", f"Waveform duplicated as '{duplicate_name}'"
            )
        else:
            QMessageBox.warning(self, "Error", "Failed to duplicate waveform")

    def delete_waveform(self):
        """Delete the current waveform"""
        if not self.current_waveform or not self.current_waveform_key:
            return

        if self.current_waveform.is_builtin:
            QMessageBox.warning(
                self, "Cannot Delete", "Built-in waveforms cannot be deleted."
            )
            return

        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete the waveform '{self.current_waveform.name}'?",
            QMessageBox.Yes | QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            if self.waveform_manager.delete_waveform(self.current_waveform_key):
                self.load_waveform_list()
                self.current_waveform = None
                self.current_waveform_key = None
                self.original_waveform_backup = None  # Clear backup
                self.unsaved_changes = False
                self.code_editor.clear()
                self.name_edit.clear()
                self.description_edit.clear()
                self.author_edit.clear()
                self.category_edit.setCurrentText("basic")
                self.complexity_combo.setCurrentText("low")

                self.builtin_checkbox.setChecked(False)

                # Update button states
                self.duplicate_button.setEnabled(False)
                self.delete_button.setEnabled(False)
                self.export_button.setEnabled(False)
                self.save_button.setEnabled(False)
                self.revert_button.setEnabled(False)
                if self.show_preview:
                    self.preview_button.setEnabled(False)

                QMessageBox.information(
                    self, "Success", "Waveform deleted successfully"
                )
            else:
                QMessageBox.warning(self, "Error", "Failed to delete waveform")

    def save_current_waveform(self) -> bool:
        """Save the current waveform"""
        if not self.current_waveform:
            return False

        # Update waveform with current editor content
        self.current_waveform.name = self.name_edit.text().strip()
        self.current_waveform.description = self.description_edit.toPlainText().strip()
        self.current_waveform.author = self.author_edit.text().strip()
        self.current_waveform.category = self.category_edit.currentText().strip()
        self.current_waveform.complexity = self.complexity_combo.currentText()

        self.current_waveform.is_builtin = self.builtin_checkbox.isChecked()
        self.current_waveform.glsl_code = self.code_editor.toPlainText()

        # Validate name
        if not self.current_waveform.name:
            QMessageBox.warning(self, "Invalid Name", "Waveform name cannot be empty.")
            return False
            
        # Check for any recent GLSL compilation errors
        if hasattr(self, 'shader_compiler') and self.shader_compiler and self.current_waveform_key:
            errors = self.shader_compiler.get_latest_errors_for_waveform(self.current_waveform_key)
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

        # Check if name changed and if new name already exists
        if (
            self.current_waveform_key != self.current_waveform.name
            and self.waveform_manager.get_waveform(self.current_waveform.name)
        ):
            QMessageBox.warning(
                self,
                "Name Conflict",
                f"A waveform named '{self.current_waveform.name}' already exists.",
            )
            return False

        # Save waveform
        overwrite = self.current_waveform_key is not None
        if self.waveform_manager.save_waveform(
            self.current_waveform,
            overwrite=overwrite,
            subdirectory=self.current_waveform.category,
        ):
            # If name changed, delete old waveform
            if (
                self.current_waveform_key
                and self.current_waveform_key != self.current_waveform.name
            ):
                self.waveform_manager.delete_waveform(self.current_waveform_key)

            self.current_waveform_key = self.current_waveform.name
            self.unsaved_changes = False
            self.save_button.setEnabled(False)
            self.revert_button.setEnabled(False)
            self.load_waveform_list()

            # Emit signal that waveform changed
            self.waveform_changed.emit(self.current_waveform.name)

            QMessageBox.information(self, "Success", "Waveform saved successfully")
            return True
        else:
            QMessageBox.warning(self, "Error", "Failed to save waveform")
            return False

    def revert_changes(self):
        """Revert changes to the current waveform"""
        if self.original_waveform_backup and self.current_waveform_key:
            reply = QMessageBox.question(
                self,
                "Confirm Revert",
                "Are you sure you want to revert all changes?",
                QMessageBox.Yes | QMessageBox.No,
            )

            if reply == QMessageBox.Yes:
                # Restore from the backup copy instead of reloading from disk
                import copy
                self.current_waveform = copy.deepcopy(self.original_waveform_backup)
                
                # Set loading flag to prevent change detection
                self._loading_waveform = True
                
                # Load the backup data into the UI
                self.code_editor.setPlainText(self.original_waveform_backup.glsl_code)
                self.name_edit.setText(self.original_waveform_backup.name)
                self.description_edit.setPlainText(self.original_waveform_backup.description)
                self.author_edit.setText(self.original_waveform_backup.author)
                self.category_edit.setCurrentText(self.original_waveform_backup.category)
                self.complexity_combo.setCurrentText(self.original_waveform_backup.complexity)
                self.builtin_checkbox.setChecked(self.original_waveform_backup.is_builtin)
                
                # Clear loading flag
                self._loading_waveform = False
                
                # Clear unsaved changes flag
                self.unsaved_changes = False
                self.save_button.setEnabled(False)
                self.revert_button.setEnabled(False)
                
                # Update preview with reverted waveform
                if self.show_preview and self.auto_preview_checkbox.isChecked():
                    self.update_preview()
                
                # Apply reverted waveform for live preview if callback exists
                if self.preview_callback:
                    self.apply_waveform_preview(self.current_waveform)
                    
                print(f"Reverted waveform '{self.current_waveform.name}' to original backup")
                
        elif not self.original_waveform_backup:
            print(f"No backup available for current waveform")
        elif not self.current_waveform_key:
            print(f"No filename key available for current waveform")

    def export_waveform(self):
        """Export the current waveform"""
        if not self.current_waveform:
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Waveform",
            f"{self.current_waveform.name}.kvwf",
            "KarmaViz Waveform Files (*.kvwf)",
        )

        if filename:
            try:
                with open(filename, "wb") as f:
                    f.write(self.current_waveform.to_binary())
                QMessageBox.information(
                    self, "Success", f"Waveform exported to {filename}"
                )
            except Exception as e:
                QMessageBox.warning(
                    self, "Export Error", f"Failed to export waveform: {e}"
                )

    def import_waveform(self):
        """Import a waveform"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Import Waveform", "", "KarmaViz Waveform Files (*.kvwf)"
        )

        if filename:
            try:
                with open(filename, "rb") as f:
                    data = f.read()
                    waveform_info = WaveformInfo.from_binary(data)

                # Check if waveform already exists
                if self.waveform_manager.get_waveform(waveform_info.name):
                    reply = QMessageBox.question(
                        self,
                        "Waveform Exists",
                        f"A waveform named '{waveform_info.name}' already exists. Overwrite?",
                        QMessageBox.Yes | QMessageBox.No,
                    )
                    if reply != QMessageBox.Yes:
                        return

                # Save imported waveform
                if self.waveform_manager.save_waveform(
                    waveform_info, overwrite=True, subdirectory=waveform_info.category
                ):
                    # Load the imported waveform into the editor
                    self.current_waveform = waveform_info
                    self.current_waveform_key = waveform_info.name
                    self.original_waveform_backup = None  # Clear backup for imported waveforms
                    
                    # Set loading flag to prevent change detection
                    self._loading_waveform = True
                    
                    # Update UI with imported waveform
                    self.code_editor.setPlainText(waveform_info.glsl_code)
                    self.name_edit.setText(waveform_info.name)
                    self.description_edit.setPlainText(waveform_info.description)
                    self.author_edit.setText(waveform_info.author)
                    self.category_edit.setCurrentText(waveform_info.category)
                    self.complexity_combo.setCurrentText(waveform_info.complexity)
                    self.builtin_checkbox.setChecked(waveform_info.is_builtin)
                    
                    # Clear loading flag and set unsaved changes
                    self._loading_waveform = False
                    self.unsaved_changes = True
                    self.save_button.setEnabled(True)
                    self.revert_button.setEnabled(False)  # No backup yet for imported waveform
                    
                    self.load_waveform_list()
                    QMessageBox.information(
                        self,
                        "Success",
                        f"Waveform '{waveform_info.name}' imported successfully",
                    )
                else:
                    QMessageBox.warning(
                        self, "Import Error", "Failed to save imported waveform"
                    )

            except Exception as e:
                QMessageBox.warning(
                    self, "Import Error", f"Failed to import waveform: {e}"
                )

    def test_compilation(self):
        """Test compile the current waveform to generate error information"""
        if not self.current_waveform or not self.shader_compiler:
            self.error_status_label.setText("No waveform loaded or shader compiler not available")
            self.error_status_label.setStyleSheet("color: #ff8800;")  # Orange
            return
        
        # Create a temporary waveform with current editor content
        temp_waveform = WaveformInfo(
            name=self.current_waveform.name,
            description=self.current_waveform.description,
            author=self.current_waveform.author,
            category=self.current_waveform.category,
            complexity=self.current_waveform.complexity,
            is_builtin=self.current_waveform.is_builtin,
            glsl_code=self.code_editor.toPlainText(),
        )
        
        # Force a test compilation
        if self.preview_callback:
            try:
                self.error_status_label.setText("Testing compilation...")
                self.error_status_label.setStyleSheet("color: #ffff00;")  # Yellow
                self.preview_callback(temp_waveform, persistent=False)  # Non-persistent test
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
        if hasattr(self, 'shader_compiler') and self.shader_compiler and self.current_waveform_key:
            errors = self.shader_compiler.get_latest_errors_for_waveform(self.current_waveform_key)
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

    def insert_template(self):
        """Insert template code"""
        template = self.get_template_code()
        self.code_editor.setPlainText(template)

    def get_template_code(self) -> str:
        """Get template GLSL code for waveforms"""
        return """// Waveform rendering function
// pos: normalized screen coordinates (-1 to 1)
// t: time in seconds
// audio_data: array of 512 audio samples (0.0 to 1.0)

vec3 render_waveform(vec2 pos, float t, float audio_data[512]) {
    // Convert position to sample index
    int sample_index = int((pos.x + 1.0) * 0.5 * 511.0);
    sample_index = clamp(sample_index, 0, 511);
    
    // Get audio sample value
    float sample_value = audio_data[sample_index];
    
    // Create waveform line
    float line_thickness = 0.02;
    float distance_to_line = abs(pos.y - (sample_value * 2.0 - 1.0));
    float line_intensity = 1.0 - smoothstep(0.0, line_thickness, distance_to_line);
    
    // Color the waveform
    vec3 color = vec3(0.0, 1.0, 0.5) * line_intensity;
    
    return color;
}"""

    def show_glsl_help(self):
        """Show GLSL help dialog"""
        help_text = """
GLSL Waveform Function Reference:

Required Function:
vec3 render_waveform(vec2 pos, float t, float audio_data[512])

Parameters:
- pos: Screen coordinates (-1 to 1, -1 to 1)
- t: Time in seconds since start
- audio_data: Array of 512 audio samples (0.0 to 1.0)

Common GLSL Functions:
- sin(x), cos(x), tan(x): Trigonometric functions
- abs(x): Absolute value
- clamp(x, min, max): Constrain value to range
- smoothstep(edge0, edge1, x): Smooth interpolation
- mix(a, b, t): Linear interpolation
- length(v): Vector length
- normalize(v): Normalize vector
- dot(a, b): Dot product

Built-in Variables:
- gl_FragCoord: Fragment coordinates
- gl_FragColor: Output color (deprecated, use return value)

Tips:
- Use pos.x to map to audio sample index
- Use pos.y to create vertical effects
- Use t for time-based animations
- Return vec3(r, g, b) where each component is 0.0-1.0
"""

        QMessageBox.information(self, "GLSL Help", help_text)

    def launch_preview_window(self):
        """Launch the standalone Pygame preview window with current code"""
        try:
            import os
            import subprocess
            import tempfile

            # Get the path to the pygame preview script
            script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pygame_waveform_preview.py")

            if not os.path.exists(script_path):
                QMessageBox.warning(
                    self, "Preview Error", f"Preview script not found at: {script_path}"
                )
                return

            # Create temporary file for live updates
            temp_dir = tempfile.gettempdir()
            self.temp_waveform_file = os.path.join(
                temp_dir, "karmaviz_temp_waveform.glsl"
            )

            # Write current waveform code to temp file
            self.update_temp_waveform_file()

            # Launch the preview window with file watching
            try:
                if (
                    hasattr(self, "preview_process")
                    and self.preview_process
                    and self.preview_process.poll() is None
                ):
                    # Process is still running, bring it to front or restart
                    self.preview_process.terminate()
                    self.preview_process.wait()

                self.preview_process = subprocess.Popen(
                    [sys.executable, script_path, self.temp_waveform_file],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                self.preview_status_label.setText(
                    "Preview window launched with live updates!"
                )
                self.launch_preview_button.setText("Restart Preview")

                # Enable auto-update for live preview
                if hasattr(self, "auto_preview_checkbox"):
                    self.auto_preview_checkbox.setChecked(True)

                print(
                    f"🚀 Launched waveform preview window (PID: {self.preview_process.pid})"
                )

            except Exception as e:
                QMessageBox.warning(
                    self, "Launch Error", f"Failed to launch preview process:\n{str(e)}"
                )

        except Exception as e:
            QMessageBox.warning(
                self, "Preview Error", f"Failed to launch preview window:\n{str(e)}"
            )

    def update_temp_waveform_file(self):
        """Update the temporary waveform file with current code"""
        if hasattr(self, "temp_waveform_file"):
            try:
                waveform_code = self.code_editor.toPlainText()
                with open(self.temp_waveform_file, "w") as f:
                    f.write(waveform_code)
            except Exception as e:
                print(f"Error updating temp waveform file: {e}")

    def update_preview(self):
        """Update the preview"""
        if not self.show_preview:
            return

        # Update temp file for live preview
        self.update_temp_waveform_file()

        # Update the placeholder text
        self.preview_label.setText(
            f"Preview of: {self.current_waveform.name if self.current_waveform else 'New Waveform'}"
        )

    def set_preview_callback(self, callback):
        """Set callback for live preview updates"""
        self.preview_callback = callback


class WaveformEditorWidget(WaveformEditor):
    """Wrapper class for compatibility with config menu"""

    waveform_saved = pyqtSignal(str)  # Signal expected by config menu

    def __init__(self, waveform_manager: WaveformManager, show_preview: bool = False, shader_compiler=None):
        super().__init__(waveform_manager, show_preview, shader_compiler)
        # Connect the waveform_changed signal to waveform_saved for compatibility
        self.waveform_changed.connect(self.waveform_saved.emit)


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    apply_dark_theme(app)

    # Create waveform manager
    waveform_manager = WaveformManager("waveforms")

    # Create and show editor
    editor = WaveformEditor(waveform_manager, show_preview=True)
    editor.show()

    sys.exit(app.exec_())

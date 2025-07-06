"""
Preset Manager Widget for KarmaViz

This widget provides a comprehensive interface for managing presets including:
- Quick preset slots (0-9) with visual indicators
- Full preset library with search and filtering
- Import/Export functionality
- Preset information display
- Save/Load operations
"""

import os
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime

from modules.logging_config import get_logger

# Get logger for this module
logger = get_logger('preset_manager_widget')

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QListWidget, QListWidgetItem, QPushButton,
    QLabel, QLineEdit, QTextEdit, QComboBox, QProgressBar,
    QFileDialog, QMessageBox, QInputDialog, QSplitter,
    QFrame, QScrollArea, QTabWidget
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QIcon

from modules.preset_manager import PresetManager, PresetInfo


class QuickPresetSlot(QWidget):
    """Widget representing a single quick preset slot"""
    
    clicked = pyqtSignal(int)  # Emits slot number when clicked
    
    def __init__(self, slot_number: int, preset_manager: PresetManager):
        super().__init__()
        self.slot_number = slot_number
        self.preset_manager = preset_manager
        self.setup_ui()
        self.update_status()
    
    def setup_ui(self):
        """Set up the UI for the quick preset slot"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Slot number label
        self.number_label = QLabel(str(self.slot_number))
        self.number_label.setAlignment(Qt.AlignCenter)
        self.number_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #333;
                background-color: #f0f0f0;
                border: 2px solid #ccc;
                border-radius: 8px;
                padding: 8px;
                min-width: 40px;
                min-height: 40px;
            }
        """)
        layout.addWidget(self.number_label)
        
        # Status indicator
        self.status_label = QLabel("Empty")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 10px; color: #666;")
        layout.addWidget(self.status_label)
        
        # Make the widget clickable
        self.setFixedSize(80, 80)
        self.setStyleSheet("""
            QuickPresetSlot {
                border: 1px solid #ddd;
                border-radius: 8px;
                background-color: #fafafa;
            }
            QuickPresetSlot:hover {
                background-color: #e8f4fd;
                border-color: #0078d4;
            }
        """)
    
    def update_status(self):
        """Update the visual status of the slot"""
        exists = self.preset_manager.quick_preset_exists(self.slot_number)
        
        if exists:
            self.status_label.setText("Saved")
            self.status_label.setStyleSheet("font-size: 10px; color: #0078d4; font-weight: bold;")
            self.number_label.setStyleSheet("""
                QLabel {
                    font-size: 18px;
                    font-weight: bold;
                    color: white;
                    background-color: #0078d4;
                    border: 2px solid #0078d4;
                    border-radius: 8px;
                    padding: 8px;
                    min-width: 40px;
                    min-height: 40px;
                }
            """)
        else:
            self.status_label.setText("Empty")
            self.status_label.setStyleSheet("font-size: 10px; color: #666;")
            self.number_label.setStyleSheet("""
                QLabel {
                    font-size: 18px;
                    font-weight: bold;
                    color: #333;
                    background-color: #f0f0f0;
                    border: 2px solid #ccc;
                    border-radius: 8px;
                    padding: 8px;
                    min-width: 40px;
                    min-height: 40px;
                }
            """)
    
    def mousePressEvent(self, event):
        """Handle mouse clicks"""
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.slot_number)
        super().mousePressEvent(event)


class PresetManagerWidget(QWidget):
    """Main preset manager widget"""
    
    def __init__(self, preset_manager: PresetManager, parent=None, config_menu=None):
        super().__init__(parent)
        self.preset_manager = preset_manager
        self.visualizer = None  # Will be set by parent
        self.config_menu = config_menu  # Reference to config menu for visualizer access
        self.current_preset_info = None
        self.setup_ui()
        self.refresh_preset_list()
    
    def set_visualizer(self, visualizer):
        """Set the visualizer reference"""
        self.visualizer = visualizer
    
    def get_visualizer(self):
        """Get the visualizer reference, trying multiple sources"""
        # Try direct reference first
        if self.visualizer:
            return self.visualizer
        
        # Try config menu reference
        if self.config_menu and hasattr(self.config_menu, 'visualizer') and self.config_menu.visualizer:
            return self.config_menu.visualizer
        
        return None
    
    def setup_ui(self):
        """Set up the main UI"""
        layout = QVBoxLayout(self)
        
        # Create tab widget for different sections
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Quick Presets tab
        self.setup_quick_presets_tab()
        
        # Preset Library tab
        self.setup_preset_library_tab()
        
        # Import/Export tab
        self.setup_import_export_tab()
    
    def setup_quick_presets_tab(self):
        """Set up the quick presets tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Title and instructions
        title = QLabel("Quick Presets")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)
        
        instructions = QLabel(
            "Quick presets allow instant save/load of your current visualizer state.\n"
            "• Ctrl+0-9: Save current state to slot\n"
            "• 0-9: Load preset from slot\n"
            "• Click slots below to load presets"
        )
        instructions.setStyleSheet("color: #666; margin-bottom: 15px;")
        layout.addWidget(instructions)
        
        # Quick preset slots grid
        slots_group = QGroupBox("Quick Preset Slots")
        slots_layout = QGridLayout(slots_group)
        
        self.quick_preset_slots = []
        for i in range(10):
            slot = QuickPresetSlot(i, self.preset_manager)
            slot.clicked.connect(self.load_quick_preset)
            self.quick_preset_slots.append(slot)
            
            # Arrange in 2 rows of 5
            row = i // 5
            col = i % 5
            slots_layout.addWidget(slot, row, col)
        
        layout.addWidget(slots_group)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.save_current_button = QPushButton("Save Current State to Slot...")
        self.save_current_button.clicked.connect(self.save_current_to_slot)
        button_layout.addWidget(self.save_current_button)
        
        self.clear_slot_button = QPushButton("Clear Slot...")
        self.clear_slot_button.clicked.connect(self.clear_slot)
        button_layout.addWidget(self.clear_slot_button)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "Quick Presets")
    
    def setup_preset_library_tab(self):
        """Set up the preset library tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Search and filter controls
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search:"))
        
        self.search_field = QLineEdit()
        self.search_field.setPlaceholderText("Search presets by name, author, or tags...")
        self.search_field.textChanged.connect(self.filter_presets)
        search_layout.addWidget(self.search_field)
        
        self.tag_filter = QComboBox()
        self.tag_filter.addItem("All Tags")
        self.tag_filter.currentTextChanged.connect(self.filter_presets)
        search_layout.addWidget(self.tag_filter)
        
        layout.addLayout(search_layout)
        
        # Main content area with splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Preset list
        list_widget = QWidget()
        list_layout = QVBoxLayout(list_widget)
        
        list_layout.addWidget(QLabel("Preset Library"))
        self.preset_list = QListWidget()
        self.preset_list.itemSelectionChanged.connect(self.on_preset_selected)
        self.preset_list.itemDoubleClicked.connect(self.load_selected_preset)
        list_layout.addWidget(self.preset_list)
        
        # List action buttons
        list_button_layout = QHBoxLayout()
        
        self.load_preset_button = QPushButton("Load Preset")
        self.load_preset_button.clicked.connect(self.load_selected_preset)
        self.load_preset_button.setEnabled(False)
        list_button_layout.addWidget(self.load_preset_button)
        
        self.delete_preset_button = QPushButton("Delete Preset")
        self.delete_preset_button.clicked.connect(self.delete_selected_preset)
        self.delete_preset_button.setEnabled(False)
        list_button_layout.addWidget(self.delete_preset_button)
        
        list_layout.addLayout(list_button_layout)
        splitter.addWidget(list_widget)
        
        # Preset details panel
        details_widget = QWidget()
        details_layout = QVBoxLayout(details_widget)
        
        details_layout.addWidget(QLabel("Preset Details"))
        
        self.preset_details = QTextEdit()
        self.preset_details.setReadOnly(True)
        self.preset_details.setMaximumHeight(200)
        details_layout.addWidget(self.preset_details)
        
        # Save new preset section
        save_group = QGroupBox("Save New Preset")
        save_layout = QVBoxLayout(save_group)
        
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Name:"))
        self.preset_name_field = QLineEdit()
        name_layout.addWidget(self.preset_name_field)
        save_layout.addLayout(name_layout)
        
        desc_layout = QHBoxLayout()
        desc_layout.addWidget(QLabel("Description:"))
        self.preset_desc_field = QLineEdit()
        desc_layout.addWidget(self.preset_desc_field)
        save_layout.addLayout(desc_layout)
        
        author_layout = QHBoxLayout()
        author_layout.addWidget(QLabel("Author:"))
        self.preset_author_field = QLineEdit()
        self.preset_author_field.setText("User")
        author_layout.addWidget(self.preset_author_field)
        save_layout.addLayout(author_layout)
        
        self.save_new_preset_button = QPushButton("Save Current State as New Preset")
        self.save_new_preset_button.clicked.connect(self.save_new_preset)
        save_layout.addWidget(self.save_new_preset_button)
        
        details_layout.addWidget(save_group)
        details_layout.addStretch()
        
        splitter.addWidget(details_widget)
        splitter.setSizes([300, 400])
        
        layout.addWidget(splitter)
        self.tab_widget.addTab(tab, "Preset Library")
    
    def setup_import_export_tab(self):
        """Set up the import/export tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Import section
        import_group = QGroupBox("Import Presets")
        import_layout = QVBoxLayout(import_group)
        
        import_info = QLabel(
            "Import presets from .kvp files created by KarmaViz or shared by other users."
        )
        import_info.setWordWrap(True)
        import_layout.addWidget(import_info)
        
        import_button_layout = QHBoxLayout()
        self.import_preset_button = QPushButton("Import Preset File...")
        self.import_preset_button.clicked.connect(self.import_preset)
        import_button_layout.addWidget(self.import_preset_button)
        
        self.import_folder_button = QPushButton("Import from Folder...")
        self.import_folder_button.clicked.connect(self.import_from_folder)
        import_button_layout.addWidget(self.import_folder_button)
        
        import_button_layout.addStretch()
        import_layout.addLayout(import_button_layout)
        
        layout.addWidget(import_group)
        
        # Export section
        export_group = QGroupBox("Export Presets")
        export_layout = QVBoxLayout(export_group)
        
        export_info = QLabel(
            "Export presets to share with others or backup your collection."
        )
        export_info.setWordWrap(True)
        export_layout.addWidget(export_info)
        
        export_button_layout = QHBoxLayout()
        self.export_preset_button = QPushButton("Export Selected Preset...")
        self.export_preset_button.clicked.connect(self.export_selected_preset)
        self.export_preset_button.setEnabled(False)
        export_button_layout.addWidget(self.export_preset_button)
        
        self.export_all_button = QPushButton("Export All Presets...")
        self.export_all_button.clicked.connect(self.export_all_presets)
        export_button_layout.addWidget(self.export_all_button)
        
        export_button_layout.addStretch()
        export_layout.addLayout(export_button_layout)
        
        layout.addWidget(export_group)
        
        # Backup section
        backup_group = QGroupBox("Backup & Restore")
        backup_layout = QVBoxLayout(backup_group)
        
        backup_info = QLabel(
            "Create backups of your entire preset collection or restore from backups."
        )
        backup_info.setWordWrap(True)
        backup_layout.addWidget(backup_info)
        
        backup_button_layout = QHBoxLayout()
        self.create_backup_button = QPushButton("Create Backup...")
        self.create_backup_button.clicked.connect(self.create_backup)
        backup_button_layout.addWidget(self.create_backup_button)
        
        self.restore_backup_button = QPushButton("Restore from Backup...")
        self.restore_backup_button.clicked.connect(self.restore_backup)
        backup_button_layout.addWidget(self.restore_backup_button)
        
        backup_button_layout.addStretch()
        backup_layout.addLayout(backup_button_layout)
        
        layout.addWidget(backup_group)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "Import/Export")
    
    def refresh_preset_list(self):
        """Refresh the preset list and quick preset slots"""
        # Update quick preset slots
        for slot in self.quick_preset_slots:
            slot.update_status()
        
        # Update preset library
        self.preset_list.clear()
        
        try:
            # Get all presets from user directory
            presets = self.preset_manager.list_presets(self.preset_manager.user_presets_dir)
            
            for filepath, preset_info in presets:
                item = QListWidgetItem(f"{preset_info.name} - {preset_info.author}")
                item.setData(Qt.UserRole, (filepath, preset_info))
                self.preset_list.addItem(item)
            
            # Update tag filter
            self.update_tag_filter(presets)
            
        except Exception as e:
            logger.error(f"Error refreshing preset list: {e}")
    
    def update_tag_filter(self, presets: List[Tuple[Path, PresetInfo]]):
        """Update the tag filter dropdown"""
        all_tags = set()
        for _, preset_info in presets:
            if preset_info.tags:
                all_tags.update(preset_info.tags)
        
        current_text = self.tag_filter.currentText()
        self.tag_filter.clear()
        self.tag_filter.addItem("All Tags")
        
        for tag in sorted(all_tags):
            self.tag_filter.addItem(tag)
        
        # Restore selection if possible
        index = self.tag_filter.findText(current_text)
        if index >= 0:
            self.tag_filter.setCurrentIndex(index)
    
    def filter_presets(self):
        """Filter presets based on search text and tag filter"""
        search_text = self.search_field.text().lower()
        tag_filter = self.tag_filter.currentText()
        
        for i in range(self.preset_list.count()):
            item = self.preset_list.item(i)
            filepath, preset_info = item.data(Qt.UserRole)
            
            # Check search text
            search_match = (
                search_text in preset_info.name.lower() or
                search_text in preset_info.author.lower() or
                search_text in preset_info.description.lower() or
                any(search_text in tag.lower() for tag in preset_info.tags)
            )
            
            # Check tag filter
            tag_match = (
                tag_filter == "All Tags" or
                tag_filter in preset_info.tags
            )
            
            item.setHidden(not (search_match and tag_match))
    
    def on_preset_selected(self):
        """Handle preset selection"""
        current_item = self.preset_list.currentItem()
        
        if current_item:
            filepath, preset_info = current_item.data(Qt.UserRole)
            self.current_preset_info = preset_info
            
            # Update details panel
            details = f"""Name: {preset_info.name}
Author: {preset_info.author}
Created: {preset_info.created_date}
Description: {preset_info.description}

Tags: {', '.join(preset_info.tags) if preset_info.tags else 'None'}

File: {os.path.basename(filepath)}"""
            
            self.preset_details.setPlainText(details)
            
            # Enable buttons
            self.load_preset_button.setEnabled(True)
            self.delete_preset_button.setEnabled(True)
            self.export_preset_button.setEnabled(True)
        else:
            self.current_preset_info = None
            self.preset_details.clear()
            self.load_preset_button.setEnabled(False)
            self.delete_preset_button.setEnabled(False)
            self.export_preset_button.setEnabled(False)
    
    def load_quick_preset(self, slot: int):
        """Load a quick preset"""
        if not self.preset_manager.quick_preset_exists(slot):
            QMessageBox.information(self, "Quick Preset", f"Quick preset {slot} is empty.")
            return
        
        # Check if visualizer is available
        visualizer = self.get_visualizer()
        if visualizer:
            success = self.preset_manager.load_quick_preset(visualizer, slot)
            if success:
                logger.debug(f"Quick preset {slot} shaders loaded successfully!")
            else:
                QMessageBox.warning(self, "Quick Preset", f"Failed to load quick preset {slot} shaders.")
        else:
            QMessageBox.warning(self, "Error", "Visualizer not available.")
    
    def save_current_to_slot(self):
        """Save current state to a quick preset slot"""
        slot, ok = QInputDialog.getInt(
            self, "Save Quick Preset", 
            "Enter slot number (0-9):", 
            0, 0, 9, 1
        )
        
        if not ok:
            return
        
        # Check if visualizer is available
        visualizer = self.get_visualizer()
        if visualizer:
            success = self.preset_manager.save_quick_preset(visualizer, slot)
            if success:
                self.refresh_preset_list()
                QMessageBox.information(self, "Quick Preset", f"Current state saved to quick preset {slot}!")
            else:
                QMessageBox.warning(self, "Quick Preset", f"Failed to save to quick preset {slot}.")
        else:
            QMessageBox.warning(self, "Error", "Visualizer not available.")
    
    def clear_slot(self):
        """Clear a quick preset slot"""
        slot, ok = QInputDialog.getInt(
            self, "Clear Quick Preset", 
            "Enter slot number to clear (0-9):", 
            0, 0, 9, 1
        )
        
        if not ok:
            return
        
        if not self.preset_manager.quick_preset_exists(slot):
            QMessageBox.information(self, "Quick Preset", f"Quick preset {slot} is already empty.")
            return
        
        reply = QMessageBox.question(
            self, "Clear Quick Preset",
            f"Are you sure you want to clear quick preset {slot}?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            filepath = self.preset_manager.quick_presets_dir / f"quick_{slot}.kviz"
            if filepath.exists():
                filepath.unlink()
                self.refresh_preset_list()
                QMessageBox.information(self, "Quick Preset", f"Quick preset {slot} cleared.")
    
    def load_selected_preset(self):
        """Load the selected preset from the library"""
        if not self.current_preset_info:
            return
        
        # Check if visualizer is available
        visualizer = self.get_visualizer()
        if visualizer:
            current_item = self.preset_list.currentItem()
            filepath, preset_info = current_item.data(Qt.UserRole)
            
            success = self.preset_manager.apply_preset(visualizer, preset_info)
            if success:
                logger.debug(f"Preset '{preset_info.name}' shaders loaded successfully!")
            else:
                logger.warning(f"Failed to load preset '{preset_info.name}' shaders.")
        else:
            logger.error("Visualizer not available.")
    
    def save_new_preset(self):
        """Save current state as a new preset"""
        name = self.preset_name_field.text().strip()
        if not name:
            QMessageBox.warning(self, "Save Preset", "Please enter a preset name.")
            return
        
        description = self.preset_desc_field.text().strip()
        author = self.preset_author_field.text().strip() or "User"
        
        # Check if visualizer is available
        visualizer = self.get_visualizer()
        if visualizer:
            success = self.preset_manager.save_preset(
                visualizer, name, description, author
            )
            if success:
                self.refresh_preset_list()
                self.preset_name_field.clear()
                self.preset_desc_field.clear()
                QMessageBox.information(self, "Preset Saved", f"Preset '{name}' saved successfully!")
            else:
                QMessageBox.warning(self, "Save Error", f"Failed to save preset '{name}'.")
        else:
            QMessageBox.warning(self, "Error", "Visualizer not available.")
    
    def delete_selected_preset(self):
        """Delete the selected preset"""
        if not self.current_preset_info:
            return
        
        reply = QMessageBox.question(
            self, "Delete Preset",
            f"Are you sure you want to delete preset '{self.current_preset_info.name}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            current_item = self.preset_list.currentItem()
            filepath, preset_info = current_item.data(Qt.UserRole)
            
            success = self.preset_manager.delete_preset(filepath)
            if success:
                self.refresh_preset_list()
                QMessageBox.information(self, "Preset Deleted", f"Preset '{preset_info.name}' deleted.")
            else:
                QMessageBox.warning(self, "Delete Error", f"Failed to delete preset '{preset_info.name}'.")
    
    def import_preset(self):
        """Import a single preset file"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Import Preset", "", "KarmaViz Presets (*.kviz);;All Files (*)"
        )
        
        if filepath:
            success = self.preset_manager.import_preset(Path(filepath))
            if success:
                self.refresh_preset_list()
                QMessageBox.information(self, "Import Success", "Preset imported successfully!")
            else:
                QMessageBox.warning(self, "Import Error", "Failed to import preset.")
    
    def import_from_folder(self):
        """Import all presets from a folder"""
        folder = QFileDialog.getExistingDirectory(self, "Import Presets from Folder")
        
        if folder:
            folder_path = Path(folder)
            imported_count = 0
            
            for preset_file in folder_path.glob("*.kviz"):
                if self.preset_manager.import_preset(preset_file):
                    imported_count += 1
            
            self.refresh_preset_list()
            QMessageBox.information(
                self, "Import Complete", 
                f"Imported {imported_count} presets from folder."
            )
    
    def export_selected_preset(self):
        """Export the selected preset"""
        if not self.current_preset_info:
            return
        
        current_item = self.preset_list.currentItem()
        filepath, preset_info = current_item.data(Qt.UserRole)
        
        export_path, _ = QFileDialog.getSaveFileName(
            self, "Export Preset", 
            f"{preset_info.name}.kviz",
            "KarmaViz Presets (*.kviz);;All Files (*)"
        )
        
        if export_path:
            success = self.preset_manager.export_preset(filepath, Path(export_path))
            if success:
                QMessageBox.information(self, "Export Success", "Preset exported successfully!")
            else:
                QMessageBox.warning(self, "Export Error", "Failed to export preset.")
    
    def export_all_presets(self):
        """Export all presets to a folder"""
        folder = QFileDialog.getExistingDirectory(self, "Export All Presets to Folder")
        
        if folder:
            folder_path = Path(folder)
            exported_count = 0
            
            presets = self.preset_manager.list_presets(self.preset_manager.user_presets_dir)
            for filepath, preset_info in presets:
                export_path = folder_path / f"{preset_info.name}.kviz"
                if self.preset_manager.export_preset(filepath, export_path):
                    exported_count += 1
            
            QMessageBox.information(
                self, "Export Complete",
                f"Exported {exported_count} presets to folder."
            )
    
    def create_backup(self):
        """Create a backup of all presets"""
        backup_path, _ = QFileDialog.getSaveFileName(
            self, "Create Preset Backup",
            f"karmaviz_presets_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            "ZIP Archives (*.zip);;All Files (*)"
        )
        
        if backup_path:
            try:
                import zipfile
                
                with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # Add all preset files
                    for preset_dir in [self.preset_manager.user_presets_dir, 
                                     self.preset_manager.quick_presets_dir]:
                        for preset_file in preset_dir.glob("*.kviz"):
                            arcname = f"{preset_dir.name}/{preset_file.name}"
                            zipf.write(preset_file, arcname)
                
                QMessageBox.information(self, "Backup Created", "Preset backup created successfully!")
                
            except Exception as e:
                QMessageBox.warning(self, "Backup Error", f"Failed to create backup: {e}")
    
    def restore_backup(self):
        """Restore presets from a backup"""
        backup_path, _ = QFileDialog.getOpenFileName(
            self, "Restore Preset Backup", "", "ZIP Archives (*.zip);;All Files (*)"
        )
        
        if backup_path:
            reply = QMessageBox.question(
                self, "Restore Backup",
                "This will restore presets from the backup. Existing presets with the same names will be overwritten. Continue?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                try:
                    import zipfile
                    
                    with zipfile.ZipFile(backup_path, 'r') as zipf:
                        zipf.extractall(self.preset_manager.presets_dir)
                    
                    self.refresh_preset_list()
                    QMessageBox.information(self, "Restore Complete", "Presets restored from backup successfully!")
                    
                except Exception as e:
                    QMessageBox.warning(self, "Restore Error", f"Failed to restore backup: {e}")
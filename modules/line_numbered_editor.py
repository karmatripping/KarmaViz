"""
Line-numbered code editor with syntax highlighting and error highlighting
"""

from PyQt5.QtWidgets import QWidget, QPlainTextEdit, QTextEdit, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import Qt, QRect, pyqtSignal
from PyQt5.QtGui import QPainter, QColor, QTextFormat, QTextCursor, QFont, QPen
from typing import List, Tuple, Optional


class LineNumberArea(QWidget):
    """Widget to display line numbers"""
    
    def __init__(self, editor):
        super().__init__(editor)
        self.code_editor = editor
        
    def sizeHint(self):
        return self.code_editor.line_number_area_width()
        
    def paintEvent(self, event):
        self.code_editor.line_number_area_paint_event(event)


class LineNumberedCodeEditor(QPlainTextEdit):
    """Code editor with line numbers and error highlighting"""
    
    error_line_changed = pyqtSignal(int)  # Emitted when cursor moves to error line
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.line_number_area = LineNumberArea(self)
        self.error_lines: List[int] = []  # 1-based line numbers with errors
        self.error_messages: dict = {}  # line_number -> error_message
        self.line_number_offset = 0  # Offset to add to displayed line numbers
        
        # Connect signals
        self.blockCountChanged.connect(self.update_line_number_area_width)
        self.updateRequest.connect(self.update_line_number_area)
        self.cursorPositionChanged.connect(self.highlight_current_line)
        
        self.update_line_number_area_width(0)
        self.highlight_current_line()
        
        # Set font
        font = QFont("Consolas", 10)
        if not font.exactMatch():
            font = QFont("Monaco", 10)
            if not font.exactMatch():
                font = QFont("Courier New", 10)
        self.setFont(font)
        
    def line_number_area_width(self):
        """Calculate the width needed for line numbers"""
        # Calculate the maximum line number that will be displayed
        max_line_in_editor = self.blockCount()
        max_displayed_line = max_line_in_editor + self.line_number_offset
        
        digits = 1
        max_num = max(1, max_displayed_line)
        while max_num >= 10:
            max_num //= 10
            digits += 1
        
        # Ensure minimum width for at least 3 digits
        digits = max(digits, 3)
        
        space = 3 + self.fontMetrics().horizontalAdvance('9') * digits
        return space
        
    def update_line_number_area_width(self, new_block_count):
        """Update the width of the line number area"""
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)
        
    def update_line_number_area(self, rect, dy):
        """Update the line number area when scrolling"""
        if dy:
            self.line_number_area.scroll(0, dy)
        else:
            self.line_number_area.update(0, rect.y(), self.line_number_area.width(), rect.height())
            
        if rect.contains(self.viewport().rect()):
            self.update_line_number_area_width(0)
            
    def resizeEvent(self, event):
        """Handle resize events"""
        super().resizeEvent(event)
        
        cr = self.contentsRect()
        self.line_number_area.setGeometry(
            QRect(cr.left(), cr.top(), self.line_number_area_width(), cr.height())
        )
        
    def line_number_area_paint_event(self, event):
        """Paint the line number area"""
        painter = QPainter(self.line_number_area)
        painter.fillRect(event.rect(), QColor(60, 60, 60))  # Dark background
        
        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = self.blockBoundingGeometry(block).translated(self.contentOffset()).top()
        bottom = top + self.blockBoundingRect(block).height()
        
        # Set up text color
        painter.setPen(QColor(150, 150, 150))  # Light gray for line numbers
        
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                line_number = block_number + 1 + self.line_number_offset
                
                # Highlight error lines in yellow
                # Convert displayed line number back to editor line number for error checking
                editor_line_number = block_number + 1
                if editor_line_number in self.error_lines:
                    painter.fillRect(
                        0, int(top), self.line_number_area.width(), 
                        int(self.blockBoundingRect(block).height()),
                        QColor(200, 50, 50)  # More visible red background for error lines
                    )
                    painter.setPen(QColor(255, 255, 255))  # White text for contrast
                else:
                    painter.setPen(QColor(150, 150, 150))  # Normal gray text
                
                painter.drawText(
                    0, int(top), self.line_number_area.width() - 3, 
                    int(self.fontMetrics().height()),
                    Qt.AlignRight, str(line_number)
                )
                
            block = block.next()
            top = bottom
            bottom = top + self.blockBoundingRect(block).height()
            block_number += 1
            
    def highlight_current_line(self):
        """Highlight the current line"""
        extra_selections = []
        
        if not self.isReadOnly():
            selection = QTextEdit.ExtraSelection()
            
            line_color = QColor(70, 70, 70)  # Dark gray for current line
            selection.format.setBackground(line_color)
            selection.format.setProperty(QTextFormat.FullWidthSelection, True)
            selection.cursor = self.textCursor()
            selection.cursor.clearSelection()
            extra_selections.append(selection)
            
        # Add error line highlighting
        for line_num in self.error_lines:
            if 1 <= line_num <= self.blockCount():
                selection = QTextEdit.ExtraSelection()
                error_color = QColor(255, 100, 100, 150)  # More visible red with transparency
                selection.format.setBackground(error_color)
                selection.format.setProperty(QTextFormat.FullWidthSelection, True)
                
                # Move cursor to the error line
                cursor = QTextCursor(self.document().findBlockByLineNumber(line_num - 1))
                selection.cursor = cursor
                extra_selections.append(selection)
        
        self.setExtraSelections(extra_selections)
        
        # Emit signal if cursor is on an error line
        current_line = self.textCursor().blockNumber() + 1
        if current_line in self.error_lines:
            self.error_line_changed.emit(current_line)
            
    def set_error_lines(self, error_lines: List[int], error_messages: dict = None):
        """Set lines that have errors using displayed line numbers (with offset)"""
        self.error_lines = []
        self.error_messages = {}
        
        for displayed_line in error_lines:
            # Convert displayed line number to editor line number
            editor_line = displayed_line - self.line_number_offset
            if editor_line > 0:  # Only show errors that fall within this editor
                self.error_lines.append(editor_line)
                if error_messages and displayed_line in error_messages:
                    self.error_messages[editor_line] = error_messages[displayed_line]
        
        self.highlight_current_line()
        self.line_number_area.update()
        
    def clear_errors(self):
        """Clear all error highlighting"""
        self.error_lines = []
        self.error_messages = {}
        self.highlight_current_line()
        self.line_number_area.update()
        
    def get_error_message(self, line_number: int) -> Optional[str]:
        """Get error message for a specific line"""
        return self.error_messages.get(line_number)
        
    def get_current_line_number(self) -> int:
        """Get the current line number (1-based)"""
        return self.textCursor().blockNumber() + 1
    
    def set_line_number_offset(self, offset: int):
        """Set the line number offset for display"""
        self.line_number_offset = offset
        self.update_line_number_area_width(0)  # Update width to accommodate new numbers
        self.line_number_area.update()  # Refresh line numbers

    def set_errors_from_main_shader(self, errors: List[Tuple[int, str]]):
        """Set errors using main shader line numbers (displayed line numbers), converting to editor line numbers"""
        self.error_lines = []
        self.error_messages = {}
        
        for displayed_line, error_msg in errors:
            # Convert displayed line number to editor line number
            editor_line = displayed_line - self.line_number_offset
            if editor_line > 0:  # Only show errors that fall within this editor
                self.error_lines.append(editor_line)
                self.error_messages[editor_line] = error_msg
        
        self.highlight_current_line()
        self.line_number_area.update()
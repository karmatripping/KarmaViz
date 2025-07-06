"""
GLSL Syntax Highlighter for KarmaViz

This module provides syntax highlighting for GLSL shader code in Qt text editors.
Shared between waveform editor and warp map editor.
"""

import re
from PyQt5.QtGui import QSyntaxHighlighter, QTextCharFormat, QColor, QFont, QTextDocument


class GLSLSyntaxHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for GLSL code"""

    def __init__(self, document: QTextDocument):
        super().__init__(document)
        self.setup_highlighting_rules()

    def setup_highlighting_rules(self):
        """Set up syntax highlighting rules for GLSL"""
        self.highlighting_rules = []

        # GLSL keywords
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor(86, 156, 214))  # Light blue
        keyword_format.setFontWeight(QFont.Bold)

        keywords = [
            "attribute",
            "const",
            "uniform",
            "varying",
            "break",
            "continue",
            "do",
            "for",
            "while",
            "if",
            "else",
            "in",
            "out",
            "inout",
            "float",
            "int",
            "void",
            "bool",
            "true",
            "false",
            "invariant",
            "discard",
            "return",
            "mat2",
            "mat3",
            "mat4",
            "vec2",
            "vec3",
            "vec4",
            "ivec2",
            "ivec3",
            "ivec4",
            "bvec2",
            "bvec3",
            "bvec4",
            "sampler2D",
            "samplerCube",
            "sampler1D",
            "sampler3D",
            "struct",
            "precision",
            "lowp",
            "mediump",
            "highp",
        ]

        for keyword in keywords:
            pattern = f'\\b{keyword}\\b'
            self.highlighting_rules.append((pattern, keyword_format))

        # GLSL built-in functions
        function_format = QTextCharFormat()
        function_format.setForeground(QColor(220, 220, 170))  # Light yellow

        functions = [
            "sin",
            "cos",
            "tan",
            "asin",
            "acos",
            "atan",
            "pow",
            "exp",
            "log",
            "exp2",
            "log2",
            "sqrt",
            "inversesqrt",
            "abs",
            "sign",
            "floor",
            "ceil",
            "fract",
            "mod",
            "min",
            "max",
            "clamp",
            "mix",
            "step",
            "smoothstep",
            "length",
            "distance",
            "dot",
            "cross",
            "normalize",
            "reflect",
            "refract",
            "texture2D",
            "texture",
            "textureCube",
        ]

        for function in functions:
            pattern = f"\\b{function}\\b"
            self.highlighting_rules.append((pattern, function_format))

        # Numbers
        number_format = QTextCharFormat()
        number_format.setForeground(QColor(181, 206, 168))  # Light green
        self.highlighting_rules.append((r"\b\d+\.?\d*f?\b", number_format))

        # Comments
        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor(106, 153, 85))  # Green
        comment_format.setFontItalic(True)
        self.highlighting_rules.append((r"//[^\n]*", comment_format))
        self.highlighting_rules.append((r"/\*.*\*/", comment_format))

        # Strings
        string_format = QTextCharFormat()
        string_format.setForeground(QColor(206, 145, 120))  # Orange
        self.highlighting_rules.append((r'".*"', string_format))

    def highlightBlock(self, text: str):
        """Apply syntax highlighting to a block of text"""
        for pattern, format_obj in self.highlighting_rules:
            for match in re.finditer(pattern, text):
                start, end = match.span()
                self.setFormat(start, end - start, format_obj)
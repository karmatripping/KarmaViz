"""
Input Sanitization Module for KarmaViz

This module provides comprehensive input sanitization for user-provided data
including GLSL code, metadata fields, file paths, and binary data validation.
"""

import os
import re
import pathlib
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass

from modules.logging_config import get_logger

logger = get_logger("input_sanitizer")


@dataclass
class ValidationResult:
    """Result of input validation"""
    is_valid: bool
    sanitized_value: Any
    warnings: List[str]
    errors: List[str]


class InputSanitizer:
    """Comprehensive input sanitization for KarmaViz editors"""
    
    # Security limits
    MAX_NAME_LENGTH = 100
    MAX_DESCRIPTION_LENGTH = 2000
    MAX_AUTHOR_LENGTH = 100
    MAX_VERSION_LENGTH = 20
    MAX_CATEGORY_LENGTH = 50
    MAX_GLSL_CODE_LENGTH = 50000  # 50KB limit for GLSL code
    MAX_FILE_SIZE = 1024 * 1024  # 1MB limit for imported files
    
    # Allowed characters for different fields
    NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\s\.\(\)]+$')  # Allow parentheses for names
    VERSION_PATTERN = re.compile(r'^[0-9]+\.[0-9]+(\.[0-9]+)?([a-zA-Z0-9\-]*)?$')
    CATEGORY_PATTERN = re.compile(r'^[a-zA-Z0-9_\-]+$')
    AUTHOR_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\s\.\@\(\)<>]+$')  # Allow @, (), <> for authors
    
    # GLSL security patterns
    GLSL_DANGEROUS_PATTERNS = [
        # System/file operations (shouldn't exist in GLSL but check anyway)
        r'#include\s*[<"].*[>"]',
        r'#pragma\s+.*',
        # Potential shader bombs or infinite loops
        r'while\s*\(\s*true\s*\)',
        r'for\s*\([^;]*;\s*true\s*;[^)]*\)',
        # Excessive recursion patterns
        r'(\w+)\s*\([^)]*\)\s*{[^}]*\1\s*\(',  # Simple recursion detection
    ]
    
    # Allowed GLSL built-in functions and keywords
    GLSL_ALLOWED_FUNCTIONS = {
        # Math functions
        'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh',
        'pow', 'exp', 'log', 'exp2', 'log2', 'sqrt', 'inversesqrt',
        'abs', 'sign', 'floor', 'ceil', 'fract', 'mod', 'min', 'max', 'clamp',
        'mix', 'step', 'smoothstep', 'length', 'distance', 'dot', 'cross',
        'normalize', 'reflect', 'refract', 'faceforward',
        # Vector functions
        'radians', 'degrees', 'dFdx', 'dFdy', 'fwidth',
        # Texture functions (limited set)
        'texture2D', 'texture', 'texelFetch',
        # Type constructors
        'vec2', 'vec3', 'vec4', 'mat2', 'mat3', 'mat4', 'float', 'int', 'bool',
        # Control flow
        'if', 'else', 'for', 'while', 'do', 'break', 'continue', 'return',
        'discard',
    }
    
    # Reserved/dangerous variable names
    GLSL_RESERVED_NAMES = {
        'gl_Position', 'gl_FragColor', 'gl_FragData', 'gl_FragDepth',
        'gl_FrontFacing', 'gl_PointCoord', 'gl_FragCoord',
    }

    def __init__(self):
        """Initialize the input sanitizer"""
        self.compiled_glsl_patterns = [re.compile(pattern, re.IGNORECASE) 
                                     for pattern in self.GLSL_DANGEROUS_PATTERNS]

    def sanitize_filename(self, filename: str) -> ValidationResult:
        """
        Sanitize filename to prevent path traversal and ensure filesystem safety
        
        Args:
            filename: Raw filename input
            
        Returns:
            ValidationResult with sanitized filename
        """
        warnings = []
        errors = []
        
        if not filename or not filename.strip():
            return ValidationResult(False, "", [], ["Filename cannot be empty"])
        
        original_filename = filename.strip()
        
        # Check for path traversal attempts BEFORE stripping path components
        if '..' in original_filename or '/' in original_filename or '\\' in original_filename:
            errors.append("Filename contains path separators or traversal patterns")
        
        # Check for hidden files
        if original_filename.startswith('.'):
            errors.append("Filename cannot start with a dot (hidden files not allowed)")
        
        # Now extract just the filename component for further processing
        filename = os.path.basename(original_filename)
        
        # Remove or replace dangerous characters
        original_filename = filename
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        filename = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', filename)  # Remove control chars
        
        if filename != original_filename:
            warnings.append("Filename contained invalid characters that were replaced")
        
        # Ensure reasonable length
        if len(filename) > 255:
            filename = filename[:255]
            warnings.append("Filename was truncated to 255 characters")
        
        # Ensure it's not a reserved name (Windows)
        reserved_names = {'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 
                         'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 
                         'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 
                         'LPT7', 'LPT8', 'LPT9'}
        
        name_without_ext = os.path.splitext(filename)[0].upper()
        if name_without_ext in reserved_names:
            filename = f"safe_{filename}"
            warnings.append("Filename was a reserved system name and was prefixed")
        
        is_valid = len(errors) == 0 and len(filename) > 0
        return ValidationResult(is_valid, filename, warnings, errors)

    def sanitize_name(self, name: str) -> ValidationResult:
        """
        Sanitize name fields (waveform name, warp map name, etc.)
        
        Args:
            name: Raw name input
            
        Returns:
            ValidationResult with sanitized name
        """
        warnings = []
        errors = []
        
        if not name or not name.strip():
            return ValidationResult(False, "", [], ["Name cannot be empty"])
        
        name = name.strip()
        
        # Check length
        if len(name) > self.MAX_NAME_LENGTH:
            name = name[:self.MAX_NAME_LENGTH]
            warnings.append(f"Name truncated to {self.MAX_NAME_LENGTH} characters")
        
        # Check for and remove potential control characters or null bytes first
        if any(ord(c) < 32 for c in name if c not in '\t\n\r'):
            name = ''.join(c for c in name if ord(c) >= 32 or c in '\t\n\r')
            warnings.append("Control characters were removed from name")
        
        # Check for valid characters after control character removal
        if not self.NAME_PATTERN.match(name):
            errors.append("Name contains invalid characters. Only letters, numbers, spaces, hyphens, underscores, dots, and parentheses are allowed")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, name, warnings, errors)

    def sanitize_description(self, description: str) -> ValidationResult:
        """
        Sanitize description fields
        
        Args:
            description: Raw description input
            
        Returns:
            ValidationResult with sanitized description
        """
        warnings = []
        errors = []
        
        if not description:
            return ValidationResult(True, "", [], [])
        
        description = description.strip()
        
        # Check length
        if len(description) > self.MAX_DESCRIPTION_LENGTH:
            description = description[:self.MAX_DESCRIPTION_LENGTH]
            warnings.append(f"Description truncated to {self.MAX_DESCRIPTION_LENGTH} characters")
        
        # Check for potential control characters (except common whitespace)
        if any(ord(c) < 32 for c in description if c not in '\t\n\r '):
            description = ''.join(c for c in description if ord(c) >= 32 or c in '\t\n\r ')
            warnings.append("Control characters were removed from description")
        
        # Normalize whitespace (convert multiple spaces/tabs to single spaces)
        description = re.sub(r'[ \t]+', ' ', description)
        description = re.sub(r'\n\s*\n', '\n', description)  # Remove empty lines
        
        return ValidationResult(True, description, warnings, errors)

    def sanitize_author(self, author: str) -> ValidationResult:
        """
        Sanitize author field
        
        Args:
            author: Raw author input
            
        Returns:
            ValidationResult with sanitized author
        """
        warnings = []
        errors = []
        
        if not author:
            return ValidationResult(True, "Unknown", [], [])
        
        author = author.strip()
        
        # Check length
        if len(author) > self.MAX_AUTHOR_LENGTH:
            author = author[:self.MAX_AUTHOR_LENGTH]
            warnings.append(f"Author name truncated to {self.MAX_AUTHOR_LENGTH} characters")
        
        # Check for and remove control characters first
        if any(ord(c) < 32 for c in author if c not in '\t\n\r '):
            author = ''.join(c for c in author if ord(c) >= 32 or c in '\t\n\r ')
            warnings.append("Control characters were removed from author name")
        
        # Basic character validation after control character removal
        if not self.AUTHOR_PATTERN.match(author):
            errors.append("Author name contains invalid characters")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, author, warnings, errors)

    def sanitize_version(self, version: str) -> ValidationResult:
        """
        Sanitize version field
        
        Args:
            version: Raw version input
            
        Returns:
            ValidationResult with sanitized version
        """
        warnings = []
        errors = []
        
        if not version:
            return ValidationResult(True, "1.0", [], [])
        
        version = version.strip()
        
        # Check length
        if len(version) > self.MAX_VERSION_LENGTH:
            version = version[:self.MAX_VERSION_LENGTH]
            warnings.append(f"Version truncated to {self.MAX_VERSION_LENGTH} characters")
        
        # Validate version format
        if not self.VERSION_PATTERN.match(version):
            errors.append("Version must follow semantic versioning format (e.g., 1.0, 1.2.3, 2.0-beta)")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, version, warnings, errors)

    def sanitize_category(self, category: str) -> ValidationResult:
        """
        Sanitize category field
        
        Args:
            category: Raw category input
            
        Returns:
            ValidationResult with sanitized category
        """
        warnings = []
        errors = []
        
        if not category:
            return ValidationResult(True, "custom", [], [])
        
        category = category.strip().lower()
        
        # Check length
        if len(category) > self.MAX_CATEGORY_LENGTH:
            category = category[:self.MAX_CATEGORY_LENGTH]
            warnings.append(f"Category truncated to {self.MAX_CATEGORY_LENGTH} characters")
        
        # Validate category format
        if not self.CATEGORY_PATTERN.match(category):
            errors.append("Category can only contain letters, numbers, hyphens, and underscores")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, category, warnings, errors)

    def sanitize_glsl_code(self, glsl_code: str) -> ValidationResult:
        """
        Sanitize GLSL code for security and validity
        
        Args:
            glsl_code: Raw GLSL code input
            
        Returns:
            ValidationResult with sanitized GLSL code
        """
        warnings = []
        errors = []
        
        if not glsl_code or not glsl_code.strip():
            return ValidationResult(False, "", [], ["GLSL code cannot be empty"])
        
        glsl_code = glsl_code.strip()
        
        # Check length
        if len(glsl_code) > self.MAX_GLSL_CODE_LENGTH:
            errors.append(f"GLSL code exceeds maximum length of {self.MAX_GLSL_CODE_LENGTH} characters")
            return ValidationResult(False, glsl_code, warnings, errors)
        
        # Check for dangerous patterns
        for pattern in self.compiled_glsl_patterns:
            if pattern.search(glsl_code):
                errors.append(f"GLSL code contains potentially dangerous pattern: {pattern.pattern}")
        
        # Check for excessive nesting (potential shader bomb)
        brace_depth = 0
        max_depth = 0
        for char in glsl_code:
            if char == '{':
                brace_depth += 1
                max_depth = max(max_depth, brace_depth)
            elif char == '}':
                brace_depth -= 1
        
        if max_depth > 20:  # Reasonable nesting limit
            warnings.append("GLSL code has very deep nesting which may impact performance")
        
        # Check for excessive line count (potential performance issue)
        line_count = glsl_code.count('\n') + 1
        if line_count > 1000:
            warnings.append("GLSL code has many lines which may impact compilation performance")
        
        # Basic syntax validation - check for balanced braces and parentheses
        if brace_depth != 0:
            errors.append("GLSL code has unbalanced braces")
        
        paren_depth = 0
        for char in glsl_code:
            if char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
                if paren_depth < 0:
                    errors.append("GLSL code has unbalanced parentheses")
                    break
        
        if paren_depth != 0:
            errors.append("GLSL code has unbalanced parentheses")
        
        # Check for use of reserved GL variables in inappropriate contexts
        for reserved_name in self.GLSL_RESERVED_NAMES:
            if reserved_name in glsl_code and 'gl_' in reserved_name:
                # This is just a warning as some uses might be legitimate
                warnings.append(f"GLSL code uses reserved OpenGL variable: {reserved_name}")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, glsl_code, warnings, errors)

    def validate_file_path(self, file_path: str, allowed_extensions: List[str] = None) -> ValidationResult:
        """
        Validate file path for security
        
        Args:
            file_path: File path to validate
            allowed_extensions: List of allowed file extensions (e.g., ['.kvwf', '.glsl'])
            
        Returns:
            ValidationResult with validation status
        """
        warnings = []
        errors = []
        
        if not file_path:
            return ValidationResult(False, "", [], ["File path cannot be empty"])
        
        try:
            # Convert to Path object for safer handling
            path = pathlib.Path(file_path).resolve()
            
            # Check if file exists
            if not path.exists():
                errors.append("File does not exist")
                return ValidationResult(False, str(path), warnings, errors)
            
            # Check if it's actually a file
            if not path.is_file():
                errors.append("Path is not a file")
                return ValidationResult(False, str(path), warnings, errors)
            
            # Check file size
            file_size = path.stat().st_size
            if file_size > self.MAX_FILE_SIZE:
                errors.append(f"File size ({file_size} bytes) exceeds maximum allowed size ({self.MAX_FILE_SIZE} bytes)")
            
            # Check file extension if specified
            if allowed_extensions:
                file_ext = path.suffix.lower()
                if file_ext not in [ext.lower() for ext in allowed_extensions]:
                    errors.append(f"File extension '{file_ext}' is not allowed. Allowed extensions: {allowed_extensions}")
            
            # Check for suspicious file paths
            path_str = str(path)
            if '..' in path_str or path_str.startswith('/etc') or path_str.startswith('/proc'):
                errors.append("File path appears to be attempting path traversal")
            
            is_valid = len(errors) == 0
            return ValidationResult(is_valid, str(path), warnings, errors)
            
        except Exception as e:
            errors.append(f"Error validating file path: {str(e)}")
            return ValidationResult(False, file_path, warnings, errors)

    def validate_binary_data(self, data: bytes, expected_magic: bytes, max_size: int = None) -> ValidationResult:
        """
        Validate binary data format and integrity
        
        Args:
            data: Binary data to validate
            expected_magic: Expected magic header bytes
            max_size: Maximum allowed size in bytes
            
        Returns:
            ValidationResult with validation status
        """
        warnings = []
        errors = []
        
        if not data:
            return ValidationResult(False, data, [], ["Binary data is empty"])
        
        # Check size limits
        if max_size and len(data) > max_size:
            errors.append(f"Binary data size ({len(data)} bytes) exceeds maximum ({max_size} bytes)")
        
        # Check magic header
        if len(data) < len(expected_magic):
            errors.append("Binary data is too short to contain valid header")
        elif data[:len(expected_magic)] != expected_magic:
            errors.append("Binary data has invalid magic header")
        
        # Basic structure validation for our binary formats
        try:
            if len(data) >= 8:  # Minimum size for header + version
                # This is a basic check - the actual parsing will do more thorough validation
                import struct
                version = struct.unpack("<I", data[4:8])[0]
                if version != 1:
                    warnings.append(f"Binary data has unexpected version: {version}")
        except Exception as e:
            errors.append(f"Error validating binary structure: {str(e)}")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, data, warnings, errors)

    def sanitize_all_metadata(self, metadata: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], List[str]]:
        """
        Sanitize all metadata fields at once
        
        Args:
            metadata: Dictionary containing metadata fields
            
        Returns:
            Tuple of (sanitized_metadata, all_warnings, all_errors)
        """
        sanitized = {}
        all_warnings = []
        all_errors = []
        
        # Define field sanitizers
        field_sanitizers = {
            'name': self.sanitize_name,
            'description': self.sanitize_description,
            'author': self.sanitize_author,
            'version': self.sanitize_version,
            'category': self.sanitize_category,
            'glsl_code': self.sanitize_glsl_code,
        }
        
        # Sanitize each field
        for field_name, value in metadata.items():
            if field_name in field_sanitizers:
                result = field_sanitizers[field_name](value)
                sanitized[field_name] = result.sanitized_value
                
                # Add field context to warnings and errors
                for warning in result.warnings:
                    all_warnings.append(f"{field_name}: {warning}")
                for error in result.errors:
                    all_errors.append(f"{field_name}: {error}")
            else:
                # For unknown fields, just pass through with basic sanitization
                if isinstance(value, str):
                    sanitized[field_name] = html.escape(value.strip(), quote=True)
                else:
                    sanitized[field_name] = value
        
        return sanitized, all_warnings, all_errors


# Global instance for easy access
input_sanitizer = InputSanitizer()
"""
Shader error parser for extracting and translating line numbers from OpenGL errors
"""

import re
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class ShaderError:
    """Represents a shader compilation error"""
    line_number: int  # Line number in the full shader
    column: Optional[int]  # Column number if available
    message: str  # Error message
    error_type: str  # Type of error (e.g., 'error', 'warning')


@dataclass
class UserCodeSection:
    """Represents a section of user code within the full shader"""
    name: str  # Name of the section (e.g., 'waveform', 'warp_map')
    start_line: int  # Starting line in the full shader (1-based)
    end_line: int  # Ending line in the full shader (1-based)
    user_start_line: int  # Starting line in user's code (1-based)


class ShaderErrorParser:
    """Parser for OpenGL shader compilation errors"""
    
    # Common OpenGL error patterns
    ERROR_PATTERNS = [
        # NVIDIA: "0(123) : error C1234: message"
        r'(\d+)\((\d+)\)\s*:\s*(error|warning)\s*[A-Z]?\d*:\s*(.+)',
        # AMD: "ERROR: 0:123: message"
        r'(ERROR|WARNING):\s*\d+:(\d+):\s*(.+)',
        # Intel: "ERROR: 123: message"
        r'(ERROR|WARNING):\s*(\d+):\s*(.+)',
        # Generic: "123: error: message"
        r'(\d+):\s*(error|warning):\s*(.+)',
        # Mesa: "123:45(67): error: message"
        r'(\d+):(\d+)\((\d+)\):\s*(error|warning):\s*(.+)',
    ]
    
    def __init__(self):
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.ERROR_PATTERNS]
    
    def parse_shader_errors(self, error_text: str) -> List[ShaderError]:
        """Parse shader compilation errors from error text"""
        errors = []
        
        for line in error_text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            for pattern in self.compiled_patterns:
                match = pattern.search(line)
                if match:
                    error = self._extract_error_from_match(match, line)
                    if error:
                        errors.append(error)
                    break
        
        return errors
    
    def _extract_error_from_match(self, match: re.Match, full_line: str) -> Optional[ShaderError]:
        """Extract error information from regex match"""
        groups = match.groups()
        
        try:
            # Try different group arrangements based on common patterns
            if len(groups) >= 4 and groups[0].isdigit():
                # Pattern: "0(123) : error C1234: message"
                line_num = int(groups[1])
                error_type = groups[2].lower()
                message = groups[3]
                return ShaderError(line_num, None, message, error_type)
            
            elif len(groups) >= 3 and groups[0].upper() in ['ERROR', 'WARNING']:
                # Pattern: "ERROR: 0:123: message"
                if groups[1].isdigit():
                    line_num = int(groups[1])
                    error_type = groups[0].lower()
                    message = groups[2]
                    return ShaderError(line_num, None, message, error_type)
            
            elif len(groups) >= 3 and groups[0].isdigit():
                # Pattern: "123: error: message"
                line_num = int(groups[0])
                error_type = groups[1].lower()
                message = groups[2]
                return ShaderError(line_num, None, message, error_type)
            
            elif len(groups) >= 5:
                # Pattern: "123:45(67): error: message"
                line_num = int(groups[0])
                col_num = int(groups[1]) if groups[1].isdigit() else None
                error_type = groups[3].lower()
                message = groups[4]
                return ShaderError(line_num, col_num, message, error_type)
                
        except (ValueError, IndexError):
            pass
        
        return None
    
    def translate_errors_to_user_code(self, errors: List[ShaderError], 
                                    user_sections: List[UserCodeSection]) -> Dict[str, List[Tuple[int, str]]]:
        """
        Translate shader errors to user code line numbers
        
        Returns:
            Dict mapping section name to list of (user_line_number, error_message) tuples
        """
        translated_errors = {}
        
        for section in user_sections:
            translated_errors[section.name] = []
        
        for error in errors:
            # Find which user section this error belongs to
            for section in user_sections:
                if section.start_line <= error.line_number <= section.end_line:
                    # Translate to user code line number
                    user_line = error.line_number - section.start_line + section.user_start_line
                    error_msg = f"{error.error_type.capitalize()}: {error.message}"
                    translated_errors[section.name].append((user_line, error_msg))
                    break
        
        return translated_errors
    
    def create_waveform_section(self, user_code: str, full_shader: str) -> Optional[UserCodeSection]:
        """Create a UserCodeSection for waveform code within the full shader"""
        return self._find_user_code_section(user_code, full_shader, 'waveform')
    
    def create_warp_map_section(self, user_code: str, full_shader: str) -> Optional[UserCodeSection]:
        """Create a UserCodeSection for warp map code within the full shader"""
        return self._find_user_code_section(user_code, full_shader, 'warp_map')
    
    def _find_user_code_section(self, user_code: str, full_shader: str, section_name: str) -> Optional[UserCodeSection]:
        """Find where user code appears in the full shader"""
        # Clean up the user code for comparison
        user_lines = [line.strip() for line in user_code.split('\n') if line.strip()]
        full_lines = full_shader.split('\n')
        
        if not user_lines:
            return None
        
        # Look for the first line of user code in the full shader
        first_user_line = user_lines[0]
        
        for i, full_line in enumerate(full_lines):
            if first_user_line in full_line.strip():
                # Found potential start, verify by checking a few more lines
                match_count = 0
                for j, user_line in enumerate(user_lines[:min(3, len(user_lines))]):
                    if i + j < len(full_lines) and user_line in full_lines[i + j]:
                        match_count += 1
                
                if match_count >= min(2, len(user_lines)):
                    # Found the section
                    start_line = i + 1  # 1-based
                    end_line = start_line + len(user_lines) - 1
                    return UserCodeSection(
                        name=section_name,
                        start_line=start_line,
                        end_line=end_line,
                        user_start_line=1
                    )
        
        return None


def test_shader_error_parser():
    """Test the shader error parser with common error formats"""
    parser = ShaderErrorParser()
    
    test_errors = [
        "0(123) : error C1234: syntax error",
        "ERROR: 0:456: undeclared identifier",
        "WARNING: 789: unused variable",
        "123: error: missing semicolon",
        "456:12(34): error: type mismatch",
    ]
    
    for error_text in test_errors:
        errors = parser.parse_shader_errors(error_text)
        print(f"Input: {error_text}")
        for error in errors:
            print(f"  -> Line {error.line_number}: {error.error_type} - {error.message}")
        print()


if __name__ == "__main__":
    test_shader_error_parser()
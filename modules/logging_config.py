"""
Logging configuration for KarmaViz
Provides centralized logging with debug flag support
"""

import logging
import sys
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        # Add color to the level name
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


def setup_logging(debug: bool = False) -> logging.Logger:
    """
    Set up logging configuration for KarmaViz
    
    Args:
        debug: If True, enable debug logging. If False, only show ERROR and CRITICAL messages.
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger('karmaviz')
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Set level based on debug flag
    if debug:
        logger.setLevel(logging.DEBUG)
        console_level = logging.DEBUG
    else:
        logger.setLevel(logging.ERROR)
        console_level = logging.ERROR
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    
    # Create formatter
    if debug:
        # Detailed format for debug mode
        formatter = ColoredFormatter(
            '%(levelname)s - %(name)s - %(message)s'
        )
    else:
        # Simple format for production mode
        formatter = ColoredFormatter('%(levelname)s: %(message)s')
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance
    
    Args:
        name: Optional name for the logger. If None, returns the main karmaviz logger.
    
    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f'karmaviz.{name}')
    return logging.getLogger('karmaviz')


# Global logger instance
_logger: Optional[logging.Logger] = None


def init_logging(debug: bool=False) -> None:
    """Initialize the global logging configuration"""
    global _logger
    if debug:
        _logger = setup_logging(debug)


def log_debug(message: str) -> None:
    """Log a debug message"""
    if _logger:
        _logger.debug(message)


def log_info(message: str) -> None:
    """Log an info message"""
    if _logger:
        _logger.info(message)


def log_warning(message: str) -> None:
    """Log a warning message"""
    if _logger:
        _logger.warning(message)


def log_error(message: str) -> None:
    """Log an error message"""
    if _logger:
        _logger.error(message)


def log_critical(message: str) -> None:
    """Log a critical message"""
    if _logger:
        _logger.critical(message)
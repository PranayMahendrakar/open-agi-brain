"""
Utility - Logger
==================
Centralized structured logging for the Open AGI Brain system.
Uses loguru for beautiful, structured, and performant logging.
"""

import sys
import os
from pathlib import Path

try:
    from loguru import logger as _loguru_logger
    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False
    import logging

# Log level from environment (default: INFO)
LOG_LEVEL = os.getenv("AGI_LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("AGI_LOG_FILE", None)


def _setup_loguru():
    """Configure loguru logger."""
    _loguru_logger.remove()  # Remove default handler
    
    # Console handler with color and formatting
    _loguru_logger.add(
        sys.stderr,
        level=LOG_LEVEL,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )
    
    # File handler (optional)
    if LOG_FILE:
        log_path = Path(LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _loguru_logger.add(
            LOG_FILE,
            level=LOG_LEVEL,
            rotation="100 MB",
            retention="30 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        )
    
    return _loguru_logger


class AGILogger:
    """
    Adapter class that wraps loguru or stdlib logging.
    Provides a consistent interface regardless of which backend is available.
    """
    
    def __init__(self, name: str):
        self.name = name
        if LOGURU_AVAILABLE:
            self._logger = _setup_loguru().bind(module=name)
        else:
            self._logger = self._setup_stdlib(name)
    
    def _setup_stdlib(self, name: str):
        """Setup stdlib logging as fallback."""
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s - %(message)s"
            ))
            logger.addHandler(handler)
        logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
        return logger
    
    def debug(self, msg: str, *args, **kwargs):
        self._logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        self._logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        self._logger.critical(msg, *args, **kwargs)
    
    def exception(self, msg: str, *args, **kwargs):
        self._logger.exception(msg, *args, **kwargs)
    
    def success(self, msg: str, *args, **kwargs):
        """Log a success message (loguru-specific)."""
        if LOGURU_AVAILABLE:
            self._logger.success(msg, *args, **kwargs)
        else:
            self._logger.info(f"✅ {msg}", *args, **kwargs)


# Cache of loggers to avoid creating duplicates
_loggers = {}


def get_logger(name: str) -> AGILogger:
    """
    Get or create a named logger.
    
    Usage:
        from utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Module initialized")
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        AGILogger instance
    """
    if name not in _loggers:
        _loggers[name] = AGILogger(name)
    return _loggers[name]


# Pre-create a default logger for quick use
default_logger = get_logger("open_agi_brain")

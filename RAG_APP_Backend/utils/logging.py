"""
Logging configuration for the RAG application
"""

import logging
import sys
import os
from pathlib import Path


# Configure logging format and level
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = logging.INFO

# Ensure logs directory exists
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True, parents=True)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name
    
    Args:
        name: Name for the logger (usually __name__)
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    
    # Clear existing handlers to avoid duplicates when reloading modules
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(LOGS_DIR / "rag_app.log")
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file_handler)
    
    return logger


# Root logger configuration
def configure_root_logger():
    """Configure the root logger for the application"""
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)
    
    # Clear existing handlers
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(LOGS_DIR / "rag_app.log")
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger.addHandler(file_handler)


# Configure third-party loggers
def configure_third_party_loggers():
    """Reduce noise from third-party libraries"""
    # Set higher log levels for noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("faiss").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)


# Run initial configuration
configure_root_logger()
configure_third_party_loggers()
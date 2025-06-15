"""
Comprehensive logging configuration for ACE-LLM.
Provides both file logging (ace.log) and console logging based on verbosity.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import ClassVar, Optional


class ACEFormatter(logging.Formatter):
    """Custom formatter with color support for console and detailed formatting for file."""

    # ANSI color codes
    COLORS: ClassVar[dict[str, str]] = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def __init__(self, use_colors: bool = False, detailed: bool = False):
        self.use_colors = use_colors
        self.detailed = detailed

        if detailed:
            # Detailed format for file logging
            fmt = "[%(asctime)s] %(name)s.%(funcName)s:%(lineno)d [%(levelname)s] %(message)s"
        else:
            # Simpler format for console
            fmt = "[%(levelname)s] %(name)s: %(message)s"

        super().__init__(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    def format(self, record):
        formatted = super().format(record)

        if self.use_colors and record.levelname in self.COLORS:
            color = self.COLORS[record.levelname]
            reset = self.COLORS["RESET"]
            return f"{color}{formatted}{reset}"

        return formatted


def configure_external_loggers(external_log_level: str = "WARNING") -> None:
    """
    Configure external libraries' loggers to reduce noise.

    Args:
        external_log_level: Log level for external libraries (ERROR, WARNING, etc.)
    """
    numeric_level = getattr(logging, external_log_level.upper(), logging.WARNING)

    # Dictionary of noisy external libraries and their specific log levels
    external_loggers: dict[str, int] = {
        # HTTP and networking libraries
        "httpcore": numeric_level,
        "httpx": numeric_level,
        "urllib3": numeric_level,
        "requests": numeric_level,
        # API client libraries
        "openai": numeric_level,
        # Embedding and vector search libraries
        "faiss": numeric_level,
        # Other potential noisy libraries
        "numexpr": logging.ERROR,
        "asyncio": numeric_level,
        "charset_normalizer": logging.ERROR,
    }

    # Apply the configured log levels
    for logger_name, level in external_loggers.items():
        logging.getLogger(logger_name).setLevel(level)


def setup_logging(
    log_level: str = "WARNING",
    log_file: Optional[str] = None,
    console_output: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    filter_external: bool = True,
    external_log_level: str = "WARNING",
) -> None:
    """
    Set up comprehensive logging for ACE-LLM.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (defaults to ace.log in current directory)
        console_output: Whether to output logs to console
        max_file_size: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        filter_external: Whether to filter external library logs
        external_log_level: Log level for external libraries if filter_external is True
    """

    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.WARNING)

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels, handlers will filter

    # Clear existing handlers
    root_logger.handlers.clear()

    # Set up file logging
    if log_file is None:
        log_file = "ace.log"

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=max_file_size, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)  # Log everything to file
    file_handler.setFormatter(ACEFormatter(use_colors=False, detailed=True))
    root_logger.addHandler(file_handler)

    # Console handler (only if requested and level is appropriate)
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)

        # Use colors if terminal supports it
        use_colors = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        console_handler.setFormatter(ACEFormatter(use_colors=use_colors, detailed=False))
        root_logger.addHandler(console_handler)

    # Filter external library logs if requested
    if filter_external:
        configure_external_loggers(external_log_level)

    # Log startup message
    logger = logging.getLogger("ace.logging_config")
    logger.info("=" * 60)
    logger.info(f"ACE-LLM Logging Started - {datetime.now().isoformat()}")
    logger.info(f"Log Level: {log_level}")
    logger.info(f"Log File: {os.path.abspath(log_file)}")
    logger.info(f"Console Output: {console_output}")
    logger.info(f"External Library Filtering: {filter_external} (level: {external_log_level})")
    logger.info("=" * 60)


class ACELogger:
    """Enhanced logger with custom methods for structured data."""

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def __getattr__(self, name):
        # Delegate all standard logging methods to the underlying logger
        return getattr(self._logger, name)

    def log_structured(self, level: str, label: str, data: str, **kwargs):
        """Log structured data (JSON, code, etc.) with escaped newlines."""
        # Escape newlines and normalize whitespace
        clean_data = data.replace("\n", "\\n").replace("\r", "\\r")
        # Limit extremely long lines
        if len(clean_data) > 2000:
            clean_data = clean_data[:1997] + "..."

        message = f"{label}: {clean_data}"
        if kwargs:
            message += f" ({', '.join(f'{k}={v}' for k, v in kwargs.items())})"

        getattr(self._logger, level.lower())(message)

    def log_plan(self, script: str, tools: Optional[list] = None, **kwargs):
        """Log an abstract plan with script and tools."""
        self.log_structured("info", "ABSTRACT_PLAN_SCRIPT", script, **kwargs)
        if tools:
            for i, tool in enumerate(tools):
                tool_info = f"name={getattr(tool, 'name', 'unknown')} desc={getattr(tool, 'description', 'none')}"
                self.log_structured("debug", f"ABSTRACT_TOOL_{i + 1}", tool_info, **kwargs)

    def log_query(self, query: str, **kwargs):
        """Log a user query."""
        self.log_structured("info", "USER_QUERY", query, **kwargs)

    def log_tool_match(self, abstract_name: str, concrete_name: str, mapping_data: Optional[dict] = None, **kwargs):
        """Log tool matching results."""
        if mapping_data:
            input_mapping = mapping_data.get("input_mapping", "none")
            output_mapping = mapping_data.get("output_mapping", "none")
            self.log_structured(
                "debug",
                f"TOOL_MATCH_{abstract_name}",
                f"concrete={concrete_name} input_map={input_mapping} output_map={output_mapping}",
                **kwargs,
            )
        else:
            self.log_structured("info", f"TOOL_MATCH_{abstract_name}", f"concrete={concrete_name}", **kwargs)

    def log_execution(self, event: str, data: Optional[str] = None, **kwargs):
        """Log execution events."""
        if data:
            self.log_structured("debug", f"EXECUTION_{event.upper()}", data, **kwargs)
        else:
            self.info(f"Execution: {event}", **kwargs)

    def log_raw_llm_output(self, output_type: str, content: str, **kwargs):
        """Log raw LLM outputs."""
        self.log_structured("debug", f"LLM_OUTPUT_{output_type.upper()}", content, **kwargs)


def get_logger(name: str) -> ACELogger:
    """Get an enhanced logger instance for the given name."""
    return ACELogger(logging.getLogger(name))


def log_execution_time(logger: ACELogger, func_name: str, duration: float):
    """Log execution time for a function."""
    logger.info(f"{func_name} completed in {duration:.3f}s")

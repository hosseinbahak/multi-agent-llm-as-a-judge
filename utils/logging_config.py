# multi_agent_llm_judge/utils/logging_config.py
from loguru import logger
import sys
from pathlib import Path

def setup_logging(level="INFO", file_path=None, rotation="10 MB", retention="7 days"):
    """
    Sets up the application's logging using loguru.
    This is an alias for configure_logging for backwards compatibility.

    Args:
        level (str): The minimum logging level to display (e.g., "DEBUG", "INFO").
        file_path (str, optional): Path to the log file. If None, logs only to console.
        rotation (str): Log file rotation size.
        retention (str): How long to keep old log files.
    """
    configure_logging(level, file_path, rotation, retention)

def configure_logging(level="INFO", file_path=None, rotation="10 MB", retention="7 days"):
    """
    Configures the application's logging using loguru.

    Args:
        level (str): The minimum logging level to display (e.g., "DEBUG", "INFO").
        file_path (str, optional): Path to the log file. If None, logs only to console.
        rotation (str): Log file rotation size.
        retention (str): How long to keep old log files.
    """
    logger.remove()  # Remove default handler to avoid duplicate outputs

    # Console logger
    logger.add(
        sys.stderr,
        level=level.upper(),
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        colorize=True
    )

    # File logger (optional)
    if file_path:
        logger.add(
            file_path,
            level="DEBUG",  # Log everything to the file
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation=rotation,
            retention=retention,
            enqueue=True,  # Make logging async-safe
            backtrace=True,
            diagnose=True,
        )

    logger.info("Logging configured.")

    # 2. File (same content as console)
    log_file = Path("/home/zeus/Projects/hb/multi_agent_llm_judge/app.log")
    logger.add(
        log_file,
        level="INFO",        # same as console
        rotation="200 MB",    # rotate at 10 MB
        retention="7 days",  # keep logs for 7 days
        encoding="utf-8"
    )

    logger.success(f"Logging configured. Logs will also be saved to {log_file}")
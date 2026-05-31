import logging
import sys
from pathlib import Path
from app.config import get_settings


def setup_logger(name: str) -> logging.Logger:
    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — UTF-8 explicitly so Windows cp1252 doesn't choke
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    if hasattr(console_handler.stream, "reconfigure"):
        console_handler.stream.reconfigure(encoding="utf-8")
    logger.addHandler(console_handler)

    # File handler
    log_dir = Path(settings.logs_dir)
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


"""
utils/logger.py — Centralised coloured logging.
"""

from __future__ import annotations

import logging
import sys

try:
    import colorlog  # type: ignore

    _handler = colorlog.StreamHandler(sys.stdout)
    _handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)-8s%(reset)s %(cyan)s%(name)s%(reset)s — %(message)s",
            log_colors={
                "DEBUG": "white",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
    )
except ImportError:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("%(levelname)-8s %(name)s — %(message)s")
    )


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(_handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger

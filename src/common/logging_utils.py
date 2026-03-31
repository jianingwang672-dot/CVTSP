from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logger(output_dir: str | Path, level: str = "INFO"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stderr, level=level, enqueue=False, backtrace=False, diagnose=False)
    logger.add(
        output_path / "run.log",
        level=level,
        enqueue=False,
        backtrace=False,
        diagnose=False,
        rotation="10 MB",
    )
    return logger


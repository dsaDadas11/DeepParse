import logging
import os

try:
    import colorlog
except ImportError:  # pragma: no cover - local fallback
    colorlog = None


def get_logger():
    log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
    level = getattr(logging, log_level, logging.INFO)

    if colorlog is None:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s.%(msecs)03d - %(levelname)s - [%(funcName)s] - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(level)
        return logger

    handler = colorlog.StreamHandler()
    formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s.%(msecs)03d - %(levelname)s - [%(funcName)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        },
    )
    handler.setFormatter(formatter)

    logger = colorlog.getLogger(__name__)
    if not logger.handlers:
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
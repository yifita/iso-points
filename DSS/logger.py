""" From https://github.com/t177398/best_python_logger """
import logging
import sys

class _CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    grey = "\x1b[0;37m"
    green = "\x1b[1;32m"
    yellow = "\x1b[1;33m"
    red = "\x1b[1;31m"
    purple = "\x1b[1;35m"
    blue = "\x1b[1;34m"
    light_blue = "\x1b[1;36m"
    reset = "\x1b[0m"
    blink_red = "\x1b[5m\x1b[1;31m"
    format_prefix = f"{purple}%(asctime)s{reset} " \
                    f"{blue}%(name)s{reset} " \
                    f"{light_blue}(%(filename)s:%(lineno)d){reset} "

    format_suffix = "%(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: format_prefix + green + format_suffix + reset,
        logging.INFO: format_prefix + grey + format_suffix + reset,
        logging.WARNING: format_prefix + yellow + format_suffix + reset,
        logging.ERROR: format_prefix + red + format_suffix + reset,
        logging.CRITICAL: format_prefix + blink_red + format_suffix + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


# Just import this function into your programs
# "from logger import get_logger"
# "logger = get_logger(__name__)"
# Use the variable __name__ so the logger will print the file's name also

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(_CustomFormatter())
    logger.addHandler(ch)
    return logger
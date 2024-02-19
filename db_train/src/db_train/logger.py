import logging
import colorlog
from typing import Any

DEFAULT_NAME: str = 'toolchain_db'
DEFAULT_LEVEL: int = logging.DEBUG
DEFAULT_SAVE_FILENAME: str = None
DEFAULT_SAVE_MODE: str = 'a'
DEFAULT_QUIET: bool = False
DEFAULT_COLOR_CONFIG: dict[str, str] = {
  'DEBUG': 'white',
  'INFO': 'green',
  'WARNING': 'yellow',
  'ERROR': 'red',
  'CRITICAL': 'bold_red'
}

logger = None

def init_logger(
    name: str = DEFAULT_NAME,
    level: int = DEFAULT_LEVEL,
    save_filename: str = DEFAULT_SAVE_FILENAME,
    save_mode: str = DEFAULT_SAVE_MODE,
    quiet: bool = DEFAULT_QUIET,
    color_config: dict[str, str] = DEFAULT_COLOR_CONFIG,
) -> None:
  global logger
  console_handler = logging.StreamHandler()
  logger = logging.getLogger(name)
  logger.setLevel(level)
  if not quiet:
    console_handler.setLevel(level)
    console_formatter = colorlog.ColoredFormatter(
      fmt='%(log_color)s[%(asctime)s.%(msecs)03d] %(filename)s -> %(funcName)s line:%(lineno)d [%(levelname)s] : %(message)s',
      datefmt='%Y-%m-%d  %H:%M:%S',
      log_colors=color_config
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    console_handler.close()

  if save_filename is not None:
    file_handler = logging.FileHandler(filename=save_filename, mode=save_mode, encoding='utf-8')
    file_handler.setLevel(level)

    file_formatter = logging.Formatter(
      fmt='[%(asctime)s.%(msecs)03d] %(filename)s -> %(funcName)s line:%(lineno)d [%(levelname)s] : %(message)s',
      datefmt='%Y-%m-%d  %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    file_handler.close()

init_logger(name='toolchain_db')
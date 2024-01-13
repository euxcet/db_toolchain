import logging
import colorlog

DEFAULT_COLOR_CONFIG = {
  'DEBUG': 'white',
  'INFO': 'green',
  'WARNING': 'yellow',
  'ERROR': 'red',
  'CRITICAL': 'bold_red'
}

class Logger():
  def __init__(self, name:str, colors:dict=DEFAULT_COLOR_CONFIG, level:int=logging.DEBUG,
               save_filename:str=None, save_mode:str='a', quiet:bool=False):
    console_handler = logging.StreamHandler()
    self.logger = logging.getLogger(name)
    self.logger.setLevel(logging.DEBUG)
    if not quiet:
      console_handler.setLevel(logging.INFO)
      console_formatter = colorlog.ColoredFormatter(
        fmt='%(log_color)s[%(asctime)s.%(msecs)03d] %(filename)s -> %(funcName)s line:%(lineno)d [%(levelname)s] : %(message)s',
        datefmt='%Y-%m-%d  %H:%M:%S',
        log_colors=colors
      )
      console_handler.setFormatter(console_formatter)
      self.logger.addHandler(console_handler)
      console_handler.close()

    if save_filename is not None:
      file_handler = logging.FileHandler(filename=save_filename, mode=save_mode, encoding='utf-8')
      file_handler.setLevel(logging.DEBUG)

      file_formatter = logging.Formatter(
        fmt='[%(asctime)s.%(msecs)03d] %(filename)s -> %(funcName)s line:%(lineno)d [%(levelname)s] : %(message)s',
        datefmt='%Y-%m-%d  %H:%M:%S'
      )
      file_handler.setFormatter(file_formatter)
      self.logger.addHandler(file_handler)
      file_handler.close()

  def debug(self, msg, *args, **kwargs):
    self.logger.debug(msg, *args, **kwargs)

  def info(self, msg, *args, **kwargs):
    self.logger.info(msg, *args, **kwargs)

  def warning(self, msg, *args, **kwargs):
    self.logger.warning(msg, *args, **kwargs)

  def error(self, msg, *args, **kwargs):
    self.logger.error(msg, *args, **kwargs)

  def critical(self, msg, *args, **kwargs):
    self.logger.critical(msg, *args, **kwargs)

logger = Logger(name='toolchain_db')
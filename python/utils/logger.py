import logging
import colorlog

output_log_filename = 'log.txt'
logger = logging.getLogger('Ring')
log_colors_config = {
  'DEBUG': 'white',
  'INFO': 'green',
  'WARNING': 'yellow',
  'ERROR': 'red',
  'CRITICAL': 'bold_red',
}
# save_string(output_log_filename, '')
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(filename=output_log_filename, mode='a', encoding='utf-8')

logger.setLevel(logging.DEBUG)
console_handler.setLevel(logging.INFO)
file_handler.setLevel(logging.DEBUG)

console_formatter = colorlog.ColoredFormatter(
  fmt='%(log_color)s[%(asctime)s.%(msecs)03d] %(filename)s -> %(funcName)s line:%(lineno)d [%(levelname)s] : %(message)s',
  datefmt='%Y-%m-%d  %H:%M:%S',
  log_colors=log_colors_config
)
file_formatter = logging.Formatter(
  fmt='[%(asctime)s.%(msecs)03d] %(filename)s -> %(funcName)s line:%(lineno)d [%(levelname)s] : %(message)s',
  datefmt='%Y-%m-%d  %H:%M:%S'
)
console_handler.setFormatter(console_formatter)
file_handler.setFormatter(file_formatter)

if not logger.handlers:
  logger.addHandler(console_handler)
  logger.addHandler(file_handler)

console_handler.close()
file_handler.close()
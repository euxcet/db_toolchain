import os
import os.path as osp
import shutil
from file_dataset import FileDataset

def makedir(path):
  try:
    os.makedirs(path)
  except:
    pass

def is_valid_user(user:str) -> bool:
  return not user.startswith('.')

def is_valid_class(class_:str) -> bool:
  return not class_.startswith('.')

def is_valid_data_file(f:str) -> bool:
  return not f.startswith('.') and (f.endswith('.txt') or f.endswith('.bin'))

def export_data(src_root, dst_root, class_map=None):
  src_dataset = FileDataset(src_root)
  for user, class_, _, files in src_dataset.records:
    if class_map is None or class_ in class_map:
      src_path = osp.join(src_root, user, class_)
      dst_path = osp.join(dst_root, user, class_ if class_map is None else class_map[class_])
      makedir(dst_path)
      for f in files:
        shutil.copyfile(osp.join(src_path, f), osp.join(dst_path, f))

if __name__ == '__main__':
  class_map = {
    '3': '4',
    '11': '19',
    '6': '15',
    '8': '16',
    '9': '14',
    '17': '17',
    '16': '18',
    '12': '11',
    '0': '0',
    '1': '0',
    '2': '0',
    '4': '5',
    '13': '12',
    '10': '3',
    '18': '10',
  }
  export_data('ring_dataset', 'old_dataset', class_map=class_map)

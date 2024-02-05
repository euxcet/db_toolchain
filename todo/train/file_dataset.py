import os
import os.path as osp
from file_utils import load_json

class FileDataset():
  def __init__(self, root:str, has_label=True):
    self.check_dataset(root)
    self.config = load_json(self._config_path)
    if has_label:
      self.label_name = self.config['label'] # key is the directory name
      self.label_id = {dir_name: id for id, dir_name in enumerate(self.label_name)}
    self.records = self.load_from_dir(root)

  def check_dataset(self, root:str):
    self._config_path = osp.join(root, 'config.json')
    if not osp.exists(self._config_path):
      raise Exception(f'The config.json file does not exist in the root directory of the dataset[{root}].')

  def load_from_dir(self, root:str):
    result_dict = dict()
    users:list[str] = os.listdir(root)
    for user in users:
      if osp.isdir(osp.join(root, user)) and self.is_valid_user(user):
        for class_ in os.listdir(osp.join(root, user)):
          if osp.isdir(osp.join(root, user, class_)) and self.is_valid_class(class_):
            src_path = osp.join(root, user, class_)
            data_files = os.listdir(src_path)
            for f in data_files:
              if self.is_valid_data_file(f):
                number = f.split('_')[0]
                if not f.split('_')[1].isdigit():
                  id = (user, class_, number)
                else:
                  id = (user, class_, number, f.split('_')[1])
                if id in result_dict:
                  result_dict[id].append(f)
                else:
                  result_dict[id] = [f]
    return [(*x, result_dict[x]) for x in result_dict]

  def is_valid_user(self, user:str) -> bool:
    return not user.startswith('.')

  def is_valid_class(self, class_:str) -> bool:
    return not class_.startswith('.')

  def is_valid_data_file(self, f:str) -> bool:
    return not f.startswith('.') and f.endswith(('txt', 'json', 'bin', 'mp3', 'npy')) and len(f.split('_')) > 1
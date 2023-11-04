import os
import os.path as osp

class FileDataset():
  def __init__(self, root:str):
    self.records = self.load_from_dir(root)
    self.label_id = self.get_label_id()

  def get_label_id(self):
    # TODO
    label_id = {str(i): i for i in range(100)}
    return label_id
    label_id = dict()
    label_count = 0
    for _, class_, _, _ in self.records:
      if class_ not in label_id:
        label_id[class_] = label_count
        label_count += 1
    return label_id

  def load_from_dir(self, root:str):
    result_dict = dict()
    users:list[str] = os.listdir(root)
    for user in users:
      if self.is_valid_user(user):
        classes = os.listdir(osp.join(root, user))
        for class_ in classes:
          if self.is_valid_class(class_):
            src_path = osp.join(root, user, class_)
            data_files = os.listdir(src_path)
            for f in data_files:
              if self.is_valid_data_file(f):
                number = f.split('_')[0]
                id = (user, class_, number)
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
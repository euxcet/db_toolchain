import yaml

class Config():
    def __init__(self, config_filename='config.yaml'):
        self.__config = {}
        self.__config_file = config_filename
        self.load()

    def get(self, key):
        return self.__config.get(key)

    def set(self, key, value):
        self.__config[key] = value
        self.save()

    def set_if_not_exist(self, key, value):
        if key not in self.__config:
            self.__config[key] = value
            self.save()
    
    def set_temp(self, key, value):
        self.__config[key] = value

    def get_all(self):
        return self.__config

    def remove_key(self, key):
        if key in self.__config:
            del self.__config[key]
        self.save()

    def remove_all(self):
        self.__config.clear()
        self.save()
    
    def save(self):
        with open(self.__config_file, 'w') as configfile:
            yaml.dump(self.__config, configfile)

    def load(self):
        try:
            with open(self.__config_file, 'r') as configfile:
                self.__config = yaml.safe_load(configfile)
        except FileNotFoundError:
            self.__config = {}

cfg = Config()

def generate_new_config():
    cfg = Config("config.yaml")
    cfg.set_if_not_exist('raw_data_path', '../../../local/dataset/0_ring_gesture_dataset_raw')
    cfg.set_if_not_exist('dataset_path', '../../../local/dataset/0_ring_gesture_dataset')
    cfg.set_if_not_exist('all_people', ["cm", "cyf", "fhy", "gsq", "gx", "hfx", "hm" ,"hsq", "jc", "jjx", "lsq", "lxt", "lzj", "mfj", "qk", "qp", "wxy", "ylc", "zqy", "zsn"])

if __name__ == "__main__":
    # generate config
    generate_new_config()
    
    
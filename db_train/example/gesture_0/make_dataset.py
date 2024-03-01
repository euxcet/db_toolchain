import numpy as np
from numpy import float32
import scipy
import pandas as pd
import os
import random
from config_train import cfg

MOVEMENTS_STUDY1 = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16"]
alpha = 1.7  # dynamic threshold
beta = 0.8  # the ratio that judgment action
frame_rate = 10  # the frame rate of the data # unused
dt = 10  # the time interval of the standard data
liftFileNum = 4
pressFileNum = 4

def get_acc_head_and_tail(data, frame_head, frame_tail, alpha=alpha):
    """获取每个动作的加速度准确起始和结束"""
    pre_detrend_data = data[:, frame_head:frame_tail]
    detrend_data = [scipy.signal.detrend(pre_detrend_data[i]) for i in range(pre_detrend_data.shape[0])]
    acc_sum = np.linalg.norm(detrend_data[:3], axis=0)
    dynamic_threshold = acc_sum.mean() + alpha * acc_sum.std()

    start_frame_smoothed = next((i for i, val in enumerate(acc_sum) if val > dynamic_threshold), 0)
    end_frame_smoothed = next((i for i, val in enumerate(acc_sum[::-1]) if val > dynamic_threshold), 0)

    acc_head = frame_head + start_frame_smoothed
    acc_tail = frame_head + len(acc_sum) - 1 - end_frame_smoothed
    return acc_head, acc_tail

def get_mark_array(data, log_acc, movement_name, beta=beta, frame_rate=frame_rate):
    """生成标记数组"""
    mark = np.zeros(data.shape[1], dtype=float32)
    for start, end in log_acc:
        length = end - start
        begin = end - int(length * (1 - beta))
        # end += frame_rate
        end = start + 200 + int(length * (1 - beta))
        mark[begin:end] = int(movement_name) + 1 # 0 表示无动作，其余动作 label + 1
    return mark


# get the acc start and end
def blurred_to_acc(data, log):
    """从模糊日志中获取加速度起始和结束"""
    acc_head_and_tail = []
    for line in log:
        if type(line) == list:
            cur_movement = (int(line[0]), int(line[1]))
        elif type(line) == str:
            parts = line.split(",")
            cur_movement = (int(parts[1]), int(parts[2]))
        else:
            print("log type error")
            return
        middle = (cur_movement[0] + cur_movement[1]) // 2
        frame_head, frame_tail = middle - 100, middle + 100

        acc_head, acc_tail = get_acc_head_and_tail(data, frame_head, frame_tail) # 获取每个动作的加速度准确起始和结束
        # print("acc_head:", acc_head, "acc_tail:", acc_tail, "cur_movement[0]:", cur_movement[0], "cur_movement[1]:", cur_movement[1])
        acc_head = max(acc_head, cur_movement[0])
        acc_tail = min(acc_tail, cur_movement[1])
        acc_head_and_tail.append([acc_head, acc_tail])

    return np.array(acc_head_and_tail)


def get_standard_data(data, log_acc, mark, dt=dt, detrend=True, normalization=True, deleteZero=True):
    """获取标准数据"""
    standard_data = []
    for start, end in log_acc:
        middle = (start + end) // 2
        for i in range(0, 200, dt):
            head, tail = middle - 100 - i, middle + 100 - i
            if head >= 0 and tail <= data.shape[1]:
                if not deleteZero or mark[tail] != 0:
                    d = data[:, head:tail]
                    # detrend
                    if detrend:
                        d = [scipy.signal.detrend(d[i]) for i in range(d.shape[0])]
                    # normalization
                    if normalization:
                        d = d / np.linalg.norm(d, axis=1).reshape(-1, 1)
                    standard_data.append((d, mark[tail]))

    # 分离标准数据和标识符
    movement_data, identifiers = zip(*standard_data) if standard_data else ([], [])
    return np.array(movement_data), np.array(identifiers)

def process_data_study(ifAddPressAndLift=False, deleteZero=True, detrend=True, normalization=True):
    """处理实验数据"""
    for person in cfg.get('all_people'):
        standard_data_each_person, standard_ident_each_person = [], []
        for movement_name in MOVEMENTS_STUDY1:
            data_path = os.path.join(cfg.get('raw_data_path'), person, f"{person}_{movement_name}.csv")
            log_path = os.path.join(cfg.get('raw_data_path'), person, f"{person}_{movement_name}.log")

            data = pd.read_csv(data_path)[["accX.1", "accY.1", "accZ.1", "gyrX.1", "gyrY.1", "gyrZ.1"]].to_numpy(dtype=np.float32).T
            with open(log_path, "r") as f:
                log = f.read().splitlines()

            log_acc = blurred_to_acc(data, log) # 获取加速度起始和结束
            mark = get_mark_array(data, log_acc, movement_name) # 生成标记数组
            standard_data, standard_ident = get_standard_data(data, log_acc, mark, detrend=detrend, normalization=normalization, deleteZero=deleteZero) # 获取标准数据

            standard_data_each_person.append(standard_data)
            standard_ident_each_person.append(standard_ident)
        # 合并标准数据和标识符
        standard_data_each_person = np.concatenate(standard_data_each_person, axis=0)
        standard_ident_each_person = np.concatenate(standard_ident_each_person, axis=0)
        # 保存标准数据和标识符
        if not os.path.exists(os.path.join(cfg.get('dataset_path'), "standard_data")):
            os.makedirs(os.path.join(cfg.get('dataset_path'), "standard_data"))
        np.save(os.path.join(cfg.get('dataset_path'), "standard_data", f"{person}_standard_data.npy"), standard_data_each_person)
        np.save(os.path.join(cfg.get('dataset_path'), "standard_data", f"{person}_standard_ident.npy"), standard_ident_each_person)
        print("standard_data_each_person.shape:", standard_data_each_person.shape)
        print("standard_ident_each_person.shape:", standard_ident_each_person.shape)
        print(f"person:{person} has been processed!")

    # if ifAddPressAndLift:
    #     liftDataset, liftLogset, pressDataset, pressLogset = getPressAndListDateset()
    #     standardLiftData = []
    #     standardLiftIdent = []
    #     standardPressData = []
    #     standardPressIdent = []
    #     for i in range(min(len(liftLogset),liftFileNum)): # 取四个数据集实验
    #         data = np.array(liftDataset[i]).T
    #         log = liftLogset[i]
    #         log_acc = blurred_to_acc(data, log)
    #         mark = get_mark_array(data, log_acc, 17) # 17 表示抬起动作
    #         standard_data, standard_ident = get_standard_data(data, log_acc, mark)
    #         standardLiftData.append(standard_data)
    #         standardLiftIdent.append(standard_ident)
    #     standardLiftData = np.concatenate(standardLiftData, axis=0)
    #     standardLiftIdent = np.concatenate(standardLiftIdent, axis=0)
    #     np.save(os.path.join(cfg.get('dataset_path'), "standard_data", "lift_standard_data.npy"), standardLiftData)
    #     np.save(os.path.join(cfg.get('dataset_path'), "standard_data", "lift_standard_ident.npy"), standardLiftIdent)
    #     print("standardLiftData.shape:", standardLiftData.shape)
    #     print("standardLiftIdent.shape:", standardLiftIdent.shape)
    #     print("lift has been processed!")
    #     for i in range(min(len(pressLogset),pressFileNum)):
    #         data = np.array(pressDataset[i]).T
    #         log = pressLogset[i]
    #         log_acc = blurred_to_acc(data, log)
    #         mark = get_mark_array(data, log_acc, 18)    # 18 表示按下动作
    #         standard_data, standard_ident = get_standard_data(data, log_acc, mark)
    #         standardPressData.append(standard_data)
    #         standardPressIdent.append(standard_ident)
    #     standardPressData = np.concatenate(standardPressData, axis=0)
    #     standardPressIdent = np.concatenate(standardPressIdent, axis=0)
    #     np.save(os.path.join(cfg.get('dataset_path'), "standard_data", "press_standard_data.npy"), standardPressData)
    #     np.save(os.path.join(cfg.get('dataset_path'), "standard_data", "press_standard_ident.npy"), standardPressIdent)
    #     print("standardPressData.shape:", standardPressData.shape)
    #     print("standardPressIdent.shape:", standardPressIdent.shape)
    #     print("press has been processed!")


def make_dataset(leave_one_out=True, test_ratio=0.2):
    if leave_one_out:
        all_people = cfg.get('all_people')
        test_num = int(len(all_people) * test_ratio)
        random.shuffle(all_people)
        test_people = all_people[:test_num]
        train_people = all_people[test_num:]
        cfg.set('test_people', test_people)
        cfg.set('train_people', train_people)
        
        # load data
        train_data_filenames = [f"{person}_standard_data.npy" for person in train_people]
        test_data_filenames = [f"{person}_standard_data.npy" for person in test_people]
        train_ident_filenames = [f"{person}_standard_ident.npy" for person in train_people]
        test_ident_filenames = [f"{person}_standard_ident.npy" for person in test_people]
        train_x = [np.load(os.path.join(cfg.get('dataset_path'), 'standard_data', filename)) for filename in train_data_filenames]
        test_x = [np.load(os.path.join(cfg.get('dataset_path'), 'standard_data', filename)) for filename in test_data_filenames]
        train_y = [np.load(os.path.join(cfg.get('dataset_path'), 'standard_data', filename)) for filename in train_ident_filenames]
        test_y = [np.load(os.path.join(cfg.get('dataset_path'), 'standard_data', filename)) for filename in test_ident_filenames]

        # concat data
        train_x = np.concatenate(train_x, axis=0)
        train_y = np.concatenate(train_y, axis=0)
        test_x = np.concatenate(test_x, axis=0)
        test_y = np.concatenate(test_y, axis=0)

        # transpose to b l c
        train_x = train_x.transpose(0, 2, 1)
        test_x = test_x.transpose(0, 2, 1)

        # save
        np.save(os.path.join(cfg.get('dataset_path'), 'train_x.npy'), train_x)
        np.save(os.path.join(cfg.get('dataset_path'), 'train_y.npy'), train_y)
        np.save(os.path.join(cfg.get('dataset_path'), 'test_x.npy'), test_x)
        np.save(os.path.join(cfg.get('dataset_path'), 'test_y.npy'), test_y)

    else:
        all_people = cfg.get('all_people')
        
        # load data
        data_filenames = [f"{person}_standard_data.npy" for person in all_people]
        ident_filenames = [f"{person}_standard_ident.npy" for person in all_people]
        x = [np.load(os.path.join(cfg.get('dataset_path'), 'standard_data', filename)) for filename in data_filenames]
        y = [np.load(os.path.join(cfg.get('dataset_path'), 'standard_data', filename)) for filename in ident_filenames]

        # concat data
        x = np.concatenate(x, axis=0)
        y = np.concatenate(y, axis=0)

        # shuffle data
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        x = x[idx]
        y = y[idx]

        # split data
        test_num = int(len(x) * test_ratio)
        test_x = x[:test_num]
        test_y = y[:test_num]
        train_x = x[test_num:]
        train_y = y[test_num:]

        # transpose to b l c
        train_x = train_x.transpose(0, 2, 1)
        test_x = test_x.transpose(0, 2, 1)

        # save
        np.save(os.path.join(cfg.get('dataset_path'), 'train_x.npy'), train_x)
        np.save(os.path.join(cfg.get('dataset_path'), 'train_y.npy'), train_y)
        np.save(os.path.join(cfg.get('dataset_path'), 'test_x.npy'), test_x)
        np.save(os.path.join(cfg.get('dataset_path'), 'test_y.npy'), test_y)


if __name__ == "__main__":
    process_data_study(ifAddPressAndLift=False, deleteZero=True, detrend=True, normalization=True)
    make_dataset(leave_one_out=True, test_ratio=0.2)
import os.path as osp
import numpy as np
import struct
import argparse
import os
from matplotlib import pyplot as plt

def load_glove_data_(filename):
    glove_data = []

    with open(filename, "rb") as f:
        while True:
            data = f.read(320)
            if len(data) == 0:
                break
            index = struct.unpack("i", data[0:4])[0]
            acc_x = struct.unpack("f", data[100:104])[0] * -9.8
            acc_y = struct.unpack("f", data[104:108])[0] * -9.8
            acc_z = struct.unpack("f", data[108:112])[0] * -9.8

            gyr_x = struct.unpack("f", data[88:92])[0]
            gyr_y = struct.unpack("f", data[92:96])[0]
            gyr_z = struct.unpack("f", data[96:100])[0]
            timestamp = struct.unpack("d", data[312:320])[0]
            glove_data.append(([acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z], timestamp))

    return glove_data


def load_ring_data(filename):
    ring_data = []

    with open(filename, "rb") as f:
        while True:
            data = f.read(36)
            if len(data) == 0:
                break
            index = struct.unpack("i", data[0:4])[0]
            acc_x = struct.unpack("f", data[4:8])[0]
            acc_y = struct.unpack("f", data[8:12])[0]
            acc_z = struct.unpack("f", data[12:16])[0]
            gyr_x = struct.unpack("f", data[16:20])[0]
            gyr_y = struct.unpack("f", data[20:24])[0]
            gyr_z = struct.unpack("f", data[24:28])[0]
            timestamp = struct.unpack("d", data[28:36])[0]
            ring_data.append(([acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z], timestamp))

    return ring_data

def load_timestamp_data(filename):
    timestamp_data = []

    with open(filename, "r") as f:
        while True:
            data = f.readline()
            if len(data) == 0:
                break
            data = data.split(" ")
            timestamp_data.append([float(data[0]), float(data[1])])

    return timestamp_data

def plot(data):
    fig = plt.figure()
    acc_axes = fig.add_subplot(2, 1, 1)
    gyr_axes = fig.add_subplot(2, 1, 2)
    t = np.arange(200)
    for i in range(3):
        acc_axes.plot(t, data[:, i])
    for i in range(3, 6):
        gyr_axes.plot(t, data[:, i])
    plt.show()

def load_glove_data(user, action, glove_filename, timestamp_filename, plot_data=False):
    raw_ring_data = load_glove_data_(glove_filename)
    raw_timestamp_data = load_timestamp_data(timestamp_filename)

    ring_data = []
    ring_pointer = 0

    for timestamp in raw_timestamp_data:
        start_timestamp, end_timestamp = timestamp[0], timestamp[1]
        while ring_pointer < len(raw_ring_data) and raw_ring_data[ring_pointer][1] < start_timestamp:
            ring_pointer += 1
        ring_data_single_action = []
        while ring_pointer < len(raw_ring_data) and raw_ring_data[ring_pointer][1] < end_timestamp:
            ring_data_single_action.append(raw_ring_data[ring_pointer][0])
            ring_pointer += 1
        ring_data.append(ring_data_single_action)
    
    # cut frames to 200 * 2
    empty_data_to_pop = []
    for i in range(len(ring_data)):
        if len(ring_data[i]) > 200:
            ring_data[i] = ring_data[i][:200]
        elif len(ring_data[i]) > 180:
            ring_data[i] += [ring_data[i][-1]] * (200 - len(ring_data[i])) # repeat the last frame, shallow copy
        else:
            empty_data_to_pop.append(i)
    for i in empty_data_to_pop[::-1]:
        # pop error data
        ring_data.pop(i)


    ring_data = [np.array(ring_data[i]) for i in range(len(ring_data))]

    if plot_data:
        print(user, action)
        for data in ring_data:
            plot(data)
    return ring_data


def load_data(user, action, ring_filename, timestamp_filename, plot_data=False):
    '''
    person: str
    action_id: str

    Return:
    ring_data: list of np.ndarray, shape=(200, 6)
    glove_data: list of np.ndarray, shape=(200, 16, 4)
    '''
    raw_ring_data = load_ring_data(ring_filename)
    raw_timestamp_data = load_timestamp_data(timestamp_filename)

    ring_data = []
    ring_pointer = 0

    for timestamp in raw_timestamp_data:
        start_timestamp, end_timestamp = timestamp[0], timestamp[1]
        while ring_pointer < len(raw_ring_data) and raw_ring_data[ring_pointer][1] < start_timestamp:
            ring_pointer += 1
        ring_data_single_action = []
        while ring_pointer < len(raw_ring_data) and raw_ring_data[ring_pointer][1] < end_timestamp:
            ring_data_single_action.append(raw_ring_data[ring_pointer][0])
            ring_pointer += 1
        ring_data.append(ring_data_single_action)
    
    # cut frames to 200 * 2
    empty_data_to_pop = []
    for i in range(len(ring_data)):
        if len(ring_data[i]) > 200:
            ring_data[i] = ring_data[i][:200]
        elif len(ring_data[i]) > 180:
            ring_data[i] += [ring_data[i][-1]] * (200 - len(ring_data[i])) # repeat the last frame, shallow copy
        else:
            empty_data_to_pop.append(i)
    for i in empty_data_to_pop[::-1]:
        # pop error data
        ring_data.pop(i)


    ring_data = [np.array(ring_data[i]) for i in range(len(ring_data))]

    # np.set_printoptions(suppress=True)
    if plot_data:
        print(user, action)
        for data in ring_data:
            plot(data)

    # if int(action) >= 20:
    #     filtered_ring_data = []
    #     for data in ring_data:
    #         norm = np.linalg.norm(data[:, :3], axis=1)
    #         pos = np.argmax(norm)
    #         if pos > 80 and pos < 160 and norm[pos] > 15:
    #             filtered_ring_data.append(data)
    #     if len(filtered_ring_data) == 0:
    #         print(user, action)
    #         for data in ring_data:
    #             plot(data)
    #     ring_data = filtered_ring_data
    return ring_data

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--user', type=str, required=True)
  parser.add_argument('--action', type=str, required=True)
  args = parser.parse_args()
  
  DATA_DIR = '1014_dataset'

  for i in range(10):
      ring_filename = os.path.join(DATA_DIR, args.user, args.action, str(i) + "_ring.bin")
      timestamp_filename = os.path.join(DATA_DIR, args.user, args.action, str(i) + "_timestamp.txt")
      if os.path.exists(ring_filename):
          ring_data_single_action = load_data(args.user, args.action, ring_filename, timestamp_filename, plot_data=True)

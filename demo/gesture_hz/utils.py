from threading import Timer
from scipy import signal
from ahrs.filters import Madgwick
from collections import deque
import pyquaternion as pyq
import numpy as np


class RepeatTimer(Timer):
    def run(self):
        while not self.finished.is_set():
            self.function(*self.args, **self.kwargs)
            self.finished.wait(self.interval)


class ButterBandpassRealTimeFilter:
    def __init__(self, fs, order=5):
        nyquist = 0.5 * fs
        # low = lowcut / nyquist
        # high = highcut / nyquist
        self.b, self.a = signal.butter(order, [10, 99], btype="bandpass", fs=fs)
        # self.zi = signal.lfilter_zi(self.b, self.a)
        self.min_value = 4e-4

    def filter(self, data):
        y = signal.lfilter(self.b, self.a, data)
        if abs(y) < self.min_value:
            y = 0
        return y


class IntegralValue:
    def __init__(self, value=0):
        self.value = value
        self.last_input_queue = deque(maxlen=10)
        for _ in range(10):
            self.last_input_queue.append(0)
        self.cur_mean = 0

    def update(self, new_input, dt):
        self.cur_mean = (self.cur_mean * 10 - self.last_input_queue[0] + new_input) / 10
        self.last_input_queue.append(new_input)
        self.value += self.cur_mean * dt
        return self.value
    
    def reset_value(self):
        self.value = 0


class MadgwickHelper:
    def __init__(self, q0: np.ndarray=np.array([1, 0, 0, 0]), Dt=1/200) -> None:
        self.madgwick_filter = Madgwick(Dt=Dt)
        self.Dt= Dt
        self.orientation = q0
        self.data = None
        self.data_without_gravity = None
        self.data_without_gravity_world = None
        self.vx, self.vy, self.vz = IntegralValue(), IntegralValue(), IntegralValue()
        self.zero_velocity_updater = self.ZeroVelocityUpdater()
    
    class ZeroVelocityUpdater():
        SAMPLES_THRESHOLD = 10
        ACC_THRESHOLD = 1
        ANGU_THRESHOLD = 0.15
        
        def __init__(self) -> None:
            self.zero_sample_cnt = 0

        def update(self, data) -> bool:
            '''
            data: [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]
            '''

            if np.all(np.abs(data[:3]) <= self.ACC_THRESHOLD) and np.all(np.abs(data[3:]) <= self.ANGU_THRESHOLD):
                self.zero_sample_cnt += 1
            else:
                self.zero_sample_cnt = 0
            return self.zero_sample_cnt >= self.SAMPLES_THRESHOLD

    def update(self, data: np.ndarray) -> np.ndarray:
        '''
        data: [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z] 
            unit: [m/s^2, m/s^2, m/s^2, rad/s, rad/s, rad/s]
        '''
        self.orientation = self.madgwick_filter.updateIMU(
            self.orientation,
            gyr=[data[3], data[4], data[5]],
            acc=[data[0], data[1], data[2]],
        )
        self.data = data
        self.data_without_gravity = self.cal_data_without_gravity()
        self.data_without_gravity_world = self.cal_data_without_gravity_world() # with acc in world coordinate

        if not self.zero_velocity_updater.update(self.data_without_gravity_world):
            self.vx.update(self.data_without_gravity_world[0], self.Dt)
            self.vy.update(self.data_without_gravity_world[1], self.Dt)
            self.vz.update(self.data_without_gravity_world[2], self.Dt)
        else:
            self.vx.reset_value()
            self.vy.reset_value()
            self.vz.reset_value()

        return self.orientation
    
    def get_orientation(self) -> np.ndarray:
        return self.orientation
    
    def get_orientation_as_pyq(self) -> pyq.Quaternion:
        return pyq.Quaternion(self.orientation[0], self.orientation[1], self.orientation[2], self.orientation[3])
    
    def get_orientation_as_euler(self) -> np.ndarray:
        return np.array(self.get_orientation_as_pyq().yaw_pitch_roll)
    
    def get_orientation_as_coords(self) -> np.ndarray:
        return np.array(self.get_orientation_as_pyq().rotate([1, 0, 0]))
    
    def cal_data_without_gravity(self) -> np.ndarray:
        GRAVITY = 9.76
        g = []
        g.append(2 * (self.orientation[1] * self.orientation[3] - self.orientation[0] * self.orientation[2]) * GRAVITY)
        g.append(2 * (self.orientation[2] * self.orientation[3] + self.orientation[0] * self.orientation[1]) * GRAVITY)
        g.append((self.orientation[0] ** 2 - self.orientation[1] ** 2 - self.orientation[2] ** 2 + self.orientation[3] ** 2) * GRAVITY)
        for _ in range(3):
            g.append(0)
        return self.data - g
    
    def cal_data_without_gravity_world(self) -> np.ndarray:
        # rotate data without gravity to world coordinate
        world_acc = self.get_orientation_as_pyq().rotate(self.data_without_gravity[:3])
        world_data = np.concatenate([world_acc, self.data_without_gravity[3:]])
        return world_data
    
    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.vx.value, self.vy.value, self.vz.value])
    
    
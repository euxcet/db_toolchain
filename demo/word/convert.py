import numpy as np
t = 0
with open('log.txt' ,'r') as f:
  for line in f.readlines():
    if line.startswith('imu'):
      d = list(map(lambda x: x + 'f,', line.strip()[5:-1].split(',')))

      print('RingImuData(listOf(', d[0] ,d[1], d[2], d[3], d[4], d[5] ,'), 0L),')
      t += 1
      if t == 500:
          break

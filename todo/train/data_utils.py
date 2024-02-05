from window import Window

def slice_data(data:list, timestamp:list, fps:int=200, lowest_fps:int=180, highest_fps:int=220) -> list:
  # Elements in the data array should have a timestamp variable and the to_numpy_float function.
  # Example is sensor.basic_data.IMUData
  sliced_data:list[Window] = [Window(fps) for _ in range(len(timestamp))]
  pointer = 0
  for interval, sample in zip(timestamp, sliced_data):
    while pointer < len(data) and data[pointer].timestamp <= interval[1]:
      if data[pointer].timestamp >= interval[0]:
        sample.push(data[pointer])
      pointer += 1

  sliced_data = list(filter(lambda x:x.capacity() >= lowest_fps and x.push_count <= highest_fps, sliced_data))
  scale = 4
  sliced_data = [d.pad().to_numpy_float().reshape(-1, scale, 6).mean(axis=1) for d in sliced_data]
  return sliced_data

def index_sequence(data:list, seq:list) -> int|None:
  for i in range(len(data) - len(seq) + 1):
    valid = True
    for j in range(len(seq)):
      if data[i + j] != seq[j]:
        valid = False
        break
    if valid:
      return i
  return None
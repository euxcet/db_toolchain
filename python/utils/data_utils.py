from utils.window import Window

def slice_data(data, timestamp, fps=200, lowest_fps=180, highest_fps=220):
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
  sliced_data = [d.pad().to_numpy_float() for d in sliced_data]
  return sliced_data
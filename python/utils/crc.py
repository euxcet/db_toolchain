def crc16(data, offset=3):
  genpoly = 0xA001
  result = 0xFFFF
  for i in range(offset, len(data)):
    result = (result & 0xFFFF) ^ (data[i] & 0xFF)
    for _ in range(8):
      if (result & 0x0001) == 1:
        result = (result >> 1) ^ genpoly
      else:
        result = result >> 1
  return result & 0xFFFF
from typing import Any
import struct

def to_bytearray(
    data: Any,
    order: str = '>',
    int_type: str = 'i',
    float_type: str = 'f',
) -> bytearray:
  if type(data) is int:
    return struct.pack(order + int_type, data)
  elif type(data) is float:
    return struct.pack(order + float_type, data)
  elif type(data) is str:
    return bytearray(data, encoding='utf-8')
  elif type(data) is list:
    result = bytearray()
    for d in data:
      result += to_bytearray(d)
    return result
  else:
    if 'to_bytearray' in dir(data):
      return data.to_bytearray()
  raise TypeError
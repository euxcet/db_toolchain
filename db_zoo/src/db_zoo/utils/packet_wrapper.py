from enum import Enum

class PacketType(Enum):
    HEARTBEAT = 0
    EYE = 1
    OBJECTS = 2

def wrap(type: PacketType, data: bytes = bytes()):
    return bytes([0x10, 0x16]) + type.value.to_bytes() + data

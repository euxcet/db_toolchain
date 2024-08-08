import socket
import signal
import sys

from typing import Any
import time


def signal_handler(sig: int, frame: Any) -> None:
    print("Ctrl+C pressed. Exiting...")
    udp_socket.close()
    sys.exit(0)


# 设置Ctrl+C信号处理
signal.signal(signal.SIGINT, signal_handler)

# 无人机单网模式下的IP地址
tello_ip = "192.168.10.1"
tello_port = 8889

# 创建UDP套接字
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 本地IP地址和端口（可以是任何可用的地址和端口）

# 绑定本地套接字
# udp_socket.bind((local_ip, local_port))
udp_socket.bind(("0.0.0.0", 9000))


def execute_command(command: str, noreport: bool = False) -> str:
    print("execute command", command)
    udp_socket.sendto(command.encode(), (tello_ip, tello_port))
    if noreport:
        return ""
    print("wait response")
    response, _ = udp_socket.recvfrom(1024)
    print("Tello's response:", response)
    return response.decode()


try:
    execute_command("command") # 会输出ok
    # 在无人机上执行 ap 命令以便设置多网模式需要接入的WiFi
    # 第一个参数是WiFi名称，第二个参数是WiFi密码，注意替换，不要直接执行
    # 请务必确认密码输入正确
    execute_command('ap 289-C3 qiyuan@1016') # 会提示自动重启，但是其实他并不会自动重启，需要手动重启，重启前不要忘了修改为组网模式

except Exception as e:
    print("An error occurred:", str(e))

finally:
    # 关闭UDP套接字
    udp_socket.close()

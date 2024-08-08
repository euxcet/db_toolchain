import asyncio
from typing import Tuple


async def main():
    for i in range(100, 120):
        print(i)
        for j in range(0, 256):
            ip = f"192.168.{i}.{j}"
            (
                transport,
                protocol,
            ) = await asyncio.get_event_loop().create_datagram_endpoint(
                lambda: ScanTtClientProtocol(),
                remote_addr=(ip, 8889),
            )
            await asyncio.sleep(0.001)


class ScanTtClientProtocol(asyncio.DatagramProtocol):
    def __init__(self):
        self.transport = None

    def connection_made(self, transport: asyncio.DatagramTransport):
        self.transport = transport
        message = b"command"
        remote_ip, _ = self.transport.get_extra_info("peername")
        self.transport.sendto(message)
        # print(f"Sent: {message.decode()} to {remote_ip}")

    def datagram_received(self, data: bytes, addr: Tuple[str, int]):
        response = data.decode()
        self.transport.close()
        print(f"Received: {response}", addr)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())

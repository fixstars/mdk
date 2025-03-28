import socket
import time


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def find_free_port(init_port: int, num_output=1):
    ret = []
    for port in range(init_port, 65536):
        if not is_port_in_use(port):
            ret.append(port)
            if len(ret) == num_output:
                break
    if len(ret) < num_output:
        raise RuntimeError("port search failed.")
    return ret


def wait_server(port: int, proc, retry_interval=1):
    # wait until http://localhost:{port} works
    while True:
        if is_port_in_use(port):
            break
        else:
            time.sleep(retry_interval)
        if proc.poll() is not None:
            raise RuntimeError()


class DummyPopen:
    """A dummy class with the same interface as popen. Prevents NameError in the finally clause."""

    def kill(self):
        pass

    def terminate(self):
        pass

    def communicate(self):
        pass

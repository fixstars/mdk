import subprocess

from termcolor import cprint

TERMINAL_COLOR = "green"


def run(cmd: list[str], accept_return_code=[0], **kwargs):
    cprint(" ".join(cmd), TERMINAL_COLOR)
    status = subprocess.run(cmd, **kwargs)
    if status.returncode not in accept_return_code:
        raise RuntimeError(status)
    return status

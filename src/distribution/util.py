import hashlib
import os
import signal
import socket
import time
import string
import random
import sys
import netifaces


def get_host():
    machine = socket.gethostname()
    interfaces = netifaces.interfaces()
    host = None
    for interface in interfaces:
        ip = netifaces.ifaddresses(interface)
        if netifaces.AF_INET in ip:
            if (ip[netifaces.AF_INET][0]['addr'][:3] in ['10.',
                                                         '172'] and 'lo' not in interface and 'docker' not in interface):
                host = ip[netifaces.AF_INET][0]['addr']
                break
    if host is not None:
        return True, machine, host
    else:
        return False, None, None


def md5_encode(s: str) -> str:
    hl = hashlib.md5()
    hl.update(s.encode(encoding='utf-8'))
    return str(hl.hexdigest())

def random_str(length: int=32) -> str:
    characters = string.ascii_letters + string.digits + string.punctuation
    random_string = ''.join(random.choices(characters, k=length))
    return random_string


def exit_simulator():
    time.sleep(2)
    print('Exit')
    parent_pid = os.getppid()
    os.kill(parent_pid, signal.SIGTERM)

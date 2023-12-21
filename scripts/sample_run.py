import sys
import time
from binaries import Binaries
from machine import Machine
from paramiko import SSHClient
import logging

# Dummy example program to execute a hello world remotely and gather some results


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise Exception("usage: python3 ./scripts/sample_run.py jdoe@remotename")
    
    logging.basicConfig(level=logging.DEBUG)
    ssh = SSHClient()
    ssh.load_system_host_keys()
    m = Machine(ssh, sys.argv[1])
    m.assesMachine()
    b = Binaries("./dist")
    m.copyBinary(b, False, True)
    m.startServer()
    time.sleep(5)
    data = m.runClientOneShot("run orca-mini hello")
    sys.stdout.buffer.write(data)
    sys.stdout.flush()
    m.stopServer()
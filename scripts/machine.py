import logging
import os
import select
import subprocess
import threading
from paramiko import SSHClient, SFTPClient, SSHConfig
import time
import sys

# Note: this doesn't quite work yet on windows due to DLLs not being in the path for the server and client binaries

class Machine:
    def __init__(self, sshClient, hostname):
        self.log = logging.getLogger(__name__)
        self.remoteDir = "." # TODO configurable and OS aware?
        self.remoteCoverDir = "." # TODO probably someplace better
        self.coverDir = "./coverage/" # TODO make configurable maybe
        self.ssh = sshClient   
        self.hostname = hostname
        self.running = False
        self.remotePort = 11434 # TODO random
        self.localPort = 1424 # TODO random

        self.ssh.load_system_host_keys()
        if self.hostname.find("@") >= 0:
            user, hostname = hostname.split("@")
            self.ssh.connect(hostname, username=user)
        else:
            self.ssh.connect(hostname)
        self.transport = self.ssh.get_transport()
        self.channel = None

    # Figure out what type of system it is
    # windows vs. linux vs. mac
    # what type of self.gpu 
    def assesMachine(self):
        self.os=""
        self.arch=""
        self.gpu=""
        # Try to determine what kind of remote system we're working with
        try:
            # Try for linux/mac first
            inp, outp, errp = self.ssh.exec_command('uname -a')
            if outp.channel.recv_exit_status() != 0:
                raise Exception("uname failed")
            inp.close()
            data = outp.read()
            errp.close()
            uname = str(data).lower()
            if uname.find("darwin") >= 0:
                self.os="darwin"
            elif uname.find("linux") >= 0:
                self.os="linux"
                inp, outp, errp = self.ssh.exec_command('lspci | grep VGA')
                inp.close()
                data = outp.read()
                errp.close()
                gpus = str(data).lower()
                if gpus.find("nvidia") >= 0:
                    self.gpu="cuda"
                elif gpus.find("amd") >= 0 or gpus.find("advanced micro devices") >= 0:
                    self.gpu="rocm"
            else:
                raise("Unrecognized unix-like self.os " + uname)
            if uname.find("arm64") >= 0:
                self.arch="arm64"
            elif uname.find("x86_64") >= 0:
                self.arch="amd64"
            else:
                raise("Unrecognized unix-like self.arch " + uname)
            if self.os == "darwin" and self.arch == "arm64":
                self.gpu="metal"
                
        except:
            # fallback to windows detection
            inp, outp, errp = self.ssh.exec_command('Get-WmiObject Win32_ComputerSystem | select SystemType')
            inp.close()
            data = outp.read()
            errp.close()
            sysType = str(data).lower()
            self.os="windows"
            BINARY_EXE=".exe"
            if sysType.find("x64") >= 0:
                self.arch="amd64"
            elif sysType.find("arm") >= 0:
                self.arch="arm64"
            else:
                raise("Unrecognized windows self.arch " + sysType)
            inp, outp, errp = self.ssh.exec_command('Get-WmiObject Win32_VideoController | select Name')
            inp.close()
            data = outp.read()
            errp.close()
            gpus = str(data).lower()
            if gpus.find("nvidia") >= 0:
                self.gpu="cuda"
            elif gpus.find("amd") >= 0 or gpus.find("advanced micro devices") >= 0:
                self.gpu="rocm"
        self.log.info("Detected remote system as %s %s %s", self.os, self.arch, self.gpu)


    # Based on assessed machine type, copy over the applicable binary
    def copyBinary(self, binaries, cpu, cov):
        filename, checksum = binaries.getBinary(self.os, self.arch, cpu, cov)
        self.remoteBinary = os.path.join(self.remoteDir, filename)
        self.log.info("checking remote system for matching checksum")
        if self.os == "windows":
            inp, outp, errp = self.ssh.exec_command('(get-filehash -algorithm md5 ' + self.remoteBinary + ').hash')
            inp.close()
            data = outp.read()
            errp.close()
            hash = data.decode('ascii').lower().strip()
        else:
            inp, outp, errp = self.ssh.exec_command('md5sum ' + self.remoteBinary + ' | cut -f1 -d\ ')
            inp.close()
            data = outp.read()
            errp.close()
            hash = data.decode('ascii').lower().strip()
        if checksum != hash:
            self.log.info("{0} does not contain the correct binary - uploading {1}".format(self.hostname, self.remoteBinary))
            binaryPath = os.path.join(binaries.distDir, filename)
            with SFTPClient.from_transport(self.transport) as sftp:
                self.log.info("copying {0} to {1}".format(self.remoteBinary, self.hostname))
                sftp.put(binaryPath, self.remoteBinary)
                sftp.chmod(self.remoteBinary, 0o755)
        else:
            self.log.info("{0} already up to date on {1}".format(self.remoteBinary, self.hostname))

    def startServer(self):
        self.serverSession = self.transport.open_session()
        self.thread = threading.Thread(target=self.runner)
        self.running = True
        self.thread.start()

    def runner(self):
        self.serverSession.setblocking(0)
        self.serverSession.get_pty()
        self.serverSession.invoke_shell()
        if self.os == "windows":
            self.serverSession.send("$env:GOCOVERDIR=" + self.remoteCoverDir + "\n")
        else:
            self.serverSession.send("export GOCOVERDIR=" + self.remoteCoverDir + "\n")
            # TODO randomize port number to avoid potential conflicts
        self.serverSession.send(self.remoteBinary + " serve\n")
        print("\n")
        while self.running:
            if self.serverSession.recv_ready():
                data = self.serverSession.recv(512)
                sys.stdout.buffer.write(data)
                sys.stdout.flush()
            time.sleep(0.001)
        self.log.info("closing server session to " + self.hostname)

    # TODO - add interactive option too
    def runClientOneShot(self, cmd_args):
        cmd = self.remoteBinary + " " + cmd_args
        self.log.info("Starting client")
        inp, outp, errp = self.ssh.exec_command(self.remoteBinary + ' ' + cmd_args)
        inp.close() # TODO consider interactive support for more complex test scenarios
        if outp.channel.recv_exit_status() != 0:
            err = errp.read()
            self.log.error(err.decode('ascii'))
            raise Exception("{0} {1} on {2} failed".format(self.remoteBinary, cmd_args, self.hostname))
        data = outp.read()
        errp.close()
        outp.close()
        return data

    def stopServer(self):
        self.log.info("signaling shutdown of remote server")
        self.running = False
        self.serverSession.send('\x03') # ^C
        time.sleep(3) # TODO more deterministic way to make sure coverage data is written out...
        if self.channel:
            self.log.info("closing channel")
            self.channel.close()
        self.log.info("joining")
        self.thread.join()
        with SFTPClient.from_transport(self.transport) as sftp:
            self.log.info("copying coverage from {0}".format(self.hostname))
            for filename in sftp.listdir(self.remoteCoverDir):
                if not filename.startswith("cov"):
                    continue
                sftp.get(os.path.join(self.remoteCoverDir, filename), os.path.join(self.coverDir, filename))

        self.log.info("done")
        # TODO copy the coverage data (if present) from the remote system locally to some well known location so we can aggregate it


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    ssh = SSHClient()
    ssh.load_system_host_keys()
    m = Machine(ssh, "daniel@desktop-puni632")
    m.assesMachine()
    

# b = Binaries("./dist")
# logging.info("Binaries" + str(b.binaries))
# sys.exit(0)
# m.copyBinary()
# m.startServer()
# time.sleep(2)
# m.runClient("run orca-mini hello")
# time.sleep(30)
# m.stopServer()
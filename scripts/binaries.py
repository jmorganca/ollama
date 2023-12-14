import hashlib
import logging
import os

BUF_SIZE = 65536

class Binaries:
    def __init__(self, distDir):
        self.distDir = distDir
        self.log = logging.getLogger(__name__)
        # Check if distDir exists and is a directory
        if not os.path.isdir(distDir):
            raise ValueError("distDir must be a valid directory")

        # Check for files in distDir
        fileList = os.listdir(distDir)
        if len(fileList) == 0:
            raise ValueError(distDir + " is empty - did you forget to build first?")

        # Split the filenames in distDir
        self.binaries = {}
        for filename in fileList:
            self.log.debug("processing "+ filename)
            parts = filename.split("-")
            hash = ""
            with open(os.path.join(distDir, filename), 'rb') as f:
                hasher = hashlib.md5()
                while True:
                    data = f.read(BUF_SIZE)
                    if not data:
                        break
                    hasher.update(data)
                hash = hasher.hexdigest().lower()
                
            cpu, cov = False, False
            if len(parts) == 1 and parts[0] == "ollama":
                # most likely the multi-arch darwin binary
                self.binaries[("darwin", "amd64", False, False)] = (filename, hash)
                self.binaries[("darwin", "arm64", False, False)] = (filename, hash)
                continue
            elif len(parts) < 3:
                self.log.debug("skipping " + filename)
                continue
            elif len(parts) == 3:
                _, OS, arch = parts[0], parts[1], parts[2]
                arch  = arch.split(".")[0]
            else:
                _, OS, arch, rest = parts[0], parts[1], parts[2], parts[3]
                # Parse the CPU and coverage info
                if rest.startswith("cpu"):
                    cpu = True
                if rest.startswith("cov"):
                    cov = True
            self.binaries[(OS, arch, cpu, cov)] = (filename, hash)

    def getBinary(self, OS, arch, cpu, cov):
        if (OS, arch, cpu, cov) not in self.binaries:
            print(self.binaries.keys())
            print(OS, arch, cpu, cov)
            raise ValueError("requested binary not present")
        return self.binaries[(OS, arch, cpu, cov)]

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    b = Binaries("./dist")
    logging.info("Discovered binaries" + str(b.binaries))


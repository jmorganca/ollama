# Development
- [Development](#development)
  - [Required Versions:](#required-versions)
  - [Mac Prerequisites](#mac-prerequisites)
  - [Linux Prerequisites](#linux-prerequisites)
  - [Local Build and Run](#local-build-and-run)
  - [Containerized Build](#containerized-build)

## Required Versions:

- cmake version 3.24 or higher
- go version 1.20 or higher
- gcc version 11.4.0 or higher


## Mac Prerequisites

```bash
brew install go cmake gcc
```

## Linux Prerequisites

- Install cmake and gcc:
  - Debian `sudo apt-get install cmake gcc`
  - Fedora/CentOS `sudo dnf install cmake gcc-c++`
  - SLES `sudo zypper install cmake gcc-c++`
- Install [golang](https://go.dev/doc/install)
- Optionally, enable GPU support:
  - Cuda (NVIDIA):
    - check [system requirements](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements)
    - select your system and follow [package manager install steps](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation)
  - ROCm (AMD)
    - Install [CLBlast](https://github.com/CNugteren/CLBlast/blob/master/doc/installation.md) and [ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html) developement packages first, as well as `cmake` and `golang`.
    - Adjust the path (correct for Arch) as appropriate for your distribution's install locations and generate dependencies: `export CLBlast_DIR=/usr/lib/cmake/CLBlast ROCM_PATH=/opt/rocm`
    - ROCm requires elevated privileges to access the GPU at runtime.  On most distros you can add your user account to the `render` group, or run as root.
- Optionally enable debugging and more verbose logging: `export CGO_CFLAGS="-g"`



## Local Build and Run
Clone
- If planning to contribure or develop, fork this repo using github's UI
- Clone the ollama repo to your local machine
```bash
git clone git@github.com:<jmorganca|username>/ollama.git
```

Get the required libraries:

```bash
cd ollama
go generate ./...
```

Build ollama:

```bash
go build .
```

Run `ollama`:

```bash
./ollama
```

## Containerized Build

If you have Docker available, you can build linux binaries with `./scripts/build_linux.sh` which has the CUDA and ROCm dependencies included.

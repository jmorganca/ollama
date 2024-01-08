//go:build darwin

package gpu

import "C"
import (
	"runtime"

	"github.com/pbnjay/memory"
)

// CheckVRAM returns the free VRAM in bytes on Linux machines with NVIDIA GPUs
func CheckVRAM() (int64, error) {
	if runtime.GOARCH == "amd64" {
		return 0, nil
	}

	// on macOS, there's already buffer for available vram (see below) so just return the total
	systemMemory := int64(memory.TotalMemory())

	// macOS limits how much memory is available to the GPU based on the amount of system memory
	// TODO: handle case where iogpu.wired_limit_mb is set to a higher value
	if systemMemory <= 36*1024*1024*1024 {
		systemMemory = systemMemory * 2 / 3
	} else {
		systemMemory = systemMemory * 3 / 4
	}

	return systemMemory, nil
}

func GetGPUInfo() GpuInfo {
	mem, _ := getCPUMem()
	return GpuInfo{
		Library: "default",
		memInfo: mem,
	}
}

func getCPUMem() (memInfo, error) {
	return memInfo{
		TotalMemory: 0,
		FreeMemory:  0,
	}, nil
}

func nativeInit() error {
	return nil
}

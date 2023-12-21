//go:build darwin

package gpu

import "C"
import (
	"github.com/jmorganca/ollama/api"
)

// CheckVRAM returns the free VRAM in bytes on Linux machines with NVIDIA GPUs
func CheckVRAM() (int64, error) {
	// TODO - assume metal, and return free memory?
	return 0, nil

}

func GetGPUInfo() GpuInfo {
	// TODO - Metal vs. x86 macs...

	return GpuInfo{
		Driver:      "METAL",
		TotalMemory: 0,
		FreeMemory:  0,
	}
}

func NumGPU(numLayer, fileSizeBytes int64, opts api.Options) int {
	// default to enable metal on macOS
	return 1
}

func nativeInit() error {
	return nil
}

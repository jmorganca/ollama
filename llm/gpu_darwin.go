//go:build darwin

package llm

import (
	"github.com/jmorganca/ollama/api"
)

// CheckVRAM returns the free VRAM in bytes on Linux machines with NVIDIA GPUs
func CheckVRAM() (int64, error) {
	// TODO - assume metal, and return free memory?
	return 0, errNvidiaSMI

}

func NumGPU(numLayer, fileSizeBytes int64, opts api.Options) int {
	// default to enable metal on macOS
	return 1
}

//go:build cuda

package llm

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"log"
	"os/exec"
	"path"
	"strconv"
	"strings"

	"github.com/jmorganca/ollama/format"
)

var (
	errNvidiaSMI     = errors.New("warning: gpu support may not be enabled, check that you have installed GPU drivers: nvidia-smi command failed")
	errAvailableVRAM = errors.New("not enough VRAM available, falling back to CPU only")
)

// acceleratedRunner returns the runner for this accelerator given the provided buildPath string.
func acceleratedRunner(buildPath string) []ModelRunner {
	return []ModelRunner{
		ModelRunner{
			Path:        path.Join(buildPath, "cuda", "bin", "ollama-runner"),
			Accelerated: true,
		},
	}
}

// CheckVRAM returns the free VRAM in bytes on Linux machines with NVIDIA GPUs
func CheckVRAM() (int64, error) {
	cmd := exec.Command("nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits")
	var stdout bytes.Buffer
	cmd.Stdout = &stdout
	err := cmd.Run()
	if err != nil {
		return 0, errNoAccel
	}

	var freeMiB int64
	scanner := bufio.NewScanner(&stdout)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.Contains(line, "[Insufficient Permissions]") {
			return 0, fmt.Errorf("GPU support may not enabled, check you have installed GPU drivers and have the necessary permissions to run nvidia-smi")
		}

		vram, err := strconv.ParseInt(strings.TrimSpace(line), 10, 64)
		if err != nil {
			return 0, fmt.Errorf("failed to parse available VRAM: %v", err)
		}

		freeMiB += vram
	}

	freeBytes := freeMiB * 1024 * 1024
	if freeBytes < 2*format.GigaByte {
		log.Printf("less than 2 GB VRAM available")
		return 0, errAvailableVRAM
	}

	return freeBytes, nil
}

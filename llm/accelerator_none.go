//go:build !rocm && !cuda

package llm

import (
	"errors"
)

var (
	errNoAccel = errors.New("no accelerator support in this binary")
)

// acceleratedRunner returns the runner for this accelerator given the provided buildPath string.
func acceleratedRunner(buildPath string) []ModelRunner {
	return make([]ModelRunner, 0, 1)
}

// CheckVRAM is a stub with no accelerator.
func CheckVRAM() (int64, error) {
	return 0, errNoGPU
}

package gpu

import (
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestBasicGetGPUInfo(t *testing.T) {
	info := GetGPUInfo()
	assert.Contains(t, "CUDA ROCM CPU METAL", info.Driver)

	switch runtime.GOOS {
	case "darwin":
		// TODO - remove this once MacOS returns some size for CPU
		return
	case "linux", "windows":
		assert.Greater(t, info.TotalMemory, uint64(0))
		assert.Greater(t, info.FreeMemory, uint64(0))
	default:
		return
	}
}

// TODO - add some logic to figure out card type through other means and actually verify we got back what we expected

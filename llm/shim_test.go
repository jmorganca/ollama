package llm

import (
	"testing"

	"github.com/jmorganca/ollama/gpu"
	"github.com/stretchr/testify/assert"
)

func TestGetShims(t *testing.T) {
	availableShims = map[string]string{
		"cpu": "X_cpu",
	}
	assert.Equal(t, false, rocmShimPresent())
	res := getShims(gpu.GpuInfo{Library: "cpu"})
	assert.Len(t, res, 2)
	assert.Equal(t, availableShims["cpu"], res[0])
	assert.Equal(t, "default", res[1])

	availableShims = map[string]string{
		"rocm_v5": "X_rocm_v5",
		"rocm_v6": "X_rocm_v6",
		"cpu":     "X_cpu",
	}
	assert.Equal(t, true, rocmShimPresent())
	res = getShims(gpu.GpuInfo{Library: "rocm"})
	assert.Len(t, res, 4)
	assert.Equal(t, availableShims["rocm_v5"], res[0])
	assert.Equal(t, availableShims["rocm_v6"], res[1])
	assert.Equal(t, availableShims["cpu"], res[2])
	assert.Equal(t, "default", res[3])

	res = getShims(gpu.GpuInfo{Library: "rocm", Variant: "v6"})
	assert.Len(t, res, 4)
	assert.Equal(t, availableShims["rocm_v6"], res[0])
	assert.Equal(t, availableShims["rocm_v5"], res[1])
	assert.Equal(t, availableShims["cpu"], res[2])
	assert.Equal(t, "default", res[3])

	res = getShims(gpu.GpuInfo{Library: "cuda"})
	assert.Len(t, res, 2)
	assert.Equal(t, availableShims["cpu"], res[0])
	assert.Equal(t, "default", res[1])

	res = getShims(gpu.GpuInfo{Library: "default"})
	assert.Len(t, res, 2)
	assert.Equal(t, availableShims["cpu"], res[0])
	assert.Equal(t, "default", res[1])

	availableShims = map[string]string{
		"rocm": "X_rocm_v5",
		"cpu":  "X_cpu",
	}
	assert.Equal(t, true, rocmShimPresent())
	res = getShims(gpu.GpuInfo{Library: "rocm", Variant: "v6"})
	assert.Len(t, res, 3)
	assert.Equal(t, availableShims["rocm"], res[0])
	assert.Equal(t, availableShims["cpu"], res[1])
	assert.Equal(t, "default", res[2])

}

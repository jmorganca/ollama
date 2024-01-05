package llm

import (
	"context"
	"fmt"
	"log"
	"os"
	"runtime"

	"github.com/pbnjay/memory"

	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/format"
	"github.com/jmorganca/ollama/gpu"
)

type LLM interface {
	Predict(context.Context, PredictOpts, func(PredictResult)) error
	Embedding(context.Context, string) ([]float64, error)
	Encode(context.Context, string) ([]int, error)
	Decode(context.Context, []int) (string, error)
	Close()
}

func New(workDir, model string, adapters, projectors []string, opts api.Options) (LLM, error) {
	if _, err := os.Stat(model); err != nil {
		return nil, err
	}

	f, err := os.Open(model)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	ggml, err := DecodeGGML(f)
	if err != nil {
		return nil, err
	}

	if runtime.GOOS == "darwin" {
		var requiredMemory int64
		var f16Multiplier int64 = 2

		switch ggml.ModelType() {
		case "3B", "7B":
			requiredMemory = 8 * format.GigaByte
		case "13B":
			requiredMemory = 16 * format.GigaByte
		case "30B", "34B", "40B":
			requiredMemory = 32 * format.GigaByte
		case "47B":
			requiredMemory = 48 * format.GigaByte
		case "65B", "70B":
			requiredMemory = 64 * format.GigaByte
		case "180B":
			requiredMemory = 128 * format.GigaByte
			f16Multiplier = 4
		}

		systemMemory := int64(memory.TotalMemory())

		if ggml.FileType() == "F16" && requiredMemory*f16Multiplier > systemMemory {
			return nil, fmt.Errorf("F16 model requires at least %s of memory", format.HumanBytes(requiredMemory))
		} else if requiredMemory > systemMemory {
			return nil, fmt.Errorf("model requires at least %s of memory", format.HumanBytes(requiredMemory))
		}
	}

	opts.NumGQA = 0
	opts.RopeFrequencyBase = 0.0
	opts.RopeFrequencyScale = 0.0
	gpuInfo := gpu.GetGPUInfo()
	return newLlmServer(gpuInfo, model, adapters, projectors, ggml.NumLayers(), opts)
}

// Give any native cgo implementations an opportunity to initialize
func Init(workdir string) error {
	return nativeInit(workdir)
}

func newLlmServer(gpuInfo gpu.GpuInfo, model string, adapters, projectors []string, numLayers int64, opts api.Options) (extServer, error) {
	for _, shim := range getShims(gpuInfo) {
		if shim == "default" {
			break
		}
		srv, err := newDynamicShimExtServer(shim, model, adapters, projectors, numLayers, opts)
		if err == nil {
			return srv, nil
		}
		log.Printf("Failed to load dynamic library %s  %s", shim, err)
	}
	return newDefaultExtServer(model, adapters, projectors, numLayers, opts)
}

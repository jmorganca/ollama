package llm

import (
	"context"
	"fmt"
	"log"
	"os"
	"runtime"

	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/gpu"
)

type LLM interface {
	Predict(context.Context, PredictOpts, func(PredictResult)) error
	Embedding(context.Context, string) ([]float64, error)
	Encode(context.Context, string) ([]int, error)
	Decode(context.Context, []int) (string, error)
	Close()
}

var AvailableShims = map[string]string{}

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

	if opts.NumCtx < 4 {
		opts.NumCtx = 4
	}

	fmt.Println("size", ggml.Size)
	fmt.Println("filetype", ggml.FileType())
	fmt.Println("architecture", ggml.ModelFamily())
	fmt.Println("type", ggml.ModelType())
	fmt.Println("name", ggml.Name())
	fmt.Println("embd", ggml.NumEmbed())
	fmt.Println("head", ggml.NumHead())
	fmt.Println("head_kv", ggml.NumHeadKv())
	fmt.Println("gqa", ggml.NumGQA())

	available, _ := gpu.CheckVRAM()

	// For now assume filesize = model size
	// TODO: use actual model size
	requiredModel := ggml.Size

	// fp16 k,v matrices each require = n_ctx * n_layer * n_embd / n_head * n_head_kv * 2 bytes each
	requiredKv := 2 * 2 * int64(opts.NumCtx) * int64(ggml.NumLayers()) * int64(ggml.NumEmbed()) * int64(ggml.NumHeadKv()) / int64(ggml.NumHead())

	// this amount is the overhead + tensors in memory
	// TODO: get this from the llama.cpp's graph calcluations instead of
	// guessing it's ~1/8th of the kv cache times gqa
	// this is slightly (~25%) higher than what it is in reality
	// TODO: this needs to account for batch size != 512
	requiredAlloc := int64(ggml.NumGQA()) * requiredKv / 8

	requiredTotal := requiredModel + requiredKv + requiredAlloc

	fmt.Println("system memory ", available)
	fmt.Println("required model", requiredModel)
	fmt.Println("required kv", requiredKv)
	fmt.Println("required alloc", requiredAlloc)
	fmt.Println("required total", requiredTotal)

	if opts.NumGPU == -1 {
		opts.NumGPU = int(ggml.NumLayers()) + 1
	}

	// decide how many layers to put on the GPU
	if opts.NumGPU > 0 {
		switch runtime.GOOS {
		case "darwin":
			if requiredTotal > available {
				fmt.Println("not enough vram available, falling back to CPU only")
				opts.NumGPU = 0
			}
		default:
			info := gpu.GetGPUInfo()
			if info.Library == "cpu" || info.Library == "default" {
				opts.NumGPU = 0
				break
			}

			if requiredTotal <= available {
				break
			}

			// overhead + tensors are always loaded into
			// scratch memory and so don't offload if there isn't even
			// enough room for that
			if requiredAlloc > available {
				fmt.Println("not enough vram available, falling back to CPU only")
				opts.NumGPU = 0
				break
			}

			available -= requiredAlloc

			// fill remaining vram with layers
			fmt.Println("splitting", available, "into layers")
			bytesPerLayer := int64((requiredModel + requiredKv) / int64(ggml.NumLayers()))
			fmt.Println("bytes per layer:", bytesPerLayer)
			layers := available/bytesPerLayer + 1
			if layers < int64(opts.NumGPU) {
				opts.NumGPU = int(layers)
			}
		}
	}

	opts.NumGQA = 0
	opts.RopeFrequencyBase = 0.0
	opts.RopeFrequencyScale = 0.0
	gpuInfo := gpu.GetGPUInfo()
	return newLlmServer(gpuInfo.Library, model, adapters, projectors, opts)
}

// Give any native cgo implementations an opportunity to initialize
func Init(workdir string) error {
	return nativeInit(workdir)
}

func newLlmServer(library, model string, adapters, projectors []string, opts api.Options) (extServer, error) {
	if _, libPresent := AvailableShims[library]; libPresent && library != "default" {
		srv, err := newDynamicShimExtServer(AvailableShims[library], model, adapters, projectors, opts)
		if err == nil {
			return srv, nil
		}
		log.Printf("Failed to load dynamic library %s - falling back to CPU mode %s", library, err)
		// TODO - update some state to indicate we were unable to load the GPU library for future "info" ux
	}

	return newDefaultExtServer(model, adapters, projectors, opts)
}

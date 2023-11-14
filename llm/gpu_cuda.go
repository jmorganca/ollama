//go:build linux || windows

package llm

import (
	"errors"
	"log"

	"github.com/jmorganca/ollama/api"
)

/*
#cgo windows LDFLAGS: -L"/Program Files/NVIDIA Corporation/NVSMI/"
#cgo linux LDFLAGS: -lnvidia-ml

#include <stdlib.h>
#include "examples/server/server.h"
*/
import "C"

// CheckVRAM returns the free VRAM in bytes on Linux machines with NVIDIA GPUs
func CheckVRAM() (int64, error) {
	return int64(C.check_vram()), nil
}

func NumGPU(numLayer, fileSizeBytes int64, opts api.Options) int {
	if opts.NumGPU != -1 {
		return opts.NumGPU
	}
	freeBytes, err := CheckVRAM()
	if err != nil {
		if !errors.Is(err, errNvidiaSMI) {
			log.Print(err.Error())
		}
		// nvidia driver not installed or no nvidia GPU found
		return 0
	}

	/*
		Calculate bytes per layer, this will roughly be the size of the model file divided by the number of layers.
		We can store the model weights and the kv cache in vram,
		to enable kv chache vram storage add two additional layers to the number of layers retrieved from the model file.
	*/
	bytesPerLayer := fileSizeBytes / numLayer

	// 75% of the absolute max number of layers we can fit in available VRAM, off-loading too many layers to the GPU can cause OOM errors
	layers := int(freeBytes/bytesPerLayer) * 3 / 4

	// TODO - not sure on this part... if we can't fit all the layers, just fallback to CPU
	// if int64(layers) < numLayer {
	// 	log.Printf("%d MB VRAM available, insufficient to load current model (reuires %d MB) - falling back to CPU %d", freeBytes/(1024*1024), fileSizeBytes/(1024*1024))
	// 	return 0
	// }
	log.Printf("%d MB VRAM available, loading up to %d GPU layers out of %d", freeBytes/(1024*1024), layers, numLayer)

	return layers
}

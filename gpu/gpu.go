//go:build linux || windows

package gpu

/*
#include "gpu_info.h"

*/
import "C"
import (
	"fmt"
	"log"
	"sync"
	"unsafe"

	"github.com/jmorganca/ollama/api"
)

type handles struct {
	cuda *C.cuda_handle_t
	rocm *C.rocm_handle_t
}

var gpuMutex sync.Mutex
var gpuHandles *handles = nil

// Note: gpuMutex must already be held
func initGPUHandles() {
	log.Printf("Detecting GPU type")
	gpuHandles = &handles{nil, nil}
	var resp C.cuda_init_resp_t
	C.cuda_init(&resp)
	if resp.err != nil {
		log.Printf("CUDA not detected: %s", C.GoString(resp.err))
		C.free(unsafe.Pointer(resp.err))

		var resp C.rocm_init_resp_t
		C.rocm_init(&resp)
		if resp.err != nil {
			log.Printf("ROCm not detected: %s", C.GoString(resp.err))
			C.free(unsafe.Pointer(resp.err))
		} else {
			log.Printf("Radeon GPU detected")
			rocm := resp.rh
			gpuHandles.rocm = &rocm
		}
	} else {
		log.Printf("Nvidia GPU detected")
		cuda := resp.ch
		gpuHandles.cuda = &cuda
	}
}

func GetGPUInfo() GpuInfo {
	// TODO - consider exploring lspci (and equivalent on windows) to check for
	// GPUs so we can report warnings if we see Nvidia/AMD but fail to load the libraries
	gpuMutex.Lock()
	defer gpuMutex.Unlock()
	if gpuHandles == nil {
		initGPUHandles()
	}

	var memInfo C.mem_info_t
	var resp GpuInfo
	if gpuHandles.cuda != nil {
		C.cuda_check_vram(*gpuHandles.cuda, &memInfo)
		resp.Driver = "CUDA"
	} else if gpuHandles.rocm != nil {
		C.rocm_check_vram(*gpuHandles.rocm, &memInfo)
		resp.Driver = "ROCM"
	} else {
		C.cpu_check_ram(&memInfo)
		resp.Driver = "CPU"
	}
	if memInfo.err != nil {
		log.Printf("error looking up GPU memory: %s", C.GoString(memInfo.err))
		C.free(unsafe.Pointer(memInfo.err))
	}
	resp.FreeMemory = uint64(memInfo.free)
	resp.TotalMemory = uint64(memInfo.total)
	return resp
}

func CheckVRAM() (int64, error) {
	gpuInfo := GetGPUInfo()
	if gpuInfo.FreeMemory > 0 && gpuInfo.Driver != "CPU" {
		return int64(gpuInfo.FreeMemory), nil
	}
	return 0, fmt.Errorf("no GPU detected") // TODO - better handling of CPU based memory determiniation
}

func NumGPU(numLayer, fileSizeBytes int64, opts api.Options) int {
	if opts.NumGPU != -1 {
		return opts.NumGPU
	}
	info := GetGPUInfo()
	if info.Driver == "CPU" {
		return 0
	}

	/*
		Calculate bytes per layer, this will roughly be the size of the model file divided by the number of layers.
		We can store the model weights and the kv cache in vram,
		to enable kv chache vram storage add two additional layers to the number of layers retrieved from the model file.
	*/
	bytesPerLayer := uint64(fileSizeBytes / numLayer)

	// 75% of the absolute max number of layers we can fit in available VRAM, off-loading too many layers to the GPU can cause OOM errors
	layers := int(info.FreeMemory/bytesPerLayer) * 3 / 4

	// TODO - not sure on this part... if we can't fit all the layers, just fallback to CPU
	// if int64(layers) < numLayer {
	// 	log.Printf("%d MB VRAM available, insufficient to load current model (reuires %d MB) - falling back to CPU %d", freeBytes/(1024*1024), fileSizeBytes/(1024*1024))
	// 	return 0
	// }
	log.Printf("%d MB VRAM available, loading up to %d GPU layers out of %d", info.FreeMemory/(1024*1024), layers, numLayer)

	return layers
}

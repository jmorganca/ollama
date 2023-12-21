package llm

import (
	"fmt"

	"github.com/jmorganca/ollama/api"
)

// no-op stubs for mac

func newRocmShimExtServer(model string, adapters, projectors []string, numLayers int64, opts api.Options) (extServer, error) {
	// should never happen...
	return nil, fmt.Errorf("ROCM GPUs not supported on Mac")
}

func nativeInit(workDir string) error {
	return nil
}

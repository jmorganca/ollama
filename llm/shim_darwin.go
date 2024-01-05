package llm

import (
	"embed"
	"fmt"

	"github.com/jmorganca/ollama/api"
)

//go:embed llama.cpp/ggml-metal.metal
var libEmbed embed.FS

func newDynamicShimExtServer(library, model string, adapters, projectors []string, numLayers int64, opts api.Options) (extServer, error) {
	// should never happen...
	return nil, fmt.Errorf("Dynamic library loading not supported on Mac")
}

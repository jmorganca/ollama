package llm

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"sync"
	"time"

	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/format"
)

const jsonGrammar = `
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws

# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= ([ \t\n] ws)?
`

type llamaModel struct {
	hyperparameters llamaHyperparameters
}

func (llm *llamaModel) ModelFamily() string {
	return "llama"
}

func llamaModelType(numLayer uint32) string {
	switch numLayer {
	case 26:
		return "3B"
	case 32:
		return "7B"
	case 40:
		return "13B"
	case 48:
		return "34B"
	case 60:
		return "30B"
	case 80:
		return "65B"
	default:
		return "unknown"
	}
}

func (llm *llamaModel) ModelType() string {
	return llamaModelType(llm.hyperparameters.NumLayer)
}

func (llm *llamaModel) FileType() string {
	return fileType(llm.hyperparameters.FileType)
}

func (llm *llamaModel) NumLayers() int64 {
	return int64(llm.hyperparameters.NumLayer)
}

type llamaHyperparameters struct {
	// NumVocab is the size of the model's vocabulary.
	NumVocab uint32

	// NumEmbd is the size of the model's embedding layer.
	NumEmbd uint32
	NumMult uint32
	NumHead uint32

	// NumLayer is the number of layers in the model.
	NumLayer uint32
	NumRot   uint32

	// FileType describes the quantization level of the model, e.g. Q4_0, Q5_K, etc.
	FileType uint32
}

type Running struct {
	Port          int
	Cmd           *exec.Cmd
	Cancel        context.CancelFunc
	exitOnce      sync.Once
	exitCh        chan error // channel to receive the exit status of the subprocess
	*StatusWriter            // captures error messages from the llama runner process
}

type ImageData struct {
	Data []byte `json:"data"`
	ID   int    `json:"id"`
}

type llama struct {
	api.Options
	ImageData []ImageData
	Running
}

var (
	errNvidiaSMI     = errors.New("warning: gpu support may not be enabled, check that you have installed GPU drivers: nvidia-smi command failed")
	errAvailableVRAM = errors.New("not enough VRAM available, falling back to CPU only")
)

// StatusWriter is a writer that captures error messages from the llama runner process
type StatusWriter struct {
	ErrCh      chan error
	LastErrMsg string
}

func NewStatusWriter() *StatusWriter {
	return &StatusWriter{
		ErrCh: make(chan error, 1),
	}
}

func (w *StatusWriter) Write(b []byte) (int, error) {
	var errMsg string
	if _, after, ok := bytes.Cut(b, []byte("error:")); ok {
		errMsg = string(bytes.TrimSpace(after))
	} else if _, after, ok := bytes.Cut(b, []byte("CUDA error")); ok {
		errMsg = string(bytes.TrimSpace(after))
	}

	if errMsg != "" {
		w.LastErrMsg = errMsg
		w.ErrCh <- fmt.Errorf("llama runner: %s", errMsg)
	}

	return os.Stderr.Write(b)
}

type prediction struct {
	Content string `json:"content"`
	Model   string `json:"model"`
	Prompt  string `json:"prompt"`
	Stop    bool   `json:"stop"`

	Timings struct {
		PredictedN  int     `json:"predicted_n"`
		PredictedMS float64 `json:"predicted_ms"`
		PromptN     int     `json:"prompt_n"`
		PromptMS    float64 `json:"prompt_ms"`
	}
}

const maxBufferSize = 512 * format.KiloByte
const maxRetries = 6

type PredictOpts struct {
	Prompt string
	Format string
	Images []api.ImageData
}

type PredictResult struct {
	Content            string
	Done               bool
	PromptEvalCount    int
	PromptEvalDuration time.Duration
	EvalCount          int
	EvalDuration       time.Duration
}

type TokenizeRequest struct {
	Content string `json:"content"`
}

type TokenizeResponse struct {
	Tokens []int `json:"tokens"`
}

type DetokenizeRequest struct {
	Tokens []int `json:"tokens"`
}

type DetokenizeResponse struct {
	Content string `json:"content"`
}

type EmbeddingRequest struct {
	Content string `json:"content"`
}

type EmbeddingResponse struct {
	Embedding []float64 `json:"embedding"`
}

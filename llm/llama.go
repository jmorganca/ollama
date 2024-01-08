package llm

import (
	"bytes"
	"context"
	_ "embed"
	"errors"
	"fmt"
	"os"
	"os/exec"
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

type Running struct {
	Port          int
	Cmd           *exec.Cmd
	Cancel        context.CancelFunc
	*StatusWriter // captures error messages from the llama runner process
}

type ImageData struct {
	Data []byte `json:"data"`
	ID   int    `json:"id"`
}

var (
	errNvidiaSMI     = errors.New("warning: gpu support may not be enabled, check that you have installed GPU drivers: nvidia-smi command failed")
	errAvailableVRAM = errors.New("not enough VRAM available, falling back to CPU only")
	payloadMissing   = fmt.Errorf("expected dynamic library payloads not included in this build of ollama")
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
const maxRetries = 3
const retryDelay = 1 * time.Second

type PredictOpts struct {
	Prompt  string
	Format  string
	Images  []api.ImageData
	Options api.Options
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

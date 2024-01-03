package llm

import (
	"bytes"
	"context"
	_ "embed"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"time"

	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/format"
)

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

func extractDynamicLibs(workDir, glob string) ([]string, error) {
	files, err := fs.Glob(libEmbed, glob)
	if err != nil || len(files) == 0 {
		return nil, payloadMissing
	}
	libs := make([]string, len(files))

	for i, file := range files {
		srcFile, err := libEmbed.Open(file)
		if err != nil {
			return nil, fmt.Errorf("read payload %s: %v", file, err)
		}
		defer srcFile.Close()
		if err := os.MkdirAll(workDir, 0o755); err != nil {
			return nil, fmt.Errorf("create payload temp dir %s: %v", workDir, err)
		}

		destFile := filepath.Join(workDir, filepath.Base(file))
		libs[i] = destFile

		_, err = os.Stat(destFile)
		switch {
		case errors.Is(err, os.ErrNotExist):
			destFile, err := os.OpenFile(destFile, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o755)
			if err != nil {
				return nil, fmt.Errorf("write payload %s: %v", file, err)
			}
			defer destFile.Close()
			if _, err := io.Copy(destFile, srcFile); err != nil {
				return nil, fmt.Errorf("copy payload %s: %v", file, err)
			}
		case err != nil:
			return nil, fmt.Errorf("stat payload %s: %v", file, err)
		}
	}
	return libs, nil
}

//go:build !darwin

package llm

/*

#include <stdlib.h>
#include "rocm_shim.h"

*/
import "C"
import (
	"context"
	"embed"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"unsafe"

	"github.com/jmorganca/ollama/api"
)

//go:embed llama.cpp/gguf/build/*/lib/*
var libEmbed embed.FS

var RocmShimMissing = fmt.Errorf("ROCm shim library not included in this build of ollama. Radeon GPUs are not supported")

type shimExtServer struct {
	s       C.struct_rocm_llama_server
	options api.Options
}

// Note: current implementation does not support concurrent instantiations
var shimMutex sync.Mutex
var llm *shimExtServer

func (llm *shimExtServer) llama_server_init(sparams *C.ext_server_params_t, err *C.ext_server_resp_t) {
	C.rocm_shim_llama_server_init(llm.s, sparams, err)
}
func (llm *shimExtServer) llama_server_start() {
	C.rocm_shim_llama_server_start(llm.s)
}
func (llm *shimExtServer) llama_server_stop() {
	C.rocm_shim_llama_server_stop(llm.s)
}

func (llm *shimExtServer) llama_server_completion(json_req *C.char, resp *C.ext_server_resp_t) {
	C.rocm_shim_llama_server_completion(llm.s, json_req, resp)
}
func (llm *shimExtServer) llama_server_completion_next_result(task_id C.int, resp *C.ext_server_task_result_t) {
	C.rocm_shim_llama_server_completion_next_result(llm.s, task_id, resp)
}
func (llm *shimExtServer) llama_server_completion_cancel(task_id C.int, err *C.ext_server_resp_t) {
	C.rocm_shim_llama_server_completion_cancel(llm.s, task_id, err)
}
func (llm *shimExtServer) llama_server_release_task_result(result *C.ext_server_task_result_t) {
	C.rocm_shim_llama_server_release_task_result(llm.s, result)
}

func (llm *shimExtServer) llama_server_tokenize(json_req *C.char, json_resp **C.char, err *C.ext_server_resp_t) {
	C.rocm_shim_llama_server_tokenize(llm.s, json_req, json_resp, err)
}
func (llm *shimExtServer) llama_server_detokenize(json_req *C.char, json_resp **C.char, err *C.ext_server_resp_t) {
	C.rocm_shim_llama_server_detokenize(llm.s, json_req, json_resp, err)
}
func (llm *shimExtServer) llama_server_embedding(json_req *C.char, json_resp **C.char, err *C.ext_server_resp_t) {
	C.rocm_shim_llama_server_embedding(llm.s, json_req, json_resp, err)
}
func (llm *shimExtServer) llama_server_release_json_resp(json_resp **C.char) {
	C.rocm_shim_llama_server_release_json_resp(llm.s, json_resp)
}

func newRocmShimExtServer(model string, adapters, projectors []string, numLayers int64, opts api.Options) (extServer, error) {
	if !ShimPresent {
		return nil, RocmShimMissing
	}
	log.Printf("Loading ROCM llm server")
	if llm == nil {
		return nil, fmt.Errorf("nativeInit wasnt called or libary load failed")
	}
	llm.options = opts
	return newExtServer(llm, model, adapters, projectors, numLayers, opts)
}

func (llm *shimExtServer) Predict(ctx context.Context, pred PredictOpts, fn func(PredictResult)) error {
	return predict(llm, llm.options, ctx, pred, fn)
}

func (llm *shimExtServer) Encode(ctx context.Context, prompt string) ([]int, error) {
	return encode(llm, ctx, prompt)
}

func (llm *shimExtServer) Decode(ctx context.Context, tokens []int) (string, error) {
	return decode(llm, ctx, tokens)
}

func (llm *shimExtServer) Embedding(ctx context.Context, input string) ([]float64, error) {
	return embedding(llm, ctx, input)
}

func (llm *shimExtServer) Close() {
	close(llm)
}

func nativeInit(workdir string) error {
	err := extractLib(workdir)
	if err != nil {
		if err == RocmShimMissing {
			log.Printf("%s", err)
			return nil
		}
		return err
	}

	// Verify we have permissions - either running as root, or we have group access to the driver
	fd, err := os.OpenFile("/dev/kfd", os.O_RDWR, 0666)
	if err != nil {
		if errors.Is(err, fs.ErrPermission) {
			log.Fatalf("Radeon card detected, but permissions not set up properly.  Either run ollama as root, or add you user account to the render group.")
			return err
		} else if errors.Is(err, fs.ErrNotExist) {
			// expected behavior without a radeon card
			return nil
		}

		return fmt.Errorf("failed to check permission on /dev/kfd: %w", err)
	}
	fd.Close()

	shimMutex.Lock()
	defer shimMutex.Unlock()
	if llm != nil {
		return nil
	}
	var libName string
	switch runtime.GOOS {
	case "darwin":
		// shouldn't happen
		return nil
	case "linux":
		libName = "librocm_server.so"
	case "windows":
		libName = "rocm_server.dll"
	default:
		// shouldn't happen
		return nil
	}
	libPath := C.CString(filepath.Join(workdir, libName))
	defer C.free(unsafe.Pointer(libPath))
	resp := newExtServerResp(128)
	defer freeExtServerResp(resp)
	var srv C.struct_rocm_llama_server
	C.rocm_shim_init(libPath, &srv, &resp)
	if resp.id < 0 {
		// TODO - consider softening this failure mode to allow fall-back to the CUDA based built-in llm
		//        and run against CPU
		return fmt.Errorf("Unable to load AMD GPU library: %s", C.GoString(resp.msg))
	}
	llm = &shimExtServer{
		s:       srv,
		options: api.DefaultOptions(),
	}
	return nil
}

func extractLib(workDir string) error {
	files, err := fs.Glob(libEmbed, "llama.cpp/gguf/build/*/lib/*rocm_server*")
	if err != nil || len(files) == 0 {
		// this is expected, ollama may be compiled without shim library packed in
		return RocmShimMissing
	}

	if len(files) != 1 {
		// Shouldn't happen, but just use the first one we find
		log.Printf("WARNING: multiple rocm libraries detected - using %s", files[0])
	}

	srcFile, err := libEmbed.Open(files[0])
	if err != nil {
		return fmt.Errorf("read ROCm shim %s: %v", files[0], err)
	}
	defer srcFile.Close()
	if err := os.MkdirAll(workDir, 0o755); err != nil {
		return fmt.Errorf("create ROCm shim temp dir %s: %v", workDir, err)
	}

	destFile := filepath.Join(workDir, filepath.Base(files[0]))

	_, err = os.Stat(destFile)
	switch {
	case errors.Is(err, os.ErrNotExist):
		destFile, err := os.OpenFile(destFile, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o755)
		if err != nil {
			return fmt.Errorf("write ROCm shim %s: %v", files[0], err)
		}
		defer destFile.Close()
		if _, err := io.Copy(destFile, srcFile); err != nil {
			return fmt.Errorf("copy ROCm shim %s: %v", files[0], err)
		}
	case err != nil:
		return fmt.Errorf("stat ROCm shim %s: %v", files[0], err)
	}
	ShimPresent = true
	return nil
}

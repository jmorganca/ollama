//go:build !darwin

package llm

/*

#include <stdlib.h>
#include "dynamic_shim.h"

*/
import "C"
import (
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"unsafe"

	"github.com/jmorganca/ollama/api"
)

type shimExtServer struct {
	s       C.struct_dynamic_llama_server
	options api.Options
}

// Note: current implementation does not support concurrent instantiations
var shimMutex sync.Mutex
var llm *shimExtServer

const pathComponentCount = 6

func (llm *shimExtServer) llama_server_init(sparams *C.ext_server_params_t, err *C.ext_server_resp_t) {
	C.dynamic_shim_llama_server_init(llm.s, sparams, err)
}
func (llm *shimExtServer) llama_server_start() {
	C.dynamic_shim_llama_server_start(llm.s)
}
func (llm *shimExtServer) llama_server_stop() {
	C.dynamic_shim_llama_server_stop(llm.s)
}

func (llm *shimExtServer) llama_server_completion(json_req *C.char, resp *C.ext_server_resp_t) {
	C.dynamic_shim_llama_server_completion(llm.s, json_req, resp)
}
func (llm *shimExtServer) llama_server_completion_next_result(task_id C.int, resp *C.ext_server_task_result_t) {
	C.dynamic_shim_llama_server_completion_next_result(llm.s, task_id, resp)
}
func (llm *shimExtServer) llama_server_completion_cancel(task_id C.int, err *C.ext_server_resp_t) {
	C.dynamic_shim_llama_server_completion_cancel(llm.s, task_id, err)
}
func (llm *shimExtServer) llama_server_release_task_result(result *C.ext_server_task_result_t) {
	C.dynamic_shim_llama_server_release_task_result(llm.s, result)
}

func (llm *shimExtServer) llama_server_tokenize(json_req *C.char, json_resp **C.char, err *C.ext_server_resp_t) {
	C.dynamic_shim_llama_server_tokenize(llm.s, json_req, json_resp, err)
}
func (llm *shimExtServer) llama_server_detokenize(json_req *C.char, json_resp **C.char, err *C.ext_server_resp_t) {
	C.dynamic_shim_llama_server_detokenize(llm.s, json_req, json_resp, err)
}
func (llm *shimExtServer) llama_server_embedding(json_req *C.char, json_resp **C.char, err *C.ext_server_resp_t) {
	C.dynamic_shim_llama_server_embedding(llm.s, json_req, json_resp, err)
}
func (llm *shimExtServer) llama_server_release_json_resp(json_resp **C.char) {
	C.dynamic_shim_llama_server_release_json_resp(llm.s, json_resp)
}

func newDynamicShimExtServer(library, model string, adapters, projectors []string, opts api.Options) (extServer, error) {
	shimMutex.Lock()
	defer shimMutex.Unlock()
	updatePath(filepath.Dir(library))
	libPath := C.CString(library)
	defer C.free(unsafe.Pointer(libPath))
	resp := newExtServerResp(128)
	defer freeExtServerResp(resp)
	var srv C.struct_dynamic_llama_server
	C.dynamic_shim_init(libPath, &srv, &resp)
	if resp.id < 0 {
		return nil, fmt.Errorf("Unable to load dynamic library: %s", C.GoString(resp.msg))
	}
	llm = &shimExtServer{
		s:       srv,
		options: opts,
	}
	log.Printf("Loading Dynamic Shim llm server: %s", library)
	return newExtServer(llm, model, adapters, projectors, opts)
}

func (llm *shimExtServer) Predict(ctx context.Context, pred PredictOpts, fn func(PredictResult)) error {
	return predict(ctx, llm, pred, fn)
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
	libs, err := extractDynamicLibs(workdir, "llama.cpp/build/*/*/lib/*")
	if err != nil {
		if err == payloadMissing {
			log.Printf("%s", payloadMissing)
			return nil
		}
		return err
	}
	for _, lib := range libs {
		// The last dir component is the variant name
		variant := filepath.Base(filepath.Dir(lib))
		AvailableShims[variant] = lib
	}

	if err := verifyDriverAccess(); err != nil {
		return err
	}

	// Report which dynamic libraries we have loaded to assist troubleshooting
	variants := make([]string, len(AvailableShims))
	i := 0
	for variant := range AvailableShims {
		variants[i] = variant
		i++
	}
	log.Printf("Dynamic LLM variants %v", variants)

	return nil
}

func extractDynamicLibs(workDir, glob string) ([]string, error) {
	files, err := fs.Glob(libEmbed, glob)
	if err != nil || len(files) == 0 {
		return nil, payloadMissing
	}
	libs := []string{}

	for _, file := range files {
		pathComps := strings.Split(file, "/")
		if len(pathComps) != pathComponentCount {
			log.Printf("unexpected payload components: %v", pathComps)
			continue
		}
		// llama.cpp/build/$OS/$VARIANT/lib/$LIBRARY
		// Include the variant in the path to avoid conflicts between multiple server libs
		targetDir := filepath.Join(workDir, pathComps[pathComponentCount-3])
		srcFile, err := libEmbed.Open(file)
		if err != nil {
			return nil, fmt.Errorf("read payload %s: %v", file, err)
		}
		defer srcFile.Close()
		if err := os.MkdirAll(targetDir, 0o755); err != nil {
			return nil, fmt.Errorf("create payload temp dir %s: %v", workDir, err)
		}

		destFile := filepath.Join(targetDir, filepath.Base(file))
		if strings.Contains(destFile, "server") {
			libs = append(libs, destFile)
		}

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

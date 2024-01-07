//go:build !darwin

package llm

/*

#include <stdlib.h>
#include "dynamic_shim.h"

*/
import "C"
import (
	"context"
	"fmt"
	"log"
	"path/filepath"
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

func newDynamicShimExtServer(library, model string, adapters, projectors []string, numLayers int64, opts api.Options) (extServer, error) {
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
	return newExtServer(llm, model, adapters, projectors, numLayers, opts)
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

//go:build !windows

package llm

/*
#include <stdlib.h>
#include "ext_server.h"

*/
import "C"
import (
	"context"

	"github.com/jmorganca/ollama/api"
)

type llamaExtServer struct {
	api.Options
}

func (llm *llamaExtServer) llama_server_init(sparams *C.ext_server_params_t, err *C.ext_server_resp_t) {
	C.llama_server_init(sparams, err)
}
func (llm *llamaExtServer) llama_server_start() {
	C.llama_server_start()
}
func (llm *llamaExtServer) llama_server_stop() {
	C.llama_server_stop()
}

func (llm *llamaExtServer) llama_server_completion(json_req *C.char, resp *C.ext_server_resp_t) {
	C.llama_server_completion(json_req, resp)
}
func (llm *llamaExtServer) llama_server_completion_next_result(task_id C.int, resp *C.ext_server_task_result_t) {
	C.llama_server_completion_next_result(task_id, resp)
}
func (llm *llamaExtServer) llama_server_completion_cancel(task_id C.int, err *C.ext_server_resp_t) {
	C.llama_server_completion_cancel(task_id, err)
}
func (llm *llamaExtServer) llama_server_release_task_result(result *C.ext_server_task_result_t) {
	C.llama_server_release_task_result(result)
}

func (llm *llamaExtServer) llama_server_tokenize(json_req *C.char, json_resp **C.char, err *C.ext_server_resp_t) {
	C.llama_server_tokenize(json_req, json_resp, err)
}
func (llm *llamaExtServer) llama_server_detokenize(json_req *C.char, json_resp **C.char, err *C.ext_server_resp_t) {
	C.llama_server_detokenize(json_req, json_resp, err)
}
func (llm *llamaExtServer) llama_server_embedding(json_req *C.char, json_resp **C.char, err *C.ext_server_resp_t) {
	C.llama_server_embedding(json_req, json_resp, err)
}
func (llm *llamaExtServer) llama_server_release_json_resp(json_resp **C.char) {
	C.llama_server_release_json_resp(json_resp)
}

func newDefaultExtServer(model string, adapters, projectors []string, opts api.Options) (extServer, error) {
	server := &llamaExtServer{opts}
	return newExtServer(server, model, adapters, projectors, opts)
}

func (llm *llamaExtServer) Predict(ctx context.Context, pred PredictOpts, fn func(PredictResult)) error {
	return predict(ctx, llm, pred, fn)
}

func (llm *llamaExtServer) Encode(ctx context.Context, prompt string) ([]int, error) {
	return encode(llm, ctx, prompt)
}

func (llm *llamaExtServer) Decode(ctx context.Context, tokens []int) (string, error) {
	return decode(llm, ctx, tokens)
}

func (llm *llamaExtServer) Embedding(ctx context.Context, input string) ([]float64, error) {
	return embedding(llm, ctx, input)
}

func (llm *llamaExtServer) Close() {
	close(llm)
}

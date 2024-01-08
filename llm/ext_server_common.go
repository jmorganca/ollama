package llm

/*
#cgo CFLAGS: -I${SRCDIR}/ext_server -I${SRCDIR}/llama.cpp -I${SRCDIR}/llama.cpp/common -I${SRCDIR}/llama.cpp/examples/server
#cgo CFLAGS: -DNDEBUG -DLLAMA_SERVER_LIBRARY=1 -D_XOPEN_SOURCE=600 -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64
#cgo CFLAGS: -Wmissing-noreturn -Wextra -Wcast-qual -Wno-unused-function -Wno-array-bounds
#cgo CPPFLAGS: -Ofast -Wextra -Wno-unused-function -Wno-unused-variable -Wno-deprecated-declarations -Wno-unused-but-set-variable
#cgo darwin CFLAGS: -D_DARWIN_C_SOURCE
#cgo darwin CPPFLAGS:  -DGGML_USE_ACCELERATE
#cgo darwin CPPFLAGS: -DGGML_USE_METAL -DGGML_METAL_NDEBUG
#cgo darwin LDFLAGS: -lc++ -framework Accelerate
#cgo darwin LDFLAGS: -framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders
#cgo darwin LDFLAGS: ${SRCDIR}/llama.cpp/build/darwin/metal/lib/libcommon.a
#cgo darwin LDFLAGS: ${SRCDIR}/llama.cpp/build/darwin/metal/lib/libext_server.a
#cgo darwin LDFLAGS: ${SRCDIR}/llama.cpp/build/darwin/metal/lib/libllama.a
#cgo darwin LDFLAGS: ${SRCDIR}/llama.cpp/build/darwin/metal/lib/libggml_static.a
#cgo linux CFLAGS: -D_GNU_SOURCE
#cgo linux windows CFLAGS: -DGGML_CUDA_DMMV_X=32 -DGGML_CUDA_MMV_Y=1 -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 -DGGML_USE_CUBLAS
#cgo linux LDFLAGS: -L/usr/local/cuda/targets/x86_64-linux/lib -L/usr/local/cuda/lib64 -L/usr/local/cuda/targets/x86_64-linux/lib/stubs
#cgo linux LDFLAGS: ${SRCDIR}/llama.cpp/build/linux/cpu/lib/libext_server.a
#cgo linux LDFLAGS: ${SRCDIR}/llama.cpp/build/linux/cpu/lib/libcommon.a
#cgo linux LDFLAGS: ${SRCDIR}/llama.cpp/build/linux/cpu/lib/libllama.a
#cgo linux LDFLAGS: ${SRCDIR}/llama.cpp/build/linux/cpu/lib/libggml_static.a
#cgo linux LDFLAGS: -lrt -ldl -lstdc++ -lm
#cgo linux windows LDFLAGS: -lpthread

#include <stdlib.h>
#include "ext_server.h"

*/
import "C"
import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strings"
	"sync"
	"time"
	"unsafe"

	"github.com/jmorganca/ollama/api"
)

type extServer interface {
	LLM
	llama_server_init(sparams *C.ext_server_params_t, err *C.ext_server_resp_t)
	llama_server_start()
	llama_server_stop()
	llama_server_completion(json_req *C.char, resp *C.ext_server_resp_t)
	llama_server_completion_next_result(task_id C.int, resp *C.ext_server_task_result_t)
	llama_server_completion_cancel(task_id C.int, err *C.ext_server_resp_t)
	llama_server_release_task_result(result *C.ext_server_task_result_t)
	llama_server_tokenize(json_req *C.char, json_resp **C.char, err *C.ext_server_resp_t)
	llama_server_detokenize(json_req *C.char, json_resp **C.char, err *C.ext_server_resp_t)
	llama_server_embedding(json_req *C.char, json_resp **C.char, err *C.ext_server_resp_t)
	llama_server_release_json_resp(json_resp **C.char)
}

// Note: current implementation does not support concurrent instantiations
var mutex sync.Mutex

func newExtServerResp(len C.size_t) C.ext_server_resp_t {
	var resp C.ext_server_resp_t
	resp.msg_len = len
	bytes := make([]byte, len)
	resp.msg = (*C.char)(C.CBytes(bytes))
	return resp
}

func freeExtServerResp(resp C.ext_server_resp_t) {
	if resp.msg_len == 0 {
		return
	}
	C.free(unsafe.Pointer(resp.msg))
}

func extServerResponseToErr(resp C.ext_server_resp_t) error {
	return fmt.Errorf(C.GoString(resp.msg))
}

func newExtServer(server extServer, model string, adapters, projectors []string, opts api.Options) (extServer, error) {
	if !mutex.TryLock() {
		log.Printf("concurrent llm servers not yet supported, waiting for prior server to complete")
		mutex.Lock()
	}

	var sparams C.ext_server_params_t
	sparams.model = C.CString(model)
	defer C.free(unsafe.Pointer(sparams.model))

	sparams.embedding = true
	sparams.n_ctx = C.uint(opts.NumCtx)
	sparams.n_batch = C.uint(opts.NumBatch)
	sparams.n_gpu_layers = C.int(opts.NumGPU)
	sparams.main_gpu = C.int(opts.MainGPU)
	sparams.n_parallel = 1 // TODO - wire up concurrency

	// Always use the value encoded in the model
	sparams.rope_freq_base = 0.0
	sparams.rope_freq_scale = 0.0
	sparams.memory_f16 = C.bool(opts.F16KV)
	sparams.use_mlock = C.bool(opts.UseMLock)
	sparams.use_mmap = C.bool(opts.UseMMap)
	sparams.numa = C.bool(opts.UseNUMA)

	sparams.lora_adapters = nil
	for i := 0; i < len(adapters); i++ {
		la := (*C.ext_server_lora_adapter_t)(C.malloc(C.sizeof_ext_server_lora_adapter_t))
		defer C.free(unsafe.Pointer(la))
		la.adapter = C.CString(adapters[i])
		defer C.free(unsafe.Pointer(la.adapter))
		la.scale = C.float(1.0) // TODO expose scale/weights up through ollama UX
		la.next = nil
		if i == 0 {
			sparams.lora_adapters = la
		} else {
			tmp := sparams.lora_adapters
			for ; tmp.next != nil; tmp = tmp.next {
			}
			tmp.next = la
		}
	}

	if len(projectors) > 0 {
		// TODO: applying multiple projectors is not supported by the llama.cpp server yet
		sparams.mmproj = C.CString(projectors[0])
		defer C.free(unsafe.Pointer(sparams.mmproj))
	} else {
		sparams.mmproj = nil
	}

	sparams.n_threads = C.uint(opts.NumThread)

	log.Printf("Initializing internal llama server")
	resp := newExtServerResp(128)
	defer freeExtServerResp(resp)
	server.llama_server_init(&sparams, &resp)
	if resp.id < 0 {
		return nil, extServerResponseToErr(resp)
	}

	log.Printf("Starting internal llama main loop")
	server.llama_server_start()
	return server, nil
}

func predict(ctx context.Context, llm extServer, predict PredictOpts, fn func(PredictResult)) error {
	resp := newExtServerResp(128)
	defer freeExtServerResp(resp)
	var imageData []ImageData
	if len(predict.Images) > 0 {
		for cnt, i := range predict.Images {
			imageData = append(imageData, ImageData{Data: i, ID: cnt})
		}
	}
	log.Printf("loaded %d images", len(imageData))

	request := map[string]any{
		"prompt":            predict.Prompt,
		"stream":            true,
		"n_predict":         predict.Options.NumPredict,
		"n_keep":            predict.Options.NumKeep,
		"temperature":       predict.Options.Temperature,
		"top_k":             predict.Options.TopK,
		"top_p":             predict.Options.TopP,
		"tfs_z":             predict.Options.TFSZ,
		"typical_p":         predict.Options.TypicalP,
		"repeat_last_n":     predict.Options.RepeatLastN,
		"repeat_penalty":    predict.Options.RepeatPenalty,
		"presence_penalty":  predict.Options.PresencePenalty,
		"frequency_penalty": predict.Options.FrequencyPenalty,
		"mirostat":          predict.Options.Mirostat,
		"mirostat_tau":      predict.Options.MirostatTau,
		"mirostat_eta":      predict.Options.MirostatEta,
		"penalize_nl":       predict.Options.PenalizeNewline,
		"seed":              predict.Options.Seed,
		"stop":              predict.Options.Stop,
		"image_data":        imageData,
		"cache_prompt":      true,
	}

	if predict.Format == "json" {
		request["grammar"] = jsonGrammar
	}

	retryDelay := 100 * time.Microsecond
	for retries := 0; retries < maxRetries; retries++ {
		if retries > 0 {
			time.Sleep(retryDelay) // wait before retrying
			retryDelay *= 2        // exponential backoff
		}

		// Handling JSON marshaling with special characters unescaped.
		buffer := &bytes.Buffer{}
		enc := json.NewEncoder(buffer)
		enc.SetEscapeHTML(false)

		if err := enc.Encode(request); err != nil {
			return fmt.Errorf("failed to marshal data: %w", err)
		}

		req := C.CString(buffer.String())
		defer C.free(unsafe.Pointer(req))

		llm.llama_server_completion(req, &resp)
		if resp.id < 0 {
			return extServerResponseToErr(resp)
		}

		retryNeeded := false
	out:
		for {
			select {
			case <-ctx.Done():
				// This handles the request cancellation
				llm.llama_server_completion_cancel(resp.id, &resp)
				if resp.id < 0 {
					return extServerResponseToErr(resp)
				} else {
					return nil
				}
			default:
				var result C.ext_server_task_result_t
				llm.llama_server_completion_next_result(resp.id, &result)
				json_resp := C.GoString(result.json_resp)
				llm.llama_server_release_task_result(&result)

				var p prediction
				if err := json.Unmarshal([]byte(json_resp), &p); err != nil {
					llm.llama_server_completion_cancel(resp.id, &resp)
					if resp.id < 0 {
						return fmt.Errorf("error unmarshaling llm prediction response: %w and cancel %s", err, C.GoString(resp.msg))
					} else {
						return fmt.Errorf("error unmarshaling llm prediction response: %w", err)
					}
				}

				if bool(result.error) && strings.Contains(json_resp, "slot unavailable") {
					retryNeeded = true
					// task will already be canceled
					break out
				}

				if p.Content != "" {
					fn(PredictResult{
						Content: p.Content,
					})
				}

				if p.Stop {
					fn(PredictResult{
						Done:               true,
						PromptEvalCount:    p.Timings.PromptN,
						PromptEvalDuration: parseDurationMs(p.Timings.PromptMS),
						EvalCount:          p.Timings.PredictedN,
						EvalDuration:       parseDurationMs(p.Timings.PredictedMS),
					})
					return nil
				}
			}
		}
		if !retryNeeded {
			return nil // success
		}
	}

	// should never reach here ideally
	return fmt.Errorf("max retries exceeded")
}

func encode(llm extServer, ctx context.Context, prompt string) ([]int, error) {
	data, err := json.Marshal(TokenizeRequest{Content: prompt})
	if err != nil {
		return nil, fmt.Errorf("marshaling encode data: %w", err)
	}
	req := C.CString(string(data))
	defer C.free(unsafe.Pointer(req))
	var json_resp *C.char
	resp := newExtServerResp(128)
	defer freeExtServerResp(resp)
	llm.llama_server_tokenize(req, &json_resp, &resp)
	if resp.id < 0 {
		return nil, extServerResponseToErr(resp)
	}
	defer llm.llama_server_release_json_resp(&json_resp)

	var encoded TokenizeResponse
	if err2 := json.Unmarshal([]byte(C.GoString(json_resp)), &encoded); err2 != nil {
		return nil, fmt.Errorf("unmarshal encode response: %w", err2)
	}

	return encoded.Tokens, err
}

func decode(llm extServer, ctx context.Context, tokens []int) (string, error) {
	if len(tokens) == 0 {
		return "", nil
	}
	data, err := json.Marshal(DetokenizeRequest{Tokens: tokens})
	if err != nil {
		return "", fmt.Errorf("marshaling decode data: %w", err)
	}

	req := C.CString(string(data))
	defer C.free(unsafe.Pointer(req))
	var json_resp *C.char
	resp := newExtServerResp(128)
	defer freeExtServerResp(resp)
	llm.llama_server_detokenize(req, &json_resp, &resp)
	if resp.id < 0 {
		return "", extServerResponseToErr(resp)
	}
	defer llm.llama_server_release_json_resp(&json_resp)

	var decoded DetokenizeResponse
	if err2 := json.Unmarshal([]byte(C.GoString(json_resp)), &decoded); err2 != nil {
		return "", fmt.Errorf("unmarshal encode response: %w", err2)
	}

	return decoded.Content, err
}

func embedding(llm extServer, ctx context.Context, input string) ([]float64, error) {
	data, err := json.Marshal(TokenizeRequest{Content: input})
	if err != nil {
		return nil, fmt.Errorf("error marshaling embed data: %w", err)
	}

	req := C.CString(string(data))
	defer C.free(unsafe.Pointer(req))
	var json_resp *C.char
	resp := newExtServerResp(128)
	defer freeExtServerResp(resp)
	llm.llama_server_embedding(req, &json_resp, &resp)
	if resp.id < 0 {
		return nil, extServerResponseToErr(resp)
	}
	defer llm.llama_server_release_json_resp(&json_resp)

	var embedding EmbeddingResponse
	if err := json.Unmarshal([]byte(C.GoString(json_resp)), &embedding); err != nil {
		return nil, fmt.Errorf("unmarshal tokenize response: %w", err)
	}

	return embedding.Embedding, nil
}

func close(llm extServer) {
	llm.llama_server_stop()
	mutex.Unlock()
}

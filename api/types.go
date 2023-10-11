package api

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"reflect"
	"strings"
	"time"
)

type StatusError struct {
	StatusCode   int
	Status       string
	ErrorMessage string `json:"error"`
}

func (e StatusError) Error() string {
	switch {
	case e.Status != "" && e.ErrorMessage != "":
		return fmt.Sprintf("%s: %s", e.Status, e.ErrorMessage)
	case e.Status != "":
		return e.Status
	case e.ErrorMessage != "":
		return e.ErrorMessage
	default:
		// this should not happen
		return "something went wrong, please see the ollama server logs for details"
	}
}

type GenerateRequest struct {
	Model    string         `json:"model"`
	Prompt   string         `json:"prompt"`
	System   string         `json:"system"`
	Template string         `json:"template"`
	Context  []int          `json:"context,omitempty"`
	Args     map[string]any `json:"args,omitempty"`

	Options map[string]interface{} `json:"options"`
}

type EmbeddingRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`

	Options map[string]interface{} `json:"options"`
}

type EmbeddingResponse struct {
	Embedding []float64 `json:"embedding"`
}

type CreateRequest struct {
	Name string `json:"name"`
	Path string `json:"path"`
}

type DeleteRequest struct {
	Name string `json:"name"`
}

type ShowRequest struct {
	Name string `json:"name"`
}

type ShowResponse struct {
	License    string `json:"license,omitempty"`
	Modelfile  string `json:"modelfile,omitempty"`
	Parameters string `json:"parameters,omitempty"`
	Template   string `json:"template,omitempty"`
	System     string `json:"system,omitempty"`
}

type CopyRequest struct {
	Source      string `json:"source"`
	Destination string `json:"destination"`
}

type PullRequest struct {
	Name     string `json:"name"`
	Insecure bool   `json:"insecure,omitempty"`
}

type ProgressResponse struct {
	Status    string `json:"status"`
	Digest    string `json:"digest,omitempty"`
	Total     int64  `json:"total,omitempty"`
	Completed int64  `json:"completed,omitempty"`
}

type PushRequest struct {
	Name     string `json:"name"`
	Insecure bool   `json:"insecure,omitempty"`
}

type ListResponse struct {
	Models []ModelResponse `json:"models"`
}

type ModelResponse struct {
	Name       string    `json:"name"`
	ModifiedAt time.Time `json:"modified_at"`
	Size       int64     `json:"size"`
	Digest     string    `json:"digest"`
}

type TokenResponse struct {
	Token string `json:"token"`
}

type GenerateResponse struct {
	Model     string    `json:"model"`
	CreatedAt time.Time `json:"created_at"`
	Response  string    `json:"response,omitempty"`

	Done    bool  `json:"done"`
	Context []int `json:"context,omitempty"`

	TotalDuration      time.Duration `json:"total_duration,omitempty"`
	LoadDuration       time.Duration `json:"load_duration,omitempty"`
	PromptEvalCount    int           `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration time.Duration `json:"prompt_eval_duration,omitempty"`
	EvalCount          int           `json:"eval_count,omitempty"`
	EvalDuration       time.Duration `json:"eval_duration,omitempty"`
}

func (r *GenerateResponse) Summary() {
	if r.TotalDuration > 0 {
		fmt.Fprintf(os.Stderr, "total duration:       %v\n", r.TotalDuration)
	}

	if r.LoadDuration > 0 {
		fmt.Fprintf(os.Stderr, "load duration:        %v\n", r.LoadDuration)
	}

	if r.PromptEvalCount > 0 {
		fmt.Fprintf(os.Stderr, "prompt eval count:    %d token(s)\n", r.PromptEvalCount)
	}

	if r.PromptEvalDuration > 0 {
		fmt.Fprintf(os.Stderr, "prompt eval duration: %s\n", r.PromptEvalDuration)
		fmt.Fprintf(os.Stderr, "prompt eval rate:     %.2f tokens/s\n", float64(r.PromptEvalCount)/r.PromptEvalDuration.Seconds())
	}

	if r.EvalCount > 0 {
		fmt.Fprintf(os.Stderr, "eval count:           %d token(s)\n", r.EvalCount)
	}

	if r.EvalDuration > 0 {
		fmt.Fprintf(os.Stderr, "eval duration:        %s\n", r.EvalDuration)
		fmt.Fprintf(os.Stderr, "eval rate:            %.2f tokens/s\n", float64(r.EvalCount)/r.EvalDuration.Seconds())
	}
}

type Options struct {
	Seed int `json:"seed,omitempty"`

	// Backend options
	UseNUMA bool `json:"numa,omitempty"`

	// Model options
	NumCtx             int     `json:"num_ctx,omitempty"`
	NumKeep            int     `json:"num_keep,omitempty"`
	NumBatch           int     `json:"num_batch,omitempty"`
	NumGQA             int     `json:"num_gqa,omitempty"`
	NumGPU             int     `json:"num_gpu,omitempty"`
	MainGPU            int     `json:"main_gpu,omitempty"`
	LowVRAM            bool    `json:"low_vram,omitempty"`
	F16KV              bool    `json:"f16_kv,omitempty"`
	LogitsAll          bool    `json:"logits_all,omitempty"`
	VocabOnly          bool    `json:"vocab_only,omitempty"`
	UseMMap            bool    `json:"use_mmap,omitempty"`
	UseMLock           bool    `json:"use_mlock,omitempty"`
	EmbeddingOnly      bool    `json:"embedding_only,omitempty"`
	RopeFrequencyBase  float32 `json:"rope_frequency_base,omitempty"`
	RopeFrequencyScale float32 `json:"rope_frequency_scale,omitempty"`

	// Predict options
	NumPredict       int      `json:"num_predict,omitempty"`
	TopK             int      `json:"top_k,omitempty"`
	TopP             float32  `json:"top_p,omitempty"`
	TFSZ             float32  `json:"tfs_z,omitempty"`
	TypicalP         float32  `json:"typical_p,omitempty"`
	RepeatLastN      int      `json:"repeat_last_n,omitempty"`
	Temperature      float32  `json:"temperature,omitempty"`
	RepeatPenalty    float32  `json:"repeat_penalty,omitempty"`
	PresencePenalty  float32  `json:"presence_penalty,omitempty"`
	FrequencyPenalty float32  `json:"frequency_penalty,omitempty"`
	Mirostat         int      `json:"mirostat,omitempty"`
	MirostatTau      float32  `json:"mirostat_tau,omitempty"`
	MirostatEta      float32  `json:"mirostat_eta,omitempty"`
	PenalizeNewline  bool     `json:"penalize_newline,omitempty"`
	Stop             []string `json:"stop,omitempty"`

	NumThread int `json:"num_thread,omitempty"`
}

func (opts *Options) FromMap(m map[string]interface{}) error {
	valueOpts := reflect.ValueOf(opts).Elem() // names of the fields in the options struct
	typeOpts := reflect.TypeOf(opts).Elem()   // types of the fields in the options struct

	// build map of json struct tags to their types
	jsonOpts := make(map[string]reflect.StructField)
	for _, field := range reflect.VisibleFields(typeOpts) {
		jsonTag := strings.Split(field.Tag.Get("json"), ",")[0]
		if jsonTag != "" {
			jsonOpts[jsonTag] = field
		}
	}

	for key, val := range m {
		if opt, ok := jsonOpts[key]; ok {
			field := valueOpts.FieldByName(opt.Name)
			if field.IsValid() && field.CanSet() {
				if val == nil {
					continue
				}

				switch field.Kind() {
				case reflect.Int:
					switch t := val.(type) {
					case int64:
						field.SetInt(t)
					case float64:
						// when JSON unmarshals numbers, it uses float64, not int
						field.SetInt(int64(t))
					default:
						log.Printf("could not convert model parameter %v of type %T to int, skipped", key, val)
					}
				case reflect.Bool:
					val, ok := val.(bool)
					if !ok {
						log.Printf("could not convert model parameter %v of type %T to bool, skipped", key, val)
						continue
					}
					field.SetBool(val)
				case reflect.Float32:
					// JSON unmarshals to float64
					val, ok := val.(float64)
					if !ok {
						log.Printf("could not convert model parameter %v of type %T to float32, skipped", key, val)
						continue
					}
					field.SetFloat(val)
				case reflect.String:
					val, ok := val.(string)
					if !ok {
						log.Printf("could not convert model parameter %v of type %T to string, skipped", key, val)
						continue
					}
					field.SetString(val)
				case reflect.Slice:
					// JSON unmarshals to []interface{}, not []string
					val, ok := val.([]interface{})
					if !ok {
						log.Printf("could not convert model parameter %v of type %T to slice, skipped", key, val)
						continue
					}
					// convert []interface{} to []string
					slice := make([]string, len(val))
					for i, item := range val {
						str, ok := item.(string)
						if !ok {
							log.Printf("could not convert model parameter %v of type %T to slice of strings, skipped", key, item)
							continue
						}
						slice[i] = str
					}
					field.Set(reflect.ValueOf(slice))
				default:
					return fmt.Errorf("unknown type loading config params: %v", field.Kind())
				}
			}
		}
	}
	return nil
}

func DefaultOptions() Options {
	return Options{
		// options set on request to runner
		NumPredict:       -1,
		NumKeep:          -1,
		Temperature:      0.8,
		TopK:             40,
		TopP:             0.9,
		TFSZ:             1.0,
		TypicalP:         1.0,
		RepeatLastN:      64,
		RepeatPenalty:    1.1,
		PresencePenalty:  0.0,
		FrequencyPenalty: 0.0,
		Mirostat:         0,
		MirostatTau:      5.0,
		MirostatEta:      0.1,
		PenalizeNewline:  true,
		Seed:             -1,

		// options set when the model is loaded
		NumCtx:             2048,
		RopeFrequencyBase:  10000.0,
		RopeFrequencyScale: 1.0,
		NumBatch:           512,
		NumGPU:             -1, // -1 here indicates that NumGPU should be set dynamically
		NumGQA:             1,
		NumThread:          0, // let the runtime decide
		LowVRAM:            false,
		F16KV:              true,
		UseMLock:           false,
		UseMMap:            true,
		UseNUMA:            false,
		EmbeddingOnly:      true,
	}
}

type Duration struct {
	time.Duration
}

func (d *Duration) UnmarshalJSON(b []byte) (err error) {
	var v any
	if err := json.Unmarshal(b, &v); err != nil {
		return err
	}

	d.Duration = 5 * time.Minute

	switch t := v.(type) {
	case float64:
		if t < 0 {
			t = math.MaxFloat64
		}

		d.Duration = time.Duration(t)
	case string:
		d.Duration, err = time.ParseDuration(t)
		if err != nil {
			return err
		}
	}

	return nil
}

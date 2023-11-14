package server

import (
	"context"
	"errors"
	"os"
	"path"
	"runtime"
	"testing"
	"time"

	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/llm"
	"github.com/stretchr/testify/require"
)

func SkipIFNoTestData(t *testing.T) {
	modelDir := getModelDir()
	if _, err := os.Stat(modelDir); errors.Is(err, os.ErrNotExist) {
		t.Skipf("%s does not exist - skipping integration tests", modelDir)
	}
}

func getModelDir() string {
	_, filename, _, _ := runtime.Caller(0)
	return path.Dir(path.Dir(filename) + "/../test_data/models/.")
}

func PrepareModelForPrompts(t *testing.T, modelName string, opts api.Options) (*Model, llm.LLM) {
	modelDir := getModelDir()
	os.Setenv("OLLAMA_MODELS", modelDir)
	model, err := GetModel(modelName)
	require.NoError(t, err, "GetModel ")
	err = opts.FromMap(model.Options)
	require.NoError(t, err, "opts from model ")
	runner, err := llm.New("unused", model.ModelPath, model.AdapterPaths, model.ProjectorPaths, opts)
	require.NoError(t, err, "llm.New failed")
	return model, runner
}

func OneShotPromptResponse(t *testing.T, ctx context.Context, req api.GenerateRequest, model *Model, runner llm.LLM) string {
	checkpointStart := time.Now()
	prompt, err := model.Prompt(PromptVars{
		System: req.System,
		Prompt: req.Prompt,
		First:  len(req.Context) == 0,
	})
	require.NoError(t, err, "prompt generation failed")
	success := make(chan bool, 1)
	response := ""
	cb := func(r llm.PredictResult) {

		if !r.Done {
			response += r.Content
		} else {
			success <- true
		}
	}
	checkpointLoaded := time.Now()
	predictReq := llm.PredictOpts{
		Prompt:           prompt,
		Format:           req.Format,
		CheckpointStart:  checkpointStart,
		CheckpointLoaded: checkpointLoaded,
	}
	err = runner.Predict(ctx, predictReq, cb)
	require.NoError(t, err, "predict call failed")

	select {
	case <-ctx.Done():
		t.Errorf("failed to complete before timeout: \n%s", response)
		return ""
	case <-success:
		return response
	}
}

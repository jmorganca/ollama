package llm

import (
	"embed"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log"
	"os"
	"path/filepath"

	"github.com/jmorganca/ollama/api"
)

//go:embed llama.cpp/ggml-metal.metal
var libEmbed embed.FS

func newDynamicShimExtServer(library, model string, adapters, projectors []string, opts api.Options) (extServer, error) {
	// should never happen...
	return nil, fmt.Errorf("Dynamic library loading not supported on Mac")
}

func nativeInit(workdir string) error {
	err := extractPayloadFiles(workdir, "llama.cpp/ggml-metal.metal")
	if err != nil {
		if err == payloadMissing {
			// TODO perhaps consider this a hard failure on arm macs?
			log.Printf("ggml-meta.metal payload missing")
			return nil
		}
		return err
	}
	os.Setenv("GGML_METAL_PATH_RESOURCES", workdir)
	return nil
}

func extractPayloadFiles(workDir, glob string) error {
	files, err := fs.Glob(libEmbed, glob)
	if err != nil || len(files) == 0 {
		return payloadMissing
	}

	for _, file := range files {
		srcFile, err := libEmbed.Open(file)
		if err != nil {
			return fmt.Errorf("read payload %s: %v", file, err)
		}
		defer srcFile.Close()
		if err := os.MkdirAll(workDir, 0o755); err != nil {
			return fmt.Errorf("create payload temp dir %s: %v", workDir, err)
		}

		destFile := filepath.Join(workDir, filepath.Base(file))
		_, err = os.Stat(destFile)
		switch {
		case errors.Is(err, os.ErrNotExist):
			destFile, err := os.OpenFile(destFile, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o755)
			if err != nil {
				return fmt.Errorf("write payload %s: %v", file, err)
			}
			defer destFile.Close()
			if _, err := io.Copy(destFile, srcFile); err != nil {
				return fmt.Errorf("copy payload %s: %v", file, err)
			}
		case err != nil:
			return fmt.Errorf("stat payload %s: %v", file, err)
		}
	}
	return nil
}

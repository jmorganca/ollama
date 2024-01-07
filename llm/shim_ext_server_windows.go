package llm

import (
	"embed"
	"log"
	"os"
	"path/filepath"
	"strings"
)

//go:embed llama.cpp/build/windows/*/lib/*.dll
var libEmbed embed.FS

func updatePath(dir string) {
	tmpDir := filepath.Dir(dir)
	pathComponents := strings.Split(os.Getenv("PATH"), ";")
	i := 0
	for _, comp := range pathComponents {
		if strings.EqualFold(comp, dir) {
			return
		}
		// Remove any other prior paths to our temp dir
		if !strings.HasPrefix(strings.ToLower(comp), strings.ToLower(tmpDir)) {
			pathComponents[i] = comp
			i++
		}
	}
	newPath := strings.Join(append([]string{dir}, pathComponents...), ";")
	log.Printf("Updating PATH to %s", newPath)
	os.Setenv("PATH", newPath)
}

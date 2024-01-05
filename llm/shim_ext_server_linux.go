package llm

import (
	"embed"
	"log"
	"os"
	"strings"
)

//go:embed llama.cpp/build/*/*/lib/*.so
var libEmbed embed.FS

func updatePath(dir string) {
	pathComponents := strings.Split(os.Getenv("PATH"), ":")
	for _, comp := range pathComponents {
		if comp == dir {
			return
		}
	}
	newPath := strings.Join(append(pathComponents, dir), ":")
	log.Printf("Updating PATH to %s", newPath)
	os.Setenv("PATH", newPath)
}

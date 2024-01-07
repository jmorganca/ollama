package server

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
)

func WriteManifest(name string, config *Layer, layers []*Layer) error {
	manifest := ManifestV2{
		SchemaVersion: 2,
		MediaType:     "application/vnd.docker.distribution.manifest.v2+json",
		Config:        config,
		Layers:        layers,
	}

	var b bytes.Buffer
	if err := json.NewEncoder(&b).Encode(manifest); err != nil {
		return err
	}

	modelpath := ParseModelPath(name)
	manifestPath, err := modelpath.GetManifestPath()
	if err != nil {
		return err
	}

	if err := os.MkdirAll(filepath.Dir(manifestPath), 0o755); err != nil {
		return err
	}

	return os.WriteFile(manifestPath, b.Bytes(), 0o644)
}

package server

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"text/template"

	"golang.org/x/exp/slices"

	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/llm"
	"github.com/jmorganca/ollama/parser"
	"github.com/jmorganca/ollama/version"
)

type RegistryOptions struct {
	Insecure bool
	Username string
	Password string
	Token    string
}

type Model struct {
	Name           string `json:"name"`
	Config         ConfigV2
	ShortName      string
	ModelPath      string
	OriginalModel  string
	AdapterPaths   []string
	ProjectorPaths []string
	Template       string
	System         string
	License        []string
	Digest         string
	Size           int64
	Options        map[string]interface{}
}

type PromptVars struct {
	System   string
	Prompt   string
	Response string
	First    bool
}

func (m *Model) Prompt(p PromptVars) (string, error) {
	var prompt strings.Builder
	// Use the "missingkey=zero" option to handle missing variables without panicking
	tmpl, err := template.New("").Option("missingkey=zero").Parse(m.Template)
	if err != nil {
		return "", err
	}

	if p.System == "" {
		// use the default system message for this model if one is not specified
		p.System = m.System
	}

	vars := map[string]any{
		"System":   p.System,
		"Prompt":   p.Prompt,
		"Response": p.Response,
		"First":    p.First,
	}

	var sb strings.Builder
	if err := tmpl.Execute(&sb, vars); err != nil {
		return "", err
	}
	prompt.WriteString(sb.String())
	prompt.WriteString(p.Response)
	return prompt.String(), nil
}

func (m *Model) ChatPrompt(msgs []api.Message) (string, []api.ImageData, error) {
	// build the prompt from the list of messages
	var prompt strings.Builder
	var currentImages []api.ImageData
	currentVars := PromptVars{
		First: true,
	}

	writePrompt := func() error {
		p, err := m.Prompt(currentVars)
		if err != nil {
			return err
		}
		prompt.WriteString(p)
		currentVars = PromptVars{}
		return nil
	}

	for _, msg := range msgs {
		switch strings.ToLower(msg.Role) {
		case "system":
			if currentVars.System != "" {
				if err := writePrompt(); err != nil {
					return "", nil, err
				}
			}
			currentVars.System = msg.Content
		case "user":
			if currentVars.Prompt != "" {
				if err := writePrompt(); err != nil {
					return "", nil, err
				}
			}
			currentVars.Prompt = msg.Content
			currentImages = msg.Images
		case "assistant":
			currentVars.Response = msg.Content
			if err := writePrompt(); err != nil {
				return "", nil, err
			}
		default:
			return "", nil, fmt.Errorf("invalid role: %s, role must be one of [system, user, assistant]", msg.Role)
		}
	}

	// Append the last set of vars if they are non-empty
	if currentVars.Prompt != "" || currentVars.System != "" {
		if err := writePrompt(); err != nil {
			return "", nil, err
		}
	}

	return prompt.String(), currentImages, nil
}

type ManifestV2 struct {
	SchemaVersion int      `json:"schemaVersion"`
	MediaType     string   `json:"mediaType"`
	Config        *Layer   `json:"config"`
	Layers        []*Layer `json:"layers"`
}

type ConfigV2 struct {
	ModelFormat   string   `json:"model_format"`
	ModelFamily   string   `json:"model_family"`
	ModelFamilies []string `json:"model_families"`
	ModelType     string   `json:"model_type"`
	FileType      string   `json:"file_type"`

	// required by spec
	Architecture string `json:"architecture"`
	OS           string `json:"os"`
	RootFS       RootFS `json:"rootfs"`
}

func (c *ConfigV2) SetModelFormat(format string) {
	if c.ModelFormat == "" {
		c.ModelFormat = format
	}
}

func (c *ConfigV2) SetModelFamily(families ...string) {
	for _, family := range families {
		if c.ModelFamily == "" {
			c.ModelFamily = family
		}

		if !slices.Contains(c.ModelFamilies, family) {
			c.ModelFamilies = append(c.ModelFamilies, family)
		}
	}
}

func (c *ConfigV2) SetModelType(modelType string) {
	if c.ModelType == "" {
		c.ModelType = modelType
	}
}

func (c *ConfigV2) SetFileType(fileType string) {
	if c.FileType == "" {
		c.FileType = fileType
	}
}

type RootFS struct {
	Type    string   `json:"type"`
	DiffIDs []string `json:"diff_ids"`
}

func (m *ManifestV2) GetTotalSize() (total int64) {
	for _, layer := range m.Layers {
		total += layer.Size
	}

	total += m.Config.Size
	return total
}

func GetManifest(mp ModelPath) (*ManifestV2, string, error) {
	fp, err := mp.GetManifestPath()
	if err != nil {
		return nil, "", err
	}

	if _, err = os.Stat(fp); err != nil {
		return nil, "", err
	}

	var manifest *ManifestV2

	bts, err := os.ReadFile(fp)
	if err != nil {
		return nil, "", fmt.Errorf("couldn't open file '%s'", fp)
	}

	shaSum := sha256.Sum256(bts)
	shaStr := hex.EncodeToString(shaSum[:])

	if err := json.Unmarshal(bts, &manifest); err != nil {
		return nil, "", err
	}

	return manifest, shaStr, nil
}

func GetModel(name string) (*Model, error) {
	mp := ParseModelPath(name)
	manifest, digest, err := GetManifest(mp)
	if err != nil {
		return nil, err
	}

	model := &Model{
		Name:      mp.GetFullTagname(),
		ShortName: mp.GetShortTagname(),
		Digest:    digest,
		Template:  "{{ .Prompt }}",
		License:   []string{},
		Size:      manifest.GetTotalSize(),
	}

	filename, err := GetBlobsPath(manifest.Config.Digest)
	if err != nil {
		return nil, err
	}

	configFile, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer configFile.Close()

	if err := json.NewDecoder(configFile).Decode(&model.Config); err != nil {
		return nil, err
	}

	for _, layer := range manifest.Layers {
		filename, err := GetBlobsPath(layer.Digest)
		if err != nil {
			return nil, err
		}

		switch layer.MediaType {
		case "application/vnd.ollama.image.model":
			model.ModelPath = filename
			model.OriginalModel = layer.From
		case "application/vnd.ollama.image.embed":
			// Deprecated in versions  > 0.1.2
			// TODO: remove this warning in a future version
			log.Print("WARNING: model contains embeddings, but embeddings in modelfiles have been deprecated and will be ignored.")
		case "application/vnd.ollama.image.adapter":
			model.AdapterPaths = append(model.AdapterPaths, filename)
		case "application/vnd.ollama.image.projector":
			model.ProjectorPaths = append(model.ProjectorPaths, filename)
		case "application/vnd.ollama.image.template":
			bts, err := os.ReadFile(filename)
			if err != nil {
				return nil, err
			}

			model.Template = string(bts)
		case "application/vnd.ollama.image.system":
			bts, err := os.ReadFile(filename)
			if err != nil {
				return nil, err
			}

			model.System = string(bts)
		case "application/vnd.ollama.image.prompt":
			bts, err := os.ReadFile(filename)
			if err != nil {
				return nil, err
			}

			model.Template = string(bts)
		case "application/vnd.ollama.image.params":
			params, err := os.Open(filename)
			if err != nil {
				return nil, err
			}
			defer params.Close()

			// parse model options parameters into a map so that we can see which fields have been specified explicitly
			if err = json.NewDecoder(params).Decode(&model.Options); err != nil {
				return nil, err
			}
		case "application/vnd.ollama.image.license":
			bts, err := os.ReadFile(filename)
			if err != nil {
				return nil, err
			}
			model.License = append(model.License, string(bts))
		}
	}

	return model, nil
}

func realpath(mfDir, from string) string {
	abspath, err := filepath.Abs(from)
	if err != nil {
		return from
	}

	home, err := os.UserHomeDir()
	if err != nil {
		return abspath
	}

	if from == "~" {
		return home
	} else if strings.HasPrefix(from, "~/") {
		return filepath.Join(home, from[2:])
	}

	if _, err := os.Stat(filepath.Join(mfDir, from)); err == nil {
		// this is a file relative to the Modelfile
		return filepath.Join(mfDir, from)
	}

	return abspath
}

func CreateModel(ctx context.Context, name, modelFileDir string, commands []parser.Command, fn func(resp api.ProgressResponse)) error {
	config := ConfigV2{
		OS:           "linux",
		Architecture: "amd64",
		RootFS: RootFS{
			Type: "layers",
		},
	}

	deleteMap := make(map[string]struct{})

	var layers Layers

	params := make(map[string][]string)
	fromParams := make(map[string]any)

	for _, c := range commands {
		log.Printf("[%s] - %s", c.Name, c.Args)
		mediatype := fmt.Sprintf("application/vnd.ollama.image.%s", c.Name)

		switch c.Name {
		case "model":
			if strings.HasPrefix(c.Args, "@") {
				blobPath, err := GetBlobsPath(strings.TrimPrefix(c.Args, "@"))
				if err != nil {
					return err
				}

				c.Args = blobPath
			}

			bin, err := os.Open(realpath(modelFileDir, c.Args))
			if err != nil {
				// not a file on disk so must be a model reference
				modelpath := ParseModelPath(c.Args)
				manifest, _, err := GetManifest(modelpath)
				switch {
				case errors.Is(err, os.ErrNotExist):
					fn(api.ProgressResponse{Status: "pulling model"})
					if err := PullModel(ctx, c.Args, &RegistryOptions{}, fn); err != nil {
						return err
					}

					manifest, _, err = GetManifest(modelpath)
					if err != nil {
						return err
					}
				case err != nil:
					return err
				}

				fn(api.ProgressResponse{Status: "reading model metadata"})
				fromConfigPath, err := GetBlobsPath(manifest.Config.Digest)
				if err != nil {
					return err
				}

				fromConfigFile, err := os.Open(fromConfigPath)
				if err != nil {
					return err
				}
				defer fromConfigFile.Close()

				var fromConfig ConfigV2
				if err := json.NewDecoder(fromConfigFile).Decode(&fromConfig); err != nil {
					return err
				}

				// if the model is not in gguf format, pull the base model to try and get it in gguf format
				if fromConfig.ModelFormat != "gguf" {
					fn(api.ProgressResponse{Status: "updating base model"})
					parent, err := GetModel(c.Args)
					if err != nil {
						return err
					}
					if err := PullModel(ctx, parent.OriginalModel, &RegistryOptions{}, fn); err != nil {
						log.Printf("error pulling model: %v", err)
					}
					// Reset the file pointer to the beginning of the file
					_, err = fromConfigFile.Seek(0, 0)
					if err != nil {
						return fmt.Errorf("update from config after pull: %w", err)
					}
					if err := json.NewDecoder(fromConfigFile).Decode(&fromConfig); err != nil {
						return err
					}
				}

				// if the model is still not in gguf format, error out
				if fromConfig.ModelFormat != "gguf" {
					return fmt.Errorf("%s is not in gguf format, this base model is not compatible with this version of ollama", c.Args)
				}

				config.SetModelFormat(fromConfig.ModelFormat)
				config.SetModelFamily(append(fromConfig.ModelFamilies, fromConfig.ModelFamily)...)
				config.SetModelType(fromConfig.ModelType)
				config.SetFileType(fromConfig.FileType)

				for _, layer := range manifest.Layers {
					deleteMap[layer.Digest] = struct{}{}
					if layer.MediaType == "application/vnd.ollama.image.params" {
						fromParamsPath, err := GetBlobsPath(layer.Digest)
						if err != nil {
							return err
						}

						fromParamsFile, err := os.Open(fromParamsPath)
						if err != nil {
							return err
						}
						defer fromParamsFile.Close()

						if err := json.NewDecoder(fromParamsFile).Decode(&fromParams); err != nil {
							return err
						}
					}

					layer, err := NewLayerFromLayer(layer.Digest, layer.MediaType, modelpath.GetShortTagname())
					if err != nil {
						return err
					}

					layers.Add(layer)
				}

				deleteMap[manifest.Config.Digest] = struct{}{}
				continue
			}
			defer bin.Close()

			var offset int64
		CREATE:
			for {
				fn(api.ProgressResponse{Status: "creating model layer"})

				bin.Seek(offset, io.SeekStart)
				ggml, err := llm.DecodeGGML(bin)
				if err != nil {
					switch {
					case errors.Is(err, io.EOF):
						break CREATE
					case errors.Is(err, llm.ErrUnsupportedFormat):
						return fmt.Errorf("model binary specified in FROM field is not a valid gguf format model, %w", err)
					default:
						return err
					}
				}

				config.SetModelFormat(ggml.Name())
				config.SetModelFamily(ggml.ModelFamily())
				config.SetModelType(ggml.ModelType())
				config.SetFileType(ggml.FileType())

				mediatype := mediatype
				if ggml.ModelFamily() == "clip" {
					mediatype = "application/vnd.ollama.image.projector"
				}

				sr := io.NewSectionReader(bin, offset, ggml.Size)
				layer, err := NewLayer(sr, mediatype)
				if err != nil {
					return err
				}

				layers.Add(layer)

				offset += ggml.Size
			}
		case "adapter":
			if strings.HasPrefix(c.Args, "@") {
				blobPath, err := GetBlobsPath(strings.TrimPrefix(c.Args, "@"))
				if err != nil {
					return err
				}

				c.Args = blobPath
			}

			fn(api.ProgressResponse{Status: "creating adapter layer"})
			bin, err := os.Open(realpath(modelFileDir, c.Args))
			if err != nil {
				return err
			}
			defer bin.Close()

			layer, err := NewLayer(bin, mediatype)
			if err != nil {
				return err
			}

			layers.Add(layer)
		case "license":
			fn(api.ProgressResponse{Status: "creating license layer"})

			bin := strings.NewReader(c.Args)
			layer, err := NewLayer(bin, mediatype)
			if err != nil {
				return err
			}

			layers.Add(layer)
		case "template", "system":
			fn(api.ProgressResponse{Status: fmt.Sprintf("creating %s layer", c.Name)})

			bin := strings.NewReader(c.Args)
			layer, err := NewLayer(bin, mediatype)
			if err != nil {
				return err
			}

			layers.Replace(layer)
		default:
			params[c.Name] = append(params[c.Name], c.Args)
		}
	}

	if len(params) > 0 {
		fn(api.ProgressResponse{Status: "creating parameters layer"})

		formattedParams, err := api.FormatParams(params)
		if err != nil {
			return err
		}

		for k, v := range fromParams {
			if _, ok := formattedParams[k]; !ok {
				formattedParams[k] = v
			}
		}

		// xxx - can this be removed?
		if config.ModelType == "65B" {
			if gqa, ok := formattedParams["gqa"].(int); ok && gqa == 8 {
				config.ModelType = "70B"
			}
		}

		var b bytes.Buffer
		if err := json.NewEncoder(&b).Encode(formattedParams); err != nil {
			return err
		}

		fn(api.ProgressResponse{Status: "creating config layer"})
		layer, err := NewLayer(&b, "application/vnd.ollama.image.params")
		if err != nil {
			return err
		}

		layers.Replace(layer)
	}

	digests := make([]string, len(layers.items))
	for i, layer := range layers.items {
		digests[i] = layer.Digest
	}

	config.RootFS.DiffIDs = digests

	var b bytes.Buffer
	if err := json.NewEncoder(&b).Encode(config); err != nil {
		return err
	}

	configLayer, err := NewLayer(&b, "application/vnd.docker.container.image.v1+json")
	if err != nil {
		return err
	}

	delete(deleteMap, configLayer.Digest)

	for _, layer := range append(layers.items, configLayer) {
		committed, err := layer.Commit()
		if err != nil {
			return err
		}

		status := "writing layer"
		if !committed {
			status = "using already created layer"
		}

		fn(api.ProgressResponse{Status: fmt.Sprintf("%s %s", status, layer.Digest)})

		delete(deleteMap, layer.Digest)
	}

	fn(api.ProgressResponse{Status: "writing manifest"})
	if err := WriteManifest(name, configLayer, layers.items); err != nil {
		return err
	}

	if noprune := os.Getenv("OLLAMA_NOPRUNE"); noprune == "" {
		if err := deleteUnusedLayers(nil, deleteMap, false); err != nil {
			return err
		}
	}

	fn(api.ProgressResponse{Status: "success"})
	return nil
}

func CopyModel(src, dest string) error {
	srcModelPath := ParseModelPath(src)
	srcPath, err := srcModelPath.GetManifestPath()
	if err != nil {
		return err
	}

	destModelPath := ParseModelPath(dest)
	destPath, err := destModelPath.GetManifestPath()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(destPath), 0o755); err != nil {
		return err
	}

	// copy the file
	input, err := os.ReadFile(srcPath)
	if err != nil {
		fmt.Println("Error reading file:", err)
		return err
	}

	err = os.WriteFile(destPath, input, 0o644)
	if err != nil {
		fmt.Println("Error reading file:", err)
		return err
	}

	return nil
}

func deleteUnusedLayers(skipModelPath *ModelPath, deleteMap map[string]struct{}, dryRun bool) error {
	fp, err := GetManifestPath()
	if err != nil {
		return err
	}

	walkFunc := func(path string, info os.FileInfo, _ error) error {
		if info.IsDir() {
			return nil
		}

		dir, file := filepath.Split(path)
		dir = strings.Trim(strings.TrimPrefix(dir, fp), string(os.PathSeparator))
		tag := strings.Join([]string{dir, file}, ":")
		fmp := ParseModelPath(tag)

		// skip the manifest we're trying to delete
		if skipModelPath != nil && skipModelPath.GetFullTagname() == fmp.GetFullTagname() {
			return nil
		}

		// save (i.e. delete from the deleteMap) any files used in other manifests
		manifest, _, err := GetManifest(fmp)
		if err != nil {
			return nil
		}

		for _, layer := range manifest.Layers {
			delete(deleteMap, layer.Digest)
		}

		delete(deleteMap, manifest.Config.Digest)
		return nil
	}

	if err := filepath.Walk(fp, walkFunc); err != nil {
		return err
	}

	// only delete the files which are still in the deleteMap
	for k := range deleteMap {
		fp, err := GetBlobsPath(k)
		if err != nil {
			log.Printf("couldn't get file path for '%s': %v", k, err)
			continue
		}
		if !dryRun {
			if err := os.Remove(fp); err != nil {
				log.Printf("couldn't remove file '%s': %v", fp, err)
				continue
			}
		} else {
			log.Printf("wanted to remove: %s", fp)
		}
	}

	return nil
}

func PruneLayers() error {
	deleteMap := make(map[string]struct{})
	p, err := GetBlobsPath("")
	if err != nil {
		return err
	}

	blobs, err := os.ReadDir(p)
	if err != nil {
		log.Printf("couldn't read dir '%s': %v", p, err)
		return err
	}

	for _, blob := range blobs {
		name := blob.Name()
		if runtime.GOOS == "windows" {
			name = strings.ReplaceAll(name, "-", ":")
		}
		if strings.HasPrefix(name, "sha256:") {
			deleteMap[name] = struct{}{}
		}
	}

	log.Printf("total blobs: %d", len(deleteMap))

	err = deleteUnusedLayers(nil, deleteMap, false)
	if err != nil {
		return err
	}

	log.Printf("total unused blobs removed: %d", len(deleteMap))

	return nil
}

func PruneDirectory(path string) error {
	info, err := os.Lstat(path)
	if err != nil {
		return err
	}

	if info.IsDir() && info.Mode()&os.ModeSymlink == 0 {
		entries, err := os.ReadDir(path)
		if err != nil {
			return err
		}

		for _, entry := range entries {
			if err := PruneDirectory(filepath.Join(path, entry.Name())); err != nil {
				return err
			}
		}

		entries, err = os.ReadDir(path)
		if err != nil {
			return err
		}

		if len(entries) > 0 {
			return nil
		}

		return os.Remove(path)
	}

	return nil
}

func DeleteModel(name string) error {
	mp := ParseModelPath(name)
	manifest, _, err := GetManifest(mp)
	if err != nil {
		return err
	}

	deleteMap := make(map[string]struct{})
	for _, layer := range manifest.Layers {
		deleteMap[layer.Digest] = struct{}{}
	}
	deleteMap[manifest.Config.Digest] = struct{}{}

	err = deleteUnusedLayers(&mp, deleteMap, false)
	if err != nil {
		return err
	}

	fp, err := mp.GetManifestPath()
	if err != nil {
		return err
	}
	err = os.Remove(fp)
	if err != nil {
		log.Printf("couldn't remove manifest file '%s': %v", fp, err)
		return err
	}

	return nil
}

func ShowModelfile(model *Model) (string, error) {
	var mt struct {
		*Model
		From       string
		Parameters map[string][]any
	}

	mt.Parameters = make(map[string][]any)
	for k, v := range model.Options {
		if s, ok := v.([]any); ok {
			mt.Parameters[k] = s
			continue
		}

		mt.Parameters[k] = []any{v}
	}

	mt.Model = model
	mt.From = model.ModelPath

	if model.OriginalModel != "" {
		mt.From = model.OriginalModel
	}

	modelFile := `# Modelfile generated by "ollama show"
# To build a new Modelfile based on this one, replace the FROM line with:
# FROM {{ .ShortName }}

FROM {{ .From }}
TEMPLATE """{{ .Template }}"""

{{- if .System }}
SYSTEM """{{ .System }}"""
{{- end }}

{{- range $adapter := .AdapterPaths }}
ADAPTER {{ $adapter }}
{{- end }}

{{- range $k, $v := .Parameters }}
{{- range $parameter := $v }}
PARAMETER {{ $k }} {{ printf "%#v" $parameter }}
{{- end }}
{{- end }}`

	tmpl, err := template.New("").Parse(modelFile)
	if err != nil {
		log.Printf("error parsing template: %q", err)
		return "", err
	}

	var buf bytes.Buffer

	if err = tmpl.Execute(&buf, mt); err != nil {
		log.Printf("error executing template: %q", err)
		return "", err
	}

	return buf.String(), nil
}

func PushModel(ctx context.Context, name string, regOpts *RegistryOptions, fn func(api.ProgressResponse)) error {
	mp := ParseModelPath(name)
	fn(api.ProgressResponse{Status: "retrieving manifest"})

	if mp.ProtocolScheme == "http" && !regOpts.Insecure {
		return fmt.Errorf("insecure protocol http")
	}

	manifest, _, err := GetManifest(mp)
	if err != nil {
		fn(api.ProgressResponse{Status: "couldn't retrieve manifest"})
		return err
	}

	var layers []*Layer
	layers = append(layers, manifest.Layers...)
	layers = append(layers, manifest.Config)

	for _, layer := range layers {
		if err := uploadBlob(ctx, mp, layer, regOpts, fn); err != nil {
			log.Printf("error uploading blob: %v", err)
			if errors.Is(err, errUnauthorized) {
				return fmt.Errorf("unable to push %s, make sure this namespace exists and you are authorized to push to it", ParseModelPath(name).GetNamespaceRepository())
			}
			return err
		}
	}

	fn(api.ProgressResponse{Status: "pushing manifest"})
	requestURL := mp.BaseURL()
	requestURL = requestURL.JoinPath("v2", mp.GetNamespaceRepository(), "manifests", mp.Tag)

	manifestJSON, err := json.Marshal(manifest)
	if err != nil {
		return err
	}

	headers := make(http.Header)
	headers.Set("Content-Type", "application/vnd.docker.distribution.manifest.v2+json")
	resp, err := makeRequestWithRetry(ctx, http.MethodPut, requestURL, headers, bytes.NewReader(manifestJSON), regOpts)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	fn(api.ProgressResponse{Status: "success"})

	return nil
}

func PullModel(ctx context.Context, name string, regOpts *RegistryOptions, fn func(api.ProgressResponse)) error {
	mp := ParseModelPath(name)

	var manifest *ManifestV2
	var err error
	var noprune string

	// build deleteMap to prune unused layers
	deleteMap := make(map[string]struct{})

	if noprune = os.Getenv("OLLAMA_NOPRUNE"); noprune == "" {
		manifest, _, err = GetManifest(mp)
		if err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}

		if manifest != nil {
			for _, l := range manifest.Layers {
				deleteMap[l.Digest] = struct{}{}
			}
			deleteMap[manifest.Config.Digest] = struct{}{}
		}
	}

	if mp.ProtocolScheme == "http" && !regOpts.Insecure {
		return fmt.Errorf("insecure protocol http")
	}

	fn(api.ProgressResponse{Status: "pulling manifest"})

	manifest, err = pullModelManifest(ctx, mp, regOpts)
	if err != nil {
		return fmt.Errorf("pull model manifest: %s", err)
	}

	var layers []*Layer
	layers = append(layers, manifest.Layers...)
	layers = append(layers, manifest.Config)

	for _, layer := range layers {
		if err := downloadBlob(
			ctx,
			downloadOpts{
				mp:      mp,
				digest:  layer.Digest,
				regOpts: regOpts,
				fn:      fn,
			}); err != nil {
			return err
		}
		delete(deleteMap, layer.Digest)
	}
	delete(deleteMap, manifest.Config.Digest)

	fn(api.ProgressResponse{Status: "verifying sha256 digest"})
	for _, layer := range layers {
		if err := verifyBlob(layer.Digest); err != nil {
			if errors.Is(err, errDigestMismatch) {
				// something went wrong, delete the blob
				fp, err := GetBlobsPath(layer.Digest)
				if err != nil {
					return err
				}
				if err := os.Remove(fp); err != nil {
					// log this, but return the original error
					log.Printf("couldn't remove file with digest mismatch '%s': %v", fp, err)
				}
			}
			return err
		}
	}

	fn(api.ProgressResponse{Status: "writing manifest"})

	manifestJSON, err := json.Marshal(manifest)
	if err != nil {
		return err
	}

	fp, err := mp.GetManifestPath()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(fp), 0o755); err != nil {
		return err
	}

	err = os.WriteFile(fp, manifestJSON, 0o644)
	if err != nil {
		log.Printf("couldn't write to %s", fp)
		return err
	}

	if noprune == "" {
		fn(api.ProgressResponse{Status: "removing any unused layers"})
		err = deleteUnusedLayers(nil, deleteMap, false)
		if err != nil {
			return err
		}
	}

	fn(api.ProgressResponse{Status: "success"})

	return nil
}

func pullModelManifest(ctx context.Context, mp ModelPath, regOpts *RegistryOptions) (*ManifestV2, error) {
	requestURL := mp.BaseURL().JoinPath("v2", mp.GetNamespaceRepository(), "manifests", mp.Tag)

	headers := make(http.Header)
	headers.Set("Accept", "application/vnd.docker.distribution.manifest.v2+json")
	resp, err := makeRequestWithRetry(ctx, http.MethodGet, requestURL, headers, nil, regOpts)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var m *ManifestV2
	if err := json.NewDecoder(resp.Body).Decode(&m); err != nil {
		return nil, err
	}

	return m, err
}

// GetSHA256Digest returns the SHA256 hash of a given buffer and returns it, and the size of buffer
func GetSHA256Digest(r io.Reader) (string, int64) {
	h := sha256.New()
	n, err := io.Copy(h, r)
	if err != nil {
		log.Fatal(err)
	}

	return fmt.Sprintf("sha256:%x", h.Sum(nil)), n
}

var errUnauthorized = fmt.Errorf("unauthorized")

func makeRequestWithRetry(ctx context.Context, method string, requestURL *url.URL, headers http.Header, body io.ReadSeeker, regOpts *RegistryOptions) (*http.Response, error) {
	resp, err := makeRequest(ctx, method, requestURL, headers, body, regOpts)
	if err != nil {
		if !errors.Is(err, context.Canceled) {
			log.Printf("request failed: %v", err)
		}

		return nil, err
	}

	switch {
	case resp.StatusCode == http.StatusUnauthorized:
		// Handle authentication error with one retry
		auth := resp.Header.Get("www-authenticate")
		authRedir := ParseAuthRedirectString(auth)
		token, err := getAuthToken(ctx, authRedir)
		if err != nil {
			return nil, err
		}
		regOpts.Token = token
		if body != nil {
			_, err = body.Seek(0, io.SeekStart)
			if err != nil {
				return nil, err
			}
		}

		resp, err := makeRequest(ctx, method, requestURL, headers, body, regOpts)
		if resp.StatusCode == http.StatusUnauthorized {
			return nil, errUnauthorized
		}

		return resp, err
	case resp.StatusCode == http.StatusNotFound:
		return nil, os.ErrNotExist
	case resp.StatusCode >= http.StatusBadRequest:
		responseBody, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("%d: %s", resp.StatusCode, err)
		}
		return nil, fmt.Errorf("%d: %s", resp.StatusCode, responseBody)
	}

	return resp, nil
}

func makeRequest(ctx context.Context, method string, requestURL *url.URL, headers http.Header, body io.Reader, regOpts *RegistryOptions) (*http.Response, error) {
	if requestURL.Scheme != "http" && regOpts != nil && regOpts.Insecure {
		requestURL.Scheme = "http"
	}

	req, err := http.NewRequestWithContext(ctx, method, requestURL.String(), body)
	if err != nil {
		return nil, err
	}

	if headers != nil {
		req.Header = headers
	}

	if regOpts != nil {
		if regOpts.Token != "" {
			req.Header.Set("Authorization", "Bearer "+regOpts.Token)
		} else if regOpts.Username != "" && regOpts.Password != "" {
			req.SetBasicAuth(regOpts.Username, regOpts.Password)
		}
	}

	req.Header.Set("User-Agent", fmt.Sprintf("ollama/%s (%s %s) Go/%s", version.Version, runtime.GOARCH, runtime.GOOS, runtime.Version()))

	if s := req.Header.Get("Content-Length"); s != "" {
		contentLength, err := strconv.ParseInt(s, 10, 64)
		if err != nil {
			return nil, err
		}

		req.ContentLength = contentLength
	}

	proxyURL, err := http.ProxyFromEnvironment(req)
	if err != nil {
		return nil, err
	}

	client := http.Client{
		Transport: &http.Transport{
			Proxy: http.ProxyURL(proxyURL),
		},
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}

	return resp, nil
}

func getValue(header, key string) string {
	startIdx := strings.Index(header, key+"=")
	if startIdx == -1 {
		return ""
	}

	// Move the index to the starting quote after the key.
	startIdx += len(key) + 2
	endIdx := startIdx

	for endIdx < len(header) {
		if header[endIdx] == '"' {
			if endIdx+1 < len(header) && header[endIdx+1] != ',' { // If the next character isn't a comma, continue
				endIdx++
				continue
			}
			break
		}
		endIdx++
	}
	return header[startIdx:endIdx]
}

func ParseAuthRedirectString(authStr string) AuthRedirect {
	authStr = strings.TrimPrefix(authStr, "Bearer ")

	return AuthRedirect{
		Realm:   getValue(authStr, "realm"),
		Service: getValue(authStr, "service"),
		Scope:   getValue(authStr, "scope"),
	}
}

var errDigestMismatch = fmt.Errorf("digest mismatch, file must be downloaded again")

func verifyBlob(digest string) error {
	fp, err := GetBlobsPath(digest)
	if err != nil {
		return err
	}

	f, err := os.Open(fp)
	if err != nil {
		return err
	}
	defer f.Close()

	fileDigest, _ := GetSHA256Digest(f)
	if digest != fileDigest {
		return fmt.Errorf("%w: want %s, got %s", errDigestMismatch, digest, fileDigest)
	}

	return nil
}

package cmd

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"syscall"
	"time"

	"github.com/olekukonko/tablewriter"
	"github.com/spf13/cobra"
	"golang.org/x/crypto/ssh"
	"golang.org/x/exp/slices"
	"golang.org/x/term"

	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/format"
	"github.com/jmorganca/ollama/parser"
	"github.com/jmorganca/ollama/progress"
	"github.com/jmorganca/ollama/readline"
	"github.com/jmorganca/ollama/server"
	"github.com/jmorganca/ollama/version"
)

type ImageData []byte

func CreateHandler(cmd *cobra.Command, args []string) error {
	filename, _ := cmd.Flags().GetString("file")
	filename, err := filepath.Abs(filename)
	if err != nil {
		return err
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	p := progress.NewProgress(os.Stderr)
	defer p.Stop()

	bars := make(map[string]*progress.Bar)

	modelfile, err := os.ReadFile(filename)
	if err != nil {
		return err
	}

	commands, err := parser.Parse(bytes.NewReader(modelfile))
	if err != nil {
		return err
	}

	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	status := "transferring model data"
	spinner := progress.NewSpinner(status)
	p.Add(status, spinner)

	for _, c := range commands {
		switch c.Name {
		case "model", "adapter":
			path := c.Args
			if path == "~" {
				path = home
			} else if strings.HasPrefix(path, "~/") {
				path = filepath.Join(home, path[2:])
			}

			if !filepath.IsAbs(path) {
				path = filepath.Join(filepath.Dir(filename), path)
			}

			bin, err := os.Open(path)
			if errors.Is(err, os.ErrNotExist) && c.Name == "model" {
				continue
			} else if err != nil {
				return err
			}
			defer bin.Close()

			hash := sha256.New()
			if _, err := io.Copy(hash, bin); err != nil {
				return err
			}
			bin.Seek(0, io.SeekStart)

			digest := fmt.Sprintf("sha256:%x", hash.Sum(nil))
			if err = client.CreateBlob(cmd.Context(), digest, bin); err != nil {
				return err
			}

			modelfile = bytes.ReplaceAll(modelfile, []byte(c.Args), []byte("@"+digest))
		}
	}

	fn := func(resp api.ProgressResponse) error {
		if resp.Digest != "" {
			spinner.Stop()

			bar, ok := bars[resp.Digest]
			if !ok {
				bar = progress.NewBar(fmt.Sprintf("pulling %s...", resp.Digest[7:19]), resp.Total, resp.Completed)
				bars[resp.Digest] = bar
				p.Add(resp.Digest, bar)
			}

			bar.Set(resp.Completed)
		} else if status != resp.Status {
			spinner.Stop()

			status = resp.Status
			spinner = progress.NewSpinner(status)
			p.Add(status, spinner)
		}

		return nil
	}

	request := api.CreateRequest{Name: args[0], Modelfile: string(modelfile)}
	if err := client.Create(cmd.Context(), &request, fn); err != nil {
		return err
	}

	return nil
}

func RunHandler(cmd *cobra.Command, args []string) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	name := args[0]
	// check if the model exists on the server
	_, err = client.Show(cmd.Context(), &api.ShowRequest{Name: name})
	var statusError api.StatusError
	switch {
	case errors.As(err, &statusError) && statusError.StatusCode == http.StatusNotFound:
		if err := PullHandler(cmd, args); err != nil {
			return err
		}
	case err != nil:
		return err
	}

	return RunGenerate(cmd, args)
}

func PushHandler(cmd *cobra.Command, args []string) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	insecure, err := cmd.Flags().GetBool("insecure")
	if err != nil {
		return err
	}

	p := progress.NewProgress(os.Stderr)
	defer p.Stop()

	bars := make(map[string]*progress.Bar)
	var status string
	var spinner *progress.Spinner

	fn := func(resp api.ProgressResponse) error {
		if resp.Digest != "" {
			if spinner != nil {
				spinner.Stop()
			}

			bar, ok := bars[resp.Digest]
			if !ok {
				bar = progress.NewBar(fmt.Sprintf("pushing %s...", resp.Digest[7:19]), resp.Total, resp.Completed)
				bars[resp.Digest] = bar
				p.Add(resp.Digest, bar)
			}

			bar.Set(resp.Completed)
		} else if status != resp.Status {
			if spinner != nil {
				spinner.Stop()
			}

			status = resp.Status
			spinner = progress.NewSpinner(status)
			p.Add(status, spinner)
		}

		return nil
	}

	request := api.PushRequest{Name: args[0], Insecure: insecure}
	if err := client.Push(cmd.Context(), &request, fn); err != nil {
		return err
	}

	spinner.Stop()
	return nil
}

func ListHandler(cmd *cobra.Command, args []string) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	models, err := client.List(cmd.Context())
	if err != nil {
		return err
	}

	var data [][]string

	for _, m := range models.Models {
		if len(args) == 0 || strings.HasPrefix(m.Name, args[0]) {
			data = append(data, []string{m.Name, m.Digest[:12], format.HumanBytes(m.Size), format.HumanTime(m.ModifiedAt, "Never")})
		}
	}

	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"NAME", "ID", "SIZE", "MODIFIED"})
	table.SetHeaderAlignment(tablewriter.ALIGN_LEFT)
	table.SetAlignment(tablewriter.ALIGN_LEFT)
	table.SetHeaderLine(false)
	table.SetBorder(false)
	table.SetNoWhiteSpace(true)
	table.SetTablePadding("\t")
	table.AppendBulk(data)
	table.Render()

	return nil
}

func DeleteHandler(cmd *cobra.Command, args []string) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	for _, name := range args {
		req := api.DeleteRequest{Name: name}
		if err := client.Delete(cmd.Context(), &req); err != nil {
			return err
		}
		fmt.Printf("deleted '%s'\n", name)
	}
	return nil
}

func ShowHandler(cmd *cobra.Command, args []string) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	if len(args) != 1 {
		return errors.New("missing model name")
	}

	license, errLicense := cmd.Flags().GetBool("license")
	modelfile, errModelfile := cmd.Flags().GetBool("modelfile")
	parameters, errParams := cmd.Flags().GetBool("parameters")
	system, errSystem := cmd.Flags().GetBool("system")
	template, errTemplate := cmd.Flags().GetBool("template")

	for _, boolErr := range []error{errLicense, errModelfile, errParams, errSystem, errTemplate} {
		if boolErr != nil {
			return errors.New("error retrieving flags")
		}
	}

	flagsSet := 0
	showType := ""

	if license {
		flagsSet++
		showType = "license"
	}

	if modelfile {
		flagsSet++
		showType = "modelfile"
	}

	if parameters {
		flagsSet++
		showType = "parameters"
	}

	if system {
		flagsSet++
		showType = "system"
	}

	if template {
		flagsSet++
		showType = "template"
	}

	if flagsSet > 1 {
		return errors.New("only one of '--license', '--modelfile', '--parameters', '--system', or '--template' can be specified")
	} else if flagsSet == 0 {
		return errors.New("one of '--license', '--modelfile', '--parameters', '--system', or '--template' must be specified")
	}

	req := api.ShowRequest{Name: args[0]}
	resp, err := client.Show(cmd.Context(), &req)
	if err != nil {
		return err
	}

	switch showType {
	case "license":
		fmt.Println(resp.License)
	case "modelfile":
		fmt.Println(resp.Modelfile)
	case "parameters":
		fmt.Println(resp.Parameters)
	case "system":
		fmt.Println(resp.System)
	case "template":
		fmt.Println(resp.Template)
	}

	return nil
}

func CopyHandler(cmd *cobra.Command, args []string) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	req := api.CopyRequest{Source: args[0], Destination: args[1]}
	if err := client.Copy(cmd.Context(), &req); err != nil {
		return err
	}
	fmt.Printf("copied '%s' to '%s'\n", args[0], args[1])
	return nil
}

func PullHandler(cmd *cobra.Command, args []string) error {
	insecure, err := cmd.Flags().GetBool("insecure")
	if err != nil {
		return err
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	p := progress.NewProgress(os.Stderr)
	defer p.Stop()

	bars := make(map[string]*progress.Bar)

	var status string
	var spinner *progress.Spinner

	fn := func(resp api.ProgressResponse) error {
		if resp.Digest != "" {
			if spinner != nil {
				spinner.Stop()
			}

			bar, ok := bars[resp.Digest]
			if !ok {
				bar = progress.NewBar(fmt.Sprintf("pulling %s...", resp.Digest[7:19]), resp.Total, resp.Completed)
				bars[resp.Digest] = bar
				p.Add(resp.Digest, bar)
			}

			bar.Set(resp.Completed)
		} else if status != resp.Status {
			if spinner != nil {
				spinner.Stop()
			}

			status = resp.Status
			spinner = progress.NewSpinner(status)
			p.Add(status, spinner)
		}

		return nil
	}

	request := api.PullRequest{Name: args[0], Insecure: insecure}
	if err := client.Pull(cmd.Context(), &request, fn); err != nil {
		return err
	}

	return nil
}

func RunGenerate(cmd *cobra.Command, args []string) error {
	interactive := true

	opts := generateOptions{
		Model:    args[0],
		WordWrap: os.Getenv("TERM") == "xterm-256color",
		Options:  map[string]interface{}{},
		Images:   []ImageData{},
	}

	format, err := cmd.Flags().GetString("format")
	if err != nil {
		return err
	}
	opts.Format = format

	prompts := args[1:]
	// prepend stdin to the prompt if provided
	if !term.IsTerminal(int(os.Stdin.Fd())) {
		in, err := io.ReadAll(os.Stdin)
		if err != nil {
			return err
		}

		prompts = append([]string{string(in)}, prompts...)
		opts.WordWrap = false
		interactive = false
	}
	opts.Prompt = strings.Join(prompts, " ")
	if len(prompts) > 0 {
		interactive = false
	}

	nowrap, err := cmd.Flags().GetBool("nowordwrap")
	if err != nil {
		return err
	}
	opts.WordWrap = !nowrap

	if !interactive {
		return generate(cmd, opts)
	}

	return generateInteractive(cmd, opts)
}

type generateContextKey string

type generateOptions struct {
	Model    string
	Prompt   string
	WordWrap bool
	Format   string
	System   string
	Template string
	Images   []ImageData
	Options  map[string]interface{}
}

func generate(cmd *cobra.Command, opts generateOptions) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	p := progress.NewProgress(os.Stderr)
	defer p.StopAndClear()

	spinner := progress.NewSpinner("")
	p.Add("", spinner)

	var latest api.GenerateResponse

	generateContext, ok := cmd.Context().Value(generateContextKey("context")).([]int)
	if !ok {
		generateContext = []int{}
	}

	termWidth, _, err := term.GetSize(int(os.Stdout.Fd()))
	if err != nil {
		opts.WordWrap = false
	}

	ctx, cancel := context.WithCancel(cmd.Context())
	defer cancel()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT)

	go func() {
		<-sigChan
		cancel()
	}()

	var currentLineLength int
	var wordBuffer string

	fn := func(response api.GenerateResponse) error {
		p.StopAndClear()

		latest = response

		termWidth, _, _ = term.GetSize(int(os.Stdout.Fd()))
		if opts.WordWrap && termWidth >= 10 {
			for _, ch := range response.Response {
				if currentLineLength+1 > termWidth-5 {
					if len(wordBuffer) > termWidth-10 {
						fmt.Printf("%s%c", wordBuffer, ch)
						wordBuffer = ""
						currentLineLength = 0
						continue
					}

					// backtrack the length of the last word and clear to the end of the line
					fmt.Printf("\x1b[%dD\x1b[K\n", len(wordBuffer))
					fmt.Printf("%s%c", wordBuffer, ch)
					currentLineLength = len(wordBuffer) + 1
				} else {
					fmt.Print(string(ch))
					currentLineLength += 1

					switch ch {
					case ' ':
						wordBuffer = ""
					case '\n':
						currentLineLength = 0
					default:
						wordBuffer += string(ch)
					}
				}
			}
		} else {
			fmt.Printf("%s%s", wordBuffer, response.Response)
			if len(wordBuffer) > 0 {
				wordBuffer = ""
			}
		}

		return nil
	}

	images := make([]api.ImageData, 0)
	for _, i := range opts.Images {
		images = append(images, api.ImageData(i))
	}
	request := api.GenerateRequest{
		Model:    opts.Model,
		Prompt:   opts.Prompt,
		Context:  generateContext,
		Format:   opts.Format,
		System:   opts.System,
		Template: opts.Template,
		Options:  opts.Options,
		Images:   images,
	}

	if err := client.Generate(ctx, &request, fn); err != nil {
		switch {
		case errors.Is(err, context.Canceled):
			return nil
		case strings.Contains(err.Error(), "unsupported model format"):
			// pull and retry to see if the model has been updated
			parts := strings.Split(opts.Model, string(os.PathSeparator))
			if len(parts) == 1 {
				// this is a library model, log some info
				fmt.Fprintln(os.Stderr, "This model is no longer compatible with Ollama. Pulling a new version...")
			}
			if err := PullHandler(cmd, []string{opts.Model}); err != nil {
				fmt.Printf("Error: %s\n", err)
				return fmt.Errorf("unsupported model, please update this model to gguf format") // relay the original error
			}
			// retry
			if err := client.Generate(ctx, &request, fn); err != nil {
				if errors.Is(err, context.Canceled) {
					return nil
				}
				return err
			}
		default:
			return err
		}
	}
	if opts.Prompt != "" {
		fmt.Println()
		fmt.Println()
	}

	if !latest.Done {
		return nil
	}

	verbose, err := cmd.Flags().GetBool("verbose")
	if err != nil {
		return err
	}

	if verbose {
		latest.Summary()
	}

	ctx = context.WithValue(cmd.Context(), generateContextKey("context"), latest.Context)
	cmd.SetContext(ctx)

	return nil
}

type MultilineState int

const (
	MultilineNone MultilineState = iota
	MultilinePrompt
	MultilineSystem
	MultilineTemplate
)

func modelIsMultiModal(cmd *cobra.Command, name string) bool {
	// get model details
	client, err := api.ClientFromEnvironment()
	if err != nil {
		fmt.Println("error: couldn't connect to ollama server")
		return false
	}

	req := api.ShowRequest{Name: name}
	resp, err := client.Show(cmd.Context(), &req)
	if err != nil {
		return false
	}

	return slices.Contains(resp.Details.Families, "clip")
}

func generateInteractive(cmd *cobra.Command, opts generateOptions) error {
	multiModal := modelIsMultiModal(cmd, opts.Model)

	// load the model
	loadOpts := generateOptions{
		Model:  opts.Model,
		Prompt: "",
		Images: []ImageData{},
	}
	if err := generate(cmd, loadOpts); err != nil {
		return err
	}

	usage := func() {
		fmt.Fprintln(os.Stderr, "Available Commands:")
		fmt.Fprintln(os.Stderr, "  /set         Set session variables")
		fmt.Fprintln(os.Stderr, "  /show        Show model information")
		fmt.Fprintln(os.Stderr, "  /bye         Exit")
		fmt.Fprintln(os.Stderr, "  /?, /help    Help for a command")
		fmt.Fprintln(os.Stderr, "")
		fmt.Fprintln(os.Stderr, "Use \"\"\" to begin a multi-line message.")
		fmt.Fprintln(os.Stderr, "")
	}

	usageSet := func() {
		fmt.Fprintln(os.Stderr, "Available Commands:")
		fmt.Fprintln(os.Stderr, "  /set parameter ...     Set a parameter")
		fmt.Fprintln(os.Stderr, "  /set system <string>   Set system message")
		fmt.Fprintln(os.Stderr, "  /set template <string> Set prompt template")
		fmt.Fprintln(os.Stderr, "  /set history           Enable history")
		fmt.Fprintln(os.Stderr, "  /set nohistory         Disable history")
		fmt.Fprintln(os.Stderr, "  /set wordwrap          Enable wordwrap")
		fmt.Fprintln(os.Stderr, "  /set nowordwrap        Disable wordwrap")
		fmt.Fprintln(os.Stderr, "  /set format json       Enable JSON mode")
		fmt.Fprintln(os.Stderr, "  /set noformat          Disable formatting")
		fmt.Fprintln(os.Stderr, "  /set verbose           Show LLM stats")
		fmt.Fprintln(os.Stderr, "  /set quiet             Disable LLM stats")
		fmt.Fprintln(os.Stderr, "")
	}

	usageShow := func() {
		fmt.Fprintln(os.Stderr, "Available Commands:")
		fmt.Fprintln(os.Stderr, "  /show license      Show model license")
		fmt.Fprintln(os.Stderr, "  /show modelfile    Show Modelfile for this model")
		fmt.Fprintln(os.Stderr, "  /show parameters   Show parameters for this model")
		fmt.Fprintln(os.Stderr, "  /show system       Show system message")
		fmt.Fprintln(os.Stderr, "  /show template     Show prompt template")
		fmt.Fprintln(os.Stderr, "")
	}

	// only list out the most common parameters
	usageParameters := func() {
		fmt.Fprintln(os.Stderr, "Available Parameters:")
		fmt.Fprintln(os.Stderr, "  /set parameter seed <int>             Random number seed")
		fmt.Fprintln(os.Stderr, "  /set parameter num_predict <int>      Max number of tokens to predict")
		fmt.Fprintln(os.Stderr, "  /set parameter top_k <int>            Pick from top k num of tokens")
		fmt.Fprintln(os.Stderr, "  /set parameter top_p <float>          Pick token based on sum of probabilities")
		fmt.Fprintln(os.Stderr, "  /set parameter num_ctx <int>          Set the context size")
		fmt.Fprintln(os.Stderr, "  /set parameter temperature <float>    Set creativity level")
		fmt.Fprintln(os.Stderr, "  /set parameter repeat_penalty <float> How strongly to penalize repetitions")
		fmt.Fprintln(os.Stderr, "  /set parameter repeat_last_n <int>    Set how far back to look for repetitions")
		fmt.Fprintln(os.Stderr, "  /set parameter num_gpu <int>          The number of layers to send to the GPU")
		fmt.Fprintln(os.Stderr, "  /set parameter stop \"<string>\", ...   Set the stop parameters")
		fmt.Fprintln(os.Stderr, "")
	}

	scanner, err := readline.New(readline.Prompt{
		Prompt:         ">>> ",
		AltPrompt:      "... ",
		Placeholder:    "Send a message (/? for help)",
		AltPlaceholder: `Use """ to end multi-line input`,
	})
	if err != nil {
		return err
	}

	fmt.Print(readline.StartBracketedPaste)
	defer fmt.Printf(readline.EndBracketedPaste)

	var multiline MultilineState
	var prompt string

	for {
		line, err := scanner.Readline()
		switch {
		case errors.Is(err, io.EOF):
			fmt.Println()
			return nil
		case errors.Is(err, readline.ErrInterrupt):
			if line == "" {
				fmt.Println("\nUse Ctrl-D or /bye to exit.")
			}

			scanner.Prompt.UseAlt = false
			prompt = ""

			continue
		case err != nil:
			return err
		}

		switch {
		case strings.HasPrefix(prompt, `"""`):
			// if the prompt so far starts with """ then we're in multiline mode
			// and we need to keep reading until we find a line that ends with """
			cut, found := strings.CutSuffix(line, `"""`)
			prompt += cut

			if !found {
				prompt += "\n"
				continue
			}

			prompt = strings.TrimPrefix(prompt, `"""`)
			scanner.Prompt.UseAlt = false

			switch multiline {
			case MultilineSystem:
				opts.System = prompt
				prompt = ""
				fmt.Println("Set system message.")
			case MultilineTemplate:
				opts.Template = prompt
				prompt = ""
				fmt.Println("Set prompt template.")
			}
			multiline = MultilineNone
		case strings.HasPrefix(line, `"""`) && len(prompt) == 0:
			scanner.Prompt.UseAlt = true
			multiline = MultilinePrompt
			prompt += line + "\n"
			continue
		case scanner.Pasting:
			prompt += line + "\n"
			continue
		case strings.HasPrefix(line, "/list"):
			args := strings.Fields(line)
			if err := ListHandler(cmd, args[1:]); err != nil {
				return err
			}
		case strings.HasPrefix(line, "/set"):
			args := strings.Fields(line)
			if len(args) > 1 {
				switch args[1] {
				case "history":
					scanner.HistoryEnable()
				case "nohistory":
					scanner.HistoryDisable()
				case "wordwrap":
					opts.WordWrap = true
					fmt.Println("Set 'wordwrap' mode.")
				case "nowordwrap":
					opts.WordWrap = false
					fmt.Println("Set 'nowordwrap' mode.")
				case "verbose":
					cmd.Flags().Set("verbose", "true")
					fmt.Println("Set 'verbose' mode.")
				case "quiet":
					cmd.Flags().Set("verbose", "false")
					fmt.Println("Set 'quiet' mode.")
				case "format":
					if len(args) < 3 || args[2] != "json" {
						fmt.Println("Invalid or missing format. For 'json' mode use '/set format json'")
					} else {
						opts.Format = args[2]
						fmt.Printf("Set format to '%s' mode.\n", args[2])
					}
				case "noformat":
					opts.Format = ""
					fmt.Println("Disabled format.")
				case "parameter":
					if len(args) < 4 {
						usageParameters()
						continue
					}
					var params []string
					for _, p := range args[3:] {
						params = append(params, p)
					}
					fp, err := api.FormatParams(map[string][]string{args[2]: params})
					if err != nil {
						fmt.Printf("Couldn't set parameter: %q\n\n", err)
						continue
					}
					fmt.Printf("Set parameter '%s' to '%s'\n\n", args[2], strings.Join(params, ", "))
					opts.Options[args[2]] = fp[args[2]]
				case "system", "template":
					if len(args) < 3 {
						usageSet()
						continue
					}
					line := strings.Join(args[2:], " ")
					line = strings.TrimPrefix(line, `"""`)
					if strings.HasPrefix(args[2], `"""`) {
						cut, found := strings.CutSuffix(line, `"""`)
						prompt += cut
						if found {
							if args[1] == "system" {
								opts.System = prompt
								fmt.Println("Set system message.")
							} else {
								opts.Template = prompt
								fmt.Println("Set prompt template.")
							}
							prompt = ""
						} else {
							prompt = `"""` + prompt + "\n"
							if args[1] == "system" {
								multiline = MultilineSystem
							} else {
								multiline = MultilineTemplate
							}
							scanner.Prompt.UseAlt = true
						}
					} else {
						opts.System = line
						fmt.Println("Set system message.")
					}
				default:
					fmt.Printf("Unknown command '/set %s'. Type /? for help\n", args[1])
				}
			} else {
				usageSet()
			}
		case strings.HasPrefix(line, "/show"):
			args := strings.Fields(line)
			if len(args) > 1 {
				client, err := api.ClientFromEnvironment()
				if err != nil {
					fmt.Println("error: couldn't connect to ollama server")
					return err
				}
				resp, err := client.Show(cmd.Context(), &api.ShowRequest{Name: opts.Model})
				if err != nil {
					fmt.Println("error: couldn't get model")
					return err
				}

				switch args[1] {
				case "license":
					if resp.License == "" {
						fmt.Print("No license was specified for this model.\n\n")
					} else {
						fmt.Println(resp.License)
					}
				case "modelfile":
					fmt.Println(resp.Modelfile)
				case "parameters":
					if resp.Parameters == "" {
						fmt.Print("No parameters were specified for this model.\n\n")
					} else {
						if len(opts.Options) > 0 {
							fmt.Println("User defined parameters:")
							for k, v := range opts.Options {
								fmt.Printf("%-*s %v\n", 30, k, v)
							}
							fmt.Println()
						}
						fmt.Println("Model defined parameters:")
						fmt.Println(resp.Parameters)
					}
				case "system":
					switch {
					case opts.System != "":
						fmt.Println(opts.System + "\n")
					case resp.System != "":
						fmt.Println(resp.System + "\n")
					default:
						fmt.Print("No system message was specified for this model.\n\n")
					}
				case "template":
					switch {
					case opts.Template != "":
						fmt.Println(opts.Template + "\n")
					case resp.Template != "":
						fmt.Println(resp.Template)
					default:
						fmt.Print("No prompt template was specified for this model.\n\n")
					}
				default:
					fmt.Printf("Unknown command '/show %s'. Type /? for help\n", args[1])
				}
			} else {
				usageShow()
			}
		case strings.HasPrefix(line, "/help"), strings.HasPrefix(line, "/?"):
			args := strings.Fields(line)
			if len(args) > 1 {
				switch args[1] {
				case "set", "/set":
					usageSet()
				case "show", "/show":
					usageShow()
				}
			} else {
				usage()
			}
		case line == "/exit", line == "/bye":
			return nil
		case strings.HasPrefix(line, "/"):
			args := strings.Fields(line)
			isFile := false

			if multiModal {
				for _, f := range extractFileNames(line) {
					if strings.HasPrefix(f, args[0]) {
						isFile = true
						break
					}
				}
			}

			if isFile {
				prompt += line
			} else {
				fmt.Printf("Unknown command '%s'. Type /? for help\n", args[0])
				continue
			}
		default:
			prompt += line
		}

		if len(prompt) > 0 && multiline == MultilineNone {
			opts.Prompt = prompt
			if multiModal {
				newPrompt, images, err := extractFileData(prompt)
				if err != nil {
					return err
				}
				opts.Prompt = newPrompt

				// reset the context if we find another image
				if len(images) > 0 {
					opts.Images = images
					ctx := cmd.Context()
					ctx = context.WithValue(ctx, generateContextKey("context"), []int{})
					cmd.SetContext(ctx)
				}
				if len(opts.Images) == 0 {
					fmt.Println("This model requires you to add a jpeg, png, or svg image.")
					fmt.Println()
					prompt = ""
					continue
				}
			}
			if err := generate(cmd, opts); err != nil {
				return err
			}

			prompt = ""
		}
	}
}

func normalizeFilePath(fp string) string {
	// Define a map of escaped characters and their replacements
	replacements := map[string]string{
		"\\ ":  " ",  // Escaped space
		"\\(":  "(",  // Escaped left parenthesis
		"\\)":  ")",  // Escaped right parenthesis
		"\\[":  "[",  // Escaped left square bracket
		"\\]":  "]",  // Escaped right square bracket
		"\\{":  "{",  // Escaped left curly brace
		"\\}":  "}",  // Escaped right curly brace
		"\\$":  "$",  // Escaped dollar sign
		"\\&":  "&",  // Escaped ampersand
		"\\;":  ";",  // Escaped semicolon
		"\\'":  "'",  // Escaped single quote
		"\\\\": "\\", // Escaped backslash
		"\\*":  "*",  // Escaped asterisk
		"\\?":  "?",  // Escaped question mark
	}

	for escaped, actual := range replacements {
		fp = strings.ReplaceAll(fp, escaped, actual)
	}
	return fp
}

func extractFileNames(input string) []string {
	// Regex to match file paths starting with / or ./ and include escaped spaces (\ or %20)
	// and followed by more characters and a file extension
	regexPattern := `(?:\./|/)[\S\\ ]+?\.(?i:jpg|jpeg|png|svg)\b`
	re := regexp.MustCompile(regexPattern)

	return re.FindAllString(input, -1)
}

func extractFileData(input string) (string, []ImageData, error) {
	filePaths := extractFileNames(input)
	var imgs []ImageData

	for _, fp := range filePaths {
		nfp := normalizeFilePath(fp)
		data, err := getImageData(nfp)
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			fmt.Printf("Couldn't process image: %q\n", err)
			return "", imgs, err
		}
		fmt.Printf("Added image '%s'\n", nfp)
		input = strings.ReplaceAll(input, fp, "")
		imgs = append(imgs, data)
	}
	return input, imgs, nil
}

func RunServer(cmd *cobra.Command, _ []string) error {
	host, port, err := net.SplitHostPort(os.Getenv("OLLAMA_HOST"))
	if err != nil {
		host, port = "127.0.0.1", "11434"
		if ip := net.ParseIP(strings.Trim(os.Getenv("OLLAMA_HOST"), "[]")); ip != nil {
			host = ip.String()
		}
	}

	if err := initializeKeypair(); err != nil {
		return err
	}

	ln, err := net.Listen("tcp", net.JoinHostPort(host, port))
	if err != nil {
		return err
	}

	return server.Serve(ln)
}

func getImageData(filePath string) ([]byte, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	buf := make([]byte, 512)
	_, err = file.Read(buf)
	if err != nil {
		return nil, err
	}

	contentType := http.DetectContentType(buf)
	allowedTypes := []string{"image/jpeg", "image/jpg", "image/svg+xml", "image/png"}
	if !slices.Contains(allowedTypes, contentType) {
		return nil, fmt.Errorf("invalid image type: %s", contentType)
	}

	info, err := file.Stat()
	if err != nil {
		return nil, err
	}

	// Check if the file size exceeds 100MB
	var maxSize int64 = 100 * 1024 * 1024 // 100MB in bytes
	if info.Size() > maxSize {
		return nil, fmt.Errorf("file size exceeds maximum limit (100MB)")
	}

	buf = make([]byte, info.Size())
	_, err = file.Seek(0, 0)
	if err != nil {
		return nil, err
	}

	_, err = io.ReadFull(file, buf)
	if err != nil {
		return nil, err
	}

	return buf, nil
}

func initializeKeypair() error {
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	privKeyPath := filepath.Join(home, ".ollama", "id_ed25519")
	pubKeyPath := filepath.Join(home, ".ollama", "id_ed25519.pub")

	_, err = os.Stat(privKeyPath)
	if os.IsNotExist(err) {
		fmt.Printf("Couldn't find '%s'. Generating new private key.\n", privKeyPath)
		_, privKey, err := ed25519.GenerateKey(rand.Reader)
		if err != nil {
			return err
		}

		privKeyBytes, err := format.OpenSSHPrivateKey(privKey, "")
		if err != nil {
			return err
		}

		err = os.MkdirAll(filepath.Dir(privKeyPath), 0o755)
		if err != nil {
			return fmt.Errorf("could not create directory %w", err)
		}

		err = os.WriteFile(privKeyPath, pem.EncodeToMemory(privKeyBytes), 0o600)
		if err != nil {
			return err
		}

		sshPrivateKey, err := ssh.NewSignerFromKey(privKey)
		if err != nil {
			return err
		}

		pubKeyData := ssh.MarshalAuthorizedKey(sshPrivateKey.PublicKey())

		err = os.WriteFile(pubKeyPath, pubKeyData, 0o644)
		if err != nil {
			return err
		}

		fmt.Printf("Your new public key is: \n\n%s\n", string(pubKeyData))
	}
	return nil
}

func startMacApp(ctx context.Context, client *api.Client) error {
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	link, err := os.Readlink(exe)
	if err != nil {
		return err
	}
	if !strings.Contains(link, "Ollama.app") {
		return fmt.Errorf("could not find ollama app")
	}
	path := strings.Split(link, "Ollama.app")
	if err := exec.Command("/usr/bin/open", "-a", path[0]+"Ollama.app").Run(); err != nil {
		return err
	}
	// wait for the server to start
	timeout := time.After(5 * time.Second)
	tick := time.Tick(500 * time.Millisecond)
	for {
		select {
		case <-timeout:
			return errors.New("timed out waiting for server to start")
		case <-tick:
			if err := client.Heartbeat(ctx); err == nil {
				return nil // server has started
			}
		}
	}
}

func checkServerHeartbeat(cmd *cobra.Command, _ []string) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}
	if err := client.Heartbeat(cmd.Context()); err != nil {
		if !strings.Contains(err.Error(), "connection refused") {
			return err
		}
		if runtime.GOOS == "darwin" {
			if err := startMacApp(cmd.Context(), client); err != nil {
				return fmt.Errorf("could not connect to ollama app, is it running?")
			}
		} else {
			return fmt.Errorf("could not connect to ollama server, run 'ollama serve' to start it")
		}
	}
	return nil
}

func versionHandler(cmd *cobra.Command, _ []string) {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return
	}

	serverVersion, err := client.Version(cmd.Context())
	if err != nil {
		fmt.Println("Warning: could not connect to a running Ollama instance")
	}

	if serverVersion != "" {
		fmt.Printf("ollama version is %s\n", serverVersion)
	}

	if serverVersion != version.Version {
		fmt.Printf("Warning: client version is %s\n", version.Version)
	}
}

func NewCLI() *cobra.Command {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	cobra.EnableCommandSorting = false

	rootCmd := &cobra.Command{
		Use:           "ollama",
		Short:         "Large language model runner",
		SilenceUsage:  true,
		SilenceErrors: true,
		CompletionOptions: cobra.CompletionOptions{
			DisableDefaultCmd: true,
		},
		Run: func(cmd *cobra.Command, args []string) {
			if version, _ := cmd.Flags().GetBool("version"); version {
				versionHandler(cmd, args)
				return
			}

			cmd.Print(cmd.UsageString())
		},
	}

	rootCmd.Flags().BoolP("version", "v", false, "Show version information")

	createCmd := &cobra.Command{
		Use:     "create MODEL",
		Short:   "Create a model from a Modelfile",
		Args:    cobra.ExactArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE:    CreateHandler,
	}

	createCmd.Flags().StringP("file", "f", "Modelfile", "Name of the Modelfile (default \"Modelfile\")")

	showCmd := &cobra.Command{
		Use:     "show MODEL",
		Short:   "Show information for a model",
		Args:    cobra.ExactArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE:    ShowHandler,
	}

	showCmd.Flags().Bool("license", false, "Show license of a model")
	showCmd.Flags().Bool("modelfile", false, "Show Modelfile of a model")
	showCmd.Flags().Bool("parameters", false, "Show parameters of a model")
	showCmd.Flags().Bool("template", false, "Show template of a model")
	showCmd.Flags().Bool("system", false, "Show system message of a model")

	runCmd := &cobra.Command{
		Use:     "run MODEL [PROMPT]",
		Short:   "Run a model",
		Args:    cobra.MinimumNArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE:    RunHandler,
	}

	runCmd.Flags().Bool("verbose", false, "Show timings for response")
	runCmd.Flags().Bool("insecure", false, "Use an insecure registry")
	runCmd.Flags().Bool("nowordwrap", false, "Don't wrap words to the next line automatically")
	runCmd.Flags().String("format", "", "Response format (e.g. json)")

	serveCmd := &cobra.Command{
		Use:     "serve",
		Aliases: []string{"start"},
		Short:   "Start ollama",
		Args:    cobra.ExactArgs(0),
		RunE:    RunServer,
	}

	pullCmd := &cobra.Command{
		Use:     "pull MODEL",
		Short:   "Pull a model from a registry",
		Args:    cobra.ExactArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE:    PullHandler,
	}

	pullCmd.Flags().Bool("insecure", false, "Use an insecure registry")

	pushCmd := &cobra.Command{
		Use:     "push MODEL",
		Short:   "Push a model to a registry",
		Args:    cobra.ExactArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE:    PushHandler,
	}

	pushCmd.Flags().Bool("insecure", false, "Use an insecure registry")

	listCmd := &cobra.Command{
		Use:     "list",
		Aliases: []string{"ls"},
		Short:   "List models",
		PreRunE: checkServerHeartbeat,
		RunE:    ListHandler,
	}

	copyCmd := &cobra.Command{
		Use:     "cp SOURCE TARGET",
		Short:   "Copy a model",
		Args:    cobra.ExactArgs(2),
		PreRunE: checkServerHeartbeat,
		RunE:    CopyHandler,
	}

	deleteCmd := &cobra.Command{
		Use:     "rm MODEL [MODEL...]",
		Short:   "Remove a model",
		Args:    cobra.MinimumNArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE:    DeleteHandler,
	}

	rootCmd.AddCommand(
		serveCmd,
		createCmd,
		showCmd,
		runCmd,
		pullCmd,
		pushCmd,
		listCmd,
		copyCmd,
		deleteCmd,
	)

	return rootCmd
}

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
	"runtime"
	"strings"
	"syscall"
	"time"

	"github.com/olekukonko/tablewriter"
	"github.com/spf13/cobra"
	"golang.org/x/crypto/ssh"
	"golang.org/x/term"

	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/format"
	"github.com/jmorganca/ollama/parser"
	"github.com/jmorganca/ollama/progress"
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
	model, err := client.Show(cmd.Context(), &api.ShowRequest{Name: name})
	var statusError api.StatusError
	switch {
	case errors.As(err, &statusError) && statusError.StatusCode == http.StatusNotFound:
		if err := PullHandler(cmd, []string{name}); err != nil {
			return err
		}
	case err != nil:
		return err
	default:
		// the model was found, check if it is in the correct format
		if model.Details.Format != "" && model.Details.Format != "gguf" {
			// pull and retry to see if the model has been updated
			parts := strings.Split(name, string(os.PathSeparator))
			if len(parts) == 1 {
				// this is a library model, log some info
				fmt.Fprintln(os.Stderr, "This model is no longer compatible with Ollama. Pulling a new version...")
			}
			if err := PullHandler(cmd, []string{name}); err != nil {
				fmt.Printf("Error: %s\n", err)
				return fmt.Errorf("unsupported model, please update this model to gguf format") // relay the original error
			}
		}
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
		if errors.Is(err, context.Canceled) {
			return nil
		}
		return err
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

func completionHandler(cmd *cobra.Command, args []string) error {
	var err error
	switch args[0] {
	case "bash":
		err = cmd.Root().GenBashCompletion(os.Stdout)
	case "zsh":
		err = cmd.Root().GenZshCompletion(os.Stdout)
	case "fish":
		err = cmd.Root().GenFishCompletion(os.Stdout, true)
	default:
		err = errors.New("unsupported shell. Supported shells: zsh, fish, bash")
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %s\n", err)
	}
	return nil
}

func autocompleteModelName(_ *cobra.Command, _ []string, toComplete string) ([]string, cobra.ShellCompDirective) {
	log.Printf("autocomplete: %s", toComplete)
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return nil, cobra.ShellCompDirectiveError
	}

	models, err := client.List(context.Background())
	if err != nil {
		return nil, cobra.ShellCompDirectiveError
	}

	var data []string

	for _, m := range models.Models {
		log.Printf("model: %s", m.Name)
		if strings.HasPrefix(m.Name, toComplete) {
			data = append(data, m.Name[:strings.IndexByte(m.Name, ':')])
		}
	}

	return data, cobra.ShellCompDirectiveNoFileComp
}

func doNotAutocomplete(_ *cobra.Command, _ []string, _ string) ([]string, cobra.ShellCompDirective) {
	return []string{}, cobra.ShellCompDirectiveNoFileComp
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
		Use:               "create MODEL",
		Short:             "Create a model from a Modelfile",
		Args:              cobra.ExactArgs(1),
		PreRunE:           checkServerHeartbeat,
		RunE:              CreateHandler,
		ValidArgsFunction: doNotAutocomplete,
	}

	createCmd.Flags().StringP("file", "f", "Modelfile", "Name of the Modelfile (default \"Modelfile\")")

	showCmd := &cobra.Command{
		Use:               "show MODEL",
		Short:             "Show information for a model",
		Args:              cobra.ExactArgs(1),
		PreRunE:           checkServerHeartbeat,
		RunE:              ShowHandler,
		ValidArgsFunction: autocompleteModelName,
	}

	showCmd.Flags().Bool("license", false, "Show license of a model")
	showCmd.Flags().Bool("modelfile", false, "Show Modelfile of a model")
	showCmd.Flags().Bool("parameters", false, "Show parameters of a model")
	showCmd.Flags().Bool("template", false, "Show template of a model")
	showCmd.Flags().Bool("system", false, "Show system message of a model")

	runCmd := &cobra.Command{
		Use:               "run MODEL [PROMPT]",
		Short:             "Run a model",
		Args:              cobra.MinimumNArgs(1),
		PreRunE:           checkServerHeartbeat,
		RunE:              RunHandler,
		ValidArgsFunction: autocompleteModelName,
	}

	runCmd.Flags().Bool("verbose", false, "Show timings for response")
	runCmd.Flags().Bool("insecure", false, "Use an insecure registry")
	runCmd.Flags().Bool("nowordwrap", false, "Don't wrap words to the next line automatically")
	runCmd.Flags().String("format", "", "Response format (e.g. json)")

	serveCmd := &cobra.Command{
		Use:               "serve",
		Aliases:           []string{"start"},
		Short:             "Start ollama",
		Args:              cobra.ExactArgs(0),
		RunE:              RunServer,
		ValidArgsFunction: doNotAutocomplete,
	}

	pullCmd := &cobra.Command{
		Use:               "pull MODEL",
		Short:             "Pull a model from a registry",
		Args:              cobra.ExactArgs(1),
		PreRunE:           checkServerHeartbeat,
		RunE:              PullHandler,
		ValidArgsFunction: doNotAutocomplete,
	}

	pullCmd.Flags().Bool("insecure", false, "Use an insecure registry")

	pushCmd := &cobra.Command{
		Use:               "push MODEL",
		Short:             "Push a model to a registry",
		Args:              cobra.ExactArgs(1),
		PreRunE:           checkServerHeartbeat,
		RunE:              PushHandler,
		ValidArgsFunction: autocompleteModelName,
	}

	pushCmd.Flags().Bool("insecure", false, "Use an insecure registry")

	listCmd := &cobra.Command{
		Use:               "list",
		Aliases:           []string{"ls"},
		Short:             "List models",
		PreRunE:           checkServerHeartbeat,
		RunE:              ListHandler,
		ValidArgsFunction: doNotAutocomplete,
	}

	copyCmd := &cobra.Command{
		Use:               "cp SOURCE TARGET",
		Short:             "Copy a model",
		Args:              cobra.ExactArgs(2),
		PreRunE:           checkServerHeartbeat,
		RunE:              CopyHandler,
		ValidArgsFunction: autocompleteModelName,
	}

	deleteCmd := &cobra.Command{
		Use:               "rm MODEL [MODEL...]",
		Short:             "Remove a model",
		Args:              cobra.MinimumNArgs(1),
		PreRunE:           checkServerHeartbeat,
		RunE:              DeleteHandler,
		ValidArgsFunction: autocompleteModelName,
	}

	completionCmd := &cobra.Command{
		Use:                   "completion [bash|zsh|fish]",
		Short:                 "Generate completion scripts",
		DisableFlagsInUseLine: true,
		Hidden:                true,
		Args:                  cobra.ExactArgs(1),
		RunE:                  completionHandler,
		ValidArgsFunction: func(_ *cobra.Command, _ []string, toComplete string) ([]string, cobra.ShellCompDirective) {
			return []string{"bash", "zsh", "fish"}, cobra.ShellCompDirectiveNoFileComp
		},
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
		completionCmd,
	)

	return rootCmd
}

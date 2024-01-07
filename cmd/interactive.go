package cmd

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"regexp"
	"strings"

	"github.com/spf13/cobra"
	"golang.org/x/exp/slices"

	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/readline"
)

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
		fmt.Fprintln(os.Stderr, "  /set          Set session variables")
		fmt.Fprintln(os.Stderr, "  /show         Show model information")
		fmt.Fprintln(os.Stderr, "  /bye          Exit")
		fmt.Fprintln(os.Stderr, "  /?, /help     Help for a command")
		fmt.Fprintln(os.Stderr, "  /? shortcuts  Help for keyboard shortcuts")
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

	usageShortcuts := func() {
		fmt.Fprintln(os.Stderr, "Available keyboard shortcuts:")
		fmt.Fprintln(os.Stderr, "  Ctrl + a            Move to the beginning of the line (Home)")
		fmt.Fprintln(os.Stderr, "  Ctrl + e            Move to the end of the line (End)")
		fmt.Fprintln(os.Stderr, "   Alt + b            Move back (left) one word")
		fmt.Fprintln(os.Stderr, "   Alt + f            Move forward (right) one word")
		fmt.Fprintln(os.Stderr, "  Ctrl + k            Delete the sentence after the cursor")
		fmt.Fprintln(os.Stderr, "  Ctrl + u            Delete the sentence before the cursor")
		fmt.Fprintln(os.Stderr, "")
		fmt.Fprintln(os.Stderr, "  Ctrl + l            Clear the screen")
		fmt.Fprintln(os.Stderr, "  Ctrl + c            Stop the model from responding")
		fmt.Fprintln(os.Stderr, "  Ctrl + d            Exit ollama (/bye)")
		fmt.Fprintln(os.Stderr, "")
	}

	usageShow := func() {
		fmt.Fprintln(os.Stderr, "Available Commands:")
		fmt.Fprintln(os.Stderr, "  /show info         Show details for this model")
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

	var sb strings.Builder
	var multiline MultilineState

	for {
		line, err := scanner.Readline()
		switch {
		case errors.Is(err, io.EOF):
			fmt.Println()
			return nil
		case errors.Is(err, readline.ErrInterrupt):
			if line == "" {
				fmt.Println("\nUse Ctrl + d or /bye to exit.")
			}

			scanner.Prompt.UseAlt = false
			sb.Reset()

			continue
		case err != nil:
			return err
		}

		switch {
		case multiline != MultilineNone:
			// check if there's a multiline terminating string
			before, ok := strings.CutSuffix(line, `"""`)
			sb.WriteString(before)
			if !ok {
				fmt.Fprintln(&sb)
				continue
			}

			switch multiline {
			case MultilineSystem:
				opts.System = sb.String()
				fmt.Println("Set system message.")
				sb.Reset()
			case MultilineTemplate:
				opts.Template = sb.String()
				fmt.Println("Set prompt template.")
				sb.Reset()
			}

			multiline = MultilineNone
			scanner.Prompt.UseAlt = false
		case strings.HasPrefix(line, `"""`):
			line := strings.TrimPrefix(line, `"""`)
			line, ok := strings.CutSuffix(line, `"""`)
			sb.WriteString(line)
			if !ok {
				// no multiline terminating string; need more input
				fmt.Fprintln(&sb)
				multiline = MultilinePrompt
				scanner.Prompt.UseAlt = true
				break
			}
		case scanner.Pasting:
			fmt.Fprintln(&sb, line)
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

					if args[1] == "system" {
						multiline = MultilineSystem
					} else if args[1] == "template" {
						multiline = MultilineTemplate
					}

					line := strings.Join(args[2:], " ")
					line, ok := strings.CutPrefix(line, `"""`)
					if !ok {
						multiline = MultilineNone
					} else {
						// only cut suffix if the line is multiline
						line, ok = strings.CutSuffix(line, `"""`)
						if ok {
							multiline = MultilineNone
						}
					}

					sb.WriteString(line)
					if multiline != MultilineNone {
						scanner.Prompt.UseAlt = true
						continue
					}

					if args[1] == "system" {
						opts.System = sb.String()
						fmt.Println("Set system message.")
					} else if args[1] == "template" {
						opts.Template = sb.String()
						fmt.Println("Set prompt template.")
					}

					sb.Reset()
					continue
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
				req := &api.ShowRequest{
					Name:     opts.Model,
					System:   opts.System,
					Template: opts.Template,
					Options:  opts.Options,
				}
				resp, err := client.Show(cmd.Context(), req)
				if err != nil {
					fmt.Println("error: couldn't get model")
					return err
				}

				switch args[1] {
				case "info":
					fmt.Println("Model details:")
					if len(resp.Details.Families) > 0 {
						fmt.Printf("Family              %s\n", strings.Join(resp.Details.Families, ", "))
					} else if resp.Details.Family != "" {
						fmt.Printf("Family              %s\n", resp.Details.Family)
					}
					fmt.Printf("Parameter Size      %s\n", resp.Details.ParameterSize)
					fmt.Printf("Quantization Level  %s\n", resp.Details.QuantizationLevel)
					fmt.Println("")
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
				case "shortcut", "shortcuts":
					usageShortcuts()
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

			if !isFile {
				fmt.Printf("Unknown command '%s'. Type /? for help\n", args[0])
				continue
			}

			sb.WriteString(line)
		default:
			sb.WriteString(line)
		}

		if sb.Len() > 0 && multiline == MultilineNone {
			opts.Prompt = sb.String()
			if multiModal {
				newPrompt, images, err := extractFileData(sb.String())
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
					sb.Reset()
					continue
				}
			}

			if err := generate(cmd, opts); err != nil {
				return err
			}

			sb.Reset()
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
	// Regex to match file paths starting with optional drive letter, / ./ \ or .\ and include escaped or unescaped spaces (\ or %20)
	// and followed by more characters and a file extension
	// This will capture non filename strings, but we'll check for file existence to remove mismatches
	regexPattern := `(?:[a-zA-Z]:)?(?:\./|/|\\)[\S\\ ]+?\.(?i:jpg|jpeg|png|svg)\b`
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

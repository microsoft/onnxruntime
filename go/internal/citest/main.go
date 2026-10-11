package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

const modulePath = "github.com/microsoft/onnxruntime/go"
const packagePath = modulePath + "/onnxruntime"

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(args []string, stdout, stderr io.Writer) (runErr error) {
	if len(args) == 0 {
		return errors.New("expected subcommand: api, results or module")
	}
	flags := flag.NewFlagSet(args[0], flag.ContinueOnError)
	flags.SetOutput(stderr)
	switch args[0] {
	case "api":
		shim := flags.String("shim", "", "shim header path")
		header := flags.String("header", "", "vendored C API header path")
		if err := flags.Parse(args[1:]); err != nil {
			return err
		}
		if flags.NArg() != 0 || *shim == "" || *header == "" {
			return errors.New("api requires -shim and -header, with no positional arguments")
		}
		s, err := openEvidence(*shim)
		if err != nil {
			return err
		}
		defer closeEvidence(s, &runErr)
		h, err := openEvidence(*header)
		if err != nil {
			return err
		}
		defer closeEvidence(h, &runErr)
		api, err := expectedAPI(s, h)
		if err != nil {
			return err
		}
		_, err = fmt.Fprintln(stdout, api)
		return err
	case "results":
		file := flags.String("file", "", "go test JSON path")
		runtime := flags.String("expect-runtime", "", "expected runtime version")
		api := flags.Int("expect-api", 0, "expected API version")
		if err := flags.Parse(args[1:]); err != nil {
			return err
		}
		if flags.NArg() != 0 || *file == "" || *runtime == "" || *api <= 0 {
			return errors.New("results requires -file, -expect-runtime and positive -expect-api, with no positional arguments")
		}
		f, err := openEvidence(*file)
		if err != nil {
			return err
		}
		defer closeEvidence(f, &runErr)
		skipped, err := inspectResults(f, *runtime, *api)
		if err != nil {
			return err
		}
		for _, name := range skipped {
			if _, err := fmt.Fprintf(stdout, "coverage limit: skipped %s\n", name); err != nil {
				return err
			}
		}
		return nil
	case "module":
		list := flags.String("list", "", "go list module JSON path")
		download := flags.String("download", "", "go mod download JSON path")
		version := flags.String("expect-version", "", "selected module version")
		commit := flags.String("expect-commit", "", "full module origin commit")
		if err := flags.Parse(args[1:]); err != nil {
			return err
		}
		if flags.NArg() != 0 || *list == "" || *download == "" || *version == "" || *commit == "" {
			return errors.New("module requires -list, -download, -expect-version and -expect-commit, with no positional arguments")
		}
		l, err := openEvidence(*list)
		if err != nil {
			return err
		}
		defer closeEvidence(l, &runErr)
		d, err := openEvidence(*download)
		if err != nil {
			return err
		}
		defer closeEvidence(d, &runErr)
		if err := checkModule(l, d, *version, *commit); err != nil {
			return err
		}
		_, err = fmt.Fprintf(stdout, "verified %s %s %s\n", modulePath, *version, *commit)
		return err
	default:
		return fmt.Errorf("unknown subcommand %q", args[0])
	}
}

func openEvidence(path string) (*os.File, error) {
	// CLI callers choose the evidence file paths; there is no restricted input root.
	return os.Open(path) //nolint:gosec // Reading caller-selected paths is the command's purpose.
}

func closeEvidence(file *os.File, result *error) {
	*result = errors.Join(*result, file.Close())
}

var sourceComments = regexp.MustCompile(`(?s)/\*.*?\*/|//[^\n]*`)
var sourceDefine = regexp.MustCompile(`(?m)^[\t ]*#[\t ]*define[\t ]+([A-Za-z_][A-Za-z_0-9]*)\b([^\n]*)`)

func definition(r io.Reader, name string) (string, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return "", err
	}
	// Preserve line boundaries so a comment cannot merge two directives.
	source := sourceComments.ReplaceAllStringFunc(string(data), func(s string) string { return " " + strings.Repeat("\n", strings.Count(s, "\n")) })
	var values []string
	for _, match := range sourceDefine.FindAllStringSubmatch(source, -1) {
		if match[1] == name {
			values = append(values, strings.TrimSpace(match[2]))
		}
	}
	if len(values) != 1 {
		return "", fmt.Errorf("expected one %s definition, found %d", name, len(values))
	}
	return values[0], nil
}

func expectedAPI(shim, header io.Reader) (int, error) {
	data, err := io.ReadAll(shim)
	if err != nil {
		return 0, err
	}
	floor, err := definition(strings.NewReader(string(data)), "ORT_GO_API_VERSION_MIN")
	if err != nil {
		return 0, err
	}
	if floor != "17" {
		return 0, fmt.Errorf("unsupported API floor %q: expected 17", floor)
	}
	cap, err := definition(strings.NewReader(string(data)), "ORT_GO_API_VERSION_MAX")
	if err != nil {
		return 0, err
	}
	switch cap {
	case "27":
		return 27, nil
	case "ORT_API_VERSION":
		value, err := definition(header, "ORT_API_VERSION")
		if err != nil {
			return 0, err
		}
		if !regexp.MustCompile(`^[0-9]+$`).MatchString(value) {
			return 0, fmt.Errorf("non-numeric ORT_API_VERSION %q", value)
		}
		api, err := strconv.Atoi(value)
		if err != nil || api < 31 {
			return 0, fmt.Errorf("unsupported ORT_API_VERSION %q: expected at least 31", value)
		}
		return 31, nil
	default:
		return 0, fmt.Errorf("unsupported API cap %q", cap)
	}
}

type testEvent struct {
	Action  string
	Package string
	Test    string
	Output  string
}

var requiredTests = []string{"TestGetVersion", "TestRunAddModel", "TestNewSessionUnicodePath", "TestSessionOptionsProfiling", "TestInitRace"}
var versionReport = regexp.MustCompile(`^ORT version: ([^\s,]+), API version: ([0-9]+)$`)

func checkResults(r io.Reader, expectedRuntime string, expectedAPI int) error {
	_, err := inspectResults(r, expectedRuntime, expectedAPI)
	return err
}

func inspectResults(r io.Reader, expectedRuntime string, expectedAPI int) ([]string, error) {
	if expectedRuntime == "" || expectedAPI <= 0 {
		return nil, errors.New("runtime and positive API expectations are required")
	}
	decoder := json.NewDecoder(r)
	states := make(map[string]string)
	var version strings.Builder
	var output strings.Builder
	var skipped []string
	packagePassed := false
	for {
		var event testEvent
		err := decoder.Decode(&event)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("decode test event: %w", err)
		}
		if event.Package != packagePath {
			return nil, fmt.Errorf("unexpected test package %q", event.Package)
		}
		if packagePassed {
			return nil, errors.New("event after final package pass")
		}
		output.WriteString(event.Output)
		if event.Action == "fail" {
			return nil, fmt.Errorf("test failure: %s", event.Test)
		}
		if event.Test == "TestGetVersion" {
			version.WriteString(event.Output)
		}
		switch event.Action {
		case "start":
			if event.Test != "" {
				return nil, errors.New("test start event must identify the package")
			}
		case "run":
			if event.Test == "" || states[event.Test] != "" {
				return nil, fmt.Errorf("invalid or duplicate run: %s", event.Test)
			}
			states[event.Test] = "run"
		case "pass", "skip":
			if event.Test == "" {
				if event.Action != "pass" {
					return nil, errors.New("package skipped")
				}
				packagePassed = true
			} else {
				if states[event.Test] != "run" {
					return nil, fmt.Errorf("terminal event without run or duplicate terminal event: %s", event.Test)
				}
				states[event.Test] = event.Action
				if event.Action == "skip" {
					skipped = append(skipped, event.Test)
				}
			}
		case "output", "pause", "cont":
		default:
			return nil, fmt.Errorf("unknown test action %q", event.Action)
		}
	}
	if !packagePassed {
		return nil, errors.New("missing final package pass")
	}
	if strings.Contains(output.String(), "(cached)") {
		return nil, errors.New("cached test output")
	}
	for _, name := range requiredTests {
		if states[name] != "pass" {
			return nil, fmt.Errorf("required test did not run and pass: %s", name)
		}
	}
	reports := 0
	for _, line := range strings.Split(version.String(), "\n") {
		index := strings.Index(line, "ORT version")
		if index < 0 {
			continue
		}
		reports++
		match := versionReport.FindStringSubmatch(strings.TrimSpace(line[index:]))
		if match == nil {
			return nil, fmt.Errorf("malformed runtime/API report: %q", line)
		}
		if match[1] != expectedRuntime {
			return nil, fmt.Errorf("runtime mismatch: got %s, expected %s", match[1], expectedRuntime)
		}
		api, err := strconv.Atoi(match[2])
		if err != nil || api != expectedAPI {
			return nil, fmt.Errorf("API mismatch: got %s, expected %d", match[2], expectedAPI)
		}
	}
	if reports != 1 {
		return nil, fmt.Errorf("expected one runtime/API report, found %d", reports)
	}
	sort.Strings(skipped)
	return skipped, nil
}

type moduleMetadata struct {
	Path    string
	Version string
	Replace json.RawMessage
	Error   json.RawMessage
	Origin  *struct {
		VCS    string
		URL    string
		Subdir string
		Hash   string
	}
}

func decodeModule(r io.Reader) (moduleMetadata, error) {
	var metadata moduleMetadata
	decoder := json.NewDecoder(r)
	if err := decoder.Decode(&metadata); err != nil {
		return metadata, fmt.Errorf("decode module metadata: %w", err)
	}
	var extra json.RawMessage
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		return metadata, errors.New("trailing module JSON data")
	}
	if metadata.Path != modulePath {
		return metadata, fmt.Errorf("module path mismatch: %q", metadata.Path)
	}
	if len(metadata.Replace) > 0 {
		return metadata, errors.New("module replacement is forbidden")
	}
	if len(metadata.Error) > 0 {
		return metadata, errors.New("module metadata contains Error")
	}
	return metadata, nil
}

func checkModule(list, download io.Reader, expectedVersion, expectedCommit string) error {
	if expectedVersion == "" {
		return errors.New("expected module version is empty")
	}
	if !regexp.MustCompile(`^[0-9a-f]{40}$`).MatchString(expectedCommit) {
		return errors.New("expected commit must be a full 40-character lowercase hexadecimal hash")
	}
	l, err := decodeModule(list)
	if err != nil {
		return fmt.Errorf("list: %w", err)
	}
	d, err := decodeModule(download)
	if err != nil {
		return fmt.Errorf("download: %w", err)
	}
	if l.Version != expectedVersion || d.Version != expectedVersion {
		return fmt.Errorf("module version mismatch: list %q, download %q, expected %q", l.Version, d.Version, expectedVersion)
	}
	if d.Origin == nil {
		return errors.New("missing download origin metadata")
	}
	if d.Origin.VCS != "git" || d.Origin.URL != "https://github.com/microsoft/onnxruntime" || d.Origin.Subdir != "go" {
		return errors.New("module origin must be Git https://github.com/microsoft/onnxruntime subdirectory go")
	}
	if d.Origin.Hash != expectedCommit {
		return fmt.Errorf("origin mismatch: got %q, expected %s", d.Origin.Hash, expectedCommit)
	}
	return nil
}

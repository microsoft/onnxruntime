package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const testCommit = "402979ac1bef89dcd55add41eaa7277c407599eb"
const testVersion = "v0.0.0-20261008210605-402979ac1bef"

func TestExpectedAPI(t *testing.T) {
	shim := "#define ORT_GO_API_VERSION_MIN 17\n#define ORT_GO_API_VERSION_MAX 27\n"
	symbolic := strings.ReplaceAll(shim, "MAX 27", "MAX ORT_API_VERSION")
	tests := []struct {
		name, shim, header string
		want               int
	}{
		{"main", shim, "#define ORT_API_VERSION 29", 27},
		{"upgrade", symbolic, "#define ORT_API_VERSION 32", 31},
		{"api31", symbolic, "#define ORT_API_VERSION 31", 31},
		{"comments", "/* #define ORT_GO_API_VERSION_MAX 26 */\n # define ORT_GO_API_VERSION_MIN 17 // floor\n\t#define ORT_GO_API_VERSION_MAX 27 /* cap */", "", 27},
		{"missing max", "#define ORT_GO_API_VERSION_MIN 17", "", 0},
		{"duplicate max", shim + "#define ORT_GO_API_VERSION_MAX 27", "", 0},
		{"duplicate min", shim + "#define ORT_GO_API_VERSION_MIN 17", "", 0},
		{"missing min", "#define ORT_GO_API_VERSION_MAX 27", "", 0},
		{"floor", strings.ReplaceAll(shim, "MIN 17", "MIN 18"), "", 0},
		{"expression", strings.ReplaceAll(shim, "MAX 27", "MAX (27)"), "", 0},
		{"cap26", strings.ReplaceAll(shim, "MAX 27", "MAX 26"), "", 0},
		{"old header", symbolic, "#define ORT_API_VERSION 30", 0},
		{"missing header", symbolic, "", 0},
		{"duplicate header", symbolic, "#define ORT_API_VERSION 32\n#define ORT_API_VERSION 32", 0},
		{"symbolic header", symbolic, "#define ORT_API_VERSION CURRENT", 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := expectedAPI(strings.NewReader(tt.shim), strings.NewReader(tt.header))
			if (err != nil) != (tt.want == 0) || got != tt.want {
				t.Fatalf("got %d, %v; want %d", got, err, tt.want)
			}
		})
	}
}

func resultFixture(api int) []testEvent {
	events := []testEvent{{Action: "start", Package: packagePath}}
	for _, name := range []string{"TestGetVersion", "TestRunAddModel", "TestNewSessionUnicodePath", "TestSessionOptionsProfiling", "TestInitRace"} {
		events = append(events, testEvent{Action: "run", Package: packagePath, Test: name})
		if name == "TestGetVersion" {
			events = append(events, testEvent{Action: "output", Package: packagePath, Test: name, Output: "    ort_test.go:58: ORT version: 1.31.0, API version: "}, testEvent{Action: "output", Package: packagePath, Test: name, Output: jsonNumber(api) + "\n"})
		}
		events = append(events, testEvent{Action: "pass", Package: packagePath, Test: name})
	}
	return append(events, testEvent{Action: "pass", Package: packagePath})
}
func jsonNumber(n int) string { b, _ := json.Marshal(n); return string(b) }
func eventStream(events []testEvent) string {
	var b strings.Builder
	for _, e := range events {
		data, _ := json.Marshal(e)
		b.Write(data)
		b.WriteByte('\n')
	}
	return b.String()
}
func removeTest(events []testEvent, name string) []testEvent {
	var out []testEvent
	for _, e := range events {
		if e.Test != name {
			out = append(out, e)
		}
	}
	return out
}

func TestCheckResults(t *testing.T) {
	type resultCase struct {
		name, stream string
		api          int
		ok           bool
	}
	good := eventStream(resultFixture(27))
	tests := []resultCase{
		{"api27", good, 27, true}, {"api31", eventStream(resultFixture(31)), 31, true},
		{"empty", "", 27, false}, {"package only", eventStream([]testEvent{{Action: "pass", Package: packagePath}}), 27, false},
		{"no tests", eventStream([]testEvent{{Action: "output", Package: packagePath, Output: "testing: warning: no tests to run\n"}, {Action: "pass", Package: packagePath}}), 27, false},
		{"wrong api", good, 28, false}, {"wrong runtime", strings.ReplaceAll(good, "1.31.0", "1.30.0"), 27, false},
		{"malformed", good + "garbage", 27, false}, {"truncated", good + `{"Action":`, 27, false},
		{"missing final", eventStream(resultFixture(27)[:len(resultFixture(27))-1]), 27, false},
		{"cached", strings.ReplaceAll(good, `"Output":""`, `"Output":"ok (cached)"`), 27, false},
		{"wrong package", strings.ReplaceAll(good, packagePath, "other/package"), 27, false},
		{"missing package", strings.ReplaceAll(good, packagePath, ""), 27, false},
		{"missing report", strings.ReplaceAll(good, "ORT version:", "Version:"), 27, false},
		{"malformed report", strings.ReplaceAll(good, "API version:", "API:"), 27, false},
		{"malformed extra report", strings.ReplaceAll(good, "ORT version:", `ORT version 1.31.0\nORT version:`), 27, false},
		{"null record", good + "null", 27, false},
	}
	for _, name := range []string{"TestGetVersion", "TestRunAddModel", "TestNewSessionUnicodePath", "TestSessionOptionsProfiling", "TestInitRace"} {
		tests = append(tests, resultCase{"missing " + name, eventStream(removeTest(resultFixture(27), name)), 27, false})
		events := resultFixture(27)
		for i := range events {
			if events[i].Test == name && events[i].Action == "pass" {
				events[i].Action = "skip"
			}
		}
		tests = append(tests, resultCase{"skipped " + name, eventStream(events), 27, false})
	}
	mutations := []struct {
		name string
		edit func([]testEvent) []testEvent
		ok   bool
	}{
		{"fragmented cache", func(es []testEvent) []testEvent {
			return append(es[:len(es)-1], testEvent{Action: "output", Package: packagePath, Output: "ok (cac"}, testEvent{Action: "output", Package: packagePath, Output: "hed)\n"}, es[len(es)-1])
		}, false},
		{"subtest only", func(es []testEvent) []testEvent {
			for i := range es {
				if es[i].Test == "TestRunAddModel" {
					es[i].Test += "/child"
				}
			}
			return es
		}, false},
		{"other test report", func(es []testEvent) []testEvent {
			for i := range es {
				if es[i].Action == "output" {
					es[i].Test = "TestRunAddModel"
				}
			}
			return es
		}, false},
		{"optional skip", func(es []testEvent) []testEvent {
			return append(es[:len(es)-1], testEvent{Action: "run", Package: packagePath, Test: "TestOptional"}, testEvent{Action: "skip", Package: packagePath, Test: "TestOptional"}, es[len(es)-1])
		}, true},
		{"package fail", func(es []testEvent) []testEvent { es[len(es)-1].Action = "fail"; return es }, false},
		{"package skip", func(es []testEvent) []testEvent { es[len(es)-1].Action = "skip"; return es }, false},
		{"trailing fail", func(es []testEvent) []testEvent { return append(es, testEvent{Action: "fail", Package: packagePath}) }, false},
		{"duplicate terminal", func(es []testEvent) []testEvent { return append(es, es[len(es)-1]) }, false},
		{"duplicate test pass", func(es []testEvent) []testEvent {
			return append(es[:len(es)-1], testEvent{Action: "pass", Package: packagePath, Test: "TestInitRace"}, es[len(es)-1])
		}, false},
		{"pass without run", func(es []testEvent) []testEvent {
			var out []testEvent
			for _, e := range es {
				if e.Test == "TestInitRace" && e.Action == "run" {
					continue
				}
				out = append(out, e)
			}
			return out
		}, false},
		{"duplicate report", func(es []testEvent) []testEvent {
			return append(es[:len(es)-1], testEvent{Action: "output", Package: packagePath, Test: "TestGetVersion", Output: "ORT version: 1.31.0, API version: 27\n"}, es[len(es)-1])
		}, false},
		{"conflicting report", func(es []testEvent) []testEvent {
			return append(es[:len(es)-1], testEvent{Action: "output", Package: packagePath, Test: "TestGetVersion", Output: "ORT version: 1.31.0, API version: 31\n"}, es[len(es)-1])
		}, false},
	}
	for _, m := range mutations {
		tests = append(tests, resultCase{m.name, eventStream(m.edit(resultFixture(27))), 27, m.ok})
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := checkResults(strings.NewReader(tt.stream), "1.31.0", tt.api)
			if (err == nil) != tt.ok {
				t.Fatalf("error=%v; want success=%v", err, tt.ok)
			}
		})
	}
}

func moduleFixture() (string, string) {
	return `{"Path":"` + modulePath + `","Version":"` + testVersion + `"}`, `{"Path":"` + modulePath + `","Version":"` + testVersion + `","Origin":{"VCS":"git","URL":"https://github.com/microsoft/onnxruntime","Subdir":"go","Hash":"` + testCommit + `"}}`
}
func TestCheckModule(t *testing.T) {
	type moduleCase struct {
		name, list, download, version, commit string
		ok                                    bool
	}
	list, download := moduleFixture()
	tests := []moduleCase{{"valid", list, download, testVersion, testCommit, true}}
	for _, field := range []string{"Path", "Version"} {
		for _, side := range []string{"list", "download"} {
			for _, value := range []string{"", "wrong", "missing"} {
				l, d := list, download
				var obj map[string]any
				source := l
				if side == "download" {
					source = d
				}
				if err := json.Unmarshal([]byte(source), &obj); err != nil {
					t.Fatal(err)
				}
				if value == "missing" {
					delete(obj, field)
				} else {
					obj[field] = value
				}
				b, _ := json.Marshal(obj)
				if side == "list" {
					l = string(b)
				} else {
					d = string(b)
				}
				tests = append(tests, moduleCase{side + field + value, l, d, testVersion, testCommit, false})
			}
		}
	}
	for _, side := range []string{"list", "download"} {
		for _, field := range []string{"Replace", "Error"} {
			l, d := list, download
			extra := `,"` + field + `":{}}`
			if side == "list" {
				l = strings.TrimSuffix(l, "}") + extra
			} else {
				d = strings.TrimSuffix(d, "}") + extra
			}
			tests = append(tests, moduleCase{side + field, l, d, testVersion, testCommit, false})
		}
	}
	for _, field := range []string{"VCS", "URL", "Subdir", "Hash"} {
		for _, value := range []string{"", "wrong", "missing"} {
			var obj map[string]any
			if err := json.Unmarshal([]byte(download), &obj); err != nil {
				t.Fatal(err)
			}
			origin := obj["Origin"].(map[string]any)
			if value == "missing" {
				delete(origin, field)
			} else {
				origin[field] = value
			}
			b, _ := json.Marshal(obj)
			tests = append(tests, moduleCase{field + value, list, string(b), testVersion, testCommit, false})
		}
	}
	for _, tt := range []struct{ name, l, d, v, c string }{
		{"missing origin", list, list, testVersion, testCommit}, {"short expected hash", list, download, testVersion, testCommit[:12]},
		{"wrong expected hash", list, download, testVersion, strings.Repeat("a", 40)}, {"empty expected version", list, download, "", testCommit},
		{"wrong expected version", list, download, "v1.31.0", testCommit}, {"bad list", "{", download, testVersion, testCommit},
		{"bad download", list, "{", testVersion, testCommit}, {"trailing list", list + "{}", download, testVersion, testCommit},
		{"trailing download", list, download + "{}", testVersion, testCommit}, {"null list", "null", download, testVersion, testCommit},
	} {
		tests = append(tests, moduleCase{tt.name, tt.l, tt.d, tt.v, tt.c, false})
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := checkModule(strings.NewReader(tt.list), strings.NewReader(tt.download), tt.version, tt.commit)
			if (err == nil) != tt.ok {
				t.Fatalf("error=%v; want success=%v", err, tt.ok)
			}
		})
	}
}

func TestRun(t *testing.T) {
	dir := t.TempDir()
	write := func(name, data string) string {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, []byte(data), 0600); err != nil {
			t.Fatal(err)
		}
		return p
	}
	shim := write("shim.h", "#define ORT_GO_API_VERSION_MIN 17\n#define ORT_GO_API_VERSION_MAX 27\n")
	header := write("header.h", "")
	events := resultFixture(27)
	events = append(events[:len(events)-1], testEvent{Action: "run", Package: packagePath, Test: "TestOptional"}, testEvent{Action: "skip", Package: packagePath, Test: "TestOptional"}, events[len(events)-1])
	results := write("results.json", eventStream(events))
	l, d := moduleFixture()
	list := write("list.json", l)
	download := write("download.json", d)
	tests := []struct {
		name   string
		args   []string
		ok     bool
		output string
	}{
		{"api", []string{"api", "-shim", shim, "-header", header}, true, "27\n"},
		{"results", []string{"results", "-file", results, "-expect-runtime", "1.31.0", "-expect-api", "27"}, true, "coverage limit: skipped TestOptional\n"},
		{"module", []string{"module", "-list", list, "-download", download, "-expect-version", testVersion, "-expect-commit", testCommit}, true, "verified " + modulePath + " " + testVersion + " " + testCommit + "\n"},
		{"no command", nil, false, ""}, {"unknown", []string{"unknown"}, false, ""}, {"bad flag", []string{"api", "-bad"}, false, ""},
		{"missing flags", []string{"api"}, false, ""}, {"missing file", []string{"api", "-shim", "missing", "-header", header}, false, ""},
		{"extra args", []string{"api", "-shim", shim, "-header", header, "extra"}, false, ""},
		{"missing result expectations", []string{"results", "-file", results}, false, ""},
		{"api mismatch", []string{"results", "-file", results, "-expect-runtime", "1.31.0", "-expect-api", "28"}, false, ""},
		{"missing module flags", []string{"module"}, false, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var out, errs bytes.Buffer
			err := run(tt.args, &out, &errs)
			if (err == nil) != tt.ok {
				t.Fatalf("error=%v; want success=%v", err, tt.ok)
			}
			if tt.ok && errs.Len() != 0 {
				t.Fatalf("unexpected stderr: %s", &errs)
			}
			if tt.ok && out.String() != tt.output {
				t.Fatalf("output %q; want %q", out.String(), tt.output)
			}
		})
	}
}

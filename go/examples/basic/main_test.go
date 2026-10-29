package main

import (
	"bytes"
	"fmt"
	"math"
	"runtime/debug"
	"strings"
	"testing"
)

func TestCheckOutput(t *testing.T) {
	good := []float32{11, 22, 33, 44, 55, 66}
	cases := []struct {
		name    string
		shape   []int64
		values  []float32
		wantErr bool
	}{
		{"valid", []int64{2, 3}, good, false},
		{"nil shape", nil, good, true},
		{"rank", []int64{6}, good, true},
		{"dimensions", []int64{3, 2}, good, true},
		{"short", []int64{2, 3}, good[:5], true},
		{"long", []int64{2, 3}, append(append([]float32{}, good...), 77), true},
	}
	for i := range good {
		for _, bad := range []float32{good[i] + 1, float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
			values := append([]float32{}, good...)
			values[i] = bad
			cases = append(cases, struct {
				name    string
				shape   []int64
				values  []float32
				wantErr bool
			}{fmt.Sprintf("value %d %v", i, bad), []int64{2, 3}, values, true})
		}
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := checkOutput(tc.shape, tc.values); (err != nil) != tc.wantErr {
				t.Fatalf("checkOutput() = %v, want error %v", err, tc.wantErr)
			}
		})
	}
}

func TestCheckRuntime(t *testing.T) {
	for _, tc := range []struct {
		name, version   string
		api             int
		expectedVersion string
		expectedAPI     int
		diagnostic      string
	}{
		{"valid", "1.31.0", 27, "1.31.0", 27, ""},
		{"optional", "1.31.0", 27, "", 0, ""},
		{"runtime", "1.31.0", 27, "0.0.0", 27, "runtime mismatch"},
		{"api", "1.31.0", 27, "1.31.0", 28, "API mismatch"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := checkRuntime(tc.version, tc.api, tc.expectedVersion, tc.expectedAPI)
			if tc.diagnostic == "" {
				if err != nil {
					t.Fatal(err)
				}
			} else if err == nil || !strings.Contains(err.Error(), tc.diagnostic) {
				t.Fatalf("error = %v, want %q", err, tc.diagnostic)
			}
		})
	}
}

func TestCheckBuildInfo(t *testing.T) {
	const version = "v0.0.0-20261008210605-402979ac1bef"
	for _, tc := range []struct {
		name    string
		info    *debug.BuildInfo
		wantErr bool
	}{
		{"valid", &debug.BuildInfo{Deps: []*debug.Module{{Path: "other", Version: "v1"}, {Path: modulePath, Version: version}}}, false},
		{"nil", nil, true},
		{"missing", &debug.BuildInfo{}, true},
		{"main module", &debug.BuildInfo{Main: debug.Module{Path: modulePath, Version: version}}, true},
		{"wrong path", &debug.BuildInfo{Deps: []*debug.Module{{Path: modulePath + "/onnxruntime", Version: version}}}, true},
		{"wrong version", &debug.BuildInfo{Deps: []*debug.Module{{Path: modulePath, Version: "v1.0.0"}}}, true},
		{"empty version", &debug.BuildInfo{Deps: []*debug.Module{{Path: modulePath}}}, true},
		{"replacement", &debug.BuildInfo{Deps: []*debug.Module{{Path: modulePath, Version: version, Replace: &debug.Module{Path: "../local"}}}}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := checkBuildInfo(tc.info, version); (err != nil) != tc.wantErr {
				t.Fatalf("checkBuildInfo() = %v, want error %v", err, tc.wantErr)
			}
		})
	}
}

func TestRunArguments(t *testing.T) {
	t.Setenv("ORT_LIB_PATH", "")
	for _, tc := range []struct {
		name       string
		args       []string
		diagnostic string
	}{
		{"unknown", []string{"-unknown"}, "flag provided but not defined"},
		{"bad api", []string{"-expect-api", "bad"}, "invalid value"},
		{"negative api", []string{"-expect-api", "-1", "model.onnx"}, "expect-api"},
		{"missing model", nil, "one model path"},
		{"extra model", []string{"a", "b"}, "one model path"},
		{"missing library", []string{"model.onnx"}, "ORT_LIB_PATH"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			err := run(tc.args, &stdout, &stderr)
			if err == nil || !strings.Contains(err.Error(), tc.diagnostic) {
				t.Fatalf("error = %v, want %q", err, tc.diagnostic)
			}
			if stdout.Len() != 0 {
				t.Fatalf("unexpected success output %q", stdout.String())
			}
		})
	}
	var stdout, stderr bytes.Buffer
	if err := run([]string{"-h"}, &stdout, &stderr); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(stderr.String(), "main module") {
		t.Fatalf("usage must explain module flag: %q", stderr.String())
	}
}

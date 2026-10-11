package onnxruntime

import (
	"bytes"
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// initRaceEnv marks the subprocess spawned by TestInitRace. When set, TestMain
// leaves the process uninitialized so TestInitRaceChild can race a cold-start
// Init against concurrent readers.
const initRaceEnv = "ORT_TEST_INIT_RACE"

const initRetryEnv = "ORT_TEST_INIT_RETRY"

var testRuntimePath string

func TestMain(m *testing.M) {
	// Native telemetry can write files during CreateEnv, before DisableTelemetry.
	if err := os.Setenv("ORT_RUNNING_UNIT_TESTS", "1"); err != nil {
		println("failed to suppress test telemetry:", err.Error())
		os.Exit(1)
	}
	libPath := os.Getenv("ORT_LIB_PATH")
	if libPath == "" {
		libPath = ".ort-lib/lib"
		if _, err := os.Stat(libPath); err != nil {
			println("failed to init ORT: set ORT_LIB_PATH to the runtime library; fallback .ort-lib/lib is unavailable:", err.Error())
			os.Exit(1)
		}
	}
	var err error
	testRuntimePath, err = filepath.Abs(resolveDir(libPath))
	if err != nil {
		println("failed to init ORT: resolve runtime path:", err.Error())
		os.Exit(1)
	}
	if _, err := os.Stat(testRuntimePath); err != nil {
		println("failed to init ORT: runtime library is unavailable:", err.Error())
		os.Exit(1)
	}
	SetSharedLibraryPath(testRuntimePath)
	if os.Getenv(initRaceEnv) != "1" && os.Getenv(initRetryEnv) != "1" {
		if err := Init(); err != nil {
			println("failed to init ORT:", err.Error())
			os.Exit(1)
		}
	}
	code := m.Run()
	if !IsInitialized() {
		println("failed to init ORT: tests did not initialize the runtime")
		code = 1
	}
	if err := Shutdown(); err != nil {
		println("failed to shut down ORT:", err.Error())
		code = 1
	}
	os.Exit(code)
}

func runtimeTestEnvironment(runtimePath, childMode string) []string {
	env := make([]string, 0, len(os.Environ())+2)
	for _, entry := range os.Environ() {
		key, _, _ := strings.Cut(entry, "=")
		if strings.EqualFold(key, "ORT_LIB_PATH") || strings.HasPrefix(strings.ToUpper(key), "ORT_TEST_") {
			continue
		}
		env = append(env, entry)
	}
	if runtimePath != "" {
		env = append(env, "ORT_LIB_PATH="+runtimePath)
	}
	if childMode != "" {
		env = append(env, childMode+"=1")
	}
	return env
}

func runRuntimeChild(t *testing.T, dir, runtimePath, childMode, selector string, extraEnv ...string) ([]byte, error) {
	t.Helper()
	binary, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, binary, "-test.run="+selector, "-test.v")
	cmd.Dir = dir
	cmd.Env = append(runtimeTestEnvironment(runtimePath, childMode), extraEnv...)
	out, err := cmd.CombinedOutput()
	if ctx.Err() != nil {
		t.Fatalf("runtime subprocess timed out: %v\n%s", ctx.Err(), out)
	}
	return out, err
}

func TestTestMainRuntimeRequired(t *testing.T) {
	dir := t.TempDir()
	invalid := filepath.Join(dir, platformLibraryName())
	if err := os.WriteFile(invalid, []byte("not a runtime library"), 0600); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name, path, mode, selector, diagnostic string
		code                                   int
	}{
		{"unset", "", "", "^TestGetVersion$", "fallback .ort-lib/lib is unavailable", 1},
		{"missing", filepath.Join(dir, "missing"), "", "^TestGetVersion$", "runtime library is unavailable", 1},
		{"invalid", invalid, "", "^TestGetVersion$", "load library", 1},
		{"valid", testRuntimePath, "", "^TestGetVersion$", "--- PASS: TestGetVersion", 0},
		{"missing_no_match", filepath.Join(dir, "missing"), "", "^NoRuntimeTestMatches$", "runtime library is unavailable", 1},
		{"cold_race_no_match", testRuntimePath, initRaceEnv, "^NoRuntimeTestMatches$", "tests did not initialize the runtime", 1},
		{"cold_retry_no_match", testRuntimePath, initRetryEnv, "^NoRuntimeTestMatches$", "tests did not initialize the runtime", 1},
		{"cold_missing", filepath.Join(dir, "missing"), initRaceEnv, "^TestInitRaceChild$", "runtime library is unavailable", 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out, err := runRuntimeChild(t, dir, tc.path, tc.mode, tc.selector)
			code := 0
			if err != nil {
				var exitErr *exec.ExitError
				if !errors.As(err, &exitErr) {
					t.Fatalf("subprocess did not exit normally: %v\n%s", err, out)
				}
				code = exitErr.ExitCode()
			}
			if code != tc.code || !bytes.Contains(out, []byte(tc.diagnostic)) {
				t.Fatalf("expected exit %d and %q, got exit %d:\n%s", tc.code, tc.diagnostic, code, out)
			}
		})
	}
}

func TestTestMainNoTelemetryFiles(t *testing.T) {
	for _, tc := range []struct {
		name, mode, selector string
	}{
		{"normal", "", "^Test(GetVersion|Telemetry)$"},
		{"cold", initRaceEnv, "^TestInitRaceChild$"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			out, err := runRuntimeChild(t, dir, testRuntimePath, tc.mode, tc.selector,
				"ORT_RUNNING_UNIT_TESTS=0", "HOME="+os.DevNull, "XDG_DATA_HOME="+os.DevNull)
			if err != nil || !bytes.Contains(out, []byte("--- PASS:")) {
				t.Fatalf("telemetry child: %v\n%s", err, out)
			}
			entries, err := os.ReadDir(dir)
			if err != nil {
				t.Fatal(err)
			}
			for _, entry := range entries {
				t.Errorf("runtime left file %q in fresh working directory", entry.Name())
			}
		})
	}
}

func TestInitIdempotent(t *testing.T) {
	if err := Init(); err != nil {
		t.Fatalf("second Init should succeed: %v", err)
	}
}

func TestIsInitialized(t *testing.T) {
	if !IsInitialized() {
		t.Fatal("expected initialized")
	}
}

func TestGetVersion(t *testing.T) {
	v, err := GetVersion()
	if err != nil {
		t.Fatal(err)
	}
	if v == "" {
		t.Fatal("expected non-empty version string")
	}
	t.Logf("ORT version: %s, API version: %d", v, APIVersion())
}

func TestTelemetry(t *testing.T) {
	if err := DisableTelemetry(); err != nil {
		t.Fatal(err)
	}
	if err := EnableTelemetry(); err != nil {
		t.Fatal(err)
	}
}

func TestAvailableProviders(t *testing.T) {
	providers, err := AvailableProviders()
	if err != nil {
		t.Fatal(err)
	}
	if len(providers) == 0 {
		t.Fatal("expected at least one provider")
	}
	found := false
	for _, p := range providers {
		if p == "CPUExecutionProvider" {
			found = true
		}
	}
	if !found {
		t.Errorf("expected CPUExecutionProvider, got %v", providers)
	}
}

func TestShutdownWithOpenSession(t *testing.T) {
	sess, err := NewSession(testdataPath("add_f32.onnx"), nil)
	if err != nil {
		t.Fatal(err)
	}

	err = Shutdown()
	if err == nil {
		t.Fatal("expected error when shutting down with open session")
	}
	sess.Close()
}

func TestInitBadPathRetryable(t *testing.T) {
	if os.Getenv(initRetryEnv) == "1" {
		if IsInitialized() {
			t.Fatal("retry subprocess must start uninitialized")
		}
		SetSharedLibraryPath(filepath.Join(t.TempDir(), "missing"))
		if err := Init(); err == nil || !strings.Contains(err.Error(), "load library") {
			t.Fatalf("expected missing-library load error, got %v", err)
		}
		if IsInitialized() || APIVersion() != 0 {
			t.Fatal("failed Init published initialization state")
		}
		if _, err := GetVersion(); !errors.Is(err, errNotInitialized) {
			t.Fatalf("expected uninitialized version error, got %v", err)
		}
		SetSharedLibraryPath(testRuntimePath)
		if err := Init(); err != nil {
			t.Fatalf("retry Init failed: %v", err)
		}
		if !IsInitialized() || APIVersion() == 0 {
			t.Fatal("retry did not publish initialization state")
		}
		if version, err := GetVersion(); err != nil || version == "" {
			t.Fatalf("retry version: %q, %v", version, err)
		}
		return
	}
	out, err := runRuntimeChild(t, t.TempDir(), testRuntimePath, initRetryEnv, "^TestInitBadPathRetryable$")
	if err != nil || !bytes.Contains(out, []byte("--- PASS: TestInitBadPathRetryable")) {
		t.Fatalf("init retry subprocess failed: %v\n%s", err, out)
	}
}

// TestCheckInitConcurrent is a read-only smoke test: the process is already
// initialized, so nothing writes the init state here. The cold-start write vs.
// read race that the atomic init flags guard is covered by TestInitRace.
func TestCheckInitConcurrent(t *testing.T) {
	const goroutines = 8
	var wg sync.WaitGroup
	wg.Add(goroutines)
	for i := 0; i < goroutines; i++ {
		go func() {
			defer wg.Done()
			for j := 0; j < 200; j++ {
				_ = IsInitialized()
				_ = APIVersion()
				_, _ = GetVersion()
				if mi, err := NewCPUMemoryInfo(); err == nil {
					_ = mi.Close()
				}
			}
		}()
	}
	wg.Wait()
}

// TestInitRace re-runs this test binary with ORT uninitialized so that a
// cold-start Init runs concurrently with readers. Init writes the init flags
// under mu, but checkInit reads them without it, so the flags must be atomic:
// under -race, plain bools report a DATA RACE here.
func TestInitRace(t *testing.T) {
	out, err := runRuntimeChild(t, t.TempDir(), testRuntimePath, initRaceEnv, "^TestInitRaceChild$")
	if bytes.Contains(out, []byte("DATA RACE")) {
		t.Fatalf("data race between Init and concurrent readers:\n%s", out)
	}
	if err != nil || !bytes.Contains(out, []byte("--- PASS: TestInitRaceChild")) {
		t.Fatalf("init race subprocess failed: %v\n%s", err, out)
	}
}

// TestInitRaceChild runs only inside the subprocess spawned by TestInitRace.
func TestInitRaceChild(t *testing.T) {
	if os.Getenv(initRaceEnv) != "1" {
		t.Skipf("only runs as the %s=1 subprocess of TestInitRace", initRaceEnv)
	}
	if IsInitialized() {
		t.Fatal("subprocess must start uninitialized")
	}

	const (
		initers = 4
		readers = 8
	)
	start := make(chan struct{})
	errs := make(chan error, initers)
	var wg sync.WaitGroup

	wg.Add(initers + readers)
	for i := 0; i < initers; i++ {
		go func() {
			defer wg.Done()
			<-start
			errs <- Init()
		}()
	}
	for i := 0; i < readers; i++ {
		go func() {
			defer wg.Done()
			<-start
			// NewCPUMemoryInfo goes through checkInit, which reads the init
			// flags without holding mu; before Init lands it just errors.
			for j := 0; j < 500; j++ {
				_ = IsInitialized()
				_ = APIVersion()
				_, _ = GetVersion()
				if opts, err := NewSessionOptions(); err == nil {
					_, _ = opts.GetExecutionMode()
					_, _ = opts.IsMemPatternEnabled()
					if err := opts.Close(); err != nil {
						t.Errorf("close reader options: %v", err)
					}
				}
				if mi, err := NewCPUMemoryInfo(); err == nil {
					_ = mi.Close()
				}
			}
		}()
	}
	close(start)
	wg.Wait()
	close(errs)

	for err := range errs {
		if err != nil {
			t.Fatalf("concurrent Init failed: %v", err)
		}
	}
	if !IsInitialized() {
		t.Fatal("expected initialized after concurrent Init calls")
	}
}

func TestOrtErrorAs(t *testing.T) {
	_, err := NewSession("nonexistent_model.onnx", nil)
	if err == nil {
		t.Fatal("expected error for nonexistent model")
	}
	var ortErr *OrtError
	if !errors.As(err, &ortErr) {
		t.Fatalf("expected *OrtError, got %T: %v", err, err)
	}
	if ortErr.Code != ErrorCodeNoSuchFile {
		t.Errorf("expected NoSuchFile, got %s", ortErr.Code)
	}
}

func TestLoadLibraryRejectsNULPath(t *testing.T) {
	fn, handle, err := loadLibrary("library\x00ignored")
	if handle != nil {
		closeLibrary(handle)
	}
	if fn != nil {
		t.Fatal("expected nil symbol for rejected path")
	}
	if err == nil {
		t.Fatal("expected error for NUL in shared library path")
	}
}

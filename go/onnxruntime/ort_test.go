package onnxruntime

import (
	"bytes"
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
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
const initFailureEnv = "ORT_TEST_INIT_FAILURE"
const epShutdownEnv = "ORT_TEST_EP_SHUTDOWN"

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
	if os.Getenv(initRaceEnv) != "1" && os.Getenv(initRetryEnv) != "1" && os.Getenv(initFailureEnv) != "1" {
		if err := Init(); err != nil {
			println("failed to init ORT:", err.Error())
			os.Exit(1)
		}
	}
	code := m.Run()
	if !IsInitialized() && (os.Getenv(epShutdownEnv) != "1" || !shutdown.Load()) {
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
		{"cold_failure_no_match", testRuntimePath, initFailureEnv, "^NoRuntimeTestMatches$", "tests did not initialize the runtime", 1},
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
		if _, err := GetBuildInfo(); !errors.Is(err, errNotInitialized) {
			t.Fatalf("expected uninitialized build-info error, got %v", err)
		}
		var opts *SessionOptions
		for _, call := range []func() error{
			func() error { return RegisterExecutionProviderLibrary("", "") },
			func() error { return UnregisterExecutionProviderLibrary("") },
			func() error { _, err := GetEPDevices(); return err },
			func() error { return opts.AppendExecutionProviderV2(nil, nil) },
			func() error { return opts.SetEPSelectionPolicy(-1) },
		} {
			if err := call(); !errors.Is(err, errNotInitialized) {
				t.Fatalf("pre-init plugin operation: %v", err)
			}
		}

		if _, err := opts.GetExecutionMode(); !errors.Is(err, errNotInitialized) {
			t.Fatalf("pre-init getter: %v", err)
		}
		if _, err := opts.IsMemPatternEnabled(); !errors.Is(err, errNotInitialized) {
			t.Fatalf("pre-init getter: %v", err)
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

func TestAPIVersionPolicy(t *testing.T) {
	version, err := GetVersion()
	if err != nil {
		t.Fatal(err)
	}
	api := APIVersion()
	if api < 17 || api > 32 {
		t.Fatalf("API %d outside loader range [17,32]", api)
	}
	if value, set := os.LookupEnv("ORT_TEST_EXPECT_API"); set {
		expected, err := strconv.Atoi(value)
		if err != nil || expected < 17 || expected > 32 {
			t.Fatalf("invalid ORT_TEST_EXPECT_API %q", value)
		}
		if api != expected {
			t.Fatalf("expected API %d, got %d", expected, api)
		}
	}
	if value, set := os.LookupEnv("ORT_TEST_EXPECT_VERSION"); set {
		if value == "" {
			t.Fatal("invalid empty ORT_TEST_EXPECT_VERSION")
		}
		if version != value {
			t.Fatalf("expected version %q, got %q", value, version)
		}
	}
	t.Logf("ORT version: %s, API version: %d", version, api)
}

func TestInitAPINegotiationQuiet(t *testing.T) {
	version, err := GetVersion()
	if err != nil {
		t.Fatal(err)
	}
	parts := strings.Split(version, ".")
	if len(parts) != 3 || parts[0] != "1" {
		t.Fatalf("expected release version 1.N.x, got %q", version)
	}
	minor, err := strconv.Atoi(parts[1])
	if err != nil {
		t.Fatalf("invalid runtime minor in %q: %v", version, err)
	}
	expected := min(minor, apiVersionMax)
	binary, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, binary, "-test.run=^TestAPIVersionPolicy$", "-test.v")
	cmd.Dir = t.TempDir()
	cmd.Env = append(runtimeTestEnvironment(testRuntimePath, ""),
		"ORT_TEST_EXPECT_API="+strconv.Itoa(expected), "ORT_TEST_EXPECT_VERSION="+version)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("negotiation child: %v\n%s\n%s", err, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "--- PASS: TestAPIVersionPolicy") {
		t.Fatalf("negotiation child did not run the API assertion:\n%s", stdout.String())
	}
	if strings.Contains(stderr.String(), "is not available") {
		t.Fatalf("Init printed an unsupported API diagnostic:\n%s", stderr.String())
	}
}

func TestAPIFeatureGates(t *testing.T) {
	opts, err := NewSessionOptions()
	if err != nil {
		t.Fatal(err)
	}
	defer opts.Close()
	for _, tc := range []struct {
		name    string
		minimum int
		call    func() error
	}{
		{"SetEPContextDataReader", 31, func() error { return opts.SetEPContextDataReader(func(string) ([]byte, error) { return nil, nil }, 1) }},
		{"ClearEPContextDataReader", 31, func() error { return opts.ClearEPContextDataReader() }},
		{"SetDeterministicCompute", 17, func() error { return opts.SetDeterministicCompute(true) }},
		{"SynchronizeInputs", 10, func() error { return featureGateSynchronize(t, true) }},
		{"SynchronizeOutputs", 10, func() error { return featureGateSynchronize(t, false) }},
		{"SetWeightlessSourceModelBuffer", 29, func() error {
			bufferOpts, err := NewSessionOptions()
			if err != nil {
				return err
			}
			defer bufferOpts.Close()
			return bufferOpts.SetWeightlessSourceModelBuffer([]byte{1, 2, 3, 4})
		}},
		{"GetBuildInfo", 15, func() error {
			info, err := GetBuildInfo()
			if err == nil && info == "" {
				t.Error("expected nonempty build info")
			}
			return err
		}},
		{"RegisterExecutionProviderLibrary", 22, func() error {
			err := RegisterExecutionProviderLibrary("go_feature_missing", filepath.Join(t.TempDir(), "missing-plugin"))
			if APIVersion() >= 22 {
				var native *OrtError
				if !errors.As(err, &native) {
					t.Fatalf("expected native load error, got %v", err)
				}
				return nil
			}
			return err
		}},
		{"UnregisterExecutionProviderLibrary", 22, func() error {
			err := UnregisterExecutionProviderLibrary("go_feature_unknown")
			if APIVersion() >= 22 {
				var native *OrtError
				if !errors.As(err, &native) {
					t.Fatalf("expected native unregister error, got %v", err)
				}
				return nil
			}
			return err
		}},
		{"GetEPDevices", 22, func() error { _, err := GetEPDevices(); return err }},
		{"AppendExecutionProviderV2", 22, func() error {
			if APIVersion() < 22 {
				return opts.AppendExecutionProviderV2(nil, nil)
			}
			return opts.AppendExecutionProviderV2([]*EPDevice{cpuEPDevice(t)}, nil)
		}},
		{"SetEPSelectionPolicy", 22, func() error { return opts.SetEPSelectionPolicy(ExecutionProviderDevicePolicyPreferCPU) }},

		{"GetExecutionMode", 27, func() error {
			mode, err := opts.GetExecutionMode()
			if err == nil && mode != ExecutionModeSequential {
				t.Errorf("expected sequential, got %v", mode)
			}
			return err
		}},
		{"IsMemPatternEnabled", 27, func() error {
			enabled, err := opts.IsMemPatternEnabled()
			if err == nil && !enabled {
				t.Error("expected memory pattern enabled")
			}
			return err
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.call()
			if APIVersion() >= tc.minimum {
				if err != nil {
					t.Fatal(err)
				}
				return
			}
			var native *OrtError
			if !errors.As(err, &native) || native.Code != ErrorCodeNotImplemented {
				t.Fatalf("expected typed unsupported error, got %v", err)
			}
			for _, text := range []string{"requires API " + strconv.Itoa(tc.minimum), "negotiated API " + strconv.Itoa(APIVersion())} {
				if !strings.Contains(native.Msg, text) {
					t.Errorf("missing %q in %q", text, native.Msg)
				}
			}
		})
	}
}

func TestVendoredHeaders(t *testing.T) {
	canonical := filepath.Join("..", "..", "include", "onnxruntime", "core", "session")
	if _, err := os.Stat(canonical); os.IsNotExist(err) {
		if _, repoErr := os.Stat(filepath.Join("..", "..", ".git")); !os.IsNotExist(repoErr) {
			t.Fatalf("repository checkout is missing canonical headers: %v", err)
		}
		t.Skip("standalone module has no canonical source tree")
	} else if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"onnxruntime_c_api.h", "onnxruntime_ep_c_api.h", "onnxruntime_error_code.h"} {
		t.Run(name, func(t *testing.T) {
			source, err := os.ReadFile(filepath.Join(canonical, name))
			if err != nil {
				t.Fatal(err)
			}
			vendored, err := os.ReadFile(name)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(source, vendored) {
				t.Fatal("vendored header differs from canonical source")
			}
		})
	}
}

func TestGetBuildInfo(t *testing.T) {
	info, err := GetBuildInfo()
	if err != nil {
		t.Fatal(err)
	}
	if info == "" {
		t.Fatal("expected nonempty build info")
	}
	t.Log(info)
}

func TestErrorCodes(t *testing.T) {
	for _, tc := range []struct {
		code   ErrorCode
		number int
		name   string
	}{
		{ErrorCodeModelLoadCanceled, 12, "ModelLoadCanceled"},
		{ErrorCodeModelRequiresCompilation, 13, "ModelRequiresCompilation"},
		{ErrorCodeNotFound, 14, "NotFound"},
		{ErrorCodeDeviceReset, 15, "DeviceReset"},
		{ErrorCode(999), 999, "ErrorCode(999)"},
	} {
		if int(tc.code) != tc.number || tc.code.String() != tc.name {
			t.Errorf("code %d: got %s, want %s", tc.code, tc.code, tc.name)
		}
	}
}

// The fixture negotiates a real ABI table layout but fails before publishing Init.
// It needs no link-time or run-time dependency on a second ORT library.
func TestInitEnvironmentFailureRetryable(t *testing.T) {
	if os.Getenv(initFailureEnv) == "1" {
		if IsInitialized() || APIVersion() != 0 {
			t.Fatal("child must start uninitialized")
		}
		SetSharedLibraryPath(os.Getenv("ORT_TEST_INIT_FAILURE_LIBRARY"))
		err := Init()
		if err == nil || !strings.Contains(err.Error(), "injected initialization failure") {
			t.Fatalf("expected injected failure, got %v", err)
		}
		if IsInitialized() || APIVersion() != 0 {
			t.Fatalf("failed Init leaked API %d", APIVersion())
		}
		if _, err := GetVersion(); !errors.Is(err, errNotInitialized) {
			t.Fatalf("version after failed Init: %v", err)
		}
		if _, err := GetBuildInfo(); !errors.Is(err, errNotInitialized) {
			t.Fatalf("build info after failed Init: %v", err)
		}
		SetSharedLibraryPath(testRuntimePath)
		if err := Init(); err != nil {
			t.Fatal(err)
		}
		version, err := GetVersion()
		if err != nil || version == "" || version == "injected-runtime" || APIVersion() == 0 {
			t.Fatalf("retry version %q API %d: %v", version, APIVersion(), err)
		}
		return
	}
	ccOutput, err := exec.Command("go", "env", "CC").Output()
	if err != nil {
		t.Fatalf("get cgo compiler: %v", err)
	}
	compiler, err := splitCompilerCommand(string(ccOutput))
	if err != nil {
		t.Fatalf("parse cgo compiler: %v", err)
	}
	if len(compiler) == 0 {
		t.Fatal("empty cgo compiler")
	}
	include, err := filepath.Abs(".")
	if err != nil {
		t.Fatal(err)
	}
	for _, memoryFailure := range []bool{false, true} {
		t.Run(strconv.FormatBool(memoryFailure), func(t *testing.T) {
			dir := t.TempDir()
			source := filepath.Join(dir, "init_failure.c")
			if err := os.WriteFile(source, []byte(initFailureSource), 0600); err != nil {
				t.Fatal(err)
			}
			library := filepath.Join(dir, platformLibraryName())
			args := append([]string{}, compiler[1:]...)
			if runtime.GOOS == "darwin" {
				args = append(args, "-dynamiclib")
			} else {
				args = append(args, "-shared")
			}
			if runtime.GOOS != "windows" {
				args = append(args, "-fPIC")
			}
			if memoryFailure {
				args = append(args, "-DFAIL_MEMORY")
			}
			args = append(args, "-I", include, source, "-o", library)
			if out, err := exec.Command(compiler[0], args...).CombinedOutput(); err != nil {
				t.Fatalf("compile Init failure fixture: %v\n%s", err, out)
			}
			out, err := runRuntimeChild(t, dir, testRuntimePath, initFailureEnv, "^TestInitEnvironmentFailureRetryable$", "ORT_TEST_INIT_FAILURE_LIBRARY="+library)
			if err != nil || !bytes.Contains(out, []byte("--- PASS: TestInitEnvironmentFailureRetryable")) {
				t.Fatalf("Init failure child: %v\n%s", err, out)
			}
		})
	}
}

const initFailureSource = `
#include "onnxruntime_c_api.h"
static char status;
static OrtStatus* ORT_API_CALL create_env(OrtLoggingLevel level, const char* id, OrtEnv** out) {
#ifdef FAIL_MEMORY
 *out = (OrtEnv*)&status;
 return NULL;
#else
 return (OrtStatus*)&status;
#endif
}
static OrtStatus* ORT_API_CALL create_memory(enum OrtAllocatorType type, enum OrtMemType mem, OrtMemoryInfo** out) {
 return (OrtStatus*)&status;
}
static void ORT_API_CALL release_env(OrtEnv* env) {}
static OrtErrorCode ORT_API_CALL error_code(const OrtStatus* s) { return ORT_FAIL; }
static const char* ORT_API_CALL error_message(const OrtStatus* s) { return "injected initialization failure"; }
static void ORT_API_CALL release_status(OrtStatus* s) {}
static const OrtApi api = {
 .CreateEnv = create_env,
 .CreateCpuMemoryInfo = create_memory,
 .ReleaseEnv = release_env,
 .GetErrorCode = error_code,
 .GetErrorMessage = error_message,
 .ReleaseStatus = release_status
};
static const OrtApi* ORT_API_CALL get_api(uint32_t version) { return version <= ORT_API_VERSION ? &api : NULL; }
static const char* ORT_API_CALL get_version(void) { return "injected-runtime"; }
static const OrtApiBase base = {get_api, get_version};
ORT_EXPORT const OrtApiBase* ORT_API_CALL OrtGetApiBase(void) { return &base; }
`

func featureGateSynchronize(t *testing.T, input bool) error {
	t.Helper()
	session, err := NewSession(testdataPath("add_f32.onnx"), nil)
	if err != nil {
		return err
	}
	defer session.Close()
	binding, err := NewIOBinding(session)
	if err != nil {
		return err
	}
	defer binding.Close()
	if input {
		return binding.SynchronizeInputs()
	}
	return binding.SynchronizeOutputs()
}

func integrationPluginPath(t *testing.T, variable string) string {
	t.Helper()
	path, present := os.LookupEnv(variable)
	if !present {
		t.Skipf("%s is absent", variable)
	}
	if path == "" {
		t.Fatalf("%s is explicitly empty", variable)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("%s plugin file %q: %v", variable, path, err)
	}
	if !info.Mode().IsRegular() {
		t.Fatalf("%s plugin path %q is not a regular file", variable, path)
	}
	return path
}

func TestEPIntegrationOptIn(t *testing.T) {
	invalid := filepath.Join(t.TempDir(), "invalid_plugin.so")
	if err := os.WriteFile(invalid, []byte("not a plugin library"), 0600); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ variable, selector string }{
		{"ORT_TEST_WEBGPU_EP_PATH", "^TestEPWebGPUIntegration$"},
		{"ORT_TEST_PLUGIN_EP_PATH", "^TestEPPluginIntegration$"},
		{"ORT_TEST_PLUGIN_EP_PATH", "^TestEPContextPluginIntegration$"},
	} {
		for _, value := range []string{"", "/definitely/missing/plugin.so", invalid} {
			t.Run(tc.selector+"/"+value, func(t *testing.T) {
				out, err := runRuntimeChild(t, ".", testRuntimePath, "", tc.selector, tc.variable+"="+value)
				diagnostic := "register plugin"
				switch value {
				case "":
					diagnostic = "explicitly empty"
				case "/definitely/missing/plugin.so":
					diagnostic = "plugin file"
				}
				if err == nil || !strings.Contains(string(out), diagnostic) {
					t.Fatalf("opt-in negative: %v\n%s", err, out)
				}
				if strings.Contains(string(out), "is not available") {
					t.Fatalf("API negotiation wrote to stderr:\n%s", out)
				}
			})
		}
	}
}

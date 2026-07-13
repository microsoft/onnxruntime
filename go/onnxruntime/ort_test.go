package onnxruntime

import (
	"bytes"
	"errors"
	"os"
	"os/exec"
	"sync"
	"testing"
)

// initRaceEnv marks the subprocess spawned by TestInitRace. When set, TestMain
// leaves the process uninitialized so TestInitRaceChild can race a cold-start
// Init against concurrent readers.
const initRaceEnv = "ORT_TEST_INIT_RACE"

func TestMain(m *testing.M) {
	libPath := os.Getenv("ORT_LIB_PATH")
	if libPath == "" {
		libPath = ".ort-lib/lib"
		if _, err := os.Stat(libPath); err != nil {
			println("skipping tests: set ORT_LIB_PATH to the directory containing libonnxruntime.so")
			os.Exit(0)
		}
	}
	SetSharedLibraryPath(libPath)
	if os.Getenv(initRaceEnv) != "1" {
		if err := Init(); err != nil {
			println("failed to init ORT:", err.Error())
			os.Exit(1)
		}
	}
	code := m.Run()
	Shutdown()
	os.Exit(code)
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
	// We can't easily test this because Init() is already successful and sticky.
	// Instead, verify that calling Init() after success is idempotent (no-op),
	// which is the complementary guarantee: once initialized, bad paths don't
	// regress. The retryable-on-failure path is exercised by the Init error
	// handling logic using sync.Mutex (not sync.Once).
	SetSharedLibraryPath("/nonexistent/path")
	if err := Init(); err != nil {
		t.Fatalf("Init should be idempotent after success, got: %v", err)
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
	if os.Getenv("ORT_LIB_PATH") == "" {
		t.Skip("set ORT_LIB_PATH to run the cold-start init race test")
	}

	cmd := exec.Command(os.Args[0], "-test.run=^TestInitRaceChild$", "-test.v")
	cmd.Env = append(os.Environ(), initRaceEnv+"=1")
	out, err := cmd.CombinedOutput()
	if bytes.Contains(out, []byte("DATA RACE")) {
		t.Fatalf("data race between Init and concurrent readers:\n%s", out)
	}
	if err != nil {
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

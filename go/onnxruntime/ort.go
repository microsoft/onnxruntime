package onnxruntime

/*
#include "cshim.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"unsafe"
)

var (
	mu           sync.Mutex
	libPath      string
	libModule    libHandle
	initialized  atomic.Bool
	shutdown     atomic.Bool
	apiVersion   int
	env          *C.OrtEnv
	cpuMemInfo   *C.OrtMemoryInfo
	sessionCount atomic.Int64
)

// SetSharedLibraryPath sets the path to the ONNX Runtime shared library.
// Must be called before Init. Ignored after a successful Init.
func SetSharedLibraryPath(path string) {
	mu.Lock()
	defer mu.Unlock()
	if !initialized.Load() {
		libPath = path
	}
}

// Init initializes the ONNX Runtime environment. It is retryable on failure
// and idempotent after success.
func Init() error {
	mu.Lock()
	defer mu.Unlock()

	if shutdown.Load() {
		return errShutdown
	}
	if initialized.Load() {
		return nil
	}

	path := resolveLibPath()
	fn, handle, err := loadLibrary(path)
	if err != nil {
		return wrapErr("load library", err)
	}
	// Init is retryable: unload the library on every failure below, or a
	// failed attempt leaks the mapping for the lifetime of the process.

	var cVersion C.int
	rc := C.ort_init_api(fn, &cVersion)
	if rc == 1 {
		closeLibrary(handle)
		return wrapErr("init api", fmt.Errorf("OrtGetApiBase returned NULL"))
	}
	if rc == 2 {
		closeLibrary(handle)
		return wrapErr("init api", fmt.Errorf("GetApi failed; ORT library may be too old (need API >= %d)", C.ORT_GO_API_VERSION_MIN))
	}
	apiVersion = int(cVersion)

	cLogID := C.CString("onnxruntime-go")
	defer C.free(unsafe.Pointer(cLogID))

	var ortEnv *C.OrtEnv
	if err := checkStatus(C.ort_CreateEnv(C.ORT_LOGGING_LEVEL_WARNING, cLogID, &ortEnv)); err != nil {
		closeLibrary(handle)
		return wrapErr("create env", err)
	}

	var memInfo *C.OrtMemoryInfo
	if err := checkStatus(C.ort_CreateCpuMemoryInfo(C.OrtDeviceAllocator, C.OrtMemTypeDefault, &memInfo)); err != nil {
		C.ort_ReleaseEnv(ortEnv)
		closeLibrary(handle)
		return wrapErr("create cpu memory info", err)
	}

	env = ortEnv
	cpuMemInfo = memInfo
	libModule = handle
	initialized.Store(true)
	return nil
}

// GetVersion returns the ORT library version string (e.g. "1.27.0").
// Must be called after Init.
func GetVersion() (string, error) {
	mu.Lock()
	defer mu.Unlock()
	if !initialized.Load() {
		return "", errNotInitialized
	}
	return C.GoString(C.ort_GetVersionString()), nil
}

// APIVersion returns the ORT API version negotiated during Init.
func APIVersion() int {
	mu.Lock()
	defer mu.Unlock()
	return apiVersion
}

// IsInitialized reports whether Init has completed successfully.
func IsInitialized() bool {
	mu.Lock()
	defer mu.Unlock()
	return initialized.Load()
}

// Shutdown releases the ONNX Runtime environment. It returns an error if
// any sessions are still open. After Shutdown, Init cannot be called again.
// Shutdown must not run concurrently with other calls into this package:
// it releases the shared environment that those calls use.
func Shutdown() error {
	mu.Lock()
	defer mu.Unlock()

	if shutdown.Load() {
		return nil
	}
	if !initialized.Load() {
		shutdown.Store(true)
		return nil
	}

	if n := sessionCount.Load(); n > 0 {
		return fmt.Errorf("ort: cannot shut down: %d session(s) still open", n)
	}

	C.ort_ReleaseMemoryInfo(cpuMemInfo)
	cpuMemInfo = nil
	C.ort_ReleaseEnv(env)
	env = nil
	initialized.Store(false)
	shutdown.Store(true)
	return nil
}

// EnableTelemetry enables platform telemetry collection.
func EnableTelemetry() error {
	mu.Lock()
	defer mu.Unlock()
	if !initialized.Load() {
		return errNotInitialized
	}
	return wrapErr("enable telemetry", checkStatus(C.ort_EnableTelemetryEvents(env)))
}

// DisableTelemetry disables platform telemetry collection.
func DisableTelemetry() error {
	mu.Lock()
	defer mu.Unlock()
	if !initialized.Load() {
		return errNotInitialized
	}
	return wrapErr("disable telemetry", checkStatus(C.ort_DisableTelemetryEvents(env)))
}

// AvailableProviders returns the execution providers available in the loaded
// ORT library.
func AvailableProviders() ([]string, error) {
	mu.Lock()
	if !initialized.Load() {
		mu.Unlock()
		return nil, errNotInitialized
	}
	mu.Unlock()

	var cProviders **C.char
	var count C.int
	if err := checkStatus(C.ort_GetAvailableProviders(&cProviders, &count)); err != nil {
		return nil, wrapErr("get available providers", err)
	}
	defer C.ort_ReleaseAvailableProviders(cProviders, count)

	n := int(count)
	providers := make([]string, n)
	ptrs := unsafe.Slice((**C.char)(unsafe.Pointer(cProviders)), n)
	for i := 0; i < n; i++ {
		providers[i] = C.GoString(ptrs[i])
	}
	return providers, nil
}

func resolveLibPath() string {
	if libPath != "" {
		return resolveDir(libPath)
	}
	if envPath := os.Getenv("ORT_LIB_PATH"); envPath != "" {
		return resolveDir(envPath)
	}
	return platformLibraryName()
}

func resolveDir(path string) string {
	info, err := os.Stat(path)
	if err == nil && info.IsDir() {
		return filepath.Join(path, platformLibraryName())
	}
	return path
}

func checkInit() error {
	if !initialized.Load() {
		if shutdown.Load() {
			return errShutdown
		}
		return errNotInitialized
	}
	return nil
}

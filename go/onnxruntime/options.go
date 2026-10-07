package onnxruntime

/*
#include "cshim.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"math"
	"sort"
	"sync"
	"unsafe"
)

// SessionOptions configures how a Session is created.
type SessionOptions struct {
	handle *C.OrtSessionOptions
	mu     sync.Mutex
}

// NewSessionOptions creates a new SessionOptions with default values.
func NewSessionOptions() (*SessionOptions, error) {
	if err := checkInit(); err != nil {
		return nil, err
	}
	var opts *C.OrtSessionOptions
	if err := checkStatus(C.ort_CreateSessionOptions(&opts)); err != nil {
		return nil, wrapErr("create session options", err)
	}
	return &SessionOptions{handle: opts}, nil
}

// SetIntraOpNumThreads sets the number of threads used within individual ops.
// 0 means use the default.
func (o *SessionOptions) SetIntraOpNumThreads(n int) error {
	if n < 0 {
		return fmt.Errorf("ort: set intra op threads: thread count must be non-negative, got %d", n)
	}
	if n > math.MaxInt32 {
		return fmt.Errorf("ort: set intra op threads: %d does not fit in C int", n)
	}
	if err := o.lockUsable("set intra op threads"); err != nil {
		return err
	}
	defer o.unlock()
	return wrapErr("set intra op threads", checkStatus(C.ort_SetIntraOpNumThreads(o.handle, C.int(n))))
}

// SetInterOpNumThreads sets the number of threads used to run independent ops in parallel.
// 0 means use the default.
func (o *SessionOptions) SetInterOpNumThreads(n int) error {
	if n < 0 {
		return fmt.Errorf("ort: set inter op threads: thread count must be non-negative, got %d", n)
	}
	if n > math.MaxInt32 {
		return fmt.Errorf("ort: set inter op threads: %d does not fit in C int", n)
	}
	if err := o.lockUsable("set inter op threads"); err != nil {
		return err
	}
	defer o.unlock()
	return wrapErr("set inter op threads", checkStatus(C.ort_SetInterOpNumThreads(o.handle, C.int(n))))
}

// SetGraphOptimizationLevel sets the level of graph optimizations applied.
func (o *SessionOptions) SetGraphOptimizationLevel(level GraphOptimizationLevel) error {
	if level < GraphOptimizationLevel(math.MinInt32) || level > GraphOptimizationLevel(math.MaxInt32) {
		return fmt.Errorf("ort: set graph optimization level: %d does not fit in C enum", level)
	}
	if err := o.lockUsable("set graph optimization level"); err != nil {
		return err
	}
	defer o.unlock()
	return wrapErr("set graph optimization level",
		checkStatus(C.ort_SetSessionGraphOptimizationLevel(o.handle, C.GraphOptimizationLevel(level))))
}

// AddConfigEntry sets an arbitrary session configuration key-value pair.
func (o *SessionOptions) AddConfigEntry(key, value string) error {
	if err := rejectNUL(key, "session config key"); err != nil {
		return err
	}
	if err := rejectNUL(value, "session config value"); err != nil {
		return err
	}
	if err := o.lockUsable("add config entry"); err != nil {
		return err
	}
	defer o.unlock()
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))
	cVal := C.CString(value)
	defer C.free(unsafe.Pointer(cVal))
	return wrapErr("add config entry", checkStatus(C.ort_AddSessionConfigEntry(o.handle, cKey, cVal)))
}

// AppendExecutionProvider adds an execution provider. CUDA and TensorRT are
// routed through their dedicated V2 APIs; all others use the generic string
// key-value API.
func (o *SessionOptions) AppendExecutionProvider(name string, options map[string]string) error {
	if err := rejectNUL(name, "execution provider name"); err != nil {
		return err
	}
	if err := validateStringMap(options, "execution provider option"); err != nil {
		return err
	}
	if err := o.lockUsable("append execution provider"); err != nil {
		return err
	}
	defer o.unlock()
	switch name {
	case "CUDA", "CUDAExecutionProvider":
		return o.appendCUDA(options)
	case "TensorRT", "TensorrtExecutionProvider":
		return o.appendTensorRT(options)
	default:
		return o.appendGenericProvider(name, options)
	}
}

// Clone creates a copy of the session options.
func (o *SessionOptions) Clone() (*SessionOptions, error) {
	if err := o.lockUsable("clone session options"); err != nil {
		return nil, err
	}
	defer o.unlock()
	var out *C.OrtSessionOptions
	if err := checkStatus(C.ort_CloneSessionOptions(o.handle, &out)); err != nil {
		return nil, wrapErr("clone session options", err)
	}
	return &SessionOptions{handle: out}, nil
}

// DisableMemPattern disables memory pattern optimization.
func (o *SessionOptions) DisableMemPattern() error {
	if err := o.lockUsable("disable mem pattern"); err != nil {
		return err
	}
	defer o.unlock()
	return wrapErr("disable mem pattern", checkStatus(C.ort_DisableMemPattern(o.handle)))
}

// EnableMemPattern enables memory pattern optimization (default).
func (o *SessionOptions) EnableMemPattern() error {
	if err := o.lockUsable("enable mem pattern"); err != nil {
		return err
	}
	defer o.unlock()
	return wrapErr("enable mem pattern", checkStatus(C.ort_EnableMemPattern(o.handle)))
}

// EnableCpuMemArena enables the CPU memory arena.
func (o *SessionOptions) EnableCpuMemArena() error {
	if err := o.lockUsable("enable cpu mem arena"); err != nil {
		return err
	}
	defer o.unlock()
	return wrapErr("enable cpu mem arena", checkStatus(C.ort_EnableCpuMemArena(o.handle)))
}

// DisableCpuMemArena disables the CPU memory arena.
func (o *SessionOptions) DisableCpuMemArena() error {
	if err := o.lockUsable("disable cpu mem arena"); err != nil {
		return err
	}
	defer o.unlock()
	return wrapErr("disable cpu mem arena", checkStatus(C.ort_DisableCpuMemArena(o.handle)))
}

// EnableProfiling enables profiling with the given file prefix.
func (o *SessionOptions) EnableProfiling(profileFilePrefix string) error {
	if err := o.lockUsable("enable profiling"); err != nil {
		return err
	}
	defer o.unlock()
	cPrefix, freePrefix, err := ortPath(profileFilePrefix)
	if err != nil {
		return err
	}
	defer freePrefix()
	return wrapErr("enable profiling", checkStatus(C.ort_EnableProfiling(o.handle, cPrefix)))
}

// DisableProfiling disables profiling.
func (o *SessionOptions) DisableProfiling() error {
	if err := o.lockUsable("disable profiling"); err != nil {
		return err
	}
	defer o.unlock()
	return wrapErr("disable profiling", checkStatus(C.ort_DisableProfiling(o.handle)))
}

// AddFreeDimensionOverride fixes a dynamic dimension by its denotation string.
func (o *SessionOptions) AddFreeDimensionOverride(dimDenotation string, dimValue int64) error {
	if err := rejectNUL(dimDenotation, "dimension denotation"); err != nil {
		return err
	}
	if err := o.lockUsable("add free dimension override"); err != nil {
		return err
	}
	defer o.unlock()
	cDim := C.CString(dimDenotation)
	defer C.free(unsafe.Pointer(cDim))
	return wrapErr("add free dimension override",
		checkStatus(C.ort_AddFreeDimensionOverride(o.handle, cDim, C.int64_t(dimValue))))
}

// AddFreeDimensionOverrideByName fixes a dynamic dimension by its name.
func (o *SessionOptions) AddFreeDimensionOverrideByName(dimName string, dimValue int64) error {
	if err := rejectNUL(dimName, "dimension name"); err != nil {
		return err
	}
	if err := o.lockUsable("add free dimension override by name"); err != nil {
		return err
	}
	defer o.unlock()
	cDim := C.CString(dimName)
	defer C.free(unsafe.Pointer(cDim))
	return wrapErr("add free dimension override by name",
		checkStatus(C.ort_AddFreeDimensionOverrideByName(o.handle, cDim, C.int64_t(dimValue))))
}

// SetExecutionMode sets sequential or parallel execution mode.
func (o *SessionOptions) SetExecutionMode(mode ExecutionMode) error {
	if mode < ExecutionMode(math.MinInt32) || mode > ExecutionMode(math.MaxInt32) {
		return fmt.Errorf("ort: set execution mode: %d does not fit in C enum", mode)
	}
	if err := o.lockUsable("set execution mode"); err != nil {
		return err
	}
	defer o.unlock()
	return wrapErr("set execution mode",
		checkStatus(C.ort_SetSessionExecutionMode(o.handle, C.ExecutionMode(mode))))
}

// GetExecutionMode returns the current execution mode. Requires ORT >= 1.27.
func (o *SessionOptions) GetExecutionMode() (ExecutionMode, error) {
	if apiVersion < 27 {
		return 0, fmt.Errorf("ort: GetExecutionMode requires ORT >= 1.27 (have API version %d)", apiVersion)
	}
	if err := o.lockUsable("get execution mode"); err != nil {
		return 0, err
	}
	defer o.unlock()
	var mode C.ExecutionMode
	if err := checkStatus(C.ort_GetSessionExecutionMode(o.handle, &mode)); err != nil {
		return 0, wrapErr("get execution mode", err)
	}
	return ExecutionMode(mode), nil
}

// IsMemPatternEnabled reports whether memory pattern optimization is enabled. Requires ORT >= 1.27.
func (o *SessionOptions) IsMemPatternEnabled() (bool, error) {
	if apiVersion < 27 {
		return false, fmt.Errorf("ort: IsMemPatternEnabled requires ORT >= 1.27 (have API version %d)", apiVersion)
	}
	if err := o.lockUsable("get mem pattern enabled"); err != nil {
		return false, err
	}
	defer o.unlock()
	var out C.int
	if err := checkStatus(C.ort_GetMemPatternEnabled(o.handle, &out)); err != nil {
		return false, wrapErr("get mem pattern enabled", err)
	}
	return out != 0, nil
}

// AddInitializer overrides a model initializer with the given tensor value.
// The tensor must outlive every session created from these options.
func (o *SessionOptions) AddInitializer(name string, value *Tensor) error {
	if err := rejectNUL(name, "initializer name"); err != nil {
		return err
	}
	if err := o.lockUsable("add initializer"); err != nil {
		return err
	}
	defer o.unlock()
	if err := value.checkUsable("add initializer"); err != nil {
		return err
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return wrapErr("add initializer", checkStatus(C.ort_AddInitializer(o.handle, cName, value.value)))
}

// SetOptimizedModelFilePath sets a path to save the optimized model to.
func (o *SessionOptions) SetOptimizedModelFilePath(path string) error {
	if err := o.lockUsable("set optimized model path"); err != nil {
		return err
	}
	defer o.unlock()
	cPath, freePath, err := ortPath(path)
	if err != nil {
		return err
	}
	defer freePath()
	return wrapErr("set optimized model path",
		checkStatus(C.ort_SetOptimizedModelFilePath(o.handle, cPath)))
}

// RegisterCustomOpsLibrary loads a shared library containing custom ops.
func (o *SessionOptions) RegisterCustomOpsLibrary(libraryPath string) error {
	if err := o.lockUsable("register custom ops library"); err != nil {
		return err
	}
	defer o.unlock()
	cPath, freePath, err := ortPath(libraryPath)
	if err != nil {
		return err
	}
	defer freePath()
	return wrapErr("register custom ops library",
		checkStatus(C.ort_RegisterCustomOpsLibrary_V2(o.handle, cPath)))
}

// HasSessionConfigEntry reports whether the config key exists.
func (o *SessionOptions) HasSessionConfigEntry(key string) (bool, error) {
	if err := rejectNUL(key, "session config key"); err != nil {
		return false, err
	}
	if err := o.lockUsable("has session config entry"); err != nil {
		return false, err
	}
	defer o.unlock()
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))
	var out C.int
	if err := checkStatus(C.ort_HasSessionConfigEntry(o.handle, cKey, &out)); err != nil {
		return false, wrapErr("has session config entry", err)
	}
	return out != 0, nil
}

// GetSessionConfigEntry returns the value for the given config key.
func (o *SessionOptions) GetSessionConfigEntry(key string) (string, error) {
	if err := rejectNUL(key, "session config key"); err != nil {
		return "", err
	}
	if err := o.lockUsable("get session config entry"); err != nil {
		return "", err
	}
	defer o.unlock()
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))
	var size C.size_t
	if err := checkStatus(C.ort_GetSessionConfigEntry(o.handle, cKey, nil, &size)); err != nil {
		return "", wrapErr("get session config entry size", err)
	}
	if size == 0 {
		return "", nil
	}
	if uint64(size) > uint64(math.MaxInt) {
		return "", fmt.Errorf("ort: get session config entry: value size %d exceeds addressable range", uint64(size))
	}
	buf := make([]C.char, size)
	if err := checkStatus(C.ort_GetSessionConfigEntry(o.handle, cKey, &buf[0], &size)); err != nil {
		return "", wrapErr("get session config entry", err)
	}
	return C.GoString(&buf[0]), nil
}

// Close releases the session options. It is idempotent.
func (o *SessionOptions) Close() error {
	if o == nil {
		return nil
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.handle != nil {
		C.ort_ReleaseSessionOptions(o.handle)
		o.handle = nil
	}
	return nil
}

func (o *SessionOptions) lockUsable(op string) error {
	if o == nil {
		return fmt.Errorf("ort: %s: session options are nil or closed", op)
	}
	o.mu.Lock()
	if o.handle == nil {
		o.mu.Unlock()
		return fmt.Errorf("ort: %s: session options are nil or closed", op)
	}
	return nil
}

func (o *SessionOptions) unlock() {
	o.mu.Unlock()
}

func (o *SessionOptions) appendCUDA(opts map[string]string) error {
	var cudaOpts *C.OrtCUDAProviderOptionsV2
	if err := checkStatus(C.ort_CreateCUDAProviderOptions(&cudaOpts)); err != nil {
		return wrapErr("create cuda options", err)
	}
	defer C.ort_ReleaseCUDAProviderOptions(cudaOpts)

	if len(opts) > 0 {
		keys, values := sortedKV(opts)
		defer freeStringArrays(keys, values)
		if err := checkStatus(C.ort_UpdateCUDAProviderOptions(
			cudaOpts, &keys[0], &values[0], C.size_t(len(opts)))); err != nil {
			return wrapErr("update cuda options", err)
		}
	}

	return wrapErr("append cuda provider",
		checkStatus(C.ort_SessionOptionsAppendExecutionProvider_CUDA_V2(o.handle, cudaOpts)))
}

func (o *SessionOptions) appendTensorRT(opts map[string]string) error {
	var trtOpts *C.OrtTensorRTProviderOptionsV2
	if err := checkStatus(C.ort_CreateTensorRTProviderOptions(&trtOpts)); err != nil {
		return wrapErr("create tensorrt options", err)
	}
	defer C.ort_ReleaseTensorRTProviderOptions(trtOpts)

	if len(opts) > 0 {
		keys, values := sortedKV(opts)
		defer freeStringArrays(keys, values)
		if err := checkStatus(C.ort_UpdateTensorRTProviderOptions(
			trtOpts, &keys[0], &values[0], C.size_t(len(opts)))); err != nil {
			return wrapErr("update tensorrt options", err)
		}
	}

	return wrapErr("append tensorrt provider",
		checkStatus(C.ort_SessionOptionsAppendExecutionProvider_TensorRT_V2(o.handle, trtOpts)))
}

func (o *SessionOptions) appendGenericProvider(name string, opts map[string]string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))

	if len(opts) == 0 {
		return wrapErr("append provider",
			checkStatus(C.ort_SessionOptionsAppendExecutionProvider(o.handle, cName, nil, nil, 0)))
	}

	keys, values := sortedKV(opts)
	defer freeStringArrays(keys, values)
	return wrapErr("append provider",
		checkStatus(C.ort_SessionOptionsAppendExecutionProvider(
			o.handle, cName, &keys[0], &values[0], C.size_t(len(opts)))))
}

func sortedKV(m map[string]string) ([]*C.char, []*C.char) {
	sorted := make([]string, 0, len(m))
	for k := range m {
		sorted = append(sorted, k)
	}
	sort.Strings(sorted)

	keys := make([]*C.char, len(sorted))
	values := make([]*C.char, len(sorted))
	for i, k := range sorted {
		keys[i] = C.CString(k)
		values[i] = C.CString(m[k])
	}
	return keys, values
}

func freeStringArrays(keys, values []*C.char) {
	for i := range keys {
		C.free(unsafe.Pointer(keys[i]))
		C.free(unsafe.Pointer(values[i]))
	}
}

func validateStringMap(values map[string]string, context string) error {
	for key, value := range values {
		if err := rejectNUL(key, context+" key"); err != nil {
			return err
		}
		if err := rejectNUL(value, context+" value"); err != nil {
			return err
		}
	}
	return nil
}

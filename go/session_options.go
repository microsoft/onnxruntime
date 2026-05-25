package onnxruntime

import (
	"unsafe"

	"github.com/microsoft/onnxruntime/go/internal/cstrings"
)

// NewSessionOptions creates a new session options object.
func NewSessionOptions() (OrtSessionOptions, error) {
	var opts OrtSessionOptions
	fn := getFuncPtr(fnCreateSessionOptions)
	status := OrtStatus(ortDispatch(fn, uintptr(unsafe.Pointer(&opts))))
	return opts, statusToGoErr(status)
}

// ReleaseSessionOptions releases a session options object.
func ReleaseSessionOptions(opts OrtSessionOptions) {
	fn := getFuncPtr(fnReleaseSessionOptions)
	if fn != 0 {
		ortDispatch(fn, uintptr(opts))
	}
}

// SetSessionGraphOptimizationLevel sets the graph optimization level for a session.
func SetSessionGraphOptimizationLevel(opts OrtSessionOptions, level GraphOptimizationLevel) error {
	fn := getFuncPtr(fnSetSessionGraphOptimizationLevel)
	status := OrtStatus(ortDispatch(fn, uintptr(opts), uintptr(level)))
	return statusToGoErr(status)
}

// SetIntraOpNumThreads sets the number of threads used for intra-op parallelism.
func SetIntraOpNumThreads(opts OrtSessionOptions, n int32) error {
	fn := getFuncPtr(fnSetIntraOpNumThreads)
	status := OrtStatus(ortDispatch(fn, uintptr(opts), uintptr(n)))
	return statusToGoErr(status)
}

// SetInterOpNumThreads sets the number of threads used for inter-op parallelism.
func SetInterOpNumThreads(opts OrtSessionOptions, n int32) error {
	fn := getFuncPtr(fnSetInterOpNumThreads)
	status := OrtStatus(ortDispatch(fn, uintptr(opts), uintptr(n)))
	return statusToGoErr(status)
}

// SetSessionExecutionMode sets whether operators run sequentially or in parallel.
func SetSessionExecutionMode(opts OrtSessionOptions, mode ExecutionMode) error {
	fn := getFuncPtr(fnSetSessionExecutionMode)
	status := OrtStatus(ortDispatch(fn, uintptr(opts), uintptr(mode)))
	return statusToGoErr(status)
}

// EnableMemPattern enables memory pattern optimization for the session.
func EnableMemPattern(opts OrtSessionOptions) error {
	fn := getFuncPtr(fnEnableMemPattern)
	status := OrtStatus(ortDispatch(fn, uintptr(opts)))
	return statusToGoErr(status)
}

// DisableMemPattern disables memory pattern optimization for the session.
func DisableMemPattern(opts OrtSessionOptions) error {
	fn := getFuncPtr(fnDisableMemPattern)
	status := OrtStatus(ortDispatch(fn, uintptr(opts)))
	return statusToGoErr(status)
}

// EnableCpuMemArena enables the CPU memory arena allocator for the session.
func EnableCpuMemArena(opts OrtSessionOptions) error {
	fn := getFuncPtr(fnEnableCpuMemArena)
	status := OrtStatus(ortDispatch(fn, uintptr(opts)))
	return statusToGoErr(status)
}

// DisableCpuMemArena disables the CPU memory arena allocator for the session.
func DisableCpuMemArena(opts OrtSessionOptions) error {
	fn := getFuncPtr(fnDisableCpuMemArena)
	status := OrtStatus(ortDispatch(fn, uintptr(opts)))
	return statusToGoErr(status)
}

// AddSessionConfigEntry adds a key-value configuration entry to session options.
func AddSessionConfigEntry(opts OrtSessionOptions, key, value string) error {
	keyBytes := cstrings.StringToCBytes(key)
	valBytes := cstrings.StringToCBytes(value)
	fn := getFuncPtr(fnAddSessionConfigEntry)
	status := OrtStatus(ortDispatch(fn, uintptr(opts),
		uintptr(unsafe.Pointer(&keyBytes[0])),
		uintptr(unsafe.Pointer(&valBytes[0]))))
	return statusToGoErr(status)
}

// AppendExecutionProviderCUDA adds the CUDA execution provider to session options.
func AppendExecutionProviderCUDA(opts OrtSessionOptions) error {
	fn := getFuncPtr(fnSessionOptionsAppendExecutionProviderCUDA)
	status := OrtStatus(ortDispatch(fn, uintptr(opts), 0))
	return statusToGoErr(status)
}

// AppendExecutionProviderTensorRT adds the TensorRT execution provider to session options.
func AppendExecutionProviderTensorRT(opts OrtSessionOptions) error {
	fn := getFuncPtr(fnSessionOptionsAppendExecutionProviderTensorRT)
	status := OrtStatus(ortDispatch(fn, uintptr(opts), 0))
	return statusToGoErr(status)
}

// SessionOptions provides a builder-style interface for configuring ONNX Runtime sessions.
type SessionOptions struct {
	ptr OrtSessionOptions
}

// NewSessionOptionsBuilder creates a new SessionOptions builder.
func NewSessionOptionsBuilder() (*SessionOptions, error) {
	opts, err := NewSessionOptions()
	if err != nil {
		return nil, err
	}
	return &SessionOptions{ptr: opts}, nil
}

// Ptr returns the underlying opaque session options pointer.
func (so *SessionOptions) Ptr() OrtSessionOptions { return so.ptr }

// WithGraphOptimizationLevel sets the graph optimization level.
func (so *SessionOptions) WithGraphOptimizationLevel(level GraphOptimizationLevel) *SessionOptions {
	SetSessionGraphOptimizationLevel(so.ptr, level)
	return so
}

// WithIntraOpNumThreads sets the number of intra-op threads.
func (so *SessionOptions) WithIntraOpNumThreads(n int32) *SessionOptions {
	SetIntraOpNumThreads(so.ptr, n)
	return so
}

// WithInterOpNumThreads sets the number of inter-op threads.
func (so *SessionOptions) WithInterOpNumThreads(n int32) *SessionOptions {
	SetInterOpNumThreads(so.ptr, n)
	return so
}

// WithExecutionMode sets the execution mode.
func (so *SessionOptions) WithExecutionMode(mode ExecutionMode) *SessionOptions {
	SetSessionExecutionMode(so.ptr, mode)
	return so
}

// WithCUDADevice adds the CUDA execution provider with default options.
func (so *SessionOptions) WithCUDADevice() *SessionOptions {
	AppendExecutionProviderCUDA(so.ptr)
	return so
}

// WithTensorRT adds the TensorRT execution provider with default options.
func (so *SessionOptions) WithTensorRT() *SessionOptions {
	AppendExecutionProviderTensorRT(so.ptr)
	return so
}

// Close releases the underlying session options object.
func (so *SessionOptions) Close() {
	if so.ptr != 0 {
		ReleaseSessionOptions(so.ptr)
		so.ptr = 0
	}
}

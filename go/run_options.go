package onnxruntime

import (
	"unsafe"

	"github.com/microsoft/onnxruntime/go/internal/cstrings"
)

// NewRunOptions creates a new run options object for configuring per-run behavior.
func NewRunOptions() (OrtRunOptions, error) {
	var opts OrtRunOptions
	fn := getFuncPtr(fnCreateRunOptions)
	status := OrtStatus(ortDispatch(fn, uintptr(unsafe.Pointer(&opts))))
	return opts, statusToGoErr(status)
}

// ReleaseRunOptions releases a run options object.
func ReleaseRunOptions(opts OrtRunOptions) {
	fn := getFuncPtr(fnReleaseRunOptions)
	if fn != 0 {
		ortDispatch(fn, uintptr(opts))
	}
}

// SetRunLogVerbosityLevel sets the logging verbosity level for a single run.
func SetRunLogVerbosityLevel(opts OrtRunOptions, level int32) error {
	fn := getFuncPtr(fnRunOptionsSetRunLogVerbosityLevel)
	status := OrtStatus(ortDispatch(fn, uintptr(opts), uintptr(level)))
	return statusToGoErr(status)
}

// SetRunTag sets a tag string for a run, useful for profiling and logging.
func SetRunTag(opts OrtRunOptions, tag string) error {
	tagBytes := cstrings.StringToCBytes(tag)
	fn := getFuncPtr(fnRunOptionsSetRunTag)
	status := OrtStatus(ortDispatch(fn, uintptr(opts), uintptr(unsafe.Pointer(&tagBytes[0]))))
	return statusToGoErr(status)
}

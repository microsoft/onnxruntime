package onnxruntime

import (
	"unsafe"

	"github.com/microsoft/onnxruntime/go/internal/cstrings"
)

// NewEnv creates a new ONNX Runtime environment with the given name and logging level.
// The environment is a process-wide singleton; subsequent calls return the same instance.
func NewEnv(name string, level LoggingLevel) (OrtEnv, error) {
	nameBytes := cstrings.StringToCBytes(name)
	var env OrtEnv
	fn := getFuncPtr(fnCreateEnv)
	status := OrtStatus(ortDispatch(fn, uintptr(level), uintptr(unsafe.Pointer(&nameBytes[0])), uintptr(unsafe.Pointer(&env))))
	return env, statusToGoErr(status)
}

// ReleaseEnv releases an ONNX Runtime environment.
func ReleaseEnv(env OrtEnv) {
	fn := getFuncPtr(fnReleaseEnv)
	if fn != 0 {
		ortDispatch(fn, uintptr(env))
	}
}

func statusToGoErr(status OrtStatus) error {
	if status == 0 {
		return nil
	}
	return statusToError(status)
}

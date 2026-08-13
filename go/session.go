package onnxruntime

import (
	"fmt"
	"os"
	"unsafe"

	"github.com/microsoft/onnxruntime/go/internal/cstrings"
)

// Session wraps an ONNX Runtime inference session, holding the loaded model
// and its default allocator.
type Session struct {
	ptr    OrtSession
	env    OrtEnv
	alloc  OrtAllocator
	closed bool
}

// NewSession loads an ONNX model from a file path and creates a new inference session.
func NewSession(env OrtEnv, modelPath string, opts OrtSessionOptions) (*Session, error) {
	pathBytes := cstrings.StringToCBytes(modelPath)
	var sess OrtSession
	fn := getFuncPtr(fnCreateSession)
	status := OrtStatus(ortDispatch(fn,
		uintptr(env),
		uintptr(unsafe.Pointer(&pathBytes[0])),
		uintptr(opts),
		uintptr(unsafe.Pointer(&sess)),
	))
	if err := statusToGoErr(status); err != nil {
		return nil, err
	}

	alloc, err := GetAllocatorWithDefaultOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to get allocator: %w", err)
	}

	return &Session{ptr: sess, env: env, alloc: alloc}, nil
}

// NewSessionFromBytes loads an ONNX model from an in-memory byte slice.
func NewSessionFromBytes(env OrtEnv, modelData []byte, opts OrtSessionOptions) (*Session, error) {
	var sess OrtSession
	fn := getFuncPtr(fnCreateSessionFromArray)
	status := OrtStatus(ortDispatch(fn,
		uintptr(env),
		uintptr(unsafe.Pointer(&modelData[0])),
		uintptr(len(modelData)),
		uintptr(opts),
		uintptr(unsafe.Pointer(&sess)),
	))
	if err := statusToGoErr(status); err != nil {
		return nil, err
	}

	alloc, err := GetAllocatorWithDefaultOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to get allocator: %w", err)
	}

	return &Session{ptr: sess, env: env, alloc: alloc}, nil
}

// NewSessionFromFile loads an ONNX model by reading the file into memory first,
// then delegates to NewSessionFromBytes.
func NewSessionFromFile(env OrtEnv, path string, opts OrtSessionOptions) (*Session, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read model file: %w", err)
	}
	return NewSessionFromBytes(env, data, opts)
}

// Run executes inference on the session with default run options.
// inputNames and outputNames specify which tensors to read and produce;
// inputs provides the input OrtValues in the same order as inputNames.
// Returns output OrtValues in the same order as outputNames.
func (s *Session) Run(inputNames []string, inputs []OrtValue, outputNames []string) ([]OrtValue, error) {
	return s.RunWithOptions(0, inputNames, inputs, outputNames)
}

// RunWithOptions executes inference with custom run options.
func (s *Session) RunWithOptions(runOpts OrtRunOptions, inputNames []string, inputs []OrtValue, outputNames []string) ([]OrtValue, error) {
	if len(inputNames) != len(inputs) {
		return nil, fmt.Errorf("inputNames length (%d) != inputs length (%d)", len(inputNames), len(inputs))
	}

	inputNamePtrs := make([]*byte, len(inputNames))
	for i, name := range inputNames {
		b := cstrings.StringToCBytes(name)
		inputNamePtrs[i] = &b[0]
	}

	outputNamePtrs := make([]*byte, len(outputNames))
	for i, name := range outputNames {
		b := cstrings.StringToCBytes(name)
		outputNamePtrs[i] = &b[0]
	}

	outputs := make([]OrtValue, len(outputNames))

	fn := getFuncPtr(fnRun)
	status := OrtStatus(ortDispatch(fn,
		uintptr(s.ptr),
		uintptr(runOpts),
		uintptr(unsafe.Pointer(&inputNamePtrs[0])),
		uintptr(unsafe.Pointer(&inputs[0])),
		uintptr(len(inputs)),
		uintptr(unsafe.Pointer(&outputNamePtrs[0])),
		uintptr(len(outputNames)),
		uintptr(unsafe.Pointer(&outputs[0])),
	))
	if err := statusToGoErr(status); err != nil {
		return nil, err
	}

	return outputs, nil
}

// InputCount returns the number of model inputs.
func (s *Session) InputCount() (int, error) {
	var count uint64
	fn := getFuncPtr(fnSessionGetInputCount)
	status := OrtStatus(ortDispatch(fn, uintptr(s.ptr), uintptr(unsafe.Pointer(&count))))
	return int(count), statusToGoErr(status)
}

// OutputCount returns the number of model outputs.
func (s *Session) OutputCount() (int, error) {
	var count uint64
	fn := getFuncPtr(fnSessionGetOutputCount)
	status := OrtStatus(ortDispatch(fn, uintptr(s.ptr), uintptr(unsafe.Pointer(&count))))
	return int(count), statusToGoErr(status)
}

// InputName returns the name of the input at the given index.
// The returned string is allocated by ORT; caller does not need to free it.
func (s *Session) InputName(index int) (string, error) {
	var namePtr *byte
	fn := getFuncPtr(fnSessionGetInputName)
	status := OrtStatus(ortDispatch(fn, uintptr(s.ptr), uintptr(index), uintptr(s.alloc), uintptr(unsafe.Pointer(&namePtr))))
	if err := statusToGoErr(status); err != nil {
		return "", err
	}
	name := cstrings.CStringToString(namePtr)
	AllocatorFree(s.alloc, unsafe.Pointer(namePtr))
	return name, nil
}

// OutputName returns the name of the output at the given index.
// The returned string is allocated by ORT; caller does not need to free it.
func (s *Session) OutputName(index int) (string, error) {
	var namePtr *byte
	fn := getFuncPtr(fnSessionGetOutputName)
	status := OrtStatus(ortDispatch(fn, uintptr(s.ptr), uintptr(index), uintptr(s.alloc), uintptr(unsafe.Pointer(&namePtr))))
	if err := statusToGoErr(status); err != nil {
		return "", err
	}
	name := cstrings.CStringToString(namePtr)
	AllocatorFree(s.alloc, unsafe.Pointer(namePtr))
	return name, nil
}

// Allocator returns the default allocator associated with this session.
func (s *Session) Allocator() OrtAllocator { return s.alloc }

// Ptr returns the underlying opaque session pointer.
func (s *Session) Ptr() OrtSession { return s.ptr }

// Close releases the session.
// After Close, the session must not be used.
func (s *Session) Close() error {
	if s.closed {
		return nil
	}
	s.closed = true

	if s.ptr != 0 {
		fn := getFuncPtr(fnReleaseSession)
		if fn != 0 {
			ortDispatch(fn, uintptr(s.ptr))
		}
		s.ptr = 0
	}
	// Default allocator is managed by ORT, do not release.
	s.alloc = 0
	return nil
}

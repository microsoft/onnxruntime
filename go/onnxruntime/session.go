package onnxruntime

/*
#include "cshim.h"
#include <stdlib.h>
*/
import "C"
import (
	"context"
	"fmt"
	"sync"
	"unsafe"
)

// IOInfo describes a single input or output of a model.
type IOInfo struct {
	Name     string
	DataType TensorElementDataType
	Shape    []int64 // dynamic dimensions are -1
}

// Session wraps an ORT inference session. It is safe for concurrent use:
// multiple goroutines may call Run simultaneously.
type Session struct {
	mu      sync.RWMutex
	handle  *C.OrtSession
	inputs  []IOInfo
	outputs []IOInfo
	closed  bool
}

// NewSession creates a session from an ONNX model file.
// If opts is nil, default options are used.
func NewSession(modelPath string, opts *SessionOptions) (*Session, error) {
	if err := checkInit(); err != nil {
		return nil, err
	}

	ownOpts, opts, err := ensureOpts(opts)
	if err != nil {
		return nil, err
	}
	if ownOpts {
		defer func() { _ = opts.Close() }()
	}

	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))

	var handle *C.OrtSession
	if err := checkStatus(C.ort_CreateSession(env, cPath, opts.handle, &handle)); err != nil {
		return nil, wrapErr("create session", err)
	}

	return finalizeSession(handle)
}

// NewSessionFromBytes creates a session from an in-memory ONNX model.
// If opts is nil, default options are used.
func NewSessionFromBytes(model []byte, opts *SessionOptions) (*Session, error) {
	if err := checkInit(); err != nil {
		return nil, err
	}
	if len(model) == 0 {
		return nil, fmt.Errorf("ort: create session: empty model data")
	}

	ownOpts, opts, err := ensureOpts(opts)
	if err != nil {
		return nil, err
	}
	if ownOpts {
		defer func() { _ = opts.Close() }()
	}

	var handle *C.OrtSession
	if err := checkStatus(C.ort_CreateSessionFromArray(
		env, unsafe.Pointer(&model[0]), C.size_t(len(model)), opts.handle, &handle)); err != nil {
		return nil, wrapErr("create session from bytes", err)
	}

	return finalizeSession(handle)
}

// Inputs returns metadata for all model inputs.
func (s *Session) Inputs() []IOInfo {
	return copyIOInfo(s.inputs)
}

// Outputs returns metadata for all model outputs.
func (s *Session) Outputs() []IOInfo {
	return copyIOInfo(s.outputs)
}

// Run executes inference. inputs maps input names to tensors. outputNames
// lists which outputs to request; if nil, all outputs are returned.
// The returned tensors must be closed by the caller.
//
// Run is safe for concurrent use on the same session.
func (s *Session) Run(ctx context.Context, inputs map[string]*Tensor, outputNames []string) (map[string]*Tensor, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.closed {
		return nil, fmt.Errorf("ort: run: session is closed")
	}

	if ctx.Err() != nil {
		return nil, ctx.Err()
	}

	if outputNames == nil {
		outputNames = make([]string, len(s.outputs))
		for i, o := range s.outputs {
			outputNames[i] = o.Name
		}
	}

	if len(outputNames) == 0 {
		return nil, fmt.Errorf("ort: run: no output names specified")
	}

	return s.runInner(ctx, inputs, outputNames, nil)
}

// RunWithOptions executes inference with explicit run options.
func (s *Session) RunWithOptions(ctx context.Context, opts *RunOptions, inputs map[string]*Tensor, outputNames []string) (map[string]*Tensor, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.closed {
		return nil, fmt.Errorf("ort: run: session is closed")
	}

	if ctx.Err() != nil {
		return nil, ctx.Err()
	}

	if outputNames == nil {
		outputNames = make([]string, len(s.outputs))
		for i, o := range s.outputs {
			outputNames[i] = o.Name
		}
	}

	if len(outputNames) == 0 {
		return nil, fmt.Errorf("ort: run: no output names specified")
	}

	var runOpts *C.OrtRunOptions
	if opts != nil {
		runOpts = opts.handle
	}

	return s.runInner(ctx, inputs, outputNames, runOpts)
}

func (s *Session) runInner(ctx context.Context, inputs map[string]*Tensor, outputNames []string, runOpts *C.OrtRunOptions) (map[string]*Tensor, error) {
	nInputs := len(inputs)
	nOutputs := len(outputNames)
	cInputNames := make([]*C.char, nInputs)
	cInputValues := make([]*C.OrtValue, nInputs)
	i := 0
	for name, tensor := range inputs {
		cInputNames[i] = C.CString(name)
		cInputValues[i] = tensor.value
		i++
	}
	defer func() {
		for _, cn := range cInputNames {
			C.free(unsafe.Pointer(cn))
		}
	}()

	cOutputNames := make([]*C.char, nOutputs)
	for i, name := range outputNames {
		cOutputNames[i] = C.CString(name)
	}
	defer func() {
		for _, cn := range cOutputNames {
			C.free(unsafe.Pointer(cn))
		}
	}()

	cOutputValues := make([]*C.OrtValue, nOutputs)

	ownRunOpts := false
	if runOpts == nil && ctx.Done() != nil {
		var err error
		if err = checkStatus(C.ort_CreateRunOptions(&runOpts)); err != nil {
			return nil, wrapErr("create run options", err)
		}
		ownRunOpts = true

		stopWatcher := make(chan struct{})
		watcherDone := make(chan struct{})
		done := ctx.Done()
		go func() {
			defer close(watcherDone)
			select {
			case <-done:
				C.ort_RunOptionsSetTerminate(runOpts)
			case <-stopWatcher:
			}
		}()
		defer func() {
			close(stopWatcher)
			<-watcherDone
		}()
	}
	if ownRunOpts {
		defer C.ort_ReleaseRunOptions(runOpts)
	}

	var inNamesPtr **C.char
	var inValuesPtr **C.OrtValue
	if nInputs > 0 {
		inNamesPtr = &cInputNames[0]
		inValuesPtr = &cInputValues[0]
	}

	err := checkStatus(C.ort_Run(
		s.handle, runOpts,
		inNamesPtr,
		inValuesPtr, C.size_t(nInputs),
		&cOutputNames[0], C.size_t(nOutputs),
		&cOutputValues[0],
	))

	if err != nil {
		for _, v := range cOutputValues {
			if v != nil {
				C.ort_ReleaseValue(v)
			}
		}
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return nil, wrapErr("run", err)
	}

	result := make(map[string]*Tensor, nOutputs)
	for i, name := range outputNames {
		t, err := wrapOutputTensor(cOutputValues[i])
		if err != nil {
			for _, v := range cOutputValues[i+1:] {
				if v != nil {
					C.ort_ReleaseValue(v)
				}
			}
			for _, existing := range result {
				_ = existing.Close()
			}
			return nil, wrapErr("wrap output "+name, err)
		}
		result[name] = t
	}

	return result, nil
}

// EndProfiling ends profiling and returns the profile file path.
func (s *Session) EndProfiling() (string, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.closed {
		return "", fmt.Errorf("ort: end profiling: session is closed")
	}

	var allocator *C.OrtAllocator
	if err := checkStatus(C.ort_GetAllocatorWithDefaultOptions(&allocator)); err != nil {
		return "", wrapErr("get allocator", err)
	}

	var cPath *C.char
	if err := checkStatus(C.ort_SessionEndProfiling(s.handle, allocator, &cPath)); err != nil {
		return "", wrapErr("end profiling", err)
	}
	path := C.GoString(cPath)
	C.ort_AllocatorFree(allocator, unsafe.Pointer(cPath))
	return path, nil
}

// Close releases the session. It is idempotent. Close waits for any
// in-flight Run calls to complete.
func (s *Session) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return nil
	}
	s.closed = true
	C.ort_ReleaseSession(s.handle)
	s.handle = nil
	sessionCount.Add(-1)
	return nil
}

func ensureOpts(opts *SessionOptions) (bool, *SessionOptions, error) {
	if opts != nil {
		return false, opts, nil
	}
	o, err := NewSessionOptions()
	if err != nil {
		return false, nil, err
	}
	return true, o, nil
}

func finalizeSession(handle *C.OrtSession) (*Session, error) {
	inputs, err := introspectIO(handle, true)
	if err != nil {
		C.ort_ReleaseSession(handle)
		return nil, err
	}

	outputs, err := introspectIO(handle, false)
	if err != nil {
		C.ort_ReleaseSession(handle)
		return nil, err
	}

	sessionCount.Add(1)
	return &Session{
		handle:  handle,
		inputs:  inputs,
		outputs: outputs,
	}, nil
}

func introspectIO(session *C.OrtSession, isInput bool) ([]IOInfo, error) {
	var count C.size_t
	var err error
	if isInput {
		err = checkStatus(C.ort_SessionGetInputCount(session, &count))
	} else {
		err = checkStatus(C.ort_SessionGetOutputCount(session, &count))
	}
	if err != nil {
		return nil, wrapErr("get io count", err)
	}

	var allocator *C.OrtAllocator
	if err := checkStatus(C.ort_GetAllocatorWithDefaultOptions(&allocator)); err != nil {
		return nil, wrapErr("get allocator", err)
	}

	infos := make([]IOInfo, int(count))
	for i := C.size_t(0); i < count; i++ {
		var cName *C.char
		if isInput {
			err = checkStatus(C.ort_SessionGetInputName(session, i, allocator, &cName))
		} else {
			err = checkStatus(C.ort_SessionGetOutputName(session, i, allocator, &cName))
		}
		if err != nil {
			return nil, wrapErr("get io name", err)
		}
		infos[i].Name = C.GoString(cName)
		C.ort_AllocatorFree(allocator, unsafe.Pointer(cName))

		var typeInfo *C.OrtTypeInfo
		if isInput {
			err = checkStatus(C.ort_SessionGetInputTypeInfo(session, i, &typeInfo))
		} else {
			err = checkStatus(C.ort_SessionGetOutputTypeInfo(session, i, &typeInfo))
		}
		if err != nil {
			return nil, wrapErr("get io type info", err)
		}

		var onnxType C.enum_ONNXType
		if err := checkStatus(C.ort_GetOnnxTypeFromTypeInfo(typeInfo, &onnxType)); err != nil {
			C.ort_ReleaseTypeInfo(typeInfo)
			return nil, wrapErr("get onnx type", err)
		}

		if onnxType == C.ONNX_TYPE_TENSOR {
			var tensorInfo *C.OrtTensorTypeAndShapeInfo
			if err := checkStatus(C.ort_CastTypeInfoToTensorInfo(typeInfo, &tensorInfo)); err != nil {
				C.ort_ReleaseTypeInfo(typeInfo)
				return nil, wrapErr("cast to tensor info", err)
			}

			var cDtype C.enum_ONNXTensorElementDataType
			if err := checkStatus(C.ort_GetTensorElementType(tensorInfo, &cDtype)); err != nil {
				C.ort_ReleaseTypeInfo(typeInfo)
				return nil, wrapErr("get element type", err)
			}
			infos[i].DataType = TensorElementDataType(cDtype)

			var ndims C.size_t
			if err := checkStatus(C.ort_GetDimensionsCount(tensorInfo, &ndims)); err != nil {
				C.ort_ReleaseTypeInfo(typeInfo)
				return nil, wrapErr("get dims count", err)
			}

			shape := make([]int64, int(ndims))
			if ndims > 0 {
				if err := checkStatus(C.ort_GetDimensions(tensorInfo,
					(*C.int64_t)(unsafe.Pointer(&shape[0])), ndims)); err != nil {
					C.ort_ReleaseTypeInfo(typeInfo)
					return nil, wrapErr("get dims", err)
				}
			}
			infos[i].Shape = shape
		}

		C.ort_ReleaseTypeInfo(typeInfo)
	}

	return infos, nil
}

func copyIOInfo(infos []IOInfo) []IOInfo {
	out := make([]IOInfo, len(infos))
	for i, info := range infos {
		out[i] = IOInfo{
			Name:     info.Name,
			DataType: info.DataType,
			Shape:    copyShape(info.Shape),
		}
	}
	return out
}

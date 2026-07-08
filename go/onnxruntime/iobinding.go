package onnxruntime

/*
#include "cshim.h"
#include <stdlib.h>
*/
import "C"
import "unsafe"

type AllocatorType int

const (
	AllocatorTypeInvalid AllocatorType = -1
	AllocatorTypeDevice  AllocatorType = 0
	AllocatorTypeArena   AllocatorType = 1
)

type MemType int

const (
	MemTypeCPUInput MemType = -2
	MemTypeCPU      MemType = -1
	MemTypeDefault  MemType = 0
)

type MemoryInfo struct {
	handle *C.OrtMemoryInfo
}

func NewCPUMemoryInfo() (*MemoryInfo, error) {
	if err := checkInit(); err != nil {
		return nil, err
	}
	var info *C.OrtMemoryInfo
	if err := checkStatus(C.ort_CreateCpuMemoryInfo(C.OrtDeviceAllocator, C.OrtMemTypeDefault, &info)); err != nil {
		return nil, wrapErr("create cpu memory info", err)
	}
	return &MemoryInfo{handle: info}, nil
}

func NewMemoryInfo(name string, allocatorType AllocatorType, id int, memType MemType) (*MemoryInfo, error) {
	if err := checkInit(); err != nil {
		return nil, err
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	var info *C.OrtMemoryInfo
	if err := checkStatus(C.ort_CreateMemoryInfo(
		cName,
		C.enum_OrtAllocatorType(allocatorType),
		C.int(id),
		C.enum_OrtMemType(memType),
		&info,
	)); err != nil {
		return nil, wrapErr("create memory info", err)
	}
	return &MemoryInfo{handle: info}, nil
}

func (m *MemoryInfo) Close() error {
	if m.handle != nil {
		C.ort_ReleaseMemoryInfo(m.handle)
		m.handle = nil
	}
	return nil
}

type IOBinding struct {
	handle  *C.OrtIoBinding
	session *Session
}

func NewIOBinding(session *Session) (*IOBinding, error) {
	if err := checkInit(); err != nil {
		return nil, err
	}
	var binding *C.OrtIoBinding
	if err := checkStatus(C.ort_CreateIoBinding(session.handle, &binding)); err != nil {
		return nil, wrapErr("create io binding", err)
	}
	return &IOBinding{handle: binding, session: session}, nil
}

func (b *IOBinding) BindInput(name string, value *Tensor) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return wrapErr("bind input", checkStatus(C.ort_BindInput(b.handle, cName, value.value)))
}

func (b *IOBinding) BindOutput(name string, value *Tensor) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return wrapErr("bind output", checkStatus(C.ort_BindOutput(b.handle, cName, value.value)))
}

func (b *IOBinding) BindOutputToDevice(name string, memInfo *MemoryInfo) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return wrapErr("bind output to device", checkStatus(C.ort_BindOutputToDevice(b.handle, cName, memInfo.handle)))
}

func (b *IOBinding) Run(opts *RunOptions) error {
	b.session.mu.RLock()
	defer b.session.mu.RUnlock()

	var runOpts *C.OrtRunOptions
	if opts != nil {
		runOpts = opts.handle
	}
	return wrapErr("run with binding", checkStatus(C.ort_RunWithBinding(b.session.handle, runOpts, b.handle)))
}

func (b *IOBinding) OutputNames() ([]string, error) {
	var allocator *C.OrtAllocator
	if err := checkStatus(C.ort_GetAllocatorWithDefaultOptions(&allocator)); err != nil {
		return nil, wrapErr("get allocator", err)
	}

	var buffer *C.char
	var lengths *C.size_t
	var count C.size_t
	if err := checkStatus(C.ort_GetBoundOutputNames(b.handle, allocator, &buffer, &lengths, &count)); err != nil {
		return nil, wrapErr("get bound output names", err)
	}
	defer C.ort_AllocatorFree(allocator, unsafe.Pointer(buffer))
	defer C.ort_AllocatorFree(allocator, unsafe.Pointer(lengths))

	n := int(count)
	if n == 0 {
		return nil, nil
	}

	lens := unsafe.Slice((*C.size_t)(unsafe.Pointer(lengths)), n)
	names := make([]string, n)
	ptr := (*C.char)(unsafe.Pointer(buffer))
	for i := 0; i < n; i++ {
		l := int(lens[i])
		names[i] = C.GoStringN(ptr, C.int(l))
		ptr = (*C.char)(unsafe.Pointer(uintptr(unsafe.Pointer(ptr)) + uintptr(l)))
	}
	return names, nil
}

func (b *IOBinding) OutputValues() ([]*Tensor, error) {
	var allocator *C.OrtAllocator
	if err := checkStatus(C.ort_GetAllocatorWithDefaultOptions(&allocator)); err != nil {
		return nil, wrapErr("get allocator", err)
	}

	var values **C.OrtValue
	var count C.size_t
	if err := checkStatus(C.ort_GetBoundOutputValues(b.handle, allocator, &values, &count)); err != nil {
		return nil, wrapErr("get bound output values", err)
	}
	defer C.ort_AllocatorFree(allocator, unsafe.Pointer(values))

	n := int(count)
	if n == 0 {
		return nil, nil
	}

	ptrs := unsafe.Slice((**C.OrtValue)(unsafe.Pointer(values)), n)
	tensors := make([]*Tensor, n)
	for i := 0; i < n; i++ {
		t, err := wrapOutputTensor(ptrs[i])
		if err != nil {
			for j := 0; j < i; j++ {
				_ = tensors[j].Close()
			}
			for j := i; j < n; j++ {
				C.ort_ReleaseValue(ptrs[j])
			}
			return nil, wrapErr("wrap bound output", err)
		}
		tensors[i] = t
	}
	return tensors, nil
}

func (b *IOBinding) ClearInputs() {
	C.ort_ClearBoundInputs(b.handle)
}

func (b *IOBinding) ClearOutputs() {
	C.ort_ClearBoundOutputs(b.handle)
}

func (b *IOBinding) Close() error {
	if b.handle != nil {
		C.ort_ReleaseIoBinding(b.handle)
		b.handle = nil
	}
	return nil
}

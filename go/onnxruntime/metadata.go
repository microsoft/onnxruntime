package onnxruntime

/*
#include "cshim.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"math"
	"sync"
	"unsafe"
)

type ModelMetadata struct {
	handle *C.OrtModelMetadata
	mu     sync.RWMutex
}

func (s *Session) ModelMetadata() (*ModelMetadata, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.closed {
		return nil, fmt.Errorf("ort: model metadata: session is closed")
	}

	var meta *C.OrtModelMetadata
	if err := checkStatus(C.ort_SessionGetModelMetadata(s.handle, &meta)); err != nil {
		return nil, wrapErr("model metadata", err)
	}
	return &ModelMetadata{handle: meta}, nil
}

func (m *ModelMetadata) ProducerName() (string, error) {
	return m.getString(func(alloc *C.OrtAllocator, out **C.char) error {
		return checkStatus(C.ort_ModelMetadataGetProducerName(m.handle, alloc, out))
	}, "producer name")
}

func (m *ModelMetadata) GraphName() (string, error) {
	return m.getString(func(alloc *C.OrtAllocator, out **C.char) error {
		return checkStatus(C.ort_ModelMetadataGetGraphName(m.handle, alloc, out))
	}, "graph name")
}

func (m *ModelMetadata) Domain() (string, error) {
	return m.getString(func(alloc *C.OrtAllocator, out **C.char) error {
		return checkStatus(C.ort_ModelMetadataGetDomain(m.handle, alloc, out))
	}, "domain")
}

func (m *ModelMetadata) Description() (string, error) {
	return m.getString(func(alloc *C.OrtAllocator, out **C.char) error {
		return checkStatus(C.ort_ModelMetadataGetDescription(m.handle, alloc, out))
	}, "description")
}

func (m *ModelMetadata) GraphDescription() (string, error) {
	return m.getString(func(alloc *C.OrtAllocator, out **C.char) error {
		return checkStatus(C.ort_ModelMetadataGetGraphDescription(m.handle, alloc, out))
	}, "graph description")
}

func (m *ModelMetadata) Version() (int64, error) {
	if err := m.lockUsable(); err != nil {
		return 0, err
	}
	defer m.mu.RUnlock()
	var version C.int64_t
	if err := checkStatus(C.ort_ModelMetadataGetVersion(m.handle, &version)); err != nil {
		return 0, wrapErr("metadata version", err)
	}
	return int64(version), nil
}

func (m *ModelMetadata) CustomMetadataKeys() ([]string, error) {
	if err := m.lockUsable(); err != nil {
		return nil, err
	}
	defer m.mu.RUnlock()
	alloc, err := defaultAllocator()
	if err != nil {
		return nil, wrapErr("metadata custom keys", err)
	}

	var cKeys **C.char
	var count C.int64_t
	if err := checkStatus(C.ort_ModelMetadataGetCustomMetadataMapKeys(m.handle, alloc, &cKeys, &count)); err != nil {
		return nil, wrapErr("metadata custom keys", err)
	}
	if cKeys != nil {
		defer C.ort_AllocatorFree(alloc, unsafe.Pointer(cKeys))
	}

	if count < 0 || uint64(count) > uint64(math.MaxInt)/uint64(unsafe.Sizeof("")) {
		return nil, fmt.Errorf("ort: metadata custom keys: count %d exceeds addressable range", int64(count))
	}
	n := int(count)
	if n == 0 {
		return nil, nil
	}
	if cKeys == nil {
		return nil, fmt.Errorf("ort: metadata custom keys: ORT returned a nil key array for %d keys", n)
	}

	ptrs := unsafe.Slice((**C.char)(unsafe.Pointer(cKeys)), n)
	keys := make([]string, n)
	for i := 0; i < n; i++ {
		keys[i] = C.GoString(ptrs[i])
		C.ort_AllocatorFree(alloc, unsafe.Pointer(ptrs[i]))
	}

	return keys, nil
}

func (m *ModelMetadata) LookupCustomMetadata(key string) (string, error) {
	if err := rejectNUL(key, "metadata key"); err != nil {
		return "", err
	}
	if err := m.lockUsable(); err != nil {
		return "", err
	}
	defer m.mu.RUnlock()
	alloc, err := defaultAllocator()
	if err != nil {
		return "", wrapErr("metadata lookup", err)
	}

	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))

	var cVal *C.char
	if err := checkStatus(C.ort_ModelMetadataLookupCustomMetadataMap(m.handle, alloc, cKey, &cVal)); err != nil {
		return "", wrapErr("metadata lookup", err)
	}

	if cVal == nil {
		return "", nil
	}

	val := C.GoString(cVal)
	C.ort_AllocatorFree(alloc, unsafe.Pointer(cVal))
	return val, nil
}

func (m *ModelMetadata) Close() error {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.handle == nil {
		return nil
	}
	C.ort_ReleaseModelMetadata(m.handle)
	m.handle = nil
	return nil
}

func (m *ModelMetadata) lockUsable() error {
	if m == nil {
		return fmt.Errorf("ort: metadata: nil or closed")
	}
	m.mu.RLock()
	if m.handle == nil {
		m.mu.RUnlock()
		return fmt.Errorf("ort: metadata: already closed")
	}
	return nil
}

func (m *ModelMetadata) getString(fn func(*C.OrtAllocator, **C.char) error, context string) (string, error) {
	if err := m.lockUsable(); err != nil {
		return "", err
	}
	defer m.mu.RUnlock()
	alloc, err := defaultAllocator()
	if err != nil {
		return "", wrapErr("metadata "+context, err)
	}

	var cStr *C.char
	if err := fn(alloc, &cStr); err != nil {
		return "", wrapErr("metadata "+context, err)
	}
	if cStr == nil {
		return "", nil
	}

	str := C.GoString(cStr)
	C.ort_AllocatorFree(alloc, unsafe.Pointer(cStr))
	return str, nil
}

func defaultAllocator() (*C.OrtAllocator, error) {
	var alloc *C.OrtAllocator
	if err := checkStatus(C.ort_GetAllocatorWithDefaultOptions(&alloc)); err != nil {
		return nil, err
	}
	return alloc, nil
}

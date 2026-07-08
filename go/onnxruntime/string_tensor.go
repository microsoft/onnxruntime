package onnxruntime

/*
#include "cshim.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// CreateStringTensor creates a tensor containing string data.
func CreateStringTensor(shape []int64, data []string) (*Tensor, error) {
	if err := checkInit(); err != nil {
		return nil, err
	}

	count := shapeElementCount(shape)
	if count < 0 {
		return nil, fmt.Errorf("ort: create string tensor: shape contains negative dimension")
	}
	if int64(len(data)) != count {
		return nil, fmt.Errorf("ort: create string tensor: data length %d does not match shape element count %d", len(data), count)
	}

	var allocator *C.OrtAllocator
	if err := checkStatus(C.ort_GetAllocatorWithDefaultOptions(&allocator)); err != nil {
		return nil, wrapErr("get allocator", err)
	}

	var cShape *C.int64_t
	if len(shape) > 0 {
		cShape = (*C.int64_t)(unsafe.Pointer(&shape[0]))
	}

	var value *C.OrtValue
	if err := checkStatus(C.ort_CreateTensorAsOrtValue(
		allocator, cShape, C.size_t(len(shape)),
		C.ONNXTensorElementDataType(TensorElementDataTypeString), &value)); err != nil {
		return nil, wrapErr("create string tensor", err)
	}

	if count > 0 {
		cStrings := make([]*C.char, len(data))
		for i, s := range data {
			cStrings[i] = C.CString(s)
		}
		err := checkStatus(C.ort_FillStringTensor(value, &cStrings[0], C.size_t(len(data))))
		for _, cs := range cStrings {
			C.free(unsafe.Pointer(cs))
		}
		if err != nil {
			C.ort_ReleaseValue(value)
			return nil, wrapErr("fill string tensor", err)
		}
	}

	return &Tensor{
		value: value,
		dtype: TensorElementDataTypeString,
		shape: copyShape(shape),
		owned: true,
	}, nil
}

// StringData reads string data from a tensor. Returns an error if the tensor
// is not a string tensor.
func (t *Tensor) StringData() ([]string, error) {
	if t.closed {
		return nil, fmt.Errorf("ort: string data: tensor is closed")
	}
	if t.dtype != TensorElementDataTypeString {
		return nil, fmt.Errorf("ort: string data: tensor is %s, not String", t.dtype)
	}

	count := shapeElementCount(t.shape)
	if count == 0 {
		return nil, nil
	}

	var totalLen C.size_t
	if err := checkStatus(C.ort_GetStringTensorDataLength(t.value, &totalLen)); err != nil {
		return nil, wrapErr("get string tensor length", err)
	}

	n := int(count)
	buf := make([]byte, int(totalLen))
	offsets := make([]C.size_t, n)

	var bufPtr unsafe.Pointer
	if totalLen > 0 {
		bufPtr = unsafe.Pointer(&buf[0])
	} else {
		bufPtr = unsafe.Pointer(&buf)
	}

	if err := checkStatus(C.ort_GetStringTensorContent(
		t.value, bufPtr, totalLen, &offsets[0], C.size_t(n))); err != nil {
		return nil, wrapErr("get string tensor content", err)
	}

	result := make([]string, n)
	for i := 0; i < n; i++ {
		start := int(offsets[i])
		var end int
		if i+1 < n {
			end = int(offsets[i+1])
		} else {
			end = int(totalLen)
		}
		result[i] = string(buf[start:end])
	}

	return result, nil
}

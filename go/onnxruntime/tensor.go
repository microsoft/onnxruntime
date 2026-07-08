package onnxruntime

/*
#include "cshim.h"
*/
import "C"
import (
	"fmt"
	"runtime"
	"unsafe"
)

// Tensor wraps an OrtValue containing tensor data.
type Tensor struct {
	value  *C.OrtValue
	dtype  TensorElementDataType
	shape  []int64
	pinner runtime.Pinner
	keep   any
	owned  bool
	closed bool
}

// CreateTensor creates a tensor backed by the provided data slice.
// The data must not be freed or resized until the tensor is closed.
// The shape must have non-negative dimensions whose product equals len(data).
func CreateTensor[T TensorElement](shape []int64, data []T) (*Tensor, error) {
	if err := checkInit(); err != nil {
		return nil, err
	}

	dtype := dtypeOf[T]()
	count := shapeElementCount(shape)
	if count < 0 {
		return nil, fmt.Errorf("ort: create tensor: shape contains negative dimension")
	}
	if int64(len(data)) != count {
		return nil, fmt.Errorf("ort: create tensor: data length %d does not match shape element count %d", len(data), count)
	}

	if count == 0 {
		return createEmptyTensor(dtype, shape)
	}

	t := &Tensor{dtype: dtype, shape: copyShape(shape), owned: false}
	t.pinner.Pin(&data[0])
	t.keep = data

	var cShape *C.int64_t
	if len(shape) > 0 {
		cShape = (*C.int64_t)(unsafe.Pointer(&shape[0]))
	}
	dataSize := C.size_t(len(data)) * C.size_t(elemSize(dtype))
	dataPtr := unsafe.Pointer(&data[0])

	var value *C.OrtValue
	if err := checkStatus(C.ort_CreateTensorWithDataAsOrtValue(
		cpuMemInfo, dataPtr, dataSize, cShape, C.size_t(len(shape)),
		C.ONNXTensorElementDataType(dtype), &value)); err != nil {
		t.pinner.Unpin()
		return nil, wrapErr("create tensor", err)
	}

	t.value = value
	return t, nil
}

// NewTensorFromBytes creates a tensor from raw bytes for types without a
// native Go mapping (Float16, BFloat16) or for fully dynamic pipelines.
// len(data) must equal the product of shape multiplied by the element size of dtype.
func NewTensorFromBytes(dtype TensorElementDataType, shape []int64, data []byte) (*Tensor, error) {
	if err := checkInit(); err != nil {
		return nil, err
	}

	count := shapeElementCount(shape)
	if count < 0 {
		return nil, fmt.Errorf("ort: create tensor: shape contains negative dimension")
	}

	es := elemSize(dtype)
	if es == 0 {
		return nil, fmt.Errorf("ort: create tensor: unsupported dtype %s", dtype)
	}

	expected := int(count) * es
	if len(data) != expected {
		return nil, fmt.Errorf("ort: create tensor: data length %d does not match expected %d bytes", len(data), expected)
	}

	if count == 0 {
		return createEmptyTensor(dtype, shape)
	}

	t := &Tensor{dtype: dtype, shape: copyShape(shape), owned: false}
	t.pinner.Pin(&data[0])
	t.keep = data

	var cShape *C.int64_t
	if len(shape) > 0 {
		cShape = (*C.int64_t)(unsafe.Pointer(&shape[0]))
	}
	dataSize := C.size_t(len(data))
	dataPtr := unsafe.Pointer(&data[0])

	var value *C.OrtValue
	if err := checkStatus(C.ort_CreateTensorWithDataAsOrtValue(
		cpuMemInfo, dataPtr, dataSize, cShape, C.size_t(len(shape)),
		C.ONNXTensorElementDataType(dtype), &value)); err != nil {
		t.pinner.Unpin()
		return nil, wrapErr("create tensor", err)
	}

	t.value = value
	return t, nil
}

// TensorData returns a typed view of the tensor's data buffer. The returned
// slice is valid until the tensor is closed. Returns an error if T's dtype
// does not match the tensor's element type.
func TensorData[T TensorElement](t *Tensor) ([]T, error) {
	if t.closed {
		return nil, fmt.Errorf("ort: tensor data: tensor is closed")
	}

	expected := dtypeOf[T]()
	if t.dtype != expected {
		return nil, fmt.Errorf("ort: tensor data: type mismatch: tensor is %s, requested %s", t.dtype, expected)
	}

	count := shapeElementCount(t.shape)
	if count == 0 {
		return nil, nil
	}

	var dataPtr unsafe.Pointer
	if err := checkStatus(C.ort_GetTensorMutableData(t.value, &dataPtr)); err != nil {
		return nil, wrapErr("tensor data", err)
	}

	return unsafe.Slice((*T)(dataPtr), count), nil
}

// Bytes returns a raw byte view of the tensor's data buffer.
func (t *Tensor) Bytes() ([]byte, error) {
	if t.closed {
		return nil, fmt.Errorf("ort: tensor bytes: tensor is closed")
	}

	count := shapeElementCount(t.shape)
	if count == 0 {
		return nil, nil
	}

	var dataPtr unsafe.Pointer
	if err := checkStatus(C.ort_GetTensorMutableData(t.value, &dataPtr)); err != nil {
		return nil, wrapErr("tensor bytes", err)
	}

	nbytes := int(count) * elemSize(t.dtype)
	return unsafe.Slice((*byte)(dataPtr), nbytes), nil
}

// Shape returns a copy of the tensor's shape.
func (t *Tensor) Shape() []int64 {
	return copyShape(t.shape)
}

// DataType returns the tensor's element data type.
func (t *Tensor) DataType() TensorElementDataType {
	return t.dtype
}

// ElementCount returns the total number of elements.
func (t *Tensor) ElementCount() int64 {
	return shapeElementCount(t.shape)
}

// ValueType returns the ONNX type of the underlying OrtValue.
func (t *Tensor) ValueType() (int, error) {
	if t.closed {
		return 0, fmt.Errorf("ort: value type: tensor is closed")
	}
	var onnxType C.enum_ONNXType
	if err := checkStatus(C.ort_GetValueType(t.value, &onnxType)); err != nil {
		return 0, wrapErr("get value type", err)
	}
	return int(onnxType), nil
}

// IsTensor reports whether this value is a tensor (as opposed to sequence/map).
func (t *Tensor) IsTensor() bool {
	vt, err := t.ValueType()
	return err == nil && vt == 1 // ONNX_TYPE_TENSOR
}

// IsSequence reports whether this value is a sequence.
func (t *Tensor) IsSequence() bool {
	vt, err := t.ValueType()
	return err == nil && vt == 3 // ONNX_TYPE_SEQUENCE
}

// IsMap reports whether this value is a map.
func (t *Tensor) IsMap() bool {
	vt, err := t.ValueType()
	return err == nil && vt == 4 // ONNX_TYPE_MAP
}

// SequenceLen returns the number of elements in a sequence value.
func (t *Tensor) SequenceLen() (int, error) {
	if t.closed {
		return 0, fmt.Errorf("ort: sequence len: value is closed")
	}
	var count C.size_t
	if err := checkStatus(C.ort_GetValueCount(t.value, &count)); err != nil {
		return 0, wrapErr("get value count", err)
	}
	return int(count), nil
}

// SequenceAt returns the element at the given index from a sequence value.
// The returned tensor must be closed by the caller.
func (t *Tensor) SequenceAt(index int) (*Tensor, error) {
	if t.closed {
		return nil, fmt.Errorf("ort: sequence at: value is closed")
	}
	var allocator *C.OrtAllocator
	if err := checkStatus(C.ort_GetAllocatorWithDefaultOptions(&allocator)); err != nil {
		return nil, wrapErr("get allocator", err)
	}
	var value *C.OrtValue
	if err := checkStatus(C.ort_GetValue(t.value, C.int(index), allocator, &value)); err != nil {
		return nil, wrapErr("get value", err)
	}
	return wrapOutputTensor(value)
}

// Close releases the tensor's resources. It is idempotent.
func (t *Tensor) Close() error {
	if t.closed {
		return nil
	}
	t.closed = true
	if t.value != nil {
		C.ort_ReleaseValue(t.value)
		t.value = nil
	}
	if !t.owned {
		t.pinner.Unpin()
	}
	t.keep = nil
	return nil
}

func createEmptyTensor(dtype TensorElementDataType, shape []int64) (*Tensor, error) {
	var allocator *C.OrtAllocator
	if err := checkStatus(C.ort_GetAllocatorWithDefaultOptions(&allocator)); err != nil {
		return nil, wrapErr("get allocator", err)
	}

	cShape := (*C.int64_t)(nil)
	if len(shape) > 0 {
		cShape = (*C.int64_t)(unsafe.Pointer(&shape[0]))
	}

	var value *C.OrtValue
	if err := checkStatus(C.ort_CreateTensorAsOrtValue(
		allocator, cShape, C.size_t(len(shape)),
		C.ONNXTensorElementDataType(dtype), &value)); err != nil {
		return nil, wrapErr("create empty tensor", err)
	}

	return &Tensor{
		value: value,
		dtype: dtype,
		shape: copyShape(shape),
		owned: true,
	}, nil
}

func wrapOutputTensor(value *C.OrtValue) (*Tensor, error) {
	var info *C.OrtTensorTypeAndShapeInfo
	if err := checkStatus(C.ort_GetTensorTypeAndShape(value, &info)); err != nil {
		return nil, wrapErr("get output tensor type", err)
	}
	defer C.ort_ReleaseTensorTypeAndShapeInfo(info)

	var cDtype C.enum_ONNXTensorElementDataType
	if err := checkStatus(C.ort_GetTensorElementType(info, &cDtype)); err != nil {
		return nil, wrapErr("get output element type", err)
	}

	var ndims C.size_t
	if err := checkStatus(C.ort_GetDimensionsCount(info, &ndims)); err != nil {
		return nil, wrapErr("get output dims count", err)
	}

	shape := make([]int64, int(ndims))
	if ndims > 0 {
		if err := checkStatus(C.ort_GetDimensions(info, (*C.int64_t)(unsafe.Pointer(&shape[0])), ndims)); err != nil {
			return nil, wrapErr("get output dims", err)
		}
	}

	return &Tensor{
		value: value,
		dtype: TensorElementDataType(cDtype),
		shape: shape,
		owned: true,
	}, nil
}

func shapeElementCount(shape []int64) int64 {
	if len(shape) == 0 {
		return 1
	}
	count := int64(1)
	for _, d := range shape {
		if d < 0 {
			return -1
		}
		count *= d
	}
	return count
}

func copyShape(s []int64) []int64 {
	out := make([]int64, len(s))
	copy(out, s)
	return out
}

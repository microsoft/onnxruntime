package onnxruntime

import (
	"fmt"
	"unsafe"
)

// NewEmptyTensor creates a new tensor value with allocated but uninitialized memory.
// Use GetTensorMutableData to fill it before running inference.
func NewEmptyTensor(alloc OrtAllocator, shape []int64, dtype TensorElementDataType) (OrtValue, error) {
	var val OrtValue
	fn := getFuncPtr(fnCreateTensorAsOrtValue)
	status := OrtStatus(ortDispatch(fn,
		uintptr(alloc),
		uintptr(unsafe.Pointer(&shape[0])),
		uintptr(len(shape)),
		uintptr(dtype),
		uintptr(unsafe.Pointer(&val)),
	))
	return val, statusToGoErr(status)
}

// NewTensor creates a tensor value from an existing Go slice.
// The data is copied into ORT-managed memory; the original Go slice is not retained.
func NewTensor(memInfo OrtMemoryInfo, data any, shape []int64) (OrtValue, error) {
	dtype, dataPtr, dataLen, err := inferTensorInfo(data)
	if err != nil {
		return 0, err
	}
	var val OrtValue
	fn := getFuncPtr(fnCreateTensorWithDataAsOrtValue)
	status := OrtStatus(ortDispatch(fn,
		uintptr(memInfo),
		uintptr(dataPtr),
		uintptr(dataLen),
		uintptr(unsafe.Pointer(&shape[0])),
		uintptr(len(shape)),
		uintptr(dtype),
		uintptr(unsafe.Pointer(&val)),
	))
	return val, statusToGoErr(status)
}

// NewTensorWithData creates a tensor from a Go slice without copying.
// The slice's backing array becomes the tensor's data buffer and must not be
// garbage-collected or modified while ORT is using it (during Run).
func NewTensorWithData(memInfo OrtMemoryInfo, data any, shape []int64) (OrtValue, error) {
	return NewTensor(memInfo, data, shape)
}

// GetTensorMutableData returns a pointer to the raw tensor data for in-place reads or writes.
func GetTensorMutableData(val OrtValue) (unsafe.Pointer, error) {
	var dataPtr unsafe.Pointer
	fn := getFuncPtr(fnGetTensorMutableData)
	status := OrtStatus(ortDispatch(fn, uintptr(val), uintptr(unsafe.Pointer(&dataPtr))))
	return dataPtr, statusToGoErr(status)
}

// GetTensorData copies tensor data into the provided Go slice pointer.
// dest must be a pointer to a slice (e.g., *[]float32).
func GetTensorData(val OrtValue, dest any) error {
	shape, dtype, err := GetTensorTypeAndShape(val)
	if err != nil {
		return fmt.Errorf("get shape: %w", err)
	}

	dataPtr, err := GetTensorMutableData(val)
	if err != nil {
		ReleaseTensorTypeAndShapeInfo(shape)
		return fmt.Errorf("get data pointer: %w", err)
	}

	count, err := GetTensorShapeElementCount(shape)
	ReleaseTensorTypeAndShapeInfo(shape)
	if err != nil {
		return fmt.Errorf("get element count: %w", err)
	}

	return copyFromPtr(dataPtr, dtype, int(count), dest)
}

// GetTensorTypeAndShape returns the type and shape info for a tensor value.
// The returned OrtTensorTypeAndShapeInfo must be released via ReleaseTensorTypeAndShapeInfo.
func GetTensorTypeAndShape(val OrtValue) (OrtTensorTypeAndShapeInfo, TensorElementDataType, error) {
	var info OrtTensorTypeAndShapeInfo
	fn := getFuncPtr(fnGetTensorTypeAndShape)
	status := OrtStatus(ortDispatch(fn, uintptr(val), uintptr(unsafe.Pointer(&info))))
	if err := statusToGoErr(status); err != nil {
		return 0, 0, err
	}

	var elemType TensorElementDataType
	elemFn := getFuncPtr(fnGetTensorElementType)
	ortDispatch(elemFn, uintptr(info), uintptr(unsafe.Pointer(&elemType)))

	return info, elemType, nil
}

// GetTensorShapeElementCount returns the total number of elements in the tensor shape.
func GetTensorShapeElementCount(info OrtTensorTypeAndShapeInfo) (int, error) {
	var count uint64
	fn := getFuncPtr(fnGetTensorShapeElementCount)
	status := OrtStatus(ortDispatch(fn, uintptr(info), uintptr(unsafe.Pointer(&count))))
	return int(count), statusToGoErr(status)
}

// GetTensorDimensions returns the dimensions of a tensor.
func GetTensorDimensions(info OrtTensorTypeAndShapeInfo) ([]int64, error) {
	var dimCount uint64
	countFn := getFuncPtr(fnGetDimensionsCount)
	status := OrtStatus(ortDispatch(countFn, uintptr(info), uintptr(unsafe.Pointer(&dimCount))))
	if err := statusToGoErr(status); err != nil {
		return nil, err
	}

	dims := make([]int64, dimCount)
	dimFn := getFuncPtr(fnGetDimensions)
	status = OrtStatus(ortDispatch(dimFn, uintptr(info), uintptr(unsafe.Pointer(&dims[0])), uintptr(len(dims))))
	if err := statusToGoErr(status); err != nil {
		return nil, err
	}

	return dims, nil
}

// ReleaseTensorTypeAndShapeInfo releases tensor type and shape info.
func ReleaseTensorTypeAndShapeInfo(info OrtTensorTypeAndShapeInfo) {
	fn := getFuncPtr(fnReleaseTensorTypeAndShapeInfo)
	if fn != 0 {
		ortDispatch(fn, uintptr(info))
	}
}

// IsTensor checks whether an OrtValue is a tensor.
func IsTensor(val OrtValue) (bool, error) {
	var isTensor int32
	fn := getFuncPtr(fnIsTensor)
	status := OrtStatus(ortDispatch(fn, uintptr(val), uintptr(unsafe.Pointer(&isTensor))))
	return isTensor != 0, statusToGoErr(status)
}

// ReleaseValue releases an OrtValue, freeing its underlying memory.
func ReleaseValue(val OrtValue) {
	fn := getFuncPtr(fnReleaseValue)
	if fn != 0 {
		ortDispatch(fn, uintptr(val))
	}
}

func inferTensorInfo(data any) (TensorElementDataType, unsafe.Pointer, int, error) {
	switch v := data.(type) {
	case []float32:
		return TensorFloat32, unsafe.Pointer(unsafe.SliceData(v)), len(v) * 4, nil
	case []float64:
		return TensorFloat64, unsafe.Pointer(unsafe.SliceData(v)), len(v) * 8, nil
	case []int32:
		return TensorInt32, unsafe.Pointer(unsafe.SliceData(v)), len(v) * 4, nil
	case []int64:
		return TensorInt64, unsafe.Pointer(unsafe.SliceData(v)), len(v) * 8, nil
	case []int8:
		return TensorInt8, unsafe.Pointer(unsafe.SliceData(v)), len(v), nil
	case []uint8:
		return TensorUint8, unsafe.Pointer(unsafe.SliceData(v)), len(v), nil
	case []int16:
		return TensorInt16, unsafe.Pointer(unsafe.SliceData(v)), len(v) * 2, nil
	case []uint16:
		return TensorUint16, unsafe.Pointer(unsafe.SliceData(v)), len(v) * 2, nil
	case []uint32:
		return TensorUint32, unsafe.Pointer(unsafe.SliceData(v)), len(v) * 4, nil
	case []uint64:
		return TensorUint64, unsafe.Pointer(unsafe.SliceData(v)), len(v) * 8, nil
	case []bool:
		return TensorBool, unsafe.Pointer(unsafe.SliceData(v)), len(v), nil
	default:
		return 0, nil, 0, fmt.Errorf("data must be a slice, got %T", data)
	}
}

func copyFromPtr(ptr unsafe.Pointer, dtype TensorElementDataType, count int, dest any) error {
	switch dtype {
	case TensorFloat32:
		src := unsafe.Slice((*float32)(ptr), count)
		d, ok := dest.(*[]float32)
		if !ok {
			return fmt.Errorf("dest must be *[]float32, got %T", dest)
		}
		*d = make([]float32, count)
		copy(*d, src)
	case TensorFloat64:
		src := unsafe.Slice((*float64)(ptr), count)
		d, ok := dest.(*[]float64)
		if !ok {
			return fmt.Errorf("dest must be *[]float64, got %T", dest)
		}
		*d = make([]float64, count)
		copy(*d, src)
	case TensorInt32:
		src := unsafe.Slice((*int32)(ptr), count)
		d, ok := dest.(*[]int32)
		if !ok {
			return fmt.Errorf("dest must be *[]int32, got %T", dest)
		}
		*d = make([]int32, count)
		copy(*d, src)
	case TensorInt64:
		src := unsafe.Slice((*int64)(ptr), count)
		d, ok := dest.(*[]int64)
		if !ok {
			return fmt.Errorf("dest must be *[]int64, got %T", dest)
		}
		*d = make([]int64, count)
		copy(*d, src)
	case TensorInt8:
		src := unsafe.Slice((*int8)(ptr), count)
		d, ok := dest.(*[]int8)
		if !ok {
			return fmt.Errorf("dest must be *[]int8, got %T", dest)
		}
		*d = make([]int8, count)
		copy(*d, src)
	case TensorUint8:
		src := unsafe.Slice((*uint8)(ptr), count)
		d, ok := dest.(*[]uint8)
		if !ok {
			return fmt.Errorf("dest must be *[]uint8, got %T", dest)
		}
		*d = make([]uint8, count)
		copy(*d, src)
	case TensorBool:
		src := unsafe.Slice((*bool)(ptr), count)
		d, ok := dest.(*[]bool)
		if !ok {
			return fmt.Errorf("dest must be *[]bool, got %T", dest)
		}
		*d = make([]bool, count)
		copy(*d, src)
	default:
		return fmt.Errorf("unsupported tensor element type %d for copy", dtype)
	}
	return nil
}

package onnxruntime

import (
	"unsafe"

	"github.com/microsoft/onnxruntime/go/internal/cstrings"
)

// NewMemoryInfo creates a memory info descriptor for a named memory region.
func NewMemoryInfo(name string, allocType AllocatorType, id int32, memType MemType) (OrtMemoryInfo, error) {
	nameBytes := cstrings.StringToCBytes(name)
	var info OrtMemoryInfo
	fn := getFuncPtr(fnCreateMemoryInfo)
	status := OrtStatus(ortDispatch(fn,
		uintptr(unsafe.Pointer(&nameBytes[0])),
		uintptr(allocType),
		uintptr(id),
		uintptr(memType),
		uintptr(unsafe.Pointer(&info)),
	))
	return info, statusToGoErr(status)
}

// NewCpuMemoryInfo creates memory info for CPU-accessible memory.
func NewCpuMemoryInfo(allocType AllocatorType, memType MemType) (OrtMemoryInfo, error) {
	var info OrtMemoryInfo
	fn := getFuncPtr(fnCreateCpuMemoryInfo)
	status := OrtStatus(ortDispatch(fn,
		uintptr(allocType),
		uintptr(memType),
		uintptr(unsafe.Pointer(&info)),
	))
	return info, statusToGoErr(status)
}

// ReleaseMemoryInfo releases a memory info object.
func ReleaseMemoryInfo(info OrtMemoryInfo) {
	fn := getFuncPtr(fnReleaseMemoryInfo)
	if fn != 0 {
		ortDispatch(fn, uintptr(info))
	}
}

// DefaultCPUAllocatorMemoryInfo returns memory info suitable for CPU tensors
// with the default arena allocator.
func DefaultCPUAllocatorMemoryInfo() (OrtMemoryInfo, error) {
	return NewCpuMemoryInfo(AllocArena, MemDefault)
}

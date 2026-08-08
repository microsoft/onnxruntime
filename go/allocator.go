package onnxruntime

import "unsafe"

// GetAllocatorWithDefaultOptions returns the default CPU arena allocator.
func GetAllocatorWithDefaultOptions() (OrtAllocator, error) {
	var alloc OrtAllocator
	fn := getFuncPtr(fnGetAllocatorWithDefaultOptions)
	status := OrtStatus(ortDispatch(fn, uintptr(unsafe.Pointer(&alloc))))
	return alloc, statusToGoErr(status)
}

// AllocatorAlloc allocates memory from the given allocator.
// The returned memory must be freed with AllocatorFree.
func AllocatorAlloc(alloc OrtAllocator, size uint64) (unsafe.Pointer, error) {
	var ptr unsafe.Pointer
	fn := getFuncPtr(fnAllocatorAlloc)
	status := OrtStatus(ortDispatch(fn, uintptr(alloc), uintptr(size), uintptr(unsafe.Pointer(&ptr))))
	return ptr, statusToGoErr(status)
}

// AllocatorFree frees memory previously allocated by AllocatorAlloc.
func AllocatorFree(alloc OrtAllocator, ptr unsafe.Pointer) {
	fn := getFuncPtr(fnAllocatorFree)
	if fn != 0 {
		ortDispatch(fn, uintptr(alloc), uintptr(ptr))
	}
}

// ReleaseAllocator releases an allocator object.
func ReleaseAllocator(alloc OrtAllocator) {
	fn := getFuncPtr(fnReleaseAllocator)
	if fn != 0 {
		ortDispatch(fn, uintptr(alloc))
	}
}

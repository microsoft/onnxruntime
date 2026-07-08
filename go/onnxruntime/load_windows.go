//go:build windows

package onnxruntime

import (
	"fmt"
	"syscall"
	"unsafe"
)

func platformLibraryName() string {
	return "onnxruntime.dll"
}

func loadLibrary(path string) (unsafe.Pointer, error) {
	dll, err := syscall.LoadDLL(path)
	if err != nil {
		return nil, fmt.Errorf("LoadDLL %s: %w", path, err)
	}

	proc, err := dll.FindProc("OrtGetApiBase")
	if err != nil {
		return nil, fmt.Errorf("FindProc OrtGetApiBase: %w", err)
	}

	return unsafe.Pointer(proc.Addr()), nil
}

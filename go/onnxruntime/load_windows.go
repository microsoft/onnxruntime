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

// libHandle is the platform handle for a loaded shared library.
type libHandle *syscall.DLL

// loadLibrary loads the shared library at path and resolves OrtGetApiBase.
// On success the caller owns the returned handle and must pass it to
// closeLibrary to unload the library.
func loadLibrary(path string) (unsafe.Pointer, libHandle, error) {
	dll, err := syscall.LoadDLL(path)
	if err != nil {
		return nil, nil, fmt.Errorf("LoadDLL %s: %w", path, err)
	}

	proc, err := dll.FindProc("OrtGetApiBase")
	if err != nil {
		_ = dll.Release()
		return nil, nil, fmt.Errorf("FindProc OrtGetApiBase: %w", err)
	}

	return unsafe.Pointer(proc.Addr()), libHandle(dll), nil
}

// closeLibrary unloads a library previously returned by loadLibrary.
func closeLibrary(h libHandle) {
	if h != nil {
		_ = (*syscall.DLL)(h).Release()
	}
}

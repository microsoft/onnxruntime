package onnxruntime

/*
#cgo LDFLAGS: -lonnxruntime
#include "onnxruntime_c_api.h"
#include <stdint.h>

static uintptr_t cgocall1(void* fn, uintptr_t a1) {
    typedef uintptr_t (*f)(uintptr_t);
    return ((f)fn)(a1);
}
*/
import "C"
import (
	"fmt"
	"unsafe"
)

func doInitialize(libPath string) error {
	apiBasePtr := uintptr(unsafe.Pointer(C.OrtGetApiBase()))
	if apiBasePtr == 0 {
		return fmt.Errorf("onnxruntime: OrtGetApiBase returned nil")
	}

	getApiAddr := *(*uintptr)(unsafe.Add(unsafe.Pointer(nil), apiBasePtr))
	apiBase = unsafe.Add(unsafe.Pointer(nil), uintptr(C.cgocall1(unsafe.Pointer(getApiAddr), C.uintptr_t(ORTAPIVersion))))
	if apiBase == nil {
		return fmt.Errorf("onnxruntime: GetApi returned nil for version %d", ORTAPIVersion)
	}

	return nil
}

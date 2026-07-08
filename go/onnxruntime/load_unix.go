//go:build !windows

package onnxruntime

/*
#cgo LDFLAGS: -ldl
#include <dlfcn.h>
#include <stdlib.h>
#include "cshim.h"

static void *ort_dlopen(const char *path) {
    return dlopen(path, RTLD_NOW | RTLD_LOCAL);
}

static void *ort_dlsym(void *handle, const char *symbol) {
    return dlsym(handle, symbol);
}

static const char *ort_dlerror(void) {
    return dlerror();
}
*/
import "C"
import (
	"fmt"
	"runtime"
	"unsafe"
)

func platformLibraryName() string {
	if runtime.GOOS == "darwin" {
		return "libonnxruntime.dylib"
	}
	return "libonnxruntime.so"
}

func loadLibrary(path string) (unsafe.Pointer, error) {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))

	handle := C.ort_dlopen(cpath)
	if handle == nil {
		errMsg := C.GoString(C.ort_dlerror())
		return nil, fmt.Errorf("dlopen %s: %s", path, errMsg)
	}

	sym := C.CString("OrtGetApiBase")
	defer C.free(unsafe.Pointer(sym))

	fn := C.ort_dlsym(handle, sym)
	if fn == nil {
		errMsg := C.GoString(C.ort_dlerror())
		return nil, fmt.Errorf("dlsym OrtGetApiBase: %s", errMsg)
	}

	return fn, nil
}

//go:build windows

package onnxruntime

/*
#include "cshim.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"syscall"
	"unsafe"
)

// ortPath converts path to a NUL-terminated ORTCHAR_T* for the ORT C API.
// The returned release func must be called once the C call has returned.
//
// On Windows ORTCHAR_T is wchar_t, so the path must be UTF-16. The []uint16
// that UTF16PtrFromString allocates holds no Go pointers, and ORT copies the
// path during the call rather than retaining it, so passing Go memory to C is
// permitted here and no copy into C memory is needed.
func ortPath(path string) (*C.ORTCHAR_T, func(), error) {
	p, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return nil, func() {}, fmt.Errorf("ort: invalid path %q: %w", path, err)
	}
	return (*C.ORTCHAR_T)(unsafe.Pointer(p)), func() {}, nil
}

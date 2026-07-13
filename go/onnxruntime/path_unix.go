//go:build !windows

package onnxruntime

/*
#include "cshim.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"strings"
	"unsafe"
)

// ortPath converts path to a NUL-terminated ORTCHAR_T* for the ORT C API.
// The returned release func must be called once the C call has returned.
//
// On POSIX ORTCHAR_T is char, so the path is passed through as UTF-8.
func ortPath(path string) (*C.ORTCHAR_T, func(), error) {
	if strings.IndexByte(path, 0) != -1 {
		return nil, func() {}, fmt.Errorf("ort: path contains NUL byte: %q", path)
	}
	// C.ORTCHAR_T is an alias for C.char here, so C.CString needs no conversion.
	p := C.CString(path)
	return p, func() { C.free(unsafe.Pointer(p)) }, nil
}

package onnxruntime

/*
#cgo LDFLAGS: -lonnxruntime
#include <stdint.h>

// Dispatch helpers: call a function pointer through cgo's stack switching.
// This avoids the assembly trampoline's goroutine-stack issue.
static uintptr_t cgocall0(void* fn) {
    typedef uintptr_t (*f)(void);
    return ((f)fn)();
}
static uintptr_t cgocall1(void* fn, uintptr_t a1) {
    typedef uintptr_t (*f)(uintptr_t);
    return ((f)fn)(a1);
}
static uintptr_t cgocall2(void* fn, uintptr_t a1, uintptr_t a2) {
    typedef uintptr_t (*f)(uintptr_t, uintptr_t);
    return ((f)fn)(a1, a2);
}
static uintptr_t cgocall3(void* fn, uintptr_t a1, uintptr_t a2, uintptr_t a3) {
    typedef uintptr_t (*f)(uintptr_t, uintptr_t, uintptr_t);
    return ((f)fn)(a1, a2, a3);
}
static uintptr_t cgocall4(void* fn, uintptr_t a1, uintptr_t a2, uintptr_t a3, uintptr_t a4) {
    typedef uintptr_t (*f)(uintptr_t, uintptr_t, uintptr_t, uintptr_t);
    return ((f)fn)(a1, a2, a3, a4);
}
static uintptr_t cgocall5(void* fn, uintptr_t a1, uintptr_t a2, uintptr_t a3, uintptr_t a4, uintptr_t a5) {
    typedef uintptr_t (*f)(uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t);
    return ((f)fn)(a1, a2, a3, a4, a5);
}
static uintptr_t cgocall6(void* fn, uintptr_t a1, uintptr_t a2, uintptr_t a3, uintptr_t a4, uintptr_t a5, uintptr_t a6) {
    typedef uintptr_t (*f)(uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t);
    return ((f)fn)(a1, a2, a3, a4, a5, a6);
}
static uintptr_t cgocall7(void* fn, uintptr_t a1, uintptr_t a2, uintptr_t a3, uintptr_t a4, uintptr_t a5, uintptr_t a6, uintptr_t a7) {
    typedef uintptr_t (*f)(uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t);
    return ((f)fn)(a1, a2, a3, a4, a5, a6, a7);
}
static uintptr_t cgocall8(void* fn, uintptr_t a1, uintptr_t a2, uintptr_t a3, uintptr_t a4, uintptr_t a5, uintptr_t a6, uintptr_t a7, uintptr_t a8) {
    typedef uintptr_t (*f)(uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t);
    return ((f)fn)(a1, a2, a3, a4, a5, a6, a7, a8);
}
*/
import "C"
import "unsafe"

// ortDispatch uses cgo helper functions (cgocall0-cgocall8) instead of
// assembly trampolines. cgo switches to the system stack before calling
// C code, which is required for libraries that use threads or call back
// into Go memory management (like ONNX Runtime).
func ortDispatch(fn uintptr, args ...uintptr) uintptr {
	ptr := unsafe.Pointer(fn)
	switch len(args) {
	case 0:
		return uintptr(C.cgocall0(ptr))
	case 1:
		return uintptr(C.cgocall1(ptr, C.uintptr_t(args[0])))
	case 2:
		return uintptr(C.cgocall2(ptr, C.uintptr_t(args[0]), C.uintptr_t(args[1])))
	case 3:
		return uintptr(C.cgocall3(ptr, C.uintptr_t(args[0]), C.uintptr_t(args[1]), C.uintptr_t(args[2])))
	case 4:
		return uintptr(C.cgocall4(ptr, C.uintptr_t(args[0]), C.uintptr_t(args[1]), C.uintptr_t(args[2]), C.uintptr_t(args[3])))
	case 5:
		return uintptr(C.cgocall5(ptr, C.uintptr_t(args[0]), C.uintptr_t(args[1]), C.uintptr_t(args[2]), C.uintptr_t(args[3]), C.uintptr_t(args[4])))
	case 6:
		return uintptr(C.cgocall6(ptr, C.uintptr_t(args[0]), C.uintptr_t(args[1]), C.uintptr_t(args[2]), C.uintptr_t(args[3]), C.uintptr_t(args[4]), C.uintptr_t(args[5])))
	case 7:
		return uintptr(C.cgocall7(ptr, C.uintptr_t(args[0]), C.uintptr_t(args[1]), C.uintptr_t(args[2]), C.uintptr_t(args[3]), C.uintptr_t(args[4]), C.uintptr_t(args[5]), C.uintptr_t(args[6])))
	case 8:
		return uintptr(C.cgocall8(ptr, C.uintptr_t(args[0]), C.uintptr_t(args[1]), C.uintptr_t(args[2]), C.uintptr_t(args[3]), C.uintptr_t(args[4]), C.uintptr_t(args[5]), C.uintptr_t(args[6]), C.uintptr_t(args[7])))
	default:
		panic("ortDispatch: too many arguments")
	}
}

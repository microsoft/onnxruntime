// Package cstrings provides utilities for converting between Go strings and C strings.
package cstrings

import "unsafe"

// CStringToString converts a null-terminated C string pointer to a Go string.
// The returned string is a copy; the original C memory can be freed safely.
func CStringToString(ptr *byte) string {
	if ptr == nil {
		return ""
	}
	var length int
	for {
		if *(*byte)(unsafe.Add(unsafe.Pointer(ptr), length)) == 0 {
			break
		}
		length++
	}
	return string(unsafe.Slice(ptr, length))
}

// StringToCBytes converts a Go string to a null-terminated byte slice suitable for passing to C.
func StringToCBytes(s string) []byte {
	b := make([]byte, len(s)+1)
	copy(b, s)
	b[len(s)] = 0
	return b
}

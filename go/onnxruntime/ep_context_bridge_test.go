//go:build orttest

package onnxruntime

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"runtime"
	"strings"
	"testing"
)

func assertNativeBuffer(t *testing.T, opts *SessionOptions, source *sourceModelBuffer, want []byte) {
	t.Helper()
	ptr, data, err := nativeSourceBuffer(opts)
	if err != nil {
		t.Fatal(err)
	}
	if ptr != uintptr(source.data) || !bytes.Equal(data, want) {
		t.Fatalf("native pointer/bytes: %x %x; want %x %x", ptr, data, uintptr(source.data), want)
	}
}

func TestWeightlessSourceModelBufferNative(t *testing.T) {
	if APIVersion() < 29 {
		opts, err := NewSessionOptions()
		if err != nil {
			t.Fatal(err)
		}
		defer opts.Close()
		_, _, err = nativeSourceBuffer(opts)
		requireOrtCode(t, err, ErrorCodeNotImplemented)
		return
	}
	opts := bufferOptions(t)
	original := []byte{1, 2, 3, 4}
	if err := opts.SetWeightlessSourceModelBuffer(original); err != nil {
		t.Fatal(err)
	}
	original[0] = 99
	source := opts.resources.source
	assertNativeBuffer(t, opts, source, []byte{1, 2, 3, 4})
	clone, err := opts.Clone()
	if err != nil {
		t.Fatal(err)
	}
	defer clone.Close()
	if err := opts.SetWeightlessSourceModelBuffer([]byte{5, 6}); err != nil {
		t.Fatal(err)
	}
	opts.Close()
	assertNativeBuffer(t, clone, source, []byte{1, 2, 3, 4})
	for _, fromBytes := range []bool{false, true} {
		observe := func(o *SessionOptions) (*SessionOptions, error) {
			return o.cloneLockedWithRestore(func(snapshot *SessionOptions) error {
				if err := restoreSourceModelBuffer(snapshot); err != nil {
					return err
				}
				ptr, data, err := nativeSourceBuffer(snapshot)
				if err != nil {
					return err
				}
				if ptr != uintptr(source.data) || !bytes.Equal(data, []byte{1, 2, 3, 4}) {
					return fmt.Errorf("constructor native snapshot lost buffer: %x %x", ptr, data)
				}
				return nil
			}, releaseNativeOptions)
		}
		var s *Session
		if fromBytes {
			model, readErr := os.ReadFile(testdataPath("add_f32.onnx"))
			if readErr != nil {
				t.Fatal(readErr)
			}
			s, err = newSessionFromBytes(model, clone, observe)
		} else {
			s, err = newSession(testdataPath("add_f32.onnx"), clone, observe)
		}
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(bufferBytes(s.resources.source), []byte{1, 2, 3, 4}) {
			t.Fatal("session lost captured storage")
		}
		s.Close()
	}
	for _, operation := range []string{"clone", "file", "bytes"} {
		sentinel := errors.New("forced buffer restoration failure")
		releases, restores := 0, 0
		failClone := func(o *SessionOptions) (*SessionOptions, error) {
			return o.cloneLockedWithRestore(func(candidate *SessionOptions) error {
				restores++
				// The clone owns a distinct native handle before restoration.
				if candidate.handle == nil || candidate.handle == o.handle {
					t.Error("invalid native clone")
				}
				return sentinel
			}, func(candidate *SessionOptions) { releases++; releaseNativeOptions(candidate) })
		}
		p, c, s := epCounts()
		refs := resourceRefCount(&source.ref)
		switch operation {
		case "clone":
			clone.mu.Lock()
			mu.Lock()
			result, callErr := failClone(clone)
			mu.Unlock()
			clone.mu.Unlock()
			err = callErr
			if result != nil {
				result.Close()
				t.Fatal("incomplete clone published")
			}
		case "file":
			result, callErr := newSession("/definitely/missing", clone, failClone)
			err = callErr
			if result != nil {
				result.Close()
				t.Fatal("session published")
			}
		case "bytes":
			result, callErr := newSessionFromBytes([]byte{1}, clone, failClone)
			err = callErr
			if result != nil {
				result.Close()
				t.Fatal("session published")
			}
		}
		if !errors.Is(err, sentinel) || releases != 1 || restores != 1 {
			t.Fatalf("%s: %v, %d restores, %d releases", operation, err, restores, releases)
		}
		if pp, cc, ss := epCounts(); pp != p || cc != c || ss != s {
			t.Fatal("restore failure changed counts")
		}
		if resourceRefCount(&source.ref) != refs {
			t.Fatal("restore failure changed references")
		}
		assertNativeBuffer(t, clone, source, []byte{1, 2, 3, 4})
	}
	clone.Close()
	if resourceRefCount(&source.ref) != 0 {
		t.Fatal("native clone leaked buffer")
	}
}

func TestWeightlessSourceModelBufferProvider(t *testing.T) {
	if APIVersion() < 29 {
		t.Skip("observer needs API 29; guards covered by native test")
	}
	if runtime.GOOS != "linux" {
		t.Skip("observer compiler flags validated only on Linux")
	}
	library, err := buildBufferObserver(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	const registration = "go_buffer_observer"
	if err := RegisterExecutionProviderLibrary(registration, library); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := UnregisterExecutionProviderLibrary(registration); err != nil {
			t.Error(err)
		}
	}()
	devices, err := GetEPDevices()
	if err != nil {
		t.Fatal(err)
	}
	var observer *EPDevice
	for _, device := range devices {
		if device.EPName() == "GoBufferObserver" {
			observer = device
		}
	}
	if observer == nil {
		t.Fatal("observer CPU device not discovered")
	}
	if observer.Device().Type() != HardwareDeviceTypeCPU {
		t.Fatal("observer did not advertise CPU")
	}
	for _, cloneFirst := range []bool{false, true} {
		opts, err := NewSessionOptions()
		if err != nil {
			t.Fatal(err)
		}
		defer opts.Close()
		if err := opts.AppendExecutionProviderV2([]*EPDevice{observer}, nil); err != nil {
			t.Fatal(err)
		}
		original := []byte{1, 2, 3, 4}
		if err := opts.SetWeightlessSourceModelBuffer(original); err != nil {
			t.Fatal(err)
		}
		original[0] = 99
		if cloneFirst {
			clone, err := opts.Clone()
			if err != nil {
				t.Fatal(err)
			}
			defer clone.Close()
			if err := opts.SetWeightlessSourceModelBuffer([]byte{5, 6}); err != nil {
				t.Fatal(err)
			}
			opts.Close()
			opts = clone
		}
		for _, fromBytes := range []bool{false, true} {
			p, c, s := epCounts()
			session, err := constructBufferSession(opts, fromBytes)
			requireOrtCode(t, err, ErrorCodeFail)
			var native *OrtError
			if !errors.As(err, &native) || !strings.HasSuffix(strings.TrimSpace(native.Msg), " : FAIL : GoBufferObserver length=4 bytes=01020304") {
				t.Fatalf("wrong provider observation: %q", native.Msg)
			}
			if session != nil {
				session.Close()
				t.Fatal("observer unexpectedly created session")
			}
			if pp, cc, ss := epCounts(); pp != p || cc != c || ss != s {
				t.Fatal("observer failure leaked counts")
			}
			if resourceRefCount(&opts.resources.source.ref) != 1 {
				t.Fatal("observer failure leaked buffer")
			}
		}
		// A conflicting constructor must stop before the observer's CreateEp sentinel.
		opts.policyAttempted = true
		for _, fromBytes := range []bool{false, true} {
			session, err := constructBufferSession(opts, fromBytes)
			requireOrtCode(t, err, ErrorCodeNotImplemented)
			if strings.Contains(err.Error(), "GoBufferObserver length") {
				t.Fatal("guard entered provider")
			}
			if session != nil {
				session.Close()
				t.Fatal("guard published session")
			}
		}
		opts.Close()
	}
}

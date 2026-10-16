//go:build orttest

package onnxruntime

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
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
	var reader *epContextReaderRegistration
	if APIVersion() >= 31 {
		if err := opts.SetEPContextDataReader(func(string) ([]byte, error) { return nil, nil }, 1); err != nil {
			t.Fatal(err)
		}
		reader = opts.resources.reader
	}
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
		readerRefs := 0
		if reader != nil {
			readerRefs = resourceRefCount(&reader.ref)
		}
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
		if reader != nil && resourceRefCount(&reader.ref) != readerRefs {
			t.Fatal("restore failure changed reader references")
		}
		assertNativeBuffer(t, clone, source, []byte{1, 2, 3, 4})
	}
	clone.Close()
	if reader != nil {
		assertReaderReleased(t, reader)
	}
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
	ccOutput, err := exec.Command("go", "env", "CC").Output()
	if err != nil {
		t.Fatalf("get cgo C compiler: %v", err)
	}
	compiler, err := splitCompilerCommand(string(ccOutput))
	if err != nil {
		t.Fatalf("parse cgo C compiler: %v", err)
	}
	library, err := buildBufferObserver(t.TempDir(), compiler)
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

type panickingReaderError struct{}

func (panickingReaderError) Error() string { panic("error formatter") }

func TestEPContextReaderBridge(t *testing.T) {
	if APIVersion() < 31 {
		t.Skip("reader bridge needs API 31; registration guards run without the tag")
	}
	for _, tc := range []struct {
		name, logicalName  string
		reader             EPContextDataReader
		limit              uint64
		mode, invalid      int
		code               ErrorCode
		message            string
		want               []byte
		allocations, frees int
	}{
		{name: "payload", logicalName: "上下文", reader: func(name string) ([]byte, error) {
			if name != "上下文" {
				return nil, errors.New("wrong name")
			}
			return []byte("abc"), nil
		}, limit: 3, want: []byte("abc")},
		{name: "empty", reader: func(string) ([]byte, error) { return nil, nil }, limit: 1},
		{name: "counted", reader: func(string) ([]byte, error) { return []byte("abc"), nil }, limit: 3, mode: 1, want: []byte("abc"), allocations: 1, frees: 1},
		{name: "hostAccessible", reader: func(string) ([]byte, error) { return []byte("abc"), nil }, limit: 3, mode: 4, want: []byte("abc"), allocations: 1, frees: 1},
		{name: "oversize", reader: func(string) ([]byte, error) { return []byte("abcd"), nil }, limit: 3, mode: 1, code: ErrorCodeInvalidArgument, message: "reader payload exceeds maxBytes"},
		{name: "error", reader: func(string) ([]byte, error) { return []byte("abc"), errors.New("user failure") }, limit: 3, mode: 1, code: ErrorCodeFail, message: "user failure"},
		{name: "nulError", reader: func(string) ([]byte, error) { return nil, errors.New("a\x00b") }, limit: 3, code: ErrorCodeFail, message: "a�b"},
		{name: "panic", reader: func(string) ([]byte, error) { panic("reader") }, limit: 3, code: ErrorCodeFail, message: "EPContext data reader panicked"},
		{name: "panicFormatter", reader: func(string) ([]byte, error) { return nil, panickingReaderError{} }, limit: 3, code: ErrorCodeFail, message: "EPContext data reader panicked"},
		{name: "utf8", logicalName: "\xff", reader: func(string) ([]byte, error) { t.Error("invalid name entered reader"); return nil, nil }, limit: 3, code: ErrorCodeInvalidArgument, message: "reader name is not UTF-8"},
		{name: "nullAllocation", reader: func(string) ([]byte, error) { return []byte("abc"), nil }, limit: 3, mode: 2, code: ErrorCodeFail, message: "reader allocator returned NULL", allocations: 1},
		{name: "deviceAllocator", reader: func(string) ([]byte, error) { t.Error("device allocator entered reader"); return nil, nil }, limit: 3, mode: 3, code: ErrorCodeInvalidArgument, message: "reader allocator memory is not host accessible"},
		{name: "missingAlloc", reader: func(string) ([]byte, error) { return nil, nil }, limit: 3, mode: 5, code: ErrorCodeInvalidArgument, message: "invalid reader allocator"},
		{name: "missingInfo", reader: func(string) ([]byte, error) { return nil, nil }, limit: 3, mode: 6, code: ErrorCodeInvalidArgument, message: "reader allocator has no memory info"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			opts, err := NewSessionOptions()
			if err != nil {
				t.Fatal(err)
			}
			defer opts.Close()
			if err := opts.SetEPContextDataReader(tc.reader, tc.limit); err != nil {
				t.Fatal(err)
			}
			registration := opts.resources.reader
			result, err := invokeNativeReader(opts, tc.logicalName, tc.mode, tc.invalid, false)
			if tc.code != ErrorCodeOK {
				requireOrtCode(t, err, tc.code)
				var native *OrtError
				if !errors.As(err, &native) || native.Msg != tc.message {
					t.Fatalf("wrong diagnostic: %v", err)
				}
				if !result.cleanFailure {
					t.Fatal("failure published outputs")
				}
			} else if err != nil || !bytes.Equal(result.data, tc.want) {
				t.Fatalf("payload %x: %v", result.data, err)
			}
			if result.allocations != tc.allocations || result.frees != tc.frees {
				t.Fatalf("allocator counts: %+v", result)
			}
			opts.Close()
			assertReaderReleased(t, registration)
		})
	}
	t.Run("clonedRegistration", func(t *testing.T) {
		opts, err := NewSessionOptions()
		if err != nil {
			t.Fatal(err)
		}
		defer opts.Close()
		if err := opts.SetEPContextDataReader(func(string) ([]byte, error) { return []byte("abc"), nil }, 3); err != nil {
			t.Fatal(err)
		}
		registration := opts.resources.reader
		clone, err := opts.Clone()
		if err != nil {
			t.Fatal(err)
		}
		defer clone.Close()
		if err := opts.ClearEPContextDataReader(); err != nil {
			t.Fatal(err)
		}
		_, err = invokeNativeReader(opts, "cleared", 0, 0, false)
		requireOrtCode(t, err, ErrorCodeFail)
		var native *OrtError
		if !errors.As(err, &native) || native.Msg != "reader not registered" {
			t.Fatalf("native Clear: %v", err)
		}
		opts.Close()
		result, err := invokeNativeReader(clone, "retained", 1, 0, false)
		if err != nil || !bytes.Equal(result.data, []byte("abc")) || result.allocations != 1 || result.frees != 1 {
			t.Fatalf("native clone registration: %+v %v", result, err)
		}
		clone.Close()
		assertReaderReleased(t, registration)
	})
	for invalid := 1; invalid <= 5; invalid++ {
		opts, err := NewSessionOptions()
		if err != nil {
			t.Fatal(err)
		}
		if err := opts.SetEPContextDataReader(func(string) ([]byte, error) { t.Error("invalid arguments entered reader"); return nil, nil }, 1); err != nil {
			t.Fatal(err)
		}
		result, err := invokeNativeReader(opts, "name", 1, invalid, false)
		requireOrtCode(t, err, ErrorCodeInvalidArgument)
		if !result.cleanFailure || result.allocations != 0 || result.frees != 0 {
			t.Fatalf("invalid args %d: %+v", invalid, result)
		}
		opts.Close()
	}
	opts, err := NewSessionOptions()
	if err != nil {
		t.Fatal(err)
	}
	defer opts.Close()
	var calls atomic.Int32
	if err := opts.SetEPContextDataReader(func(string) ([]byte, error) { calls.Add(1); return []byte("abc"), nil }, 3); err != nil {
		t.Fatal(err)
	}
	registration := opts.resources.reader
	var workers sync.WaitGroup
	for range 16 {
		workers.Go(func() {
			result, err := invokeNativeReader(opts, "concurrent", 1, 0, false)
			if err != nil || !bytes.Equal(result.data, []byte("abc")) || result.allocations != 1 || result.frees != 1 {
				t.Errorf("concurrent callback: %+v %v", result, err)
			}
		})
	}
	workers.Wait()
	if calls.Load() != 16 {
		t.Fatalf("calls: %d", calls.Load())
	}
	if runtime.GOOS == "linux" {
		result, err := invokeNativeReader(opts, "C thread", 1, 0, true)
		if err != nil || !bytes.Equal(result.data, []byte("abc")) || result.allocations != 1 || result.frees != 1 {
			t.Fatalf("C thread: %+v %v", result, err)
		}
	}
	opts.Close()
	assertReaderReleased(t, registration)
}

func TestEPContextReaderReentry(t *testing.T) {
	if APIVersion() < 31 {
		t.Skip("observer callback needs API 31")
	}
	if runtime.GOOS != "linux" {
		t.Skip("observer compiler flags validated only on Linux")
	}
	marker := os.Getenv("ORT_GO_REENTRY_CHILD")
	library := os.Getenv("ORT_GO_REENTRY_LIBRARY")
	if marker == "" {
		ccOutput, err := exec.Command("go", "env", "CC").Output()
		if err != nil {
			t.Fatalf("get cgo C compiler: %v", err)
		}
		compiler, err := splitCompilerCommand(string(ccOutput))
		if err != nil {
			t.Fatalf("parse cgo C compiler: %v", err)
		}
		library, err = buildBufferObserver(t.TempDir(), compiler)
		if err != nil {
			t.Fatal(err)
		}
	}
	for _, fromBytes := range []bool{false, true} {
		for _, action := range []string{"direct", "worker", "buffer", "clear", "replace", "close"} {
			name := fmt.Sprintf("bytes=%t/%s", fromBytes, action)
			t.Run(name, func(t *testing.T) {
				if marker == "" {
					ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
					defer cancel()
					cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestEPContextReaderReentry$/^"+fmt.Sprintf("bytes=%t", fromBytes)+"$/^"+action+"$", "-test.v")
					cmd.Env = append(os.Environ(), "ORT_GO_REENTRY_CHILD="+name, "ORT_GO_REENTRY_LIBRARY="+library)
					output, err := cmd.CombinedOutput()
					if ctx.Err() != nil {
						t.Fatalf("reader re-entry timed out: %v\n%s", ctx.Err(), output)
					}
					if err != nil {
						t.Fatalf("reader re-entry child: %v\n%s", err, output)
					}
					if !strings.Contains(string(output), "REENTRY_ASSERTIONS_OK "+name) {
						t.Fatalf("missing child assertions:\n%s", output)
					}
					if strings.Contains(string(output), "The requested API version") {
						t.Fatalf("API negotiation wrote to stderr:\n%s", output)
					}
					return
				}
				if marker != name {
					t.Fatalf("wrong child selected %q", name)
				}
				runReaderReentry(t, library, fromBytes, action)
				t.Log("REENTRY_ASSERTIONS_OK " + name)
			})
		}
	}
}

func runReaderReentry(t *testing.T, library string, fromBytes bool, action string) {
	t.Helper()
	p, c, s := epCounts()
	if err := RegisterExecutionProviderLibrary("go_reader_observer", library); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := UnregisterExecutionProviderLibrary("go_reader_observer"); err != nil {
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
		t.Fatal("observer not discovered")
	}
	opts, err := NewSessionOptions()
	if err != nil {
		t.Fatal(err)
	}
	defer opts.Close()
	if err := opts.AppendExecutionProviderV2([]*EPDevice{observer}, nil); err != nil {
		t.Fatal(err)
	}
	if err := opts.AddConfigEntry("reader.key", "value"); err != nil {
		t.Fatal(err)
	}
	if err := opts.SetWeightlessSourceModelBuffer([]byte{1, 2, 3, 4}); err != nil {
		t.Fatal(err)
	}
	source := opts.resources.source
	calls := 0
	var registration *epContextReaderRegistration
	permitted := func() error {
		value, err := opts.GetSessionConfigEntry("reader.key")
		if err != nil {
			return err
		}
		if value != "value" {
			return errors.New("wrong config")
		}
		if _, err := opts.GetExecutionMode(); err != nil {
			return err
		}
		if _, err := opts.IsMemPatternEnabled(); err != nil {
			return err
		}
		clone, err := opts.Clone()
		if err != nil {
			return err
		}
		ptr, data, inspectErr := nativeSourceBuffer(clone)
		if inspectErr == nil && (ptr != uintptr(source.data) || !bytes.Equal(data, []byte{1, 2, 3, 4})) {
			inspectErr = errors.New("reentrant clone lost buffer")
		}
		closeErr := clone.Close()
		if inspectErr != nil {
			return inspectErr
		}
		return closeErr
	}
	reader := func(name string) ([]byte, error) {
		calls++
		if name != "observer/上下文" {
			return nil, fmt.Errorf("wrong logical name: %q", name)
		}
		if resourceRefCount(&registration.ref) != 2 || resourceRefCount(&source.ref) != 2 {
			return nil, errors.New("snapshot references not retained")
		}
		var err error
		if action == "worker" {
			done := make(chan error, 1)
			go func() { done <- permitted() }()
			err = <-done
		} else {
			err = permitted()
		}
		if err != nil {
			return nil, err
		}
		switch action {
		case "buffer":
			err = opts.SetWeightlessSourceModelBuffer([]byte{9, 8})
		case "clear":
			err = opts.ClearEPContextDataReader()
		case "replace":
			err = opts.SetEPContextDataReader(func(string) ([]byte, error) { return nil, errors.New("replacement invoked") }, 1)
		case "close":
			err = opts.Close()
		}
		if err != nil {
			return nil, err
		}
		if !bytes.Equal(bufferBytes(source), []byte{1, 2, 3, 4}) {
			return nil, errors.New("snapshot buffer changed")
		}
		return []byte("abc"), nil
	}
	if err := opts.SetEPContextDataReader(reader, 3); err != nil {
		t.Fatal(err)
	}
	registration = opts.resources.reader
	session, err := constructBufferSession(opts, fromBytes)
	requireOrtCode(t, err, ErrorCodeFail)
	var native *OrtError
	if !errors.As(err, &native) || !strings.HasSuffix(strings.TrimSpace(native.Msg), " : FAIL : GoBufferObserver length=4 bytes=01020304 callback=abc source=unchanged") {
		t.Fatalf("wrong observation: %v", err)
	}
	if session != nil || calls != 1 {
		t.Fatalf("session %v calls %d", session, calls)
	}
	opts.Close()
	assertReaderReleased(t, registration)
	if resourceRefCount(&source.ref) != 0 {
		t.Fatal("buffer leaked")
	}
	if pp, cc, ss := epCounts(); pp != p || cc != c || ss != s {
		t.Fatalf("re-entry leaked counts: %d %d %d; want %d %d %d", pp, cc, ss, p, c, s)
	}
}

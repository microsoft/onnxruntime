package onnxruntime

import (
	"bytes"
	"context"
	"errors"
	"os"
	"reflect"
	"strings"
	"testing"
	"unsafe"
)

func requireOrtCode(t *testing.T, err error, code ErrorCode) {
	t.Helper()
	var native *OrtError
	if !errors.As(err, &native) || native.Code != code {
		t.Fatalf("expected %s, got %v", code, err)
	}
}

func bufferBytes(buffer *sourceModelBuffer) []byte {
	if buffer == nil {
		return nil
	}
	return bytes.Clone(unsafe.Slice((*byte)(buffer.data), int(buffer.size)))
}

func bufferOptions(t *testing.T) *SessionOptions {
	t.Helper()
	opts, err := NewSessionOptions()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { opts.Close() })
	if err := opts.AppendExecutionProviderV2([]*EPDevice{cpuEPDevice(t)}, nil); err != nil {
		t.Fatal(err)
	}
	if err := opts.SetWeightlessSourceModelBuffer([]byte{1, 2, 3, 4}); err != nil {
		t.Fatal(err)
	}
	return opts
}

func TestWeightlessSourceModelBuffer(t *testing.T) {
	opts, err := NewSessionOptions()
	if err != nil {
		t.Fatal(err)
	}
	defer opts.Close()
	if APIVersion() < 29 {
		requireOrtCode(t, opts.SetWeightlessSourceModelBuffer(nil), ErrorCodeNotImplemented)
		checkEPAdd(t, opts, false)
		return
	}
	requireOrtCode(t, opts.SetWeightlessSourceModelBuffer(nil), ErrorCodeInvalidArgument)
	original := []byte{1, 2, 3, 4}
	if err := opts.SetWeightlessSourceModelBuffer(original); err != nil {
		t.Fatal(err)
	}
	original[0] = 99
	clone, err := opts.Clone()
	if err != nil {
		t.Fatal(err)
	}
	defer clone.Close()
	retained := clone.resources.source
	if err := opts.SetWeightlessSourceModelBuffer([]byte{5, 6}); err != nil {
		t.Fatal(err)
	}
	opts.Close()
	if !bytes.Equal(bufferBytes(retained), []byte{1, 2, 3, 4}) {
		t.Fatal("source bytes not retained")
	}
	clone.Close()
	if resourceRefCount(&retained.ref) != 0 {
		t.Fatal("clone leaked buffer")
	}
	if err := opts.SetWeightlessSourceModelBuffer(original); err == nil {
		t.Fatal("closed options accepted")
	}
	var nilOpts *SessionOptions
	if err := nilOpts.SetWeightlessSourceModelBuffer(original); err == nil {
		t.Fatal("nil options accepted")
	}
}

func constructBufferSession(opts *SessionOptions, fromBytes bool) (*Session, error) {
	if fromBytes {
		data, err := os.ReadFile(testdataPath("add_f32.onnx"))
		if err != nil {
			return nil, err
		}
		return NewSessionFromBytes(data, opts)
	}
	return NewSession(testdataPath("add_f32.onnx"), opts)
}

func TestWeightlessSourceModelBufferPolicy(t *testing.T) {
	if APIVersion() < 29 {
		opts, err := NewSessionOptions()
		if err != nil {
			t.Fatal(err)
		}
		defer opts.Close()
		requireOrtCode(t, opts.SetWeightlessSourceModelBuffer([]byte{1}), ErrorCodeNotImplemented)
		return
	}
	for policy := ExecutionProviderDevicePolicyDefault; policy <= ExecutionProviderDevicePolicyMinOverallPower; policy++ {
		for _, policyFirst := range []bool{false, true} {
			opts, err := NewSessionOptions()
			if err != nil {
				t.Fatal(err)
			}
			if policyFirst {
				if err := opts.SetEPSelectionPolicy(policy); err != nil {
					t.Fatal(err)
				}
				if err := opts.AppendExecutionProviderV2([]*EPDevice{cpuEPDevice(t)}, nil); err != nil {
					t.Fatal(err)
				}
			} else {
				if err := opts.SetWeightlessSourceModelBuffer([]byte{1, 2, 3, 4}); err != nil {
					t.Fatal(err)
				}
			}
			p, c, s := epCounts()
			v2, protected := opts.successfulV2, opts.protected
			if policyFirst {
				err = opts.SetWeightlessSourceModelBuffer([]byte{5})
			} else {
				err = opts.SetEPSelectionPolicy(policy)
			}
			requireOrtCode(t, err, ErrorCodeNotImplemented)
			if !strings.Contains(err.Error(), "explicit V2") {
				t.Fatalf("missing selection diagnostic: %v", err)
			}
			if pp, cc, ss := epCounts(); pp != p || cc != c || ss != s {
				t.Fatal("rejection changed counts")
			}
			if opts.successfulV2 != v2 || opts.protected != protected {
				t.Fatal("rejection changed selection/protection")
			}
			if opts.policyAttempted != policyFirst {
				t.Fatal("rejection changed policy flag")
			}
			if !policyFirst && !bytes.Equal(bufferBytes(opts.resources.source), []byte{1, 2, 3, 4}) {
				t.Fatal("rejection changed bytes")
			}
			clone, err := opts.Clone()
			if err != nil {
				t.Fatal(err)
			}
			if policyFirst {
				requireOrtCode(t, clone.SetWeightlessSourceModelBuffer([]byte{5}), ErrorCodeNotImplemented)
			} else {
				requireOrtCode(t, clone.SetEPSelectionPolicy(policy), ErrorCodeNotImplemented)
			}
			clone.Close()
			opts.Close()
		}
	}
	for _, fromBytes := range []bool{false, true} {
		opts, err := NewSessionOptions()
		if err != nil {
			t.Fatal(err)
		}
		if err := opts.SetWeightlessSourceModelBuffer([]byte{1}); err != nil {
			t.Fatal(err)
		}
		session, err := constructBufferSession(opts, fromBytes)
		requireOrtCode(t, err, ErrorCodeInvalidArgument)
		if session != nil {
			session.Close()
			t.Fatal("missing V2 accepted")
		}
		if err := opts.AppendExecutionProviderV2([]*EPDevice{cpuEPDevice(t)}, nil); err != nil {
			t.Fatal(err)
		}
		// Inject conflicting state to exercise each constructor independently of setters.
		opts.policyAttempted = true
		p, c, s := epCounts()
		session, err = constructBufferSession(opts, fromBytes)
		requireOrtCode(t, err, ErrorCodeNotImplemented)
		if session != nil {
			session.Close()
			t.Fatal("policy plus V2 plus buffer accepted")
		}
		if pp, cc, ss := epCounts(); pp != p || cc != c || ss != s {
			t.Fatal("constructor rejection changed counts")
		}
		clone, err := opts.Clone()
		if err != nil {
			t.Fatal(err)
		}
		session, err = constructBufferSession(clone, fromBytes)
		requireOrtCode(t, err, ErrorCodeNotImplemented)
		if session != nil {
			session.Close()
			t.Fatal("conflicting clone accepted")
		}
		clone.Close()
		opts.Close()
	}
}

func TestSessionResourceLifetime(t *testing.T) {
	if APIVersion() < 29 {
		t.Skip("buffer unavailable; guard covered by TestWeightlessSourceModelBuffer")
	}
	for _, fromBytes := range []bool{false, true} {
		opts := bufferOptions(t)
		var reader *epContextReaderRegistration
		if APIVersion() >= 31 {
			if err := opts.SetEPContextDataReader(func(string) ([]byte, error) { return nil, nil }, 1); err != nil {
				t.Fatal(err)
			}
			reader = opts.resources.reader
		}
		source := opts.resources.source
		s, err := constructBufferSession(opts, fromBytes)
		if err != nil {
			t.Fatal(err)
		}
		if reader != nil && resourceRefCount(&reader.ref) != 2 {
			t.Fatal("session did not retain reader")
		}
		if resourceRefCount(&source.ref) != 2 {
			t.Fatalf("session did not retain buffer: %d", resourceRefCount(&source.ref))
		}
		if err := opts.SetWeightlessSourceModelBuffer([]byte{5}); err != nil {
			t.Fatal(err)
		}
		if reader != nil {
			if err := opts.ClearEPContextDataReader(); err != nil {
				t.Fatal(err)
			}
			if resourceRefCount(&reader.ref) != 1 {
				t.Fatal("Clear lost session reader")
			}
		}
		opts.Close()
		if reader != nil && s.resources.reader != reader {
			t.Fatal("session lost exact reader")
		}
		if !bytes.Equal(bufferBytes(s.resources.source), []byte{1, 2, 3, 4}) {
			t.Fatal("session lost snapshot bytes")
		}
		assertBufferSessionAdd(t, s)
		s.Close()
		s.Close()
		if reader != nil {
			assertReaderReleased(t, reader)
		}
		if resourceRefCount(&source.ref) != 0 {
			t.Fatal("session leaked buffer")
		}
	}
	opts := bufferOptions(t)
	var reader *epContextReaderRegistration
	if APIVersion() >= 31 {
		if err := opts.SetEPContextDataReader(func(string) ([]byte, error) { return nil, nil }, 1); err != nil {
			t.Fatal(err)
		}
		reader = opts.resources.reader
	}
	source := opts.resources.source
	p, c, s := epCounts()
	for _, fromBytes := range []bool{false, true} {
		var session *Session
		var err error
		if fromBytes {
			session, err = NewSessionFromBytes([]byte{1}, opts)
		} else {
			session, err = NewSession("/definitely/missing", opts)
		}
		if err == nil || session != nil {
			t.Fatal("invalid model accepted")
		}
		if reader != nil && resourceRefCount(&reader.ref) != 1 {
			t.Fatal("failed creation leaked reader")
		}
		if resourceRefCount(&source.ref) != 1 {
			t.Fatal("failed creation leaked buffer")
		}
		if pp, cc, ss := epCounts(); pp != p || cc != c || ss != s {
			t.Fatal("failed creation leaked counters")
		}
	}
	for _, fromBytes := range []bool{false, true} {
		for _, failInput := range []bool{false, true} {
			session, err := constructBufferSession(opts, fromBytes)
			if err != nil {
				t.Fatal(err)
			}
			snapshot, _, err := reserveSessionCreation(opts)
			if err != nil {
				t.Fatal(err)
			}
			sentinel := errors.New("forced introspection failure")
			result, err := finalizeSessionWithIO(session.handle, func(input bool) ([]IOInfo, error) {
				if input == failInput {
					return nil, sentinel
				}
				return introspectIO(session.handle, input)
			})
			session.handle = nil
			session.Close()
			finishSessionCreation(snapshot, nil)
			if result != nil || !errors.Is(err, sentinel) {
				t.Fatalf("introspection: %v", err)
			}
			if reader != nil && resourceRefCount(&reader.ref) != 1 {
				t.Fatal("introspection leaked reader")
			}
			if resourceRefCount(&source.ref) != 1 {
				t.Fatal("introspection leaked buffer")
			}
			if pp, cc, ss := epCounts(); pp != p || cc != c || ss != s {
				t.Fatal("introspection leaked counters")
			}
		}
	}
}

func resourceRefCount(r *sessionResource) int { r.mu.Lock(); defer r.mu.Unlock(); return r.refs }

func assertBufferSessionAdd(t *testing.T, session *Session) {
	t.Helper()
	a, err := CreateTensor([]int64{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	b, err := CreateTensor([]int64{2, 3}, []float32{10, 20, 30, 40, 50, 60})
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	out, err := session.Run(context.Background(), map[string]*Tensor{"A": a, "B": b}, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		for _, tensor := range out {
			tensor.Close()
		}
	}()
	values, err := TensorData[float32](out["C"])
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(values, []float32{11, 22, 33, 44, 55, 66}) || !reflect.DeepEqual(out["C"].Shape(), []int64{2, 3}) {
		t.Fatalf("buffer session Add: %v shape %v", values, out["C"].Shape())
	}
}

func TestEPContextReaderRegistration(t *testing.T) {
	opts, err := NewSessionOptions()
	if err != nil {
		t.Fatal(err)
	}
	defer opts.Close()
	reader := func(string) ([]byte, error) { return []byte{1, 2, 3}, nil }
	if APIVersion() < 31 {
		requireOrtCode(t, opts.SetEPContextDataReader(nil, 0), ErrorCodeNotImplemented)
		requireOrtCode(t, opts.ClearEPContextDataReader(), ErrorCodeNotImplemented)
		checkEPAdd(t, opts, false)
		return
	}
	for _, limit := range []uint64{0, ^uint64(0)} {
		requireOrtCode(t, opts.SetEPContextDataReader(reader, limit), ErrorCodeInvalidArgument)
	}
	requireOrtCode(t, opts.SetEPContextDataReader(nil, 1), ErrorCodeInvalidArgument)
	if err := opts.SetEPContextDataReader(reader, 3); err != nil {
		t.Fatal(err)
	}
	registration := opts.resources.reader
	requireOrtCode(t, opts.SetEPContextDataReader(nil, 3), ErrorCodeInvalidArgument)
	if opts.resources.reader != registration || resourceRefCount(&registration.ref) != 1 {
		t.Fatal("invalid replacement changed registration")
	}
	clone, err := opts.Clone()
	if err != nil {
		t.Fatal(err)
	}
	defer clone.Close()
	if resourceRefCount(&registration.ref) != 2 {
		t.Fatal("clone did not retain reader")
	}
	if err := opts.SetEPContextDataReader(reader, 4); err != nil {
		t.Fatal(err)
	}
	if resourceRefCount(&registration.ref) != 1 {
		t.Fatal("replacement lost clone reader")
	}
	replacement := opts.resources.reader
	if err := opts.ClearEPContextDataReader(); err != nil {
		t.Fatal(err)
	}
	assertReaderReleased(t, replacement)
	if err := opts.ClearEPContextDataReader(); err != nil {
		t.Fatal(err)
	}
	opts.Close()
	for _, fromBytes := range []bool{false, true} {
		session, err := constructBufferSession(clone, fromBytes)
		if err != nil {
			t.Fatal(err)
		}
		if session.resources.reader != registration || resourceRefCount(&registration.ref) != 2 {
			t.Fatal("session lost reader snapshot")
		}
		session.Close()
		var failed *Session
		if fromBytes {
			failed, err = NewSessionFromBytes([]byte{1}, clone)
		} else {
			failed, err = NewSession("/definitely/missing", clone)
		}
		if err == nil || failed != nil || resourceRefCount(&registration.ref) != 1 {
			t.Fatal("failed construction leaked reader")
		}
	}
	clone.Close()
	assertReaderReleased(t, registration)
	if err := opts.ClearEPContextDataReader(); err == nil {
		t.Fatal("closed options accepted")
	}
	var nilOpts *SessionOptions
	if err := nilOpts.SetEPContextDataReader(reader, 3); err == nil {
		t.Fatal("nil options accepted")
	}
}

func assertReaderReleased(t *testing.T, registration *epContextReaderRegistration) {
	t.Helper()
	if resourceRefCount(&registration.ref) != 0 {
		t.Fatal("reader reference leaked")
	}
	deleted := false
	func() { defer func() { deleted = recover() != nil }(); registration.handle.Value() }()
	if !deleted {
		t.Fatal("cgo handle leaked")
	}
}

type contextReaderFailureCase struct {
	name       string
	reader     EPContextDataReader
	limit      uint64
	code       ErrorCode
	diagnostic string
}

func contextReaderFailureCases(payload []byte) []contextReaderFailureCase {
	return []contextReaderFailureCase{
		{"error", func(string) ([]byte, error) { return nil, errors.New("integration reader error") }, uint64(len(payload)), ErrorCodeFail, "integration reader error"},
		{"panic", func(string) ([]byte, error) { panic("integration panic") }, uint64(len(payload)), ErrorCodeFail, "panicked"},
		{"oversize", func(string) ([]byte, error) { return payload, nil }, uint64(len(payload) - 1), ErrorCodeInvalidArgument, "maxBytes"},
	}
}

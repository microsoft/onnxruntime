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
		source := opts.resources.source
		s, err := constructBufferSession(opts, fromBytes)
		if err != nil {
			t.Fatal(err)
		}
		if resourceRefCount(&source.ref) != 2 {
			t.Fatalf("session did not retain buffer: %d", resourceRefCount(&source.ref))
		}
		if err := opts.SetWeightlessSourceModelBuffer([]byte{5}); err != nil {
			t.Fatal(err)
		}
		opts.Close()
		if !bytes.Equal(bufferBytes(s.resources.source), []byte{1, 2, 3, 4}) {
			t.Fatal("session lost snapshot bytes")
		}
		assertBufferSessionAdd(t, s)
		s.Close()
		s.Close()
		if resourceRefCount(&source.ref) != 0 {
			t.Fatal("session leaked buffer")
		}
	}
	opts := bufferOptions(t)
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

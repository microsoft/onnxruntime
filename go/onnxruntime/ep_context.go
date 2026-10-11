package onnxruntime

/*
#include "cshim.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"sync"
	"unsafe"
)

// sessionResource shares one native resource across options and sessions.
// Retain while an owner is locked; release only after its native handle is released.
type sessionResource struct {
	mu   sync.Mutex
	refs int
	free func()
}

func (r *sessionResource) retain() {
	r.mu.Lock()
	r.refs++
	r.mu.Unlock()
}

func (r *sessionResource) release() {
	r.mu.Lock()
	r.refs--
	last := r.refs == 0
	r.mu.Unlock()
	if last {
		r.free()
	}
}

type sourceModelBuffer struct {
	ref  sessionResource
	data unsafe.Pointer
	size C.size_t
}

// sessionResources is the exact resource snapshot attached to a native owner.
type sessionResources struct{ source *sourceModelBuffer }

func (r sessionResources) retain() {
	if r.source != nil {
		r.source.ref.retain()
	}
}

func (r *sessionResources) release() {
	if r.source != nil {
		r.source.ref.release()
	}
	*r = sessionResources{}
}

func policyBufferError() error {
	return wrapErr("weightless source model buffer", &OrtError{Code: ErrorCodeNotImplemented,
		Msg: "automatic EP selection with a source buffer is unsupported; use explicit V2 selection on fresh options"})
}

func (o *SessionOptions) validateSessionResourcesLocked() error {
	if o.resources.source == nil {
		return nil
	}
	if o.policyAttempted {
		return policyBufferError()
	}
	if !o.successfulV2 {
		return wrapErr("create session", &OrtError{Code: ErrorCodeInvalidArgument, Msg: "source buffer requires successful explicit V2 selection"})
	}
	return nil
}

// SetWeightlessSourceModelBuffer copies serialized source ONNX model bytes from
// model into native storage. Requires API 29. Clones and sessions retain their
// captured buffer until Close; changing or closing these options does not change
// that snapshot.
// Sessions with a buffer require successful AppendExecutionProviderV2 selection.
// Automatic EP selection policies cannot be combined with a source buffer.
// Empty buffers are invalid; the native API does not provide a clear operation.
func (o *SessionOptions) SetWeightlessSourceModelBuffer(model []byte) error {
	mu.Lock()
	err := checkInit()
	if err == nil && apiVersion < 29 {
		err = wrapErr("set weightless source model buffer", &OrtError{Code: ErrorCodeNotImplemented,
			Msg: fmt.Sprintf("SessionOptionsSetWeightlessSourceModelBuffer requires API 29; negotiated API %d (function unavailable)", apiVersion)})
	}
	mu.Unlock()
	if err != nil {
		return err
	}
	if err := o.lockUsable("set weightless source model buffer"); err != nil {
		return err
	}
	defer o.unlock()
	if o.policyAttempted {
		return policyBufferError()
	}
	if len(model) == 0 || uint64(C.size_t(len(model))) != uint64(len(model)) {
		return wrapErr("set weightless source model buffer", &OrtError{Code: ErrorCodeInvalidArgument, Msg: "source buffer must have a nonzero addressable size"})
	}
	var data unsafe.Pointer
	if err := checkStatus(C.ort_copy_buffer(unsafe.Pointer(&model[0]), C.size_t(len(model)), &data)); err != nil {
		return wrapErr("copy weightless source model buffer", err)
	}
	candidate := &sourceModelBuffer{data: data, size: C.size_t(len(model))}
	candidate.ref = sessionResource{refs: 1, free: func() { C.free(data) }}
	if err := checkStatus(C.ort_SessionOptionsSetWeightlessSourceModelBuffer(o.handle, data, candidate.size)); err != nil {
		candidate.ref.release()
		return wrapErr("set weightless source model buffer", err)
	}
	old := o.resources.source
	o.resources.source = candidate
	if old != nil {
		old.ref.release()
	}
	return nil
}

func restoreSourceModelBuffer(o *SessionOptions) error {
	if source := o.resources.source; source != nil {
		return wrapErr("restore weightless source model buffer", checkStatus(C.ort_SessionOptionsSetWeightlessSourceModelBuffer(o.handle, source.data, source.size)))
	}
	return nil
}

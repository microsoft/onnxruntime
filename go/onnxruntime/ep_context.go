package onnxruntime

/*
#include "cshim.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"math"
	"runtime/cgo"
	"strings"
	"sync"
	"unicode/utf8"
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
type sessionResources struct {
	source *sourceModelBuffer
	reader *epContextReaderRegistration
}

func (r sessionResources) retain() {
	if r.reader != nil {
		r.reader.ref.retain()
	}
	if r.source != nil {
		r.source.ref.retain()
	}
}

func (r *sessionResources) release() {
	if r.reader != nil {
		r.reader.ref.release()
	}
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

// EPContextDataReader reads a logical UTF-8 data name, without filesystem fallback.
// The reader must bound its own I/O and allocations and support concurrent calls.
// During construction it may call getters, Clone, setters, Clear or Close on the
// originating options, directly or through another goroutine. Constructors capture
// options before calling the reader; later changes affect later constructions.
// Readers must not construct sessions recursively, call Shutdown, mutate the
// plugin registry or wait for completion of the current construction.
type EPContextDataReader func(name string) ([]byte, error)

type epContextReaderRegistration struct {
	ref      sessionResource
	reader   EPContextDataReader
	maxBytes uint64
	state    unsafe.Pointer
	handle   cgo.Handle
}

func checkReaderAPI() error {
	mu.Lock()
	defer mu.Unlock()
	if err := checkInit(); err != nil {
		return err
	}
	if apiVersion < 31 {
		return wrapErr("EPContext data reader", &OrtError{Code: ErrorCodeNotImplemented,
			Msg: fmt.Sprintf("SessionOptionsSetEpContextDataReadFunc requires API 31; negotiated API %d (function unavailable)", apiVersion)})
	}
	return nil
}

// SetEPContextDataReader registers a reader for external EPContext data. Requires
// API 31. maxBytes bounds each native copy and must fit size_t and int. Options,
// clones and sessions retain their exact registration until the last Close.
// An empty returned slice is a successful empty payload. Reader errors and panics
// fail native loading. See EPContextDataReader for concurrency and re-entry rules.
func (o *SessionOptions) SetEPContextDataReader(reader EPContextDataReader, maxBytes uint64) error {
	if err := checkReaderAPI(); err != nil {
		return err
	}
	if err := o.lockUsable("set EPContext data reader"); err != nil {
		return err
	}
	defer o.unlock()
	if reader == nil || maxBytes == 0 || maxBytes > uint64(math.MaxInt) || uint64(C.size_t(maxBytes)) != maxBytes {
		return wrapErr("set EPContext data reader", &OrtError{Code: ErrorCodeInvalidArgument, Msg: "reader and a nonzero addressable limit are required"})
	}
	candidate := &epContextReaderRegistration{reader: reader, maxBytes: maxBytes}
	candidate.handle = cgo.NewHandle(candidate)
	if err := checkStatus(C.ort_reader_state(C.uintptr_t(candidate.handle), &candidate.state)); err != nil {
		candidate.handle.Delete()
		return wrapErr("allocate EPContext reader state", err)
	}
	candidate.ref = sessionResource{refs: 1, free: func() { candidate.handle.Delete(); C.free(candidate.state) }}
	if err := checkStatus(C.ort_SessionOptionsSetEpContextDataReadFunc(o.handle, C.OrtReadNamedBufferFunc(C.ort_reader_trampoline), candidate.state)); err != nil {
		candidate.ref.release()
		return wrapErr("set EPContext data reader", err)
	}
	old := o.resources.reader
	o.resources.reader = candidate
	if old != nil {
		old.ref.release()
	}
	return nil
}

// ClearEPContextDataReader clears future constructions' reader. Requires API 31.
// Existing clones and sessions retain their captured registration.
func (o *SessionOptions) ClearEPContextDataReader() error {
	if err := checkReaderAPI(); err != nil {
		return err
	}
	if err := o.lockUsable("clear EPContext data reader"); err != nil {
		return err
	}
	defer o.unlock()
	if err := checkStatus(C.ort_SessionOptionsSetEpContextDataReadFunc(o.handle, nil, nil)); err != nil {
		return wrapErr("clear EPContext data reader", err)
	}
	old := o.resources.reader
	o.resources.reader = nil
	if old != nil {
		old.ref.release()
	}
	return nil
}

func readerError(code C.OrtErrorCode, message string) *C.OrtStatus {
	message = strings.ReplaceAll(message, "\x00", "\uFFFD")
	return C.ort_reader_error(code, (*C.char)(unsafe.Pointer(unsafe.StringData(message))), C.size_t(len(message)))
}

//export ortGoReadNamedBuffer
func ortGoReadNamedBuffer(handle C.uintptr_t, name *C.char, allocator *C.OrtAllocator, buffer *unsafe.Pointer, size *C.size_t) (status *C.OrtStatus) {
	defer func() {
		if recover() != nil {
			status = C.ort_reader_panic()
		}
	}()
	if buffer != nil {
		*buffer = nil
	}
	if size != nil {
		*size = 0
	}
	if handle == 0 || name == nil || allocator == nil || buffer == nil || size == nil {
		return readerError(C.ORT_INVALID_ARGUMENT, "invalid reader callback arguments")
	}
	logicalName := C.GoString(name)
	if !utf8.ValidString(logicalName) {
		return readerError(C.ORT_INVALID_ARGUMENT, "reader name is not UTF-8")
	}
	if status := C.ort_reader_validate(allocator); status != nil {
		return status
	}
	registration := cgo.Handle(handle).Value().(*epContextReaderRegistration)
	payload, err := registration.reader(logicalName)
	if err != nil {
		return readerError(C.ORT_FAIL, err.Error())
	}
	if uint64(len(payload)) > registration.maxBytes {
		return readerError(C.ORT_INVALID_ARGUMENT, "reader payload exceeds maxBytes")
	}
	if len(payload) == 0 {
		return nil
	}
	return C.ort_reader_copy(allocator, unsafe.Pointer(&payload[0]), C.size_t(len(payload)), buffer, size)
}

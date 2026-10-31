package onnxruntime

/*
#include "cshim.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"math"
	"sync"
	"unsafe"
)

type RunOptions struct {
	handle             *C.OrtRunOptions
	mu                 sync.RWMutex
	terminateMu        sync.Mutex
	terminateRequested bool
}

func NewRunOptions() (*RunOptions, error) {
	if err := checkInit(); err != nil {
		return nil, err
	}
	var opts *C.OrtRunOptions
	if err := checkStatus(C.ort_CreateRunOptions(&opts)); err != nil {
		return nil, wrapErr("create run options", err)
	}
	return &RunOptions{handle: opts}, nil
}

func (o *RunOptions) SetLogVerbosityLevel(level int) error {
	if level < math.MinInt32 || level > math.MaxInt32 {
		return fmt.Errorf("ort: set run log verbosity level: %d does not fit in C int", level)
	}
	if err := o.lockMutable("set run log verbosity level"); err != nil {
		return err
	}
	defer o.mu.Unlock()
	return wrapErr("set run log verbosity level",
		checkStatus(C.ort_RunOptionsSetRunLogVerbosityLevel(o.handle, C.int(level))))
}

func (o *RunOptions) SetLogSeverityLevel(level int) error {
	if level < math.MinInt32 || level > math.MaxInt32 {
		return fmt.Errorf("ort: set run log severity level: %d does not fit in C int", level)
	}
	if err := o.lockMutable("set run log severity level"); err != nil {
		return err
	}
	defer o.mu.Unlock()
	return wrapErr("set run log severity level",
		checkStatus(C.ort_RunOptionsSetRunLogSeverityLevel(o.handle, C.int(level))))
}

func (o *RunOptions) SetTag(tag string) error {
	if err := rejectNUL(tag, "run tag"); err != nil {
		return err
	}
	if err := o.lockMutable("set run tag"); err != nil {
		return err
	}
	defer o.mu.Unlock()
	cTag := C.CString(tag)
	defer C.free(unsafe.Pointer(cTag))
	return wrapErr("set run tag", checkStatus(C.ort_RunOptionsSetRunTag(o.handle, cTag)))
}

func (o *RunOptions) SetTerminate() error {
	if err := o.lockUsable("set terminate"); err != nil {
		return err
	}
	defer o.mu.RUnlock()
	o.terminateMu.Lock()
	defer o.terminateMu.Unlock()
	if err := checkStatus(C.ort_RunOptionsSetTerminate(o.handle)); err != nil {
		return wrapErr("set terminate", err)
	}
	o.terminateRequested = true
	return nil
}

func (o *RunOptions) UnsetTerminate() error {
	if err := o.lockUsable("unset terminate"); err != nil {
		return err
	}
	defer o.mu.RUnlock()
	o.terminateMu.Lock()
	defer o.terminateMu.Unlock()
	if err := checkStatus(C.ort_RunOptionsUnsetTerminate(o.handle)); err != nil {
		return wrapErr("unset terminate", err)
	}
	o.terminateRequested = false
	return nil
}

func (o *RunOptions) AddConfigEntry(key, value string) error {
	if err := rejectNUL(key, "run config key"); err != nil {
		return err
	}
	if err := rejectNUL(value, "run config value"); err != nil {
		return err
	}
	if err := o.lockMutable("add run config entry"); err != nil {
		return err
	}
	defer o.mu.Unlock()
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))
	cVal := C.CString(value)
	defer C.free(unsafe.Pointer(cVal))
	return wrapErr("add run config entry", checkStatus(C.ort_AddRunConfigEntry(o.handle, cKey, cVal)))
}

func (o *RunOptions) Close() error {
	if o == nil {
		return nil
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.handle != nil {
		C.ort_ReleaseRunOptions(o.handle)
		o.handle = nil
	}
	return nil
}

// restoreTerminate restores the caller-controlled terminate state after the
// context watcher temporarily changed the underlying run options.
func (o *RunOptions) restoreTerminate(handle *C.OrtRunOptions) {
	o.terminateMu.Lock()
	defer o.terminateMu.Unlock()
	if o.terminateRequested {
		_ = checkStatus(C.ort_RunOptionsSetTerminate(handle))
	} else {
		_ = checkStatus(C.ort_RunOptionsUnsetTerminate(handle))
	}
}

// lockUsable holds a read lock that prevents Close from releasing the handle.
func (o *RunOptions) lockUsable(op string) error {
	if o == nil {
		return fmt.Errorf("ort: %s: run options are nil or closed", op)
	}
	o.mu.RLock()
	if o.handle == nil {
		o.mu.RUnlock()
		return fmt.Errorf("ort: %s: run options are nil or closed", op)
	}
	return nil
}

func (o *RunOptions) lockMutable(op string) error {
	if o == nil {
		return fmt.Errorf("ort: %s: run options are nil or closed", op)
	}
	o.mu.Lock()
	if o.handle == nil {
		o.mu.Unlock()
		return fmt.Errorf("ort: %s: run options are nil or closed", op)
	}
	return nil
}

func (o *RunOptions) unlock() {
	o.mu.RUnlock()
}

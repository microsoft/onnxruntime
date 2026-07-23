package onnxruntime

/*
#include "cshim.h"
#include <stdlib.h>
*/
import "C"
import "unsafe"

type RunOptions struct {
	handle *C.OrtRunOptions
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
	return wrapErr("set run log verbosity level",
		checkStatus(C.ort_RunOptionsSetRunLogVerbosityLevel(o.handle, C.int(level))))
}

func (o *RunOptions) SetLogSeverityLevel(level int) error {
	return wrapErr("set run log severity level",
		checkStatus(C.ort_RunOptionsSetRunLogSeverityLevel(o.handle, C.int(level))))
}

func (o *RunOptions) SetTag(tag string) error {
	cTag := C.CString(tag)
	defer C.free(unsafe.Pointer(cTag))
	return wrapErr("set run tag", checkStatus(C.ort_RunOptionsSetRunTag(o.handle, cTag)))
}

func (o *RunOptions) SetTerminate() error {
	return wrapErr("set terminate", checkStatus(C.ort_RunOptionsSetTerminate(o.handle)))
}

func (o *RunOptions) UnsetTerminate() error {
	return wrapErr("unset terminate", checkStatus(C.ort_RunOptionsUnsetTerminate(o.handle)))
}

func (o *RunOptions) AddConfigEntry(key, value string) error {
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))
	cVal := C.CString(value)
	defer C.free(unsafe.Pointer(cVal))
	return wrapErr("add run config entry", checkStatus(C.ort_AddRunConfigEntry(o.handle, cKey, cVal)))
}

func (o *RunOptions) Close() error {
	if o.handle != nil {
		C.ort_ReleaseRunOptions(o.handle)
		o.handle = nil
	}
	return nil
}

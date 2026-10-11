//go:build orttest

package onnxruntime

/*
#include "cshim.h"

static OrtStatus* test_source_buffer(void* fn, int version,
                                    const OrtSessionOptions* opts,
                                    const void** data, size_t* size) {
    const OrtApiBase* base = ((const OrtApiBase* (*)(void))fn)();
    const OrtApi* api = base->GetApi((uint32_t)version);
    if (version < 29) return api->CreateStatus(ORT_NOT_IMPLEMENTED, "source getter requires API 29");
    if (api->GetEpApi == NULL) return api->CreateStatus(ORT_NOT_IMPLEMENTED, "GetEpApi unavailable");
    const OrtEpApi* ep = api->GetEpApi();
    if (ep == NULL || ep->SessionOptionsGetWeightlessSourceModelBuffer == NULL)
        return api->CreateStatus(ORT_NOT_IMPLEMENTED, "source getter unavailable");
    return ep->SessionOptionsGetWeightlessSourceModelBuffer(opts, data, size);
}
*/
import "C"
import (
	"bytes"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"unsafe"
)

func nativeSourceBuffer(o *SessionOptions) (uintptr, []byte, error) {
	if err := o.lockUsable("inspect native source buffer"); err != nil {
		return 0, nil, err
	}
	defer o.unlock()
	// Tests serialize lifecycle changes. The clone hook already holds package mu.
	path, version := resolveLibPath(), apiVersion
	fn, library, err := loadLibrary(path)
	if err != nil {
		return 0, nil, err
	}
	defer closeLibrary(library)
	var data unsafe.Pointer
	var size C.size_t
	if err := checkStatus(C.test_source_buffer(fn, C.int(version), o.handle, &data, &size)); err != nil {
		return 0, nil, err
	}
	if uint64(size) > uint64(math.MaxInt) || (size != 0 && data == nil) {
		return 0, nil, fmt.Errorf("invalid native source buffer")
	}
	return uintptr(data), bytes.Clone(unsafe.Slice((*byte)(data), int(size))), nil
}

func buildBufferObserver(directory string) (string, error) {
	if runtime.GOOS != "linux" {
		return "", fmt.Errorf("observer compiler flags are validated only on Linux")
	}
	compiler, err := exec.Command("go", "env", "CC").Output()
	if err != nil {
		return "", fmt.Errorf("get cgo C compiler: %w", err)
	}
	parts := strings.Fields(strings.TrimSpace(string(compiler)))
	if len(parts) == 0 {
		return "", fmt.Errorf("cgo C compiler unavailable")
	}
	headers, err := filepath.Abs(".")
	if err != nil {
		return "", err
	}
	source := filepath.Join(directory, "buffer_observer.c")
	library := filepath.Join(directory, "libbuffer_observer.so")
	if err := os.WriteFile(source, []byte(bufferObserverSource), 0600); err != nil {
		return "", err
	}
	args := append([]string{}, parts[1:]...)
	args = append(args, "-std=c11", "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror", "-I", headers, source, "-o", library)
	output, err := exec.Command(parts[0], args...).CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("compile buffer observer: %w\n%s", err, output)
	}
	return library, nil
}

const bufferObserverSource = `
#include "onnxruntime_c_api.h"
#include <stdio.h>
#include <stdlib.h>

static const OrtApi* api;
static const OrtEpApi* ep_api;
static const char* ORT_API_CALL name(const OrtEpFactory* self) { (void)self; return "GoBufferObserver"; }
static const char* ORT_API_CALL vendor(const OrtEpFactory* self) { (void)self; return "Go tests"; }
static uint32_t ORT_API_CALL vendor_id(const OrtEpFactory* self) { (void)self; return 0; }
static const char* ORT_API_CALL version(const OrtEpFactory* self) { (void)self; return "1.0.0"; }
static void ORT_API_CALL release_ep(OrtEpFactory* self, OrtEp* ep) { (void)self; (void)ep; }
static OrtStatus* ORT_API_CALL data_transfer(OrtEpFactory* self, OrtDataTransferImpl** out) {
    (void)self;
    *out = NULL;
    return NULL;
}
static OrtStatus* ORT_API_CALL supported(OrtEpFactory* self,
    const OrtHardwareDevice* const* devices, size_t count,
    OrtEpDevice** out, size_t capacity, size_t* added) {
    *added = 0;
    for (size_t i = 0; i < count && *added < capacity; i++) {
        if (api->HardwareDevice_Type(devices[i]) != OrtHardwareDeviceType_CPU) continue;
        OrtStatus* status = ep_api->CreateEpDevice(self, devices[i], NULL, NULL, &out[*added]);
        if (status != NULL) return status;
        ++*added;
        break;
    }
    return NULL;
}
static OrtStatus* ORT_API_CALL create(OrtEpFactory* self,
    const OrtHardwareDevice* const* devices, const OrtKeyValuePairs* const* metadata,
    size_t count, const OrtSessionOptions* options, const OrtLogger* logger, OrtEp** out) {
    (void)self; (void)devices; (void)metadata; (void)count; (void)logger;
    *out = NULL;
    const void* data = NULL;
    size_t size = 0;
    OrtStatus* status = ep_api->SessionOptionsGetWeightlessSourceModelBuffer(options, &data, &size);
    if (status != NULL) return status;
    char message[256];
    int used = snprintf(message, sizeof(message), "GoBufferObserver length=%zu bytes=", size);
    if (size > 64 || (size != 0 && data == NULL)) return api->CreateStatus(ORT_FAIL, "invalid observer buffer");
    for (size_t i = 0; i < size; i++) {
        used += snprintf(message + used, sizeof(message) - (size_t)used, "%02x", ((const unsigned char*)data)[i]);
    }
    return api->CreateStatus(ORT_FAIL, message);
}
__attribute__((visibility("default"))) OrtStatus* CreateEpFactories(
    const char* registered_name, const OrtApiBase* base, const OrtLogger* logger,
    OrtEpFactory** out, size_t capacity, size_t* added) {
    (void)registered_name; (void)logger;
    *added = 0;
    unsigned minor = 0;
    if (sscanf(base->GetVersionString(), "1.%u.", &minor) != 1) minor = ORT_API_VERSION;
    if (minor > ORT_API_VERSION) minor = ORT_API_VERSION;
    for (; minor >= 29; minor--) {
        api = base->GetApi(minor);
        if (api != NULL) break;
    }
    if (api == NULL) return NULL;
    if (api->GetEpApi == NULL) return api->CreateStatus(ORT_NOT_IMPLEMENTED, "observer GetEpApi unavailable");
    ep_api = api->GetEpApi();
    if (ep_api == NULL || ep_api->SessionOptionsGetWeightlessSourceModelBuffer == NULL)
        return api->CreateStatus(ORT_NOT_IMPLEMENTED, "observer getter unavailable");
    if (capacity == 0) return api->CreateStatus(ORT_FAIL, "observer factory capacity is zero");
    OrtEpFactory* factory = calloc(1, sizeof(OrtEpFactory));
    if (factory == NULL) return api->CreateStatus(ORT_FAIL, "observer factory allocation failed");
    factory->ort_version_supported = minor;
    factory->GetName = name;
    factory->GetVendor = vendor;
    factory->GetVendorId = vendor_id;
    factory->GetVersion = version;
    factory->GetSupportedDevices = supported;
    factory->CreateEp = create;
    factory->ReleaseEp = release_ep;
    factory->CreateDataTransfer = data_transfer;
    out[0] = factory;
    *added = 1;
    return NULL;
}
__attribute__((visibility("default"))) OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
    free(factory);
    return NULL;
}
`

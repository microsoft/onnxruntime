//go:build orttest

package onnxruntime

/*
#include "cshim.h"
#include <stdlib.h>
#include <string.h>
#if defined(__linux__)
#include <pthread.h>
#endif

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

typedef struct {
    OrtAllocator allocator;
    const OrtMemoryInfo* info;
    int allocations;
    int frees;
    int fail;
} TestReaderAllocator;
static void* ORT_API_CALL test_alloc(OrtAllocator* a, size_t size) {
    TestReaderAllocator* test = (TestReaderAllocator*)a;
    test->allocations++;
    return test->fail ? NULL : malloc(size);
}
static void ORT_API_CALL test_free(OrtAllocator* a, void* data) {
    ((TestReaderAllocator*)a)->frees++;
    free(data);
}
static const OrtMemoryInfo* ORT_API_CALL test_info(const OrtAllocator* a) {
    return ((const TestReaderAllocator*)a)->info;
}
typedef struct {
    OrtReadNamedBufferFunc reader;
    void* state;
    const char* name;
    OrtAllocator* allocator;
    void** data;
    size_t* size;
    OrtStatus* status;
} TestReaderCall;
#if defined(__linux__)
static void* test_reader_thread(void* value) {
    TestReaderCall* call = value;
    call->status = call->reader(call->state, call->name, call->allocator, call->data, call->size);
    return NULL;
}
#endif
static OrtStatus* test_reader_call(void* fn, int version, const OrtSessionOptions* opts,
    const char* name, int mode, int invalid, int threaded, void** data, size_t* size,
    int* allocations, int* frees, int* clean_failure) {
    const OrtApiBase* base = ((const OrtApiBase* (*)(void))fn)();
    const OrtApi* api = base->GetApi((uint32_t)version);
    if (version < 31) return api->CreateStatus(ORT_NOT_IMPLEMENTED, "reader bridge requires API 31");
    if (api->GetEpApi == NULL) return api->CreateStatus(ORT_NOT_IMPLEMENTED, "reader GetEpApi unavailable");
    const OrtEpApi* ep = api->GetEpApi();
    if (ep == NULL || ep->SessionOptionsGetEpContextConfig == NULL ||
        ep->EpContextConfigGetEpContextDataReadFunc == NULL || ep->ReleaseEpContextConfig == NULL)
        return api->CreateStatus(ORT_NOT_IMPLEMENTED, "reader config unavailable");
    OrtEpContextConfig* config = NULL;
    OrtStatus* status = ep->SessionOptionsGetEpContextConfig(opts, &config);
    if (status != NULL) return status;
    OrtReadNamedBufferFunc reader = NULL;
    void* state = NULL;
    status = ep->EpContextConfigGetEpContextDataReadFunc(config, &reader, &state);
    ep->ReleaseEpContextConfig(config);
    if (status != NULL) return status;
    if (reader == NULL) return api->CreateStatus(ORT_FAIL, "reader not registered");
    OrtAllocator* allocator = NULL;
    OrtMemoryInfo* info = NULL;
    TestReaderAllocator test = {0};
    if (mode == 0) {
        status = api->GetAllocatorWithDefaultOptions(&allocator);
    } else {
        if (mode == 3 || mode == 4) {
            status = api->CreateMemoryInfo_V2("test", OrtMemoryInfoDeviceType_GPU, 0, 0,
                mode == 4 ? OrtDeviceMemoryType_HOST_ACCESSIBLE : OrtDeviceMemoryType_DEFAULT,
                0, OrtDeviceAllocator, &info);
        } else {
            status = api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &info);
        }
        test.allocator.version = ORT_API_VERSION;
        test.allocator.Alloc = test_alloc;
        test.allocator.Free = test_free;
        test.allocator.Info = test_info;
        test.info = info;
        test.fail = mode == 2;
        allocator = &test.allocator;
        if (mode == 5) test.allocator.Alloc = NULL;
        if (mode == 6) test.info = NULL;
    }
    if (status != NULL) return status;
    *data = (void*)1;
    *size = 123;
    TestReaderCall call = {reader, invalid == 1 ? NULL : state, invalid == 2 ? NULL : name,
        invalid == 3 ? NULL : allocator, invalid == 4 ? NULL : data,
        invalid == 5 ? NULL : size, NULL};
#if defined(__linux__)
    if (threaded) {
        pthread_t thread;
        if (pthread_create(&thread, NULL, test_reader_thread, &call) != 0) {
            if (info != NULL) api->ReleaseMemoryInfo(info);
            return api->CreateStatus(ORT_FAIL, "reader thread creation failed");
        }
        pthread_join(thread, NULL);
        status = call.status;
    } else
#else
    (void)threaded;
#endif
    {
        status = reader(call.state, call.name, call.allocator, call.data, call.size);
    }
    *clean_failure = (invalid == 4 || *data == NULL) && (invalid == 5 || *size == 0);
    if (status == NULL && *size != 0) {
        void* copy = malloc(*size);
        if (copy == NULL) {
            allocator->Free(allocator, *data);
            *data = NULL; *size = 0;
            status = api->CreateStatus(ORT_FAIL, "test copy allocation failed");
        } else {
            memcpy(copy, *data, *size);
            allocator->Free(allocator, *data);
            *data = copy;
        }
    }
    *allocations = test.allocations;
    *frees = test.frees;
    if (info != NULL) api->ReleaseMemoryInfo(info);
    return status;
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

func buildBufferObserver(directory string, compiler []string) (string, error) {
	if runtime.GOOS != "linux" {
		return "", fmt.Errorf("observer compiler flags are validated only on Linux")
	}
	if len(compiler) == 0 {
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
	args := append([]string{}, compiler[1:]...)
	args = append(args, "-std=c11", "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror", "-I", headers, source, "-o", library)
	output, err := exec.Command(compiler[0], args...).CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("compile buffer observer: %w\n%s", err, output)
	}
	return library, nil
}

const bufferObserverSource = `
#include "onnxruntime_c_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const OrtApi* api;
static const OrtEpApi* ep_api;
static unsigned api_version;
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
    unsigned char before[64];
    if (size > sizeof(before) || (size != 0 && data == NULL))
        return api->CreateStatus(ORT_FAIL, "invalid observer buffer");
    if (size != 0) memcpy(before, data, size);
    int called = 0;
    if (api_version >= 31) {
        if (ep_api->SessionOptionsGetEpContextConfig == NULL ||
            ep_api->EpContextConfigGetEpContextDataReadFunc == NULL ||
            ep_api->ReleaseEpContextConfig == NULL)
            return api->CreateStatus(ORT_NOT_IMPLEMENTED, "observer callback slots unavailable");
        OrtEpContextConfig* config = NULL;
        status = ep_api->SessionOptionsGetEpContextConfig(options, &config);
        if (status != NULL) return status;
        OrtReadNamedBufferFunc reader = NULL;
        void* state = NULL;
        status = ep_api->EpContextConfigGetEpContextDataReadFunc(config, &reader, &state);
        if (status != NULL) { ep_api->ReleaseEpContextConfig(config); return status; }
        if (reader != NULL) {
            OrtAllocator* allocator = NULL;
            status = api->GetAllocatorWithDefaultOptions(&allocator);
            if (status == NULL) {
                void* payload = NULL;
                size_t payload_size = 0;
                status = reader(state, "observer/上下文", allocator, &payload, &payload_size);
                if (status == NULL && (payload_size != 3 || payload == NULL || memcmp(payload, "abc", 3) != 0))
                    status = api->CreateStatus(ORT_FAIL, "observer callback payload mismatch");
                if (payload != NULL) allocator->Free(allocator, payload);
                called = 1;
            }
        }
        ep_api->ReleaseEpContextConfig(config);
        if (status != NULL) return status;
    }
    const void* after = NULL;
    size_t after_size = 0;
    status = ep_api->SessionOptionsGetWeightlessSourceModelBuffer(options, &after, &after_size);
    if (status != NULL) return status;
    if (after != data || after_size != size || (size != 0 && memcmp(after, before, size) != 0))
        return api->CreateStatus(ORT_FAIL, "observer source changed during callback");
    char message[256];
    int used = snprintf(message, sizeof(message), "GoBufferObserver length=%zu bytes=", size);
    if (size > 64 || (size != 0 && data == NULL)) return api->CreateStatus(ORT_FAIL, "invalid observer buffer");
    for (size_t i = 0; i < size; i++) {
        used += snprintf(message + used, sizeof(message) - (size_t)used, "%02x", ((const unsigned char*)data)[i]);
    }
    if (called) snprintf(message + used, sizeof(message) - (size_t)used, " callback=abc source=unchanged");
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
    api_version = minor;
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

type readerBridgeResult struct {
	data               []byte
	allocations, frees int
	cleanFailure       bool
}

// Callers retain options and serialize mutations for the duration of this call.
func invokeNativeReader(o *SessionOptions, name string, mode, invalid int, threaded bool) (readerBridgeResult, error) {
	fn, library, err := loadLibrary(resolveLibPath())
	if err != nil {
		return readerBridgeResult{}, err
	}
	defer closeLibrary(library)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	var data unsafe.Pointer
	var size C.size_t
	var allocations, frees, clean C.int
	thread := C.int(0)
	if threaded {
		thread = 1
	}
	status := C.test_reader_call(fn, C.int(APIVersion()), o.handle, cName, C.int(mode), C.int(invalid), thread, &data, &size, &allocations, &frees, &clean)
	result := readerBridgeResult{allocations: int(allocations), frees: int(frees), cleanFailure: clean != 0}
	if err := checkStatus(status); err != nil {
		return result, err
	}
	if size != 0 {
		result.data = bytes.Clone(unsafe.Slice((*byte)(data), int(size)))
		C.free(data)
	}
	return result, nil
}

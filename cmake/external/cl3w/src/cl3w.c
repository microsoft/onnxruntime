/*
 * This file was generated with cl3w_gen.py, part of cl3w
 * (hosted at https://github.com/cloudhan/cl3w)
*/
#define CL3W_NO_CL_API_DEFINES
#include "cl3w.h"

#include <stdio.h>

#define CL3W_RET_IF_OK(expr) if ((expr) == CL3W_OK) return CL3W_OK
#define CL3W_RET_IF_ERROR(expr) do {       \
    CL3W_STATUS status = (expr);           \
    if (status != CL3W_OK) return status;  \
} while(0)

static const char* get_probe_api_name(void);
static CL3WclAPI get_api(const char *api_name);
static CL3W_STATUS load_libcl(void);
static void load_apis(void);
static void unload_apis(void);

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif
#include <windows.h>

static const char* default_lib_paths[] = {
  "OpenCL.dll"
};

static HMODULE libcl;

static CL3W_STATUS open_lib(const char* path, HMODULE* out_lib) {
    HMODULE lib = LoadLibraryA(path);
    if (!lib) {
        return CL3W_ERROR_LIBRARY_OPEN;
    }

    if (!GetProcAddress(lib, get_probe_api_name())) {
        FreeLibrary(lib);
        return CL3W_ERROR_LIBRARY_OPEN;
    }

    *out_lib = lib;
    return CL3W_OK;
}


static CL3W_STATUS open_libcl(const char** libpaths, size_t npaths) {
    if (libpaths && npaths) {
        int i;
        for (i = 0; i < npaths; i++) {
            CL3W_RET_IF_OK(open_lib(libpaths[i], &libcl));
        }
    }
    libcl = (PVOID)0;
    return CL3W_ERROR_LIBRARY_OPEN;
}

static void close_libcl(void) {
    FreeLibrary(libcl);
}

static CL3WclAPI get_api(const char *api_name) {
    CL3WclAPI res;
    res = (CL3WclAPI)GetProcAddress(libcl, api_name);
    return res;
}
#else // POSIX
#include <dlfcn.h>

#if defined(__ANDROID__)
static const char *default_lib_paths[] = {
  "/system/lib64/libOpenCL.so",
  "/system/vendor/lib64/libOpenCL.so",
  "/system/vendor/lib64/egl/libGLES_mali.so",
  "/system/vendor/lib64/libPVROCL.so",
  "/data/data/org.pocl.libs/files/lib64/libpocl.so",
  "/system/lib/libOpenCL.so",
  "/system/vendor/lib/libOpenCL.so",
  "/system/vendor/lib/egl/libGLES_mali.so",
  "/system/vendor/lib/libPVROCL.so",
  "/data/data/org.pocl.libs/files/lib/libpocl.so",
  "libOpenCL.so"
};
#elif defined(__linux__)
static const char *default_lib_paths[] = {
  "/usr/lib/libOpenCL.so",
  "/usr/local/lib/libOpenCL.so",
  "/usr/local/lib/libpocl.so",
  "/usr/lib64/libOpenCL.so",
  "/usr/lib32/libOpenCL.so",
  "libOpenCL.so"
};
#elif defined(__APPLE__) || defined(__MACOSX)
static const char *default_lib_paths[] = {
  "/System/Library/Frameworks/OpenCL.framework/OpenCL"
  "libOpenCL.so",
};
#endif

static void* libcl;

static CL3W_STATUS open_lib(const char* path, void** out_lib) {
    void* lib = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
    if (!lib) {
        return CL3W_ERROR_LIBRARY_OPEN;
    }

    if (!dlsym(lib, get_probe_api_name())) {
        dlclose(lib);
        return CL3W_ERROR_LIBRARY_OPEN;
    }

    *out_lib = lib;
    return CL3W_OK;
}

static CL3W_STATUS open_libcl(const char** libpaths, size_t npaths) {
    if (libpaths && npaths) {
        int i;
        for (i = 0; i < npaths; i++) {
            CL3W_RET_IF_OK(open_lib(libpaths[i], &libcl));
        }
    }
    libcl = NULL;
    return CL3W_ERROR_LIBRARY_OPEN;
}

static void close_libcl(void)
{
    if (libcl) {
        dlclose(libcl);
        libcl = NULL;
    }
}

static CL3WclAPI get_api(const char *api_name) {
    CL3WclAPI res;
    res = (CL3WclAPI)dlsym(libcl, api_name);
    return res;
}
#endif

static const char* cl3w_api_names[] = {
    "clGetPlatformIDs",
    "clGetPlatformInfo",
    "clGetDeviceIDs",
    "clGetDeviceInfo",
    "clCreateSubDevices",
    "clRetainDevice",
    "clReleaseDevice",
    "clCreateContext",
    "clCreateContextFromType",
    "clRetainContext",
    "clReleaseContext",
    "clGetContextInfo",
    "clRetainCommandQueue",
    "clReleaseCommandQueue",
    "clGetCommandQueueInfo",
    "clCreateBuffer",
    "clCreateSubBuffer",
    "clCreateImage",
    "clRetainMemObject",
    "clReleaseMemObject",
    "clGetSupportedImageFormats",
    "clGetMemObjectInfo",
    "clGetImageInfo",
    "clSetMemObjectDestructorCallback",
    "clRetainSampler",
    "clReleaseSampler",
    "clGetSamplerInfo",
    "clCreateProgramWithSource",
    "clCreateProgramWithBinary",
    "clCreateProgramWithBuiltInKernels",
    "clRetainProgram",
    "clReleaseProgram",
    "clBuildProgram",
    "clCompileProgram",
    "clLinkProgram",
    "clUnloadPlatformCompiler",
    "clGetProgramInfo",
    "clGetProgramBuildInfo",
    "clCreateKernel",
    "clCreateKernelsInProgram",
    "clRetainKernel",
    "clReleaseKernel",
    "clSetKernelArg",
    "clGetKernelInfo",
    "clGetKernelArgInfo",
    "clGetKernelWorkGroupInfo",
    "clWaitForEvents",
    "clGetEventInfo",
    "clCreateUserEvent",
    "clRetainEvent",
    "clReleaseEvent",
    "clSetUserEventStatus",
    "clSetEventCallback",
    "clGetEventProfilingInfo",
    "clFlush",
    "clFinish",
    "clEnqueueReadBuffer",
    "clEnqueueReadBufferRect",
    "clEnqueueWriteBuffer",
    "clEnqueueWriteBufferRect",
    "clEnqueueFillBuffer",
    "clEnqueueCopyBuffer",
    "clEnqueueCopyBufferRect",
    "clEnqueueReadImage",
    "clEnqueueWriteImage",
    "clEnqueueFillImage",
    "clEnqueueCopyImage",
    "clEnqueueCopyImageToBuffer",
    "clEnqueueCopyBufferToImage",
    "clEnqueueMapBuffer",
    "clEnqueueMapImage",
    "clEnqueueUnmapMemObject",
    "clEnqueueMigrateMemObjects",
    "clEnqueueNDRangeKernel",
    "clEnqueueNativeKernel",
    "clEnqueueMarkerWithWaitList",
    "clEnqueueBarrierWithWaitList",
    "clGetExtensionFunctionAddressForPlatform",
    "clSetCommandQueueProperty",
    "clCreateImage2D",
    "clCreateImage3D",
    "clEnqueueMarker",
    "clEnqueueWaitForEvents",
    "clEnqueueBarrier",
    "clUnloadCompiler",
    "clGetExtensionFunctionAddress",
    "clCreateCommandQueue",
    "clCreateSampler",
    "clEnqueueTask",
};

char cl3w_msg_prefix[] = "[cl3w] OpenCL API";
char cl3w_msg_suffix[] = "is not loaded/supported";

cl_int clGetPlatformIDsDummyImpl(cl_uint num_entries, cl_platform_id* platforms, cl_uint* num_platforms) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[0], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clGetPlatformInfoDummyImpl(cl_platform_id platform, cl_platform_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[1], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clGetDeviceIDsDummyImpl(cl_platform_id platform, cl_device_type device_type, cl_uint num_entries, cl_device_id* devices, cl_uint* num_devices) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[2], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clGetDeviceInfoDummyImpl(cl_device_id device, cl_device_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[3], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clCreateSubDevicesDummyImpl(cl_device_id in_device, const cl_device_partition_property* properties, cl_uint num_devices, cl_device_id* out_devices, cl_uint* num_devices_ret) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[4], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clRetainDeviceDummyImpl(cl_device_id device) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[5], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clReleaseDeviceDummyImpl(cl_device_id device) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[6], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_context clCreateContextDummyImpl(const cl_context_properties* properties, cl_uint num_devices, const cl_device_id* devices, void (CL_CALLBACK* pfn_notify)(const char* errinfo, const void* private_info, size_t cb, void* user_data), void* user_data, cl_int* errcode_ret) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[7], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    *errcode_ret = CL_INVALID_HOST_PTR;
    return NULL;
}

cl_context clCreateContextFromTypeDummyImpl(const cl_context_properties* properties, cl_device_type device_type, void (CL_CALLBACK* pfn_notify)(const char* errinfo, const void* private_info, size_t cb, void* user_data), void* user_data, cl_int* errcode_ret) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[8], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    *errcode_ret = CL_INVALID_HOST_PTR;
    return NULL;
}

cl_int clRetainContextDummyImpl(cl_context context) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[9], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clReleaseContextDummyImpl(cl_context context) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[10], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clGetContextInfoDummyImpl(cl_context context, cl_context_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[11], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clRetainCommandQueueDummyImpl(cl_command_queue command_queue) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[12], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clReleaseCommandQueueDummyImpl(cl_command_queue command_queue) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[13], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clGetCommandQueueInfoDummyImpl(cl_command_queue command_queue, cl_command_queue_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[14], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_mem clCreateBufferDummyImpl(cl_context context, cl_mem_flags flags, size_t size, void* host_ptr, cl_int* errcode_ret) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[15], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    *errcode_ret = CL_INVALID_HOST_PTR;
    return NULL;
}

cl_mem clCreateSubBufferDummyImpl(cl_mem buffer, cl_mem_flags flags, cl_buffer_create_type buffer_create_type, const void* buffer_create_info, cl_int* errcode_ret) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[16], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    *errcode_ret = CL_INVALID_HOST_PTR;
    return NULL;
}

cl_mem clCreateImageDummyImpl(cl_context context, cl_mem_flags flags, const cl_image_format* image_format, const cl_image_desc* image_desc, void* host_ptr, cl_int* errcode_ret) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[17], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    *errcode_ret = CL_INVALID_HOST_PTR;
    return NULL;
}

cl_int clRetainMemObjectDummyImpl(cl_mem memobj) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[18], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clReleaseMemObjectDummyImpl(cl_mem memobj) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[19], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clGetSupportedImageFormatsDummyImpl(cl_context context, cl_mem_flags flags, cl_mem_object_type image_type, cl_uint num_entries, cl_image_format* image_formats, cl_uint* num_image_formats) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[20], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clGetMemObjectInfoDummyImpl(cl_mem memobj, cl_mem_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[21], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clGetImageInfoDummyImpl(cl_mem image, cl_image_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[22], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clSetMemObjectDestructorCallbackDummyImpl(cl_mem memobj, void (CL_CALLBACK* pfn_notify)(cl_mem memobj, void* user_data), void* user_data) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[23], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clRetainSamplerDummyImpl(cl_sampler sampler) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[24], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clReleaseSamplerDummyImpl(cl_sampler sampler) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[25], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clGetSamplerInfoDummyImpl(cl_sampler sampler, cl_sampler_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[26], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_program clCreateProgramWithSourceDummyImpl(cl_context context, cl_uint count, const char ** strings, const size_t* lengths, cl_int* errcode_ret) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[27], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    *errcode_ret = CL_INVALID_HOST_PTR;
    return NULL;
}

cl_program clCreateProgramWithBinaryDummyImpl(cl_context context, cl_uint num_devices, const cl_device_id* device_list, const size_t* lengths, const unsigned char ** binaries, cl_int* binary_status, cl_int* errcode_ret) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[28], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    *errcode_ret = CL_INVALID_HOST_PTR;
    return NULL;
}

cl_program clCreateProgramWithBuiltInKernelsDummyImpl(cl_context context, cl_uint num_devices, const cl_device_id* device_list, const char* kernel_names, cl_int* errcode_ret) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[29], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    *errcode_ret = CL_INVALID_HOST_PTR;
    return NULL;
}

cl_int clRetainProgramDummyImpl(cl_program program) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[30], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clReleaseProgramDummyImpl(cl_program program) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[31], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clBuildProgramDummyImpl(cl_program program, cl_uint num_devices, const cl_device_id* device_list, const char* options, void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data), void* user_data) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[32], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clCompileProgramDummyImpl(cl_program program, cl_uint num_devices, const cl_device_id* device_list, const char* options, cl_uint num_input_headers, const cl_program* input_headers, const char ** header_include_names, void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data), void* user_data) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[33], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_program clLinkProgramDummyImpl(cl_context context, cl_uint num_devices, const cl_device_id* device_list, const char* options, cl_uint num_input_programs, const cl_program* input_programs, void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data), void* user_data, cl_int* errcode_ret) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[34], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    *errcode_ret = CL_INVALID_HOST_PTR;
    return NULL;
}

cl_int clUnloadPlatformCompilerDummyImpl(cl_platform_id platform) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[35], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clGetProgramInfoDummyImpl(cl_program program, cl_program_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[36], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clGetProgramBuildInfoDummyImpl(cl_program program, cl_device_id device, cl_program_build_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[37], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_kernel clCreateKernelDummyImpl(cl_program program, const char* kernel_name, cl_int* errcode_ret) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[38], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    *errcode_ret = CL_INVALID_HOST_PTR;
    return NULL;
}

cl_int clCreateKernelsInProgramDummyImpl(cl_program program, cl_uint num_kernels, cl_kernel* kernels, cl_uint* num_kernels_ret) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[39], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clRetainKernelDummyImpl(cl_kernel kernel) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[40], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clReleaseKernelDummyImpl(cl_kernel kernel) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[41], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clSetKernelArgDummyImpl(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void* arg_value) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[42], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clGetKernelInfoDummyImpl(cl_kernel kernel, cl_kernel_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[43], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clGetKernelArgInfoDummyImpl(cl_kernel kernel, cl_uint arg_index, cl_kernel_arg_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[44], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clGetKernelWorkGroupInfoDummyImpl(cl_kernel kernel, cl_device_id device, cl_kernel_work_group_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[45], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clWaitForEventsDummyImpl(cl_uint num_events, const cl_event* event_list) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[46], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clGetEventInfoDummyImpl(cl_event event, cl_event_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[47], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_event clCreateUserEventDummyImpl(cl_context context, cl_int* errcode_ret) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[48], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    *errcode_ret = CL_INVALID_HOST_PTR;
    return NULL;
}

cl_int clRetainEventDummyImpl(cl_event event) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[49], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clReleaseEventDummyImpl(cl_event event) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[50], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clSetUserEventStatusDummyImpl(cl_event event, cl_int execution_status) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[51], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clSetEventCallbackDummyImpl(cl_event event, cl_int command_exec_callback_type, void (CL_CALLBACK* pfn_notify)(cl_event event, cl_int event_command_status, void *user_data), void* user_data) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[52], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clGetEventProfilingInfoDummyImpl(cl_event event, cl_profiling_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[53], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clFlushDummyImpl(cl_command_queue command_queue) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[54], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clFinishDummyImpl(cl_command_queue command_queue) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[55], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clEnqueueReadBufferDummyImpl(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read, size_t offset, size_t size, void* ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[56], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clEnqueueReadBufferRectDummyImpl(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read, const size_t* buffer_origin, const size_t* host_origin, const size_t* region, size_t buffer_row_pitch, size_t buffer_slice_pitch, size_t host_row_pitch, size_t host_slice_pitch, void* ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[57], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clEnqueueWriteBufferDummyImpl(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write, size_t offset, size_t size, const void* ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[58], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clEnqueueWriteBufferRectDummyImpl(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write, const size_t* buffer_origin, const size_t* host_origin, const size_t* region, size_t buffer_row_pitch, size_t buffer_slice_pitch, size_t host_row_pitch, size_t host_slice_pitch, const void* ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[59], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clEnqueueFillBufferDummyImpl(cl_command_queue command_queue, cl_mem buffer, const void* pattern, size_t pattern_size, size_t offset, size_t size, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[60], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clEnqueueCopyBufferDummyImpl(cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_buffer, size_t src_offset, size_t dst_offset, size_t size, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[61], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clEnqueueCopyBufferRectDummyImpl(cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_buffer, const size_t* src_origin, const size_t* dst_origin, const size_t* region, size_t src_row_pitch, size_t src_slice_pitch, size_t dst_row_pitch, size_t dst_slice_pitch, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[62], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clEnqueueReadImageDummyImpl(cl_command_queue command_queue, cl_mem image, cl_bool blocking_read, const size_t* origin, const size_t* region, size_t row_pitch, size_t slice_pitch, void* ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[63], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clEnqueueWriteImageDummyImpl(cl_command_queue command_queue, cl_mem image, cl_bool blocking_write, const size_t* origin, const size_t* region, size_t input_row_pitch, size_t input_slice_pitch, const void* ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[64], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clEnqueueFillImageDummyImpl(cl_command_queue command_queue, cl_mem image, const void* fill_color, const size_t* origin, const size_t* region, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[65], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clEnqueueCopyImageDummyImpl(cl_command_queue command_queue, cl_mem src_image, cl_mem dst_image, const size_t* src_origin, const size_t* dst_origin, const size_t* region, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[66], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clEnqueueCopyImageToBufferDummyImpl(cl_command_queue command_queue, cl_mem src_image, cl_mem dst_buffer, const size_t* src_origin, const size_t* region, size_t dst_offset, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[67], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clEnqueueCopyBufferToImageDummyImpl(cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_image, size_t src_offset, const size_t* dst_origin, const size_t* region, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[68], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

void* clEnqueueMapBufferDummyImpl(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_map, cl_map_flags map_flags, size_t offset, size_t size, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event, cl_int* errcode_ret) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[69], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    *errcode_ret = CL_INVALID_HOST_PTR;
    return NULL;
}

void* clEnqueueMapImageDummyImpl(cl_command_queue command_queue, cl_mem image, cl_bool blocking_map, cl_map_flags map_flags, const size_t* origin, const size_t* region, size_t* image_row_pitch, size_t* image_slice_pitch, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event, cl_int* errcode_ret) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[70], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    *errcode_ret = CL_INVALID_HOST_PTR;
    return NULL;
}

cl_int clEnqueueUnmapMemObjectDummyImpl(cl_command_queue command_queue, cl_mem memobj, void* mapped_ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[71], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clEnqueueMigrateMemObjectsDummyImpl(cl_command_queue command_queue, cl_uint num_mem_objects, const cl_mem* mem_objects, cl_mem_migration_flags flags, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[72], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clEnqueueNDRangeKernelDummyImpl(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim, const size_t* global_work_offset, const size_t* global_work_size, const size_t* local_work_size, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[73], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clEnqueueNativeKernelDummyImpl(cl_command_queue command_queue, void (CL_CALLBACK* user_func)(void*), void* args, size_t cb_args, cl_uint num_mem_objects, const cl_mem* mem_list, const void ** args_mem_loc, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[74], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clEnqueueMarkerWithWaitListDummyImpl(cl_command_queue command_queue, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[75], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clEnqueueBarrierWithWaitListDummyImpl(cl_command_queue command_queue, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[76], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

void* clGetExtensionFunctionAddressForPlatformDummyImpl(cl_platform_id platform, const char* func_name) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[77], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    /* no error handling here*/
    return NULL;
}

cl_int clSetCommandQueuePropertyDummyImpl(cl_command_queue command_queue, cl_command_queue_properties properties, cl_bool enable, cl_command_queue_properties* old_properties) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[78], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_mem clCreateImage2DDummyImpl(cl_context context, cl_mem_flags flags, const cl_image_format* image_format, size_t image_width, size_t image_height, size_t image_row_pitch, void* host_ptr, cl_int* errcode_ret) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[79], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    *errcode_ret = CL_INVALID_HOST_PTR;
    return NULL;
}

cl_mem clCreateImage3DDummyImpl(cl_context context, cl_mem_flags flags, const cl_image_format* image_format, size_t image_width, size_t image_height, size_t image_depth, size_t image_row_pitch, size_t image_slice_pitch, void* host_ptr, cl_int* errcode_ret) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[80], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    *errcode_ret = CL_INVALID_HOST_PTR;
    return NULL;
}

cl_int clEnqueueMarkerDummyImpl(cl_command_queue command_queue, cl_event* event) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[81], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clEnqueueWaitForEventsDummyImpl(cl_command_queue command_queue, cl_uint num_events, const cl_event* event_list) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[82], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clEnqueueBarrierDummyImpl(cl_command_queue command_queue) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[83], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

cl_int clUnloadCompilerDummyImpl() {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[84], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

void* clGetExtensionFunctionAddressDummyImpl(const char* func_name) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[85], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    /* no error handling here*/
    return NULL;
}

cl_command_queue clCreateCommandQueueDummyImpl(cl_context context, cl_device_id device, cl_command_queue_properties properties, cl_int* errcode_ret) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[86], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    *errcode_ret = CL_INVALID_HOST_PTR;
    return NULL;
}

cl_sampler clCreateSamplerDummyImpl(cl_context context, cl_bool normalized_coords, cl_addressing_mode addressing_mode, cl_filter_mode filter_mode, cl_int* errcode_ret) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[87], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    *errcode_ret = CL_INVALID_HOST_PTR;
    return NULL;
}

cl_int clEnqueueTaskDummyImpl(cl_command_queue command_queue, cl_kernel kernel, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    fprintf(stderr, "%s %s %s\n", cl3w_msg_prefix, cl3w_api_names[88], cl3w_msg_suffix);
    /* We reuse CL_INVALID_HOST_PTR as an indicator of unloaded function, aka, function pointer is invalid. */
    return CL_INVALID_HOST_PTR;
}

CL3W_API union CL3WAPIs cl3w_apis;

cl_int clGetPlatformIDs(cl_uint num_entries, cl_platform_id* platforms, cl_uint* num_platforms) {
    return cl3w_apis.cl.clGetPlatformIDs(num_entries, platforms, num_platforms);
}

cl_int clGetPlatformInfo(cl_platform_id platform, cl_platform_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    return cl3w_apis.cl.clGetPlatformInfo(platform, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_int clGetDeviceIDs(cl_platform_id platform, cl_device_type device_type, cl_uint num_entries, cl_device_id* devices, cl_uint* num_devices) {
    return cl3w_apis.cl.clGetDeviceIDs(platform, device_type, num_entries, devices, num_devices);
}

cl_int clGetDeviceInfo(cl_device_id device, cl_device_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    return cl3w_apis.cl.clGetDeviceInfo(device, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_int clCreateSubDevices(cl_device_id in_device, const cl_device_partition_property* properties, cl_uint num_devices, cl_device_id* out_devices, cl_uint* num_devices_ret) {
    return cl3w_apis.cl.clCreateSubDevices(in_device, properties, num_devices, out_devices, num_devices_ret);
}

cl_int clRetainDevice(cl_device_id device) {
    return cl3w_apis.cl.clRetainDevice(device);
}

cl_int clReleaseDevice(cl_device_id device) {
    return cl3w_apis.cl.clReleaseDevice(device);
}

cl_context clCreateContext(const cl_context_properties* properties, cl_uint num_devices, const cl_device_id* devices, void (CL_CALLBACK* pfn_notify)(const char* errinfo, const void* private_info, size_t cb, void* user_data), void* user_data, cl_int* errcode_ret) {
    return cl3w_apis.cl.clCreateContext(properties, num_devices, devices, pfn_notify, user_data, errcode_ret);
}

cl_context clCreateContextFromType(const cl_context_properties* properties, cl_device_type device_type, void (CL_CALLBACK* pfn_notify)(const char* errinfo, const void* private_info, size_t cb, void* user_data), void* user_data, cl_int* errcode_ret) {
    return cl3w_apis.cl.clCreateContextFromType(properties, device_type, pfn_notify, user_data, errcode_ret);
}

cl_int clRetainContext(cl_context context) {
    return cl3w_apis.cl.clRetainContext(context);
}

cl_int clReleaseContext(cl_context context) {
    return cl3w_apis.cl.clReleaseContext(context);
}

cl_int clGetContextInfo(cl_context context, cl_context_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    return cl3w_apis.cl.clGetContextInfo(context, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_int clRetainCommandQueue(cl_command_queue command_queue) {
    return cl3w_apis.cl.clRetainCommandQueue(command_queue);
}

cl_int clReleaseCommandQueue(cl_command_queue command_queue) {
    return cl3w_apis.cl.clReleaseCommandQueue(command_queue);
}

cl_int clGetCommandQueueInfo(cl_command_queue command_queue, cl_command_queue_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    return cl3w_apis.cl.clGetCommandQueueInfo(command_queue, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_mem clCreateBuffer(cl_context context, cl_mem_flags flags, size_t size, void* host_ptr, cl_int* errcode_ret) {
    return cl3w_apis.cl.clCreateBuffer(context, flags, size, host_ptr, errcode_ret);
}

cl_mem clCreateSubBuffer(cl_mem buffer, cl_mem_flags flags, cl_buffer_create_type buffer_create_type, const void* buffer_create_info, cl_int* errcode_ret) {
    return cl3w_apis.cl.clCreateSubBuffer(buffer, flags, buffer_create_type, buffer_create_info, errcode_ret);
}

cl_mem clCreateImage(cl_context context, cl_mem_flags flags, const cl_image_format* image_format, const cl_image_desc* image_desc, void* host_ptr, cl_int* errcode_ret) {
    return cl3w_apis.cl.clCreateImage(context, flags, image_format, image_desc, host_ptr, errcode_ret);
}

cl_int clRetainMemObject(cl_mem memobj) {
    return cl3w_apis.cl.clRetainMemObject(memobj);
}

cl_int clReleaseMemObject(cl_mem memobj) {
    return cl3w_apis.cl.clReleaseMemObject(memobj);
}

cl_int clGetSupportedImageFormats(cl_context context, cl_mem_flags flags, cl_mem_object_type image_type, cl_uint num_entries, cl_image_format* image_formats, cl_uint* num_image_formats) {
    return cl3w_apis.cl.clGetSupportedImageFormats(context, flags, image_type, num_entries, image_formats, num_image_formats);
}

cl_int clGetMemObjectInfo(cl_mem memobj, cl_mem_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    return cl3w_apis.cl.clGetMemObjectInfo(memobj, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_int clGetImageInfo(cl_mem image, cl_image_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    return cl3w_apis.cl.clGetImageInfo(image, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_int clSetMemObjectDestructorCallback(cl_mem memobj, void (CL_CALLBACK* pfn_notify)(cl_mem memobj, void* user_data), void* user_data) {
    return cl3w_apis.cl.clSetMemObjectDestructorCallback(memobj, pfn_notify, user_data);
}

cl_int clRetainSampler(cl_sampler sampler) {
    return cl3w_apis.cl.clRetainSampler(sampler);
}

cl_int clReleaseSampler(cl_sampler sampler) {
    return cl3w_apis.cl.clReleaseSampler(sampler);
}

cl_int clGetSamplerInfo(cl_sampler sampler, cl_sampler_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    return cl3w_apis.cl.clGetSamplerInfo(sampler, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_program clCreateProgramWithSource(cl_context context, cl_uint count, const char ** strings, const size_t* lengths, cl_int* errcode_ret) {
    return cl3w_apis.cl.clCreateProgramWithSource(context, count, strings, lengths, errcode_ret);
}

cl_program clCreateProgramWithBinary(cl_context context, cl_uint num_devices, const cl_device_id* device_list, const size_t* lengths, const unsigned char ** binaries, cl_int* binary_status, cl_int* errcode_ret) {
    return cl3w_apis.cl.clCreateProgramWithBinary(context, num_devices, device_list, lengths, binaries, binary_status, errcode_ret);
}

cl_program clCreateProgramWithBuiltInKernels(cl_context context, cl_uint num_devices, const cl_device_id* device_list, const char* kernel_names, cl_int* errcode_ret) {
    return cl3w_apis.cl.clCreateProgramWithBuiltInKernels(context, num_devices, device_list, kernel_names, errcode_ret);
}

cl_int clRetainProgram(cl_program program) {
    return cl3w_apis.cl.clRetainProgram(program);
}

cl_int clReleaseProgram(cl_program program) {
    return cl3w_apis.cl.clReleaseProgram(program);
}

cl_int clBuildProgram(cl_program program, cl_uint num_devices, const cl_device_id* device_list, const char* options, void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data), void* user_data) {
    return cl3w_apis.cl.clBuildProgram(program, num_devices, device_list, options, pfn_notify, user_data);
}

cl_int clCompileProgram(cl_program program, cl_uint num_devices, const cl_device_id* device_list, const char* options, cl_uint num_input_headers, const cl_program* input_headers, const char ** header_include_names, void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data), void* user_data) {
    return cl3w_apis.cl.clCompileProgram(program, num_devices, device_list, options, num_input_headers, input_headers, header_include_names, pfn_notify, user_data);
}

cl_program clLinkProgram(cl_context context, cl_uint num_devices, const cl_device_id* device_list, const char* options, cl_uint num_input_programs, const cl_program* input_programs, void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data), void* user_data, cl_int* errcode_ret) {
    return cl3w_apis.cl.clLinkProgram(context, num_devices, device_list, options, num_input_programs, input_programs, pfn_notify, user_data, errcode_ret);
}

cl_int clUnloadPlatformCompiler(cl_platform_id platform) {
    return cl3w_apis.cl.clUnloadPlatformCompiler(platform);
}

cl_int clGetProgramInfo(cl_program program, cl_program_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    return cl3w_apis.cl.clGetProgramInfo(program, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_int clGetProgramBuildInfo(cl_program program, cl_device_id device, cl_program_build_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    return cl3w_apis.cl.clGetProgramBuildInfo(program, device, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_kernel clCreateKernel(cl_program program, const char* kernel_name, cl_int* errcode_ret) {
    return cl3w_apis.cl.clCreateKernel(program, kernel_name, errcode_ret);
}

cl_int clCreateKernelsInProgram(cl_program program, cl_uint num_kernels, cl_kernel* kernels, cl_uint* num_kernels_ret) {
    return cl3w_apis.cl.clCreateKernelsInProgram(program, num_kernels, kernels, num_kernels_ret);
}

cl_int clRetainKernel(cl_kernel kernel) {
    return cl3w_apis.cl.clRetainKernel(kernel);
}

cl_int clReleaseKernel(cl_kernel kernel) {
    return cl3w_apis.cl.clReleaseKernel(kernel);
}

cl_int clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void* arg_value) {
    return cl3w_apis.cl.clSetKernelArg(kernel, arg_index, arg_size, arg_value);
}

cl_int clGetKernelInfo(cl_kernel kernel, cl_kernel_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    return cl3w_apis.cl.clGetKernelInfo(kernel, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_int clGetKernelArgInfo(cl_kernel kernel, cl_uint arg_index, cl_kernel_arg_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    return cl3w_apis.cl.clGetKernelArgInfo(kernel, arg_index, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_int clGetKernelWorkGroupInfo(cl_kernel kernel, cl_device_id device, cl_kernel_work_group_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    return cl3w_apis.cl.clGetKernelWorkGroupInfo(kernel, device, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_int clWaitForEvents(cl_uint num_events, const cl_event* event_list) {
    return cl3w_apis.cl.clWaitForEvents(num_events, event_list);
}

cl_int clGetEventInfo(cl_event event, cl_event_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    return cl3w_apis.cl.clGetEventInfo(event, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_event clCreateUserEvent(cl_context context, cl_int* errcode_ret) {
    return cl3w_apis.cl.clCreateUserEvent(context, errcode_ret);
}

cl_int clRetainEvent(cl_event event) {
    return cl3w_apis.cl.clRetainEvent(event);
}

cl_int clReleaseEvent(cl_event event) {
    return cl3w_apis.cl.clReleaseEvent(event);
}

cl_int clSetUserEventStatus(cl_event event, cl_int execution_status) {
    return cl3w_apis.cl.clSetUserEventStatus(event, execution_status);
}

cl_int clSetEventCallback(cl_event event, cl_int command_exec_callback_type, void (CL_CALLBACK* pfn_notify)(cl_event event, cl_int event_command_status, void *user_data), void* user_data) {
    return cl3w_apis.cl.clSetEventCallback(event, command_exec_callback_type, pfn_notify, user_data);
}

cl_int clGetEventProfilingInfo(cl_event event, cl_profiling_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    return cl3w_apis.cl.clGetEventProfilingInfo(event, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_int clFlush(cl_command_queue command_queue) {
    return cl3w_apis.cl.clFlush(command_queue);
}

cl_int clFinish(cl_command_queue command_queue) {
    return cl3w_apis.cl.clFinish(command_queue);
}

cl_int clEnqueueReadBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read, size_t offset, size_t size, void* ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    return cl3w_apis.cl.clEnqueueReadBuffer(command_queue, buffer, blocking_read, offset, size, ptr, num_events_in_wait_list, event_wait_list, event);
}

cl_int clEnqueueReadBufferRect(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read, const size_t* buffer_origin, const size_t* host_origin, const size_t* region, size_t buffer_row_pitch, size_t buffer_slice_pitch, size_t host_row_pitch, size_t host_slice_pitch, void* ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    return cl3w_apis.cl.clEnqueueReadBufferRect(command_queue, buffer, blocking_read, buffer_origin, host_origin, region, buffer_row_pitch, buffer_slice_pitch, host_row_pitch, host_slice_pitch, ptr, num_events_in_wait_list, event_wait_list, event);
}

cl_int clEnqueueWriteBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write, size_t offset, size_t size, const void* ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    return cl3w_apis.cl.clEnqueueWriteBuffer(command_queue, buffer, blocking_write, offset, size, ptr, num_events_in_wait_list, event_wait_list, event);
}

cl_int clEnqueueWriteBufferRect(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write, const size_t* buffer_origin, const size_t* host_origin, const size_t* region, size_t buffer_row_pitch, size_t buffer_slice_pitch, size_t host_row_pitch, size_t host_slice_pitch, const void* ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    return cl3w_apis.cl.clEnqueueWriteBufferRect(command_queue, buffer, blocking_write, buffer_origin, host_origin, region, buffer_row_pitch, buffer_slice_pitch, host_row_pitch, host_slice_pitch, ptr, num_events_in_wait_list, event_wait_list, event);
}

cl_int clEnqueueFillBuffer(cl_command_queue command_queue, cl_mem buffer, const void* pattern, size_t pattern_size, size_t offset, size_t size, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    return cl3w_apis.cl.clEnqueueFillBuffer(command_queue, buffer, pattern, pattern_size, offset, size, num_events_in_wait_list, event_wait_list, event);
}

cl_int clEnqueueCopyBuffer(cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_buffer, size_t src_offset, size_t dst_offset, size_t size, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    return cl3w_apis.cl.clEnqueueCopyBuffer(command_queue, src_buffer, dst_buffer, src_offset, dst_offset, size, num_events_in_wait_list, event_wait_list, event);
}

cl_int clEnqueueCopyBufferRect(cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_buffer, const size_t* src_origin, const size_t* dst_origin, const size_t* region, size_t src_row_pitch, size_t src_slice_pitch, size_t dst_row_pitch, size_t dst_slice_pitch, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    return cl3w_apis.cl.clEnqueueCopyBufferRect(command_queue, src_buffer, dst_buffer, src_origin, dst_origin, region, src_row_pitch, src_slice_pitch, dst_row_pitch, dst_slice_pitch, num_events_in_wait_list, event_wait_list, event);
}

cl_int clEnqueueReadImage(cl_command_queue command_queue, cl_mem image, cl_bool blocking_read, const size_t* origin, const size_t* region, size_t row_pitch, size_t slice_pitch, void* ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    return cl3w_apis.cl.clEnqueueReadImage(command_queue, image, blocking_read, origin, region, row_pitch, slice_pitch, ptr, num_events_in_wait_list, event_wait_list, event);
}

cl_int clEnqueueWriteImage(cl_command_queue command_queue, cl_mem image, cl_bool blocking_write, const size_t* origin, const size_t* region, size_t input_row_pitch, size_t input_slice_pitch, const void* ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    return cl3w_apis.cl.clEnqueueWriteImage(command_queue, image, blocking_write, origin, region, input_row_pitch, input_slice_pitch, ptr, num_events_in_wait_list, event_wait_list, event);
}

cl_int clEnqueueFillImage(cl_command_queue command_queue, cl_mem image, const void* fill_color, const size_t* origin, const size_t* region, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    return cl3w_apis.cl.clEnqueueFillImage(command_queue, image, fill_color, origin, region, num_events_in_wait_list, event_wait_list, event);
}

cl_int clEnqueueCopyImage(cl_command_queue command_queue, cl_mem src_image, cl_mem dst_image, const size_t* src_origin, const size_t* dst_origin, const size_t* region, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    return cl3w_apis.cl.clEnqueueCopyImage(command_queue, src_image, dst_image, src_origin, dst_origin, region, num_events_in_wait_list, event_wait_list, event);
}

cl_int clEnqueueCopyImageToBuffer(cl_command_queue command_queue, cl_mem src_image, cl_mem dst_buffer, const size_t* src_origin, const size_t* region, size_t dst_offset, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    return cl3w_apis.cl.clEnqueueCopyImageToBuffer(command_queue, src_image, dst_buffer, src_origin, region, dst_offset, num_events_in_wait_list, event_wait_list, event);
}

cl_int clEnqueueCopyBufferToImage(cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_image, size_t src_offset, const size_t* dst_origin, const size_t* region, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    return cl3w_apis.cl.clEnqueueCopyBufferToImage(command_queue, src_buffer, dst_image, src_offset, dst_origin, region, num_events_in_wait_list, event_wait_list, event);
}

void* clEnqueueMapBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_map, cl_map_flags map_flags, size_t offset, size_t size, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event, cl_int* errcode_ret) {
    return cl3w_apis.cl.clEnqueueMapBuffer(command_queue, buffer, blocking_map, map_flags, offset, size, num_events_in_wait_list, event_wait_list, event, errcode_ret);
}

void* clEnqueueMapImage(cl_command_queue command_queue, cl_mem image, cl_bool blocking_map, cl_map_flags map_flags, const size_t* origin, const size_t* region, size_t* image_row_pitch, size_t* image_slice_pitch, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event, cl_int* errcode_ret) {
    return cl3w_apis.cl.clEnqueueMapImage(command_queue, image, blocking_map, map_flags, origin, region, image_row_pitch, image_slice_pitch, num_events_in_wait_list, event_wait_list, event, errcode_ret);
}

cl_int clEnqueueUnmapMemObject(cl_command_queue command_queue, cl_mem memobj, void* mapped_ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    return cl3w_apis.cl.clEnqueueUnmapMemObject(command_queue, memobj, mapped_ptr, num_events_in_wait_list, event_wait_list, event);
}

cl_int clEnqueueMigrateMemObjects(cl_command_queue command_queue, cl_uint num_mem_objects, const cl_mem* mem_objects, cl_mem_migration_flags flags, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    return cl3w_apis.cl.clEnqueueMigrateMemObjects(command_queue, num_mem_objects, mem_objects, flags, num_events_in_wait_list, event_wait_list, event);
}

cl_int clEnqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim, const size_t* global_work_offset, const size_t* global_work_size, const size_t* local_work_size, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    return cl3w_apis.cl.clEnqueueNDRangeKernel(command_queue, kernel, work_dim, global_work_offset, global_work_size, local_work_size, num_events_in_wait_list, event_wait_list, event);
}

cl_int clEnqueueNativeKernel(cl_command_queue command_queue, void (CL_CALLBACK* user_func)(void*), void* args, size_t cb_args, cl_uint num_mem_objects, const cl_mem* mem_list, const void ** args_mem_loc, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    return cl3w_apis.cl.clEnqueueNativeKernel(command_queue, user_func, args, cb_args, num_mem_objects, mem_list, args_mem_loc, num_events_in_wait_list, event_wait_list, event);
}

cl_int clEnqueueMarkerWithWaitList(cl_command_queue command_queue, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    return cl3w_apis.cl.clEnqueueMarkerWithWaitList(command_queue, num_events_in_wait_list, event_wait_list, event);
}

cl_int clEnqueueBarrierWithWaitList(cl_command_queue command_queue, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    return cl3w_apis.cl.clEnqueueBarrierWithWaitList(command_queue, num_events_in_wait_list, event_wait_list, event);
}

void* clGetExtensionFunctionAddressForPlatform(cl_platform_id platform, const char* func_name) {
    return cl3w_apis.cl.clGetExtensionFunctionAddressForPlatform(platform, func_name);
}

cl_int clSetCommandQueueProperty(cl_command_queue command_queue, cl_command_queue_properties properties, cl_bool enable, cl_command_queue_properties* old_properties) {
    return cl3w_apis.cl.clSetCommandQueueProperty(command_queue, properties, enable, old_properties);
}

cl_mem clCreateImage2D(cl_context context, cl_mem_flags flags, const cl_image_format* image_format, size_t image_width, size_t image_height, size_t image_row_pitch, void* host_ptr, cl_int* errcode_ret) {
    return cl3w_apis.cl.clCreateImage2D(context, flags, image_format, image_width, image_height, image_row_pitch, host_ptr, errcode_ret);
}

cl_mem clCreateImage3D(cl_context context, cl_mem_flags flags, const cl_image_format* image_format, size_t image_width, size_t image_height, size_t image_depth, size_t image_row_pitch, size_t image_slice_pitch, void* host_ptr, cl_int* errcode_ret) {
    return cl3w_apis.cl.clCreateImage3D(context, flags, image_format, image_width, image_height, image_depth, image_row_pitch, image_slice_pitch, host_ptr, errcode_ret);
}

cl_int clEnqueueMarker(cl_command_queue command_queue, cl_event* event) {
    return cl3w_apis.cl.clEnqueueMarker(command_queue, event);
}

cl_int clEnqueueWaitForEvents(cl_command_queue command_queue, cl_uint num_events, const cl_event* event_list) {
    return cl3w_apis.cl.clEnqueueWaitForEvents(command_queue, num_events, event_list);
}

cl_int clEnqueueBarrier(cl_command_queue command_queue) {
    return cl3w_apis.cl.clEnqueueBarrier(command_queue);
}

cl_int clUnloadCompiler() {
    return cl3w_apis.cl.clUnloadCompiler();
}

void* clGetExtensionFunctionAddress(const char* func_name) {
    return cl3w_apis.cl.clGetExtensionFunctionAddress(func_name);
}

cl_command_queue clCreateCommandQueue(cl_context context, cl_device_id device, cl_command_queue_properties properties, cl_int* errcode_ret) {
    return cl3w_apis.cl.clCreateCommandQueue(context, device, properties, errcode_ret);
}

cl_sampler clCreateSampler(cl_context context, cl_bool normalized_coords, cl_addressing_mode addressing_mode, cl_filter_mode filter_mode, cl_int* errcode_ret) {
    return cl3w_apis.cl.clCreateSampler(context, normalized_coords, addressing_mode, filter_mode, errcode_ret);
}

cl_int clEnqueueTask(cl_command_queue command_queue, cl_kernel kernel, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    return cl3w_apis.cl.clEnqueueTask(command_queue, kernel, num_events_in_wait_list, event_wait_list, event);
}


#define CL3W_ARRAY_SIZE(x)  (sizeof(x) / sizeof((x)[0]))

static void load_apis(void) {
    unload_apis();
    size_t i;
    for (i = 0; i < CL3W_ARRAY_SIZE(cl3w_api_names); i++) {
        CL3WclAPI func = get_api(cl3w_api_names[i]);
        if (func) {
            // printf("%s loaded\n", cl3w_api_names[i]);
            cl3w_apis.ptr[i] = func;
        }
    }
}

static void unload_apis(void) {
    cl3w_apis.ptr[0] = (CL3WclAPI)clGetPlatformIDsDummyImpl;
    cl3w_apis.ptr[1] = (CL3WclAPI)clGetPlatformInfoDummyImpl;
    cl3w_apis.ptr[2] = (CL3WclAPI)clGetDeviceIDsDummyImpl;
    cl3w_apis.ptr[3] = (CL3WclAPI)clGetDeviceInfoDummyImpl;
    cl3w_apis.ptr[4] = (CL3WclAPI)clCreateSubDevicesDummyImpl;
    cl3w_apis.ptr[5] = (CL3WclAPI)clRetainDeviceDummyImpl;
    cl3w_apis.ptr[6] = (CL3WclAPI)clReleaseDeviceDummyImpl;
    cl3w_apis.ptr[7] = (CL3WclAPI)clCreateContextDummyImpl;
    cl3w_apis.ptr[8] = (CL3WclAPI)clCreateContextFromTypeDummyImpl;
    cl3w_apis.ptr[9] = (CL3WclAPI)clRetainContextDummyImpl;
    cl3w_apis.ptr[10] = (CL3WclAPI)clReleaseContextDummyImpl;
    cl3w_apis.ptr[11] = (CL3WclAPI)clGetContextInfoDummyImpl;
    cl3w_apis.ptr[12] = (CL3WclAPI)clRetainCommandQueueDummyImpl;
    cl3w_apis.ptr[13] = (CL3WclAPI)clReleaseCommandQueueDummyImpl;
    cl3w_apis.ptr[14] = (CL3WclAPI)clGetCommandQueueInfoDummyImpl;
    cl3w_apis.ptr[15] = (CL3WclAPI)clCreateBufferDummyImpl;
    cl3w_apis.ptr[16] = (CL3WclAPI)clCreateSubBufferDummyImpl;
    cl3w_apis.ptr[17] = (CL3WclAPI)clCreateImageDummyImpl;
    cl3w_apis.ptr[18] = (CL3WclAPI)clRetainMemObjectDummyImpl;
    cl3w_apis.ptr[19] = (CL3WclAPI)clReleaseMemObjectDummyImpl;
    cl3w_apis.ptr[20] = (CL3WclAPI)clGetSupportedImageFormatsDummyImpl;
    cl3w_apis.ptr[21] = (CL3WclAPI)clGetMemObjectInfoDummyImpl;
    cl3w_apis.ptr[22] = (CL3WclAPI)clGetImageInfoDummyImpl;
    cl3w_apis.ptr[23] = (CL3WclAPI)clSetMemObjectDestructorCallbackDummyImpl;
    cl3w_apis.ptr[24] = (CL3WclAPI)clRetainSamplerDummyImpl;
    cl3w_apis.ptr[25] = (CL3WclAPI)clReleaseSamplerDummyImpl;
    cl3w_apis.ptr[26] = (CL3WclAPI)clGetSamplerInfoDummyImpl;
    cl3w_apis.ptr[27] = (CL3WclAPI)clCreateProgramWithSourceDummyImpl;
    cl3w_apis.ptr[28] = (CL3WclAPI)clCreateProgramWithBinaryDummyImpl;
    cl3w_apis.ptr[29] = (CL3WclAPI)clCreateProgramWithBuiltInKernelsDummyImpl;
    cl3w_apis.ptr[30] = (CL3WclAPI)clRetainProgramDummyImpl;
    cl3w_apis.ptr[31] = (CL3WclAPI)clReleaseProgramDummyImpl;
    cl3w_apis.ptr[32] = (CL3WclAPI)clBuildProgramDummyImpl;
    cl3w_apis.ptr[33] = (CL3WclAPI)clCompileProgramDummyImpl;
    cl3w_apis.ptr[34] = (CL3WclAPI)clLinkProgramDummyImpl;
    cl3w_apis.ptr[35] = (CL3WclAPI)clUnloadPlatformCompilerDummyImpl;
    cl3w_apis.ptr[36] = (CL3WclAPI)clGetProgramInfoDummyImpl;
    cl3w_apis.ptr[37] = (CL3WclAPI)clGetProgramBuildInfoDummyImpl;
    cl3w_apis.ptr[38] = (CL3WclAPI)clCreateKernelDummyImpl;
    cl3w_apis.ptr[39] = (CL3WclAPI)clCreateKernelsInProgramDummyImpl;
    cl3w_apis.ptr[40] = (CL3WclAPI)clRetainKernelDummyImpl;
    cl3w_apis.ptr[41] = (CL3WclAPI)clReleaseKernelDummyImpl;
    cl3w_apis.ptr[42] = (CL3WclAPI)clSetKernelArgDummyImpl;
    cl3w_apis.ptr[43] = (CL3WclAPI)clGetKernelInfoDummyImpl;
    cl3w_apis.ptr[44] = (CL3WclAPI)clGetKernelArgInfoDummyImpl;
    cl3w_apis.ptr[45] = (CL3WclAPI)clGetKernelWorkGroupInfoDummyImpl;
    cl3w_apis.ptr[46] = (CL3WclAPI)clWaitForEventsDummyImpl;
    cl3w_apis.ptr[47] = (CL3WclAPI)clGetEventInfoDummyImpl;
    cl3w_apis.ptr[48] = (CL3WclAPI)clCreateUserEventDummyImpl;
    cl3w_apis.ptr[49] = (CL3WclAPI)clRetainEventDummyImpl;
    cl3w_apis.ptr[50] = (CL3WclAPI)clReleaseEventDummyImpl;
    cl3w_apis.ptr[51] = (CL3WclAPI)clSetUserEventStatusDummyImpl;
    cl3w_apis.ptr[52] = (CL3WclAPI)clSetEventCallbackDummyImpl;
    cl3w_apis.ptr[53] = (CL3WclAPI)clGetEventProfilingInfoDummyImpl;
    cl3w_apis.ptr[54] = (CL3WclAPI)clFlushDummyImpl;
    cl3w_apis.ptr[55] = (CL3WclAPI)clFinishDummyImpl;
    cl3w_apis.ptr[56] = (CL3WclAPI)clEnqueueReadBufferDummyImpl;
    cl3w_apis.ptr[57] = (CL3WclAPI)clEnqueueReadBufferRectDummyImpl;
    cl3w_apis.ptr[58] = (CL3WclAPI)clEnqueueWriteBufferDummyImpl;
    cl3w_apis.ptr[59] = (CL3WclAPI)clEnqueueWriteBufferRectDummyImpl;
    cl3w_apis.ptr[60] = (CL3WclAPI)clEnqueueFillBufferDummyImpl;
    cl3w_apis.ptr[61] = (CL3WclAPI)clEnqueueCopyBufferDummyImpl;
    cl3w_apis.ptr[62] = (CL3WclAPI)clEnqueueCopyBufferRectDummyImpl;
    cl3w_apis.ptr[63] = (CL3WclAPI)clEnqueueReadImageDummyImpl;
    cl3w_apis.ptr[64] = (CL3WclAPI)clEnqueueWriteImageDummyImpl;
    cl3w_apis.ptr[65] = (CL3WclAPI)clEnqueueFillImageDummyImpl;
    cl3w_apis.ptr[66] = (CL3WclAPI)clEnqueueCopyImageDummyImpl;
    cl3w_apis.ptr[67] = (CL3WclAPI)clEnqueueCopyImageToBufferDummyImpl;
    cl3w_apis.ptr[68] = (CL3WclAPI)clEnqueueCopyBufferToImageDummyImpl;
    cl3w_apis.ptr[69] = (CL3WclAPI)clEnqueueMapBufferDummyImpl;
    cl3w_apis.ptr[70] = (CL3WclAPI)clEnqueueMapImageDummyImpl;
    cl3w_apis.ptr[71] = (CL3WclAPI)clEnqueueUnmapMemObjectDummyImpl;
    cl3w_apis.ptr[72] = (CL3WclAPI)clEnqueueMigrateMemObjectsDummyImpl;
    cl3w_apis.ptr[73] = (CL3WclAPI)clEnqueueNDRangeKernelDummyImpl;
    cl3w_apis.ptr[74] = (CL3WclAPI)clEnqueueNativeKernelDummyImpl;
    cl3w_apis.ptr[75] = (CL3WclAPI)clEnqueueMarkerWithWaitListDummyImpl;
    cl3w_apis.ptr[76] = (CL3WclAPI)clEnqueueBarrierWithWaitListDummyImpl;
    cl3w_apis.ptr[77] = (CL3WclAPI)clGetExtensionFunctionAddressForPlatformDummyImpl;
    cl3w_apis.ptr[78] = (CL3WclAPI)clSetCommandQueuePropertyDummyImpl;
    cl3w_apis.ptr[79] = (CL3WclAPI)clCreateImage2DDummyImpl;
    cl3w_apis.ptr[80] = (CL3WclAPI)clCreateImage3DDummyImpl;
    cl3w_apis.ptr[81] = (CL3WclAPI)clEnqueueMarkerDummyImpl;
    cl3w_apis.ptr[82] = (CL3WclAPI)clEnqueueWaitForEventsDummyImpl;
    cl3w_apis.ptr[83] = (CL3WclAPI)clEnqueueBarrierDummyImpl;
    cl3w_apis.ptr[84] = (CL3WclAPI)clUnloadCompilerDummyImpl;
    cl3w_apis.ptr[85] = (CL3WclAPI)clGetExtensionFunctionAddressDummyImpl;
    cl3w_apis.ptr[86] = (CL3WclAPI)clCreateCommandQueueDummyImpl;
    cl3w_apis.ptr[87] = (CL3WclAPI)clCreateSamplerDummyImpl;
    cl3w_apis.ptr[88] = (CL3WclAPI)clEnqueueTaskDummyImpl;
}

static const char* get_probe_api_name(void) {
    /* clCreateContext */
    return cl3w_api_names[7];
}

CL3W_STATUS cl3wInit(void) {
    return cl3wInit2(default_lib_paths, CL3W_ARRAY_SIZE(default_lib_paths));
}

CL3W_STATUS cl3wInit2(const char** libpaths, size_t npaths) {
    CL3W_RET_IF_ERROR(open_libcl(libpaths, npaths));
    load_apis();
    return CL3W_OK;
}

CL3W_STATUS cl3wUnload() {
    unload_apis();
    close_libcl();
    return CL3W_OK;
}

#undef CL3W_ARRAY_SIZE

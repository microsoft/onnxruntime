// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/*
 * This file was generated with cl3w_gen.py, part of cl3w
 * (hosted at https://github.com/cloudhan/cl3w)
*/
#ifndef __cl3w_h_
#define __cl3w_h_

#include <CL/opencl.h>

#ifndef CL3W_API
#define CL3W_API
#endif

#if defined(_WIN32)
#ifndef CL_API_ENTRY
#define CL_API_ENTRY
#endif
#ifndef CL_API_ENTRYP
#define CL_API_ENTRYP CL_API_ENTRY*
#endif
#ifndef CL_API_CALL
#define CL_API_CALL     __stdcall
#endif
#ifndef CL_CALLBACK
#define CL_CALLBACK     __stdcall
#endif
#else
#ifndef CL_API_ENTRY
#define CL_API_ENTRY
#endif
#ifndef CL_API_ENTRYP
#define CL_API_ENTRYP CL_API_ENTRY*
#endif
#ifndef CL_API_CALL
#define CL_API_CALL
#endif
#ifndef CL_CALLBACK
#define CL_CALLBACK
#endif
#endif

#ifndef CLAPI
#define CLAPI
#endif
#ifndef CLAPIP
#define CLAPIP CLAPI *
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define CL3W_STATUS int
#define CL3W_OK (CL3W_STATUS)0
#define CL3W_ERROR_INIT (CL3W_STATUS)-1
#define CL3W_ERROR_LIBRARY_OPEN (CL3W_STATUS)-2

typedef void (*CL3WclAPI)(void);

/// Init with default path
CL3W_API CL3W_STATUS cl3wInit();

/// Init with user defined heuristics
CL3W_API CL3W_STATUS cl3wInit2(const char** libpaths, size_t npaths);

/// Unload OpenCL()
CL3W_API CL3W_STATUS cl3wUnload();

typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLGETPLATFORMIDSFUNC)(cl_uint num_entries, cl_platform_id* platforms, cl_uint* num_platforms);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLGETPLATFORMINFOFUNC)(cl_platform_id platform, cl_platform_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLGETDEVICEIDSFUNC)(cl_platform_id platform, cl_device_type device_type, cl_uint num_entries, cl_device_id* devices, cl_uint* num_devices);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLGETDEVICEINFOFUNC)(cl_device_id device, cl_device_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLCREATESUBDEVICESFUNC)(cl_device_id in_device, const cl_device_partition_property* properties, cl_uint num_devices, cl_device_id* out_devices, cl_uint* num_devices_ret);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLRETAINDEVICEFUNC)(cl_device_id device);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLRELEASEDEVICEFUNC)(cl_device_id device);
typedef cl_context (CL_API_CALL CL_API_ENTRYP PFNCLCREATECONTEXTFUNC)(const cl_context_properties* properties, cl_uint num_devices, const cl_device_id* devices, void (CL_CALLBACK* pfn_notify)(const char* errinfo, const void* private_info, size_t cb, void* user_data), void* user_data, cl_int* errcode_ret);
typedef cl_context (CL_API_CALL CL_API_ENTRYP PFNCLCREATECONTEXTFROMTYPEFUNC)(const cl_context_properties* properties, cl_device_type device_type, void (CL_CALLBACK* pfn_notify)(const char* errinfo, const void* private_info, size_t cb, void* user_data), void* user_data, cl_int* errcode_ret);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLRETAINCONTEXTFUNC)(cl_context context);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLRELEASECONTEXTFUNC)(cl_context context);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLGETCONTEXTINFOFUNC)(cl_context context, cl_context_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLRETAINCOMMANDQUEUEFUNC)(cl_command_queue command_queue);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLRELEASECOMMANDQUEUEFUNC)(cl_command_queue command_queue);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLGETCOMMANDQUEUEINFOFUNC)(cl_command_queue command_queue, cl_command_queue_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret);
typedef cl_mem (CL_API_CALL CL_API_ENTRYP PFNCLCREATEBUFFERFUNC)(cl_context context, cl_mem_flags flags, size_t size, void* host_ptr, cl_int* errcode_ret);
typedef cl_mem (CL_API_CALL CL_API_ENTRYP PFNCLCREATESUBBUFFERFUNC)(cl_mem buffer, cl_mem_flags flags, cl_buffer_create_type buffer_create_type, const void* buffer_create_info, cl_int* errcode_ret);
typedef cl_mem (CL_API_CALL CL_API_ENTRYP PFNCLCREATEIMAGEFUNC)(cl_context context, cl_mem_flags flags, const cl_image_format* image_format, const cl_image_desc* image_desc, void* host_ptr, cl_int* errcode_ret);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLRETAINMEMOBJECTFUNC)(cl_mem memobj);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLRELEASEMEMOBJECTFUNC)(cl_mem memobj);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLGETSUPPORTEDIMAGEFORMATSFUNC)(cl_context context, cl_mem_flags flags, cl_mem_object_type image_type, cl_uint num_entries, cl_image_format* image_formats, cl_uint* num_image_formats);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLGETMEMOBJECTINFOFUNC)(cl_mem memobj, cl_mem_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLGETIMAGEINFOFUNC)(cl_mem image, cl_image_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLSETMEMOBJECTDESTRUCTORCALLBACKFUNC)(cl_mem memobj, void (CL_CALLBACK* pfn_notify)(cl_mem memobj, void* user_data), void* user_data);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLRETAINSAMPLERFUNC)(cl_sampler sampler);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLRELEASESAMPLERFUNC)(cl_sampler sampler);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLGETSAMPLERINFOFUNC)(cl_sampler sampler, cl_sampler_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret);
typedef cl_program (CL_API_CALL CL_API_ENTRYP PFNCLCREATEPROGRAMWITHSOURCEFUNC)(cl_context context, cl_uint count, const char ** strings, const size_t* lengths, cl_int* errcode_ret);
typedef cl_program (CL_API_CALL CL_API_ENTRYP PFNCLCREATEPROGRAMWITHBINARYFUNC)(cl_context context, cl_uint num_devices, const cl_device_id* device_list, const size_t* lengths, const unsigned char ** binaries, cl_int* binary_status, cl_int* errcode_ret);
typedef cl_program (CL_API_CALL CL_API_ENTRYP PFNCLCREATEPROGRAMWITHBUILTINKERNELSFUNC)(cl_context context, cl_uint num_devices, const cl_device_id* device_list, const char* kernel_names, cl_int* errcode_ret);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLRETAINPROGRAMFUNC)(cl_program program);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLRELEASEPROGRAMFUNC)(cl_program program);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLBUILDPROGRAMFUNC)(cl_program program, cl_uint num_devices, const cl_device_id* device_list, const char* options, void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data), void* user_data);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLCOMPILEPROGRAMFUNC)(cl_program program, cl_uint num_devices, const cl_device_id* device_list, const char* options, cl_uint num_input_headers, const cl_program* input_headers, const char ** header_include_names, void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data), void* user_data);
typedef cl_program (CL_API_CALL CL_API_ENTRYP PFNCLLINKPROGRAMFUNC)(cl_context context, cl_uint num_devices, const cl_device_id* device_list, const char* options, cl_uint num_input_programs, const cl_program* input_programs, void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data), void* user_data, cl_int* errcode_ret);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLUNLOADPLATFORMCOMPILERFUNC)(cl_platform_id platform);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLGETPROGRAMINFOFUNC)(cl_program program, cl_program_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLGETPROGRAMBUILDINFOFUNC)(cl_program program, cl_device_id device, cl_program_build_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret);
typedef cl_kernel (CL_API_CALL CL_API_ENTRYP PFNCLCREATEKERNELFUNC)(cl_program program, const char* kernel_name, cl_int* errcode_ret);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLCREATEKERNELSINPROGRAMFUNC)(cl_program program, cl_uint num_kernels, cl_kernel* kernels, cl_uint* num_kernels_ret);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLRETAINKERNELFUNC)(cl_kernel kernel);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLRELEASEKERNELFUNC)(cl_kernel kernel);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLSETKERNELARGFUNC)(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void* arg_value);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLGETKERNELINFOFUNC)(cl_kernel kernel, cl_kernel_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLGETKERNELARGINFOFUNC)(cl_kernel kernel, cl_uint arg_index, cl_kernel_arg_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLGETKERNELWORKGROUPINFOFUNC)(cl_kernel kernel, cl_device_id device, cl_kernel_work_group_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLWAITFOREVENTSFUNC)(cl_uint num_events, const cl_event* event_list);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLGETEVENTINFOFUNC)(cl_event event, cl_event_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret);
typedef cl_event (CL_API_CALL CL_API_ENTRYP PFNCLCREATEUSEREVENTFUNC)(cl_context context, cl_int* errcode_ret);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLRETAINEVENTFUNC)(cl_event event);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLRELEASEEVENTFUNC)(cl_event event);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLSETUSEREVENTSTATUSFUNC)(cl_event event, cl_int execution_status);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLSETEVENTCALLBACKFUNC)(cl_event event, cl_int command_exec_callback_type, void (CL_CALLBACK* pfn_notify)(cl_event event, cl_int event_command_status, void *user_data), void* user_data);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLGETEVENTPROFILINGINFOFUNC)(cl_event event, cl_profiling_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLFLUSHFUNC)(cl_command_queue command_queue);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLFINISHFUNC)(cl_command_queue command_queue);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLENQUEUEREADBUFFERFUNC)(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read, size_t offset, size_t size, void* ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLENQUEUEREADBUFFERRECTFUNC)(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read, const size_t* buffer_origin, const size_t* host_origin, const size_t* region, size_t buffer_row_pitch, size_t buffer_slice_pitch, size_t host_row_pitch, size_t host_slice_pitch, void* ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLENQUEUEWRITEBUFFERFUNC)(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write, size_t offset, size_t size, const void* ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLENQUEUEWRITEBUFFERRECTFUNC)(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write, const size_t* buffer_origin, const size_t* host_origin, const size_t* region, size_t buffer_row_pitch, size_t buffer_slice_pitch, size_t host_row_pitch, size_t host_slice_pitch, const void* ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLENQUEUEFILLBUFFERFUNC)(cl_command_queue command_queue, cl_mem buffer, const void* pattern, size_t pattern_size, size_t offset, size_t size, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLENQUEUECOPYBUFFERFUNC)(cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_buffer, size_t src_offset, size_t dst_offset, size_t size, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLENQUEUECOPYBUFFERRECTFUNC)(cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_buffer, const size_t* src_origin, const size_t* dst_origin, const size_t* region, size_t src_row_pitch, size_t src_slice_pitch, size_t dst_row_pitch, size_t dst_slice_pitch, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLENQUEUEREADIMAGEFUNC)(cl_command_queue command_queue, cl_mem image, cl_bool blocking_read, const size_t* origin, const size_t* region, size_t row_pitch, size_t slice_pitch, void* ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLENQUEUEWRITEIMAGEFUNC)(cl_command_queue command_queue, cl_mem image, cl_bool blocking_write, const size_t* origin, const size_t* region, size_t input_row_pitch, size_t input_slice_pitch, const void* ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLENQUEUEFILLIMAGEFUNC)(cl_command_queue command_queue, cl_mem image, const void* fill_color, const size_t* origin, const size_t* region, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLENQUEUECOPYIMAGEFUNC)(cl_command_queue command_queue, cl_mem src_image, cl_mem dst_image, const size_t* src_origin, const size_t* dst_origin, const size_t* region, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLENQUEUECOPYIMAGETOBUFFERFUNC)(cl_command_queue command_queue, cl_mem src_image, cl_mem dst_buffer, const size_t* src_origin, const size_t* region, size_t dst_offset, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLENQUEUECOPYBUFFERTOIMAGEFUNC)(cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_image, size_t src_offset, const size_t* dst_origin, const size_t* region, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);
typedef void* (CL_API_CALL CL_API_ENTRYP PFNCLENQUEUEMAPBUFFERFUNC)(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_map, cl_map_flags map_flags, size_t offset, size_t size, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event, cl_int* errcode_ret);
typedef void* (CL_API_CALL CL_API_ENTRYP PFNCLENQUEUEMAPIMAGEFUNC)(cl_command_queue command_queue, cl_mem image, cl_bool blocking_map, cl_map_flags map_flags, const size_t* origin, const size_t* region, size_t* image_row_pitch, size_t* image_slice_pitch, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event, cl_int* errcode_ret);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLENQUEUEUNMAPMEMOBJECTFUNC)(cl_command_queue command_queue, cl_mem memobj, void* mapped_ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLENQUEUEMIGRATEMEMOBJECTSFUNC)(cl_command_queue command_queue, cl_uint num_mem_objects, const cl_mem* mem_objects, cl_mem_migration_flags flags, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLENQUEUENDRANGEKERNELFUNC)(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim, const size_t* global_work_offset, const size_t* global_work_size, const size_t* local_work_size, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLENQUEUENATIVEKERNELFUNC)(cl_command_queue command_queue, void (CL_CALLBACK* user_func)(void*), void* args, size_t cb_args, cl_uint num_mem_objects, const cl_mem* mem_list, const void ** args_mem_loc, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLENQUEUEMARKERWITHWAITLISTFUNC)(cl_command_queue command_queue, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLENQUEUEBARRIERWITHWAITLISTFUNC)(cl_command_queue command_queue, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);
typedef void* (CL_API_CALL CL_API_ENTRYP PFNCLGETEXTENSIONFUNCTIONADDRESSFORPLATFORMFUNC)(cl_platform_id platform, const char* func_name);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLSETCOMMANDQUEUEPROPERTYFUNC)(cl_command_queue command_queue, cl_command_queue_properties properties, cl_bool enable, cl_command_queue_properties* old_properties);
typedef cl_mem (CL_API_CALL CL_API_ENTRYP PFNCLCREATEIMAGE2DFUNC)(cl_context context, cl_mem_flags flags, const cl_image_format* image_format, size_t image_width, size_t image_height, size_t image_row_pitch, void* host_ptr, cl_int* errcode_ret);
typedef cl_mem (CL_API_CALL CL_API_ENTRYP PFNCLCREATEIMAGE3DFUNC)(cl_context context, cl_mem_flags flags, const cl_image_format* image_format, size_t image_width, size_t image_height, size_t image_depth, size_t image_row_pitch, size_t image_slice_pitch, void* host_ptr, cl_int* errcode_ret);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLENQUEUEMARKERFUNC)(cl_command_queue command_queue, cl_event* event);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLENQUEUEWAITFOREVENTSFUNC)(cl_command_queue command_queue, cl_uint num_events, const cl_event* event_list);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLENQUEUEBARRIERFUNC)(cl_command_queue command_queue);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLUNLOADCOMPILERFUNC)();
typedef void* (CL_API_CALL CL_API_ENTRYP PFNCLGETEXTENSIONFUNCTIONADDRESSFUNC)(const char* func_name);
typedef cl_command_queue (CL_API_CALL CL_API_ENTRYP PFNCLCREATECOMMANDQUEUEFUNC)(cl_context context, cl_device_id device, cl_command_queue_properties properties, cl_int* errcode_ret);
typedef cl_sampler (CL_API_CALL CL_API_ENTRYP PFNCLCREATESAMPLERFUNC)(cl_context context, cl_bool normalized_coords, cl_addressing_mode addressing_mode, cl_filter_mode filter_mode, cl_int* errcode_ret);
typedef cl_int (CL_API_CALL CL_API_ENTRYP PFNCLENQUEUETASKFUNC)(cl_command_queue command_queue, cl_kernel kernel, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);

union CL3WAPIs {
    CL3WclAPI ptr[89];
    struct {
        PFNCLGETPLATFORMIDSFUNC                                 clGetPlatformIDs;
        PFNCLGETPLATFORMINFOFUNC                                clGetPlatformInfo;
        PFNCLGETDEVICEIDSFUNC                                   clGetDeviceIDs;
        PFNCLGETDEVICEINFOFUNC                                  clGetDeviceInfo;
        PFNCLCREATESUBDEVICESFUNC                               clCreateSubDevices;
        PFNCLRETAINDEVICEFUNC                                   clRetainDevice;
        PFNCLRELEASEDEVICEFUNC                                  clReleaseDevice;
        PFNCLCREATECONTEXTFUNC                                  clCreateContext;
        PFNCLCREATECONTEXTFROMTYPEFUNC                          clCreateContextFromType;
        PFNCLRETAINCONTEXTFUNC                                  clRetainContext;
        PFNCLRELEASECONTEXTFUNC                                 clReleaseContext;
        PFNCLGETCONTEXTINFOFUNC                                 clGetContextInfo;
        PFNCLRETAINCOMMANDQUEUEFUNC                             clRetainCommandQueue;
        PFNCLRELEASECOMMANDQUEUEFUNC                            clReleaseCommandQueue;
        PFNCLGETCOMMANDQUEUEINFOFUNC                            clGetCommandQueueInfo;
        PFNCLCREATEBUFFERFUNC                                   clCreateBuffer;
        PFNCLCREATESUBBUFFERFUNC                                clCreateSubBuffer;
        PFNCLCREATEIMAGEFUNC                                    clCreateImage;
        PFNCLRETAINMEMOBJECTFUNC                                clRetainMemObject;
        PFNCLRELEASEMEMOBJECTFUNC                               clReleaseMemObject;
        PFNCLGETSUPPORTEDIMAGEFORMATSFUNC                       clGetSupportedImageFormats;
        PFNCLGETMEMOBJECTINFOFUNC                               clGetMemObjectInfo;
        PFNCLGETIMAGEINFOFUNC                                   clGetImageInfo;
        PFNCLSETMEMOBJECTDESTRUCTORCALLBACKFUNC                 clSetMemObjectDestructorCallback;
        PFNCLRETAINSAMPLERFUNC                                  clRetainSampler;
        PFNCLRELEASESAMPLERFUNC                                 clReleaseSampler;
        PFNCLGETSAMPLERINFOFUNC                                 clGetSamplerInfo;
        PFNCLCREATEPROGRAMWITHSOURCEFUNC                        clCreateProgramWithSource;
        PFNCLCREATEPROGRAMWITHBINARYFUNC                        clCreateProgramWithBinary;
        PFNCLCREATEPROGRAMWITHBUILTINKERNELSFUNC                clCreateProgramWithBuiltInKernels;
        PFNCLRETAINPROGRAMFUNC                                  clRetainProgram;
        PFNCLRELEASEPROGRAMFUNC                                 clReleaseProgram;
        PFNCLBUILDPROGRAMFUNC                                   clBuildProgram;
        PFNCLCOMPILEPROGRAMFUNC                                 clCompileProgram;
        PFNCLLINKPROGRAMFUNC                                    clLinkProgram;
        PFNCLUNLOADPLATFORMCOMPILERFUNC                         clUnloadPlatformCompiler;
        PFNCLGETPROGRAMINFOFUNC                                 clGetProgramInfo;
        PFNCLGETPROGRAMBUILDINFOFUNC                            clGetProgramBuildInfo;
        PFNCLCREATEKERNELFUNC                                   clCreateKernel;
        PFNCLCREATEKERNELSINPROGRAMFUNC                         clCreateKernelsInProgram;
        PFNCLRETAINKERNELFUNC                                   clRetainKernel;
        PFNCLRELEASEKERNELFUNC                                  clReleaseKernel;
        PFNCLSETKERNELARGFUNC                                   clSetKernelArg;
        PFNCLGETKERNELINFOFUNC                                  clGetKernelInfo;
        PFNCLGETKERNELARGINFOFUNC                               clGetKernelArgInfo;
        PFNCLGETKERNELWORKGROUPINFOFUNC                         clGetKernelWorkGroupInfo;
        PFNCLWAITFOREVENTSFUNC                                  clWaitForEvents;
        PFNCLGETEVENTINFOFUNC                                   clGetEventInfo;
        PFNCLCREATEUSEREVENTFUNC                                clCreateUserEvent;
        PFNCLRETAINEVENTFUNC                                    clRetainEvent;
        PFNCLRELEASEEVENTFUNC                                   clReleaseEvent;
        PFNCLSETUSEREVENTSTATUSFUNC                             clSetUserEventStatus;
        PFNCLSETEVENTCALLBACKFUNC                               clSetEventCallback;
        PFNCLGETEVENTPROFILINGINFOFUNC                          clGetEventProfilingInfo;
        PFNCLFLUSHFUNC                                          clFlush;
        PFNCLFINISHFUNC                                         clFinish;
        PFNCLENQUEUEREADBUFFERFUNC                              clEnqueueReadBuffer;
        PFNCLENQUEUEREADBUFFERRECTFUNC                          clEnqueueReadBufferRect;
        PFNCLENQUEUEWRITEBUFFERFUNC                             clEnqueueWriteBuffer;
        PFNCLENQUEUEWRITEBUFFERRECTFUNC                         clEnqueueWriteBufferRect;
        PFNCLENQUEUEFILLBUFFERFUNC                              clEnqueueFillBuffer;
        PFNCLENQUEUECOPYBUFFERFUNC                              clEnqueueCopyBuffer;
        PFNCLENQUEUECOPYBUFFERRECTFUNC                          clEnqueueCopyBufferRect;
        PFNCLENQUEUEREADIMAGEFUNC                               clEnqueueReadImage;
        PFNCLENQUEUEWRITEIMAGEFUNC                              clEnqueueWriteImage;
        PFNCLENQUEUEFILLIMAGEFUNC                               clEnqueueFillImage;
        PFNCLENQUEUECOPYIMAGEFUNC                               clEnqueueCopyImage;
        PFNCLENQUEUECOPYIMAGETOBUFFERFUNC                       clEnqueueCopyImageToBuffer;
        PFNCLENQUEUECOPYBUFFERTOIMAGEFUNC                       clEnqueueCopyBufferToImage;
        PFNCLENQUEUEMAPBUFFERFUNC                               clEnqueueMapBuffer;
        PFNCLENQUEUEMAPIMAGEFUNC                                clEnqueueMapImage;
        PFNCLENQUEUEUNMAPMEMOBJECTFUNC                          clEnqueueUnmapMemObject;
        PFNCLENQUEUEMIGRATEMEMOBJECTSFUNC                       clEnqueueMigrateMemObjects;
        PFNCLENQUEUENDRANGEKERNELFUNC                           clEnqueueNDRangeKernel;
        PFNCLENQUEUENATIVEKERNELFUNC                            clEnqueueNativeKernel;
        PFNCLENQUEUEMARKERWITHWAITLISTFUNC                      clEnqueueMarkerWithWaitList;
        PFNCLENQUEUEBARRIERWITHWAITLISTFUNC                     clEnqueueBarrierWithWaitList;
        PFNCLGETEXTENSIONFUNCTIONADDRESSFORPLATFORMFUNC         clGetExtensionFunctionAddressForPlatform;
        PFNCLSETCOMMANDQUEUEPROPERTYFUNC                        clSetCommandQueueProperty;
        PFNCLCREATEIMAGE2DFUNC                                  clCreateImage2D;
        PFNCLCREATEIMAGE3DFUNC                                  clCreateImage3D;
        PFNCLENQUEUEMARKERFUNC                                  clEnqueueMarker;
        PFNCLENQUEUEWAITFOREVENTSFUNC                           clEnqueueWaitForEvents;
        PFNCLENQUEUEBARRIERFUNC                                 clEnqueueBarrier;
        PFNCLUNLOADCOMPILERFUNC                                 clUnloadCompiler;
        PFNCLGETEXTENSIONFUNCTIONADDRESSFUNC                    clGetExtensionFunctionAddress;
        PFNCLCREATECOMMANDQUEUEFUNC                             clCreateCommandQueue;
        PFNCLCREATESAMPLERFUNC                                  clCreateSampler;
        PFNCLENQUEUETASKFUNC                                    clEnqueueTask;
    } cl;
};
CL3W_API extern union CL3WAPIs cl3w_apis;

#ifndef CL3W_NO_CL_API_DEFINES
#define clGetPlatformIDs                                   cl3w_apis.cl.clGetPlatformIDs
#define clGetPlatformInfo                                  cl3w_apis.cl.clGetPlatformInfo
#define clGetDeviceIDs                                     cl3w_apis.cl.clGetDeviceIDs
#define clGetDeviceInfo                                    cl3w_apis.cl.clGetDeviceInfo
#define clCreateSubDevices                                 cl3w_apis.cl.clCreateSubDevices
#define clRetainDevice                                     cl3w_apis.cl.clRetainDevice
#define clReleaseDevice                                    cl3w_apis.cl.clReleaseDevice
#define clCreateContext                                    cl3w_apis.cl.clCreateContext
#define clCreateContextFromType                            cl3w_apis.cl.clCreateContextFromType
#define clRetainContext                                    cl3w_apis.cl.clRetainContext
#define clReleaseContext                                   cl3w_apis.cl.clReleaseContext
#define clGetContextInfo                                   cl3w_apis.cl.clGetContextInfo
#define clRetainCommandQueue                               cl3w_apis.cl.clRetainCommandQueue
#define clReleaseCommandQueue                              cl3w_apis.cl.clReleaseCommandQueue
#define clGetCommandQueueInfo                              cl3w_apis.cl.clGetCommandQueueInfo
#define clCreateBuffer                                     cl3w_apis.cl.clCreateBuffer
#define clCreateSubBuffer                                  cl3w_apis.cl.clCreateSubBuffer
#define clCreateImage                                      cl3w_apis.cl.clCreateImage
#define clRetainMemObject                                  cl3w_apis.cl.clRetainMemObject
#define clReleaseMemObject                                 cl3w_apis.cl.clReleaseMemObject
#define clGetSupportedImageFormats                         cl3w_apis.cl.clGetSupportedImageFormats
#define clGetMemObjectInfo                                 cl3w_apis.cl.clGetMemObjectInfo
#define clGetImageInfo                                     cl3w_apis.cl.clGetImageInfo
#define clSetMemObjectDestructorCallback                   cl3w_apis.cl.clSetMemObjectDestructorCallback
#define clRetainSampler                                    cl3w_apis.cl.clRetainSampler
#define clReleaseSampler                                   cl3w_apis.cl.clReleaseSampler
#define clGetSamplerInfo                                   cl3w_apis.cl.clGetSamplerInfo
#define clCreateProgramWithSource                          cl3w_apis.cl.clCreateProgramWithSource
#define clCreateProgramWithBinary                          cl3w_apis.cl.clCreateProgramWithBinary
#define clCreateProgramWithBuiltInKernels                  cl3w_apis.cl.clCreateProgramWithBuiltInKernels
#define clRetainProgram                                    cl3w_apis.cl.clRetainProgram
#define clReleaseProgram                                   cl3w_apis.cl.clReleaseProgram
#define clBuildProgram                                     cl3w_apis.cl.clBuildProgram
#define clCompileProgram                                   cl3w_apis.cl.clCompileProgram
#define clLinkProgram                                      cl3w_apis.cl.clLinkProgram
#define clUnloadPlatformCompiler                           cl3w_apis.cl.clUnloadPlatformCompiler
#define clGetProgramInfo                                   cl3w_apis.cl.clGetProgramInfo
#define clGetProgramBuildInfo                              cl3w_apis.cl.clGetProgramBuildInfo
#define clCreateKernel                                     cl3w_apis.cl.clCreateKernel
#define clCreateKernelsInProgram                           cl3w_apis.cl.clCreateKernelsInProgram
#define clRetainKernel                                     cl3w_apis.cl.clRetainKernel
#define clReleaseKernel                                    cl3w_apis.cl.clReleaseKernel
#define clSetKernelArg                                     cl3w_apis.cl.clSetKernelArg
#define clGetKernelInfo                                    cl3w_apis.cl.clGetKernelInfo
#define clGetKernelArgInfo                                 cl3w_apis.cl.clGetKernelArgInfo
#define clGetKernelWorkGroupInfo                           cl3w_apis.cl.clGetKernelWorkGroupInfo
#define clWaitForEvents                                    cl3w_apis.cl.clWaitForEvents
#define clGetEventInfo                                     cl3w_apis.cl.clGetEventInfo
#define clCreateUserEvent                                  cl3w_apis.cl.clCreateUserEvent
#define clRetainEvent                                      cl3w_apis.cl.clRetainEvent
#define clReleaseEvent                                     cl3w_apis.cl.clReleaseEvent
#define clSetUserEventStatus                               cl3w_apis.cl.clSetUserEventStatus
#define clSetEventCallback                                 cl3w_apis.cl.clSetEventCallback
#define clGetEventProfilingInfo                            cl3w_apis.cl.clGetEventProfilingInfo
#define clFlush                                            cl3w_apis.cl.clFlush
#define clFinish                                           cl3w_apis.cl.clFinish
#define clEnqueueReadBuffer                                cl3w_apis.cl.clEnqueueReadBuffer
#define clEnqueueReadBufferRect                            cl3w_apis.cl.clEnqueueReadBufferRect
#define clEnqueueWriteBuffer                               cl3w_apis.cl.clEnqueueWriteBuffer
#define clEnqueueWriteBufferRect                           cl3w_apis.cl.clEnqueueWriteBufferRect
#define clEnqueueFillBuffer                                cl3w_apis.cl.clEnqueueFillBuffer
#define clEnqueueCopyBuffer                                cl3w_apis.cl.clEnqueueCopyBuffer
#define clEnqueueCopyBufferRect                            cl3w_apis.cl.clEnqueueCopyBufferRect
#define clEnqueueReadImage                                 cl3w_apis.cl.clEnqueueReadImage
#define clEnqueueWriteImage                                cl3w_apis.cl.clEnqueueWriteImage
#define clEnqueueFillImage                                 cl3w_apis.cl.clEnqueueFillImage
#define clEnqueueCopyImage                                 cl3w_apis.cl.clEnqueueCopyImage
#define clEnqueueCopyImageToBuffer                         cl3w_apis.cl.clEnqueueCopyImageToBuffer
#define clEnqueueCopyBufferToImage                         cl3w_apis.cl.clEnqueueCopyBufferToImage
#define clEnqueueMapBuffer                                 cl3w_apis.cl.clEnqueueMapBuffer
#define clEnqueueMapImage                                  cl3w_apis.cl.clEnqueueMapImage
#define clEnqueueUnmapMemObject                            cl3w_apis.cl.clEnqueueUnmapMemObject
#define clEnqueueMigrateMemObjects                         cl3w_apis.cl.clEnqueueMigrateMemObjects
#define clEnqueueNDRangeKernel                             cl3w_apis.cl.clEnqueueNDRangeKernel
#define clEnqueueNativeKernel                              cl3w_apis.cl.clEnqueueNativeKernel
#define clEnqueueMarkerWithWaitList                        cl3w_apis.cl.clEnqueueMarkerWithWaitList
#define clEnqueueBarrierWithWaitList                       cl3w_apis.cl.clEnqueueBarrierWithWaitList
#define clGetExtensionFunctionAddressForPlatform           cl3w_apis.cl.clGetExtensionFunctionAddressForPlatform
#define clSetCommandQueueProperty                          cl3w_apis.cl.clSetCommandQueueProperty
#define clCreateImage2D                                    cl3w_apis.cl.clCreateImage2D
#define clCreateImage3D                                    cl3w_apis.cl.clCreateImage3D
#define clEnqueueMarker                                    cl3w_apis.cl.clEnqueueMarker
#define clEnqueueWaitForEvents                             cl3w_apis.cl.clEnqueueWaitForEvents
#define clEnqueueBarrier                                   cl3w_apis.cl.clEnqueueBarrier
#define clUnloadCompiler                                   cl3w_apis.cl.clUnloadCompiler
#define clGetExtensionFunctionAddress                      cl3w_apis.cl.clGetExtensionFunctionAddress
#define clCreateCommandQueue                               cl3w_apis.cl.clCreateCommandQueue
#define clCreateSampler                                    cl3w_apis.cl.clCreateSampler
#define clEnqueueTask                                      cl3w_apis.cl.clEnqueueTask
#endif

#ifdef __cplusplus
}
#endif
#endif

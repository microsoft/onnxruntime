/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define CUDA_LIB_NAME "cuda"

#if defined(_WIN32)
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif // defined(WIN32_LEAN_AND_MEAN)
#include <windows.h>
#define dllOpen(name) (void*) LoadLibraryA("nv" name ".dll")
#define dllClose(handle) FreeLibrary(static_cast<HMODULE>(handle))
#define dllGetSym(handle, name) GetProcAddress(static_cast<HMODULE>(handle), name)
#else
#include <dlfcn.h>
#define dllOpen(name) dlopen("lib" name ".so", RTLD_LAZY)
#define dllClose(handle) dlclose(handle)
#define dllGetSym(handle, name) dlsym(handle, name)
#endif

#include "cudaDriverWrapper.h"
// #include "plugin.h"
#include <cuda.h>
#include <stdio.h>

// using namespace nvinfer1;

CUDADriverWrapper::CUDADriverWrapper()
{
    handle = dllOpen(CUDA_LIB_NAME);
    // ASSERT(handle != nullptr); // TODO check
    
    auto load_sym = [](void* handle, const char* name) {
        void* ret = dllGetSym(handle, name);
        // ASSERT(ret != nullptr); // TODO check
        return ret;
    };

    *(void**) (&_cuGetErrorName) = load_sym(handle, "cuGetErrorName");
    *(void**) (&_cuFuncSetAttribute) = load_sym(handle, "cuFuncSetAttribute");
    *(void**) (&_cuLinkComplete) = load_sym(handle, "cuLinkComplete");
    *(void**) (&_cuModuleUnload) = load_sym(handle, "cuModuleUnload");
    *(void**) (&_cuLinkDestroy) = load_sym(handle, "cuLinkDestroy");
    *(void**) (&_cuModuleLoadData) = load_sym(handle, "cuModuleLoadData");
    *(void**) (&_cuLinkCreate) = load_sym(handle, "cuLinkCreate_v2");
    *(void**) (&_cuModuleGetFunction) = load_sym(handle, "cuModuleGetFunction");
    *(void**) (&_cuLinkAddFile) = load_sym(handle, "cuLinkAddFile_v2");
    *(void**) (&_cuLinkAddData) = load_sym(handle, "cuLinkAddData_v2");
    *(void**) (&_cuLaunchCooperativeKernel) = load_sym(handle, "cuLaunchCooperativeKernel");
    *(void**) (&_cuLaunchKernel) = load_sym(handle, "cuLaunchKernel");
}

CUDADriverWrapper::~CUDADriverWrapper()
{
    dllClose(handle);
}

CUresult CUDADriverWrapper::cuGetErrorName(CUresult error, const char** pStr) const
{
    return (*_cuGetErrorName)(error, pStr);
}

CUresult CUDADriverWrapper::cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value) const
{
    return (*_cuFuncSetAttribute)(hfunc, attrib, value);
}

CUresult CUDADriverWrapper::cuLinkComplete(CUlinkState state, void** cubinOut, size_t* sizeOut) const
{
    return (*_cuLinkComplete)(state, cubinOut, sizeOut);
}

CUresult CUDADriverWrapper::cuModuleUnload(CUmodule hmod) const
{
    return (*_cuModuleUnload)(hmod);
}

CUresult CUDADriverWrapper::cuLinkDestroy(CUlinkState state) const
{
    return (*_cuLinkDestroy)(state);
}

CUresult CUDADriverWrapper::cuModuleLoadData(CUmodule* module, const void* image) const
{
    return (*_cuModuleLoadData)(module, image);
}

CUresult CUDADriverWrapper::cuLinkCreate(
    unsigned int numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut) const
{
    return (*_cuLinkCreate)(numOptions, options, optionValues, stateOut);
}

CUresult CUDADriverWrapper::cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) const
{
    return (*_cuModuleGetFunction)(hfunc, hmod, name);
}

CUresult CUDADriverWrapper::cuLinkAddFile(CUlinkState state, CUjitInputType type, const char* path,
    unsigned int numOptions, CUjit_option* options, void** optionValues) const
{
    return (*_cuLinkAddFile)(state, type, path, numOptions, options, optionValues);
}

CUresult CUDADriverWrapper::cuLinkAddData(CUlinkState state, CUjitInputType type, void* data, size_t size,
    const char* name, unsigned int numOptions, CUjit_option* options, void** optionValues) const
{
    return (*_cuLinkAddData)(state, type, data, size, name, numOptions, options, optionValues);
}

CUresult CUDADriverWrapper::cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, CUstream hStream, void** kernelParams) const
{
    return (*_cuLaunchCooperativeKernel)(
        f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams);
}

CUresult CUDADriverWrapper::cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra) const
{
    return (*_cuLaunchKernel)(
        f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
}

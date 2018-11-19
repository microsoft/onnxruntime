// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
//define ONNX_RUNTIME_DLL_IMPORT if your program is dynamically linked to onnxruntime
//No dllexport here. Because we are using a def file
#ifdef _WIN32
#ifdef ONNX_RUNTIME_DLL_IMPORT
#define ONNX_RUNTIME_EXPORT __declspec(dllimport)
#else
#define ONNX_RUNTIME_EXPORT
#endif
#else
#define ONNX_RUNTIME_EXPORT
#endif

//SAL2 staffs
#ifndef _WIN32
#define _In_
#define _In_opt_
#define _Out_
#define _Out_opt_
#define _Inout_
#define _Inout_opt_
#define _Frees_ptr_opt_
#define ONNXRUNTIME_ALL_ARGS_NONNULL __attribute__((nonnull))
#else
#include <specstrings.h>
#define ONNXRUNTIME_ALL_ARGS_NONNULL
#endif

#ifdef _WIN32
#define ONNX_RUNTIME_MUST_USE_RESULT
#else
#define ONNX_RUNTIME_MUST_USE_RESULT __attribute__((warn_unused_result))
#endif
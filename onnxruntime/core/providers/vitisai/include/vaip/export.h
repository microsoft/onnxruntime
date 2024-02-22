// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#pragma once
#if defined(_WIN32)
#if VAIP_USE_DLL
#if VAIP_EXPORT_DLL == 1
#define VAIP_DLL_SPEC __declspec(dllexport)
#else
#define VAIP_DLL_SPEC __declspec(dllimport)
#endif
#else
#define VAIP_DLL_SPEC
#endif
#else
#define VAIP_DLL_SPEC __attribute__((visibility("default")))
#endif

#if defined(_WIN32)
#if VAIP_USE_DLL == 1
#define VAIP_PASS_ENTRY __declspec(dllexport)
#else
#define VAIP_PASS_ENTRY
#endif
#else
#define VAIP_PASS_ENTRY __attribute__((visibility("default")))
#endif

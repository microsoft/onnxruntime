// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Experimental C API function declarations (per-function plumbing).
//
// This is an internal detail header for onnxruntime_experimental_c_api.h. Do NOT include it directly (enforced via the
// #error below) — include onnxruntime_experimental_c_api.h instead. It must be included after onnxruntime_c_api.h (for
// ORT_API_CALL / NO_EXCEPTION) and after any auxiliary type declarations that the experimental functions reference
// (e.g. ORT_RUNTIME_CLASS(...)).
//
// It performs an X-macro pass over onnxruntime_experimental_c_api.inc to declare, for each experimental function, its
// C function pointer typedef and lookup name constant.

#pragma once

#ifndef ORT_INCLUDING_EXPERIMENTAL_C_API_FNS
#error "Include onnxruntime_experimental_c_api.h; do not include this detail header directly."
#endif

// For each ORT_EXPERIMENTAL_API(VER, RET, NAME, ...) entry in the .inc file, this generates:
//
//   // Function pointer typedef:
//   typedef RET(ORT_API_CALL* OrtExperimental_<NAME>_SinceV<VER>_Fn)(...) NO_EXCEPTION;
//
//   // Name constant for lookup:
//   static const char* const kOrtExperimental_<NAME>_SinceV<VER>_FnName = "<NAME>_SinceV<VER>";
//
// Example: ORT_EXPERIMENTAL_API(28, OrtStatusPtr, OrtApi_ExperimentalApiTest, _Out_ int64_t* out) produces:
//   typedef OrtStatusPtr(ORT_API_CALL* OrtExperimental_OrtApi_ExperimentalApiTest_SinceV28_Fn)(
//       _Out_ int64_t* out) NO_EXCEPTION;
//   static const char* const kOrtExperimental_OrtApi_ExperimentalApiTest_SinceV28_FnName =
//       "OrtApi_ExperimentalApiTest_SinceV28";
#define ORT_EXPERIMENTAL_API(VER, RET, NAME, ...)                                                 \
  typedef RET(ORT_API_CALL* OrtExperimental_##NAME##_SinceV##VER##_Fn)(__VA_ARGS__) NO_EXCEPTION; \
  static const char* const kOrtExperimental_##NAME##_SinceV##VER##_FnName = #NAME "_SinceV" #VER;

#include "onnxruntime_experimental_c_api.inc"

#undef ORT_EXPERIMENTAL_API

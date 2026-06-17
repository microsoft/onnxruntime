// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Experimental C API consumer header.
//
// This header provides C function pointer typedefs and name constants for experimental ORT functions.
// It should be used together with the experimental header lookup function `OrtApi::GetExperimentalFunction()`.
//
// For C++ consumers, the companion header onnxruntime_experimental_cxx_api.h provides typed inline accessors in the
// Ort::Experimental namespace (including throwing variants).
//
// The per-function typedefs and name constants are produced from onnxruntime_experimental_c_api.inc (which defines the
// list of experimental API functions) by the detail header onnxruntime_experimental_c_api_fns.h, which this header
// includes. Hand-written auxiliary declarations (e.g. opaque types the experimental APIs use) live directly in this
// header.
//
// IMPORTANT: Experimental functions are NOT part of the stable ABI. They may be added, changed, or removed between
// releases without notice. A function's availability should always be checked at runtime (the lookup returns nullptr
// if the function is not present).
//
// C usage:
//   OrtExperimental_OrtApi_ExperimentalApiTest_SinceV28_Fn fn =
//       (OrtExperimental_OrtApi_ExperimentalApiTest_SinceV28_Fn)api->GetExperimentalFunction(
//           kOrtExperimental_OrtApi_ExperimentalApiTest_SinceV28_FnName);
//   if (fn) {
//     OrtStatusPtr status = fn(&result);
//   }
//
// C++ usage (see onnxruntime_experimental_cxx_api.h):
//   if (auto* fn = Ort::Experimental::Get_OrtApi_ExperimentalApiTest_SinceV28_Fn(api)) {
//     Ort::Status status(fn(&result));
//   }

#pragma once

#include "onnxruntime_c_api.h"

//
// Auxiliary declarations
//
// Declarations of auxiliary types or typedefs required by experimental APIs go here.
//
// For example, if an experimental API uses a new type, OrtExperimentalType, we would declare it in this file:
//   ORT_RUNTIME_CLASS(ExperimentalType);
//

ORT_RUNTIME_CLASS(ModelPackageOptions);
ORT_RUNTIME_CLASS(ModelPackageContext);
ORT_RUNTIME_CLASS(ModelPackageComponentContext);

//
// C: function pointer typedefs and name constants
//
// Generated from onnxruntime_experimental_c_api.inc. Kept in a separate detail header so this header stays focused on
// the curated, hand-written surface (auxiliary declarations above, plus any custom helpers). The define/undef guard
// enforces that the detail header is only ever pulled in through this header.

#define ORT_INCLUDING_EXPERIMENTAL_C_API_FNS
#include "onnxruntime_experimental_c_api_fns.h"
#undef ORT_INCLUDING_EXPERIMENTAL_C_API_FNS

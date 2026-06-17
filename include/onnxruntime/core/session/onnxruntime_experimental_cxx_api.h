// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Experimental C++ API consumer header.
//
// This header provides typed inline accessors in the Ort::Experimental namespace for experimental ORT functions. It is
// the C++ companion to onnxruntime_experimental_c_api.h, which provides the underlying C function pointer typedefs and
// name constants.
//
// The per-function accessors are produced from onnxruntime_experimental_c_api.inc (which defines the list of
// experimental API functions) by the detail header onnxruntime_experimental_cxx_api_fns.h, which this header includes
// inside the Ort::Experimental namespace. Hand-written C++ helpers/wrappers live directly in this header.
//
// IMPORTANT: Experimental functions are NOT part of the stable ABI. They may be added, changed, or removed between
// releases without notice.
//
// Two accessor flavors are generated for each experimental function:
//
//   1. Get_<NAME>_SinceV<VER>_Fn(api)        -> returns the typed function pointer, or nullptr if the function is not
//                                               available in this build. Use this to check availability at runtime.
//
//   2. Get_<NAME>_SinceV<VER>_FnOrThrow(api) -> returns the typed function pointer, or throws Ort::Exception
//                                               (ORT_NOT_IMPLEMENTED) if the function is not available in this build.
//                                               Use this when the function is required.
//
// C++ usage (nullable):
//   if (auto* fn = Ort::Experimental::Get_OrtApi_ExperimentalApiTest_SinceV28_Fn(api)) {
//     Ort::Status status(fn(&result));
//   }
//
// C++ usage (throwing):
//   auto* fn = Ort::Experimental::Get_OrtApi_ExperimentalApiTest_SinceV28_FnOrThrow(api);
//   Ort::Status status(fn(&result));

#pragma once

#include "onnxruntime_experimental_c_api.h"
#include "onnxruntime_cxx_api.h"

//
// Typed accessors (nullable and throwing)
//
// Generated from onnxruntime_experimental_c_api.inc and declared in the Ort::Experimental namespace. Kept in a separate
// detail header so this header stays focused on the curated, hand-written surface. Custom C++ helpers can be added
// below in their own `namespace Ort { namespace Experimental { ... } }` block. The define/undef guard enforces that the
// detail header is only ever pulled in through this header.

#define ORT_INCLUDING_EXPERIMENTAL_CXX_API_FNS
#include "onnxruntime_experimental_cxx_api_fns.h"
#undef ORT_INCLUDING_EXPERIMENTAL_CXX_API_FNS

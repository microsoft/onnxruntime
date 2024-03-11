// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnxruntime_config.h"

#if defined(__GNUC__)
#pragma GCC diagnostic push

// Disable warning from protobuf code.
//
// In file included from coreml_proto/Model.pb.h:30:
// In file included from _deps/protobuf-src/src/google/protobuf/extension_set.h:53:
// _deps/protobuf-src/src/google/protobuf/parse_context.h:328:47:
//     error: implicit conversion loses integer precision: 'long' to 'int' [-Werror,-Wshorten-64-to-32]
#ifdef HAS_SHORTEN_64_TO_32
#pragma GCC diagnostic ignored "-Wshorten-64-to-32"
#endif
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4244)  // conversion from long to int
#endif

// Model.pb.h is generated in the build output directory from the CoreML protobuf files in
// <build output directory>/_deps/coremltools-src/mlmodel/format
#include "coreml_proto/Model.pb.h"

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace COREML_SPEC = CoreML::Specification;

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// Disable warning from protobuf code.
//
// In file included from coreml_proto/Model.pb.h:30:
// In file included from _deps/protobuf-src/src/google/protobuf/extension_set.h:53:
// _deps/protobuf-src/src/google/protobuf/parse_context.h:328:47:
//     error: implicit conversion loses integer precision: 'long' to 'int' [-Werror,-Wshorten-64-to-32]

#if defined(__GNUC__)
#pragma GCC diagnostic push
#if defined(__has_warning)
#if __has_warning("-Wshorten-64-to-32")
#pragma GCC diagnostic ignored "-Wshorten-64-to-32"
#endif
#endif  // defined(__has_warning)
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4244)
#endif

// Model.pb.h is generated in the build output directory from the CoreML protobuf files in
// onnxruntime/core/providers/coreml/coremltools/mlmodel/format
#include "coreml_proto/Model.pb.h"

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace COREML_SPEC = CoreML::Specification;

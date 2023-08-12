// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/js/js_kernel.h"

#include "gather.h"

namespace onnxruntime {
namespace js {

using AllSupportedSize =
    TypeList<
        float,
        double,
        int64_t,
        uint64_t,
        int32_t,
        uint32_t>;

#define REGISTER_GATHER_VERSIONED_KERNEL(GatherOp, sinceVersion, endVersion)                            \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                                                    \
      GatherOp,                                                                                         \
      kOnnxDomain,                                                                                      \
      sinceVersion, endVersion,                                                                         \
      kJsExecutionProvider,                                                                             \
      KernelDefBuilder()                                                                                \
          .TypeConstraint("T", BuildKernelDefConstraintsFromTypeList<AllSupportedSize>())               \
          .TypeConstraint("Tind", BuildKernelDefConstraintsFromTypeList<TypeList<int32_t, int64_t>>()), \
      GatherOp);

#define REGISTER_GATHER_KERNEL(GatherOp, sinceVersion)                                                  \
  ONNX_OPERATOR_KERNEL_EX(                                                                              \
      GatherOp,                                                                                         \
      kOnnxDomain,                                                                                      \
      sinceVersion,                                                                                     \
      kJsExecutionProvider,                                                                             \
      KernelDefBuilder()                                                                                \
          .TypeConstraint("T", BuildKernelDefConstraintsFromTypeList<AllSupportedSize>())               \
          .TypeConstraint("Tind", BuildKernelDefConstraintsFromTypeList<TypeList<int32_t, int64_t>>()), \
      GatherOp);

REGISTER_GATHER_VERSIONED_KERNEL(Gather, 1, 10);
REGISTER_GATHER_VERSIONED_KERNEL(Gather, 11, 12);
REGISTER_GATHER_KERNEL(Gather, 13);

REGISTER_GATHER_VERSIONED_KERNEL(GatherElements, 11, 12);
REGISTER_GATHER_KERNEL(GatherElements, 13);

REGISTER_GATHER_VERSIONED_KERNEL(GatherND, 11, 11);
REGISTER_GATHER_VERSIONED_KERNEL(GatherND, 12, 12);
REGISTER_GATHER_KERNEL(GatherND, 13);
}  // namespace js
}  // namespace onnxruntime

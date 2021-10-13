// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/optional.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/reduction/reduction_ops.h"
#include "core/providers/cuda/reduction/reduction_functions.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/dlpack/dlpack_converter.h"
#include "orttraining/training_ops/cpu/aten_ops/aten_op.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
class ReduceSumTraining final : public ReduceKernel<true> {
 public:
  ReduceSumTraining(const OpKernelInfo& info) : ReduceKernel<true>(info) {
    fast_reduction_ = true;
  }

  Status ComputeInternal(OpKernelContext* ctx) const override {
#if 0
    return ComputeImplEx<T>(ctx, CUDNN_REDUCE_TENSOR_ADD);
#else
    auto* p_ctx_internal = static_cast<OpKernelContextInternal*>(p_ctx);
    std::vector<DLManagedTensor*> dlpacks;
    for (int i = 0; i < p_ctx_internal->InputCount(); i++) {
      const OrtValue* p_ort_value = p_ctx_internal->GetInputMLValue(i);
      if (!p_ort_value) {
        dlpacks.emplace_back(nullptr);
      } else {
        OrtValue ort_value = *p_ort_value;
        dlpacks.emplace_back(dlpack::OrtValueToDlpack(ort_value));
      }
    }

    auto result = aten_ops::ATenOperatorExecutor::Instance()("ReduceSum", "", dlpacks);
    for (size_t i = 0; i < result.size(); i++) {
      ORT_RETURN_IF_ERROR(p_ctx_internal->SetOutputMLValue(static_cast<int>(i), dlpack::DlpackToOrtValue(result[i])));
    }

    return Status::OK();
#endif
  }
};

}  // namespace cuda
}  // namespace onnxruntime

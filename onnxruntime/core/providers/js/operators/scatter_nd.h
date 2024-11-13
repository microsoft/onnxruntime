// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/js/js_kernel.h"
#include "core/framework/data_transfer_manager.h"

namespace onnxruntime {
namespace js {

enum class ScatterNDReduction : int {
  None = 0,
  Add = 1,
  Mul = 2,
  Min = 3,
  Max = 4,
};

class ScatterND : public JsKernel {
 public:
  ScatterND(const OpKernelInfo& info) : JsKernel(info) {
    std::string reduction = info.GetAttrOrDefault<std::string>("reduction", "none");
    if (reduction == "add") {
      reduction_ = ScatterNDReduction::Add;
    } else if (reduction == "mul") {
      reduction_ = ScatterNDReduction::Mul;
    } else if (reduction == "min") {
      reduction_ = ScatterNDReduction::Min;
    } else if (reduction == "max") {
      reduction_ = ScatterNDReduction::Max;
    } else if (reduction == "none") {
      LOGS_DEFAULT(WARNING) << "ScatterND with reduction=='none' only guarantees "
                            << "to be correct if indices are not duplicated.";
    } else {
      ORT_THROW("Reduction '", reduction, "' is not supported on webgpu when opset <= 13.");
    }

    JSEP_INIT_KERNEL_ATTRIBUTE(ScatterND, ({
                                 "reduction" : UTF8ToString($1),
                               }),
                               reduction.c_str());
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* X = context->Input<Tensor>(0);
    if (X == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
    const TensorShape& X_shape = X->Shape();

    Tensor* Y = context->Output(0, X_shape);
    const void* source = X->DataRaw();
    void* target = Y->MutableDataRaw();
    // If source and target pointers are not equal (non-inplace operation), we need to copy the data.
    if (target != source) {
      ORT_RETURN_IF_ERROR(Info().GetDataTransferManager().CopyTensor(*X, *Y));
    }
    return ComputeInternal(context);
  }

 private:
  ScatterNDReduction reduction_{ScatterNDReduction::None};
};

}  // namespace js
}  // namespace onnxruntime

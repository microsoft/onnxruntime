// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/pool_base.h"

namespace onnxruntime {
namespace contrib {

class QLinearAveragePool final : public OpKernel, public PoolBase {
 public:
  QLinearAveragePool(const OpKernelInfo& info) : OpKernel(info), PoolBase(info) {
    channels_last_ = (info.GetAttrOrDefault<int64_t>("channels_last", static_cast<int64_t>(0)) != 0);

    int32_t input_type = info.node().InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
    is_input_signed_ = ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8 == input_type;
  }

  ~QLinearAveragePool() override = default;

  Status Compute(OpKernelContext* context) const override;

 private:
  template <typename T8Bits>
  Status ComputeImpl(OpKernelContext* context) const;

  PoolProcessContext pool_context_;
  bool channels_last_;
  bool is_input_signed_;
};

}  // namespace contrib
}  // namespace onnxruntime

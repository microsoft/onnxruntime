// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/rocm_kernel.h"
#include "core/providers/rocm/miopen_common.h"
#include "core/providers/rocm/nn/conv.h"
#include "core/providers/cpu/nn/conv_transpose_attributes.h"

namespace onnxruntime {
namespace rocm {

template <typename T>
class ConvTranspose : public RocmKernel {
 public:
  ConvTranspose(const OpKernelInfo& info) : RocmKernel(info), conv_transpose_attrs_(info){};
  Status ComputeInternal(OpKernelContext* context) const override;
  Status DoConvTranspose(OpKernelContext* context, bool dynamic_padding) const;

 private:
  ConvTransposeAttributes conv_transpose_attrs_;

  mutable MiopenConvState<miopenConvAlgoPerf_t> s_;
};

}  // namespace rocm
}  // namespace onnxruntime

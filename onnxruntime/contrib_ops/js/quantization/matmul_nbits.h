// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace js {

using onnxruntime::js::JsKernel;

class MatMulNBits final : public JsKernel {
 public:
  MatMulNBits(const OpKernelInfo& info) : JsKernel(info),
                                          K_{narrow<size_t>(info.GetAttr<int64_t>("K"))},
                                          N_{narrow<size_t>(info.GetAttr<int64_t>("N"))},
                                          accuracy_level_{info.GetAttrOrDefault<int64_t>("accuracy_level", 0)},
                                          nbits_{narrow<size_t>(info.GetAttr<int64_t>("bits"))},
                                          block_size_{narrow<size_t>(info.GetAttr<int64_t>("block_size"))} {
    ORT_ENFORCE(nbits_ == 4,
                "Only 4b quantization is supported for MatMulNBits op, additional bits support is planned.");
    ORT_ENFORCE(block_size_ >= 16 && !(block_size_ & (block_size_ - 1)),
                "Block size must be a power of 2 and greater than or equal to 16.");
    JSEP_INIT_KERNEL_ATTRIBUTE(MatMulNBits, ({
                                 "k" : $1,
                                 "n" : $2,
                                 "accuracyLevel" : $3,
                                 "bits" : $4,
                                 "blockSize" : $5
                               }),
                               static_cast<int32_t>(K_),
                               static_cast<int32_t>(N_),
                               static_cast<int32_t>(accuracy_level_),
                               static_cast<int32_t>(nbits_),
                               static_cast<int32_t>(block_size_));
  }

 private:
  const size_t K_;
  const size_t N_;
  const int64_t accuracy_level_;
  const size_t nbits_;
  const size_t block_size_;
};

}  // namespace js
}  // namespace contrib
}  // namespace onnxruntime

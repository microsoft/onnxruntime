// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/tensor.h"
#include "core/providers/cpu/tensor/split.h"
#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace js {

class Split : public JsKernel, public SplitBase {
 public:
  Split(const OpKernelInfo& info, uint32_t opset) : JsKernel(info), SplitBase(info, opset) {
    const bool is_uneven_split_allowed = num_outputs_ >= 0;
    std::vector<int32_t> split_sizes;
    if (split_sizes_.size() > 0) {
      ORT_ENFORCE(split_sizes_.size() == info.node().OutputDefs().size(),
                  "Number of outputs (", info.node().OutputDefs().size(), ") does not match split_sizes (",
                  split_sizes_.size(), ")");
      split_sizes.resize(split_sizes_.size());
      for (size_t i = 0; i < split_sizes_.size(); ++i) {
        split_sizes[i] = gsl::narrow_cast<int32_t>(split_sizes_[i]);
      }
      if (num_outputs_ < 0) {
        num_outputs_ = split_sizes.size();
      }
    } else {
      if (num_outputs_ < 0) {
        num_outputs_ = info.node().OutputDefs().size();
      } else {
        ORT_ENFORCE(num_outputs_ == info.node().OutputDefs().size(),
                    "Number of outputs (", info.node().OutputDefs().size(), ") does not match num_outputs (",
                    num_outputs_, ")");
      }
    }

    JSEP_INIT_KERNEL_ATTRIBUTE(Split, ({"axis" : $1,
                                        "numOutputs" : $2,
                                        "splitSizes" : $3 ? Array.from(HEAP32.subarray(Number($3), Number($4))) : [],
                                        "isUnevenSplitAllowed" : !!$5}),
                               static_cast<int32_t>(axis_),
                               static_cast<int32_t>(num_outputs_),
                               JSEP_HEAP32_INDEX_START(split_sizes),
                               JSEP_HEAP32_INDEX_END(split_sizes),
                               static_cast<int32_t>(is_uneven_split_allowed));
  }
};

class Split_1 final : public Split {
 public:
  Split_1(const OpKernelInfo& info) : Split(info, 1) {}
};

class Split_2_10 final : public Split {
 public:
  Split_2_10(const OpKernelInfo& info) : Split(info, 2) {}
};

class Split_11_12 final : public Split {
 public:
  Split_11_12(const OpKernelInfo& info) : Split(info, 11) {}
};

class Split_13_17 final : public Split {
 public:
  Split_13_17(const OpKernelInfo& info) : Split(info, 13) {}
};

class Split_18 final : public Split {
 public:
  Split_18(const OpKernelInfo& info) : Split(info, 18) {}
};

}  // namespace js
}  // namespace onnxruntime

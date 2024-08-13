// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/js/js_data_types.h"
#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace js {

using onnxruntime::js::JsKernel;

class GatherBlockQuantized : public JsKernel {
 public:
  GatherBlockQuantized(const OpKernelInfo& info) : JsKernel(info) {
    int64_t gather_axis;
    int64_t quantize_axis;
    int64_t block_size;
    if (!info.GetAttr<int64_t>("gather_axis", &gather_axis).IsOK()) {
      gather_axis = 0;
    }

    if (!info.GetAttr<int64_t>("quantize_axis", &quantize_axis).IsOK()) {
      quantize_axis = 1;
    }

    if (!info.GetAttr<int64_t>("block_size", &block_size).IsOK()) {
      block_size = 128;
    }

    ORT_ENFORCE(block_size >= 16 && ((block_size - 1) & block_size) == 0,
                "'block_size' must be 2's power and not less than 16.");
    JSEP_INIT_KERNEL_ATTRIBUTE(GatherBlockQuantized, ({
                                 "gatherAxis" : $1,
                                 "quantizeAxis" : $2,
                                 "blockSize" : $3
                               }),
                               static_cast<int32_t>(gather_axis),
                               static_cast<int32_t>(quantize_axis),
                               static_cast<int32_t>(block_size));
  }
};

}  // namespace js
}  // namespace contrib
}  // namespace onnxruntime

/**
 * Copyright (c) 2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/

#pragma once
#include "core/common/status.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {

class HailoMemcpy final : public OpKernel {
public:
    HailoMemcpy(const OpKernelInfo& info);

    Status Compute(OpKernelContext* context) const override;
};

}  // namespace onnxruntime
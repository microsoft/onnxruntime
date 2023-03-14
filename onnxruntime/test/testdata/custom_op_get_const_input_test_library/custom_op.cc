// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "custom_op.h"
#include "common.h"

TestCustomKernel::TestCustomKernel(const OrtKernelInfo* info)
{
    Ort::ConstKernelInfo kinfo(info);
    int is_constant = 0;

    // Get weights (constant inputs) from kernel info
    for (size_t i = 0; i < kinfo.GetInputCount(); i++) {
        Ort::ConstValue const_input = kinfo.GetTensorConstantInput(i, &is_constant);
        if (is_constant) {
            const float* value  = const_input.GetTensorData<float>();
            ORT_ENFORCE(value[0] == 1.0);
            ORT_ENFORCE(value[1] == 2.0);
            ORT_ENFORCE(value[2] == 3.0);
            ORT_ENFORCE(value[3] == 4.0);
        }
    }
    ORT_ENFORCE(is_constant == 1);
}

void TestCustomKernel::Compute(OrtKernelContext* context) {}

//
// TestCustomOp 
//

TestCustomOp::TestCustomOp() {}

void* TestCustomOp::CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const { 
    return new TestCustomKernel(info);
}

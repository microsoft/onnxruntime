// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

#include "armnn/ArmNN.hpp"


namespace onnxruntime {
namespace armnn_ep {

armnn::TensorShape ArmNNTensorShape(const TensorShape& tensorShape, unsigned int extDim = 0);

}  // namespace armnn_ep
}  // namespace onnxruntime

/**
* Copyright (c) 2016-present, Facebook, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
/* Modifications Copyright (c) Microsoft. */

#include "core/providers/cpu/nn/batch_norm.h"
#include "core/providers/cpu/nn/batch_norm_helper.h"

namespace onnxruntime {
// spec: https://github.com/onnx/onnx/blob/master/docs/Operators.md#BatchNormalization

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(BatchNormalization, 7, 8, float,
                                         KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                                         BatchNorm<float>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(BatchNormalization, 7, 8, double,
                                         KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
                                         BatchNorm<double>);

// We alias the running mean to the mean so it stays preserved across multiple batches
ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(BatchNormalization, 9, 13, float,
                               KernelDefBuilder().Alias(3,1).Alias(4,2).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                               BatchNorm<float>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(BatchNormalization, 9, 13, double,
                               KernelDefBuilder().Alias(3,1).Alias(4,2).TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
                               BatchNorm<double>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(BatchNormalization, 14, float,
                               KernelDefBuilder().Alias(3,1).Alias(4,2).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                               BatchNorm<float>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(BatchNormalization, 14, double,
                               KernelDefBuilder().Alias(3,1).Alias(4,2).TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
                               BatchNorm<double>);
}  // namespace onnxruntime

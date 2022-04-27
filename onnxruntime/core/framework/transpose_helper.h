// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <sstream>

#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/framework/tensor_shape.h"
#include "core/framework/tensor.h"

#include "gsl/gsl"

namespace onnxruntime {
bool IsMovingSingleAxis(const gsl::span<const size_t>& permutations, size_t& from, size_t& to);
void SingleAxisTranspose(const gsl::span<const size_t>& permutations, const Tensor& input, Tensor& output, size_t from,
                         size_t to, const TensorShape* input_shape_override = nullptr);
}  // namespace onnxruntime
// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License

#pragma once

#include "dnnl.hpp"
#include <string>

namespace onnxruntime {
namespace ort_dnnl {
namespace dnnl_util {
bool IsGPURuntimeAvalible();

bool IsBF16Supported();

dnnl::algorithm OrtOperatorToDnnlAlgorithm(std::string op);

}  // namespace dnnl_util
}  // namespace ort_dnnl
}  // namespace onnxruntime

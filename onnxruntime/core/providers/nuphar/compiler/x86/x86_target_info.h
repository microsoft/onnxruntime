// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/codegen/common/target_info.h"

namespace onnxruntime {

class CodeGenTargetX86 final : public CodeGenTarget {
  int max_vector_bits_;
  int vector_unit_num_;

 public:
  CodeGenTargetX86(const std::string& target_name, int max_vector_bits, int vector_unit_num)
      : CodeGenTarget(target_name), max_vector_bits_(max_vector_bits), vector_unit_num_(vector_unit_num) {}

  int NaturalVectorWidth(int bits) const override {
    return max_vector_bits_ * vector_unit_num_ / bits;
  }

  ~CodeGenTargetX86() override = default;
};

std::unique_ptr<CodeGenTarget> CodeGenTarget_AVX2();

std::unique_ptr<CodeGenTarget> CodeGenTarget_AVX512();

}  // namespace onnxruntime

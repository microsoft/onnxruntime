// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

#include <memory>

namespace onnxruntime {
namespace contrib {
class Tokenizer final : public OpKernel {
 public:
  explicit Tokenizer(const OpKernelInfo& info);
  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;
  ~Tokenizer();

  Status Compute(OpKernelContext* context) const override;

 private:
  Status CharTokenize(OpKernelContext* context, size_t N, size_t C,
                      const std::vector<int64_t>& input_dims) const;

  Status SeparatorTokenize(OpKernelContext* context, size_t N, size_t C,
                           const std::vector<int64_t>& input_dims) const;

  bool mark_;
  std::string pad_value_;
  int64_t mincharnum_;
  bool char_tokenezation_;
  struct SearchData;
  std::unique_ptr<SearchData> search_data_;
};
}  // namespace contrib
}  // namespace onnxruntime

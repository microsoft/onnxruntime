// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "gsl/gsl_util"
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include <sstream>

namespace onnxruntime {

class TransposeBase {
 public:
  /**
  Transpose the input Tensor into the output Tensor using the provided permutations.
  Both Tensors must have the same data type. 
  */
  static Status DoTranspose(const std::vector<int64_t>& permutations, const Tensor& input, Tensor& output);

 protected:
  TransposeBase(const OpKernelInfo& info) {
    Status status = info.GetAttrs<int64_t>("perm", perm_);

    if (status.IsOK()) {
      perm_specified_ = true;
      size_t rank = perm_.size();
      std::vector<bool> seen(rank, false);
      // Check that perm_ is a valid permutation of [0,rank-1]
      for (auto i : perm_) {
        if ((i < 0) || (i >= gsl::narrow<int64_t>(rank)))
          ORT_THROW("Attribute perm of Transpose has an invalid value. Value ", i, " is outside range.");
        if (seen[i])
          ORT_THROW("Attribute perm of Transpose has an invalid value. Value ", i, " is repeated.");
        seen[i] = true;
      }
    }
  }

  Status ComputeOutputShape(const Tensor& X, std::vector<int64_t>& output_dims,
                          std::vector<int64_t>& default_perm, const std::vector<int64_t>*& p_perm) const {
    size_t rank = X.Shape().NumDimensions();
    const auto& input_dims = X.Shape().GetDims();

    // Determine permutation to use:
    // If no permutation was specified in the attributes, the default is [rank-1, ..., 0]
    default_perm.resize(rank);

    if (perm_specified_)
      p_perm = &perm_;
    else {
      for (int i = 0; i < rank; ++i)
        default_perm[i] = rank - i - 1;
      p_perm = &default_perm;
    }

    // Determine shape of output
    output_dims.resize(rank);
    for (int i = 0; i < rank; i++) {
      size_t inpdim = (*p_perm)[i];
      if (inpdim >= rank) {
        std::ostringstream ss;
        ss << "[ ";
        for (const auto& p : *p_perm)
          ss << p << " ";
        ss << "]";
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "perm: ", ss.str(), " does not align with rank of input data: ", std::to_string(rank));
      }
      output_dims[i] = input_dims[inpdim];
    }
    return Status::OK();
  }

  bool perm_specified_ = false;
  std::vector<int64_t> perm_;
};

class Transpose final : public OpKernel, public TransposeBase {
 public:
  Transpose(const OpKernelInfo& info) : OpKernel(info), TransposeBase(info) {}

  Status Compute(OpKernelContext* context) const override;
};
}  // namespace onnxruntime

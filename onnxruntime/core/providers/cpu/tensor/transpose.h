// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifndef SHARED_PROVIDER
#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/framework/tensor_shape.h"
#include "core/framework/op_kernel.h"
#endif

#include "core/common/gsl.h"
#include <sstream>

namespace onnxruntime {

/** Tells if the transpose is equivalent to a reshape:
 empty dimensions can change place, not empty dimensions must be in
 the same order in the permuted tenosr.
*/
bool IsTransposeReshape(const gsl::span<const size_t>& perm, gsl::span<const int64_t> input_dims);

// Public function for element-wise transpose, primarily to unit test any out of bounds access
Status DoTransposeEltWise(int64_t num_axes, gsl::span<const int64_t> target_dims, size_t num_blocks,
                          const gsl::span<const size_t>& stride, const uint8_t* source, uint8_t* target,
                          size_t element_size);

class TransposeBase {
 public:
  /**
  Transpose the input Tensor into the output Tensor using the provided permutations.
  Both Tensors must have the same data type. `input_shape_override` overrides the shape of `input` for compute purposes.
  */
  static Status DoTranspose(const gsl::span<const size_t>& permutations, const Tensor& input, Tensor& output,
                            const TensorShape* input_shape_override = nullptr);

 protected:
  TransposeBase(const OpKernelInfo& info) {
    std::vector<int64_t> temp_perm;
    Status status = info.GetAttrs<int64_t>("perm", temp_perm);
    if (status.IsOK()) {
      size_t rank = temp_perm.size();
      perm_.resize(temp_perm.size());
      // Check that perm_ is a valid permutation of [0,rank-1]
      for (size_t i = 0; i != temp_perm.size(); ++i) {
        int64_t v = temp_perm[i];
        ORT_ENFORCE(v >= 0 && static_cast<uint64_t>(v) <= std::numeric_limits<size_t>::max());
        if (static_cast<size_t>(v) >= rank)
          ORT_THROW("Attribute perm of Transpose has an invalid value. Value ", i, " is outside range.");
        perm_[i] = static_cast<size_t>(v);
      }
      perm_specified_ = true;
      std::vector<bool> seen(rank, false);
      for (auto i : perm_) {
        if (seen[i])
          ORT_THROW("Attribute perm of Transpose has an invalid value. Value ", i, " is repeated.");
        seen[i] = true;
      }
    }
  }

  Status ComputeOutputShape(const Tensor& X, TensorShapeVector& output_dims, InlinedVector<size_t>& default_perm,
                            const InlinedVector<size_t>*& p_perm) const {
    const size_t rank = X.Shape().NumDimensions();
    const auto& input_dims = X.Shape().GetDims();

    if (perm_specified_)
      p_perm = &perm_;
    else {
      // Determine permutation to use:
      // If no permutation was specified in the attributes, the default is [rank-1, ..., 0]
      default_perm.resize(rank);
      for (size_t i = 0; i < rank; ++i) default_perm[i] = rank - i - 1;
      p_perm = &default_perm;
    }

    if (p_perm->size() != rank) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "perm size: ", p_perm->size(), " does not match input rank: ", std::to_string(rank));
    }

    // Determine shape of output
    output_dims.resize(rank);
    for (size_t i = 0; i < rank; i++) {
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
  InlinedVector<size_t> perm_;
};

class Transpose final : public OpKernel, public TransposeBase {
 public:
  Transpose(const OpKernelInfo& info) : OpKernel(info), TransposeBase(info) {}

  Status Compute(OpKernelContext* context) const override;
};
}  // namespace onnxruntime

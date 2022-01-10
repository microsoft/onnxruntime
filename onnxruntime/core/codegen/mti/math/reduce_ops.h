// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <tvm/te/operation.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::te::Tensor ArgMax(const tvm::te::Tensor& X,
                   int64_t axis,
                   bool keep_dims,
                   const std::string& name = "argmax");

tvm::te::Tensor ArgMin(const tvm::te::Tensor& X,
                   int64_t axis,
                   bool keep_dims,
                   const std::string& name = "argmin");

tvm::te::Tensor ReduceL1(const tvm::te::Tensor& X,
                     const std::vector<int64_t>& axes,
                     bool keep_dims,
                     const std::string& name = "reduce_l1");

tvm::te::Tensor ReduceL2(const tvm::te::Tensor& X,
                     const std::vector<int64_t>& axes,
                     bool keep_dims,
                     const std::string& name = "reduce_l2");

tvm::te::Tensor ReduceLogSum(const tvm::te::Tensor& X,
                         const std::vector<int64_t>& axes,
                         bool keep_dims,
                         const std::string& name = "reduce_log_sum");

tvm::te::Tensor ReduceLogSumExp(const tvm::te::Tensor& X,
                            const std::vector<int64_t>& axes,
                            bool keep_dims,
                            const std::string& name = "argmareduce_log_sum_exp");

tvm::te::Tensor ReduceMax(const tvm::te::Tensor& X,
                      const std::vector<int64_t>& axes,
                      bool keep_dims,
                      const std::string& name = "reduce_max");

tvm::te::Tensor ReduceMean(const tvm::te::Tensor& X,
                       const std::vector<int64_t>& axes,
                       bool keep_dims,
                       const std::string& name = "reduce_mean");

tvm::te::Tensor ReduceMin(const tvm::te::Tensor& X,
                      const std::vector<int64_t>& axes,
                      bool keep_dims,
                      const std::string& name = "reduce_min");

tvm::te::Tensor ReduceProd(const tvm::te::Tensor& X,
                       const std::vector<int64_t>& axes,
                       bool keep_dims,
                       const std::string& name = "reduce_prod");

tvm::te::Tensor ReduceSum(const tvm::te::Tensor& X,
                      const std::vector<int64_t>& axes,
                      bool keep_dims,
                      const std::string& name = "reduce_sum");

tvm::te::Tensor ReduceSumSquare(const tvm::te::Tensor& X,
                            const std::vector<int64_t>& axes,
                            bool keep_dims,
                            const std::string& name = "reduce_sum_square");

}  // namespace tvm_codegen
}  // namespace onnxruntime

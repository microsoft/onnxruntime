// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <tvm/tvm.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::Tensor ArgMax(const tvm::Tensor& X,
                   int64_t axis,
                   bool keep_dims,
                   const std::string& name = "argmax");

tvm::Tensor ArgMin(const tvm::Tensor& X,
                   int64_t axis,
                   bool keep_dims,
                   const std::string& name = "argmin");

tvm::Tensor ReduceL1(const tvm::Tensor& X,
                     const std::vector<int64_t>& axes,
                     bool keep_dims,
                     const std::string& name = "reduce_l1");

tvm::Tensor ReduceL2(const tvm::Tensor& X,
                     const std::vector<int64_t>& axes,
                     bool keep_dims,
                     const std::string& name = "reduce_l2");

tvm::Tensor ReduceLogSum(const tvm::Tensor& X,
                         const std::vector<int64_t>& axes,
                         bool keep_dims,
                         const std::string& name = "reduce_log_sum");

tvm::Tensor ReduceLogSumExp(const tvm::Tensor& X,
                            const std::vector<int64_t>& axes,
                            bool keep_dims,
                            const std::string& name = "argmareduce_log_sum_exp");

tvm::Tensor ReduceMax(const tvm::Tensor& X,
                      const std::vector<int64_t>& axes,
                      bool keep_dims,
                      const std::string& name = "reduce_max");

tvm::Tensor ReduceMean(const tvm::Tensor& X,
                       const std::vector<int64_t>& axes,
                       bool keep_dims,
                       const std::string& name = "reduce_mean");

tvm::Tensor ReduceMin(const tvm::Tensor& X,
                      const std::vector<int64_t>& axes,
                      bool keep_dims,
                      const std::string& name = "reduce_min");

tvm::Tensor ReduceProd(const tvm::Tensor& X,
                       const std::vector<int64_t>& axes,
                       bool keep_dims,
                       const std::string& name = "reduce_prod");

tvm::Tensor ReduceSum(const tvm::Tensor& X,
                      const std::vector<int64_t>& axes,
                      bool keep_dims,
                      const std::string& name = "reduce_sum");

tvm::Tensor ReduceSumSquare(const tvm::Tensor& X,
                            const std::vector<int64_t>& axes,
                            bool keep_dims,
                            const std::string& name = "reduce_sum_square");

}  // namespace tvm_codegen
}  // namespace onnxruntime

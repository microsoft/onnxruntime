// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <tvm/tvm.h>

namespace onnxruntime {
namespace nuphar {

constexpr auto kNupharVReduce = "nuphar_v_reduce";

constexpr auto kNupharVReduceFuseDim = "nuphar_v_reduce_fuse_dim";

tvm::Tensor ReduceSum(const tvm::Tensor& X,
                      const std::vector<int64_t>& axes,
                      bool keep_dims,
                      const int32_t vector_size,
                      bool last_dim_aligned = false,
                      int32_t fuse_dim = 0,
                      const std::string& name = "reduce_sum_v");

tvm::Tensor ReduceMax(const tvm::Tensor& X,
                      const std::vector<int64_t>& axes,
                      bool keep_dims,
                      const int32_t vector_size,
                      bool last_dim_aligned = false,
                      int32_t fuse_dim = 0,
                      const std::string& name = "reduce_max_v");

tvm::Tensor ReduceMin(const tvm::Tensor& X,
                      const std::vector<int64_t>& axes, bool keep_dims,
                      const int32_t vector_size,
                      bool last_dim_aligned = false,
                      int32_t fuse_dim = 0,
                      const std::string& name = "reduce_min_v");

tvm::Tensor ReduceMean(const tvm::Tensor& X,
                       const std::vector<int64_t>& axes, bool keep_dims,
                       const int32_t vector_size,
                       bool last_dim_aligned = false,
                       int32_t fuse_dim = 0,
                       const std::string& name = "reduce_mean_v");

}  // namespace nuphar
}  // namespace onnxruntime

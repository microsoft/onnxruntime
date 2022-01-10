// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/mti/nn/pool_ops.h"

#include "core/codegen/mti/mti_tvm_utils.h"
#include "core/mlas/inc/mlas.h"
#include "core/providers/cpu/nn/pool_attributes.h"
#include <tvm/topi/nn/pooling.h>

namespace onnxruntime {
namespace tvm_codegen {

// TODO: topi only support 2d-pool, MaxPool1d and MaxPool3d will need to be added if necessary.
// only support version < 8 for topi doesn't come with implementation to output index tensor
tvm::te::Tensor MaxPool(const tvm::te::Tensor& input,
                    const PoolAttributes& pool_attrs,
                    const tvm::Array<tvm::PrimExpr>& /*output_shape*/,
                    const std::string& /*name*/) {
  return topi::nn::pool(input,
                        ToTvmArray(pool_attrs.kernel_shape),
                        ToTvmArray(pool_attrs.strides),
                        ToTvmArray(pool_attrs.pads),
                        /*pool_type*/ tvm::topi::nn::kMaxPool,
                        /*ceil_mode*/ false,
                        /*layout*/ pool_attrs.storage_order == 0 ? "NCWH" : "NCHW",
                        pool_attrs.count_include_pad);
}

tvm::te::Tensor AveragePool(const tvm::te::Tensor& input,
                        const PoolAttributes& pool_attrs,
                        const tvm::Array<tvm::PrimExpr>& /*output_shape*/,
                        const std::string& /*name*/) {
  tvm::te::Tensor pool;
  return pool;
  #if 0
  return tvm::topi::nn::pool2d(input,
                        ToTvmArray(pool_attrs.kernel_shape),
                        ToTvmArray(pool_attrs.strides),
                        ToTvmArray(pool_attrs.pads),
                        /*pool_type*/ tvm::topi::nn::kAvgPool,
                        /*ceil_mode*/ false,
                        /*layout*/ "NCHW",
                        pool_attrs.count_include_pad);
}

tvm::te::Tensor GlobalMaxPool(const tvm::te::Tensor& input,
                          const PoolAttributes& /*pool_attrs*/,
                          const tvm::Array<tvm::PrimExpr>& /*output_shape*/,
                          const std::string& /*name*/) {
  return tvm::topi::nn::global_pool(input,
                               /*pool_type*/ tvm::topi::nn::kMaxPool,
                               /*layout*/ "NCHW");
}

tvm::te::Tensor GlobalAveragePool(const tvm::te::Tensor& input,
                              const PoolAttributes& /*pool_attrs*/,
                              const tvm::Array<tvm::PrimExpr>& /*output_shape*/,
                              const std::string& /*name*/) {
  return tvm::topi::nn::global_pool(input,
                               /*pool_type*/ tvm::topi::nn::kAvgPool,
                               /*layout*/ "NCHW");
}

}  // namespace tvm_codegen
}  // namespace onnxruntime

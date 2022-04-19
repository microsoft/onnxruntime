// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/mti_x86/nn/pool_ops.h"

#include "core/codegen/mti/mti_tvm_utils.h"
#include "core/mlas/inc/mlas.h"
#include "core/providers/cpu/nn/pool_attributes.h"
#include <topi/detail/extern.h>

namespace onnxruntime {
namespace nuphar {

TVM_REGISTER_GLOBAL("tvm.contrib.onnxruntime.pool_f32")
    .set_body([](tvm::TVMArgs args, tvm::TVMRetValue* /*ret*/) {
      // input
      DLTensor* X = args[0];
      DCHECK(tvm::runtime::TypeMatch(X->dtype, kDLFloat, 32));
      // output
      DLTensor* Y = args[1];
      DCHECK(tvm::runtime::TypeMatch(Y->dtype, kDLFloat, 32));

      // enum is not an integral type
      int k = args[2];
      MLAS_POOLING_KIND kind = static_cast<MLAS_POOLING_KIND>(k);

      int num_args = args.size();
      DCHECK(num_args > 3);
      int arg_idx = 3;

      auto extract_values_fn = [&]() {
        std::vector<int64_t> vec;

        DCHECK(arg_idx < num_args);
        int64_t num_vec = args[arg_idx++];
        for (int i = 0; i < num_vec; i++, arg_idx++) {
          DCHECK(arg_idx < num_args);
          int64_t v = args[arg_idx];
          vec.push_back(v);
        }
        return vec;
      };

      std::vector<int64_t> kernel_shape = extract_values_fn();
      std::vector<int64_t> padding = extract_values_fn();
      std::vector<int64_t> strides = extract_values_fn();

      MlasPool(kind,
               /*num_pooling_dims*/ kernel_shape.size(),
               /*input_shape*/ X->shape,
               kernel_shape.data(),
               padding.data(),
               strides.data(),
               /*output_shape*/ Y->shape,
               reinterpret_cast<float*>(static_cast<char*>(X->data) + X->byte_offset),
               reinterpret_cast<float*>(static_cast<char*>(Y->data) + Y->byte_offset),
               /*thread_pool*/ nullptr);
    });

TVM_REGISTER_GLOBAL("tvm.contrib.onnxruntime.global_pool_f32")
    .set_body([](tvm::TVMArgs args, tvm::TVMRetValue* /*ret*/) {
      // input
      DLTensor* X = args[0];
      DCHECK(tvm::runtime::TypeMatch(X->dtype, kDLFloat, 32));
      // output
      DLTensor* Y = args[1];
      DCHECK(tvm::runtime::TypeMatch(Y->dtype, kDLFloat, 32));

      // enum is not an integral type
      int k = args[2];
      MLAS_POOLING_KIND kind = static_cast<MLAS_POOLING_KIND>(k);

      MlasPool(kind,
               /*num_pooling_dims*/ X->ndim - 2,
               /*input_shape*/ X->shape,
               /*kernel_shape*/ nullptr,
               /*padding*/ nullptr,
               /*strides*/ nullptr,
               /*output_shape*/ Y->shape,
               reinterpret_cast<float*>(static_cast<char*>(X->data) + X->byte_offset),
               reinterpret_cast<float*>(static_cast<char*>(Y->data) + Y->byte_offset),
               /*thread_pool*/ nullptr);
    });

static tvm::Tensor MakeGlobalPoolCommon(const tvm::Tensor& X,
                                        const MLAS_POOLING_KIND kind,
                                        const tvm::Array<tvm::Expr>& output_shape,
                                        const std::string& name) {
  return topi::detail::make_extern(
           /*output_shapes*/ {output_shape},
           /*output_types*/ {X->dtype},
           /*inputs*/ {X},
           [&](tvm::Array<tvm::Buffer> ins, tvm::Array<tvm::Buffer> outs) {
             return topi::detail::call_packed({tvm::Expr("tvm.contrib.onnxruntime.global_pool_f32"),
                                               topi::detail::pack_buffer(ins[0]),
                                               topi::detail::pack_buffer(outs[0]),
                                               static_cast<int>(kind)});
           },
           name, /*tag*/ "", /*attrs*/ {})[0];
}

static tvm::Tensor MakePoolCommon(const tvm::Tensor& X,
                                  const PoolAttributes& pool_attrs,
                                  const MLAS_POOLING_KIND kind,
                                  const tvm::Array<tvm::Expr>& output_shape,
                                  const std::string& name) {
  size_t num_input_dims = X.ndim();
  ORT_ENFORCE(num_input_dims >= 3, "Input dimension must be >= 3");
  size_t num_pooling_dims = num_input_dims - 2;
  ORT_ENFORCE(num_pooling_dims <= 3, "pooling size must be <= 3");
  ORT_ENFORCE(num_pooling_dims == pool_attrs.kernel_shape.size(),
              "kernel_shape num_dims is not compatible with X num_dims.");

  tvm::Array<tvm::Expr> pooling_args;
  auto add_args_fn = [&](const TensorShapeVector& v) {
    pooling_args.push_back(tvm::make_const(tvm::Int(64), static_cast<int64_t>(v.size())));
    for (auto n : v) {
      pooling_args.push_back(tvm::make_const(tvm::Int(64), n));
    }
  };
  add_args_fn(pool_attrs.kernel_shape);
  add_args_fn(pool_attrs.pads);
  add_args_fn(pool_attrs.strides);

  return topi::detail::make_extern(
           /*output_shapes*/ {output_shape},
           /*output_types*/ {X->dtype},
           /*inputs*/ {X},
           [&](tvm::Array<tvm::Buffer> ins, tvm::Array<tvm::Buffer> outs) {
             tvm::Array<tvm::Expr> args = {tvm::Expr("tvm.contrib.onnxruntime.pool_f32"),
                                           topi::detail::pack_buffer(ins[0]),
                                           topi::detail::pack_buffer(outs[0]),
                                           static_cast<int>(kind)};
             // kernel_shape, padds and strides are directly passed into the external function
             for (size_t i = 0; i < pooling_args.size(); i++) {
               args.push_back(pooling_args[i]);
             }
             return topi::detail::call_packed(args);
           },
           name, /*tag*/ "", /*attrs*/ {})[0];
}

tvm::Tensor AveragePool(const tvm::Tensor& X,
                        const PoolAttributes& pool_attrs,
                        const tvm::Array<tvm::Expr>& output_shape,
                        const std::string& name) {
  MLAS_POOLING_KIND kind = pool_attrs.count_include_pad ? MlasAveragePoolingIncludePad
                                                        : MlasAveragePoolingExcludePad;
  return MakePoolCommon(X, pool_attrs, kind, output_shape, name);
}

tvm::Tensor GlobalAveragePool(const tvm::Tensor& X,
                              const PoolAttributes& pool_attrs,
                              const tvm::Array<tvm::Expr>& output_shape,
                              const std::string& name) {
  MLAS_POOLING_KIND kind = pool_attrs.count_include_pad ? MlasAveragePoolingIncludePad
                                                        : MlasAveragePoolingExcludePad;
  return MakeGlobalPoolCommon(X, kind, output_shape, name);
}

tvm::Tensor MaxPool(const tvm::Tensor& X,
                    const PoolAttributes& pool_attrs,
                    const tvm::Array<tvm::Expr>& output_shape,
                    const std::string& name) {
  return MakePoolCommon(X, pool_attrs, MlasMaximumPooling, output_shape, name);
}

tvm::Tensor GlobalMaxPool(const tvm::Tensor& X,
                          const PoolAttributes& /*pool_attrs*/,
                          const tvm::Array<tvm::Expr>& output_shape,
                          const std::string& name) {
  return MakeGlobalPoolCommon(X, MlasMaximumPooling, output_shape, name);
}

}  // namespace nuphar
}  // namespace onnxruntime

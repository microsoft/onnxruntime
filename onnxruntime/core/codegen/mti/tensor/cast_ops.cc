// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/mti/tensor/cast_ops.h"

#include "core/codegen/mti/mti_tvm_utils.h"
#include <tvm/topi/broadcast.h>
#include <tvm/topi/elemwise.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::te::Tensor CastTensor(const tvm::te::Tensor& X, tvm::DataType type, const std::string& name) {
  return tvm::topi::cast(X, type, name);
}

// handle cases where bool is reprented as uint8 (e.g. in ONNX).
tvm::te::Tensor CastTensorToUInt8Bool(const tvm::te::Tensor& X, const std::string& name) {
  return tvm::te::compute(
      X->shape,
      [&](const tvm::Array<tvm::tir::Var>& indices) {
        auto val = X(indices);
        // A special cast from float16 to bool, first cast up to float32,
        // to workaround a float16 bug in many TVM backends.
        // Intel Skylake is one of them. https://github.com/dmlc/tvm/issues/2959
        // TODO: remove it, after TVM is fixed
        if (X->dtype == tvm::DataType::Float(16))
          val = tvm::cast(tvm::DataType::Float(32), val);
        return tvm::tir::Select(tvm::topi::equal(val, tvm::tir::make_zero(val.dtype())),
                                tvm::tir::make_zero(tvm::DataType::UInt(8)),
                                tvm::tir::make_const(tvm::DataType::UInt(8), 1));
      },
      name);
}

}  // namespace tvm_codegen
}  // namespace onnxruntime

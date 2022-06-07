// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/mti/tensor/cast_ops.h"

#include "core/codegen/mti/mti_tvm_utils.h"
#include <topi/broadcast.h>
#include <topi/elemwise.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::Tensor Cast(const tvm::Tensor& X, tvm::Type type, const std::string& name) {
  return topi::cast(X, type, name);
}

// handle cases where bool is reprented as uint8 (e.g. in ONNX).
tvm::Tensor CastToUInt8Bool(const tvm::Tensor& X, const std::string& name) {
  return tvm::compute(
      X->shape,
      [&](const tvm::Array<tvm::Var>& indices) {
        auto val = X(indices);
        // A special cast from float16 to bool, first cast up to float32,
        // to workaround a float16 bug in many TVM backends.
        // Intel Skylake is one of them. https://github.com/dmlc/tvm/issues/2959
        // TODO: remove it, after TVM is fixed
        if (X->dtype == HalideIR::Float(16))
          val = tvm::cast(HalideIR::Float(32), val);
        return tvm::ir::Select::make(topi::equal(val, tvm::make_zero(val.type())),
                                     tvm::make_zero(HalideIR::UInt(8)),
                                     tvm::make_const(HalideIR::UInt(8), 1));
      },
      name);
}

}  // namespace tvm_codegen
}  // namespace onnxruntime

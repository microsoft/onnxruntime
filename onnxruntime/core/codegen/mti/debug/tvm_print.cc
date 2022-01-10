// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/mti/debug/tvm_print.h"

#include "core/codegen/common/utils.h"
#include "core/codegen/common/dump_array.h"
#include "core/codegen/mti/common.h"
#include <tvm/topi/detail/extern.h>

namespace onnxruntime {
namespace tvm_codegen {

TVM_REGISTER_GLOBAL("tvm.contrib.onnxruntime.print")
    .set_body([](tvm::TVMArgs args, tvm::TVMRetValue* /*ret*/) {
      DLTensor* X = args[0];
      DLTensor* Y = args[1];

      DLDataType dtype = X->dtype;
      std::vector<int64_t> shape;
      int64_t total_size = 1;
      for (int i = 0; i < X->ndim; ++i) {
        shape.push_back(X->shape[i]);
        total_size *= X->shape[i];
      }

      // pass X to Y
      memcpy(static_cast<char*>(Y->data) + Y->byte_offset,
             static_cast<char*>(X->data) + X->byte_offset,
             total_size * dtype.bits / 8);

      if (tvm::runtime::TypeMatch(dtype, kDLFloat, 32)) {
        float* data = reinterpret_cast<float*>(static_cast<char*>(X->data) + X->byte_offset);
        DumpArray("float tensor:", data, shape);
      } else if (tvm::runtime::TypeMatch(dtype, kDLInt, 8)) {
        int8_t* data = reinterpret_cast<int8_t*>(static_cast<char*>(X->data) + X->byte_offset);
        DumpArray("int8 tensor:", data, shape);
      } else if (tvm::runtime::TypeMatch(dtype, kDLInt, 16)) {
        int16_t* data = reinterpret_cast<int16_t*>(static_cast<char*>(X->data) + X->byte_offset);
        DumpArray("int16 tensor:", data, shape);
      } else if (tvm::runtime::TypeMatch(dtype, kDLInt, 32)) {
        int32_t* data = reinterpret_cast<int32_t*>(static_cast<char*>(X->data) + X->byte_offset);
        DumpArray("int32 tensor:", data, shape);
      } else if (tvm::runtime::TypeMatch(dtype, kDLUInt, 8)) {
        uint8_t* data = reinterpret_cast<uint8_t*>(static_cast<char*>(X->data) + X->byte_offset);
        DumpArray("uint8 tensor:", data, shape);
      } else if (tvm::runtime::TypeMatch(dtype, kDLUInt, 16)) {
        uint16_t* data = reinterpret_cast<uint16_t*>(static_cast<char*>(X->data) + X->byte_offset);
        DumpArray("uint16 tensor:", data, shape);
      } else if (tvm::runtime::TypeMatch(dtype, kDLUInt, 32)) {
        uint32_t* data = reinterpret_cast<uint32_t*>(static_cast<char*>(X->data) + X->byte_offset);
        DumpArray("uint32 tensor:", data, shape);
      } else {
        MTI_ASSERT(0 && "not implemented!");
      }
    });

tvm::Array<tvm::te::Tensor>
PrintTVMTensorExtern(const tvm::te::Tensor& X,
                     const std::string& name) {
  return tvm::topi::detail::make_extern(
      {X->shape},
      {X->dtype},
      {X},
      [&](tvm::Array<tvm::te::Buffer> ins, tvm::Array<tvm::te::Buffer> outs) {
        return tvm::topi::detail::call_packed({tvm::tir::StringImm("tvm.contrib.onnxruntime.print"),
                                               tvm::topi::detail::pack_buffer(ins[0]),
                                               tvm::topi::detail::pack_buffer(outs[0])});
      },
      name + "_print", "", {});
}

tvm::te::Tensor PrintImmutable(const tvm::te::Tensor& X) {
  auto outputs = PrintTVMTensorExtern(X, X->op->name + "_print");
  return outputs[0];
}

void Print(tvm::te::Tensor& X) {
  X = PrintImmutable(X);
}

}  // namespace tvm_codegen
}  // namespace onnxruntime

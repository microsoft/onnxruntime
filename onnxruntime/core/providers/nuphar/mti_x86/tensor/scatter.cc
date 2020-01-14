// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/mti_x86/tensor/scatter.h"

#include "core/codegen/common/utils.h"
#include "core/codegen/mti/mti_tvm_utils.h"
#include "core/common/common.h"
#include <topi/detail/extern.h>

namespace onnxruntime {
namespace nuphar {

#define STRINGIFY_1(x) #x
#define STRINGIFY(x) STRINGIFY_1(x)
#define GET_EXTERN_SCATTER_STR(input_type, index_type)                \
  STRINGIFY(tvm.contrib.onnxruntime.scatter_##input_type##index_type)

static int64_t DLTensorSize(const DLTensor* dl_tensor) {
  int64_t sz = 1;
  for (int i = 0; i < dl_tensor->ndim; ++i) {
    sz *= dl_tensor->shape[i];
  }
  return sz;
}

template<class T, class Tind>
void ScatterCommon(tvm::TVMArgs args, tvm::TVMRetValue* /*ret*/) {
  DLTensor* input = args[0];
  DLTensor* indices = args[1];
  DLTensor* updates = args[2];
  DLTensor* output = args[3];
  int axis = args[4];

  int num_dims = input->ndim;
  DCHECK(axis < num_dims);

  for (int i = 0; i < num_dims; i++) {
    if (indices->shape[i] != updates->shape[i]) {
      LOG(FATAL) <<
        "Indices vs updates dimensions differs at position=" <<
        i << " " << indices->shape[i] << " vs " << updates->shape[i];
    }
  }

  // extract indices from raw data
  Tind* indices_data = reinterpret_cast<Tind*>(static_cast<char*>(indices->data) + indices->byte_offset);
  int64_t indices_size = DLTensorSize(indices);
  std::vector<Tind> indices_data_vec(indices_size);
  int64_t axis_size = input->shape[axis];
  for (int64_t i = 0; i < indices_size; i++) {
    Tind idx = indices_data[i];
    // indices can be negative values
    if (idx >= -axis_size && idx < axis_size) {
      indices_data_vec[i] = idx >= 0 ? idx : idx + static_cast<Tind>(axis_size);
    } else {
      LOG(FATAL) <<
        "indices element out of data bounds, idx=" << idx <<
        " must be within the inclusive range [" << -axis_size <<
        "," << axis_size - 1 << "]";
    }
  }

  // copy input data into output
  int64_t input_size = DLTensorSize(input);
  memcpy(static_cast<char*>(output->data) + output->byte_offset,
         static_cast<char*>(input->data) + input->byte_offset,
         input_size * input->dtype.bits / 8);

  std::vector<int64_t> input_strides;
  GetStrides(input->shape, num_dims, input_strides);

  T* output_data = reinterpret_cast<T*>(static_cast<char*>(output->data) + output->byte_offset);
  T* updates_data = reinterpret_cast<T*>(static_cast<char*>(updates->data) + updates->byte_offset);
  const std::vector<int> indices_shape(indices->shape, indices->shape + num_dims);
  // Because indices data is flat, running_indices maintains indices's original dimensions.
  // We will use its dimensions to compute the corresponding index (or offset) to output_data,
  // which is also flat.
  std::vector<int64_t> running_indices(num_dims, 0);

  for (int64_t i = 0; i < indices_size; i++) {
    Tind idx = indices_data_vec[i];
    // output indices come from running_indices
    std::vector<int64_t> curr_output_indices = running_indices;
    curr_output_indices[axis] = static_cast<int64_t>(idx);

    // get the index into output_data
    int64_t output_idx = 0;
    for (int j = 0; j < num_dims; j++) {
      output_idx += curr_output_indices[j] * input_strides[j];
    }

    // update data
    output_data[output_idx] = updates_data[i];

    // update running_indices
    Tind carry = 1;
    for (int j = num_dims - 1; j >= 0; j--) {
      if (carry == 0) break;
      Tind curr_idx = running_indices[j] + carry;
      running_indices[j] = curr_idx % indices_shape[j];
      carry = curr_idx / indices_shape[j];
    }
  }
}

#define REGISTER_EXTERN_SCATTER(input_type, index_type)               \
  TVM_REGISTER_GLOBAL(GET_EXTERN_SCATTER_STR(input_type, index_type)) \
      .set_body([](tvm::TVMArgs args, tvm::TVMRetValue* ret) {        \
        ScatterCommon<input_type, index_type>(args, ret);             \
      });

#define REGISTER_EXTERN_SCATTER_PAIR(input_type) \
  REGISTER_EXTERN_SCATTER(input_type, int32_t)   \
  REGISTER_EXTERN_SCATTER(input_type, int64_t)   \

REGISTER_EXTERN_SCATTER_PAIR(bool)
REGISTER_EXTERN_SCATTER_PAIR(int8_t)
REGISTER_EXTERN_SCATTER_PAIR(uint8_t)
REGISTER_EXTERN_SCATTER_PAIR(int16_t)
REGISTER_EXTERN_SCATTER_PAIR(uint16_t)
REGISTER_EXTERN_SCATTER_PAIR(int32_t)
REGISTER_EXTERN_SCATTER_PAIR(uint32_t)
REGISTER_EXTERN_SCATTER_PAIR(int64_t)
REGISTER_EXTERN_SCATTER_PAIR(uint64_t)
REGISTER_EXTERN_SCATTER_PAIR(float)
REGISTER_EXTERN_SCATTER_PAIR(double)

#undef REGISTER_EXTERN_SCATTER

static tvm::Tensor MakeExternScatter(const tvm::Tensor& t,
                                     int64_t axis_p,
                                     const tvm::Tensor& indices,
                                     const tvm::Tensor& updates,
                                     const std::string& name,
                                     const char* extern_scatter) {
  // handle negative axis
  int64_t input_rank = static_cast<int64_t>(t->shape.size());
  DCHECK(input_rank >= 1);
  DCHECK(input_rank == static_cast<int64_t>(indices->shape.size()));
  DCHECK(input_rank == static_cast<int64_t>(updates->shape.size()));
  int axis = static_cast<int>(tvm_codegen::HandleNegativeAxis(axis_p, input_rank));

  // output has the same shape as input
  tvm::Array<tvm::Expr> output_shape;
  for (int64_t i = 0; i < input_rank; i++) {
    output_shape.push_back(t->shape[i]);
  }

  return topi::detail::make_extern(
           /*output_shapes*/ {output_shape},
           /*output_types*/ {t->dtype},
           /*inputs*/ {t, indices, updates},
           [&](tvm::Array<tvm::Buffer> ins, tvm::Array<tvm::Buffer> outs) {
             tvm::Array<tvm::Expr> args = {tvm::Expr(extern_scatter),
                                           topi::detail::pack_buffer(ins[0]),
                                           topi::detail::pack_buffer(ins[1]),
                                           topi::detail::pack_buffer(ins[2]),
                                           topi::detail::pack_buffer(outs[0]),
                                           axis};
             return topi::detail::call_packed(args);
           },
           name, /*tag*/ "", /*attrs*/ {})[0];
}

tvm::Tensor Scatter(const tvm::Tensor& t,
                    int64_t axis_p,
                    const tvm::Tensor& indices,
                    const tvm::Tensor& updates,
                    const std::string& name) {

#define MAKE_EXTERN_SCATTER_IF_MATCH(input_tensor_type, index_tensor_type, input_type, index_type)               \
  if (t->dtype == input_tensor_type && indices->dtype == index_tensor_type) {                                    \
    return MakeExternScatter(t, axis_p, indices, updates, name, GET_EXTERN_SCATTER_STR(input_type, index_type)); \
  }

#define MAKE_EXTERN_SCATTER_PAIR_IF_MATCH(input_tensor_type, input_type)             \
  MAKE_EXTERN_SCATTER_IF_MATCH(input_tensor_type, tvm::Int(32), input_type, int32_t) \
  MAKE_EXTERN_SCATTER_IF_MATCH(input_tensor_type, tvm::Int(64), input_type, int64_t)

  MAKE_EXTERN_SCATTER_PAIR_IF_MATCH(tvm::Bool(), bool)
  MAKE_EXTERN_SCATTER_PAIR_IF_MATCH(tvm::Int(8), int8_t)
  MAKE_EXTERN_SCATTER_PAIR_IF_MATCH(tvm::UInt(8), uint8_t)
  MAKE_EXTERN_SCATTER_PAIR_IF_MATCH(tvm::Int(16), int16_t)
  MAKE_EXTERN_SCATTER_PAIR_IF_MATCH(tvm::UInt(16), uint16_t)
  MAKE_EXTERN_SCATTER_PAIR_IF_MATCH(tvm::Int(32), int32_t)
  MAKE_EXTERN_SCATTER_PAIR_IF_MATCH(tvm::UInt(32), uint32_t)
  MAKE_EXTERN_SCATTER_PAIR_IF_MATCH(tvm::Int(64), int64_t)
  MAKE_EXTERN_SCATTER_PAIR_IF_MATCH(tvm::UInt(64), uint64_t)
  MAKE_EXTERN_SCATTER_PAIR_IF_MATCH(tvm::Float(32), float)
  MAKE_EXTERN_SCATTER_PAIR_IF_MATCH(tvm::Float(64), double)

#undef MAKE_EXTERN_SCATTER_PAIR_IF_MATCH
#undef MAKE_EXTERN_SCATTER_IF_MATCH

  ORT_NOT_IMPLEMENTED("input type is not implementated");
}

#undef STRINGIFY_1
#undef STRINGIFY
#undef GET_EXTERN_SCATTER_STR

}  // namespace tvm_codegen
}  // namespace onnxruntime

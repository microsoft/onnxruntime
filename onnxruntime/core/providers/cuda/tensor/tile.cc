// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/tile.h"
#include "core/providers/cpu/tensor/utils.h"
#include "tile_impl.h"

using namespace onnxruntime::common;
namespace onnxruntime {
namespace cuda {

namespace {

#ifdef BUILD_CUDA_EP_AS_PLUGIN
// PLUGIN BUILD ADAPTATION: TileOp::IsTileMemcpy (CPU provider) cannot be
// linked into the plugin. Reimplement the memcpy fast-path check here.
// Keep in sync with TileOp::IsTileMemcpy in cpu/tensor/tile.cc.
bool IsTileMemcpyForPlugin(const TensorShape& input_shape,
                           const int64_t* repeats,
                           size_t rank,
                           /*out*/ bool& is_batched_memcpy,
                           /*out*/ size_t& num_of_elements_per_batch,
                           /*out*/ size_t& num_of_copies_per_batch,
                           /*out*/ size_t& num_of_batch_copies) {
  for (int64_t i = static_cast<int64_t>(rank) - 1; i >= 0; --i) {
    if (repeats[i] != 1) {
      if (input_shape.SizeToDimension(onnxruntime::narrow<size_t>(i)) == 1) {
        num_of_copies_per_batch = 1;
        for (int64_t j = 0; j <= i; ++j) {
          num_of_copies_per_batch *= onnxruntime::narrow<size_t>(repeats[onnxruntime::narrow<size_t>(j)]);
        }
        is_batched_memcpy = false;
        return true;
      } else if (i == 1) {
        num_of_elements_per_batch = static_cast<size_t>(input_shape.SizeFromDimension(1));
        num_of_copies_per_batch = onnxruntime::narrow<size_t>(repeats[onnxruntime::narrow<size_t>(i)]);
        num_of_batch_copies = onnxruntime::narrow<size_t>(repeats[0]);
        is_batched_memcpy = true;
        return true;
      } else {
        break;
      }
    }
  }
  return false;
}
#endif

}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Tile,
    kOnnxDomain,
    6,
    12,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                              DataTypeImpl::GetTensorType<double>(),
                              DataTypeImpl::GetTensorType<int32_t>(),
                              DataTypeImpl::GetTensorType<int64_t>(),
                              DataTypeImpl::GetTensorType<MLFloat16>()})
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    Tile);

ONNX_OPERATOR_KERNEL_EX(
    Tile,
    kOnnxDomain,
    13,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                              DataTypeImpl::GetTensorType<double>(),
                              DataTypeImpl::GetTensorType<int32_t>(),
                              DataTypeImpl::GetTensorType<int64_t>(),
                              DataTypeImpl::GetTensorType<MLFloat16>(),
                              DataTypeImpl::GetTensorType<BFloat16>()})
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    Tile);

#define CASE_TILE(type)                                                                                            \
  case sizeof(type): {                                                                                             \
    TileImpl(Stream(ctx), input_rank, fdm_input_shape, input_strides,                                              \
             reinterpret_cast<const typename ToCudaType<type>::MappedType*>(input_data), fdm_output_strides,       \
             reinterpret_cast<typename ToCudaType<type>::MappedType*>(output_data), output_tensor.Shape().Size()); \
  } break

#define CASE_TILE_MEMCPY(type)                                                                                \
  case sizeof(type): {                                                                                        \
    TileMemcpyImpl(Stream(ctx), reinterpret_cast<const typename ToCudaType<type>::MappedType*>(input_data),   \
                   reinterpret_cast<typename ToCudaType<type>::MappedType*>(output_data), input_shape.Size(), \
                   num_of_copies_per_batch);                                                                  \
  } break

#define CASE_TILE_BATCHED_MEMCPY(type)                                                                             \
  case sizeof(type): {                                                                                             \
    TileBatchedMemcpyImpl(Stream(ctx), reinterpret_cast<const typename ToCudaType<type>::MappedType*>(input_data), \
                          reinterpret_cast<typename ToCudaType<type>::MappedType*>(output_data),                   \
                          num_of_elements_per_batch, input_shape.Size(), num_of_batch_copies,                      \
                          num_of_copies_per_batch);                                                                \
  } break

Status Tile::ComputeInternal(OpKernelContext* ctx) const {
  auto& input_tensor = *ctx->Input<Tensor>(0);
  auto& repeats_tensor = *ctx->Input<Tensor>(1);
  int32_t input_rank = static_cast<int32_t>(input_tensor.Shape().NumDimensions());

  if (repeats_tensor.Shape().NumDimensions() != 1)
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "'repeat' input tensor must be 1 dimensional");
  if (repeats_tensor.Shape().Size() != input_rank)
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "'repeat' input tensor must have the same length as the 'input' tensor");

  // Calculate the shape of the output tensor
  auto* repeats = repeats_tensor.Data<int64_t>();
  const auto& input_shape = input_tensor.Shape();
  const auto input_dims = input_shape.GetDims();
  auto output_dims(input_shape.AsShapeVector());
  for (int32_t axis = 0; axis < input_rank; axis++) {
    if (repeats[axis] < 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Tile repeat value must be non-negative, got: ", repeats[axis]);
    }
    output_dims[axis] = SafeInt<int64_t>(output_dims[axis]) * repeats[axis];
  }

  TensorShape output_shape(output_dims);
  auto& output_tensor = *ctx->Output(0, output_shape);

  void* output_data = output_tensor.MutableDataRaw();
  const void* input_data = input_tensor.DataRaw();
  const auto element_size = input_tensor.DataType()->Size();

  // Repeat tensor input can have 0 as a valid value
  // check if the computed output_shape size is 0 and
  // return an empty tensor if so.
  if (output_shape.Size() == 0) {
    return Status::OK();
  }

  // Repeat tensor has all 1s in it
  if (output_shape == input_shape) {
    return CUDA_CALL(cudaMemcpyAsync(output_tensor.MutableDataRaw(), input_tensor.DataRaw(), input_tensor.SizeInBytes(), cudaMemcpyDeviceToDevice, Stream(ctx)));
  }

  bool is_batched_memcpy = false;
  size_t num_of_elements_per_batch = 1;
  size_t num_of_copies_per_batch = 1;
  size_t num_of_batch_copies = 1;
#ifdef BUILD_CUDA_EP_AS_PLUGIN
  if (IsTileMemcpyForPlugin(input_shape,
                            repeats,
                            input_rank,
                            is_batched_memcpy,
                            num_of_elements_per_batch,
                            num_of_copies_per_batch,
                            num_of_batch_copies)) {
#else
  if (TileOp::IsTileMemcpy(input_shape,
                           repeats,
                           input_rank,
                           is_batched_memcpy,
                           num_of_elements_per_batch,
                           num_of_copies_per_batch,
                           num_of_batch_copies)) {
#endif
    if (!is_batched_memcpy) {
      switch (element_size) {
        CASE_TILE_MEMCPY(float);
        CASE_TILE_MEMCPY(double);
        CASE_TILE_MEMCPY(MLFloat16);
        default:
          ORT_THROW("Unsupported value attribute datatype with sizeof=: ", element_size);
          break;
      }
    } else {
      switch (element_size) {
        CASE_TILE_BATCHED_MEMCPY(float);
        CASE_TILE_BATCHED_MEMCPY(double);
        CASE_TILE_BATCHED_MEMCPY(MLFloat16);
        default:
          ORT_THROW("Unsupported value attribute datatype with sizeof=: ", element_size);
          break;
      }
    }

    return Status::OK();
  }

  TensorPitches input_pitches(input_dims);
  TArray<int64_t> input_strides(input_pitches);

  TArray<fast_divmod> fdm_input_shape(input_rank);
  for (size_t i = 0; i < input_dims.size(); ++i) {
    fdm_input_shape[gsl::narrow_cast<int>(i)] = fast_divmod(gsl::narrow_cast<int>(input_dims[i]));
  }

  TArray<fast_divmod> fdm_output_strides(input_rank);
  TensorPitches output_pitches(output_dims);
  for (auto i = 0; i < input_rank; i++) {
    fdm_output_strides[i] = fast_divmod(static_cast<int>(output_pitches[i]));
  }

  static_assert(sizeof(float) == sizeof(int32_t), "Float and Int32 are of different sizes");
  static_assert(sizeof(double) == sizeof(int64_t), "Double and Int64 are of different sizes");

  if (output_tensor.Shape().Size() > 0) {
    switch (element_size) {
      CASE_TILE(float);
      CASE_TILE(double);
      CASE_TILE(MLFloat16);
      default:
        ORT_THROW("Unsupported value attribute datatype with sizeof=: ", element_size);
        break;
    }
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime

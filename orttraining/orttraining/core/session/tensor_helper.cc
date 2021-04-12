// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/session/tensor_helper.h"

#ifdef USE_CUDA
#include "core/providers/cuda/cuda_common.h"
#endif

namespace onnxruntime {
namespace training {

// Return the shape of a tensor slice.
std::vector<int64_t> GetSliceShape(
    const std::vector<int64_t>& shape,  // before-slicing tensor shape
    const size_t slice_axis,            // axis to slice along
    const size_t num_slices) {          // number of slices along the slicing axis
  ORT_ENFORCE(shape.size() > 0);
  ORT_ENFORCE(slice_axis < shape.size());
  ORT_ENFORCE(num_slices > 0);
  ORT_ENFORCE(shape.at(slice_axis) > 0);
  ORT_ENFORCE(shape.at(slice_axis) % num_slices == 0);

  // Shape of slice along slice_axis.
  std::vector<int64_t> slice_shape(shape.size());
  // Compute original slice's shape.
  std::copy(shape.begin(), shape.end(), slice_shape.begin());
  // Replace the sliced dimension.
  slice_shape.at(slice_axis) = shape.at(slice_axis) / num_slices;

  return slice_shape;
}

// Given tensor's element type and shape, this function creates a tensor in the passed-in session.
OrtValue CreateCpuTensorValue(
    const MLDataType elem_type,
    std::vector<int64_t> shape,
    onnxruntime::InferenceSession& session_state) {
  ORT_ENFORCE(elem_type->AsPrimitiveDataType(), "Tensor's element type must be a scalar type.");
  ORT_ENFORCE(shape.size() > 0, "Shape vector must be non-empty.");

  // Get CPU allocator from the session.
  OrtMemoryInfo cpu_location(onnxruntime::CPU, OrtDeviceAllocator);
  AllocatorPtr cpu_allocator = session_state.GetAllocator(cpu_location);

  // Given a shape, allocate a tensor using CPU allocator.
  auto cpu_tensor = onnxruntime::make_unique<Tensor>(elem_type, shape, cpu_allocator);

  // Create type definition for the created tensor.
  auto tensor_type = DataTypeImpl::GetType<Tensor>();

  // Create OrtValue to wrap the allocated tensor.
  OrtValue cpu_value{cpu_tensor.release(), tensor_type, tensor_type->GetDeleteFunc()};

  return cpu_value;
}

// Copy a chunk of memory to CPU from GPU.
void CopyGpuToCpu(
    void* dst_ptr,
    const void* src_ptr,
    const size_t size,
    const OrtMemoryInfo& dst_location,
    const OrtMemoryInfo& src_location) {
  ORT_ENFORCE(dst_location.device.Type() == OrtDevice::CPU);

#ifdef USE_CUDA
  // Current CUDA device.
  int device;
  CUDA_CALL(cudaGetDevice(&device));

  if (device != src_location.id) {
    // Need to switch to the allocating device.
    CUDA_CALL(cudaSetDevice(src_location.id));
    // Copy from GPU to CPU.
    CUDA_CALL(cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyDeviceToHost));
    // Switch back to current device.
    CUDA_CALL(cudaSetDevice(device));
  } else {
    // Copy from GPU to CPU.
    CUDA_CALL(cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyDeviceToHost));
  }
#else
  ORT_UNUSED_PARAMETER(dst_ptr);
  ORT_UNUSED_PARAMETER(src_ptr);
  ORT_UNUSED_PARAMETER(size);
  ORT_UNUSED_PARAMETER(dst_location);
  ORT_UNUSED_PARAMETER(src_location);
  ORT_THROW("CPU-to-CPU copy is not implemented.");
#endif
}

// Copy a chunk of memory to CPU from CPU.
void CopyCpuToCpu(
    void* dst_ptr,
    const void* src_ptr,
    const size_t size,
    const OrtMemoryInfo& dst_location,
    const OrtMemoryInfo& src_location) {
  ORT_ENFORCE(src_location.device.Type() == OrtDevice::CPU);
  ORT_ENFORCE(dst_location.device.Type() == OrtDevice::CPU);
  memcpy(dst_ptr, src_ptr, size);
}

// Copy a tensor allocated on GPU/CPU to CPU buffer.
void CopyToCpuTensor(Tensor& dst, const Tensor& src) {
  // Get the tensor shape.
  const TensorShape& dst_shape = dst.Shape();
  const TensorShape& src_shape = src.Shape();

  ORT_ENFORCE(dst_shape == src_shape);

  // Get tensor's element type.
  const MLDataType src_type = src.DataType();
  const MLDataType dst_type = dst.DataType();

  ORT_ENFORCE(dst_type == src_type);

  // Get tensor's memory location.
  const OrtMemoryInfo& dst_location = dst.Location();
  const OrtMemoryInfo& src_location = src.Location();

  ORT_ENFORCE(src_location.device.Type() == OrtDevice::CPU || src_location.device.Type() == OrtDevice::GPU,
              "The copy function can only copy source tensor from CPU/GPU to CPU.");
  ORT_ENFORCE(dst_location.device.Type() == OrtDevice::CPU, "Destination tensor must be a CPU tensor.");

  auto src_ptr = src.DataRaw();
  auto dst_ptr = dst.MutableDataRaw();

  if (src_location.device.Type() == OrtDevice::GPU) {
    CopyGpuToCpu(dst_ptr, src_ptr, src.SizeInBytes(), dst_location, src_location);
  } else {
    CopyCpuToCpu(dst_ptr, src_ptr, src.SizeInBytes(), dst_location, src_location);
  }
}

// Update the linear index as if one extra axis is appended to the original shape.
// For a tensor with shape [D1, D2, D3], the linear index of element at (x, y, z) is
// i = x * (D2 * D3) + y * D3 + z. If we append one dimension to form a new shape [D1, D2, D3, D4],
// the new linear index at (x, y, z, u) can be computed using i * D4 + u.
size_t UpdateLinearIndex(const size_t linear_index, const size_t new_axis_index, const size_t new_axis_dim) {
  return linear_index * new_axis_dim + new_axis_index;
}

// If we slice tensor with shape [D_1, D_2, ..., D_j, D_{axis}, D_k, ..., D_n], then segment_size is Dk * ... * Dn and
// the num_segments is D1 * ... * Dj. "axis" is the axis to slice or concatenate along.
void ComputeSegment(const size_t axis, const TensorShape& shape, size_t& num_segments, size_t& segment_size) {
  segment_size = 1;
  num_segments = 1;

  for (size_t i = 0; i < static_cast<size_t>(shape.NumDimensions()); ++i) {
    if (i > axis) {
      segment_size *= shape[i];
    }
    if (i < axis) {
      num_segments *= shape[i];
    }
  }
}

// Copy a slice from source tensor to destination tensor.
// Assume that input shape is [10, 8, 2], slice_axis=1, num_slices=4.
// The destination's tensor is computed using
//  dst = src[:, lower:upper, :],
// where
//  slice_stride = 8 / num_slices,
//  lower = slice_id * slice_stride,
//  upper = (slice_id + 1) * slice_stride.
void CopySlice(Tensor& dst, const Tensor& src, const size_t slice_id, const size_t slice_axis, const size_t num_slices) {
  ORT_ENFORCE(dst.Location().device.Type() == OrtDevice::CPU, "Source must be a CPU tensor.");
  ORT_ENFORCE(src.Location().device.Type() == OrtDevice::CPU, "Destination must be a CPU tensor.");

  const TensorShape& src_shape = src.Shape();

  ORT_ENFORCE(src_shape[slice_axis] % num_slices == 0,
              "The dimension along the sliced axis must be divisible by the number of slices.",
              " We have sliced dimension ", src_shape[slice_axis], " and number slices ", num_slices, ".");

  // Cache sliced dimension.
  const size_t slice_dim = src_shape[slice_axis];
  // The number of slices along the sliced axis.
  const size_t slice_size = src_shape[slice_axis] / num_slices;

  auto src_ptr = src.DataRaw();
  auto dst_ptr = dst.MutableDataRaw();

  // If we slice tensor with shape [D1, D2, ..., Dj, sliced_dim, Dk, ..., Dn], then segment_size is Dk * ... * Dn.
  size_t segment_size = 0;
  // The total number of combinations of (D1, D2, ..., Dj). It's used as the total count of segments.
  size_t num_segments = 0;

  ComputeSegment(slice_axis, src_shape, num_segments, segment_size);

  // For each segment, we have several consecutive memory blocks to copy. For example, the first segment is
  // input[0, ..., 0, slice_id*slice_size : (slice_id + 1) * slice_size, :, ..., :], where its memory blocks
  // are
  //   input[0, ..., 0, 0, :, ..., :],
  //   input[0, ..., 0, 1, :, ..., :],
  //   ...
  //   input[0, ..., slize_size - 1, 1, :, ..., :].
  for (size_t i = 0; i < num_segments; ++i) {
    // Do pointer arithmetic operations using "char*" because things are stored in terms of bytes.
    // Copy input[i, slice_id*slice_size : (slice_id + 1) * slice_size, :, ..., :] to buffer.
    const void* src_addr = reinterpret_cast<const char*>(src_ptr) + UpdateLinearIndex(i, slice_id * slice_size, slice_dim) * segment_size * src.DataType()->Size();
    void* dst_addr = reinterpret_cast<char*>(dst_ptr) + UpdateLinearIndex(i, 0 * slice_size, 1) * segment_size * dst.DataType()->Size();
    memcpy(dst_addr, src_addr, segment_size * slice_size * src.DataType()->Size());
  }
}

// Slice the input tensor "value" along the axis indicated by "slice_axis".
// It's the "slice_id"-th slice along the indicated axis and the total number
// of slices is "num_slices".
OrtValue SliceTensor(
    const OrtValue& value,
    const size_t slice_id,
    const size_t slice_axis,
    const size_t num_slices,
    onnxruntime::InferenceSession& session_state) {
  ORT_ENFORCE(value.IsTensor(), "Sliced value must be a tensor.");
  auto& src = value.Get<Tensor>();
  auto src_shape = src.Shape().GetDims();

  auto buf_value = CreateCpuTensorValue(src.DataType(), src_shape, session_state);
  ORT_ENFORCE(buf_value.IsTensor(), "Buffer value must be a tensor.");
  auto& buf = *buf_value.GetMutable<Tensor>();
  CopyToCpuTensor(buf, src);

  // Compute the shape of the slice_id-th slice in the original tensor.
  auto slice_shape = GetSliceShape(src_shape, slice_axis, num_slices);

  // Allocate the slice as a tensor.
  auto dst_value = CreateCpuTensorValue(src.DataType(), slice_shape, session_state);
  ORT_ENFORCE(dst_value.IsTensor(), "Buffer value must be a tensor.");
  auto& dst = *dst_value.GetMutable<Tensor>();

  // Copy the content of slice from the original tensor to the newly allocated tensor.
  CopySlice(dst, buf, slice_id, slice_axis, num_slices);

  return dst_value;
}

void CopyConcat(
    const size_t concat_axis,
    const std::vector<OrtValue>& values,
    OrtValue& result) {
  // Dimension product before concat_axis; d_0 x d_1 x ... x d_{concat_axis - 1}.
  size_t num_segments = 0;
  // Assume the concatenated tensors are n-dimensional.
  // Dimension product after concat_axis; d_{concat_axis + 1} x d_{concat_axis + 2} x ... x d_n.
  size_t segment_size = 0;

  ComputeSegment(concat_axis, result.Get<Tensor>().Shape(), num_segments, segment_size);

  auto& dst = *result.GetMutable<Tensor>();
  auto& dst_shape = dst.Shape();
  for (size_t i_seg = 0; i_seg < static_cast<size_t>(num_segments); ++i_seg) {
    // Accumulated dimension sum along the "concat_axis" of processed input tensors.
    // If we concatenate 3 tensors with shape [1, 2, 3] and "concat_axis" is 1, "anchor_bias" would be
    // 0 when processing the 1st tensor, 2 when processing the 2nd tensor, 2+2 when processing the last tensor.
    size_t anchor_bias = 0;

    for (size_t i_tensor = 0; i_tensor < values.size(); ++i_tensor) {
      auto& src = values[i_tensor].Get<Tensor>();
      auto& src_shape = src.Shape();

      auto src_ptr = src.DataRaw();
      auto dst_ptr = dst.MutableDataRaw();

      // Copied chunk size. Its unit is the number of tensor elements.
      auto chunk_size = src_shape[concat_axis] * segment_size;

      // Bias of the i_seg-th segment in input tensor. Its unit is the number of tensor elements.
      // chunk_size * 
      auto src_bias = UpdateLinearIndex(i_seg, 0, src_shape[concat_axis]) * segment_size;
      auto dst_bias = UpdateLinearIndex(i_seg, anchor_bias, dst_shape[concat_axis]) * segment_size;

      memcpy(reinterpret_cast<char*>(dst_ptr) + dst_bias * dst.DataType()->Size(),
             reinterpret_cast<const char*>(src_ptr) + src_bias * src.DataType()->Size(),
             chunk_size * src.DataType()->Size());

      anchor_bias += src_shape[concat_axis];
    }
  }
}

OrtValue ConcatenateTensors(
    const std::vector<OrtValue>& orig_values,
    const size_t axis,
    onnxruntime::InferenceSession& session_state) {
  // Concatenated tensors in CPU buffers.
  std::vector<OrtValue> cpu_values;
  // Result tensor's shape.
  std::vector<int64_t> new_shape = orig_values.front().Get<Tensor>().Shape().GetDims();
  // Tensor elements' type.
  MLDataType elem_type = orig_values.front().Get<Tensor>().DataType();
  int64_t new_dim = 0;

  for (auto& src : orig_values) {
    ORT_ENFORCE(src.IsTensor(), "Only tensors can be concatenated.");
    // Extract the shape of the original tensor.
    auto& src_tensor = src.Get<Tensor>();
    auto src_shape = src_tensor.Shape().GetDims();
    ORT_ENFORCE(src_shape.size() == new_shape.size(), "Tensors to be concatenated must have the same rank.");
    ORT_ENFORCE(src_tensor.DataType() == elem_type, "Tensors to be concatenated must have the same rank.");

    // Allocate the same size of tensor on CPU.
    auto cpu_value = CreateCpuTensorValue(elem_type, src_shape, session_state);
    auto& cpu_tensor = *cpu_value.GetMutable<Tensor>();

    // Get source and destination for memory copy.
    auto src_ptr = src_tensor.DataRaw();
    auto dst_ptr = cpu_tensor.MutableDataRaw();

    const OrtMemoryInfo& dst_location = cpu_tensor.Location();
    const OrtMemoryInfo& src_location = src_tensor.Location();
    if (src_location.device.Type() == OrtDevice::GPU) {
      CopyGpuToCpu(dst_ptr, src_ptr, src_tensor.SizeInBytes(), dst_location, src_location);
    } else {
      CopyCpuToCpu(dst_ptr, src_ptr, src_tensor.SizeInBytes(), dst_location, src_location);
    }

    cpu_values.push_back(cpu_value);
    new_dim += src_shape[axis];
  }

  new_shape[axis] = new_dim;

  auto result_value = CreateCpuTensorValue(elem_type, new_shape, session_state);

  // With concatenated tensors in CPU buffer and allocated result tensor, we start the concatenation.
  CopyConcat(axis, cpu_values, result_value);

  return result_value;
}
}  // namespace training
}  // namespace onnxruntime
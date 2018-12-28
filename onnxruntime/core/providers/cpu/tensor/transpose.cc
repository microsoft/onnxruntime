// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/transpose.h"

namespace onnxruntime {

/* A permutation [a,b,c,...] indicates that 
   - The 0-th dimension of the output corresponds to the a-th dimension of input
   - The 1-st dimension of the output corresponds to the b-th dimension of input
   - The 2-nd dimension of the output corresponds to the c-th dimension of input
   etc.
   */

// ComputeOffset: compute offset into a tensor. This is essentially the dot-product of
// index and stride, restricted to the specified number of axes.
size_t ComputeOffset(const std::vector<int64_t>& index, const std::vector<size_t>& stride, int64_t num_axes) {
  size_t offset = 0;
  for (int64_t j = 0; j < num_axes; ++j) {
    offset += index[j] * stride[j];
  }
  return offset;
}

// IncrementIndex: Increment an index into a tensor (in lexicographic ordering), wrapping
// around the specified upper_bound.
void IncrementIndex(std::vector<int64_t>& index, const std::vector<int64_t>& upper_bound, int64_t num_axes) {
  for (int64_t k = num_axes - 1; k >= 0; --k) {
    index[k]++;
    if (index[k] < upper_bound[k]) break;
    index[k] = 0;
  }
}

// DoTranspose: copies source tensor to target, transposing elements.
// The stride vector indicates the transposition.
template <typename T>
static void DoTransposeImpl(int64_t num_axes, const std::vector<int64_t>& target_dims,
                            size_t num_blocks, size_t num_elts_in_block, const std::vector<size_t>& stride,
                            const T* source, T* target) {
  size_t blocksize = num_elts_in_block * sizeof(float);
  // index used to iterate over target iteration-space
  std::vector<int64_t> target_index(num_axes, 0);
  for (size_t i = 0; i < num_blocks; ++i) {
    // convert target_index into an offset in source data
    size_t source_offset = ComputeOffset(target_index, stride, num_axes);

    // copy
    memcpy(target, source + source_offset, blocksize);

    // increment target_index:
    IncrementIndex(target_index, target_dims, num_axes);
    target += num_elts_in_block;
  }
}

// DoTransposeEltWise: specialization of DoTranspose for the num_elts_in_block=1 case.
// copies source tensor to target, transposing elements.
// The stride vector indicates the transposition.
template <typename T>
static void DoTransposeEltWise(int64_t num_axes, const std::vector<int64_t>& target_dims, size_t num_blocks,
                               const std::vector<size_t>& stride, const T* source, T* target) {
  // index used to iterate over target iteration-space
  std::vector<int64_t> target_index(num_axes, 0);
  for (size_t i = 0; i < num_blocks; ++i) {
    // convert target_index into an offset in source data
    size_t source_offset = ComputeOffset(target_index, stride, num_axes);

    // copy
    *target = *(source + source_offset);

    // increment target_index:
    IncrementIndex(target_index, target_dims, num_axes);
    target++;
  }
}

// DoTransposeSingleBlock: specialization of DoTranspose for the num_blocks=1 case.
// copies source tensor to target, transposing elements.
template <typename T>
static void DoTransposeSingleBlock(size_t num_elts_in_block, const T* source, T* target) {
  size_t blocksize = num_elts_in_block * sizeof(T);
  // copy
  memcpy(target, source, blocksize);
}

template <typename T>
Status TransposeBase::DoTranspose(const std::vector<int64_t>& permutations, const Tensor& input, Tensor& output) {
  const auto& input_shape = input.Shape();
  const auto& input_dims = input_shape.GetDims();
  auto rank = input_shape.NumDimensions();

  std::vector<size_t> stride(rank);
  for (int i = 0; i < rank; i++) {
    size_t inpdim = permutations[i];
    if (inpdim + 1 < rank)
      stride[i] = input_shape.SizeFromDimension(inpdim + 1);
    else
      stride[i] = 1;
  }

  // Partition the permutation into a prefix and the largest suffix such that
  // every axis i in the suffix is mapped to i.
  int64_t num_axes_in_prefix = 0;  // number of axes in prefix
  size_t suffix_blocksize = 1;     // product of dimensions in the suffix
  size_t prefix_blocksize = 1;     // product of dimensions in the prefix
  bool is_suffix = true;

  for (int64_t i = rank - 1; i >= 0; --i) {
    int64_t input_axis = permutations[i];
    if (is_suffix && (input_axis == i)) {
      suffix_blocksize *= input_dims[input_axis];
    } else {
      is_suffix = false;
      prefix_blocksize *= input_dims[input_axis];
      ++num_axes_in_prefix;
    }
  }

  const T* input_data = input.Data<T>();
  T* output_data = output.MutableData<T>();

  if (1 == prefix_blocksize)
    DoTransposeSingleBlock<T>(suffix_blocksize, input_data, output_data);
  else if (1 == suffix_blocksize)
    DoTransposeEltWise<T>(num_axes_in_prefix, output.Shape().GetDims(), prefix_blocksize, stride,
                          input_data, output_data);
  else
    DoTransposeImpl<T>(num_axes_in_prefix, output.Shape().GetDims(), prefix_blocksize, suffix_blocksize, stride,
                       input_data, output_data);

  return Status::OK();
}

template <>
Status Transpose<float>::Compute(OpKernelContext* ctx) const {
  // Get input and output:
  const Tensor* input_tensor_ptr = ctx->Input<Tensor>(0);
  ORT_ENFORCE(input_tensor_ptr != nullptr);
  const Tensor& X = *input_tensor_ptr;
  const TensorShape& input_shape = X.Shape();
  const std::vector<int64_t>& input_dims = input_shape.GetDims();
  size_t rank = input_dims.size();

  std::vector<int64_t> output_dims(rank);
  const std::vector<int64_t>* p_perm;
  std::vector<int64_t> default_perm(rank);
  ComputeOutputShape(X, output_dims, default_perm, p_perm);

  TensorShape output_shape{output_dims};
  Tensor& Y = *ctx->Output(0, output_shape);

  TransposeBase::DoTranspose<float>(*p_perm, X, Y);

  return Status::OK();
}

ONNX_CPU_OPERATOR_KERNEL(
    Transpose,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Transpose<float>);

}  // namespace onnxruntime

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
void DoTranspose(int64_t num_axes, const std::vector<int64_t>& target_dims,
                 size_t num_blocks, size_t num_elts_in_block,
                 const std::vector<size_t>& stride,
                 float* target, const float* source) {
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
void DoTransposeEltWise(int64_t num_axes, const std::vector<int64_t>& target_dims,
                        size_t num_blocks,
                        const std::vector<size_t>& stride,
                        float* target, const float* source) {
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
// The stride vector indicates the transposition.
void DoTransposeSingleBlock(size_t num_elts_in_block, float* target, const float* source) {
  size_t blocksize = num_elts_in_block * sizeof(float);
  // copy
  memcpy(target, source, blocksize);
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

  std::vector<size_t> stride(rank);
  for (int i = 0; i < rank; i++) {
    size_t inpdim = (*p_perm)[i];
    if (inpdim + 1 < rank)
      stride[i] = input_shape.SizeFromDimension(inpdim + 1);
    else
      stride[i] = 1;
  }

  TensorShape output_shape{output_dims};
  Tensor* Y = ctx->Output(0, output_shape);
  const float* Xdata = X.template Data<float>();
  float* Ydata = Y->template MutableData<float>();

  // Partition the permutation into a prefix and the largest suffix such that
  // every axis i in the suffix is mapped to i.
  int64_t num_axes_in_prefix = 0;  // number of axes in prefix
  size_t suffix_blocksize = 1;     // product of dimensions in the suffix
  size_t prefix_blocksize = 1;     // product of dimensions in the prefix
  bool is_suffix = true;
  for (int64_t i = rank - 1; i >= 0; --i) {
    int64_t inpaxis = (*p_perm)[i];
    if (is_suffix && (inpaxis == i)) {
      suffix_blocksize *= input_dims[inpaxis];
    } else {
      is_suffix = false;
      prefix_blocksize *= input_dims[inpaxis];
      ++num_axes_in_prefix;
    }
  }

  if (1 == prefix_blocksize)
    DoTransposeSingleBlock(suffix_blocksize, Ydata, Xdata);
  else if (1 == suffix_blocksize)
    DoTransposeEltWise(num_axes_in_prefix, output_dims, prefix_blocksize, stride, Ydata, Xdata);
  else
    DoTranspose(num_axes_in_prefix, output_dims, prefix_blocksize, suffix_blocksize, stride, Ydata, Xdata);

  return Status::OK();
}

ONNX_CPU_OPERATOR_KERNEL(
    Transpose,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Transpose<float>);

}  // namespace onnxruntime

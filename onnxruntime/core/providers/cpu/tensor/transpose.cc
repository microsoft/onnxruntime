// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/transpose.h"
#include "core/framework/utils.h"

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

// IncrementIndex: Increment an index into a tensor (in lexicographic ordering), wrapping
// around the specified upper_bound.
void IncrementIndexAndUpdateOffset(std::vector<int64_t>& index, const std::vector<int64_t>& upper_bound, int64_t num_axes, 
                                   int64_t& offset, const std::vector<size_t>& stride){
  for (int64_t k = num_axes - 1; k >= 0; --k) {
    index[k]++;
    offset += (int64_t)stride[k];
    if (index[k] < upper_bound[k]) break;
    offset -= index[k] * (int64_t)stride[k];
    index[k] = 0;
  }
}


#ifdef USE_OPENMP
template <typename T>
static void OMP_DoTransposeImpl(int64_t num_axes, int64_t split_axes, const std::vector<int64_t>& target_dims,
                                size_t num_blocks, size_t num_elts_in_block, 
                                const std::vector<size_t>& mapped_strides, const T* source, T* target) {
  const int64_t max_splits = 32;
  int64_t num_splits = std::min(target_dims[split_axes], max_splits);
  int64_t remain_splits = target_dims[split_axes] % num_splits;
  int64_t num_blocks_per_col = num_blocks / target_dims[split_axes];
  int64_t num_cols_per_split = target_dims[split_axes] / num_splits;

  #pragma omp parallel for
  for (int64_t split = 0; split < num_splits; ++split) {
    int64_t start_col = (split < remain_splits) ? 
                        (split * (1 + num_cols_per_split)) : 
                        ((split - remain_splits) * num_cols_per_split + remain_splits * (1+num_cols_per_split));
    int64_t blocks_in_split = num_blocks_per_col * ((split < remain_splits)? (num_cols_per_split + 1) : num_cols_per_split);

    // index used to iterate over target iteration-space
    std::vector<int64_t> target_index(num_axes, 0LL);
    target_index[split_axes] = start_col;

    int64_t stride_after_split = 1;
    for (int64_t n = ((int64_t)target_dims.size()) - 1; n > split_axes; --n) {
      stride_after_split *= (int64_t)target_dims[n];
    }
    T* target_in_split = target + start_col * stride_after_split;
    size_t blocksize = num_elts_in_block * sizeof(T);
    int64_t source_offset = (int64_t)ComputeOffset(target_index, mapped_strides, num_axes);
    for (int64_t i = 0; i < blocks_in_split; ++i) {
      // copy
      memcpy(target_in_split, source + source_offset, blocksize);
      target_in_split += num_elts_in_block;

      // increment target_index,  update offset
      IncrementIndexAndUpdateOffset(target_index, target_dims, num_axes, source_offset, mapped_strides);
    }
  }
}
#endif

// DoTranspose: copies source tensor to target, transposing elements.
// The stride vector indicates the transposition.
template <typename T>
static void DoTransposeImpl(int64_t num_axes, const std::vector<int64_t>& target_dims,
                            size_t num_blocks, size_t num_elts_in_block, const std::vector<size_t>& stride,
                            const T* source, T* target) {
  #ifdef USE_OPENMP
  int64_t split_axes = 0;
  while (split_axes < num_axes-1 && target_dims[split_axes] == 1) ++split_axes;
  OMP_DoTransposeImpl(num_axes, split_axes, target_dims, num_blocks, num_elts_in_block, stride, source, target);
  return;
  #endif

  int64_t blocksize = num_elts_in_block * sizeof(float);
  // index used to iterate over target iteration-space
  std::vector<int64_t> target_index(num_axes, 0);
  // convert target_index into an offset in source data
  int64_t	source_offset = (int64_t)ComputeOffset(target_index, stride, num_axes);
  for (int64_t i = 0; i < (int64_t)num_blocks; ++i) {
    // copy
    memcpy(target, source + source_offset, blocksize);
    target += num_elts_in_block;

    // increment target_index:
    IncrementIndexAndUpdateOffset(target_index, target_dims, num_axes, source_offset, stride);
  }
}

#ifdef USE_OPENMP
template <typename T>
static void OMP_DoTransposeEltWise(int64_t num_axes, int64_t split_axes, const std::vector<int64_t>& target_dims,
                                   size_t num_blocks, const std::vector<size_t>& mapped_strides, const T* source, T* target) {
  const int64_t max_splits = 32;
  int64_t num_splits = std::min(target_dims[split_axes], max_splits);
  int64_t remain_splits = target_dims[split_axes] % num_splits;
  int64_t num_blocks_per_col = num_blocks / target_dims[split_axes];
  int64_t num_cols_per_split = target_dims[split_axes] / num_splits;

  #pragma omp parallel for
  for (int64_t split = 0; split < num_splits; ++split) {
    int64_t start_col = (split < remain_splits) ? 
                        (split * (1 + num_cols_per_split)) : 
                        ((split - remain_splits) * num_cols_per_split + remain_splits * (1+num_cols_per_split));
    int64_t blocks_in_split = num_blocks_per_col * ((split < remain_splits)? (num_cols_per_split + 1) : num_cols_per_split);

    // index used to iterate over target iteration-space
    std::vector<int64_t> target_index(num_axes, 0LL);
    target_index[split_axes] = start_col;
    int64_t stride_after_split = 1;
    for (int64_t n = ((int64_t)target_dims.size()) - 1; n > split_axes; --n) {
      stride_after_split *= (int64_t)target_dims[n];
    }
    T* target_in_split = target + start_col * stride_after_split;
    // convert target_index into an offset in source data
    int64_t source_offset = (int64_t)ComputeOffset(target_index, mapped_strides, num_axes);
    for (int64_t i = 0; i < blocks_in_split; ++i) {
      // copy
      *target_in_split++ = *(source + source_offset);

      // increment target_index:
      IncrementIndexAndUpdateOffset(target_index, target_dims, num_axes, source_offset, mapped_strides);
    }
  }                                     
}
#endif


// DoTransposeEltWise: specialization of DoTranspose for the num_elts_in_block=1 case.
// copies source tensor to target, transposing elements.
// The stride vector indicates the transposition.
template <typename T>
static void DoTransposeEltWise(int64_t num_axes, const std::vector<int64_t>& target_dims, size_t num_blocks,
                               const std::vector<size_t>& stride, const T* source, T* target) {
  #ifdef USE_OPENMP
  int64_t split_axes = 0;
  while (split_axes < num_axes-1 && target_dims[split_axes] == 1) ++split_axes;
  OMP_DoTransposeEltWise(num_axes, split_axes, target_dims, num_blocks, stride, source, target);
  return;
  #endif

  // index used to iterate over target iteration-space
  std::vector<int64_t> target_index(num_axes, 0);
  // convert target_index into an offset in source data
  int64_t source_offset = (int64_t)ComputeOffset(target_index, stride, num_axes);
  for (int64_t i = 0; i < (int64_t)num_blocks; ++i) {
    // copy
    *target++ = *(source + source_offset);

    // increment target_index:
    IncrementIndexAndUpdateOffset(target_index, target_dims, num_axes, source_offset, stride);
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
static Status DoTypedTranspose(const std::vector<int64_t>& permutations, const Tensor& input, Tensor& output) {
  const auto& input_shape = input.Shape();
  const auto& input_dims = input_shape.GetDims();
  int64_t rank = static_cast<int64_t>(input_shape.NumDimensions());

  std::vector<size_t> stride(rank);
  for (int64_t i = 0; i < rank; i++) {
    auto inpdim = permutations[i];
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

Status TransposeBase::DoTranspose(const std::vector<int64_t>& permutations, const Tensor& input, Tensor& output) {
  Status status = Status::OK();

  auto input_type = input.DataType();
  auto output_type = output.DataType();

  if (input_type != output_type) {
    status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Mismatched data types between input and output Tensors. ",
                             input_type, " != ", output_type);
  } else {
    DispatchOnTensorTypeWithReturn(input_type, status, DoTypedTranspose, permutations, input, output);
  }

  return status;
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

  DoTypedTranspose<float>(*p_perm, X, Y);

  return Status::OK();
}

ONNX_CPU_OPERATOR_KERNEL(
    Transpose,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Transpose<float>);

}  // namespace onnxruntime

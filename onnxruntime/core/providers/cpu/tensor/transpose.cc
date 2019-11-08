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
static inline size_t ComputeOffset(const std::vector<int64_t>& index, const std::vector<size_t>& stride, int64_t num_axes) {
  size_t offset = 0;
  for (int64_t j = 0; j < num_axes; ++j) {
    offset += index[j] * stride[j];
  }
  return offset;
}

// IncrementIndex: Increment an index into a tensor (in lexicographic ordering), wrapping
// around the specified upper_bound.
static inline void IncrementIndex(std::vector<int64_t>& index, const std::vector<int64_t>& upper_bound, int64_t num_axes) {
  for (int64_t k = num_axes - 1; k >= 0; --k) {
    index[k]++;
    if (index[k] < upper_bound[k]) break;
    index[k] = 0;
  }
}

// DoTransposeSingleBlock: specialization of DoTranspose for the num_blocks=1 case.
// copies source tensor to target, transposing elements.
static inline void DoTransposeSingleBlock(size_t num_elts_in_block, const void* source, void* target,
                                          size_t element_size) {
  size_t blocksize = num_elts_in_block * element_size;
  // copy
  memcpy(target, source, blocksize);
}

static inline void DoTransposeSingleBlock(size_t num_elts_in_block, const std::string* source, std::string* target) {
  const std::string* end = source + num_elts_in_block;
  std::copy(source, end, target);
}

// DoTranspose: copies source tensor to target, transposing elements.
// The stride vector indicates the transposition.
static void DoTransposeImpl(int64_t num_axes, const std::vector<int64_t>& target_dims,
                            size_t num_blocks, size_t num_elts_in_block, const std::vector<size_t>& stride,
                            const uint8_t* source, uint8_t* target, size_t element_size) {
  size_t blocksize = num_elts_in_block * element_size;
  // index used to iterate over target iteration-space
  std::vector<int64_t> target_index(num_axes, 0);
  for (size_t i = 0; i < num_blocks; ++i) {
    // convert target_index into an offset in source data
    size_t source_offset = ComputeOffset(target_index, stride, num_axes);

    // copy
    memcpy(target, source + source_offset * element_size, blocksize);

    // increment target_index:
    IncrementIndex(target_index, target_dims, num_axes);
    target += blocksize;
  }
}

static void DoTransposeImpl(int64_t num_axes, const std::vector<int64_t>& target_dims,
                            size_t num_blocks, size_t num_elts_in_block, const std::vector<size_t>& stride,
                            const std::string* source, std::string* target) {
  // index used to iterate over target iteration-space
  std::vector<int64_t> target_index(num_axes, 0);
  for (size_t i = 0; i < num_blocks; ++i) {
    // convert target_index into an offset in source data
    size_t source_offset = ComputeOffset(target_index, stride, num_axes);

    // copy
    DoTransposeSingleBlock(num_elts_in_block, source + source_offset, target);

    // increment target_index:
    IncrementIndex(target_index, target_dims, num_axes);
    target += num_elts_in_block;
  }
}

template <class T>
inline void CopyPrim(uint8_t* target, const uint8_t* source) {
  *reinterpret_cast<T*>(target) = *reinterpret_cast<const T*>(source);
}

// DoTransposeEltWise: specialization of DoTranspose for the num_elts_in_block=1 case.
// copies source tensor to target, transposing elements.
// The stride vector indicates the transposition.
static void DoTransposeEltWise(int64_t num_axes, const std::vector<int64_t>& target_dims, size_t num_blocks,
                               const std::vector<size_t>& stride, const uint8_t* source, uint8_t* target,
                               size_t element_size) {
  // index used to iterate over target iteration-space
  std::vector<int64_t> target_index(num_axes, 0);

  switch (element_size) {
    case sizeof(uint64_t):
      for (size_t i = 0; i < num_blocks; ++i) {
        // convert target_index into an offset in source data
        size_t source_offset = ComputeOffset(target_index, stride, num_axes);

        // copy
        CopyPrim<uint64_t>(target, source + (source_offset * element_size));

        // increment target_index:
        IncrementIndex(target_index, target_dims, num_axes);
        target += element_size;
      }
      break;
    case sizeof(uint32_t):
      for (size_t i = 0; i < num_blocks; ++i) {
        // convert target_index into an offset in source data
        size_t source_offset = ComputeOffset(target_index, stride, num_axes);

        // copy
        CopyPrim<uint32_t>(target, source + (source_offset * element_size));

        // increment target_index:
        IncrementIndex(target_index, target_dims, num_axes);
        target += element_size;
      }
      break;
    case sizeof(uint16_t):
      for (size_t i = 0; i < num_blocks; ++i) {
        // convert target_index into an offset in source data
        size_t source_offset = ComputeOffset(target_index, stride, num_axes);

        // copy
        CopyPrim<uint16_t>(target, source + (source_offset * element_size));

        // increment target_index:
        IncrementIndex(target_index, target_dims, num_axes);
        target += element_size;
      }
      break;
    case sizeof(uint8_t):
      for (size_t i = 0; i < num_blocks; ++i) {
        // convert target_index into an offset in source data
        size_t source_offset = ComputeOffset(target_index, stride, num_axes);

        // copy
        *target = *(source + (source_offset * element_size));

        // increment target_index:
        IncrementIndex(target_index, target_dims, num_axes);
        target += element_size;
      }
      break;
    default:
      assert(false);
  }
}

static void DoTransposeEltWise(int64_t num_axes, const std::vector<int64_t>& target_dims, size_t num_blocks,
                               const std::vector<size_t>& stride, const std::string* source, std::string* target) {
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

static Status DoUntypedTranspose(const std::vector<size_t>& permutations, const Tensor& input, Tensor& output) {
  const auto& input_shape = input.Shape();
  const auto& input_dims = input_shape.GetDims();
  auto rank = input_shape.NumDimensions();

  const auto element_size = input.DataType()->Size();
  const bool is_string_type = input.IsDataTypeString();

  std::vector<size_t> stride(rank);
  for (size_t i = 0; i < rank; i++) {
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

  if (is_string_type) {
    const auto* input_data = input.template Data<std::string>();
    auto* output_data = output.template MutableData<std::string>();
    if (1 == prefix_blocksize) {
      DoTransposeSingleBlock(suffix_blocksize, input_data, output_data);
    } else if (1 == suffix_blocksize) {
      DoTransposeEltWise(num_axes_in_prefix, output.Shape().GetDims(), prefix_blocksize, stride,
                         input_data, output_data);
    } else {
      DoTransposeImpl(num_axes_in_prefix, output.Shape().GetDims(), prefix_blocksize, suffix_blocksize, stride,
                      input_data, output_data);
    }
  } else {
    const auto* input_data = reinterpret_cast<const uint8_t*>(input.DataRaw());
    auto* output_data = reinterpret_cast<uint8_t*>(output.MutableDataRaw());
    if (1 == prefix_blocksize) {
      DoTransposeSingleBlock(suffix_blocksize, input_data, output_data, element_size);
    } else if (1 == suffix_blocksize) {
      DoTransposeEltWise(num_axes_in_prefix, output.Shape().GetDims(), prefix_blocksize, stride,
                         input_data, output_data, element_size);
    } else {
      DoTransposeImpl(num_axes_in_prefix, output.Shape().GetDims(), prefix_blocksize, suffix_blocksize, stride,
                      input_data, output_data, element_size);
    }
  }

  return Status::OK();
}

Status TransposeBase::DoTranspose(const std::vector<size_t>& permutations, const Tensor& input, Tensor& output) {
  Status status = Status::OK();

  auto input_type = input.DataType();
  auto output_type = output.DataType();

  if (input_type != output_type) {
    status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Mismatched data types between input and output Tensors. ",
                             input_type, " != ", output_type);
  } else {
    status = DoUntypedTranspose(permutations, input, output);
  }

  return status;
}

Status Transpose::Compute(OpKernelContext* ctx) const {
  // Get input and output:
  const auto* input_tensor_ptr = ctx->Input<Tensor>(0);
  ORT_ENFORCE(input_tensor_ptr != nullptr);
  const Tensor& X = *input_tensor_ptr;
  const TensorShape& input_shape = X.Shape();
  const std::vector<int64_t>& input_dims = input_shape.GetDims();
  size_t rank = input_dims.size();

  std::vector<int64_t> output_dims(rank);
  const std::vector<size_t>* p_perm;
  std::vector<size_t> default_perm(rank);
  const auto& status = ComputeOutputShape(X, output_dims, default_perm, p_perm);
  if (!status.IsOK())
    return status;

  TensorShape output_shape{output_dims};
  Tensor& Y = *ctx->Output(0, output_shape);

  DoUntypedTranspose(*p_perm, X, Y);

  return Status::OK();
}

ONNX_CPU_OPERATOR_KERNEL(
    Transpose,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Transpose);

}  // namespace onnxruntime

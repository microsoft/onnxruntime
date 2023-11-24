// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/transpose.h"

#include "core/framework/element_type_lists.h"
#include "core/framework/utils.h"
#include "core/framework/transpose_helper.h"
#include "core/framework/op_kernel_type_control_utils.h"
#include "core/mlas/inc/mlas.h"
#include "core/providers/op_kernel_type_control.h"
#include "utils.h"

namespace onnxruntime {

namespace {
using DefaultDataTypes = element_type_lists::All;
}  // namespace

namespace op_kernel_type_control {
// we're using one set of types for all opsets
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Transpose, Input, 0,
    DefaultDataTypes);

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
// enable all types for layout transformation
ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Transpose, Input, 0,
    DefaultDataTypes);
#endif
}  // namespace op_kernel_type_control

namespace {
// reduce the supported types with any global or op specific lists
using EnabledDataTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(kCpuExecutionProvider, kOnnxDomain,
                                                                        Transpose, Input, 0);
}  // namespace

/* A permutation [a,b,c,...] indicates that
   - The 0-th dimension of the output corresponds to the a-th dimension of input
   - The 1-st dimension of the output corresponds to the b-th dimension of input
   - The 2-nd dimension of the output corresponds to the c-th dimension of input
   etc.
   */

struct MultiIndex {
  size_t n_axes;
  std::vector<size_t> index;
  std::vector<size_t> upper_bound;
  std::vector<int64_t> stride;

  /* There is one MultiIndex instance per axis in the tensor.
   * The array keeps track of the position of a pointer walking through the data.
   * Any function using it creates an array of MultiIndex
   * then calls function IncrementIndexAndComputeOffsetSetup
   * to initialize the array. This constructor does not initialize
   * anything because it would be overwritten by function
   * IncrementIndexAndComputeOffsetSetup. This one calls method Init.
   * Function IncrementIndexAndComputeOffset is called to increment
   * the array of MultiIndex to move to the next data in the tensor.
   */
  MultiIndex() : index(), upper_bound(), stride() { n_axes = 0; }

  void Init(size_t num_axes) {
    index.resize(num_axes);
    upper_bound.resize(num_axes);
    stride.resize(num_axes);
    n_axes = num_axes;
  }

  void InitAxis(size_t n_axis, size_t i, size_t n, int64_t s) {
    index[n_axis] = i;
    upper_bound[n_axis] = n;
    stride[n_axis] = s;
  }
};

/* This function initializes an array of MultiIndex of size num_axes (one instance per axis).
 * target_dims is the shape of the transposed tensor, stride is linked to the tensor to
 * be transposed, if source_dims is the shape, stride[i] = source_dims[i+1] * source_dims[i+2] * ... * 1.
 * element_size is the size of the tensor element (sizeof(float), sizeof(double)).
 */
static void IncrementIndexAndComputeOffsetSetup(MultiIndex& mindex, size_t num_axes, gsl::span<const int64_t> target_dims,
                                                const gsl::span<const size_t>& stride, size_t element_size) {
  mindex.Init(num_axes);
  size_t naxes = 0;
  for (size_t i = 0; i < num_axes; ++i) {
    if (target_dims[i] == 1)
      continue;
    mindex.InitAxis(naxes, 0, static_cast<size_t>(target_dims[i]), stride[i] * static_cast<int64_t>(element_size));
    ++naxes;
  }
  ORT_ENFORCE(naxes > 0, "Method IncrementIndexAndComputeOffset assumes this value is strictly positive.");
  mindex.n_axes = naxes;
}

/* This function increments an array of MultiIndex initialized by function IncrementIndexAndComputeOffsetSetup.
 * It increments the last dimension, checks if it stays within boundary. If it stays in, it returns,
 * otherwise, it reset the dimension to zero and increments the previous one.
 * While doing that, every modification brought to the array of indices is applied on the
 * pointer local_source. It avoids computing again local_source from the source tensor.
 * At every time, the following condition is verified:
 * local_source = source + (sum_i mindex[i].index * mindex[i].stride
 */
template <typename T>
static inline void IncrementIndexAndComputeOffset(MultiIndex& mindex, const T*& local_source) {
  // Increment the last dimension.
  int pos = static_cast<int>(mindex.n_axes) - 1;
  local_source += mindex.stride[pos];
  // Checks it stays within boundaries.
  if (++mindex.index[pos] < mindex.upper_bound[pos])
    return;
  // If not, loops on other indices.
  // The first test is outside the loop to be faster.
  // As it is the most common case.
  local_source -= mindex.stride[pos] * mindex.index[pos];
  mindex.index[pos] = 0;
  --pos;
  for (; pos >= 0; --pos) {
    local_source += mindex.stride[pos];
    if (++mindex.index[pos] < mindex.upper_bound[pos])
      break;
    local_source -= mindex.stride[pos] * mindex.index[pos];
    mindex.index[pos] = 0;
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
static void DoTransposeImpl(int64_t num_axes, gsl::span<const int64_t> target_dims,
                            size_t num_blocks, size_t num_elts_in_block, const gsl::span<const size_t>& stride,
                            const uint8_t* source, uint8_t* target, size_t element_size) {
  size_t blocksize = num_elts_in_block * element_size;
  MultiIndex mindex;
  IncrementIndexAndComputeOffsetSetup(mindex, onnxruntime::narrow<size_t>(num_axes), target_dims, stride, element_size);

  const uint8_t* local_source = source;
  for (size_t i = 0; i < num_blocks; ++i) {
    ORT_ENFORCE((local_source >= source) && (local_source < source + num_blocks * blocksize));
    memcpy(target, local_source, blocksize);
    IncrementIndexAndComputeOffset(mindex, local_source);
    target += blocksize;
  }
}

static void DoTransposeImpl(int64_t num_axes, gsl::span<const int64_t> target_dims,
                            size_t num_blocks, size_t num_elts_in_block, const gsl::span<const size_t>& stride,
                            const std::string* source, std::string* target) {
  ORT_ENFORCE(num_axes > 0, "Transpose not implemented for empty tensors.");
  MultiIndex mindex;
  IncrementIndexAndComputeOffsetSetup(mindex, onnxruntime::narrow<size_t>(num_axes), target_dims, stride, 1);

  const std::string* local_source = source;
  for (size_t i = 0; i < num_blocks; ++i) {
    ORT_ENFORCE((local_source >= source) && (local_source < source + num_blocks * num_elts_in_block));
    DoTransposeSingleBlock(num_elts_in_block, local_source, target);
    IncrementIndexAndComputeOffset(mindex, local_source);
    target += num_elts_in_block;
  }
}

template <class T>
inline void CopyPrim(uint8_t* target, const uint8_t* source) {
  *reinterpret_cast<T*>(target) = *reinterpret_cast<const T*>(source);
}

// The function does not check num_axes > 0 but this is expected.
template <class T>
static bool TypedDoTransposeEltWise(int64_t num_axes, gsl::span<const int64_t> target_dims, size_t num_blocks,
                                    const gsl::span<const size_t>& stride, const uint8_t* source, uint8_t* target) {
  constexpr bool enabled = utils::HasTypeWithSameSize<EnabledDataTypes, T>();

  if (enabled) {
    MultiIndex mindex;
    IncrementIndexAndComputeOffsetSetup(mindex, onnxruntime::narrow<size_t>(num_axes), target_dims, stride, sizeof(T));

    const uint8_t* local_source = source;
    uint8_t* target_end = target + sizeof(T) * num_blocks;
    for (; target != target_end; target += sizeof(T)) {
      ORT_ENFORCE((local_source >= source) && (local_source < source + sizeof(T) * num_blocks));
      CopyPrim<T>(target, local_source);
      IncrementIndexAndComputeOffset(mindex, local_source);
    }
  }

  return enabled;
}

// DoTransposeEltWise: specialization of DoTranspose for the num_elts_in_block=1 case.
// copies source tensor to target, transposing elements.
// The stride vector indicates the transposition.
Status DoTransposeEltWise(int64_t num_axes, gsl::span<const int64_t> target_dims, size_t num_blocks,
                          const gsl::span<const size_t>& stride, const uint8_t* source, uint8_t* target,
                          size_t element_size) {
  bool enabled = false;
  switch (element_size) {
    case sizeof(uint64_t):
      enabled = TypedDoTransposeEltWise<uint64_t>(num_axes, target_dims, num_blocks, stride, source, target);
      break;
    case sizeof(uint32_t):
      enabled = TypedDoTransposeEltWise<uint32_t>(num_axes, target_dims, num_blocks, stride, source, target);
      break;
    case sizeof(uint16_t):
      enabled = TypedDoTransposeEltWise<uint16_t>(num_axes, target_dims, num_blocks, stride, source, target);
      break;
    case sizeof(uint8_t):
      enabled = TypedDoTransposeEltWise<uint8_t>(num_axes, target_dims, num_blocks, stride, source, target);
      break;
    default:
      // leave enabled as false
      break;
  }

  return enabled ? Status::OK()
                 : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Transpose of element size not supported in this build. Size=",
                                   element_size);
}

static void DoTransposeEltWise(int64_t num_axes, gsl::span<const int64_t> target_dims, size_t num_blocks,
                               const gsl::span<const size_t>& stride, const std::string* source, std::string* target) {
  ORT_ENFORCE(num_axes > 0, "Transpose not implemented for empty tensors.");
  MultiIndex mindex;
  IncrementIndexAndComputeOffsetSetup(mindex, onnxruntime::narrow<size_t>(num_axes), target_dims, stride, 1);

  // index used to iterate over target iteration-space
  const std::string* local_source = source;
  for (size_t i = 0; i < num_blocks; ++i) {
    ORT_ENFORCE((local_source >= source) && (local_source < source + num_blocks));
    *target = *local_source;
    IncrementIndexAndComputeOffset(mindex, local_source);
    target++;
  }
}

//  `input_shape_override` overrides the shape of `input` for compute purposes.
static Status DoUntypedTranspose(const gsl::span<const size_t>& permutations, const Tensor& input, Tensor& output,
                                 const TensorShape* input_shape_override = nullptr) {
  const auto& input_shape = input_shape_override ? *input_shape_override : input.Shape();
  const auto& input_dims = input_shape.GetDims();
  auto rank = input_shape.NumDimensions();

  const auto element_size = input.DataType()->Size();
  const bool is_string_type = input.IsDataTypeString();

  InlinedVector<size_t> stride(rank);
  for (size_t i = 0; i < rank; i++) {
    size_t inpdim = permutations[i];
    if (inpdim + 1 < rank)
      stride[i] = onnxruntime::narrow<size_t>(input_shape.SizeFromDimension(inpdim + 1));
    else
      stride[i] = 1;
  }

  // Partition the permutation into a prefix and the largest suffix such that
  // every axis i in the suffix is mapped to i.
  int64_t num_axes_in_prefix = 0;  // number of axes in prefix
  size_t suffix_blocksize = 1;     // product of dimensions in the suffix
  size_t prefix_blocksize = 1;     // product of dimensions in the prefix
  bool is_suffix = true;

  for (int64_t i = SafeInt<int64_t>(rank) - 1; i >= 0; --i) {
    int64_t input_axis = onnxruntime::narrow<int64_t>(permutations[onnxruntime::narrow<size_t>(i)]);
    if (is_suffix && (input_axis == i)) {
      suffix_blocksize *= static_cast<size_t>(input_dims[onnxruntime::narrow<size_t>(input_axis)]);
    } else {
      is_suffix = false;
      prefix_blocksize *= static_cast<size_t>(input_dims[onnxruntime::narrow<size_t>(input_axis)]);
      ++num_axes_in_prefix;
    }
  }

  Status status = Status::OK();

  if (is_string_type) {
    constexpr bool string_enabled = utils::HasType<EnabledDataTypes, std::string>();

    if (string_enabled) {
      const auto* input_data = input.Data<std::string>();
      auto* output_data = output.MutableData<std::string>();
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
      status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Transpose of std::string is not supported in this build.");
    }
  } else {
    const auto* input_data = reinterpret_cast<const uint8_t*>(input.DataRaw());
    auto* output_data = reinterpret_cast<uint8_t*>(output.MutableDataRaw());
    if (1 == prefix_blocksize) {
      DoTransposeSingleBlock(suffix_blocksize, input_data, output_data, element_size);
    } else if (1 == suffix_blocksize) {
      // this may return a failed status if the data size is not supported in this build
      status = DoTransposeEltWise(num_axes_in_prefix, output.Shape().GetDims(), prefix_blocksize, stride,
                                  input_data, output_data, element_size);
    } else {
      DoTransposeImpl(num_axes_in_prefix, output.Shape().GetDims(), prefix_blocksize, suffix_blocksize, stride,
                      input_data, output_data, element_size);
    }
  }

  return status;
}

bool IsTransposeReshape(const gsl::span<const size_t>& perm, gsl::span<const int64_t> input_dims) {
  // As long as the dims with values > 1 stay in the same order, it's a reshape.
  // Example: Shape=(1,1,1024,4096) -> perm=(2,0,3,1).
  size_t last_permuted_axis = 0;
  for (size_t i = 0; i < perm.size(); ++i) {
    if (input_dims[perm[i]] == 1)
      continue;
    if (perm[i] < last_permuted_axis)
      return false;
    last_permuted_axis = perm[i];
  }
  return true;
}

//`input_shape_override` overrides the shape of `input` for compute purposes.
Status TransposeBase::DoTranspose(const gsl::span<const size_t>& permutations, const Tensor& input, Tensor& output,
                                  const TensorShape* input_shape_override) {
  Status status = Status::OK();

  auto input_type = input.DataType();
  auto output_type = output.DataType();

  if (input_type != output_type) {
    status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Mismatched data types between input and output Tensors. ",
                             input_type, " != ", output_type);
  } else {
    TensorShape shape = input_shape_override ? *input_shape_override : input.Shape();
    if (IsTransposeReshape(permutations, shape.GetDims())) {
      // As long as the dims with values > 1 stay in the same order, it's a reshape.
      // Example: Shape=(1,1,1024,4096) -> perm=(2,0,3,1).
      CopyCpuTensor(&input, &output);
      return Status::OK();
    }

    size_t from = 0, to = 0;
    bool moving_single_axis = IsTransposeMovingSingleAxis(permutations, from, to);

    if (moving_single_axis && !input.IsDataTypeString()) {
      SingleAxisTranspose(permutations, input, output, from, to, input_shape_override);
    } else {
      // fall back to default implementation
      status = DoUntypedTranspose(permutations, input, output, input_shape_override);
    }
  }

  return status;
}

Status Transpose::Compute(OpKernelContext* ctx) const {
  const auto* input_tensor_ptr = ctx->Input<Tensor>(0);
  ORT_ENFORCE(input_tensor_ptr != nullptr);
  const Tensor& X = *input_tensor_ptr;
  const TensorShape& input_shape = X.Shape();
  auto input_dims = input_shape.GetDims();
  size_t rank = input_dims.size();

  TensorShapeVector output_dims(rank);
  const InlinedVector<size_t>* p_perm;
  InlinedVector<size_t> default_perm(rank);
  Status status = ComputeOutputShape(X, output_dims, default_perm, p_perm);
  if (!status.IsOK())
    return status;

  TensorShape output_shape{output_dims};
  Tensor& Y = *ctx->Output(0, output_shape);

  if (output_shape.Size() == 0)
    return Status::OK();

  if (IsTransposeReshape(*p_perm, input_dims)) {
    // As long as the dims with values > 1 stay in the same order, it's a reshape.
    // Example: Shape=(1,1,1024,4096) -> perm=(2,0,3,1).
    CopyCpuTensor(&X, &Y);
    return Status::OK();
  }

  size_t from = 0, to = 0;
  bool moving_single_axis = IsTransposeMovingSingleAxis(*p_perm, from, to);

  if (moving_single_axis && !X.IsDataTypeString()) {
    SingleAxisTranspose(*p_perm, X, Y, from, to, nullptr, ctx->GetOperatorThreadPool());
  } else {
    // fall back to default implementation
    status = DoUntypedTranspose(*p_perm, X, Y);
  }

  return status;
}

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Transpose,
    1,
    12,
    KernelDefBuilder().TypeConstraint("T", BuildKernelDefConstraintsFromTypeList<EnabledDataTypes>()),
    Transpose);

ONNX_CPU_OPERATOR_KERNEL(
    Transpose,
    13,
    KernelDefBuilder().TypeConstraint("T", BuildKernelDefConstraintsFromTypeList<EnabledDataTypes>()),
    Transpose);

}  // namespace onnxruntime

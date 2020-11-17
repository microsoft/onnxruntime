// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/transpose.h"
#include "core/framework/utils.h"
#include "core/mlas/inc/mlas.h"
#include "utils.h"

namespace onnxruntime {

/* A permutation [a,b,c,...] indicates that
   - The 0-th dimension of the output corresponds to the a-th dimension of input
   - The 1-st dimension of the output corresponds to the b-th dimension of input
   - The 2-nd dimension of the output corresponds to the c-th dimension of input
   etc.
   */

typedef struct MultiIndex {
  size_t index;
  size_t upper_bound;
  int64_t stride;
  MultiIndex() {}
  MultiIndex(size_t i, size_t n, int64_t s) {
    index = i;
    upper_bound = n;
    stride = s;
  }
} MultiIndex;

static size_t IncrementIndexAndComputeOffsetSetup(MultiIndex* mindex, int64_t num_axes, const std::vector<int64_t>& target_dims,
                                                  const std::vector<size_t>& stride, size_t element_size) {
  size_t naxes = 0;
  for (int64_t i = 0; i < num_axes; ++i) {
    if (target_dims[i] == 1)
      continue;
    mindex[naxes] = MultiIndex(0, static_cast<size_t>(target_dims[i]), stride[i] * element_size);
    ++naxes;
  }
  return naxes;
}

// Combines multi-index increment and corresponding pointer in the tensor to transpose.
static void IncrementIndexAndComputeOffset(MultiIndex* mindex, size_t naxes, const uint8_t*& local_source) {
  MultiIndex* it = mindex + (naxes - 1);
  local_source += it->stride;
  if (++it->index < it->upper_bound)
    return;
  local_source -= it->stride * it->index;
  it->index = 0;
  --it;
  MultiIndex* rend = mindex - 1;
  for (; it != rend; --it) {
    local_source += it->stride;
    if (++it->index < it->upper_bound)
      break;
    local_source -= it->stride * it->index;
    it->index = 0;
  }
}

// Combines multi-index increment and corresponding pointer in the string tensor to transpose.
static void IncrementIndexAndComputeOffset(MultiIndex* mindex, size_t naxes, const std::string*& local_source) {
  MultiIndex* it = mindex + (naxes - 1);
  local_source += it->stride;
  if (++it->index < it->upper_bound)
    return;
  local_source -= it->stride * it->index;
  it->index = 0;
  --it;
  MultiIndex* rend = mindex - 1;
  for (; it != rend; --it) {
    local_source += it->stride;
    if (++it->index < it->upper_bound)
      break;
    local_source -= it->stride * it->index;
    it->index = 0;
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
  std::vector<MultiIndex> mindex(num_axes);
  size_t naxes = IncrementIndexAndComputeOffsetSetup(mindex.data(), num_axes, target_dims, stride, element_size);

  const uint8_t* local_source = source;
  for (size_t i = 0; i < num_blocks; ++i) {
    memcpy(target, local_source, blocksize);
    IncrementIndexAndComputeOffset(mindex.data(), naxes, local_source);
    target += blocksize;
  }
}

static void DoTransposeImpl(int64_t num_axes, const std::vector<int64_t>& target_dims,
                            size_t num_blocks, size_t num_elts_in_block, const std::vector<size_t>& stride,
                            const std::string* source, std::string* target) {
  ORT_ENFORCE(num_axes > 0, "Transpose not implemented for empty tensors.");
  std::vector<MultiIndex> mindex(num_axes);
  size_t naxes = IncrementIndexAndComputeOffsetSetup(mindex.data(), num_axes, target_dims, stride, 1);

  const std::string* local_source = source;
  for (size_t i = 0; i < num_blocks; ++i) {
    DoTransposeSingleBlock(num_elts_in_block, local_source, target);
    IncrementIndexAndComputeOffset(mindex.data(), naxes, local_source);
    target += num_elts_in_block;
  }
}

template <class T>
inline void CopyPrim(uint8_t* target, const uint8_t* source) {
  *reinterpret_cast<T*>(target) = *reinterpret_cast<const T*>(source);
}

// The function does not check num_axes > 0 but this is expected.
template <class T>
static void TypedDoTransposeEltWise(int64_t num_axes, const std::vector<int64_t>& target_dims, size_t num_blocks,
                                    const std::vector<size_t>& stride, const uint8_t* source, uint8_t* target) {
  std::vector<MultiIndex> mindex(num_axes);
  size_t naxes = IncrementIndexAndComputeOffsetSetup(mindex.data(), num_axes, target_dims, stride, sizeof(T));

  const uint8_t* local_source = source;
  for (size_t i = 0; i < num_blocks; ++i) {
    CopyPrim<uint64_t>(target, local_source);
    IncrementIndexAndComputeOffset(mindex.data(), naxes, local_source);
    target += sizeof(T);
  }
}

// DoTransposeEltWise: specialization of DoTranspose for the num_elts_in_block=1 case.
// copies source tensor to target, transposing elements.
// The stride vector indicates the transposition.
static void DoTransposeEltWise(int64_t num_axes, const std::vector<int64_t>& target_dims, size_t num_blocks,
                               const std::vector<size_t>& stride, const uint8_t* source, uint8_t* target,
                               size_t element_size) {
  switch (element_size) {
    case sizeof(uint64_t):
      TypedDoTransposeEltWise<uint64_t>(num_axes, target_dims, num_blocks, stride, source, target);
      break;
    case sizeof(uint32_t):
      TypedDoTransposeEltWise<uint32_t>(num_axes, target_dims, num_blocks, stride, source, target);
      break;
    case sizeof(uint16_t):
      TypedDoTransposeEltWise<uint16_t>(num_axes, target_dims, num_blocks, stride, source, target);
      break;
    case sizeof(uint8_t):
      TypedDoTransposeEltWise<uint8_t>(num_axes, target_dims, num_blocks, stride, source, target);
      break;
    default:
      assert(false);
  }
}

static void DoTransposeEltWise(int64_t num_axes, const std::vector<int64_t>& target_dims, size_t num_blocks,
                               const std::vector<size_t>& stride, const std::string* source, std::string* target) {
  ORT_ENFORCE(num_axes > 0, "Transpose not implemented for empty tensors.");
  std::vector<MultiIndex> mindex(num_axes);
  size_t naxes = IncrementIndexAndComputeOffsetSetup(mindex.data(), num_axes, target_dims, stride, 1);

  // index used to iterate over target iteration-space
  const std::string* local_source = source;
  for (size_t i = 0; i < num_blocks; ++i) {
    *target = *local_source;
    IncrementIndexAndComputeOffset(mindex.data(), naxes, local_source);
    target++;
  }
}

//  `input_shape_override` overrides the shape of `input` for compute purposes.
static Status DoUntypedTranspose(const std::vector<size_t>& permutations, const Tensor& input, Tensor& output,
                                 const TensorShape* input_shape_override = nullptr) {
  const auto& input_shape = input_shape_override ? *input_shape_override : input.Shape();
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

/*
Optimizations for moving a single axis either inwards or outwards.

If moving outwards we can use a single reader and multiple writers. The number of writers is equal to the value of
the axis being moved.

  e.g. if the input is NHWC with shape {N, 300, 300, 3}, we can transpose to NCHW by reading once and having
       one writer for each of the 3 channels at a different offset in the output, updating the offset for each item
       in the batch of N.

Similarly if one axis is moving inwards we can use a single writer and multiple readers. The number of readers is equal
to the value of the axis being moved.

  e.g. if the input is NCHW with shape {N, 3, 300, 300}, we can transpose to NHWC by writing once using one reader for
       each of the 3 channels at a different offset in the input, updating the read offset for each item in the batch
       of N.

This can be generalized for any input where only one axis is being moved, with the block size for each read/write
being dependent on which axis is moving, what direction it's moving in, and where it's moving to.

We use simple pointer arithmetic if the size of each read/write is a power of 2 and between 8 and 64 bits.
We use memcpy if the block size is larger.

We fall back to the default implementation in all other cases, and if the input is std::string.
*/

// moving a single axis outwards where the read/write size is a power of 2 and between 8 and 64 bits.
template <typename T>
static void SimpleTransposeSingleAxisOutwards(const T* input_data, T* output_data,
                                              int64_t num_loops, int64_t num_writers,
                                              int64_t writes_per_loop, int64_t writes_per_writer_per_loop) {
  const T* end;
  for (int64_t l = 0; l < num_loops; ++l) {
    T* output_for_first_writer = output_data;

    for (auto wwpl = 0; wwpl < writes_per_writer_per_loop; ++wwpl) {
      T* output_for_current_writer = output_for_first_writer;

      end = input_data + num_writers;
      for (; input_data != end;) {
        *output_for_current_writer = *input_data++;

        // skip to output position for next writer
        output_for_current_writer += writes_per_writer_per_loop;
      }

      ++output_for_first_writer;
    }

    output_data += writes_per_loop;
  }
}

#ifdef MLAS_SUPPORTS_TRANSPOSE

static void SimpleTransposeSingleAxisOutwards(const uint8_t* input_data, uint8_t* output_data,
                                              int64_t num_loops, int64_t num_writers,
                                              int64_t writes_per_loop, int64_t writes_per_writer_per_loop) {
  for (int64_t l = 0; l < num_loops; ++l) {
    MlasTranspose(input_data,
                  output_data,
                  static_cast<size_t>(writes_per_writer_per_loop),
                  static_cast<size_t>(num_writers));
    input_data += writes_per_loop;
    output_data += writes_per_loop;
  }
}

#endif

//  `input_shape_override` overrides the shape of `input` for compute purposes.
static void TransposeSingleAxisOutwards(const std::vector<size_t>& permutations, const Tensor& input, Tensor& output,
                                        int64_t from, int64_t to, const TensorShape* input_shape_override = nullptr) {
  ORT_UNUSED_PARAMETER(permutations);

  const auto& input_shape = input_shape_override ? *input_shape_override : input.Shape();
  const auto& input_dims = input_shape.GetDims();

  const auto element_size = input.DataType()->Size();

  const auto* input_data = reinterpret_cast<const uint8_t*>(input.DataRaw());
  auto* output_data = reinterpret_cast<uint8_t*>(output.MutableDataRaw());

  auto num_loops = input_shape.SizeToDimension(to);
  auto num_writers = input_dims[from];
  auto block_size = input_shape.SizeFromDimension(from + 1);
  auto writes_per_loop = int64_t(input_shape.Size() / num_loops / block_size);
  auto writes_per_writer_per_loop = int64_t(writes_per_loop / num_writers);
  const int64_t bytes_per_write = block_size * element_size;

  switch (bytes_per_write) {
    case (sizeof(uint8_t)): {
      SimpleTransposeSingleAxisOutwards(input_data, output_data,
                                        num_loops, num_writers, writes_per_loop, writes_per_writer_per_loop);
      break;
    }
    case (sizeof(uint16_t)): {
      SimpleTransposeSingleAxisOutwards(reinterpret_cast<const uint16_t*>(input_data),
                                        reinterpret_cast<uint16_t*>(output_data),
                                        num_loops, num_writers, writes_per_loop, writes_per_writer_per_loop);
      break;
    }
    case (sizeof(uint32_t)): {
      SimpleTransposeSingleAxisOutwards(reinterpret_cast<const uint32_t*>(input_data),
                                        reinterpret_cast<uint32_t*>(output_data),
                                        num_loops, num_writers, writes_per_loop, writes_per_writer_per_loop);
      break;
    }
    case (sizeof(uint64_t)): {
      SimpleTransposeSingleAxisOutwards(reinterpret_cast<const uint64_t*>(input_data),
                                        reinterpret_cast<uint64_t*>(output_data),
                                        num_loops, num_writers, writes_per_loop, writes_per_writer_per_loop);
      break;
    }
    default: {
      // we need to use memcpy for each block
      for (int64_t l = 0; l < num_loops; ++l) {
        uint8_t* output_for_first_writer = output_data;

        for (auto wwpl = 0; wwpl < writes_per_writer_per_loop; ++wwpl) {
          uint8_t* output_for_current_writer = output_for_first_writer;

          for (int64_t w = 0; w < num_writers; ++w) {
            memcpy(output_for_current_writer, input_data, bytes_per_write);
            // skip to output position for next writer
            output_for_current_writer += (writes_per_writer_per_loop * bytes_per_write);
            input_data += bytes_per_write;
          }

          output_for_first_writer += bytes_per_write;
        }

        output_data += writes_per_loop * bytes_per_write;
      }
    }
  }
}

template <typename T>
static void SimpleTransposeSingleAxisInwards(const T* input_data, T* output_data,
                                             int64_t num_loops, int64_t num_readers,
                                             int64_t reads_per_loop, int64_t reads_per_reader_per_loop) {
  T* end;
  for (int64_t l = 0; l < num_loops; ++l) {
    const T* input_for_first_reader = input_data;

    for (auto rrpl = 0; rrpl < reads_per_reader_per_loop; ++rrpl) {
      const T* input_for_current_reader = input_for_first_reader;

      end = output_data + num_readers;
      for (; output_data != end;) {
        *output_data++ = *input_for_current_reader;
        // skip to input position for next reader
        input_for_current_reader += reads_per_reader_per_loop;
      }

      ++input_for_first_reader;
    }

    input_data += reads_per_loop;
  }
}

#ifdef MLAS_SUPPORTS_TRANSPOSE

static void SimpleTransposeSingleAxisInwards(const uint8_t* input_data, uint8_t* output_data,
                                             int64_t num_loops, int64_t num_readers,
                                             int64_t reads_per_loop, int64_t reads_per_reader_per_loop) {
  for (int64_t l = 0; l < num_loops; ++l) {
    MlasTranspose(input_data,
                  output_data,
                  static_cast<size_t>(num_readers),
                  static_cast<size_t>(reads_per_reader_per_loop));
    input_data += reads_per_loop;
    output_data += reads_per_loop;
  }
}

#endif

// moving a single axis inwards where the read/write size is a power of 2 and between 8 and 64 bits.
//  `input_shape_override` overrides the shape of `input` for compute purposes.
static void TransposeSingleAxisInwards(const std::vector<size_t>& permutations, const Tensor& input, Tensor& output,
                                       int64_t from, int64_t to, const TensorShape* input_shape_override = nullptr) {
  ORT_UNUSED_PARAMETER(permutations);

  const auto& input_shape = input_shape_override ? *input_shape_override : input.Shape();
  const auto& input_dims = input_shape.GetDims();

  const auto element_size = input.DataType()->Size();

  const auto* input_data = reinterpret_cast<const uint8_t*>(input.DataRaw());
  auto* output_data = reinterpret_cast<uint8_t*>(output.MutableDataRaw());

  auto num_loops = input_shape.SizeToDimension(from);
  auto num_readers = input_dims[from];
  auto block_size = input_shape.SizeFromDimension(to + 1);
  auto reads_per_loop = int64_t(input_shape.Size() / num_loops / block_size);
  auto reads_per_reader_per_loop = int64_t(reads_per_loop / num_readers);
  const int64_t bytes_per_read = block_size * element_size;

  switch (bytes_per_read) {
    case (sizeof(uint8_t)): {
      SimpleTransposeSingleAxisInwards(input_data, output_data,
                                       num_loops, num_readers, reads_per_loop, reads_per_reader_per_loop);
      break;
    }
    case (sizeof(uint16_t)): {
      SimpleTransposeSingleAxisInwards(reinterpret_cast<const uint16_t*>(input_data),
                                       reinterpret_cast<uint16_t*>(output_data),
                                       num_loops, num_readers, reads_per_loop, reads_per_reader_per_loop);
      break;
    }
    case (sizeof(uint32_t)): {
      SimpleTransposeSingleAxisInwards(reinterpret_cast<const uint32_t*>(input_data),
                                       reinterpret_cast<uint32_t*>(output_data),
                                       num_loops, num_readers, reads_per_loop, reads_per_reader_per_loop);
      break;
    }
    case (sizeof(uint64_t)): {
      SimpleTransposeSingleAxisInwards(reinterpret_cast<const uint64_t*>(input_data),
                                       reinterpret_cast<uint64_t*>(output_data),
                                       num_loops, num_readers, reads_per_loop, reads_per_reader_per_loop);
      break;
    }
    default: {
      // we need to use memcpy for each block
      for (int64_t l = 0; l < num_loops; ++l) {
        const uint8_t* input_for_first_reader = input_data;

        for (auto rrpl = 0; rrpl < reads_per_reader_per_loop; ++rrpl) {
          const uint8_t* input_for_current_reader = input_for_first_reader;

          for (int64_t r = 0; r < num_readers; ++r) {
            memcpy(output_data, input_for_current_reader, bytes_per_read);
            output_data += bytes_per_read;

            // skip to input position for next reader
            input_for_current_reader += (reads_per_reader_per_loop * bytes_per_read);
          }

          input_for_first_reader += bytes_per_read;
        }

        input_data += reads_per_loop * bytes_per_read;
      }
    }
  }
}

//  `input_shape_override` overrides the shape of `input` for compute purposes.
static void SingleAxisTranspose(const std::vector<size_t>& permutations, const Tensor& input, Tensor& output,
                                size_t from, size_t to, const TensorShape* input_shape_override = nullptr) {
  if (from > to) {
    TransposeSingleAxisOutwards(permutations, input, output, from, to, input_shape_override);
  } else {
    TransposeSingleAxisInwards(permutations, input, output, from, to, input_shape_override);
  }
}

static bool IsMovingSingleAxis(const std::vector<size_t>& permutations, size_t& from, size_t& to) {
  // if a single axis moved to an outer dimension, the values should be one lower than the index until the slot the
  // axis was moved from, and equal to the index after that.
  // e.g. axis 3 moves out to 1 would be: 0, 3, 1, 2, 4
  auto check_moved_outwards = [&permutations](size_t cur, size_t moved_from) {
    // we start processing with the slot after the moved one, so the expected value is one less than the index
    size_t expected = cur - 1;
    for (size_t end = permutations.size(); cur < end; ++cur) {
      if (permutations[cur] != expected) {
        return false;
      }

      // we are at the slot the axis moved from, so do an additional increment before checking the next value
      if (cur == moved_from) {
        ++expected;
      }

      ++expected;
    }

    return true;
  };

  // if a single axis moved to an inner dimension, the values should be one higher than the index until the slot the
  // axis was moved to, and equal to the index after that.
  // e.g. axis 1 moves inwards to 3 would be: 0, 2, 3, 1, 4
  auto check_moved_inwards = [&permutations](size_t cur, size_t& moved_to) {
    size_t started_at = cur;
    size_t expected = cur + 1;
    moved_to = std::numeric_limits<size_t>::max();

    for (size_t end = permutations.size(); cur < end; ++cur) {
      if (permutations[cur] != expected) {
        // if a single axis moved it must have come from the location we started at
        if (started_at != permutations[cur]) {
          return false;
        }

        moved_to = cur;
      } else {
        ++expected;
      }
    }

    return moved_to != std::numeric_limits<size_t>::max();
  };

  bool single_axis_moved = false;
  // check axis moving outwards (earlier entry in permutations)
  for (size_t i = 0, end = permutations.size(); i < end; ++i) {
    size_t axis = permutations[i];

    if (axis != i) {
      if (check_moved_outwards(i + 1, axis)) {
        single_axis_moved = true;
        to = i;
        from = axis;
      } else if (check_moved_inwards(i, to)) {
        single_axis_moved = true;
        from = i;
      }

      break;
    }
  }

  return single_axis_moved;
}

bool IsReshape(const std::vector<size_t>& perm, const std::vector<int64_t>& input_dims) {
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
Status TransposeBase::DoTranspose(const std::vector<size_t>& permutations, const Tensor& input, Tensor& output,
                                  const TensorShape* input_shape_override) {
  Status status = Status::OK();

  auto input_type = input.DataType();
  auto output_type = output.DataType();

  if (input_type != output_type) {
    status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Mismatched data types between input and output Tensors. ",
                             input_type, " != ", output_type);
  } else {
    TensorShape shape = input_shape_override ? *input_shape_override : input.Shape();
    if (IsReshape(permutations, shape.GetDims())) {
      // As long as the dims with values > 1 stay in the same order, it's a reshape.
      // Example: Shape=(1,1,1024,4096) -> perm=(2,0,3,1).
      CopyCpuTensor(&input, &output);
      return Status::OK();
    }

    size_t from = 0, to = 0;
    bool moving_single_axis = IsMovingSingleAxis(permutations, from, to);

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
  const std::vector<int64_t>& input_dims = input_shape.GetDims();
  size_t rank = input_dims.size();

  std::vector<int64_t> output_dims(rank);
  const std::vector<size_t>* p_perm;
  std::vector<size_t> default_perm(rank);
  Status status = ComputeOutputShape(X, output_dims, default_perm, p_perm);
  if (!status.IsOK())
    return status;

  TensorShape output_shape{output_dims};
  Tensor& Y = *ctx->Output(0, output_shape);

  if (output_shape.Size() == 0)
    return Status::OK();

  if (IsReshape(*p_perm, input_dims)) {
    // As long as the dims with values > 1 stay in the same order, it's a reshape.
    // Example: Shape=(1,1,1024,4096) -> perm=(2,0,3,1).
    CopyCpuTensor(&X, &Y);
    return Status::OK();
  }

  size_t from = 0, to = 0;
  bool moving_single_axis = IsMovingSingleAxis(*p_perm, from, to);

  if (moving_single_axis && !X.IsDataTypeString()) {
    SingleAxisTranspose(*p_perm, X, Y, from, to);
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
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Transpose);

ONNX_CPU_OPERATOR_KERNEL(
    Transpose,
    13,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Transpose);

}  // namespace onnxruntime

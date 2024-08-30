// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/copy.h"
#include "core/framework/element_type_lists.h"
#include "core/framework/transpose_helper.h"
#include "core/mlas/inc/mlas.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {

template <typename T>
struct has_mlas_transpose : std::false_type {};

template <>
struct has_mlas_transpose<uint8_t> : std::true_type {};

template <>
struct has_mlas_transpose<uint32_t> : std::true_type {};

// moving a single axis outwards where the read/write size is a power of 2 and between 8 and 64 bits.
template <typename T>
typename std::enable_if<!has_mlas_transpose<T>::value, void>::type SimpleTransposeSingleAxisOutwards(
    const T* input_data, T* output_data, int64_t num_loops, int64_t num_writers, int64_t writes_per_loop,
    int64_t writes_per_writer_per_loop) {
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

template <typename T>
typename std::enable_if<has_mlas_transpose<T>::value, void>::type SimpleTransposeSingleAxisOutwards(
    const T* input_data, T* output_data, int64_t num_loops, int64_t num_writers, int64_t writes_per_loop,
    int64_t writes_per_writer_per_loop) {
  for (int64_t l = 0; l < num_loops; ++l) {
    MlasTranspose(input_data, output_data, static_cast<size_t>(writes_per_writer_per_loop),
                  static_cast<size_t>(num_writers));
    input_data += writes_per_loop;
    output_data += writes_per_loop;
  }
}

//  `input_shape_override` overrides the shape of `input` for compute purposes.
void TransposeSingleAxisOutwards(gsl::span<const size_t> permutations, const Tensor& input, Tensor& output,
                                 size_t from, size_t to, const TensorShape* input_shape_override = nullptr,
                                 concurrency::ThreadPool* tp = nullptr) {
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
  // TODO: check integer overflow
  const size_t bytes_per_write = static_cast<size_t>(block_size) * element_size;

  switch (bytes_per_write) {
    case (sizeof(uint8_t)): {
      SimpleTransposeSingleAxisOutwards(input_data, output_data, num_loops, num_writers, writes_per_loop,
                                        writes_per_writer_per_loop);
      break;
    }
    case (sizeof(uint16_t)): {
      SimpleTransposeSingleAxisOutwards(reinterpret_cast<const uint16_t*>(input_data),
                                        reinterpret_cast<uint16_t*>(output_data), num_loops, num_writers,
                                        writes_per_loop, writes_per_writer_per_loop);
      break;
    }
    case (sizeof(uint32_t)): {
      SimpleTransposeSingleAxisOutwards(reinterpret_cast<const uint32_t*>(input_data),
                                        reinterpret_cast<uint32_t*>(output_data), num_loops, num_writers,
                                        writes_per_loop, writes_per_writer_per_loop);
      break;
    }
    case (sizeof(uint64_t)): {
      SimpleTransposeSingleAxisOutwards(reinterpret_cast<const uint64_t*>(input_data),
                                        reinterpret_cast<uint64_t*>(output_data), num_loops, num_writers,
                                        writes_per_loop, writes_per_writer_per_loop);
      break;
    }
    default: {
      TensorPitches src_strides(input_dims);

      TensorPitches contig_dst_strides(output);

      const auto dims = input_dims.size();
      TensorShapeVector dst_strides(dims);
      for (size_t dim = 0; dim < dims; ++dim) {
        dst_strides[permutations[dim]] = contig_dst_strides[dim];
      }

      ORT_THROW_IF_ERROR(DispatchStridedCopy<element_type_lists::All>(tp,
                                                                      output, 0, dst_strides,
                                                                      input_shape,
                                                                      input, 0, src_strides));
    }
  }
}

template <typename T>
typename std::enable_if<!has_mlas_transpose<T>::value, void>::type SimpleTransposeSingleAxisInwards(
    const T* input_data, T* output_data, int64_t num_loops, int64_t num_readers, int64_t reads_per_loop,
    int64_t reads_per_reader_per_loop) {
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

template <typename T>
typename std::enable_if<has_mlas_transpose<T>::value, void>::type SimpleTransposeSingleAxisInwards(
    const T* input_data, T* output_data, int64_t num_loops, int64_t num_readers, int64_t reads_per_loop,
    int64_t reads_per_reader_per_loop) {
  for (int64_t l = 0; l < num_loops; ++l) {
    MlasTranspose(input_data, output_data, static_cast<size_t>(num_readers),
                  static_cast<size_t>(reads_per_reader_per_loop));
    input_data += reads_per_loop;
    output_data += reads_per_loop;
  }
}

// moving a single axis inwards where the read/write size is a power of 2 and between 8 and 64 bits.
//  `input_shape_override` overrides the shape of `input` for compute purposes.
void TransposeSingleAxisInwards(gsl::span<const size_t> permutations, const Tensor& input, Tensor& output,
                                size_t from, size_t to, const TensorShape* input_shape_override = nullptr) {
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
  // TODO: check integer overflow
  const size_t bytes_per_read = static_cast<size_t>(block_size) * element_size;

  switch (bytes_per_read) {
    case (sizeof(uint8_t)): {
      SimpleTransposeSingleAxisInwards(input_data, output_data, num_loops, num_readers, reads_per_loop,
                                       reads_per_reader_per_loop);
      break;
    }
    case (sizeof(uint16_t)): {
      SimpleTransposeSingleAxisInwards(reinterpret_cast<const uint16_t*>(input_data),
                                       reinterpret_cast<uint16_t*>(output_data), num_loops, num_readers, reads_per_loop,
                                       reads_per_reader_per_loop);
      break;
    }
    case (sizeof(uint32_t)): {
      SimpleTransposeSingleAxisInwards(reinterpret_cast<const uint32_t*>(input_data),
                                       reinterpret_cast<uint32_t*>(output_data), num_loops, num_readers, reads_per_loop,
                                       reads_per_reader_per_loop);
      break;
    }
    case (sizeof(uint64_t)): {
      SimpleTransposeSingleAxisInwards(reinterpret_cast<const uint64_t*>(input_data),
                                       reinterpret_cast<uint64_t*>(output_data), num_loops, num_readers, reads_per_loop,
                                       reads_per_reader_per_loop);
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
void SingleAxisTranspose(gsl::span<const size_t> permutations, const Tensor& input, Tensor& output, size_t from,
                         size_t to, const TensorShape* input_shape_override, concurrency::ThreadPool* tp) {
  if (from > to) {
    TransposeSingleAxisOutwards(permutations, input, output, from, to, input_shape_override, tp);
  } else {
    TransposeSingleAxisInwards(permutations, input, output, from, to, input_shape_override);
  }
}

bool IsTransposeMovingSingleAxis(gsl::span<const size_t> permutations, size_t& from, size_t& to) {
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

}  // namespace onnxruntime
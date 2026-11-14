// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstring>

#if defined(__SSE2__) || defined(_M_X64)
#include <emmintrin.h>
#endif

#include "core/providers/cpu/tensor/pack_unpack.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {
namespace {

template <typename Word, bool Unpack, int Width = 0>
void ComputePacking(OpKernelContext* ctx, const Tensor& input, Tensor& output,
                    const packing::Parameters& p, const packing::Layout& l) {
  const uint64_t units = Unpack ? l.groups : l.words;
  const ptrdiff_t work = SafeInt<ptrdiff_t>(output.Shape().Size() / (Unpack ? l.count : l.words)) * units;
  const double cost = static_cast<double>(Unpack ? p.group_size : (sizeof(Word) * 8 / p.width + 2) * p.group_size);
  concurrency::ThreadPool::TryParallelFor(
      ctx->GetOperatorThreadPool(), work, cost,
      [&](ptrdiff_t begin, ptrdiff_t end) {
        for (ptrdiff_t i = begin; i < end; ++i) {
          const uint64_t index = static_cast<uint64_t>(i);
          const uint64_t inner = index % l.inner;
          const uint64_t unit = (index / l.inner) % units;
          const uint64_t outer = index / l.inner / units;
#if defined(__SSE2__) || defined(_M_X64)
          if constexpr (Width == 1) {
            if (l.inner == 1) {
              if constexpr (Unpack) {
                if (unit % 8 == 0 && l.count - unit >= 8 && end - i >= 8) {
                  constexpr int word_bits = sizeof(Word) * 8;
                  const auto* row = input.Data<Word>() + outer * l.words;
                  const auto byte = static_cast<char>(row[unit / word_bits] >> (unit % word_bits));
                  const __m128i masks = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, -128, 64, 32, 16, 8, 4, 2, 1);
                  const __m128i clear = _mm_cmpeq_epi8(_mm_and_si128(_mm_set1_epi8(byte), masks), _mm_setzero_si128());
                  _mm_storel_epi64(reinterpret_cast<__m128i*>(output.MutableData<uint8_t>() + index),
                                   _mm_andnot_si128(clear, _mm_set1_epi8(1)));
                  i += 7;
                  continue;
                }
              } else {
                constexpr int word_bits = sizeof(Word) * 8;
                const uint64_t first = unit * word_bits;
                if (l.count - first >= word_bits) {
                  const auto* codes = input.Data<uint8_t>() + outer * l.count + first;
                  if constexpr (sizeof(Word) == 1) {
                    const __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(codes));
                    output.MutableData<Word>()[index] = static_cast<Word>(_mm_movemask_epi8(_mm_slli_epi16(bytes, 7)));
                  } else {
                    const __m128i low = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes));
                    const __m128i high = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes + 16));
                    output.MutableData<Word>()[index] =
                        static_cast<Word>(_mm_movemask_epi8(_mm_slli_epi16(low, 7))) |
                        (static_cast<Word>(_mm_movemask_epi8(_mm_slli_epi16(high, 7))) << 16);
                  }
                  continue;
                }
              }
            }
          }
#endif
          if constexpr (Unpack) {
            packing::DecodeAndStoreGroup<Word, Width>(
                input.Data<Word>() + outer * l.words * l.inner + inner,
                output.MutableData<uint8_t>() + outer * l.count * l.inner + inner, unit, p, l);
          } else {
            output.MutableData<Word>()[index] = packing::EncodeWord<Word, Width>(
                input.Data<uint8_t>() + outer * l.count * l.inner + inner, unit, p, l);
          }
        }
      });
}

template <typename Word, bool Unpack>
void DispatchPacking(OpKernelContext* ctx, const Tensor& input, Tensor& output,
                     const packing::Parameters& p, const packing::Layout& l) {
  if (p.group_size == 1) {
    switch (p.width) {
      case 1:
        return ComputePacking<Word, Unpack, 1>(ctx, input, output, p, l);
      case 2:
        return ComputePacking<Word, Unpack, 2>(ctx, input, output, p, l);
      case 4:
        return ComputePacking<Word, Unpack, 4>(ctx, input, output, p, l);
      case 8:
        return ComputePacking<Word, Unpack, 8>(ctx, input, output, p, l);
    }
  }
  ComputePacking<Word, Unpack>(ctx, input, output, p, l);
}

template <bool Unpack>
class PackUnpack final : public OpKernel {
 public:
  explicit PackUnpack(const OpKernelInfo& info) : OpKernel(info), attributes_(info) {}

  Status Compute(OpKernelContext* ctx) const override {
    const auto& input = *ctx->Input<Tensor>(0);
    TensorShapeVector shape;
    packing::Layout layout{};
    ORT_RETURN_IF_ERROR(attributes_.Prepare(input, Unpack ? ctx->Input<Tensor>(1) : nullptr, Unpack, shape, layout));
    auto& output = *ctx->Output(0, TensorShape(shape));
    if (output.Shape().Size() == 0) return Status::OK();
    const auto& p = attributes_.parameters;
    const bool word32 = (Unpack ? input.GetElementType() : attributes_.output_type) == ONNX_NAMESPACE::TensorProto_DataType_UINT32;
    if (!word32 && p.group_size == 1 && p.radix == 256) {
      std::memcpy(output.MutableDataRaw(), input.DataRaw(), input.SizeInBytes());
    } else if (word32) {
      DispatchPacking<uint32_t, Unpack>(ctx, input, output, p, layout);
    } else {
      DispatchPacking<uint8_t, Unpack>(ctx, input, output, p, layout);
    }
    return Status::OK();
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(PackUnpack);
  const packing::Attributes attributes_;
};

using Pack = PackUnpack<false>;
using Unpack = PackUnpack<true>;
}  // namespace

ONNX_OPERATOR_KERNEL_EX(
    Pack, kOnnxDomain, 29, kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<uint8_t>()).TypeConstraint("TStorage", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<uint32_t>()}),
    Pack);

ONNX_OPERATOR_KERNEL_EX(
    Unpack, kOnnxDomain, 29, kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<uint8_t>()).TypeConstraint("TStorage", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<uint32_t>()}),
    Unpack);
}  // namespace onnxruntime

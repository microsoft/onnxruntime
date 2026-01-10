// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/common/narrow.h"
#include "core/common/safeint.h"
#include "core/common/utf8_util.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"
#include "re2/re2.h"

#ifdef _MSC_VER
#include <memory_resource>
#define ORT_PMR_ALLOCATOR_SUPPORTED
#endif

#include <optional>
#include <type_traits>
#include <vector>

#ifdef ORT_PMR_ALLOCATOR_SUPPORTED
using SlicesVector = std::pmr::vector<re2::StringPiece>;
#else
using SlicesVector = std::vector<re2::StringPiece>;
#endif

namespace onnxruntime {
namespace contrib {

class Tokenizer final : public OpKernel {
 public:
  explicit Tokenizer(const OpKernelInfo& info);
  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;
  ~Tokenizer() = default;

  Status Compute(OpKernelContext* context) const override;

 private:
  Status EstimateNumberOfTokens(gsl::span<const std::string> input_span,
                                size_t& max_tokens_per_row,
                                size_t& total_tokens_estimate) const;

  Status CharTokenize(OpKernelContext* context, size_t N, size_t C,
                      gsl::span<const int64_t> input_dims) const;

  Status SeparatorExpressionTokenizer(OpKernelContext* context, size_t N, size_t C,
                                      gsl::span<const int64_t> input_dims) const;

  Status TokenExpression(OpKernelContext* ctx,
                         size_t N, size_t C,
                         gsl::span<const int64_t> input_dims) const;

  void OutputData(gsl::span<const SlicesVector> rows,
                  size_t max_tokens, size_t max_output_index, std::string* output_data) const;

  bool mark_{false};
  std::string pad_value_;
  size_t mincharnum_{0};
  bool char_tokenezation_{false};
  InlinedVector<std::unique_ptr<re2::RE2>> separators_;
  std::unique_ptr<re2::RE2> regex_;
};

using namespace utf8_util;

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    Tokenizer,
    1,
    string,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<std::string>()),
    contrib::Tokenizer);

namespace tokenizer_details {
constexpr char kStartMarker = 0x2;
constexpr char kEndMarker = 0x3;
}  // namespace tokenizer_details

using namespace tokenizer_details;

Tokenizer::Tokenizer(const OpKernelInfo& info) : OpKernel(info) {
  int64_t mark = 0;
  auto status = info.GetAttr("mark", &mark);
  ORT_ENFORCE(status.IsOK(), "attribute mark is not set");
  mark_ = mark != 0;

  status = info.GetAttr("pad_value", &pad_value_);
  ORT_ENFORCE(status.IsOK(), "attribute pad_value is not set");

  int64_t mincharnum = 0;
  status = info.GetAttr("mincharnum", &mincharnum);
  ORT_ENFORCE(status.IsOK(), "attribute mincharnum is not set");
  ORT_ENFORCE(mincharnum > 0, "attribute mincharnum must have a positive value");
  mincharnum_ = narrow<size_t>(mincharnum);

  // Optional attributes either or
  std::vector<std::string> separators;
  std::string tokenexp;
  status = info.GetAttrs("separators", separators);
  if (!status.IsOK()) {
    status = info.GetAttr("tokenexp", &tokenexp);
    ORT_ENFORCE(status.IsOK(), "Either one of the separators OR tokenexp attributes required but none is set");
    ORT_ENFORCE(!tokenexp.empty(), "Expecting a non-empty tokenexp");
    char_tokenezation_ = (tokenexp == ".");
  } else {
    ORT_ENFORCE(!separators.empty(), "separators must not be empty");
    if (separators.size() == 1 && separators[0].empty()) {
      char_tokenezation_ = true;
    }
  }

  ORT_ENFORCE(!char_tokenezation_ || mincharnum_ < 2,
              "mincharnum is too big for char level tokenezation");

  // Check if we have separators or tokenexp
  if (!char_tokenezation_) {
    if (!separators.empty()) {
      re2::RE2::Options options;
      options.set_longest_match(true);
      for (const auto& sep : separators) {
        std::unique_ptr<re2::RE2> regex = std::make_unique<re2::RE2>(sep, options);
        if (!regex->ok()) {
          ORT_THROW("Can not digest separators: ", sep, " ", regex->error());
        }
        separators_.push_back(std::move(regex));
      }
    } else {
      // Use tokenexp
      assert(!tokenexp.empty());
      re2::RE2::Options options;
      options.set_longest_match(true);
      std::unique_ptr<re2::RE2> regex = std::make_unique<re2::RE2>(tokenexp, options);
      if (!regex->ok()) {
        ORT_THROW("Can not digest tokenexp: ", regex->error());
      }
      regex_.swap(regex);
    }
  }
}

Status Tokenizer::EstimateNumberOfTokens(gsl::span<const std::string> input_span,
                                         size_t& max_tokens_per_row, size_t& total_tokens_estimate) const {
  total_tokens_estimate = 0;
  max_tokens_per_row = 0;
  for (const auto& s : input_span) {
    size_t utf8_chars = 0;  // length in utf8 chars
    if (!utf8_validate(reinterpret_cast<const unsigned char*>(s.data()), s.size(),
                       utf8_chars)) {
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                    "Input string contains invalid utf8 chars: " + s);
    }
    auto tokens = std::max<size_t>(1, utf8_chars / mincharnum_);
    total_tokens_estimate += tokens;
    max_tokens_per_row = std::max(max_tokens_per_row, tokens);
  }

  return Status::OK();
}

Status Tokenizer::CharTokenize(OpKernelContext* ctx, size_t N, size_t C,
                               gsl::span<const int64_t> input_dims) const {
  // With char tokenzation we get as many tokens as the number of
  // utf8 characters in the string. So for every string we calculate its character(utf8) length
  // add padding and add start/end test separators if necessary
  size_t max_tokens = 0;
  auto X = ctx->Input<Tensor>(0);
  auto const input_data = X->Data<std::string>();
  auto curr_input = input_data;
  auto const last = input_data + N * C;
  while (curr_input != last) {
    const auto& s = *curr_input;
    size_t tokens = 0;  // length in utf8 chars
    if (!utf8_validate(reinterpret_cast<const unsigned char*>(s.data()), s.size(),
                       tokens)) {
      // Please do not include the input text in the error message as it could
      // be deemed as a compliance violation by teams using this operator
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input string contains invalid utf8 chars:", s);
    }
    max_tokens = std::max(max_tokens, tokens);
    ++curr_input;
  }

  TensorShapeVector output_dims(input_dims.begin(), input_dims.end());
  // Check if we have no output due to apparently empty strings input.
  if (max_tokens == 0) {
    output_dims.push_back(0);
    TensorShape output_shape(output_dims);
    ctx->Output(0, output_shape);
    return Status::OK();
  }

  if (mark_) {
    max_tokens += 2;  // Start/end markers as separate tokens
  }

  output_dims.push_back(max_tokens);
  TensorShape output_shape(output_dims);
  auto output_tensor = ctx->Output(0, output_shape);
  auto const output_data = output_tensor->MutableData<std::string>();
  size_t output_index = 0;
  curr_input = input_data;
  while (curr_input != last) {
    const auto& s = *curr_input;
    if (mark_) {
      output_data[output_index].assign(&kStartMarker, 1);
      ++output_index;
    }
    size_t tokens = 0;
    const size_t str_len = s.size();
    for (size_t token_idx = 0; token_idx < str_len;) {
      size_t tlen = 0;
      [[maybe_unused]] bool result = utf8_bytes(static_cast<unsigned char>(s[token_idx]), tlen);
      assert(result);
      assert(token_idx + tlen <= str_len);
      output_data[output_index] = s.substr(token_idx, tlen);
      ++output_index;
      token_idx += tlen;
      ++tokens;
    }
    if (mark_) {
      output_data[output_index].assign(&kEndMarker, 1);
      ++output_index;
    }
    // Padding strings
    assert(tokens + (static_cast<size_t>(mark_) * 2) <= max_tokens);
    const size_t pads = max_tokens - (static_cast<size_t>(mark_) * 2) - tokens;
    for (size_t p = 0; p < pads; ++p) {
      output_data[output_index] = pad_value_;
      ++output_index;
    }
    ++curr_input;
  }
  return Status::OK();
}

namespace {

// We use std::vector in this case, because InlinedVector::clear() is incompatible
// with std::vector. It also deallocates memory, which is not what we want.

// The compiler we are using GCC on Linux and Clang on MacOS does not
// have the library that support C++17 PMR. So we are only using it on Windows
// since the problem is acute on the platform.

#ifdef ORT_PMR_ALLOCATOR_SUPPORTED
/// <summary>
/// This class provides a thin abstraction over the std::pmr::monotonic_buffer_resource
/// If the allocated buffer is not enough, additional allocations are done using
/// new/delete.
/// </summary>
class MonotonicAllocatorWithDefault : public std::pmr::monotonic_buffer_resource {
 public:
  MonotonicAllocatorWithDefault(void* ptr, size_t size_in_bytes)
      : monotonic_buffer_resource(ptr, size_in_bytes, std::pmr::get_default_resource()) {}
  MonotonicAllocatorWithDefault(void* ptr, size_t size_in_bytes, std::pmr::memory_resource* upstream)
      : monotonic_buffer_resource(ptr, size_in_bytes, upstream) {}
};

class MemoryAllocator {
 public:
  explicit MemoryAllocator(size_t num_of_slices) {
    size_t allocated_size = 0;
    void* ptr = AlignedAllocate(num_of_slices, allocated_size);
    resource_.emplace(ptr, allocated_size);
  }

  SlicesVector CreateVectorWithAllocator() {
    return SlicesVector(&resource_.value());
  }

  SlicesVector& EmplaceBack(std::vector<SlicesVector>& rows) {
    return rows.emplace_back(&resource_.value());
  }

 private:
  /// <summary>
  /// Pre-allocate memory for the tokens to reduce a number of individual
  /// allocations and thus memory contention.
  /// Used in conjunction with PMR memory allocatior
  /// </summary>
  /// <param name="num">number of objects of T</param>
  /// <param name="buf">buffer holder</param>
  /// <param name="allocated_size">aligned allocated size</param>
  /// <returns>pointer to the buffer</returns>
  void* AlignedAllocate(size_t num, size_t& allocated_size) {
    constexpr size_t alignment = alignof(re2::StringPiece);
    const size_t size_bytes = SafeInt<size_t>(num) * sizeof(re2::StringPiece) + alignment;
    buf_holder_ = std::make_unique<uint8_t[]>(size_bytes);
    void* ptr = buf_holder_.get();
    allocated_size = size_bytes;
    return std::align(alignment, size_bytes, ptr, allocated_size);
  }

  std::unique_ptr<uint8_t[]> buf_holder_;
  std::optional<MonotonicAllocatorWithDefault> resource_;
};

#else

class MemoryAllocator {
 public:
  explicit MemoryAllocator(size_t /* num_of_slices */) {
  }

  SlicesVector CreateVectorWithAllocator() const {
    return SlicesVector{};
  }

  SlicesVector& EmplaceBack(std::vector<SlicesVector>& rows) const {
    return rows.emplace_back();
  }
};

#endif
}  // namespace

void Tokenizer::OutputData(gsl::span<const SlicesVector> rows,
                           size_t max_tokens, [[maybe_unused]] size_t max_output_index, std::string* output_data) const {
  size_t output_index = 0;
  for (const auto& row : rows) {
    [[maybe_unused]] size_t c_idx = output_index;
    if (mark_) {
      output_data[output_index++].assign(&kStartMarker, 1);
    }
    // Output tokens for this row
    for (const auto& token : row) {
      output_data[output_index++].assign(token.data(), token.length());
    }
    if (mark_) {
      output_data[output_index++].assign(&kEndMarker, 1);
    }
    const size_t pads = max_tokens - (static_cast<size_t>(mark_) * 2) - row.size();
    for (size_t p = 0; p < pads; ++p) {
      output_data[output_index++] = pad_value_;
    }
    assert(output_index <= max_output_index);
    assert((output_index - c_idx) <= max_tokens);
  }
}

Status Tokenizer::SeparatorExpressionTokenizer(OpKernelContext* ctx,
                                               size_t N, size_t C,
                                               gsl::span<const int64_t> input_dims) const {
  using namespace re2;

  auto X = ctx->Input<Tensor>(0);
  const auto input_span = X->DataAsSpan<std::string>();

  // Let's estimate maximum number of tokens
  // It is hard to estimate the number of separate characters that would not appear in the
  // output.
  size_t total_tokens_estimate = 0;
  size_t max_tokens_per_row = 0;
  ORT_RETURN_IF_ERROR(EstimateNumberOfTokens(input_span, max_tokens_per_row, total_tokens_estimate));
  // Add a scratch token vector allocation
  total_tokens_estimate += max_tokens_per_row;

  // Pre-allocate memory for all tokens (StringPieces)
  MemoryAllocator allocator(total_tokens_estimate);

  // Make sure the vectors below are destroyed before the allocator
  const size_t vector_num = SafeInt<size_t>(N) * C;

  std::vector<SlicesVector> rows;
  rows.reserve(vector_num);

  // Re-use the same vector for each tokenization round
  SlicesVector tokens = allocator.CreateVectorWithAllocator();
  tokens.reserve(max_tokens_per_row);

  // We do not constraint the search to match
  // on the beginning or end of the string
  constexpr RE2::Anchor anchor = RE2::UNANCHORED;

  // Scan all strings and attempt to find separators in them
  // collect all the output tokens here
  size_t max_tokens = 0;
  for (const auto& s : input_span) {
    size_t utf8_chars = 0;  // length in utf8 chars
    if (!utf8_len(reinterpret_cast<const unsigned char*>(s.data()), s.size(),
                  utf8_chars)) {
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                    "Input string contains invalid utf8 chars: " + s);
    }

    const auto expected_tokens = std::max<size_t>(1, utf8_chars / mincharnum_);
    auto& row = allocator.EmplaceBack(rows);
    row.reserve(expected_tokens);
    row.emplace_back(s);

    for (const auto& sep : separators_) {
      for (const auto& text : row) {
        const auto end_pos = text.length();
        size_t start_pos = 0;
        StringPiece submatch;

        bool match = true;
        do {
          match = sep->Match(text, start_pos, end_pos, anchor, &submatch, 1);
          if (match) {
            // Record  pos/len
            assert(submatch.data() != nullptr);
            size_t match_pos = submatch.data() - text.data();
            assert(match_pos >= start_pos);
            auto token_len = match_pos - start_pos;
            utf8_chars = 0;
            bool valid = utf8_len(reinterpret_cast<const unsigned char*>(text.data() + start_pos),
                                  token_len, utf8_chars);
            if (!valid) {
              return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                            "Match contains invalid utf8 chars: " + std::string{submatch});
            }
            if (utf8_chars >= mincharnum_) {
              tokens.emplace_back(text.data() + start_pos, token_len);
            }
            // Update starting position
            // Guard against empty string match
            auto match_len = submatch.length();
            if (match_len > 0) {
              start_pos = match_pos + match_len;
            } else {
              size_t bytes = 0;
              utf8_bytes(*submatch.data(), bytes);
              start_pos = match_pos + bytes;
            }
          } else {
            // record trailing token
            auto trailing_len = end_pos - start_pos;
            utf8_chars = 0;
            utf8_len(reinterpret_cast<const unsigned char*>(text.data() + start_pos),
                     trailing_len, utf8_chars);
            if (utf8_chars >= mincharnum_) {
              tokens.emplace_back(text.data() + start_pos, trailing_len);
            }
          }
        } while (match);
      }  // row

      // We want to preserve the buffer for the next separator
      // copying slices is cheaper than allocating new memory
      if (!tokens.empty()) {
        row = tokens;
        tokens.clear();
        continue;
      }

      // Nothing more to match for any remaining separators
      row.clear();
      tokens.clear();
      break;
    }  // separators_
    max_tokens = std::max(max_tokens, row.size());
  }

  TensorShapeVector output_dims(input_dims.begin(), input_dims.end());
  // Check if we have no output due to either empty input
  // or everything is a separator
  if (max_tokens == 0) {
    output_dims.push_back(0);
    TensorShape output_shape(output_dims);
    ctx->Output(0, output_shape);
    return Status::OK();
  }

  if (mark_) {
    max_tokens += 2;  // Start/end markers as separate tokens
  }

  output_dims.push_back(max_tokens);
  TensorShape output_shape(output_dims);

  auto output_tensor = ctx->Output(0, output_shape);
  auto const output_data = output_tensor->MutableData<std::string>();

  OutputData(rows, max_tokens, narrow<size_t>(output_shape.Size()), output_data);

  return Status::OK();
}

Status Tokenizer::TokenExpression(OpKernelContext* ctx,
                                  size_t N, size_t C,
                                  gsl::span<const int64_t> input_dims) const {
  using namespace re2;

  size_t max_tokens = 0;
  auto X = ctx->Input<Tensor>(0);
  const auto input_span = X->DataAsSpan<std::string>();

  // Let's estimate maximum number of tokens
  size_t total_tokens_estimate = 0;
  size_t max_tokens_per_row = 0;
  ORT_RETURN_IF_ERROR(EstimateNumberOfTokens(input_span, max_tokens_per_row, total_tokens_estimate));

  // Pre-allocate memory for all tokens (StringPieces)
  MemoryAllocator allocator(total_tokens_estimate);

  // Make sure the vectors below are destroyed before the allocator
  const size_t vector_num = SafeInt<size_t>(N) * C;

  // We use std::vector in this case, because InlinedVector::clear() is incompatible
  // with std::vector. It also deallocates memory, which is not what we want.
  std::vector<SlicesVector> rows;
  rows.reserve(vector_num);

  // We do not constraint the search to match
  // on the beginning or end of the string
  constexpr RE2::Anchor anchor = RE2::UNANCHORED;

  for (const auto& s : input_span) {
    size_t utf8_chars = 0;
    utf8_len(reinterpret_cast<const unsigned char*>(s.data()), s.size(), utf8_chars);

    auto& row = allocator.EmplaceBack(rows);

    if (utf8_chars >= mincharnum_) {
      auto estimated_tokens = std::max<size_t>(1, utf8_chars / mincharnum_);
      row.reserve(estimated_tokens);

      StringPiece text(s);
      const auto end_pos = s.length();
      size_t start_pos = 0;
      StringPiece submatch;

      bool match = true;
      do {
        match = regex_->Match(text, start_pos, end_pos, anchor, &submatch, 1);
        if (match) {
          // Record  pos/len
          assert(submatch.data() != nullptr);
          size_t match_pos = submatch.data() - s.data();
          assert(match_pos >= start_pos);
          // Guard against empty match and make
          // sure we make progress either way
          auto token_len = submatch.length();
          utf8_chars = 0;
          if (!utf8_len(reinterpret_cast<const unsigned char*>(submatch.data()), token_len, utf8_chars)) {
            return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                          "Match contains invalid utf8 chars: " + std::string{submatch});
          }
          if (utf8_chars >= mincharnum_) {
            row.push_back(submatch);
            start_pos = match_pos + token_len;
          } else {
            size_t bytes = 0;
            utf8_bytes(*submatch.data(), bytes);
            start_pos = match_pos + bytes;
          }
        }
      } while (match);
    }
    max_tokens = std::max(max_tokens, row.size());
  }

  // Check for empty output
  TensorShapeVector output_dims(input_dims.begin(), input_dims.end());
  // Check if we have no output due to either empty input
  // everything is a separator
  if (max_tokens == 0) {
    output_dims.push_back(0);
    TensorShape output_shape(output_dims);
    ctx->Output(0, output_shape);
    return Status::OK();
  }

  if (mark_) {
    max_tokens += 2;  // Start/end markers as separate tokens
  }

  output_dims.push_back(max_tokens);
  TensorShape output_shape(output_dims);

  auto output_tensor = ctx->Output(0, output_shape);
  auto const output_data = output_tensor->MutableData<std::string>();

  OutputData(rows, max_tokens, narrow<size_t>(output_shape.Size()), output_data);

  return Status::OK();
}

Status Tokenizer::Compute(OpKernelContext* ctx) const {
  // Get input buffer ptr
  auto X = ctx->Input<Tensor>(0);
  if (X == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  if (!X->IsDataTypeString()) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "tensor(string) expected as input");
  }

  auto& input_shape = X->Shape();
  auto input_dims = input_shape.GetDims();
  size_t N = 0;
  size_t C = 0;
  if (input_dims.size() == 1) {
    N = 1;
    C = narrow<size_t>(input_dims[0]);
  } else if (input_dims.size() == 2) {
    N = narrow<size_t>(input_dims[0]);
    C = narrow<size_t>(input_dims[1]);
  } else {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "Input dimensions are either [C] or [N][C] allowed");
  }

  // Empty input
  Status s;
  if (input_shape.Size() == 0) {
    std::vector<int64_t> output_dims;
    if (input_dims.size() == 2) {
      output_dims.push_back(input_dims[0]);
    }
    output_dims.push_back(0);

    TensorShape output_shape(output_dims);
    ctx->Output(0, output_shape);
    return s;
  }

  if (char_tokenezation_) {
    s = CharTokenize(ctx, N, C, input_dims);
  } else {
    if (!separators_.empty()) {
      s = SeparatorExpressionTokenizer(ctx, N, C, input_dims);
    } else {
      assert(regex_ != nullptr);
      s = TokenExpression(ctx, N, C, input_dims);
    }
  }
  return s;
}
}  // namespace contrib
}  // namespace onnxruntime

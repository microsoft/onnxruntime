// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/common/utf8_util.h"
#include "core/framework/tensor.h"
#include "core/framework/op_kernel.h"
#include "re2/re2.h"

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
  Status CharTokenize(OpKernelContext* context, size_t N, size_t C,
                      const std::vector<int64_t>& input_dims) const;

  Status SeparatorExpressionTokenizer(OpKernelContext* context, size_t N, size_t C,
                                      const std::vector<int64_t>& input_dims) const;

  Status TokenExpression(OpKernelContext* ctx,
                         size_t N, size_t C,
                         const std::vector<int64_t>& input_dims) const;

  bool mark_{false};
  std::string pad_value_;
  int64_t mincharnum_{0};
  bool char_tokenezation_{false};
  std::vector<std::unique_ptr<re2::RE2>> separators_;
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
const char start_text = 0x2;
const char end_text = 0x3;
}  // namespace tokenizer_details

using namespace tokenizer_details;

Tokenizer::Tokenizer(const OpKernelInfo& info) : OpKernel(info) {
  int64_t mark = 0;
  auto status = info.GetAttr("mark", &mark);
  ORT_ENFORCE(status.IsOK(), "attribute mark is not set");
  mark_ = mark != 0;

  status = info.GetAttr("pad_value", &pad_value_);
  ORT_ENFORCE(status.IsOK(), "attribute pad_value is not set");

  status = info.GetAttr("mincharnum", &mincharnum_);
  ORT_ENFORCE(status.IsOK(), "attribute mincharnum is not set");
  ORT_ENFORCE(mincharnum_ > 0, "attribute mincharnum must have a positive value");

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
        std::unique_ptr<re2::RE2> regex(new re2::RE2(sep, options));
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
      std::unique_ptr<re2::RE2> regex(new re2::RE2(tokenexp, options));
      if (!regex->ok()) {
        ORT_THROW("Can not digest tokenexp: ", regex->error());
      }
      regex_.swap(regex);
    }
  }
}

Status Tokenizer::CharTokenize(OpKernelContext* ctx, size_t N, size_t C,
                               const std::vector<int64_t>& input_dims) const {
  // With char tokenzation we get as many tokens as the number of
  // utf8 characters in the string. So for every string we calculate its character(utf8) length
  // add padding and add start/end test separators if necessary
  size_t max_tokens = 0;
  auto X = ctx->Input<Tensor>(0);
  auto const input_data = X->template Data<std::string>();
  auto curr_input = input_data;
  auto const last = input_data + N * C;
  while (curr_input != last) {
    const auto& s = *curr_input;
    size_t tokens = 0;  // length in utf8 chars
    if (!utf8_validate(reinterpret_cast<const unsigned char*>(s.data()), s.size(),
                       tokens)) {
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                    "Input string contains invalid utf8 chars: " + s);
    }
    max_tokens = std::max(max_tokens, tokens);
    ++curr_input;
  }

  std::vector<int64_t> output_dims(input_dims);
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
  auto const output_data = output_tensor->template MutableData<std::string>();
  size_t output_index = 0;
  curr_input = input_data;
  while (curr_input != last) {
    const auto& s = *curr_input;
    if (mark_) {
      (output_data + output_index)->assign(&start_text, 1);
      ++output_index;
    }
    size_t tokens = 0;
    const size_t str_len = s.size();
    for (size_t token_idx = 0; token_idx < str_len;) {
      size_t tlen = 0;
      bool result = utf8_bytes(static_cast<unsigned char>(s[token_idx]), tlen);
      assert(result);
      (void)result;
      assert(token_idx + tlen <= str_len);
      *(output_data + output_index) = s.substr(token_idx, tlen);
      ++output_index;
      token_idx += tlen;
      ++tokens;
    }
    if (mark_) {
      (output_data + output_index)->assign(&end_text, 1);
      ++output_index;
    }
    // Padding strings
    assert(tokens + (mark_ * 2) <= max_tokens);
    const size_t pads = max_tokens - (mark_ * 2) - tokens;
    for (size_t p = 0; p < pads; ++p) {
      *(output_data + output_index) = pad_value_;
      ++output_index;
    }
    ++curr_input;
  }
  return Status::OK();
}

Status Tokenizer::SeparatorExpressionTokenizer(OpKernelContext* ctx,
                                               size_t N, size_t C,
                                               const std::vector<int64_t>& input_dims) const {
  using namespace re2;
  std::vector<std::vector<StringPiece>> rows;
  rows.reserve(N * C);

  // We do not constraint the search to match
  // on the beginning or end of the string
  const RE2::Anchor anchor = RE2::UNANCHORED;

  // Scan all strings and attempt to find separators in them
  // collect all the output tokens here
  size_t max_tokens = 0;
  auto X = ctx->Input<Tensor>(0);
  auto const input_data = X->template Data<std::string>();
  auto curr_input = input_data;
  auto const last = input_data + N * C;
  while (curr_input != last) {
    const auto& s = *curr_input;
    size_t utf8_chars = 0;  // length in utf8 chars
    if (!utf8_validate(reinterpret_cast<const unsigned char*>(s.data()), s.size(),
                       utf8_chars)) {
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                    "Input string contains invalid utf8 chars: " + s);
    }

    std::vector<StringPiece> row{s};

    for (const auto& sep : separators_) {
      std::vector<StringPiece> tokens;
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
                            "Match contains invalid utf8 chars: " + submatch.as_string());
            }
            if (utf8_chars >= size_t(mincharnum_)) {
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
            if (utf8_chars >= size_t(mincharnum_)) {
              tokens.emplace_back(text.data() + start_pos, trailing_len);
            }
          }
        } while (match);
      }  // row
      // Replace the row with the results of this tokenezation
      row.swap(tokens);
    }  // separators_
    max_tokens = std::max(max_tokens, row.size());
    rows.push_back(std::move(row));
    ++curr_input;
  }

  std::vector<int64_t> output_dims(input_dims);
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
  auto const output_data = output_tensor->template MutableData<std::string>();

#ifdef _DEBUG
  const size_t max_output_index = N * C * max_tokens;
#endif
  size_t output_index = 0;
  curr_input = input_data;
  for (auto& row : rows) {
#ifdef _DEBUG
    size_t c_idx = output_index;
#endif
    if (mark_) {
      (output_data + output_index)->assign(&start_text, 1);
      ++output_index;
    }
    // Output tokens for this row
    for (const auto& token : row) {
      (output_data + output_index)->assign(token.data(), token.size());
      ++output_index;
    }
    if (mark_) {
      (output_data + output_index)->assign(&end_text, 1);
      ++output_index;
    }
    const size_t pads = max_tokens - (mark_ * 2) - row.size();
    for (size_t p = 0; p < pads; ++p) {
      *(output_data + output_index) = pad_value_;
      ++output_index;
    }
#ifdef _DEBUG
    assert(output_index <= max_output_index);
    assert((output_index - c_idx) <= max_tokens);
#endif
    ++curr_input;
  }
  return Status::OK();
}

Status Tokenizer::TokenExpression(OpKernelContext* ctx,
                                  size_t N, size_t C,
                                  const std::vector<int64_t>& input_dims) const {
  using namespace re2;
  // Represents a token that will be output after
  // first is the index, second is the size;
  std::vector<std::vector<StringPiece>> tokens;
  tokens.reserve(N * C);

  size_t max_tokens = 0;
  auto X = ctx->Input<Tensor>(0);
  auto const input_data = X->template Data<std::string>();
  auto curr_input = input_data;
  auto const last = input_data + N * C;

  // We do not constraint the search to match
  // on the beginning or end of the string
  const RE2::Anchor anchor = RE2::UNANCHORED;

  while (curr_input != last) {
    const auto& s = *curr_input;

    size_t utf8_chars = 0;
    if (!utf8_validate(reinterpret_cast<const unsigned char*>(s.data()), s.size(),
                       utf8_chars)) {
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                    "Input string contains invalid utf8 chars: " + s);
    }

    tokens.emplace_back();
    auto& row = tokens.back();

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
                        "Match contains invalid utf8 chars: " + submatch.as_string());
        }
        if (utf8_chars >= size_t(mincharnum_)) {
          row.push_back(submatch);
          start_pos = match_pos + token_len;
        } else {
          size_t bytes = 0;
          utf8_bytes(*submatch.data(), bytes);
          start_pos = match_pos + bytes;
        }
      }
    } while (match);
    max_tokens = std::max(max_tokens, row.size());
    ++curr_input;
  }

  // Check for empty output
  std::vector<int64_t> output_dims(input_dims);
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
  auto const output_data = output_tensor->template MutableData<std::string>();

#ifdef _DEBUG
  const size_t max_output_index = N * C * max_tokens;
#endif
  curr_input = input_data;
  size_t output_index = 0;
  for (const auto& row : tokens) {
    assert(curr_input != last);
#ifdef _DEBUG
    size_t c_idx = output_index;
#endif
    if (mark_) {
      (output_data + output_index)->assign(&start_text, 1);
      ++output_index;
    }
    // Output tokens for this row
    for (const auto& token : row) {
      (output_data + output_index)->assign(token.data(), token.length());
      ++output_index;
    }
    if (mark_) {
      (output_data + output_index)->assign(&end_text, 1);
      ++output_index;
    }
    const size_t pads = max_tokens - (mark_ * 2) - row.size();
    for (size_t p = 0; p < pads; ++p) {
      *(output_data + output_index) = pad_value_;
      ++output_index;
    }
#ifdef _DEBUG
    assert(output_index <= max_output_index);
    assert((output_index - c_idx) <= max_tokens);
#endif
    ++curr_input;
  }

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
  auto& input_dims = input_shape.GetDims();
  size_t N = 0;
  size_t C = 0;
  if (input_dims.size() == 1) {
    N = 1;
    C = gsl::narrow<size_t>(input_dims[0]);
  } else if (input_dims.size() == 2) {
    N = gsl::narrow<size_t>(input_dims[0]);
    C = gsl::narrow<size_t>(input_dims[1]);
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

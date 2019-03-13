// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/tensor.h"
#include "core/framework/op_kernel.h"
#include "core/graph/onnx_protobuf.h"
#include "onnx/defs/schema.h"

#include "core/common/utf8_util.h"
#include "re2/re2.h"

#include <codecvt>
#include <locale>

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

  Status SeparatorTokenize(OpKernelContext* context, size_t N, size_t C,
                           const std::vector<int64_t>& input_dims) const;

  Status ExpressionTokenize(OpKernelContext* ctx,
                            size_t N, size_t C,
                            const std::vector<int64_t>& input_dims) const;

  bool mark_;
  std::string pad_value_;
  int64_t mincharnum_;
  bool char_tokenezation_;
  struct SearchData;
  std::unique_ptr<SearchData> search_data_;
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

const std::string conv_error("Conversion Error");
const std::wstring wconv_error(L"Conversion Error");

// Use a Trie like structure for searching multiple strings
// at once but convert it to a ternary tree for saving space.
// We insert separators in the same order they are specified.
// Template parameter is a CharT which can be a char/wchar_t
// or anything else that supports operator ><,== as long as
// this is not a variable length sequence. We convert utf8 to utf16
// before inserting.
// Value is a supplementary information useful for search hit
// and is present in the nodes that terminate the whole search pattern
template <class CharT, class Value>
class TernarySearchTree {
 private:
  struct Node {
    std::unique_ptr<Node> left_;
    std::unique_ptr<Node> mid_;
    std::unique_ptr<Node> right_;
    CharT c_;  // character
    Value value_;
    bool has_val_;

    explicit Node(CharT c) : c_(c), value_(), has_val_(false) {
    }
    ~Node() = default;
  };

  struct GetState {
    const CharT* const str_;
    const size_t len_;
    size_t depth_;
    const Value* result_;
  };

 public:
  TernarySearchTree() = default;
  ~TernarySearchTree() = default;

  /**
  * Returns a ptr to an associated value and nullptr on search miss.
  * Must use default constructed state.
  */
  const Value* get(const CharT* str, size_t len) const {
    if (len == 0) {
      return nullptr;
    }
    GetState get_state{str, len, 0, nullptr};
    get(root_.get(), get_state);
    return get_state.result_;
  }

  /**
  * Returns true if successful and false on empty strings
  * and duplicates.
  */
  bool put(const CharT* str, size_t len, const Value& v) {
    if (len < 1) {
      assert(false);
      return false;
    }
    Node* new_root = put(root_.get(), str, len, v, 0);
    if (new_root != nullptr) {
      root_.release();
      root_.reset(new_root);
      return true;
    }
    return false;
  }

 private:
  void update_state(const Node* node, GetState& state) const {
    if (node->has_val_) {
      if (state.result_ == nullptr) {
        state.result_ = &node->value_;
      } else if (node->value_ < *state.result_) {
        state.result_ = &node->value_;
      }
    }
  }
  void get(const Node* node, GetState& state) const {
    if (node == nullptr) {
      return;
    }
    assert(state.depth_ < state.len_);
    CharT c = state.str_[state.depth_];
    if (c < node->c_) {
      get(node->left_.get(), state);
      return;
    } else if (c > node->c_) {
      get(node->right_.get(), state);
      return;
    } else if (state.depth_ < (state.len_ - 1)) {
      // Check if we have a match at this node
      update_state(node, state);
      if (node->mid_ != nullptr) {
        ++state.depth_;
        get(node->mid_.get(), state);
      }
      return;
    }
    update_state(node, state);
  }

  Node* put(Node* node, const CharT* str, size_t len, const Value& v, size_t depth) {
    CharT c = str[depth];

    std::unique_ptr<Node> new_node;
    if (node == nullptr) {
      new_node.reset(new Node(c));
    }

    Node* new_link = nullptr;
    Node* n = (node != nullptr) ? node : new_node.get();
    if (c < n->c_) {
      new_link = put(n->left_.get(), str, len, v, depth);
      if (new_link != nullptr) {
        n->left_.release();
        n->left_.reset(new_link);
      }
    } else if (c > n->c_) {
      new_link = put(n->right_.get(), str, len, v, depth);
      if (new_link != nullptr) {
        n->right_.release();
        n->right_.reset(new_link);
      }
    } else if (depth < (len - 1)) {
      new_link = put(n->mid_.get(), str, len, v, depth + 1);
      if (new_link != nullptr) {
        n->mid_.release();
        n->mid_.reset(new_link);
      }
    } else {
      if (!n->has_val_) {
        n->value_ = v;
        n->has_val_ = true;
        new_link = n;
      }
    }
    if (new_link != nullptr) {
      new_node.release();
      return n;
    }
    return nullptr;
  }
  std::unique_ptr<Node> root_;
};

// We store the length of the original pattern within the
// Ternary Tree. This allows us to cut out the length of the matching
// separator from the original string.
struct SearchValue {
  size_t w_len;
  int priority_;
  bool operator<(const SearchValue& o) const {
    return priority_ < o.priority_;
  }
};

}  // namespace tokenizer_details

using namespace tokenizer_details;

struct Tokenizer::SearchData {
  TernarySearchTree<wchar_t, SearchValue> tst_;
};

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
  } else {
    ORT_ENFORCE(!separators.empty(), "Expect at least one item within separators");
  }

  char_tokenezation_ = (separators.size() == 1 &&
                        separators[0].empty());

  ORT_ENFORCE(!char_tokenezation_ || mincharnum_ < 2,
              "mincharnum is too big for char level tokenezation");

  // Check if we have separators or tokenexp
  if (!char_tokenezation_) {
    if (!separators.empty()) {
      std::unique_ptr<SearchData> sd(std::make_unique<SearchData>());
      std::wstring_convert<std::codecvt_utf8<wchar_t>> converter(conv_error, wconv_error);
      int priority = 0;  // earlier search patterns get priority
      for (const auto& sep : separators) {
        ORT_ENFORCE(!sep.empty(), "No empty separators allowed");
        std::wstring wsep = converter.from_bytes(sep);
        ORT_ENFORCE(wsep != wconv_error, "Separator strings contains invalid utf8 chars");
        bool result = sd->tst_.put(wsep.c_str(), wsep.length(), {wsep.length(), priority});
        ORT_ENFORCE(result, "duplicate separator detected");
        ++priority;
      }
      search_data_.swap(sd);
    } else {
      // Use tokenexp
      re2::RE2::Options options;
      options.set_longest_match(true);
      options.set_posix_syntax(true);
      std::unique_ptr<re2::RE2> regex(new re2::RE2(tokenexp, options));
      if (!regex->ok()) {
        ORT_THROW("Can not digest regex: ", regex->error());
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

Status Tokenizer::SeparatorTokenize(OpKernelContext* ctx,
                                    size_t N, size_t C,
                                    const std::vector<int64_t>& input_dims) const {
  struct Match {
    int priority_;
    size_t offset_;
    size_t size_;
    // create a conflict for overlapping matches
    // thus if they overlap neither is less than the other
    // and they are considered equal
    bool operator<(const Match& o) const {
      return (offset_ + size_) <= o.offset_;
    }
  };

  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter(conv_error, wconv_error);
  // Scan all strings and attempt to find separators in them
  // collect all the output tokens here
  size_t max_tokens = 0;
  std::vector<std::vector<std::wstring>> tokenized_strings;
  tokenized_strings.reserve(N * C);
  auto X = ctx->Input<Tensor>(0);
  auto const input_data = X->template Data<std::string>();
  auto curr_input = input_data;
  auto const last = input_data + N * C;
  while (curr_input != last) {
    const auto& s = *curr_input;
    std::wstring wstr = converter.from_bytes(s);
    if (wstr == wconv_error) {
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                    "Invalid utf8 chars in the input: " + s);
    }

    std::set<Match> matches;
    const wchar_t* ws = wstr.c_str();
    size_t len_remaining = wstr.length();
    size_t offset = 0;
    while (len_remaining > 0) {
      const auto* val = search_data_->tst_.get(ws, len_remaining);
      if (val != nullptr) {
        auto p = matches.insert({val->priority_, offset, val->w_len});
        while (!p.second && val->priority_ < p.first->priority_) {
          // if overlapping matches of the same pattern(priority), then
          // the earlier match naturally wins
          matches.erase(p.first);
          p = matches.insert({val->priority_, offset, val->w_len});
        }
      }
      ++ws;
      ++offset;
      --len_remaining;
    }

    // Tokenize
    tokenized_strings.emplace_back();
    auto& row_tokens = tokenized_strings.back();
    row_tokens.reserve(matches.size() + 1);
    ws = wstr.c_str();
    offset = 0;
    for (const auto& m : matches) {
      assert(m.offset_ >= offset);
      size_t sz = (m.offset_ - offset);
      if (sz > 0 && sz >= size_t(mincharnum_)) {
        row_tokens.emplace_back(ws, sz);
      }
      offset = m.offset_ + m.size_;
      ws = wstr.c_str() + offset;
    }
    assert(offset <= wstr.length());
    if (offset < wstr.length()) {
      row_tokens.emplace_back(ws, wstr.length() - offset);
    }

    max_tokens = std::max(max_tokens, row_tokens.size());
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
  for (auto& row : tokenized_strings) {
#ifdef _DEBUG
    size_t c_idx = output_index;
#endif
    if (mark_) {
      (output_data + output_index)->assign(&start_text, 1);
      ++output_index;
    }
    // Output tokens for this row
    for (auto& token : row) {
      *(output_data + output_index) = converter.to_bytes(token);
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
  }
  return Status::OK();
}

Status Tokenizer::ExpressionTokenize(OpKernelContext* ctx,
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
        if (token_len > 0) {
          row.push_back(submatch);
          start_pos = match_pos + token_len;
        } else {
          start_pos = match_pos + 1;
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
  if (X->DataType() != DataTypeImpl::GetType<std::string>()) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "tensor(string) expected as input");
  }

  auto& input_dims = X->Shape().GetDims();
  size_t N = 0;
  size_t C = 0;
  if (input_dims.size() == 1) {
    N = 1;
    if (input_dims[0] < 1) {
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                    "Invalid C dimension value");
    }
    C = input_dims[0];
  } else if (input_dims.size() == 2) {
    if (input_dims[0] < 1 || input_dims[1] < 1) {
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                    "Invalid N and/or C dimension values");
    }
    N = input_dims[0];
    C = input_dims[1];
  } else {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "Input dimensions are either [C] or [N][C] allowed");
  }

  Status s;
  if (char_tokenezation_) {
    s = CharTokenize(ctx, N, C, input_dims);
  } else {
    if (regex_ != nullptr) {
      s = ExpressionTokenize(ctx, N, C, input_dims);
    } else {
      s = SeparatorTokenize(ctx, N, C, input_dims);
    }
  }
  return s;
}
}  // namespace contrib
}  // namespace onnxruntime

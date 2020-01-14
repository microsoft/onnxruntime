// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "tfidfvectorizer.h"
#include "onnx/defs/schema.h"
#include "core/common/common.h"
#include "core/framework/tensor.h"

#include <functional>
#include <unordered_set>
#include <ostream>
#include <iterator>

namespace onnxruntime {

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    TfIdfVectorizer,
    9,
    string,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<std::string>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>()),
    TfIdfVectorizer);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    TfIdfVectorizer,
    9,
    int32_t,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>()),
    TfIdfVectorizer);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    TfIdfVectorizer,
    9,
    int64_t,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>()),
    TfIdfVectorizer);

namespace ngram_details {

class NgramEntryBase {
  size_t id_;  // Id in the pool
 protected:
  NgramEntryBase(size_t id) : id_(id) {}
  ~NgramEntryBase() = default;

 public:
  size_t Id() const { return id_; }
};

template <class T>
class NgramEntry;

template <>
class NgramEntry<int64_t> : public NgramEntryBase {
  std::vector<int64_t> items_;
  size_t hash_ = 0;

  void RunningHash(int64_t v) {
    std::hash<int64_t> hf;
    hash_ ^= hf(v) + 0x9e3779b9 + (hash_ << 6) + (hash_ >> 2);
  }

 public:
  template <typename ForwardIter>
  explicit NgramEntry(size_t id, ForwardIter first, ForwardIter last) : NgramEntryBase(id) {
    while (first != last) {
      RunningHash(*first);
      items_.push_back(*first);
      ++first;
    }
    assert(!items_.empty());
  }
  // For sampling
  explicit NgramEntry() : NgramEntryBase(0) {}
  void AddItem(int64_t v) {
    items_.push_back(v);
    RunningHash(v);
  }
  void DebugPrint() const {
    std::copy(items_.cbegin(), items_.cend(), std::ostream_iterator<int64_t>(std::cout, ","));
    std::cout << std::endl;
  }
  void Clear() {
    items_.clear();
    hash_ = 0;
  }
  bool operator==(const NgramEntry& o) const {
    return items_ == o.items_;
  }
  size_t Hash() const {
    return hash_;
  }
};

template <>
class NgramEntry<int32_t> : public NgramEntry<int64_t> {
 public:
  template <typename ForwardIter>
  explicit NgramEntry(size_t id, ForwardIter first, ForwardIter last) : NgramEntry<int64_t>(id, first, last) {}
  explicit NgramEntry() = default;
};

template <>
class NgramEntry<std::string> : public NgramEntryBase {
 private:
  std::vector<std::reference_wrapper<const std::string>> items_;
  size_t hash_ = 0;

  void RunningHash(const std::string& s) {
    std::hash<std::string> hf;
    hash_ ^= hf(s) + 0x9e3779b9 + (hash_ << 6) + (hash_ >> 2);
  }

 public:
  template <typename ForwardIter>
  explicit NgramEntry(size_t id, ForwardIter first, ForwardIter last) : NgramEntryBase(id) {
    while (first != last) {
      RunningHash(*first);
      items_.push_back(std::cref(*first));
      ++first;
    }
    assert(!items_.empty());
  }
  explicit NgramEntry() : NgramEntryBase(0) {}
  void AddItem(const std::string& s) {
    items_.push_back(std::cref(s));
    RunningHash(s);
  }
  void DebugPrint() const {
    std::copy(items_.cbegin(), items_.cend(), std::ostream_iterator<std::string>(std::cout, ","));
    std::cout << std::endl;
  }
  void Clear() {
    items_.clear();
    hash_ = 0;
  }

  bool operator==(const NgramEntry& o) const {
    if (items_.size() == o.items_.size()) {
      std::equal_to<std::string> pred;
      for (size_t i = 0; i < items_.size(); ++i) {
        if (!pred(items_[i], o.items_[i])) {
          return false;
        }
      }
      return true;
    }
    return false;
  }
  size_t Hash() const {
    return hash_;
  }
};

template <typename ForwardIter, typename Cont>
inline void Emplace(ForwardIter first, size_t ngrams, size_t ngram_size, size_t& ngram_id, Cont& c) {
  for (; ngrams > 0; --ngrams) {
    c.emplace(ngram_id, first, first + ngram_size);
    first += ngram_size;
    ++ngram_id;
  }
}

}  // namespace ngram_details
}  // namespace onnxruntime

using namespace onnxruntime::ngram_details;

namespace std {
template <typename T>
struct hash<NgramEntry<T>> {
  typedef NgramEntry<T> argument_type;
  typedef size_t result_type;
  result_type operator()(const argument_type& a) const {
    return a.Hash();
  }
};
}  // namespace std

namespace onnxruntime {

using IntegerPoolSet = std::unordered_set<NgramEntry<int64_t>>;
// Does not own strings, contains references to them. This helps
// to search by string references that point to the current input.
using StringPoolSet = std::unordered_set<NgramEntry<std::string>>;

template <typename T>
struct Return;

template <>
struct Return<int64_t> {
  using type = IntegerPoolSet::const_iterator;
};

template <>
struct Return<int32_t> : Return<int64_t> {};

template <>
struct Return<std::string> {
  using type = StringPoolSet::const_iterator;
};

// The weighting criteria.
// "TF"(term frequency),
//    the counts are propagated to output
// "IDF"(inverse document frequency),
//    all the counts larger than 1
//    would be truncated to 1 and the i-th element
//    in weights would be used to scale (by multiplication)
//    the count of the i-th n-gram in pool
// "TFIDF" (the combination of TF and IDF).
//  counts are scaled by the associated values in the weights attribute.

enum WeightingCriteria {
  kNone = 0,
  kTF = 1,
  kIDF = 2,
  kTFIDF = 3
};

struct TfIdfVectorizer::Impl {
  WeightingCriteria weighting_criteria_ = kNone;
  int64_t max_gram_length_ = 0;
  int64_t min_gram_length_ = 0;
  int64_t max_skip_count_ = 0;
  // This is the content of ngram_counts attribute.
  // The starting indexes of 1-grams, 2-grams,
  // and so on in pool. For example, if ngram_counts is [0, 17, 36],
  // the first index (zero-based) of 1-gram/2-gram/3-gram
  // in pool are 0/17/36.
  std::vector<int64_t> ngram_counts_;
  // Contains output indexes
  // represents ngram_indexes output
  std::vector<int64_t> ngram_indexes_;
  std::vector<float> weights_;

  std::vector<std::string> pool_strings_;
  // This set contains references to pool_string_ entries
  // of pool_strings attribute
  StringPoolSet str_set_;
  // This set contains pool_int64s entries
  IntegerPoolSet int64_set_;
  size_t output_size_ = 0;

  Impl() = default;
  ~Impl() = default;
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;

  template <typename T>
  typename Return<T>::type PoolEnd() const;

  template <typename T>
  typename Return<T>::type PoolFind(const ngram_details::NgramEntry<T>&) const;

  void IncrementCount(size_t ngram_id, size_t row_num,
                      std::vector<uint32_t>& frequencies) const {
    assert(ngram_id < ngram_indexes_.size());
    auto output_idx = row_num * output_size_ + ngram_indexes_[ngram_id];
    assert(static_cast<size_t>(output_idx) < frequencies.size());
    ++frequencies[output_idx];
  }
};

template <>
inline Return<int64_t>::type TfIdfVectorizer::Impl::PoolEnd<int64_t>() const {
  return int64_set_.cend();
}

template <>
inline Return<int32_t>::type TfIdfVectorizer::Impl::PoolEnd<int32_t>() const {
  return PoolEnd<int64_t>();
}

template <>
inline Return<std::string>::type TfIdfVectorizer::Impl::PoolEnd<std::string>() const {
  return str_set_.cend();
}

template <>
inline Return<int64_t>::type TfIdfVectorizer::Impl::PoolFind<int64_t>(const NgramEntry<int64_t>& i) const {
  return int64_set_.find(i);
}

template <>
inline Return<int32_t>::type TfIdfVectorizer::Impl::PoolFind<int32_t>(const NgramEntry<int32_t>& i) const {
  return int64_set_.find(i);
}

template <>
inline Return<std::string>::type TfIdfVectorizer::Impl::PoolFind<std::string>(const NgramEntry<std::string>& i) const {
  return str_set_.find(i);
}

TfIdfVectorizer::TfIdfVectorizer(const OpKernelInfo& info) : OpKernel(info), impl_(new Impl) {
  std::string mode;
  Status status = info.GetAttr("mode", &mode);
  ORT_ENFORCE(status.IsOK(), "mode is required");
  if (mode == "TF") {
    impl_->weighting_criteria_ = kTF;
  } else if (mode == "IDF") {
    impl_->weighting_criteria_ = kIDF;
  } else if (mode == "TFIDF") {
    impl_->weighting_criteria_ = kTFIDF;
  }
  ORT_ENFORCE(impl_->weighting_criteria_ != kNone, "mode: ", mode, " is unrecognized, acceptable values are TF,IDF,TFIDF");

  status = info.GetAttr("min_gram_length", &impl_->min_gram_length_);
  ORT_ENFORCE(status.IsOK(), "min_gram_length is required");
  ORT_ENFORCE(impl_->min_gram_length_ > 0, "Required min_gram_length must be positive: ", std::to_string(impl_->min_gram_length_));

  status = info.GetAttr("max_gram_length", &impl_->max_gram_length_);
  ORT_ENFORCE(status.IsOK(), "min_gram_length is required");
  ORT_ENFORCE(impl_->max_gram_length_ >= impl_->min_gram_length_,
              "min_gram_length >= max_gram_length required: ",
              std::to_string(impl_->max_gram_length_), " >= ", std::to_string(impl_->min_gram_length_));

  status = info.GetAttr("max_skip_count", &impl_->max_skip_count_);
  ORT_ENFORCE(status.IsOK(), "max_skip_count is required");
  ORT_ENFORCE(impl_->max_skip_count_ >= 0, "max_skip_count must be non-negative: ", std::to_string(impl_->max_skip_count_));

  status = info.GetAttrs(std::string("ngram_counts"), impl_->ngram_counts_);
  ORT_ENFORCE(status.IsOK() && !impl_->ngram_counts_.empty(), "Non-empty ngram_counts is required");
  ORT_ENFORCE(size_t(impl_->min_gram_length_) <= impl_->ngram_counts_.size(),
              "min_gram_length must be inbounds of ngram_counts: ",
              std::to_string(impl_->min_gram_length_), " <= ", std::to_string(impl_->ngram_counts_.size()));
  ORT_ENFORCE(size_t(impl_->max_gram_length_) <= impl_->ngram_counts_.size(),
              "max_gram_length must be inbounds of ngram_counts: ",
              std::to_string(impl_->max_gram_length_), " <= ", std::to_string(impl_->ngram_counts_.size()));

  status = info.GetAttrs("ngram_indexes", impl_->ngram_indexes_);
  ORT_ENFORCE(status.IsOK() && !impl_->ngram_indexes_.empty(), "Non-empty ngram_indexes is required");
  {
    // Check that all are positive
    ORT_ENFORCE(std::all_of(impl_->ngram_indexes_.cbegin(), impl_->ngram_indexes_.cend(),
                            [](int64_t i) { return i >= 0; }),
                "Negative ngram_indexes values are not allowed");
    // Set output size to max output index + 1;
    auto greatest_hit = std::max_element(impl_->ngram_indexes_.cbegin(), impl_->ngram_indexes_.cend());
    impl_->output_size_ = *greatest_hit + 1;
  }

  status = info.GetAttrs("weights", impl_->weights_);
  if (status.IsOK()) {
    ORT_ENFORCE(impl_->weights_.size() == impl_->ngram_indexes_.size(),
                "Got weights of size: ", std::to_string(impl_->weights_.size()),
                " but ngram_indexes size: ", std::to_string(impl_->ngram_indexes_.size()),
                " must be of equal size");
  }

  std::vector<int64_t> pool_int64s;
  status = info.GetAttrs("pool_strings", impl_->pool_strings_);
  if (status.IsOK()) {
    ORT_ENFORCE(!impl_->pool_strings_.empty(), "pool_strings must not be empty if specified");
  } else {
    status = info.GetAttrs("pool_int64s", pool_int64s);
    ORT_ENFORCE(status.IsOK() && !pool_int64s.empty(), "non-empty pool_int64s is required if pool_strings not provided");
  }

  // Iterator via the pool. Insert 1 item for 1-grams, 2 items for 2-grams, etc.
  const auto total_items = (impl_->pool_strings_.empty()) ? pool_int64s.size() : impl_->pool_strings_.size();
  size_t ngram_id = 0;
  // Load into dictionary only required gram sizes
  const size_t min_gram_length = impl_->min_gram_length_;
  const size_t max_gram_length = impl_->max_gram_length_;
  size_t ngram_size = 1;
  for (size_t i = 0; i < impl_->ngram_counts_.size(); ++i) {
    size_t start_idx = impl_->ngram_counts_[i];
    size_t end_idx = ((i + 1) < impl_->ngram_counts_.size()) ? impl_->ngram_counts_[i + 1] : total_items;
    ORT_ENFORCE(end_idx >= start_idx && end_idx <= total_items,
                "n-gram counts out of bounds for ", std::to_string(ngram_size), "-grams");
    auto items = end_idx - start_idx;
    if (items > 0) {
      ORT_ENFORCE((items % ngram_size == 0),
                  "Number of items must compose whole ", std::to_string(ngram_size), "-grams");
      auto ngrams = items / ngram_size;
      // Skip loading into hash_set ngrams that are not in the range of [min_gram_length-max_gram_length]
      if (ngram_size >= min_gram_length && ngram_size <= max_gram_length) {
        if (impl_->pool_strings_.empty()) {
          auto before_insert = impl_->int64_set_.size();
          Emplace(pool_int64s.begin() + start_idx, ngrams, ngram_size, ngram_id, impl_->int64_set_);
          ORT_ENFORCE((before_insert + ngrams) == impl_->int64_set_.size(), "pool_int64s duplicate ", std::to_string(ngram_size), "-grams detected");
        } else {
          auto before_insert = impl_->str_set_.size();
          Emplace(impl_->pool_strings_.begin() + start_idx, ngrams, ngram_size, ngram_id, impl_->str_set_);
          ORT_ENFORCE((before_insert + ngrams) == impl_->str_set_.size(), "poll_strings duplicate ", std::to_string(ngram_size), "-grams detected");
        }
      } else {
        ngram_id += ngrams;
      }
    }
    ++ngram_size;
  }
}

TfIdfVectorizer::~TfIdfVectorizer() = default;

void TfIdfVectorizer::OutputResult(OpKernelContext* ctx, size_t B, const std::vector<uint32_t>& frequences) const {
  const Impl& impl = *impl_;
  std::vector<int64_t> output_dims;
  if (B == 0) {
    output_dims.push_back(impl.output_size_);
  } else {
    output_dims.push_back(B);
    output_dims.push_back(impl.output_size_);
  }

  TensorShape output_shape(output_dims);
  assert(frequences.size() == static_cast<size_t>(output_shape.Size()));

  auto Y = ctx->Output(0, output_shape);
  auto output_data = Y->MutableData<float>();
  const auto& w = impl.weights_;
  switch (impl.weighting_criteria_) {
    case kTF: {
      for (auto f : frequences) {
        *output_data++ = static_cast<float>(f);
      }
    } break;
    case kIDF: {
      if (!w.empty()) {
        assert(frequences.size() == w.size());
        for (size_t i = 0; i < frequences.size(); ++i) {
          *output_data++ = (frequences[i] > 0) ? w[i] : 0;
        }
      } else {
        for (auto f : frequences) {
          *output_data++ = (f > 0) ? 1.0f : 0;
        }
      }
    } break;
    case kTFIDF: {
      if (!w.empty()) {
        assert(frequences.size() == w.size());
        for (size_t i = 0; i < frequences.size(); ++i) {
          *output_data++ = frequences[i] * w[i];
        }
      } else {
        for (auto f : frequences) {
          *output_data++ = static_cast<float>(f);
        }
      }
    } break;
    case kNone:  // fall-through
    default:
      assert(false);
  }
}

template <typename T>
Status TfIdfVectorizer::ComputeImpl(OpKernelContext* ctx) const {
  const auto& impl = *impl_;
  auto const set_end = impl.PoolEnd<T>();

  auto X = ctx->Input<Tensor>(0);
  auto& input_shape = X->Shape();
  const size_t total_items = input_shape.Size();

  size_t b_dim = 0;
  size_t B = 0;
  size_t C = 0;
  auto& input_dims = input_shape.GetDims();
  if (input_dims.empty()) {
    b_dim = 1;
    C = 1;
    assert(total_items == 1);
  } else if (input_dims.size() == 1) {
    b_dim = 1;
    C = input_dims[0];
  } else if (input_dims.size() == 2) {
    B = input_dims[0];
    C = input_dims[1];
    b_dim = B;
    if (B < 1) {
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                    "Input shape must have either [C] or [B,C] dimensions with B > 0.");
    }
  } else {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "Input shape must have either [C] or [B,C] dimensions with B > 0.");
  }

  // Frequency holder allocate [B..output_size_]
  // and init all to zero
  std::vector<uint32_t> frequencies;
  frequencies.resize(b_dim * impl.output_size_, 0);

  if (input_shape.Size() == 0) {
    // TfidfVectorizer may receive an empty input when it follows a Tokenizer
    // (for example for a string containing only stopwords).
    // TfidfVectorizer returns a zero tensor of shape
    // {b_dim, output_size} when b_dim is the number of received observations
    // and output_size the is the maximum value in ngram_indexes attribute plus 1.
    OutputResult(ctx, B, frequencies);
    return Status::OK();
  }

  assert((b_dim * C) == total_items);

  const auto max_gram_length = impl.max_gram_length_;
  const auto max_skip_distance = impl.max_skip_count_ + 1;  // Convert to distance
  auto start_ngram_size = impl.min_gram_length_;
  auto const input_data = X->template Data<T>();
  auto const end_data = input_data + total_items;
  NgramEntry<T> sample;

  // Treat 1-grams in a special way
  if (start_ngram_size == 1) {
    size_t row_num = 0;
    auto ngram_start = input_data;
    while (ngram_start < end_data) {
      auto const ngram_row_end = ngram_start + C;
      while (ngram_start < ngram_row_end) {
        sample.Clear();
        sample.AddItem(*ngram_start);
        auto hit = impl.PoolFind<T>(sample);
        if (hit != set_end) {
          // record frequency
          auto ngram_id = hit->Id();
          impl.IncrementCount(ngram_id, row_num, frequencies);
        }
        ++ngram_start;
      }
      ++row_num;
      ngram_start = ngram_row_end;
    }
    if (++start_ngram_size > max_gram_length) {
      OutputResult(ctx, B, frequencies);
      return Status::OK();
    }
  }

  for (auto skip_distance = 1; skip_distance <= max_skip_distance; ++skip_distance) {
    auto ngram_start = input_data;
    size_t row_num = 0;
    while (ngram_start < end_data) {
      assert((B == 0) || (row_num < B));
      auto const ngram_row_end = ngram_start + C;
      assert(ngram_row_end <= end_data);
      while (ngram_start < ngram_row_end) {
        // Check if any n-gram size in [start_ngram_size..max_gram_length] range
        // fit before the end of the row so we do not waste time adding [1..start_ngram_size)
        // At least items of start_ngram_size should fit
        // last row should match end_data
        auto at_least_this = ngram_start + skip_distance * (start_ngram_size - 1);
        if (at_least_this >= ngram_row_end) {
          break;
        }
        sample.Clear();
        auto ngram_item = ngram_start;
        for (auto ngram_size = 1;
             ngram_size <= max_gram_length &&
             ngram_item < ngram_row_end;
             ++ngram_size, ngram_item += skip_distance) {
          sample.AddItem(*ngram_item);

          // Do not test anything before start_ngram_size
          if (ngram_size >= start_ngram_size) {
            auto hit = impl.PoolFind<T>(sample);
            if (hit != set_end) {
              // record frequency
              auto ngram_id = hit->Id();
              impl.IncrementCount(ngram_id, row_num, frequencies);
            }
          }
        }
        // Sliding window shift
        ++ngram_start;
      }
      // Next row
      ngram_start = ngram_row_end;
      ++row_num;
    }
  }
  OutputResult(ctx, B, frequencies);
  return Status::OK();
}

Status TfIdfVectorizer::Compute(OpKernelContext* ctx) const {
  Status s;

  auto X = ctx->Input<Tensor>(0);

  if (X->IsDataType<int32_t>()) {
    s = ComputeImpl<int32_t>(ctx);
  } else if (X->IsDataType<int64_t>()) {
    s = ComputeImpl<int64_t>(ctx);
  } else if (X->IsDataTypeString()) {
    s = ComputeImpl<std::string>(ctx);
  } else {
    s = Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
               "Invalid type of the input argument");
  }

  return s;
}

}  // namespace onnxruntime

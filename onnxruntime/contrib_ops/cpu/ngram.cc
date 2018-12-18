// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ngram.h"
#include "onnx/defs/schema.h"
#include "core/common/common.h"
#include "core/framework/tensor.h"

#include <functional>
#include <unordered_set>
#include <ostream>
#include <iterator>

namespace onnxruntime {
namespace contrib {

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    Ngram,
    1,
    string,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<std::string>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>()),
    contrib::Ngram);

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    Ngram,
    1,
    int32_t,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>()),
    contrib::Ngram);

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    Ngram,
    1,
    int64_t,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>()),
    contrib::Ngram);

namespace ngram_details {

class NgramElementBase {
  size_t id_;  // id in the pool
 protected:
  NgramElementBase(size_t id) : id_(id) {}
  ~NgramElementBase() = default;

 public:
  size_t id() const { return id_; }
};

template <class T>
class NGramItem;

template <>
class NGramItem<int64_t> : public NgramElementBase {
  std::vector<int64_t> items_;

 public:
  template <typename ForwardIter>
  explicit NGramItem(size_t id, ForwardIter first, ForwardIter last) : NgramElementBase(id),
                                                                       items_(first, last) {
    assert(!items_.empty());
  }
  // For sampling
  explicit NGramItem() : NgramElementBase(0) {}
  void AddItem(int64_t t) { items_.push_back(t); }
  void DebugPrint() const {
    std::copy(items_.cbegin(), items_.cend(), std::ostream_iterator<int64_t>(std::cout, ","));
    std::cout << std::endl;
  }
  void Clear() { items_.clear(); }
  bool operator==(const NGramItem& o) const {
    return items_ == o.items_;
  }
  size_t hash() const {
    if (items_.empty()) return 0;
    auto first = items_.cbegin();
    auto const end = items_.cend();
    std::hash<int64_t> hf{};
    auto hash = hf(*first);
    while (++first != end) {
      hash ^= hf(*first) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    return hash;
  }
};

template <>
class NGramItem<int32_t> : public NGramItem<int64_t> {
 public:
  template <typename ForwardIter>
  explicit NGramItem(size_t id, ForwardIter first, ForwardIter last) : NGramItem<int64_t>(id, first, last) {}
  explicit NGramItem() = default;
};

template <>
class NGramItem<std::string> : public NgramElementBase {
 private:
  std::vector<std::reference_wrapper<const std::string>> items_;

 public:
  template <typename ForwardIter>
  explicit NGramItem(size_t id, ForwardIter first, ForwardIter last) : NgramElementBase(id) {
    std::transform(first, last, std::back_inserter(items_),
                   [](const std::string& s) { return std::cref(s); });
    assert(!items_.empty());
  }
  explicit NGramItem() : NgramElementBase(0) {}
  void AddItem(const std::string& s) { items_.push_back(std::cref(s)); }
  void DebugPrint() const {
    std::copy(items_.cbegin(), items_.cend(), std::ostream_iterator<std::string>(std::cout, ","));
    std::cout << std::endl;
  }
  void Clear() { items_.clear(); }

  bool operator==(const NGramItem& o) const {
    if (items_.size() == o.items_.size()) {
      return std::equal(items_.cbegin(), items_.cend(),
                        o.items_.cbegin(), o.items_.cend(),
                        std::equal_to<std::string>());
    }
    return false;
  }
  size_t hash() const {
    if (items_.empty()) return 0;
    auto first = items_.cbegin();
    auto const end = items_.cend();
    std::hash<std::string> hf{};
    auto hash = hf(*first);
    while (++first != end) {
      hash ^= hf(*first) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    return hash;
  }
};

using IntegerPoolSet = std::unordered_set<NGramItem<int64_t>>;
// Does not own strings, contains references to them. This helps
// to search by string references that point to the current input.
using StringPoolSet = std::unordered_set<NGramItem<std::string>>;

template <typename ForwardIter, typename Cont>
inline void Emplace(ForwardIter first, size_t ngrams, size_t ngram_size, size_t& ngram_id, Cont& c) {
  for (; ngrams > 0; --ngrams) {
    c.emplace(ngram_id, first, first + ngram_size);
    first += ngram_size;
    ++ngram_id;
  }
}

}  // namespace ngram_details
}  // namespace contrib
}  // namespace onnxruntime

using namespace onnxruntime::contrib::ngram_details;

namespace std {
template <typename T>
struct hash<NGramItem<T>> {
  typedef NGramItem<T> argument_type;
  typedef size_t result_type;
  result_type operator()(const argument_type& a) const {
    return a.hash();
  }
};
}  // namespace std

namespace onnxruntime {
namespace contrib {

enum Mode {
  kNone = 0,
  kTF = 1,
  kIDF = 2,
  kTFIDF = 3
};

struct Ngram::Impl {
  Mode mode_ = kNone;
  int64_t N_ = 0;
  int64_t M_ = 0;
  int64_t S_ = 0;
  bool all_ = false;
  std::vector<int64_t> ngram_counts_;
  std::vector<int64_t> ngram_indexes_;
  std::vector<float> weights_;

  std::vector<std::string> pool_strings_;
  StringPoolSet str_set_;
  IntegerPoolSet int_set_;
  size_t output_size_ = 0;

  MLDataType int32_dt_;
  MLDataType int64_dt_;
  MLDataType string_dt_;
  Impl() {
    int32_dt_ = DataTypeImpl::GetType<int32_t>();
    int64_dt_ = DataTypeImpl::GetType<int64_t>();
    string_dt_ = DataTypeImpl::GetType<std::string>();
  }

  template <typename T>
  auto PoolEnd() const;

  template <typename T>
  auto Find(const ngram_details::NGramItem<T>&) const;

  void IncrementCount(size_t ngram_id, std::vector<uint32_t>& frequencies) const {
    assert(ngram_id < ngram_indexes_.size());
    auto output_idx = ngram_indexes_[ngram_id];
    ORT_ENFORCE(output_idx >= 0, "ngram_indxes has a negative index");
    assert(static_cast<size_t>(output_idx) < frequencies.size());
    ++frequencies[output_idx];
  }
};

template <>
inline auto Ngram::Impl::PoolEnd<int64_t>() const {
  return int_set_.cend();
}

template <>
inline auto Ngram::Impl::PoolEnd<int32_t>() const {
  return PoolEnd<int64_t>();
}

template <>
inline auto Ngram::Impl::PoolEnd<std::string>() const {
  return str_set_.cend();
}

template <>
inline auto Ngram::Impl::Find<int64_t>(const NGramItem<int64_t>& i) const {
  return int_set_.find(i);
}

template <>
inline auto Ngram::Impl::Find<int32_t>(const NGramItem<int32_t>& i) const {
  return int_set_.find(i);
}

template <>
inline auto Ngram::Impl::Find<std::string>(const NGramItem<std::string>& i) const {
  return str_set_.find(i);
}

Ngram::Ngram(const OpKernelInfo& info) : OpKernel(info), impl_(new Impl) {
  std::string mode;
  Status status = info.GetAttr("mode", &mode);
  ORT_ENFORCE(status.IsOK(), "mode is required");
  if (mode == "TF") {
    impl_->mode_ = kTF;
  } else if (mode == "IDF") {
    impl_->mode_ = kIDF;
  } else if (mode == "TFIDF") {
    impl_->mode_ = kTFIDF;
  }
  ORT_ENFORCE(impl_->mode_ != kNone, "Unrecognized mode");

  status = info.GetAttr("M", &impl_->M_);
  ORT_ENFORCE(status.IsOK() && impl_->M_ > 0, "Positive Attr M is required");
  status = info.GetAttr("N", &impl_->N_);
  ORT_ENFORCE(status.IsOK() && impl_->N_ >= impl_->M_, "Positive M >= N is required");
  status = info.GetAttr("S", &impl_->S_);
  ORT_ENFORCE(status.IsOK() && impl_->N_ >= 0, "Non-negative number of skips S is required");

  int64_t all = 0;
  status = info.GetAttr("all", &all);
  ORT_ENFORCE(status.IsOK(), "Attribute all is required");
  impl_->all_ = (all != 0);

  status = info.GetAttrs(std::string("ngram_counts"), impl_->ngram_counts_);
  ORT_ENFORCE(status.IsOK() && !impl_->ngram_counts_.empty(), "Non-empty ngram_counts is required");
  ORT_ENFORCE(size_t(impl_->M_) <= impl_->ngram_counts_.size(), "M must be inbounds of ngram_counts");
  ORT_ENFORCE(size_t(impl_->N_) <= impl_->ngram_counts_.size(), "N must be inbounds of ngram_counts");

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
                "weights and indexes must have equal size");
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
  // Load into dictionary only required Ns
  const size_t M = impl_->M_;
  const size_t N = impl_->N_;
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
      // Skip loading into hash_set ngrams that are not N or not in the range of [M-N] for all=true;
      if ((impl_->all_ && (ngram_size >= M && ngram_size <= N)) ||
          ngram_size == N) {
        if (impl_->pool_strings_.empty()) {
          auto before_insert = impl_->int_set_.size();
          Emplace(pool_int64s.begin() + start_idx, ngrams, ngram_size, ngram_id, impl_->int_set_);
          ORT_ENFORCE((before_insert + ngrams) == impl_->int_set_.size(), "pool_int64s duplicate ", std::to_string(ngram_size), "-grams detected");
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

Ngram::~Ngram() {
}

void Ngram::OutputResult(OpKernelContext* ctx, const std::vector<uint32_t>& frequences) const {
  std::vector<int64_t> output_dims;
  output_dims.push_back(frequences.size());

  TensorShape output_shape(output_dims);
  auto Y = ctx->Output(0, output_shape);
  auto output_data = Y->MutableData<float>();
  const auto& w = impl_->weights_;
  switch (impl_->mode_) {
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
      break;
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
}

template <typename T>
void Ngram::ComputeImpl(OpKernelContext* ctx, size_t total_items) const {
  const auto& impl = *impl_;
  auto const set_end = impl.PoolEnd<T>();
  // Frequency holder, init all to zero
  std::vector<uint32_t> frequencies;
  frequencies.resize(impl.output_size_, 0);

  const auto N = impl.N_;
  const auto S = impl.S_ + 1;  // Convert to distance
  auto const start_ngram_size = (impl.all_) ? impl.M_ : N;

  auto X = ctx->Input<Tensor>(0);
  auto const input_data = X->template Data<T>();
  auto const end_data = input_data + total_items;
  NGramItem<T> sample;
  // Convert skip into distance between n-gram items
  // by adding 1
  for (auto si = 1; si <= S; ++si) {
    auto ngram_start = input_data;
    while (ngram_start < end_data) {
      // Check if any ni in [start_ngram_size..N]
      // fit before end_data so we do not waste time adding [1..start_ngram_size)
      // For that at least items with start_ngram_size should fit
      auto at_least_this = ngram_start + (si - 1) * (start_ngram_size - 1) + (start_ngram_size - 1);
      if (at_least_this >= end_data) {
        break;
      }
      sample.Clear();
      auto ngram_item = ngram_start;
      for (auto ni = 1;
           ni <= N &&
           ngram_item < end_data;
           ++ni, ngram_item += si) {
        sample.AddItem(*ngram_item);

        // Do not test anything before start_ngram_size
        if (ni >= start_ngram_size) {
          auto hit = impl.Find<T>(sample);
          if (hit != set_end) {
            // record frequency
            auto ngram_id = hit->id();
            impl.IncrementCount(ngram_id, frequencies);
          }
        }
      }
      ++ngram_start;
    }
  }
  OutputResult(ctx, frequencies);
}

Status Ngram::Compute(OpKernelContext* ctx) const {
  Status s;

  auto X = ctx->Input<Tensor>(0);
  auto& input_dims = X->Shape().GetDims();
  size_t total_items = 1;
  // Scalar
  if (input_dims.empty() || (input_dims.size() == 1 && input_dims[0] == 0)) {
    total_items = 1;
  } else {
    for (const auto& dim : input_dims) {
      total_items *= dim;
    }
  }

  if (X->DataType() == impl_->int32_dt_) {
    ComputeImpl<int32_t>(ctx, total_items);
  } else if (X->DataType() == impl_->int64_dt_) {
    ComputeImpl<int64_t>(ctx, total_items);
  } else if (X->DataType() == impl_->string_dt_) {
    ComputeImpl<std::string>(ctx, total_items);
  } else {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "Invalid type of the input argument");
  }

  return s;
}

}  // namespace contrib
}  // namespace onnxruntime

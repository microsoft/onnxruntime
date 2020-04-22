// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "tfidfvectorizer.h"
#include "onnx/defs/schema.h"
#include "core/common/common.h"
#include "core/framework/tensor.h"
#include "core/platform/threadpool.h"

#include <functional>
#include <unordered_map>

namespace onnxruntime {

ONNX_CPU_OPERATOR_KERNEL(
    TfIdfVectorizer,
    9,
    KernelDefBuilder()
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<std::string>(),
                              DataTypeImpl::GetTensorType<int32_t>(),
                              DataTypeImpl::GetTensorType<int64_t>()})
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>()),
    TfIdfVectorizer);

namespace ngram_details {

// NgrampPart implements a Trie like structure
// for a unigram (1) it would insert into a root map with a valid id.
// for (1,2,3) node 2 would be a child of 1 but have id == 0
// because (1,2) does not exists. Node 3 would have a valid id.
template <class T>
struct NgramPart;

template <>
struct NgramPart<int64_t>;

template <>
struct NgramPart<std::string>;

using NgramPartInt = NgramPart<int64_t>;
using NgramPartString = NgramPart<std::string>;

// Avoid recursive class definitions using unique_ptr + forward declaration
using IntMap = std::unordered_map<int64_t, std::unique_ptr<NgramPartInt>>;

using StrMap = std::unordered_map<std::reference_wrapper<const std::string>, std::unique_ptr<NgramPartString>,
                                  std::hash<std::string>, std::equal_to<std::string>>;

template <>
struct NgramPart<int64_t> {
  size_t id_;  // 0 - means no entry, search for a bigger N
  IntMap leafs_;
  explicit NgramPart(size_t id) : id_(id) {}
};

template <>
struct NgramPart<std::string> {
  size_t id_;  // 0 - means no entry, search for a bigger N
  StrMap leafs_;
  explicit NgramPart(size_t id) : id_(id) {}
};

// Returns next ngram_id
template <class K, class ForwardIter, class Map>
inline size_t PopulateGrams(ForwardIter first, size_t ngrams, size_t ngram_size, size_t ngram_id,
                            Map& c) {
  for (; ngrams > 0; --ngrams) {
    size_t n = 1;
    Map* m = &c;
    while (true) {
      auto p = m->emplace(*first, onnxruntime::make_unique<NgramPart<K>>(0));
      ++first;
      if (n == ngram_size) {
        ORT_ENFORCE(p.first->second->id_ == 0, "Duplicate ngram detected, size: ", ngram_size, " id: ", ngram_id);
        p.first->second->id_ = ngram_id;
        ++ngram_id;
        break;
      }
      ++n;
      m = &p.first->second->leafs_;
    }
  }
  return ngram_id;
}

}  // namespace ngram_details
}  // namespace onnxruntime

using namespace onnxruntime::ngram_details;

namespace onnxruntime {

inline const void* AdvanceElementPtr(const void* p, size_t elements, size_t element_size) {
  return reinterpret_cast<const uint8_t*>(p) + elements * element_size;
}

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
  // This map contains references to pool_string_ entries
  // of pool_strings attribute
  StrMap str_map_;
  // This map contains pool_int64s entries
  IntMap int64_map_;

  size_t output_size_ = 0;

  Impl() = default;
  ~Impl() = default;
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;

  void IncrementCount(size_t ngram_id, size_t row_num,
                      std::vector<uint32_t>& frequencies) const {
    assert(ngram_id != 0);
    --ngram_id;
    assert(ngram_id < ngram_indexes_.size());
    auto output_idx = row_num * output_size_ + ngram_indexes_[ngram_id];
    assert(static_cast<size_t>(output_idx) < frequencies.size());
    ++frequencies[output_idx];
  }
};

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
  size_t ngram_id = 1;  // start with 1, 0 - means no n-gram
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
          ngram_id = PopulateGrams<int64_t>(pool_int64s.begin() + start_idx, ngrams, ngram_size, ngram_id, impl_->int64_map_);
        } else {
          ngram_id = PopulateGrams<std::string>(impl_->pool_strings_.begin() + start_idx, ngrams, ngram_size, ngram_id, impl_->str_map_);
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
    B = 1;  // For use in the loops below
  } else {
    output_dims.push_back(B);
    output_dims.push_back(impl.output_size_);
  }

  const auto row_size = impl.output_size_;

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
        const auto* freqs = frequences.data();
        for (size_t batch = 0; batch < B; ++batch) {
          for (size_t i = 0; i < row_size; ++i) {
            *output_data++ = (*freqs++ > 0) ? w[i] : 0;
          }
        }
      } else {
        for (auto f : frequences) {
          *output_data++ = (f > 0) ? 1.0f : 0;
        }
      }
    } break;
    case kTFIDF: {
      if (!w.empty()) {
        const auto* freqs = frequences.data();
        for (size_t batch = 0; batch < B; ++batch) {
          for (size_t i = 0; i < row_size; ++i) {
            *output_data++ = *freqs++ * w[i];
          }
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

void TfIdfVectorizer::ComputeImpl(OpKernelContext* ctx, ptrdiff_t row_num, size_t row_size,
                                  std::vector<uint32_t>& frequencies) const {
  auto X = ctx->Input<Tensor>(0);
  const auto elem_size = X->DataType()->Size();

  const void* row_begin = AdvanceElementPtr(X->DataRaw(), row_num * row_size, elem_size);
  const void* const row_end = AdvanceElementPtr(row_begin, row_size, elem_size);

  const auto& impl = *impl_;
  const auto max_gram_length = impl.max_gram_length_;
  const auto max_skip_distance = impl.max_skip_count_ + 1;  // Convert to distance
  auto start_ngram_size = impl.min_gram_length_;

  for (auto skip_distance = 1; skip_distance <= max_skip_distance; ++skip_distance) {
    auto ngram_start = row_begin;
    auto const ngram_row_end = row_end;

    while (ngram_start < ngram_row_end) {
      // We went far enough so no n-grams of any size can be gathered
      auto at_least_this = AdvanceElementPtr(ngram_start, skip_distance * (start_ngram_size - 1), elem_size);
      if (at_least_this >= ngram_row_end) {
        break;
      }

      auto ngram_item = ngram_start;
      if (X->IsDataTypeString()) {
        const std::string* str_item = reinterpret_cast<const std::string*>(ngram_item);
        const StrMap* str_map = &impl.str_map_;
        for (auto ngram_size = 1;
             !str_map->empty() &&
             ngram_size <= max_gram_length &&
             str_item < ngram_row_end;
             ++ngram_size, str_item += skip_distance) {
          auto hit = str_map->find(*str_item);
          if (hit == str_map->end()) {
            break;
          }
          if (ngram_size >= start_ngram_size && hit->second->id_ != 0) {
            impl.IncrementCount(hit->second->id_, row_num, frequencies);
          }
          str_map = &hit->second->leafs_;
        }
      } else {
        const IntMap* int_map = &impl.int64_map_;
        for (auto ngram_size = 1;
             !int_map->empty() &&
             ngram_size <= max_gram_length &&
             ngram_item < ngram_row_end;
             ++ngram_size, ngram_item = AdvanceElementPtr(ngram_item, skip_distance, elem_size)) {
          int64_t val = (X->IsDataType<int32_t>()) ? int64_t{*reinterpret_cast<const int32_t*>(ngram_item)} : *reinterpret_cast<const int64_t*>(ngram_item);
          auto hit = int_map->find(val);
          if (hit == int_map->end()) {
            break;
          }
          if (ngram_size >= start_ngram_size && hit->second->id_ != 0) {
            impl.IncrementCount(hit->second->id_, row_num, frequencies);
          }
          int_map = &hit->second->leafs_;
        }
      }
      // Sliding window shift
      ngram_start = AdvanceElementPtr(ngram_start, 1, elem_size);
    }
    // We count UniGrams only once since they are not affected
    // by skip distance
    if (start_ngram_size == 1 && ++start_ngram_size > max_gram_length) {
      break;
    }
  }
}

Status TfIdfVectorizer::Compute(OpKernelContext* ctx) const {
  auto X = ctx->Input<Tensor>(0);
  auto& input_shape = X->Shape();
  const size_t total_items = input_shape.Size();

  int32_t num_rows = 0;
  size_t B = 0;
  size_t C = 0;
  auto& input_dims = input_shape.GetDims();
  if (input_dims.empty()) {
    num_rows = 1;
    C = 1;
    assert(total_items == 1);
  } else if (input_dims.size() == 1) {
    num_rows = 1;
    C = input_dims[0];
  } else if (input_dims.size() == 2) {
    B = input_dims[0];
    C = input_dims[1];
    num_rows = static_cast<int32_t>(B);
    if (B < 1) {
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                    "Input shape must have either [C] or [B,C] dimensions with B > 0.");
    }
  } else {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "Input shape must have either [C] or [B,C] dimensions with B > 0.");
  }

  assert((num_rows * C) == total_items);
  // Frequency holder allocate [B..output_size_]
  // and init all to zero
  std::vector<uint32_t> frequencies;
  frequencies.resize(num_rows * impl_->output_size_, 0);

  if (total_items == 0 ||
      (X->IsDataTypeString() && impl_->str_map_.empty()) ||
      ((X->IsDataType<int32_t>() || X->IsDataType<int64_t>()) && impl_->int64_map_.empty())) {
    // TfidfVectorizer may receive an empty input when it follows a Tokenizer
    // (for example for a string containing only stopwords).
    // TfidfVectorizer returns a zero tensor of shape
    // {b_dim, output_size} when b_dim is the number of received observations
    // and output_size the is the maximum value in ngram_indexes attribute plus 1.
    OutputResult(ctx, B, frequencies);
    return Status::OK();
  }

  std::function<void(ptrdiff_t)> fn = [this, ctx, C, &frequencies](ptrdiff_t row_num) {
    ComputeImpl(ctx, row_num, C, frequencies);
  };

  concurrency::ThreadPool::TryBatchParallelFor(ctx->GetOperatorThreadPool(), num_rows, std::move(fn), 0);

  OutputResult(ctx, B, frequencies);

  return Status::OK();
}

}  // namespace onnxruntime

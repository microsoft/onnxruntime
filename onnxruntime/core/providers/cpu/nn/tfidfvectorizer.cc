// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "tfidfvectorizer.h"
#include "core/common/common.h"
#include "core/framework/tensor.h"
#include "core/platform/threadpool.h"

#include <functional>
#include <unordered_map>
#include <core/common/safeint.h>

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
      auto p = m->emplace(*first, std::make_unique<NgramPart<K>>(0));
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
  gsl::span<const int64_t> ngram_counts_;
  // Contains output indexes
  // represents ngram_indexes output
  gsl::span<const int64_t> ngram_indexes_;
  gsl::span<const float> weights_;

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

  inline size_t OutputIdToIncrement(size_t ngram_id) const {
    assert(ngram_id != 0);
    --ngram_id;
    assert(ngram_id < ngram_indexes_.size());
    return SafeInt<size_t>(ngram_indexes_[ngram_id]);
  }
};

TfIdfVectorizer::TfIdfVectorizer(const OpKernelInfo& info) : OpKernel(info), impl_(std::make_unique<Impl>()) {
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

  status = info.GetAttrsAsSpan("ngram_counts", impl_->ngram_counts_);
  ORT_ENFORCE(status.IsOK() && !impl_->ngram_counts_.empty(), "Non-empty ngram_counts is required");
  ORT_ENFORCE(size_t(impl_->min_gram_length_) <= impl_->ngram_counts_.size(),
              "min_gram_length must be inbounds of ngram_counts: ",
              std::to_string(impl_->min_gram_length_), " <= ", std::to_string(impl_->ngram_counts_.size()));
  ORT_ENFORCE(size_t(impl_->max_gram_length_) <= impl_->ngram_counts_.size(),
              "max_gram_length must be inbounds of ngram_counts: ",
              std::to_string(impl_->max_gram_length_), " <= ", std::to_string(impl_->ngram_counts_.size()));

  status = info.GetAttrsAsSpan("ngram_indexes", impl_->ngram_indexes_);
  ORT_ENFORCE(status.IsOK() && !impl_->ngram_indexes_.empty(), "Non-empty ngram_indexes is required");
  {
    // Check that all are positive
    ORT_ENFORCE(std::all_of(impl_->ngram_indexes_.begin(), impl_->ngram_indexes_.end(),
                            [](int64_t i) { return i >= 0; }),
                "Negative ngram_indexes values are not allowed");
    // Set output size to max output index + 1;
    auto greatest_hit = std::max_element(impl_->ngram_indexes_.begin(), impl_->ngram_indexes_.end());
    impl_->output_size_ = SafeInt<size_t>(*greatest_hit) + 1;
  }

  status = info.GetAttrsAsSpan("weights", impl_->weights_);
  if (status.IsOK()) {
    ORT_ENFORCE(impl_->weights_.size() == impl_->ngram_indexes_.size(),
                "Got weights of size: ", std::to_string(impl_->weights_.size()),
                " but ngram_indexes size: ", std::to_string(impl_->ngram_indexes_.size()),
                " must be of equal size");
  }

  gsl::span<const int64_t> pool_int64s;
  std::vector<std::reference_wrapper<const std::string>> pool_strings;
  status = info.GetAttrsStringRefs("pool_strings", pool_strings);
  if (status.IsOK()) {
    ORT_ENFORCE(!pool_strings.empty(), "pool_strings must not be empty if specified");
  } else {
    status = info.GetAttrsAsSpan("pool_int64s", pool_int64s);
    ORT_ENFORCE(status.IsOK() && !pool_int64s.empty(), "non-empty pool_int64s is required if pool_strings not provided");
  }

  // Iterator via the pool. Insert 1 item for 1-grams, 2 items for 2-grams, etc.
  const auto total_items = (pool_strings.empty()) ? pool_int64s.size() : pool_strings.size();
  size_t ngram_id = 1;  // start with 1, 0 - means no n-gram
  // Load into dictionary only required gram sizes
  const size_t min_gram_length = onnxruntime::narrow<size_t>(impl_->min_gram_length_);
  const size_t max_gram_length = onnxruntime::narrow<size_t>(impl_->max_gram_length_);
  size_t ngram_size = 1;
  for (size_t i = 0; i < impl_->ngram_counts_.size(); ++i) {
    size_t start_idx = onnxruntime::narrow<size_t>(impl_->ngram_counts_[i]);
    size_t end_idx = onnxruntime::narrow<size_t>((i + 1) < impl_->ngram_counts_.size() ? impl_->ngram_counts_[i + 1] : total_items);
    ORT_ENFORCE(end_idx >= start_idx && end_idx <= total_items,
                "n-gram counts out of bounds for ", std::to_string(ngram_size), "-grams");
    auto items = end_idx - start_idx;
    if (items > 0) {
      ORT_ENFORCE((items % ngram_size == 0),
                  "Number of items must compose whole ", std::to_string(ngram_size), "-grams");
      auto ngrams = items / ngram_size;
      // Skip loading into hash_set ngrams that are not in the range of [min_gram_length-max_gram_length]
      if (ngram_size >= min_gram_length && ngram_size <= max_gram_length) {
        if (pool_strings.empty()) {
          ngram_id = PopulateGrams<int64_t>(pool_int64s.begin() + start_idx, ngrams, ngram_size, ngram_id, impl_->int64_map_);
        } else {
          ngram_id = PopulateGrams<std::string>(pool_strings.begin() + start_idx, ngrams, ngram_size, ngram_id, impl_->str_map_);
        }
      } else {
        ngram_id += ngrams;
      }
    }
    ++ngram_size;
  }
}

TfIdfVectorizer::~TfIdfVectorizer() = default;

void TfIdfVectorizer::ComputeImpl(const void* x_data_raw, size_t elem_size, ptrdiff_t row_num, size_t row_size,
                                  bool is_input_string, gsl::span<float> output_data,
                                  std::function<void(size_t, gsl::span<float>&)>& fn_weight) const {
  const void* const row_begin = AdvanceElementPtr(x_data_raw, row_num * row_size, elem_size);
  const void* const row_end = AdvanceElementPtr(row_begin, row_size, elem_size);

  const auto& impl = *impl_;
  const auto max_gram_length = impl.max_gram_length_;
  const auto max_skip_distance = impl.max_skip_count_ + 1;  // Convert to distance
  auto start_ngram_size = impl.min_gram_length_;
  size_t output_idx;

  for (auto skip_distance = 1; skip_distance <= max_skip_distance; ++skip_distance) {
    auto ngram_start = row_begin;
    auto const ngram_row_end = row_end;

    while (ngram_start < ngram_row_end) {
      // We went far enough so no n-grams of any size can be gathered
      auto at_least_this = AdvanceElementPtr(ngram_start, SafeInt<size_t>(skip_distance) * (start_ngram_size - 1), elem_size);
      if (at_least_this >= ngram_row_end) {
        break;
      }

      auto ngram_item = ngram_start;
      if (is_input_string) {
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
            output_idx = impl.OutputIdToIncrement(hit->second->id_);
            fn_weight(output_idx, output_data);
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
          int64_t val = (elem_size == 4) ? int64_t{*reinterpret_cast<const int32_t*>(ngram_item)} : *reinterpret_cast<const int64_t*>(ngram_item);
          auto hit = int_map->find(val);
          if (hit == int_map->end()) {
            break;
          }
          if (ngram_size >= start_ngram_size && hit->second->id_ != 0) {
            output_idx = impl.OutputIdToIncrement(hit->second->id_);
            fn_weight(output_idx, output_data);
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
  const size_t total_items = onnxruntime::narrow<size_t>(input_shape.Size());

  int32_t num_rows = 0;
  size_t B = 0;
  size_t C = 0;
  auto input_dims = input_shape.GetDims();
  if (input_dims.empty()) {
    num_rows = 1;
    C = 1;
    assert(total_items == 1);
  } else if (input_dims.size() == 1) {
    num_rows = 1;
    C = onnxruntime::narrow<size_t>(input_dims[0]);
  } else if (input_dims.size() == 2) {
    B = onnxruntime::narrow<size_t>(input_dims[0]);
    C = onnxruntime::narrow<size_t>(input_dims[1]);
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
  const Impl& impl = *impl_;
  TensorShapeVector output_dims;
  if (B == 0) {
    output_dims.push_back(impl.output_size_);
    B = 1;  // For use in the loops below
  } else {
    output_dims.push_back(B);
    output_dims.push_back(impl.output_size_);
  }
  TensorShape output_shape(output_dims);

  auto Y = ctx->Output(0, output_shape);
  auto output_data = Y->MutableData<float>();
  const bool is_input_string = X->IsDataTypeString();

  if (total_items == 0 ||
      (is_input_string && impl_->str_map_.empty()) ||
      ((X->IsDataType<int32_t>() || X->IsDataType<int64_t>()) && impl_->int64_map_.empty())) {
    // TfidfVectorizer may receive an empty input when it follows a Tokenizer
    // (for example for a string containing only stopwords).
    // TfidfVectorizer returns a zero tensor of shape
    // {b_dim, output_size} when b_dim is the number of received observations
    // and output_size the is the maximum value in ngram_indexes attribute plus 1.
    memset(output_data, 0, static_cast<size_t>(output_shape.Size() * sizeof(float)));
    return Status::OK();
  }

  auto x_data_raw = ctx->Input<Tensor>(0)->DataRaw();
  const auto elem_size = X->DataType()->Size();
  int32_t num_batches = std::min<int32_t>(concurrency::ThreadPool::DegreeOfParallelism(ctx->GetOperatorThreadPool()) * 2, num_rows);

  const auto& w = impl.weights_;
  std::function<void(size_t, gsl::span<float>&)> fn_weight;

  switch (impl.weighting_criteria_) {
    case kTF:
      fn_weight = [&w](size_t i, gsl::span<float>& out) { out[i] += 1.0f; };
      break;
    case kIDF:
      if (!w.empty()) {
        fn_weight = [&w](size_t i, gsl::span<float>& out) { out[i] = w[i]; };
      } else {
        fn_weight = [&w](size_t i, gsl::span<float>& out) { out[i] = 1.0f; };
      }
      break;
    case kTFIDF:
      if (!w.empty()) {
        fn_weight = [&w](size_t i, gsl::span<float>& out) { out[i] += w[i]; };
      } else {
        fn_weight = [&w](size_t i, gsl::span<float>& out) { out[i] += 1.0f; };
      }
      break;
    case kNone:  // fall-through
    default:
      assert(false);
  }

  std::function<void(ptrdiff_t)> fn = [this, C, output_data, x_data_raw, elem_size,
                                       is_input_string, num_batches, num_rows, &fn_weight](ptrdiff_t batch_num) {
    // Frequency holder allocate [B..output_size_] and init all to zero.
    auto work = concurrency::ThreadPool::PartitionWork(batch_num, num_batches, static_cast<size_t>(num_rows));
    std::vector<uint32_t> frequencies(this->impl_->output_size_);
    for (auto row_num = work.start; row_num < work.end; ++row_num) {
      auto out = gsl::span<float>(output_data + row_num * this->impl_->output_size_, this->impl_->output_size_);
      std::fill(out.begin(), out.end(), 0.0f);
      ComputeImpl(x_data_raw, elem_size, row_num, C, is_input_string, out, fn_weight);
    }
  };

  concurrency::ThreadPool::TrySimpleParallelFor(ctx->GetOperatorThreadPool(), num_batches, std::move(fn));
  return Status::OK();
}

}  // namespace onnxruntime

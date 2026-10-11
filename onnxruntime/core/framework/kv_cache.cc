// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/kv_cache.h"
#include "core/framework/ort_value.h"
#include "core/common/safeint.h"

#include <algorithm>
#include <limits>

namespace onnxruntime {
namespace {

size_t PayloadSize(const KVCacheOptions& o, int64_t tokens, int64_t width) {
  ORT_ENFORCE(tokens >= 0, "Negative KV token count");
  const size_t elements = SafeInt<size_t>(o.batch_size) * o.num_kv_heads * tokens * width;
  ORT_ENFORCE(elements <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
              "KV payload exceeds signed indexing range");
  [[maybe_unused]] const size_t bytes = SafeInt<size_t>(elements) * sizeof(float);
  return elements;
}

void Validate(const KVCacheOptions& o) {
  ORT_ENFORCE(o.batch_size > 0 && o.num_kv_heads > 0 &&
                  o.key_head_size > 0 && o.value_head_size > 0,
              "KV cache requires positive B, N, Dk, Dv");
  ORT_ENFORCE(o.page_size > 0 && o.sliding_window >= 0, "Invalid page size or sliding window");
  ORT_ENFORCE(o.storage == KVCacheStorage::Contiguous || o.storage == KVCacheStorage::Paged ||
                  o.storage == KVCacheStorage::Ring,
              "Unsupported KV cache storage kind");
  ORT_ENFORCE(o.storage != KVCacheStorage::Ring || o.sliding_window > 0,
              "Ring storage requires a declared sliding window");
  PayloadSize(o, o.page_size, o.key_head_size);
  PayloadSize(o, o.page_size, o.value_head_size);
  if (o.sliding_window) {
    PayloadSize(o, o.sliding_window, o.key_head_size);
    PayloadSize(o, o.sliding_window, o.value_head_size);
  }
  ORT_ENFORCE(o.batch_size <= std::numeric_limits<int64_t>::max() / o.num_kv_heads,
              "KV head count overflow");
}

struct Page {
  int64_t position;
  int64_t count;
  InlinedVector<float> keys;
  InlinedVector<float> values;
};

std::shared_ptr<const Page> MakePage(const KVCacheOptions& o, int64_t position, int64_t count,
                                     gsl::span<const float> keys, gsl::span<const float> values,
                                     int64_t source_tokens, int64_t source_offset) {
  auto page = std::make_shared<Page>();
  page->position = position;
  page->count = count;
  page->keys.reserve(PayloadSize(o, count, o.key_head_size));
  page->values.reserve(PayloadSize(o, count, o.value_head_size));
  for (int64_t h = 0; h < o.batch_size * o.num_kv_heads; ++h) {
    auto k = keys.subspan(static_cast<size_t>((h * source_tokens + source_offset) * o.key_head_size),
                          static_cast<size_t>(count * o.key_head_size));
    auto v = values.subspan(static_cast<size_t>((h * source_tokens + source_offset) * o.value_head_size),
                            static_cast<size_t>(count * o.value_head_size));
    page->keys.insert(page->keys.end(), k.begin(), k.end());
    page->values.insert(page->values.end(), v.begin(), v.end());
  }
  return page;
}

class BlockCache;
class RingCache;
std::shared_ptr<const KVCache> SealCache(const KVCache& cache);

class BlockCache final : public KVCache {
 public:
  BlockCache(KVCacheOptions options, int64_t end, int64_t begin,
             InlinedVector<std::shared_ptr<const Page>> pages)
      : options_(options), end_(end), begin_(begin), pages_(std::move(pages)) {}

  const KVCacheOptions& Options() const noexcept override { return options_; }
  int64_t End() const noexcept override { return end_; }
  int64_t Begin() const noexcept override { return begin_; }
  void Visit(const std::function<void(const KVCacheBlock&)>& visitor) const override {
    for (const auto& p : pages_) {
      visitor({p->position, p->count, p->count, p->keys.data(), p->values.data()});
    }
  }

  std::shared_ptr<const KVCache> Append(gsl::span<const float> keys,
                                        gsl::span<const float> values, int64_t tokens) const override {
    ORT_ENFORCE(tokens > 0 && tokens <= std::numeric_limits<int64_t>::max() - end_,
                "KV append must be nonempty and must not overflow computed length");
    ORT_ENFORCE(keys.size() == PayloadSize(options_, tokens, options_.key_head_size) &&
                    values.size() == PayloadSize(options_, tokens, options_.value_head_size),
                "KV append payload does not match logical geometry");
    InlinedVector<std::shared_ptr<const Page>> pages;
    if (options_.storage == KVCacheStorage::Contiguous) {
      const int64_t length = end_ - begin_ + tokens;
      InlinedVector<float> k(PayloadSize(options_, length, options_.key_head_size));
      InlinedVector<float> v(PayloadSize(options_, length, options_.value_head_size));
      Visit([&](const KVCacheBlock& block) {
        for (int64_t h = 0; h < options_.batch_size * options_.num_kv_heads; ++h) {
          std::copy_n(block.keys + h * block.head_stride * options_.key_head_size,
                      block.count * options_.key_head_size,
                      k.data() + (h * length + block.position - begin_) * options_.key_head_size);
          std::copy_n(block.values + h * block.head_stride * options_.value_head_size,
                      block.count * options_.value_head_size,
                      v.data() + (h * length + block.position - begin_) * options_.value_head_size);
        }
      });
      for (int64_t h = 0; h < options_.batch_size * options_.num_kv_heads; ++h) {
        std::copy_n(keys.data() + h * tokens * options_.key_head_size, tokens * options_.key_head_size,
                    k.data() + (h * length + end_ - begin_) * options_.key_head_size);
        std::copy_n(values.data() + h * tokens * options_.value_head_size, tokens * options_.value_head_size,
                    v.data() + (h * length + end_ - begin_) * options_.value_head_size);
      }
      if (length) pages.push_back(MakePage(options_, begin_, length, k, v, length, 0));
    } else {
      pages = pages_;  // Committed pages are immutable and shareable.
      for (int64_t t = 0; t < tokens;) {
        const int64_t n = std::min(options_.page_size, tokens - t);
        pages.push_back(MakePage(options_, end_ + t, n, keys, values, tokens, t));
        t += n;
      }
    }
    return std::make_shared<BlockCache>(options_, end_ + tokens, begin_, std::move(pages));
  }

  std::shared_ptr<const KVCache> Seal() const override {
    if (options_.storage == KVCacheStorage::Ring) return SealCache(*this);
    const int64_t begin = options_.sliding_window
                              ? std::max(begin_, end_ - std::min(end_, options_.sliding_window))
                              : begin_;
    InlinedVector<std::shared_ptr<const Page>> pages;
    for (const auto& p : pages_) {
      if (p->position + p->count <= begin) continue;
      if (p->position >= begin) {
        pages.push_back(p);
      } else {
        pages.push_back(MakePage(options_, begin, p->position + p->count - begin,
                                 p->keys, p->values, p->count, begin - p->position));
      }
    }
    return std::make_shared<BlockCache>(options_, end_, begin, std::move(pages));
  }

 private:
  const KVCacheOptions options_;
  const int64_t end_;
  const int64_t begin_;
  const InlinedVector<std::shared_ptr<const Page>> pages_;
};

// Committed rings own exactly W slots per head, addressed by absolute p % W.
// Appending stages the previous ring and the entire new chunk outside this bank.
class RingCache final : public KVCache {
 public:
  explicit RingCache(const KVCache& source)
      : options_(source.Options()), end_(source.End()), begin_(std::max(source.Begin(), end_ - std::min(end_, options_.sliding_window))), keys_(PayloadSize(options_, options_.sliding_window, options_.key_head_size)), values_(PayloadSize(options_, options_.sliding_window, options_.value_head_size)) {
    source.Visit([&](const KVCacheBlock& block) {
      for (int64_t p = std::max(begin_, block.position); p < block.position + block.count; ++p) {
        for (int64_t h = 0; h < options_.batch_size * options_.num_kv_heads; ++h) {
          std::copy_n(block.keys + (h * block.head_stride + p - block.position) * options_.key_head_size,
                      options_.key_head_size,
                      keys_.data() + (h * options_.sliding_window + p % options_.sliding_window) *
                                         options_.key_head_size);
          std::copy_n(block.values + (h * block.head_stride + p - block.position) * options_.value_head_size,
                      options_.value_head_size,
                      values_.data() + (h * options_.sliding_window + p % options_.sliding_window) *
                                           options_.value_head_size);
        }
      }
    });
  }
  const KVCacheOptions& Options() const noexcept override { return options_; }
  int64_t End() const noexcept override { return end_; }
  int64_t Begin() const noexcept override { return begin_; }
  void Visit(const std::function<void(const KVCacheBlock&)>& visitor) const override {
    for (int64_t p = begin_; p < end_;) {
      const int64_t slot = p % options_.sliding_window;
      const int64_t n = std::min(end_ - p, options_.sliding_window - slot);
      visitor({p, n, options_.sliding_window, keys_.data() + slot * options_.key_head_size,
               values_.data() + slot * options_.value_head_size});
      p += n;
    }
  }
  std::shared_ptr<const KVCache> Append(gsl::span<const float> keys, gsl::span<const float> values,
                                        int64_t tokens) const override {
    InlinedVector<std::shared_ptr<const Page>> pages;
    Visit([&](const KVCacheBlock& b) {
      // MakePage reads each head with the physical ring stride, not retained length.
      const size_t k_size = PayloadSize(options_, b.head_stride, options_.key_head_size) -
                            static_cast<size_t>((b.position % options_.sliding_window) * options_.key_head_size);
      const size_t v_size = PayloadSize(options_, b.head_stride, options_.value_head_size) -
                            static_cast<size_t>((b.position % options_.sliding_window) * options_.value_head_size);
      pages.push_back(MakePage(options_, b.position, b.count, {b.keys, k_size}, {b.values, v_size},
                               b.head_stride, 0));
    });
    BlockCache staging(options_, end_, begin_, std::move(pages));
    return staging.Append(keys, values, tokens);
  }
  std::shared_ptr<const KVCache> Seal() const override { return std::make_shared<RingCache>(*this); }

 private:
  const KVCacheOptions options_;
  const int64_t end_;
  const int64_t begin_;
  InlinedVector<float> keys_;
  InlinedVector<float> values_;
};

std::shared_ptr<const KVCache> SealCache(const KVCache& cache) {
  ORT_ENFORCE(cache.Options().storage == KVCacheStorage::Ring, "Expected ring retention");
  return std::make_shared<RingCache>(cache);
}

}  // namespace

KVCacheStorage ParseKVCacheStorage(std::string_view storage) {
  if (storage == "contiguous") return KVCacheStorage::Contiguous;
  if (storage == "paged") return KVCacheStorage::Paged;
  if (storage == "ring") return KVCacheStorage::Ring;
  ORT_THROW("Unsupported KV cache backend: ", storage);
}

const char* KVCacheStorageName(KVCacheStorage storage) {
  switch (storage) {
    case KVCacheStorage::Contiguous:
      return "contiguous";
    case KVCacheStorage::Paged:
      return "paged";
    case KVCacheStorage::Ring:
      return "ring";
  }
  ORT_THROW("Invalid KV cache storage kind");
}

std::shared_ptr<const KVCache> CreateKVCache(const KVCacheOptions& options) {
  Validate(options);
  auto empty = std::make_shared<BlockCache>(options, 0, 0, InlinedVector<std::shared_ptr<const Page>>{});
  return options.storage == KVCacheStorage::Ring ? empty->Seal() : empty;
}

void CreateKVCacheOrtValue(const KVCacheOptions& options, OrtValue& value) {
  auto payload = std::make_unique<KVCacheValue>();
  payload->cache = CreateKVCache(options);
  value.Init(payload.release(), DataTypeImpl::GetType<KVCacheValue>(),
             DataTypeImpl::GetType<KVCacheValue>()->GetDeleteFunc());
}

Status ValidateKVCacheType(const KVCache& cache, const ONNX_NAMESPACE::TypeProto& type) {
  ORT_RETURN_IF_NOT(type.has_kv_cache_type(), "Expected a dedicated kv_cache TypeProto");
  const auto& proto = type.kv_cache_type();
  const auto& o = cache.Options();
  ORT_RETURN_IF_NOT(proto.elem_type() == ONNX_NAMESPACE::TensorProto::FLOAT, "Only float32 cache supported");
  const ONNX_NAMESPACE::TensorShapeProto_Dimension* dims[]{
      &proto.batch_size(), &proto.num_kv_heads(), &proto.key_head_size(), &proto.value_head_size()};
  const int64_t sizes[]{o.batch_size, o.num_kv_heads, o.key_head_size, o.value_head_size};
  for (size_t i = 0; i < 4; ++i) {
    ORT_RETURN_IF(dims[i]->has_dim_value() && dims[i]->dim_value() != sizes[i],
                  "KV cache logical geometry differs from graph type at dimension ", i);
  }
  ORT_RETURN_IF_NOT(proto.sliding_window() == o.sliding_window, "KV cache retention differs from graph type");
  return Status::OK();
}

}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <memory>
#include <string_view>

#include "core/common/inlined_containers.h"
#include "core/framework/data_types.h"

namespace onnxruntime {

enum class KVCacheStorage { Contiguous,
                            Paged,
                            Ring };

struct KVCacheOptions {
  int64_t batch_size;
  int64_t num_kv_heads;
  int64_t key_head_size;
  int64_t value_head_size;
  KVCacheStorage storage{KVCacheStorage::Contiguous};
  int64_t page_size{16};
  int64_t sliding_window{0};
};

// A borrowed BNSD block. Position is absolute, and each head has its own stride.
struct KVCacheBlock {
  int64_t position;
  int64_t count;
  int64_t head_stride;
  const float* keys;
  const float* values;
};

class KVCache {
 public:
  virtual ~KVCache() = default;
  virtual const KVCacheOptions& Options() const noexcept = 0;
  virtual int64_t End() const noexcept = 0;
  virtual int64_t Begin() const noexcept = 0;
  virtual void Visit(const std::function<void(const KVCacheBlock&)>& visitor) const = 0;

  // Append returns a private, untrimmed version: all queries in a chunk can still
  // see their required history. Seal performs retention only after attention.
  virtual std::shared_ptr<const KVCache> Append(gsl::span<const float> keys,
                                                gsl::span<const float> values,
                                                int64_t tokens) const = 0;
  virtual std::shared_ptr<const KVCache> Seal() const = 0;
};

// OrtValues share immutable backing, never mutable per-layer published state.
struct KVCacheValue {
  std::shared_ptr<const KVCache> cache;
};

template <>
MLDataType DataTypeImpl::GetType<KVCacheValue>();

KVCacheStorage ParseKVCacheStorage(std::string_view storage);
const char* KVCacheStorageName(KVCacheStorage storage);
std::shared_ptr<const KVCache> CreateKVCache(const KVCacheOptions& options);
void CreateKVCacheOrtValue(const KVCacheOptions& options, OrtValue& value);
Status ValidateKVCacheType(const KVCache& cache, const ONNX_NAMESPACE::TypeProto& type);

}  // namespace onnxruntime

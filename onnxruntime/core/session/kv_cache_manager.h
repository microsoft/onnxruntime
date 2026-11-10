// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/inlined_containers.h"
#include "core/framework/ort_value.h"
#include "core/framework/stateful_execution_provider.h"

namespace onnxruntime {

class NodeArg;

struct KVCacheSpec {
  std::string name;
  int32_t element_type;
  // Non-sequence dimensions are positive; the sequence dimension is initially zero.
  TensorShapeVector shape;
  int64_t sequence_axis;
  int64_t initial_capacity = 0;
};

enum class KVCacheUpdateKind { Share,
                               Copy,
                               Append };

struct KVCacheView {
  OrtValue value;
  int64_t length = 0;
  int64_t capacity = 0;
};

struct KVCacheStats {
  size_t allocations = 0;
  size_t bytes_copied = 0;
  size_t shared_updates = 0;
};

// Sequential tensor-backed storage. Returned views retain their allocation across updates/reset/release.
class KVCacheManager {
 public:
  static Status ValidateSpecs(gsl::span<const KVCacheSpec> specs);
  static Status CheckBinding(const KVCacheSpec& spec, const NodeArg& argument, bool is_input);
  static Status Create(InlinedVector<KVCacheSpec> specs, AllocatorPtr allocator,
                       std::unique_ptr<KVCacheManager>& out, StatefulProviderBinding provider = {});
  Status Get(std::string_view name, KVCacheView& out) const;
  Status Update(std::string_view name, const OrtValue& value, KVCacheUpdateKind kind);
  Status Reserve(std::string_view name, int64_t capacity);
  Status Reset();
  bool RequiresReset() const { return requires_reset_; }
  const StatefulDispatch& Dispatch() const { return dispatch_; }
  const KVCacheStats& Stats() const { return stats_; }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(KVCacheManager);

 private:
  struct Entry {
    KVCacheSpec spec;
    OrtValue storage;
    KVCacheView view;
  };
  struct CpuWork;
  KVCacheManager(InlinedVector<KVCacheSpec> specs, AllocatorPtr allocator, StatefulProviderBinding provider);
  Status Allocate(const Entry& entry, int64_t length, int64_t capacity, Entry& out, KVCacheStats& stats);
  static OrtValue MakeView(Entry& entry, int64_t length);
  Status ValidateValue(const Entry& entry, const OrtValue& value) const;
  Status UpdateCpu(Entry& entry, const OrtValue& value, KVCacheUpdateKind kind, KVCacheStats& stats);
  Status ReserveCpu(Entry& entry, int64_t capacity, KVCacheStats& stats);
  Status Apply(OrtStatefulProviderOperation operation, Entry& entry, const OrtValue* value, int64_t capacity);
  static OrtStatus* ORT_API_CALL CpuReference(void* context, OrtStatefulCacheOperationResult* out) noexcept;

  InlinedHashMap<std::string, Entry> entries_;
  const AllocatorPtr allocator_;
  KVCacheStats stats_;
  StatefulDispatch dispatch_;
  bool requires_reset_ = false;
};

struct LoopKVCacheBinding {
  KVCacheSpec cache;
  std::string input_name;
  std::string output_name;
  KVCacheUpdateKind update_kind = KVCacheUpdateKind::Share;
};

}  // namespace onnxruntime

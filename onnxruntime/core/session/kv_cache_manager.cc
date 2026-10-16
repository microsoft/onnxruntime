// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/kv_cache_manager.h"

#include <algorithm>
#include <cstring>

#include "core/common/safeint.h"
#include "core/framework/data_transfer.h"
#include "core/graph/graph.h"

namespace onnxruntime {
namespace {

MLDataType ElementType(const KVCacheSpec& spec) {
  return DataTypeImpl::TensorTypeFromONNXEnum(spec.element_type)->GetElementType();
}

size_t StorageBytes(const KVCacheSpec& spec, int64_t length) {
  SafeInt<int64_t> elements = length;
  for (size_t i = 0; i < spec.shape.size(); ++i) {
    if (i != static_cast<size_t>(spec.sequence_axis)) {
      elements *= spec.shape[i];
    }
  }
  return SafeInt<size_t>(static_cast<int64_t>(elements)) * ElementType(spec)->Size();
}

}  // namespace

Status KVCacheManager::ValidateSpecs(gsl::span<const KVCacheSpec> specs) {
  InlinedHashSet<std::string> names;
  for (const auto& spec : specs) {
    ORT_RETURN_IF(spec.name.empty() || !names.insert(spec.name).second, "Empty or duplicate KV cache name");
    ORT_RETURN_IF(spec.element_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
                      spec.element_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 &&
                      spec.element_type != ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16,
                  "CPU KV cache supports float32, float16 and bfloat16");
    ORT_RETURN_IF(spec.sequence_axis < 0 || static_cast<uint64_t>(spec.sequence_axis) >= spec.shape.size() ||
                      spec.initial_capacity < 0,
                  "Invalid KV cache sequence axis or capacity: ", spec.name);
    for (size_t i = 0; i < spec.shape.size(); ++i) {
      ORT_RETURN_IF(i == static_cast<size_t>(spec.sequence_axis) ? spec.shape[i] != 0 : spec.shape[i] <= 0,
                    "KV cache requires positive fixed dimensions and an initially empty sequence: ", spec.name);
    }
    ORT_UNUSED_PARAMETER(StorageBytes(spec, spec.initial_capacity));
  }
  return Status::OK();
}

Status KVCacheManager::CheckBinding(const KVCacheSpec& spec, const NodeArg& argument, bool is_input) {
  const auto* type = argument.TypeAsProto();
  ORT_RETURN_IF_NOT(type && type->has_tensor_type() && type->tensor_type().elem_type() == spec.element_type,
                    "KV cache binding dtype mismatch: ", argument.Name());
  const auto* shape = argument.Shape();
  if (shape) {
    ORT_RETURN_IF(shape->dim_size() != static_cast<int>(spec.shape.size()),
                  "KV cache binding rank mismatch: ", argument.Name());
    for (int i = 0; i < shape->dim_size(); ++i) {
      const auto& dimension = shape->dim(i);
      ORT_RETURN_IF(dimension.has_dim_value() &&
                        (i == spec.sequence_axis ? (is_input && dimension.dim_value() != 0)
                                                 : dimension.dim_value() != spec.shape[i]),
                    "KV cache binding needs a dynamic sequence and compatible fixed dimensions: ", argument.Name());
    }
  }
  return Status::OK();
}

KVCacheManager::KVCacheManager(InlinedVector<KVCacheSpec> specs, AllocatorPtr allocator)
    : allocator_(std::move(allocator)) {
  for (auto& spec : specs) {
    const auto name = spec.name;
    entries_.emplace(name, Entry{std::move(spec), {}, {}});
  }
}

Status KVCacheManager::Create(InlinedVector<KVCacheSpec> specs, AllocatorPtr allocator,
                              std::unique_ptr<KVCacheManager>& out) {
  ORT_RETURN_IF_ERROR(ValidateSpecs(specs));
  ORT_RETURN_IF(specs.empty(), "KVCacheManager requires at least one cache");
  ORT_RETURN_IF_NOT(allocator && allocator->Info().device.Type() == OrtDevice::CPU,
                    "KVCacheManager requires a CPU allocator; native CUDA cache support is unavailable");
  auto manager = std::unique_ptr<KVCacheManager>(new KVCacheManager(std::move(specs), std::move(allocator)));
  for (auto& [name, entry] : manager->entries_) {
    Entry initialized;
    ORT_RETURN_IF_ERROR(manager->Allocate(entry, 0, entry.spec.initial_capacity, initialized));
    entry = std::move(initialized);
  }
  out = std::move(manager);
  return Status::OK();
}

OrtValue KVCacheManager::MakeView(Entry& entry, int64_t length) {
  auto shape = entry.spec.shape;
  shape[entry.spec.sequence_axis] = length;
  auto& storage = *entry.storage.GetMutable<Tensor>();
  auto tensor = std::make_unique<Tensor>(storage.DataType(), TensorShape(shape),
                                         storage.MutableDataRaw(), storage.Location());
  OrtValue value;
  // The compact logical tensor may be smaller than the allocation it retains.
  std::function<void(void*)> deleter = [owner = entry.storage](void* data) mutable {
    delete static_cast<Tensor*>(data);
    owner = {};
  };
  value.Init(tensor.release(), DataTypeImpl::GetType<Tensor>(), std::move(deleter));
  return value;
}

Status KVCacheManager::Allocate(const Entry& entry, int64_t length, int64_t capacity, Entry& out) {
  const size_t bytes = StorageBytes(entry.spec, capacity);
  auto shape = entry.spec.shape;
  shape[entry.spec.sequence_axis] = capacity;
  Entry next{entry.spec, {}, {{}, length, capacity}};
  Tensor::InitOrtValue(ElementType(entry.spec), TensorShape(shape), allocator_, next.storage);
  ORT_RETURN_IF(bytes != 0 && next.storage.Get<Tensor>().DataRaw() == nullptr, "CPU KV cache allocation failed");
  next.view.value = MakeView(next, length);
  if (bytes != 0) {
    stats_.allocations = SafeInt<size_t>(stats_.allocations) + 1;
  }
  out = std::move(next);
  return Status::OK();
}

Status KVCacheManager::ValidateValue(const Entry& entry, const OrtValue& value) {
  ORT_RETURN_IF_NOT(value.IsTensor(), "KV cache update must be a tensor: ", entry.spec.name);
  const auto& tensor = value.Get<Tensor>();
  ORT_RETURN_IF_NOT(tensor.Location().device.Type() == OrtDevice::CPU,
                    "CPU KV cache requires a CPU tensor: ", entry.spec.name);
#ifdef ENABLE_STRIDED_TENSORS
  ORT_RETURN_IF_NOT(tensor.IsContiguous(), "CPU KV cache requires a contiguous tensor: ", entry.spec.name);
#endif
  ORT_RETURN_IF(tensor.GetElementType() != entry.spec.element_type ||
                    tensor.Shape().NumDimensions() != entry.spec.shape.size(),
                "KV cache update dtype or rank mismatch: ", entry.spec.name);
  for (size_t i = 0; i < entry.spec.shape.size(); ++i) {
    ORT_RETURN_IF(i != static_cast<size_t>(entry.spec.sequence_axis) && tensor.Shape()[i] != entry.spec.shape[i],
                  "KV cache non-sequence dimension mismatch: ", entry.spec.name);
  }
  ORT_UNUSED_PARAMETER(StorageBytes(entry.spec, tensor.Shape()[entry.spec.sequence_axis]));
  return Status::OK();
}

Status KVCacheManager::Get(std::string_view name, KVCacheView& out) const {
  auto it = entries_.find(std::string(name));
  ORT_RETURN_IF(it == entries_.end(), "Unknown KV cache: ", name);
  out = it->second.view;
  return Status::OK();
}

Status KVCacheManager::Update(std::string_view name, const OrtValue& value, KVCacheUpdateKind kind) {
  auto it = entries_.find(std::string(name));
  ORT_RETURN_IF(it == entries_.end(), "Unknown KV cache: ", name);
  Entry& entry = it->second;
  ORT_RETURN_IF_ERROR(ValidateValue(entry, value));
  const auto& source = value.Get<Tensor>();
  const int64_t incoming = source.Shape()[entry.spec.sequence_axis];
  ORT_RETURN_IF(kind != KVCacheUpdateKind::Share && kind != KVCacheUpdateKind::Copy &&
                    kind != KVCacheUpdateKind::Append,
                "Unknown KV cache update kind");
  if (kind == KVCacheUpdateKind::Share) {
    ORT_RETURN_IF_NOT(source.OwnsBuffer(), "Sharing requires an allocator-owned KV output; copy borrowed tensors");
    ORT_RETURN_IF(incoming < entry.view.length, "KV cache update cannot shrink logical length; use Reset");
    Entry next{entry.spec, value, {value, incoming, incoming}};
    stats_.shared_updates = SafeInt<size_t>(stats_.shared_updates) + 1;
    entry = std::move(next);
    return Status::OK();
  }
  if (kind == KVCacheUpdateKind::Append && incoming == 0) {
    return Status::OK();
  }
  const int64_t length = kind == KVCacheUpdateKind::Append
                             ? static_cast<int64_t>(SafeInt<int64_t>(entry.view.length) + incoming)
                             : incoming;
  ORT_RETURN_IF(length < entry.view.length, "KV cache update cannot shrink logical length; use Reset");
  Entry next;
  ORT_RETURN_IF_ERROR(Allocate(entry, length, std::max(length, entry.view.capacity), next));
  size_t copied = 0;
  if (kind == KVCacheUpdateKind::Copy) {
    copied = source.SizeInBytes();
    if (copied != 0) {
      CPUDataTransfer transfer;
      ORT_RETURN_IF_ERROR(transfer.CopyTensor(source, *next.view.value.GetMutable<Tensor>()));
    }
  } else {
    SafeInt<size_t> outer = 1;
    SafeInt<size_t> inner_bytes = ElementType(entry.spec)->Size();
    for (size_t i = 0; i < entry.spec.shape.size(); ++i) {
      if (i < static_cast<size_t>(entry.spec.sequence_axis)) {
        outer *= entry.spec.shape[i];
      } else if (i > static_cast<size_t>(entry.spec.sequence_axis)) {
        inner_bytes *= entry.spec.shape[i];
      }
    }
    const size_t old_row = inner_bytes * entry.view.length;
    const size_t added_row = inner_bytes * incoming;
    const size_t new_row = inner_bytes * length;
    auto* destination = static_cast<char*>(next.view.value.GetMutable<Tensor>()->MutableDataRaw());
    const auto* previous = static_cast<const char*>(entry.view.value.Get<Tensor>().DataRaw());
    const auto* added = static_cast<const char*>(source.DataRaw());
    for (size_t row = 0; row < outer; ++row) {
      const size_t destination_offset = SafeInt<size_t>(row) * new_row;
      if (old_row != 0) {
        const size_t old_offset = SafeInt<size_t>(row) * old_row;
        std::memcpy(destination + destination_offset, previous + old_offset, old_row);
      }
      const size_t added_offset = SafeInt<size_t>(row) * added_row;
      const size_t append_offset = SafeInt<size_t>(destination_offset) + old_row;
      std::memcpy(destination + append_offset, added + added_offset, added_row);
    }
    copied = outer * new_row;
  }
  stats_.bytes_copied = SafeInt<size_t>(stats_.bytes_copied) + copied;
  entry = std::move(next);
  return Status::OK();
}

Status KVCacheManager::Reserve(std::string_view name, int64_t capacity) {
  auto it = entries_.find(std::string(name));
  ORT_RETURN_IF(it == entries_.end(), "Unknown KV cache: ", name);
  ORT_RETURN_IF(capacity < 0, "KV cache capacity must be nonnegative");
  Entry& entry = it->second;
  if (capacity <= entry.view.capacity) {
    return Status::OK();
  }
  Entry next;
  ORT_RETURN_IF_ERROR(Allocate(entry, entry.view.length, capacity, next));
  const size_t copied = StorageBytes(entry.spec, entry.view.length);
  if (copied != 0) {
    CPUDataTransfer transfer;
    ORT_RETURN_IF_ERROR(transfer.CopyTensor(entry.view.value.Get<Tensor>(), *next.view.value.GetMutable<Tensor>()));
  }
  stats_.bytes_copied = SafeInt<size_t>(stats_.bytes_copied) + copied;
  entry = std::move(next);
  return Status::OK();
}

void KVCacheManager::Reset() {
  for (auto& [name, entry] : entries_) {
    entry.view.value = MakeView(entry, 0);
    entry.view.length = 0;
  }
}

}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/kv_cache_manager.h"

#include <algorithm>
#include <cstring>

#include "core/common/safeint.h"
#include "core/framework/data_transfer.h"
#include "core/framework/error_code_helper.h"
#include "core/graph/graph.h"
#include "core/session/allocator_adapters.h"
#include "core/session/ort_apis.h"

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

KVCacheManager::KVCacheManager(InlinedVector<KVCacheSpec> specs, AllocatorPtr allocator,
                               StatefulProviderBinding provider)
    : allocator_(std::move(allocator)), dispatch_(std::move(provider)) {
  for (auto& spec : specs) {
    const auto name = spec.name;
    entries_.emplace(name, Entry{std::move(spec), {}, {}});
  }
}

Status KVCacheManager::Create(InlinedVector<KVCacheSpec> specs, AllocatorPtr allocator,
                              std::unique_ptr<KVCacheManager>& out, StatefulProviderBinding provider) {
  ORT_RETURN_IF_ERROR(ValidateSpecs(specs));
  ORT_RETURN_IF(specs.empty(), "KVCacheManager requires at least one cache");
  const auto& selection = provider.Selection();
  ORT_RETURN_IF_NOT(allocator &&
                        static_cast<int>(allocator->Info().device.Type()) == static_cast<int>(selection.device_type) &&
                        allocator->Info().device.Id() == selection.device_id,
                    "KVCacheManager requires the selected CPU/provider device allocator");
  auto manager = std::unique_ptr<KVCacheManager>(
      new KVCacheManager(std::move(specs), std::move(allocator), std::move(provider)));
  for (auto& [name, entry] : manager->entries_) {
    ORT_RETURN_IF_ERROR(manager->Apply(ORT_STATEFUL_CACHE_INITIALIZE, entry, nullptr, entry.spec.initial_capacity));
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

Status KVCacheManager::Allocate(const Entry& entry, int64_t length, int64_t capacity, Entry& out,
                                KVCacheStats& stats) {
  const size_t bytes = StorageBytes(entry.spec, capacity);
  auto shape = entry.spec.shape;
  shape[entry.spec.sequence_axis] = capacity;
  Entry next{entry.spec, {}, {{}, length, capacity}};
  Tensor::InitOrtValue(ElementType(entry.spec), TensorShape(shape), allocator_, next.storage);
  ORT_RETURN_IF(bytes != 0 && next.storage.Get<Tensor>().DataRaw() == nullptr, "CPU KV cache allocation failed");
  next.view.value = MakeView(next, length);
  if (bytes != 0) {
    stats.allocations = SafeInt<size_t>(stats.allocations) + 1;
  }
  out = std::move(next);
  return Status::OK();
}

Status KVCacheManager::ValidateValue(const Entry& entry, const OrtValue& value) const {
  ORT_RETURN_IF_NOT(value.IsTensor(), "KV cache update must be a tensor: ", entry.spec.name);
  const auto& tensor = value.Get<Tensor>();
  const auto& expected = allocator_->Info().device;
  const bool compatible = expected.Type() == OrtDevice::CPU ? tensor.Location().device.Type() == OrtDevice::CPU
                                                            : tensor.Location().device == expected;
  ORT_RETURN_IF_NOT(compatible,
                    "KV cache requires matching CPU/provider tensor storage: ", entry.spec.name);
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
  ORT_RETURN_IF(requires_reset_, "KV cache state requires Reset after a provider failure");
  auto it = entries_.find(std::string(name));
  ORT_RETURN_IF(it == entries_.end(), "Unknown KV cache: ", name);
  out = it->second.view;
  return Status::OK();
}

Status KVCacheManager::Update(std::string_view name, const OrtValue& value, KVCacheUpdateKind kind) {
  ORT_RETURN_IF(requires_reset_, "KV cache requires Reset after a provider failure");
  auto it = entries_.find(std::string(name));
  ORT_RETURN_IF(it == entries_.end(), "Unknown KV cache: ", name);
  Entry& entry = it->second;
  ORT_RETURN_IF_ERROR(ValidateValue(entry, value));
  const auto& source = value.Get<Tensor>();
  const int64_t incoming = source.Shape()[entry.spec.sequence_axis];
  ORT_RETURN_IF(kind != KVCacheUpdateKind::Share && kind != KVCacheUpdateKind::Copy &&
                    kind != KVCacheUpdateKind::Append,
                "Unknown KV cache update kind");
  ORT_RETURN_IF(kind == KVCacheUpdateKind::Share && !source.OwnsBuffer(),
                "Sharing requires an allocator-owned KV output; copy borrowed tensors");
  const int64_t length = kind == KVCacheUpdateKind::Append
                             ? static_cast<int64_t>(SafeInt<int64_t>(entry.view.length) + incoming)
                             : incoming;
  ORT_RETURN_IF(length < entry.view.length, "KV cache update cannot shrink logical length; use Reset");
  if (kind == KVCacheUpdateKind::Append && incoming == 0) {
    return Status::OK();
  }
  return Apply(static_cast<OrtStatefulProviderOperation>(static_cast<int>(kind) + 1), entry, &value, 0);
}

Status KVCacheManager::UpdateCpu(Entry& entry, const OrtValue& value, KVCacheUpdateKind kind, KVCacheStats& stats) {
  const auto& source = value.Get<Tensor>();
  const int64_t incoming = source.Shape()[entry.spec.sequence_axis];
  if (kind == KVCacheUpdateKind::Share) {
    Entry next{entry.spec, value, {value, incoming, incoming}};
    stats.shared_updates = SafeInt<size_t>(stats.shared_updates) + 1;
    entry = std::move(next);
    return Status::OK();
  }
  const int64_t length = kind == KVCacheUpdateKind::Append
                             ? static_cast<int64_t>(SafeInt<int64_t>(entry.view.length) + incoming)
                             : incoming;
  Entry next;
  ORT_RETURN_IF_ERROR(Allocate(entry, length, std::max(length, entry.view.capacity), next, stats));
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
  stats.bytes_copied = SafeInt<size_t>(stats.bytes_copied) + copied;
  entry = std::move(next);
  return Status::OK();
}

Status KVCacheManager::Reserve(std::string_view name, int64_t capacity) {
  ORT_RETURN_IF(requires_reset_, "KV cache requires Reset after a provider failure");
  auto it = entries_.find(std::string(name));
  ORT_RETURN_IF(it == entries_.end(), "Unknown KV cache: ", name);
  ORT_RETURN_IF(capacity < 0, "KV cache capacity must be nonnegative");
  Entry& entry = it->second;
  if (capacity <= entry.view.capacity) {
    return Status::OK();
  }
  return Apply(ORT_STATEFUL_CACHE_RESERVE, entry, nullptr, capacity);
}

Status KVCacheManager::ReserveCpu(Entry& entry, int64_t capacity, KVCacheStats& stats) {
  Entry next;
  ORT_RETURN_IF_ERROR(Allocate(entry, entry.view.length, capacity, next, stats));
  const size_t copied = StorageBytes(entry.spec, entry.view.length);
  if (copied != 0) {
    CPUDataTransfer transfer;
    ORT_RETURN_IF_ERROR(transfer.CopyTensor(entry.view.value.Get<Tensor>(), *next.view.value.GetMutable<Tensor>()));
  }
  stats.bytes_copied = SafeInt<size_t>(stats.bytes_copied) + copied;
  entry = std::move(next);
  return Status::OK();
}

struct KVCacheManager::CpuWork {
  KVCacheManager& manager;
  Entry next;
  OrtStatefulProviderOperation operation;
  const OrtValue* value;
  int64_t capacity;
  KVCacheStats stats{};
  bool called = false;

  Status Run() {
    ORT_RETURN_IF(called, "CPU reference may be invoked only once");
    called = true;
    ORT_RETURN_IF(manager.allocator_->Info().device.Type() != OrtDevice::CPU,
                  "CPU reference requires CPU storage; no device transfer is permitted");
    switch (operation) {
      case ORT_STATEFUL_CACHE_INITIALIZE: {
        Entry initialized;
        ORT_RETURN_IF_ERROR(manager.Allocate(next, 0, capacity, initialized, stats));
        next = std::move(initialized);
        return Status::OK();
      }
      case ORT_STATEFUL_CACHE_RESERVE:
        return manager.ReserveCpu(next, capacity, stats);
      case ORT_STATEFUL_CACHE_RESET:
        next.view.value = MakeView(next, 0);
        next.view.length = 0;
        return Status::OK();
      default:
        return manager.UpdateCpu(next, *value, static_cast<KVCacheUpdateKind>(static_cast<int>(operation) - 1), stats);
    }
  }
};

OrtStatus* ORT_API_CALL KVCacheManager::CpuReference(void* context, OrtStatefulCacheOperationResult* out) noexcept {
  API_IMPL_BEGIN
  ORT_API_RETURN_IF(!context || !out, ORT_INVALID_ARGUMENT, "CPU reference context/output is NULL");
  auto& work = *static_cast<CpuWork*>(context);
  ORT_API_RETURN_IF(work.called, ORT_INVALID_ARGUMENT, "CPU reference may be invoked only once");
  work.manager.dispatch_.RecordDelegation(work.operation);
  ORT_API_RETURN_IF_STATUS_NOT_OK(work.Run());
  auto storage = std::make_unique<OrtValue>(work.next.storage);
  const OrtStatefulCacheOperationResult result{
      sizeof(result), storage.get(), work.next.view.length, work.next.view.capacity,
      work.stats.allocations, work.stats.bytes_copied, work.stats.shared_updates};
  *out = result;
  storage.release();
  return nullptr;
  API_IMPL_END
}

Status KVCacheManager::Apply(OrtStatefulProviderOperation operation, Entry& entry,
                             const OrtValue* value, int64_t capacity) {
  bool provider = false;
  ORT_RETURN_IF_ERROR(dispatch_.Select(operation, entry.spec.element_type,
                                       allocator_->Info().device.Type() == OrtDevice::CPU, provider));
  CpuWork work{*this, entry, operation, value, capacity};
  dispatch_.RecordCall(operation, provider);
  Status status;
  if (!provider) {
    status = work.Run();
  } else {
    requires_reset_ = true;
    const OrtStatefulKVCacheSpec spec{entry.spec.name.c_str(),
                                      static_cast<ONNXTensorElementDataType>(entry.spec.element_type),
                                      entry.spec.shape.data(), entry.spec.shape.size(),
                                      entry.spec.sequence_axis, entry.spec.initial_capacity};
    auto allocator = std::make_shared<OrtAllocatorImplWrappingIAllocator>(AllocatorPtr{allocator_});
    const OrtStatefulCacheOperationRequest request{
        sizeof(request), OrtGetApiBase()->GetApi(ORT_API_VERSION), operation, &spec,
        entry.storage.IsTensor() ? &entry.storage : nullptr,
        entry.view.value.IsTensor() ? &entry.view.value : nullptr, value,
        entry.view.length, entry.view.capacity, capacity, allocator.get(), &work, &CpuReference};
    OrtStatefulCacheOperationResult result{};
    result.struct_size = sizeof(result);
    const auto& descriptor = dispatch_.Binding().Descriptor();
    status = ToStatusAndRelease(descriptor.CacheOperation(descriptor.context, &request, &result));
    std::unique_ptr<OrtValue> storage(result.storage);
    if (status.IsOK()) {
      status = [&]() -> Status {
        const size_t minimum_size =
            offsetof(OrtStatefulCacheOperationResult, shared_updates) + sizeof(result.shared_updates);
        ORT_RETURN_IF(result.struct_size < minimum_size ||
                          !storage,
                      "Provider returned an undersized or empty cache result");
        ORT_RETURN_IF_ERROR(ValidateValue(entry, *storage));
        ORT_RETURN_IF(result.length < 0 || result.capacity < result.length ||
                          storage->Get<Tensor>().Shape()[entry.spec.sequence_axis] != result.capacity,
                      "Provider returned invalid cache length/capacity");
        const int64_t incoming = value ? value->Get<Tensor>().Shape()[entry.spec.sequence_axis] : 0;
        const int64_t expected = operation == ORT_STATEFUL_CACHE_INITIALIZE || operation == ORT_STATEFUL_CACHE_RESET
                                     ? 0
                                 : operation == ORT_STATEFUL_CACHE_RESERVE ? entry.view.length
                                 : operation == ORT_STATEFUL_CACHE_APPEND
                                     ? static_cast<int64_t>(SafeInt<int64_t>(entry.view.length) + incoming)
                                     : incoming;
        const int64_t minimum = operation == ORT_STATEFUL_CACHE_INITIALIZE ? capacity
                                : operation == ORT_STATEFUL_CACHE_RESERVE  ? capacity
                                : operation == ORT_STATEFUL_CACHE_RESET    ? 0
                                : operation == ORT_STATEFUL_CACHE_SHARE    ? incoming
                                                                           : entry.view.capacity;
        ORT_RETURN_IF(result.length != expected || result.capacity < minimum,
                      "Provider violated cache operation extents");
        ORT_RETURN_IF(operation == ORT_STATEFUL_CACHE_SHARE &&
                          (result.capacity != incoming ||
                           storage->Get<Tensor>().DataRaw() != value->Get<Tensor>().DataRaw()),
                      "Provider Share must retain the allocator-owned output buffer");
        ORT_RETURN_IF(StorageBytes(entry.spec, result.capacity) != 0 &&
                          storage->Get<Tensor>().DataRaw() == nullptr,
                      "Provider returned NULL cache storage");
        Entry next{entry.spec, {}, {{}, result.length, result.capacity}};
        const bool existing = entry.storage.IsTensor() &&
                              storage->GetMutable<Tensor>() == entry.storage.GetMutable<Tensor>();
        next.storage = existing ? std::move(*storage)
                                : dispatch_.Binding().RetainValue(std::move(*storage), allocator);
        next.view.value = MakeView(next, result.length);
        work.next = std::move(next);
        work.stats = {result.allocations, result.bytes_copied, result.shared_updates};
        return Status::OK();
      }();
    }
  }
  if (!status.IsOK()) {
    dispatch_.RecordFailure(operation);
    return status;
  }
  KVCacheStats stats{
      SafeInt<size_t>(stats_.allocations) + work.stats.allocations,
      SafeInt<size_t>(stats_.bytes_copied) + work.stats.bytes_copied,
      SafeInt<size_t>(stats_.shared_updates) + work.stats.shared_updates};
  entry = std::move(work.next);
  stats_ = stats;
  if (operation != ORT_STATEFUL_CACHE_RESET) {
    requires_reset_ = false;
  }
  return Status::OK();
}

Status KVCacheManager::Reset() {
  requires_reset_ = true;
  for (auto& [name, entry] : entries_) {
    ORT_RETURN_IF_ERROR(Apply(ORT_STATEFUL_CACHE_RESET, entry, nullptr, 0));
  }
  requires_reset_ = false;
  return Status::OK();
}

}  // namespace onnxruntime

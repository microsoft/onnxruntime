// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/stateful_execution_provider.h"

#include "core/common/safeint.h"
#include "core/framework/execution_providers.h"
#include "core/framework/error_code_helper.h"
#include "core/session/IOBinding.h"
#include "core/framework/stateful_provider.h"
#include "core/session/inference_session.h"
#include "core/session/ort_apis.h"

namespace onnxruntime {
namespace {
constexpr uint64_t kCacheOperations = (uint64_t{1} << ORT_STATEFUL_SAMPLE_GREEDY) - 1;
constexpr uint64_t kOperations = (uint64_t{1} << 7) - 1;
constexpr uint64_t kCacheTypes = (uint64_t{1} << ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) |
                                 (uint64_t{1} << ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) |
                                 (uint64_t{1} << ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
constexpr uint64_t kSamplerTypes = kCacheTypes & ~(uint64_t{1} << ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);

bool ValidOperation(OrtStatefulProviderOperation operation) {
  return operation >= ORT_STATEFUL_CACHE_INITIALIZE && operation <= ORT_STATEFUL_SAMPLE_GREEDY;
}

Status FindProvider(InferenceSession& session, const char* name, std::shared_ptr<IExecutionProvider>& out) {
  ORT_RETURN_IF_NOT(name && *name, "Provider name must be nonempty");
  std::unique_ptr<IOBinding> binding;
  ORT_RETURN_IF_ERROR(session.NewIOBinding(&binding));
  for (const auto& candidate : session.GetExecutionProviders()) {
    if (candidate->Type() == name) {
      out = candidate;
      return Status::OK();
    }
  }
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unknown Session execution provider: ", name);
}
}  // namespace

struct StatefulProviderBinding::Lease {
  OrtStatefulProviderDescriptor descriptor{};
  std::shared_ptr<const void> logging_resources;
  std::shared_ptr<IExecutionProvider> provider;
  ~Lease() {
    if (descriptor.context) {
      descriptor.Release(descriptor.context);
    }
  }
};

Status StatefulProviderSelection::Parse(
    const OrtStatefulProviderSelection* input, StatefulProviderSelection& out) {
  const size_t minimum_size = offsetof(OrtStatefulProviderSelection, device_id) + sizeof(input->device_id);
  ORT_RETURN_IF_NOT(input &&
                        input->struct_size >= minimum_size,
                    "Provider selection is NULL or undersized");
  ORT_RETURN_IF(input->mode < ORT_STATEFUL_PROVIDER_CPU || input->mode > ORT_STATEFUL_PROVIDER_REQUIRE ||
                    input->device_type < OrtMemoryInfoDeviceType_CPU ||
                    input->device_type > OrtMemoryInfoDeviceType_NPU ||
                    input->device_id < -1,
                "Invalid provider policy or device");
  StatefulProviderSelection value;
  value.mode = input->mode;
  value.provider_name = input->provider_name ? input->provider_name : "";
  value.device_type = input->device_type;
  value.device_id = input->device_id;
  ORT_RETURN_IF(value.mode == ORT_STATEFUL_PROVIDER_CPU
                    ? (!value.provider_name.empty() || value.device_type != OrtMemoryInfoDeviceType_CPU ||
                       value.device_id != 0)
                    : value.provider_name.empty(),
                "CPU policy needs CPU device 0 and no name; other policies need a provider name");
  out = std::move(value);
  return Status::OK();
}

Status StatefulProviderBinding::ValidateDescriptor(const OrtStatefulProviderDescriptor& descriptor) {
  ORT_RETURN_IF(descriptor.struct_size <
                    offsetof(OrtStatefulProviderDescriptor, Release) + sizeof(descriptor.Release),
                "Undersized stateful provider descriptor");
  const auto& caps = descriptor.capabilities;
  ORT_RETURN_IF(caps.struct_size <
                    offsetof(OrtStatefulProviderCapabilities, device_id) + sizeof(caps.device_id),
                "Undersized stateful provider descriptor");
  ORT_RETURN_IF(caps.version != 1 || (caps.operations & ~kOperations) != 0 ||
                    (caps.cache_element_types & ~kCacheTypes) != 0 ||
                    (caps.sampler_element_types & ~kSamplerTypes) != 0 ||
                    (caps.memory_types & ~uint32_t{15}) != 0 || caps.device_id < -1,
                "Unsupported stateful descriptor version/capability");
  const bool cache = (caps.operations & kCacheOperations) != 0;
  const bool sampler = (caps.operations & (uint64_t{1} << ORT_STATEFUL_SAMPLE_GREEDY)) != 0;
  ORT_RETURN_IF(cache != (descriptor.CacheOperation != nullptr) ||
                    sampler != (descriptor.Greedy != nullptr) ||
                    (cache && caps.cache_element_types == 0) || (sampler && caps.sampler_element_types == 0) ||
                    (caps.operations != 0 && caps.memory_types == 0) ||
                    (descriptor.context && (!descriptor.Retain || !descriptor.Release)) ||
                    ((descriptor.Retain != nullptr) != (descriptor.Release != nullptr)),
                "Stateful capabilities and callbacks are inconsistent");
  return Status::OK();
}

OrtStatus* StatefulCpuBoundary(void* context, Status (*operation)(void*)) noexcept {
  API_IMPL_BEGIN
  ORT_API_RETURN_IF_STATUS_NOT_OK(operation(context));
  return nullptr;
  API_IMPL_END
}

Status StatefulProviderBinding::Resolve(InferenceSession* session, const StatefulProviderSelection& selection,
                                        StatefulProviderBinding& out) {
  OrtStatefulProviderSelection input{sizeof(input), selection.mode, selection.provider_name.c_str(),
                                     selection.device_type, selection.device_id};
  StatefulProviderBinding value;
  ORT_RETURN_IF_ERROR(StatefulProviderSelection::Parse(&input, value.selection_));
  if (selection.mode == ORT_STATEFUL_PROVIDER_CPU) {
    out = std::move(value);
    return Status::OK();
  }
  ORT_RETURN_IF_NOT(session, "Provider selection requires an initialized Session");
  std::shared_ptr<IExecutionProvider> provider;
  ORT_RETURN_IF_ERROR(FindProvider(*session, selection.provider_name.c_str(), provider));
  auto lease = std::make_shared<Lease>();
  lease->logging_resources = session->RetainProviderLoggingResources();
  if (const auto* descriptor = provider->GetStatefulProvider()) {
    ORT_RETURN_IF_ERROR(ValidateDescriptor(*descriptor));
    lease->descriptor = *descriptor;
    // Acquire before publishing a context to the lease's destructor.
    if (descriptor->context) {
      descriptor->Retain(descriptor->context);
    }
  } else {
    lease->descriptor.struct_size = sizeof(OrtStatefulProviderDescriptor);
    lease->descriptor.capabilities = {sizeof(OrtStatefulProviderCapabilities), 1, 0, 0, 0, 0, -1};
  }
  lease->provider = std::move(provider);
  const int supported_id = lease->descriptor.capabilities.device_id;
  if (value.selection_.device_id == -1) {
    ORT_RETURN_IF(supported_id == -1, "Ambiguous stateful provider device; select an explicit device id");
    value.selection_.device_id = supported_id;
  }
  ORT_RETURN_IF(supported_id >= 0 && supported_id != value.selection_.device_id,
                "Stateful provider device id mismatch");
  value.lease_ = std::move(lease);
  out = std::move(value);
  return Status::OK();
}

Status StatefulProviderBinding::Query(InferenceSession& session, const char* name,
                                      OrtStatefulProviderCapabilities& out) {
  std::shared_ptr<IExecutionProvider> provider;
  ORT_RETURN_IF_ERROR(FindProvider(session, name, provider));
  OrtStatefulProviderCapabilities caps{sizeof(caps), 1, 0, 0, 0, 0, -1};
  if (const auto* descriptor = provider->GetStatefulProvider()) {
    ORT_RETURN_IF_ERROR(ValidateDescriptor(*descriptor));
    caps = descriptor->capabilities;
  }
  out = caps;
  return Status::OK();
}

const OrtStatefulProviderDescriptor& StatefulProviderBinding::Descriptor() const {
  return lease_ ? lease_->descriptor : *GetCpuStatefulProviderDescriptor();
}

const char* StatefulProviderBinding::Name() const {
  return selection_.mode == ORT_STATEFUL_PROVIDER_CPU ? kCpuExecutionProvider : selection_.provider_name.c_str();
}

bool StatefulProviderBinding::CacheCompatible(const StatefulProviderBinding& other) const {
  const auto& a = Descriptor();
  const auto& b = other.Descriptor();
  return selection_.mode == other.selection_.mode && selection_.provider_name == other.selection_.provider_name &&
         selection_.device_type == other.selection_.device_type && selection_.device_id == other.selection_.device_id &&
         (a.capabilities.operations & kCacheOperations) == (b.capabilities.operations & kCacheOperations) &&
         a.capabilities.cache_element_types == b.capabilities.cache_element_types &&
         a.capabilities.memory_types == b.capabilities.memory_types && a.CacheOperation == b.CacheOperation;
}

OrtValue StatefulProviderBinding::RetainValue(OrtValue value, std::shared_ptr<const void> resource) const {
  if (!lease_ && !resource) {
    return value;
  }
  OrtValue retained;
  auto* tensor = value.GetMutable<Tensor>();
  retained.Init(tensor, DataTypeImpl::GetType<Tensor>(),
                [owner = std::move(value), lease = lease_, resource = std::move(resource)](void*) mutable {
                  owner = {};
                  resource.reset();
                  lease.reset();
                });
  return retained;
}

StatefulDispatch::StatefulDispatch(StatefulProviderBinding binding) : binding_(std::move(binding)) {
  for (size_t i = 0; i < records_.size(); ++i) {
    records_[i].struct_size = sizeof(OrtStatefulDispatchInfo);
    records_[i].operation = static_cast<OrtStatefulProviderOperation>(i);
    records_[i].origin = ORT_STATEFUL_DISPATCH_NOT_INVOKED;
  }
}

Status StatefulDispatch::Select(OrtStatefulProviderOperation operation, int element_type, bool cpu_compatible,
                                bool& provider) const {
  ORT_RETURN_IF_NOT(ValidOperation(operation) && element_type > 0 && element_type < 64,
                    "Invalid stateful operation or element type");
  const auto& selection = binding_.Selection();
  const auto& caps = binding_.Descriptor().capabilities;
  const uint64_t types =
      operation == ORT_STATEFUL_SAMPLE_GREEDY ? caps.sampler_element_types : caps.cache_element_types;
  const bool supported = (caps.operations & (uint64_t{1} << operation)) != 0 &&
                         (types & (uint64_t{1} << element_type)) != 0 &&
                         (caps.memory_types & (uint32_t{1} << selection.device_type)) != 0;
  provider = selection.mode != ORT_STATEFUL_PROVIDER_CPU && supported;
  if (selection.mode == ORT_STATEFUL_PROVIDER_REQUIRE && !supported) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Required stateful provider capability is unavailable: ",
                           binding_.Name(), " operation ", operation);
  }
  if (!provider && (!cpu_compatible || selection.device_type != OrtMemoryInfoDeviceType_CPU)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "CPU stateful fallback requires CPU-compatible storage; no implicit device transfer");
  }
  return Status::OK();
}

void StatefulDispatch::RecordCall(OrtStatefulProviderOperation operation, bool provider) {
  auto& info = records_.at(static_cast<size_t>(operation));
  info.origin = provider ? ORT_STATEFUL_DISPATCH_PROVIDER
                : binding_.Selection().mode == ORT_STATEFUL_PROVIDER_CPU
                    ? ORT_STATEFUL_DISPATCH_CPU
                    : ORT_STATEFUL_DISPATCH_CPU_FALLBACK;
  auto& count = provider ? info.provider_calls : info.cpu_calls;
  count = SafeInt<size_t>(count) + 1;
}

void StatefulDispatch::RecordDelegation(OrtStatefulProviderOperation operation) {
  auto& count = records_.at(static_cast<size_t>(operation)).cpu_delegations;
  count = SafeInt<size_t>(count) + 1;
}

void StatefulDispatch::RecordFailure(OrtStatefulProviderOperation operation) {
  auto& count = records_.at(static_cast<size_t>(operation)).failures;
  count = SafeInt<size_t>(count) + 1;
}

Status StatefulDispatch::GetInfo(OrtStatefulProviderOperation operation, OrtStatefulDispatchInfo& out) const {
  ORT_RETURN_IF_NOT(ValidOperation(operation), "Invalid stateful operation");
  auto info = records_[static_cast<size_t>(operation)];
  const auto& selection = binding_.Selection();
  info.provider_name = info.origin == ORT_STATEFUL_DISPATCH_PROVIDER      ? binding_.Name()
                       : info.origin == ORT_STATEFUL_DISPATCH_NOT_INVOKED ? ""
                                                                          : kCpuExecutionProvider;
  info.requested_provider_name = binding_.Name();
  info.policy = selection.mode;
  info.device_type = selection.device_type;
  info.device_id = selection.device_id;
  out = info;
  return Status::OK();
}

}  // namespace onnxruntime

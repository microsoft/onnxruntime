// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "allocator_adapters.h"
#include "core/framework/error_code_helper.h"
#include "core/session/abi_key_value_pairs.h"
#include "core/session/inference_session.h"
#include "core/session/ort_env.h"
#include "core/session/ort_apis.h"

namespace onnxruntime {

namespace {
// The ORT API maintains ABI and backward compatibility, allowing applications to be built with an older version
// and run with a newer one. Users may call `RegisterAllocator` with a custom allocator. However, any new
// function pointers introduced in the newer version may contain invalid values, as the older application
// is unaware of them.
// Therefore, it's necessary to check the version value in `OrtAllocatorImplWrappingIAllocator` and
// `IAllocatorImplWrappingOrtAllocator` to ensure compatibility.
constexpr uint32_t kOrtAllocatorReserveMinVersion = 18;
constexpr uint32_t kOrtAllocatorStatsMinVersion = 23;
}  // namespace

OrtAllocatorImplWrappingIAllocator::OrtAllocatorImplWrappingIAllocator(onnxruntime::AllocatorPtr&& i_allocator)
    : i_allocator_(std::move(i_allocator)) {
  OrtAllocator::version = ORT_API_VERSION;
  OrtAllocator::Alloc =
      [](OrtAllocator* this_, size_t size) { return static_cast<OrtAllocatorImplWrappingIAllocator*>(this_)->Alloc(size); };
  OrtAllocator::Free =
      [](OrtAllocator* this_, void* p) { static_cast<OrtAllocatorImplWrappingIAllocator*>(this_)->Free(p); };
  OrtAllocator::Info =
      [](const OrtAllocator* this_) { return static_cast<const OrtAllocatorImplWrappingIAllocator*>(this_)->Info(); };
  if (OrtAllocator::version >= kOrtAllocatorReserveMinVersion) {
    OrtAllocator::Reserve =
        [](OrtAllocator* this_, size_t size) { return static_cast<OrtAllocatorImplWrappingIAllocator*>(this_)->Reserve(size); };
  }
  if (OrtAllocator::version >= kOrtAllocatorStatsMinVersion) {
    OrtAllocator::GetStats =
        [](const OrtAllocator* this_, OrtKeyValuePairs** stats) noexcept -> OrtStatusPtr {
      API_IMPL_BEGIN
      auto kvp = std::make_unique<OrtKeyValuePairs>();
      auto stats_map = static_cast<const OrtAllocatorImplWrappingIAllocator*>(this_)->Stats();
      kvp->CopyFromMap(stats_map);
      *stats = reinterpret_cast<OrtKeyValuePairs*>(kvp.release());
      return nullptr;
      API_IMPL_END
    };
  }
}

void* OrtAllocatorImplWrappingIAllocator::Alloc(size_t size) {
  return i_allocator_->Alloc(size);
}

void* OrtAllocatorImplWrappingIAllocator::Reserve(size_t size) {
  return i_allocator_->Reserve(size);
}

void OrtAllocatorImplWrappingIAllocator::Free(void* p) {
  i_allocator_->Free(p);
}

const OrtMemoryInfo* OrtAllocatorImplWrappingIAllocator::Info() const {
  return &i_allocator_->Info();
}

std::unordered_map<std::string, std::string> OrtAllocatorImplWrappingIAllocator::Stats() const {
  AllocatorStats stats{};
  i_allocator_->GetStats(&stats);

  // Allocators which does not implement GetStats() will return empty stats
  std::unordered_map<std::string, std::string> entries;
  if (stats.num_allocs > 0 || stats.bytes_limit != 0) {
    entries.insert_or_assign("Limit", std::to_string(stats.bytes_limit));
    entries.insert_or_assign("InUse", std::to_string(stats.bytes_in_use));
    entries.insert_or_assign("TotalAllocated", std::to_string(stats.total_allocated_bytes));
    entries.insert_or_assign("MaxInUse", std::to_string(stats.max_bytes_in_use));
    entries.insert_or_assign("NumAllocs", std::to_string(stats.num_allocs));
    entries.insert_or_assign("NumReserves", std::to_string(stats.num_reserves));
    entries.insert_or_assign("NumArenaExtensions", std::to_string(stats.num_arena_extensions));
    entries.insert_or_assign("NumArenaShrinkages", std::to_string(stats.num_arena_shrinkages));
    entries.insert_or_assign("MaxAllocSize", std::to_string(stats.max_alloc_size));
  }
  return entries;
}

onnxruntime::AllocatorPtr OrtAllocatorImplWrappingIAllocator::GetWrappedIAllocator() {
  return i_allocator_;
}

IAllocatorImplWrappingOrtAllocator::IAllocatorImplWrappingOrtAllocator(OrtAllocator* ort_allocator)
    : IAllocator(*ort_allocator->Info(ort_allocator)), ort_allocator_(ort_allocator) {}

void* IAllocatorImplWrappingOrtAllocator::Alloc(size_t size) {
  return ort_allocator_->Alloc(ort_allocator_, size);
}

void* IAllocatorImplWrappingOrtAllocator::Reserve(size_t size) {
  if (ort_allocator_->version >= kOrtAllocatorReserveMinVersion && ort_allocator_->Reserve) {
    return ort_allocator_->Reserve(ort_allocator_, size);
  }

  return ort_allocator_->Alloc(ort_allocator_, size);
}

void IAllocatorImplWrappingOrtAllocator::Free(void* p) {
  return ort_allocator_->Free(ort_allocator_, p);
}

void IAllocatorImplWrappingOrtAllocator::GetStats(AllocatorStats* stats) {
  *stats = {};

  if (ort_allocator_->version >= kOrtAllocatorStatsMinVersion && ort_allocator_->GetStats) {
    OrtKeyValuePairs* kvps = nullptr;
    Ort::ThrowOnError(ort_allocator_->GetStats(ort_allocator_, &kvps));

    auto release_fn = [](OrtKeyValuePairs** kvp) {
      OrtApis::ReleaseKeyValuePairs(*kvp);
    };

    std::unique_ptr<OrtKeyValuePairs*, decltype(release_fn)> kvp_guard(&kvps, release_fn);

    const auto keys = kvps->Keys(), values = kvps->Values();

    for (size_t i = 0; i < keys.size(); ++i) {
      if (strcmp(keys[i], "Limit") == 0) {
        stats->bytes_limit = std::stoll(values[i]);
      } else if (strcmp(keys[i], "InUse") == 0) {
        stats->bytes_in_use = std::stoll(values[i]);
      } else if (strcmp(keys[i], "TotalAllocated") == 0) {
        stats->total_allocated_bytes = std::stoll(values[i]);
      } else if (strcmp(keys[i], "MaxInUse") == 0) {
        stats->max_bytes_in_use = std::stoll(values[i]);
      } else if (strcmp(keys[i], "NumAllocs") == 0) {
        stats->num_allocs = std::stoll(values[i]);
      } else if (strcmp(keys[i], "NumReserves") == 0) {
        stats->num_reserves = std::stoll(values[i]);
      } else if (strcmp(keys[i], "NumArenaExtensions") == 0) {
        stats->num_arena_extensions = std::stoll(values[i]);
      } else if (strcmp(keys[i], "NumArenaShrinkages") == 0) {
        stats->num_arena_shrinkages = std::stoll(values[i]);
      } else if (strcmp(keys[i], "MaxAllocSize") == 0) {
        stats->max_alloc_size = std::stoll(values[i]);
      }
    }
  }
}

}  // namespace onnxruntime
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(disable : 26409)
#endif
ORT_API_STATUS_IMPL(OrtApis::CreateAllocator, const OrtSession* sess,
                    const OrtMemoryInfo* mem_info, _Outptr_ OrtAllocator** out) {
  API_IMPL_BEGIN
  auto* session = reinterpret_cast<const ::onnxruntime::InferenceSession*>(sess);
  auto allocator_ptr = session->GetAllocator(*mem_info);
  if (!allocator_ptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "No requested allocator available");
  }
  *out = new onnxruntime::OrtAllocatorImplWrappingIAllocator(std::move(allocator_ptr));
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateAndRegisterAllocator, _Inout_ OrtEnv* env,
                    _In_ const OrtMemoryInfo* mem_info,
                    _In_ const OrtArenaCfg* arena_cfg) {
  using namespace onnxruntime;
  if (!env) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Env is null");
  }

  if (!mem_info) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "OrtMemoryInfo is null");
  }

  auto st = env->CreateAndRegisterAllocator(*mem_info, arena_cfg);

  if (!st.IsOK()) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, st.ErrorMessage().c_str());
  }
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::RegisterAllocator, _Inout_ OrtEnv* env,
                    _In_ OrtAllocator* allocator) {
  using namespace onnxruntime;
  if (!env) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Env is null");
  }

  if (!allocator) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Provided allocator is null");
  }

  if (allocator->Info(allocator)->alloc_type == OrtAllocatorType::OrtArenaAllocator) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Please register the allocator as OrtDeviceAllocator "
                                 "even if the provided allocator has arena logic built-in. "
                                 "OrtArenaAllocator is reserved for internal arena logic based "
                                 "allocators only.");
  }

  std::shared_ptr<IAllocator> i_alloc_ptr =
      std::make_shared<onnxruntime::IAllocatorImplWrappingOrtAllocator>(allocator);

  auto st = env->RegisterAllocator(i_alloc_ptr);

  if (!st.IsOK()) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, st.ErrorMessage().c_str());
  }
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::UnregisterAllocator, _Inout_ OrtEnv* env,
                    _In_ const OrtMemoryInfo* mem_info) {
  using namespace onnxruntime;
  if (!env) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Env is null");
  }

  if (!mem_info) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Provided OrtMemoryInfo is null");
  }

  auto st = env->UnregisterAllocator(*mem_info);

  if (!st.IsOK()) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, st.ErrorMessage().c_str());
  }
  return nullptr;
}

ORT_API(void, OrtApis::ReleaseAllocator, _Frees_ptr_opt_ OrtAllocator* allocator) {
  delete static_cast<onnxruntime::OrtAllocatorImpl*>(allocator);
}

ORT_API_STATUS_IMPL(OrtApis::CreateAndRegisterAllocatorV2, _Inout_ OrtEnv* env, _In_ const char* provider_type, _In_ const OrtMemoryInfo* mem_info, _In_ const OrtArenaCfg* arena_cfg,
                    _In_reads_(num_keys) const char* const* provider_options_keys, _In_reads_(num_keys) const char* const* provider_options_values, _In_ size_t num_keys) {
  using namespace onnxruntime;
  std::unordered_map<std::string, std::string> options;
  for (size_t i = 0; i != num_keys; i++) {
    if (provider_options_keys[i] == nullptr || provider_options_keys[i][0] == '\0' ||
        provider_options_values[i] == nullptr || provider_options_values[i][0] == '\0') {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Provider options key/value cannot be empty");
    }

    if (strlen(provider_options_keys[i]) > 1024 || strlen(provider_options_values[i]) > 1024) {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                   "Maximum string length for a provider options key/value is 1024.");
    }

    options[provider_options_keys[i]] = provider_options_values[i];
  }

  if (!env) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Env is null");
  }

  if (!mem_info) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "OrtMemoryInfo is null");
  }

  auto st = env->CreateAndRegisterAllocatorV2(provider_type, *mem_info, options, arena_cfg);

  if (!st.IsOK()) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, st.ErrorMessage().c_str());
  }
  return nullptr;
}

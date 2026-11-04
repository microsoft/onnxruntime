// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

#include "core/common/common.h"
#include "core/common/safeint.h"
#include "core/framework/ortdevice.h"
#include "core/platform/env.h"

struct OrtMemoryInfo;

namespace onnxruntime {
class IAllocator;
#ifndef SHARED_PROVIDER
class Tensor;
#endif
class Stream;

namespace common {
class Status;
}

inline constexpr size_t kExternalDataLoaderParallelReadThreshold = 8 * 1024 * 1024;

#if !defined(ORT_MINIMAL_BUILD)
struct ExternalDataLoaderOptions {
  static constexpr size_t kMaxReadingThreads = 64;
  static constexpr size_t kDefaultReadingThreads = 4;
  static constexpr size_t kDefaultMinAutoThreads = 2;
  static constexpr size_t kDefaultMaxAutoThreads = 8;

  static constexpr size_t kMinPinnedMemoryBytes = 1024 * 1024;           // 1 MiB (512 KiB per buffer)
  static constexpr size_t kMaxPinnedMemoryBytes = 1024 * 1024 * 1024;    // 1 GiB
  static constexpr size_t kPinnedMemoryAlignment = 8192;                 // 8 KiB total (4 KiB per buffer)
  static constexpr size_t kDefaultPinnedMemoryBytes = 32 * 1024 * 1024;  // 32 MiB total (two 16 MiB buffers)

  static constexpr size_t kMinChunkBytes = 4096;                 // 4 KiB
  static constexpr size_t kMaxChunkBytes = 64 * 1024 * 1024;     // 64 MiB
  static constexpr size_t kChunkAlignment = 4096;                // 4 KiB
  static constexpr size_t kDefaultChunkBytes = 2 * 1024 * 1024;  // 2 MiB

  static constexpr uint64_t kMaxMinSizeBytesU64 = uint64_t{4} * 1024 * 1024 * 1024;  // 4 GiB
  static constexpr size_t kDefaultMinSizeBytes = 16 * 1024 * 1024;                   // 16 MiB

  std::optional<size_t> reading_threads;
  std::optional<size_t> auto_reading_threads;
  std::optional<size_t> pinned_memory_bytes;
  std::optional<size_t> chunk_bytes;
  std::optional<size_t> min_size_bytes;

  static size_t GetDefaultReadingThreads() noexcept {
#if !defined(SHARED_PROVIDER)
    const int physical_cores = Env::Default().GetNumPhysicalCpuCores();
    if (physical_cores > 0) {
      return std::clamp<size_t>(static_cast<size_t>(physical_cores) / 2,
                                kDefaultMinAutoThreads,
                                kDefaultMaxAutoThreads);
    }
#endif
    const size_t hw_threads = std::thread::hardware_concurrency();
    if (hw_threads > 0) {
      return std::clamp<size_t>(hw_threads / 4,
                                kDefaultMinAutoThreads,
                                kDefaultMaxAutoThreads);
    }
    return kDefaultReadingThreads;
  }

  static constexpr bool IsValidReadingThreads(size_t value) noexcept {
    return value <= kMaxReadingThreads;
  }

  static constexpr bool IsValidPinnedMemoryBytes(size_t value) noexcept {
    return value >= kMinPinnedMemoryBytes &&
           value <= kMaxPinnedMemoryBytes &&
           (value % kPinnedMemoryAlignment) == 0;
  }

  static constexpr bool IsValidChunkBytes(size_t value) noexcept {
    return value >= kMinChunkBytes &&
           value <= kMaxChunkBytes &&
           (value % kChunkAlignment) == 0;
  }

  static constexpr bool IsValidMinSizeBytes(size_t value) noexcept {
    return static_cast<uint64_t>(value) <= kMaxMinSizeBytesU64;
  }

  static bool TryParseSizeT(std::string_view value_str, size_t& out) noexcept {
    if (value_str.empty()) {
      return false;
    }
    uint64_t parsed = 0;
    const char* const begin = value_str.data();
    const char* const end = begin + value_str.size();
    const auto res = std::from_chars(begin, end, parsed);
    if (res.ec != std::errc{} || res.ptr != end ||
        parsed > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
      return false;
    }
    out = static_cast<size_t>(parsed);
    return true;
  }

  static common::Status ValidateReadingThreads(size_t value, std::string_view key_name) {
    if (!IsValidReadingThreads(value)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             key_name, " must be between 0 and ", kMaxReadingThreads, ".");
    }
    return common::Status::OK();
  }

  static common::Status ValidatePinnedMemoryBytes(size_t value, std::string_view key_name) {
    if (!IsValidPinnedMemoryBytes(value)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             key_name, " must be between ", kMinPinnedMemoryBytes, " and ",
                             kMaxPinnedMemoryBytes, " bytes and a multiple of ",
                             kPinnedMemoryAlignment, ".");
    }
    return common::Status::OK();
  }

  static common::Status ValidateChunkBytes(size_t value, std::string_view key_name) {
    if (!IsValidChunkBytes(value)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             key_name, " must be between ", kMinChunkBytes, " and ",
                             kMaxChunkBytes, " bytes and a multiple of ",
                             kChunkAlignment, ".");
    }
    return common::Status::OK();
  }

  static common::Status ValidateMinSizeBytes(size_t value, std::string_view key_name) {
    if (!IsValidMinSizeBytes(value)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             key_name, " must be between 0 and ", kMaxMinSizeBytesU64, " bytes.");
    }
    return common::Status::OK();
  }

  static common::Status ParseReadingThreads(std::string_view value_str,
                                            std::string_view key_name,
                                            size_t& out) {
    size_t parsed = 0;
    if (!TryParseSizeT(value_str, parsed)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             key_name, " must be between 0 and ", kMaxReadingThreads, ".");
    }
    ORT_RETURN_IF_ERROR(ValidateReadingThreads(parsed, key_name));
    out = parsed;
    return common::Status::OK();
  }

  static common::Status ParsePinnedMemoryBytes(std::string_view value_str,
                                               std::string_view key_name,
                                               size_t& out) {
    size_t parsed = 0;
    if (!TryParseSizeT(value_str, parsed)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             key_name, " must be between ", kMinPinnedMemoryBytes, " and ",
                             kMaxPinnedMemoryBytes, " bytes and a multiple of ",
                             kPinnedMemoryAlignment, ".");
    }
    ORT_RETURN_IF_ERROR(ValidatePinnedMemoryBytes(parsed, key_name));
    out = parsed;
    return common::Status::OK();
  }

  static common::Status ParseChunkBytes(std::string_view value_str,
                                        std::string_view key_name,
                                        size_t& out) {
    size_t parsed = 0;
    if (!TryParseSizeT(value_str, parsed)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             key_name, " must be between ", kMinChunkBytes, " and ",
                             kMaxChunkBytes, " bytes and a multiple of ",
                             kChunkAlignment, ".");
    }
    ORT_RETURN_IF_ERROR(ValidateChunkBytes(parsed, key_name));
    out = parsed;
    return common::Status::OK();
  }

  static common::Status ParseMinSizeBytes(std::string_view value_str,
                                          std::string_view key_name,
                                          size_t& out) {
    size_t parsed = 0;
    if (!TryParseSizeT(value_str, parsed)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             key_name, " must be between 0 and ", kMaxMinSizeBytesU64, " bytes.");
    }
    ORT_RETURN_IF_ERROR(ValidateMinSizeBytes(parsed, key_name));
    out = parsed;
    return common::Status::OK();
  }
};
#endif  // !defined(ORT_MINIMAL_BUILD)

// Data transfer interface.
// Keep default virtual methods inline so shared providers do not require a non-exported RTTI symbol.
class IExternalDataLoader {
 public:
  virtual ~IExternalDataLoader() = default;

  virtual bool CanLoad(const OrtMemoryInfo& target_memory_info) const = 0;

  virtual bool SupportsDataType([[maybe_unused]] int32_t tensor_data_type) const { return true; }

#if !defined(ORT_MINIMAL_BUILD)
  virtual bool IsEnabled() const noexcept { return true; }

  virtual common::Status Configure([[maybe_unused]] const ExternalDataLoaderOptions& options) const {
    return common::Status::OK();
  }
#endif

#if !defined(ORT_MINIMAL_BUILD) || defined(ENABLE_D3D12_FILE_LOADING)
  // Begins one external-weight load transaction for a SessionState initialization.
  // Implementations should discard stale transaction state and initialize any
  // transaction-wide resources. Returning an error prevents the transaction from
  // continuing and causes EndLoad() to be called.
  virtual common::Status BeginLoad() const { return common::Status::OK(); }

  // Optionally reports the total external-weight bytes this loader is expected to receive in the active
  // transaction. FinalizeLoad() may be called more than once per transaction, so loaders can use this to make
  // size-based decisions from the full total instead of each partial batch.
  virtual void SetExpectedLoadBytes([[maybe_unused]] size_t total_bytes) const {}

  // Flushes and synchronizes any queued or in-flight external-weight loads before
  // SessionState initialization proceeds to kernel creation and prepacking.
  virtual common::Status FinalizeLoad(
      [[maybe_unused]] const std::function<bool()>& is_canceled = {}) const {
    return common::Status::OK();
  }

  // Ends the active transaction. Cancel outstanding work and release candidate
  // metadata and resources still owned by the loader, without invalidating allocations
  // already transferred by successful LoadTensor() calls. This is called after all
  // expected LoadTensor() calls or after any earlier failure or cancellation. It may
  // be called more than once, so implementations must be noexcept and idempotent.
  virtual void EndLoad() const noexcept {}
#endif

#if defined(ENABLE_D3D12_FILE_LOADING)
  // Returns true when the loader creates the tensor's backing allocation instead of
  // writing into a tensor allocated by the framework.
  virtual bool CreatesTensorForDevice([[maybe_unused]] const OrtDevice& target_device) const { return false; }

  // Registers one external tensor with the active transaction. The path, offset, and length
  // have already been resolved and validated against the TensorProto by the caller.
  // This incremental registration lets ORT report candidates as it discovers them
  // without retaining a separate core-side candidate collection. The loader owns the
  // candidate metadata needed to plan file access, destination allocation, and shared
  // device work. ORT does not call LoadTensor() during this phase.
  //
  // RegisterLoadCandidate() calls occur after a successful BeginLoad() and before
  // CommitLoadCandidates(). The same file may be referenced by multiple tensors, and
  // a transaction may contain no candidates. Returning an error ends the transaction.
  // The default implementation retains no candidate state.
  virtual common::Status RegisterLoadCandidate([[maybe_unused]] const Env& env,
                                               [[maybe_unused]] const std::filesystem::path& data_file_path,
                                               [[maybe_unused]] std::string_view tensor_name,
                                               [[maybe_unused]] FileOffsetType data_offset,
                                               [[maybe_unused]] SafeInt<size_t> data_length) const {
    return common::Status::OK();
  }

  // Commits the complete set of registered candidates before tensor loading begins.
  // This is where a loader can coalesce file ranges, allocate destinations, perform
  // shared I/O or device work, and synchronize the results. On success, the loader
  // must either make each accepted candidate ready for a subsequent LoadTensor() call
  // or make CanLoad() return false so normal loading can take over.
  //
  // ORT then makes a separate initializer-deserialization pass and calls LoadTensor()
  // for each tensor still claimed by this loader. The loader must retain unclaimed
  // committed tensors until LoadTensor() transfers their ownership or EndLoad()
  // releases them. This method is called even when no candidates were registered.
  // Long-running work should check is_canceled and return MODEL_LOAD_CANCELED when
  // cancellation is observed. Returning any error causes EndLoad() to be called.
  // The default implementation has no work to commit.
  virtual common::Status CommitLoadCandidates(
      [[maybe_unused]] const std::function<bool()>& is_canceled) const {
    return common::Status::OK();
  }
#endif

  // Tensor should be allocated with the correct memory info and size. A loader that
  // creates tensors for the target device replaces it with one backed by memory
  // owned through allocator.
#if defined(ENABLE_D3D12_FILE_LOADING)
  virtual common::Status LoadTensor([[maybe_unused]] const Env& env,
                                    [[maybe_unused]] const std::filesystem::path& data_file_path,
                                    [[maybe_unused]] std::string_view tensor_name,
                                    [[maybe_unused]] FileOffsetType data_offset,
                                    [[maybe_unused]] SafeInt<size_t> data_length,
                                    [[maybe_unused]] const std::shared_ptr<IAllocator>& allocator,
                                    [[maybe_unused]] Tensor& tensor) const {
    ORT_NOT_IMPLEMENTED(__FUNCTION__, " is not implemented");
  }
#endif

  // Tensor should be already allocated with the correct memory info and size.
#if defined(__wasm__)
  virtual common::Status LoadTensor([[maybe_unused]] const Env& env,
                                    [[maybe_unused]] const std::filesystem::path& data_file_path,
                                    [[maybe_unused]] FileOffsetType data_offset,
                                    [[maybe_unused]] SafeInt<size_t> data_length,
                                    [[maybe_unused]] Tensor& tensor) const {
    ORT_NOT_IMPLEMENTED(__FUNCTION__, " is not implemented");
  }
#else
  virtual common::Status LoadTensor([[maybe_unused]] const RandomAccessFile& file,
                                    [[maybe_unused]] FileOffsetType data_offset,
                                    [[maybe_unused]] SafeInt<size_t> data_length,
                                    [[maybe_unused]] Tensor& tensor) const {
    ORT_NOT_IMPLEMENTED(__FUNCTION__, " is not implemented");
  }
#endif
};

#if defined(__wasm__)

enum class ExternalDataLoadType {
  CPU = 0,
#if defined(USE_JSEP) || defined(USE_WEBGPU)
  WEBGPU_BUFFER = 1,
#endif
};

// Entry point for loading external data implementation using inline JavaScript.
common::Status LoadWebAssemblyExternalData(const Env& env,
                                           const std::filesystem::path& data_file_path,
                                           FileOffsetType data_offset,
                                           SafeInt<size_t> data_length,
                                           ExternalDataLoadType load_type,
                                           void* tensor_data);

#endif

}  // namespace onnxruntime

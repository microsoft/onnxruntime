// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifdef _WIN32
// Define NOMINMAX (and WIN32_LEAN_AND_MEAN) before <windows.h> so the min/max macros it would otherwise pull in do
// not clobber std::numeric_limits<...>::max() and std::min/std::max used in this header.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "plugin_ep_utils.h"
#include "onnxruntime_experimental_cxx_api.h"

/**
 * \file
 * \brief Sample-only EPContext data helpers shared by the example plugin EP and its tests.
 *
 * These helpers are intentionally outside the ORT C and EP ABI and are provided as a reference for EP authors that
 * need to handle external (non-embedded) EPContext binary data.
 *
 * The intended entry points for EP implementers are:
 *  - ReadEpContextData(api, config, file_name, graph, out, allocator): reads via an application-supplied
 *    OrtReadNamedBufferFunc (carried by OrtEpContextConfig) or the file fallback, returning an owning EpContextData
 *    buffer that avoids copying large data.
 *  - WriteEpContextDataWithFileFallback(api, config, ...): writes via an application-supplied OrtWriteNamedBufferFunc
 *    or the file fallback.
 *
 * The other functions are lower-level building blocks. Production EPs should additionally apply their own sandboxing,
 * size limits, and path policies; see the per-function notes on how untrusted, model-derived names are treated.
 */
namespace ep_context_data_utils {

#ifdef _WIN32
inline std::string WindowsLastErrorMessage(std::string_view message, DWORD error_code) {
  return std::string{message} + " GetLastError=" + std::to_string(error_code);
}

// Converts a UTF-8 string to a wide string. Reports conversion failures (e.g., invalid UTF-8) via OrtStatus* instead
// of silently returning an empty string. An empty input yields an empty output and a success status.
inline OrtStatus* Utf8ToWideString(const OrtApi& api, std::string_view value, std::wstring& wide_value) {
  wide_value.clear();
  if (value.empty()) {
    return nullptr;
  }
  if (value.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return api.CreateStatus(ORT_INVALID_ARGUMENT, "EPContext data file name is too long to convert");
  }

  const int wide_length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                                              static_cast<int>(value.size()), nullptr, 0);
  if (wide_length <= 0) {
    const std::string message = WindowsLastErrorMessage(
        "EPContext data file name is not valid UTF-8 or could not be converted to a wide string.", GetLastError());
    return api.CreateStatus(ORT_INVALID_ARGUMENT, message.c_str());
  }

  wide_value.resize(static_cast<size_t>(wide_length));
  const int converted = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                                            static_cast<int>(value.size()), wide_value.data(), wide_length);
  if (converted != wide_length) {
    wide_value.clear();
    const std::string message = WindowsLastErrorMessage("Failed to convert EPContext data file name to a wide string.",
                                                        GetLastError());
    return api.CreateStatus(ORT_FAIL, message.c_str());
  }
  return nullptr;
}

// Converts a wide string to UTF-8. Reports conversion failures via OrtStatus* instead of silently returning an empty
// string. An empty input yields an empty output and a success status.
inline OrtStatus* WideToUtf8String(const OrtApi& api, std::wstring_view value, std::string& utf8_value) {
  utf8_value.clear();
  if (value.empty()) {
    return nullptr;
  }
  if (value.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return api.CreateStatus(ORT_INVALID_ARGUMENT, "EPContext data file name is too long to convert");
  }

  const int utf8_length = WideCharToMultiByte(CP_UTF8, 0, value.data(), static_cast<int>(value.size()),
                                              nullptr, 0, nullptr, nullptr);
  if (utf8_length <= 0) {
    const std::string message = WindowsLastErrorMessage(
        "EPContext data file name could not be converted to UTF-8.", GetLastError());
    return api.CreateStatus(ORT_INVALID_ARGUMENT, message.c_str());
  }

  utf8_value.resize(static_cast<size_t>(utf8_length));
  const int converted = WideCharToMultiByte(CP_UTF8, 0, value.data(), static_cast<int>(value.size()),
                                            utf8_value.data(), utf8_length, nullptr, nullptr);
  if (converted != utf8_length) {
    utf8_value.clear();
    const std::string message = WindowsLastErrorMessage("Failed to convert EPContext data file name to UTF-8.",
                                                        GetLastError());
    return api.CreateStatus(ORT_FAIL, message.c_str());
  }
  return nullptr;
}
#endif

// Converts a UTF-8 path to a std::filesystem::path. A null or empty input yields an empty path and a success status;
// conversion failures are reported via OrtStatus*.
inline OrtStatus* Utf8Path(const OrtApi& api, const char* path, std::filesystem::path& out_path) {
  out_path.clear();
  if (path == nullptr || path[0] == '\0') {
    return nullptr;
  }

#ifdef _WIN32
  std::wstring wide_path;
  RETURN_IF_ERROR(Utf8ToWideString(api, path, wide_path));
  out_path = std::filesystem::path{wide_path};
#else
  (void)api;
  out_path = std::filesystem::path{path};
#endif
  return nullptr;
}

inline OrtStatus* PathToUtf8String(const OrtApi& api, const std::filesystem::path& path, std::string& utf8_path) {
  utf8_path.clear();
#ifdef _WIN32
  RETURN_IF_ERROR(WideToUtf8String(api, path.wstring(), utf8_path));
#else
  (void)api;
  utf8_path = path.string();
#endif
  return nullptr;
}

inline std::string PathToUtf8StringForMessage(const std::filesystem::path& path) {
  std::string utf8_path;
  Ort::Status status{PathToUtf8String(Ort::GetApi(), path, utf8_path)};
  return status.IsOK() ? utf8_path : std::string{"<path conversion failed>"};
}

inline bool HasAbsoluteOrRootedPath(const std::filesystem::path& path) {
  return path.is_absolute() || path.has_root_name() || path.has_root_directory();
}

// Read gate: the resolved path must exist and be a regular file. status() follows symlinks, matching the
// canonicalization used for containment. This rejects directories and special files - notably a FIFO, which
// containment does not catch and which blocks an ifstream open, making it a cheap denial of service. Advisory, not a
// security boundary: the path can change between this check and the open (TOCTOU).
inline OrtStatus* EnsureRegularFileForRead(const OrtApi& api, const std::filesystem::path& data_path) {
  std::error_code ec;
  const std::filesystem::file_status file_status = std::filesystem::status(data_path, ec);
  if (ec || !std::filesystem::exists(file_status)) {
    const std::string message = "Failed to open EPContext data file for read: " +
                                PathToUtf8StringForMessage(data_path);
    return api.CreateStatus(ORT_FAIL, message.c_str());
  }

  if (!std::filesystem::is_regular_file(file_status)) {
    const std::string message = "EPContext data file must be a regular file: " +
                                PathToUtf8StringForMessage(data_path);
    return api.CreateStatus(ORT_INVALID_ARGUMENT, message.c_str());
  }

  return nullptr;
}

// Write counterpart: a not-yet-existing target is the normal case and is allowed, but an existing target must be a
// regular file. Same advisory (TOCTOU) caveat as EnsureRegularFileForRead().
//
// A symlink leaf is rejected. This closes a gap containment alone cannot: a *dangling* symlink (one whose target
// does not exist yet) reports as not_found, so weakly_canonical() leaves the link path untouched and it passes
// containment while pointing outside the model directory - an ofstream would then follow it and create the target
// there. Detecting that requires symlink_status(), which inspects the link instead of following it; status() would
// report the missing target as not_found and wave the write through.
//
// Note the division of labor: for a model-relative name that resolves, weakly_canonical() has already replaced the
// link with its target before this runs, so containment is what constrains those writes. This check is what covers
// the dangling case, and trusted (graph == nullptr) paths, which skip resolution altogether.
inline OrtStatus* EnsureRegularFileForWrite(const OrtApi& api, const std::filesystem::path& data_path) {
  std::error_code symlink_ec;
  const std::filesystem::file_status link_status = std::filesystem::symlink_status(data_path, symlink_ec);
  if (!symlink_ec && std::filesystem::is_symlink(link_status)) {
    const std::string message = "EPContext data file must not be a symlink: " +
                                PathToUtf8StringForMessage(data_path);
    return api.CreateStatus(ORT_INVALID_ARGUMENT, message.c_str());
  }

  std::error_code ec;
  const std::filesystem::file_status file_status = std::filesystem::status(data_path, ec);
  if (ec || !std::filesystem::exists(file_status)) {
    return nullptr;
  }

  if (!std::filesystem::is_regular_file(file_status)) {
    const std::string message = "EPContext data file must be a regular file: " +
                                PathToUtf8StringForMessage(data_path);
    return api.CreateStatus(ORT_INVALID_ARGUMENT, message.c_str());
  }

  return nullptr;
}

// Returns true if `candidate_full` (a base-relative name already combined with `base`) resolves to a location inside
// `base`. Both are normalized with std::filesystem::weakly_canonical, which resolves "." / ".." and any symlinks in
// the existing portion of the path, so a name that escapes `base` directly or through a symlink is rejected. On
// success the canonical resolved path is written to `resolved`.
inline bool IsResolvedPathWithinBase(const std::filesystem::path& base, const std::filesystem::path& candidate_full,
                                     std::filesystem::path& resolved) {
  std::error_code ec;
  const std::filesystem::path base_for_canon = base.empty() ? std::filesystem::path{"."} : base;
  const std::filesystem::path canonical_base = std::filesystem::weakly_canonical(base_for_canon, ec);
  if (ec) {
    return false;
  }
  std::filesystem::path candidate_resolved = std::filesystem::weakly_canonical(candidate_full, ec);
  if (ec) {
    return false;
  }
  const std::filesystem::path relative = candidate_resolved.lexically_relative(canonical_base);
  if (relative.empty() || *relative.begin() == std::filesystem::path{".."}) {
    return false;
  }

  resolved = std::move(candidate_resolved);
  return true;
}

// Resolves `file_name` to a filesystem path for reading or writing EPContext data (used by both the read path and
// the write-fallback path).
//
// When `graph` is null the caller is trusted and owns the path: `file_name` is returned as-is and may be absolute and
// may contain ".." (there is no model directory to contain against, and absolute paths are already permitted, so no
// traversal check is applied). When `graph` is non-null, `file_name` originates from the
// untrusted EPContext model "ep_cache_context" attribute: the graph must have a model path, the name must be
// relative, and after combining it with the model's directory the result must stay within that directory. Symlinks and
// ".." are resolved (via weakly_canonical), so a name that escapes the model directory - including through a symlink -
// is rejected.
// This helper only decides whether a model-derived file name resolves inside the model directory; the file type is
// checked by EnsureRegularFileForRead() / EnsureRegularFileForWrite() at the point of use. Production EPs should
// still choose an application-approved storage root (sandbox) and cap the number of bytes they will read or write for
// a single EPContext payload.
inline OrtStatus* ResolveEpContextDataPath(const OrtApi& api, const char* file_name, const OrtGraph* graph,
                                           std::filesystem::path& data_path) {
  data_path.clear();

  if (file_name == nullptr || file_name[0] == '\0') {
    return api.CreateStatus(ORT_INVALID_ARGUMENT, "EPContext data file name must not be empty");
  }

  std::filesystem::path candidate_path;
  RETURN_IF_ERROR(Utf8Path(api, file_name, candidate_path));
  if (candidate_path.empty()) {
    return api.CreateStatus(ORT_INVALID_ARGUMENT, "EPContext data file name is not a valid path");
  }

  // Trusted direct callers (graph == nullptr) own the path and may pass an absolute physical path, including one with
  // ".." components. There is no model directory to contain against here, and absolute paths are already allowed, so
  // no traversal check is applied; the untrusted (model-relative) branch below is the one constrained to the model
  // directory.
  if (graph == nullptr) {
    data_path = candidate_path;
    return nullptr;
  }

  // Untrusted (model-derived) name: must be relative and must resolve within the model directory.
  if (HasAbsoluteOrRootedPath(candidate_path)) {
    return api.CreateStatus(ORT_INVALID_ARGUMENT, "EPContext data file name must not be absolute or rooted");
  }

  const ORTCHAR_T* model_path = nullptr;
  RETURN_IF_ERROR(api.Graph_GetModelPath(graph, &model_path));
  if (model_path == nullptr || model_path[0] == 0) {
    return api.CreateStatus(ORT_INVALID_ARGUMENT,
                            "EPContext data file fallback requires a model path to resolve relative names");
  }

  const std::filesystem::path base_dir = std::filesystem::path{model_path}.parent_path();
  std::filesystem::path resolved;
  if (!IsResolvedPathWithinBase(base_dir, base_dir / candidate_path, resolved)) {
    return api.CreateStatus(ORT_INVALID_ARGUMENT,
                            "EPContext data file name must resolve to a path within the model directory");
  }

  data_path = resolved;
  return nullptr;
}

inline OrtStatus* WriteEpContextDataToResolvedFile(const OrtApi& api, const std::filesystem::path& data_path,
                                                   const void* buffer, size_t buffer_size) {
  RETURN_IF_ERROR(EnsureRegularFileForWrite(api, data_path));

  std::ofstream output_stream(data_path, std::ios::binary);
  if (!output_stream) {
    const std::string message = "Failed to open EPContext data file for write: " +
                                PathToUtf8StringForMessage(data_path);
    return api.CreateStatus(ORT_FAIL, message.c_str());
  }

  if (buffer_size != 0) {
    if (buffer_size > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
      return api.CreateStatus(ORT_INVALID_ARGUMENT, "EPContext data buffer is too large to write");
    }

    output_stream.write(static_cast<const char*>(buffer), static_cast<std::streamsize>(buffer_size));
    if (!output_stream) {
      const std::string message = "Failed to write EPContext data file: " +
                                  PathToUtf8StringForMessage(data_path);
      return api.CreateStatus(ORT_FAIL, message.c_str());
    }
  }

  return nullptr;
}

inline OrtStatus* ReadEpContextDataFromFile(const OrtApi& api, const char* file_name, const OrtGraph* graph,
                                            std::vector<char>& data) {
  data.clear();

  std::filesystem::path data_path;
  RETURN_IF_ERROR(ResolveEpContextDataPath(api, file_name, graph, data_path));
  RETURN_IF_ERROR(EnsureRegularFileForRead(api, data_path));

  std::ifstream input_stream(data_path, std::ios::binary);
  if (!input_stream) {
    const std::string message = "Failed to open EPContext data file for read: " +
                                PathToUtf8StringForMessage(data_path);
    return api.CreateStatus(ORT_FAIL, message.c_str());
  }

  data.assign(std::istreambuf_iterator<char>{input_stream}, std::istreambuf_iterator<char>{});
  if (!input_stream) {
    const std::string message = "Failed to read EPContext data file: " +
                                PathToUtf8StringForMessage(data_path);
    return api.CreateStatus(ORT_FAIL, message.c_str());
  }

  return nullptr;
}

// Reads the resolved EPContext data file into a buffer allocated from `allocator`, so a caller-supplied allocator is
// honored on the file-fallback path just as it is on the callback path. On success `*out_buffer` owns `*out_size`
// bytes and must be freed by the caller via the same `allocator`; on failure (and for an empty file) `*out_buffer` is
// null and `*out_size` is 0. `graph` governs name resolution exactly as in ReadEpContextDataFromFile().
//
// `allocator`, `out_buffer` and `out_size` must all be non-null. Unlike the other helpers here, which take C++
// references for their outputs, this one takes raw out-pointers, so it validates them and reports a bad argument as
// an OrtStatus* rather than dereferencing null.
inline OrtStatus* ReadEpContextDataFromFileWithAllocator(const OrtApi& api, const char* file_name,
                                                         const OrtGraph* graph, OrtAllocator* allocator,
                                                         void** out_buffer, size_t* out_size) {
  if (out_buffer == nullptr || out_size == nullptr) {
    return api.CreateStatus(ORT_INVALID_ARGUMENT,
                            "EPContext data file read requires non-null out_buffer and out_size pointers");
  }

  if (allocator == nullptr) {
    return api.CreateStatus(ORT_INVALID_ARGUMENT, "EPContext data file read requires a non-null allocator");
  }

  *out_buffer = nullptr;
  *out_size = 0;

  std::filesystem::path data_path;
  RETURN_IF_ERROR(ResolveEpContextDataPath(api, file_name, graph, data_path));
  RETURN_IF_ERROR(EnsureRegularFileForRead(api, data_path));

  // Open at the end (std::ios::ate) so tellg() reports the byte count; binary mode keeps that count exact.
  std::ifstream input_stream(data_path, std::ios::binary | std::ios::ate);
  if (!input_stream) {
    const std::string message = "Failed to open EPContext data file for read: " +
                                PathToUtf8StringForMessage(data_path);
    return api.CreateStatus(ORT_FAIL, message.c_str());
  }

  const std::streampos end_pos = input_stream.tellg();
  if (end_pos < 0) {
    const std::string message = "Failed to determine EPContext data file size: " +
                                PathToUtf8StringForMessage(data_path);
    return api.CreateStatus(ORT_FAIL, message.c_str());
  }

  const auto byte_count_wide = static_cast<std::uintmax_t>(end_pos);
  if (byte_count_wide > static_cast<std::uintmax_t>(std::numeric_limits<size_t>::max()) ||
      byte_count_wide > static_cast<std::uintmax_t>(std::numeric_limits<std::streamsize>::max())) {
    return api.CreateStatus(ORT_INVALID_ARGUMENT, "EPContext data file is too large to read");
  }
  const size_t byte_count = static_cast<size_t>(byte_count_wide);
  if (byte_count == 0) {
    return nullptr;  // Empty file: leave *out_buffer null / *out_size 0 (no allocation needed).
  }

  input_stream.seekg(0, std::ios::beg);
  if (!input_stream) {
    const std::string message = "Failed to read EPContext data file: " +
                                PathToUtf8StringForMessage(data_path);
    return api.CreateStatus(ORT_FAIL, message.c_str());
  }

  void* buffer = nullptr;
  RETURN_IF_ERROR(api.AllocatorAlloc(allocator, byte_count, &buffer));
  if (buffer == nullptr) {
    return api.CreateStatus(ORT_FAIL, "Allocator returned a null buffer for the EPContext data file read");
  }

  // Free the freshly allocated buffer via the same allocator on any error path below; release it to the caller on
  // success. Release any AllocatorFree status without throwing, keeping errors on the OrtStatus* path.
  auto buffer_deleter = [&api, allocator](void* buffer_to_free) {
    if (buffer_to_free != nullptr) {
      Ort::Status free_status{api.AllocatorFree(allocator, buffer_to_free)};
      static_cast<void>(free_status);
    }
  };
  std::unique_ptr<void, decltype(buffer_deleter)> buffer_guard(buffer, buffer_deleter);

  input_stream.read(static_cast<char*>(buffer), static_cast<std::streamsize>(byte_count));
  if (!input_stream || static_cast<size_t>(input_stream.gcount()) != byte_count) {
    const std::string message = "Failed to read EPContext data file: " +
                                PathToUtf8StringForMessage(data_path);
    return api.CreateStatus(ORT_FAIL, message.c_str());
  }

  *out_buffer = buffer_guard.release();
  *out_size = byte_count;
  return nullptr;
}

inline OrtStatus* WriteEpContextDataToFile(const OrtApi& api, const char* file_name, const OrtGraph* graph,
                                           const void* buffer, size_t buffer_size) {
  if (buffer == nullptr && buffer_size != 0) {
    return api.CreateStatus(ORT_INVALID_ARGUMENT, "EPContext data buffer must not be null for non-empty data");
  }

  std::filesystem::path data_path;
  RETURN_IF_ERROR(ResolveEpContextDataPath(api, file_name, graph, data_path));
  return WriteEpContextDataToResolvedFile(api, data_path, buffer, buffer_size);
}

// Forward declaration so EpContextData can grant the populating read helper access to its internals.
class EpContextData;
inline OrtStatus* ReadEpContextData(const OrtApi& api, OrtReadNamedBufferFunc read_func, void* read_state,
                                    const char* file_name, const OrtGraph* graph, EpContextData& out,
                                    OrtAllocator* allocator = nullptr);

// RAII owner for the bytes returned by an EPContext read, used to avoid copying potentially large data. Both the
// app-supplied read-callback path and the file-fallback path place the bytes in a buffer obtained from an
// OrtAllocator: the callback path adopts the buffer the callback allocated, and the file path reads straight into a
// buffer allocated from the same (optionally caller-supplied) allocator. Either way the bytes are freed via that
// allocator on destruction and are accessed through data()/size() without an extra copy. EPs that handle large
// EPContext blobs should prefer ReadEpContextData() + EpContextData.
class EpContextData {
 public:
  EpContextData() = default;
  ~EpContextData() { FreeAllocatorBuffer(); }

  EpContextData(EpContextData&& other) noexcept { MoveFrom(other); }
  EpContextData& operator=(EpContextData&& other) noexcept {
    if (this != &other) {
      Reset();
      MoveFrom(other);
    }
    return *this;
  }

  EpContextData(const EpContextData&) = delete;
  EpContextData& operator=(const EpContextData&) = delete;

  // Pointer to the bytes, owned by this object (valid until it is destroyed or reassigned). May be null only when
  // size() == 0.
  const char* data() const noexcept { return static_cast<const char*>(buffer_); }
  size_t size() const noexcept { return buffer_size_; }
  bool empty() const noexcept { return buffer_size_ == 0; }

  // Frees any owned bytes and returns to the empty state. The read entry points call this before doing any work, so
  // an EpContextData is safe to reuse across reads; callers may also use it to release a large buffer early.
  void Reset() noexcept { FreeAllocatorBuffer(); }

 private:
  friend OrtStatus* ReadEpContextData(const OrtApi& api, OrtReadNamedBufferFunc read_func, void* read_state,
                                      const char* file_name, const OrtGraph* graph, EpContextData& out,
                                      OrtAllocator* allocator);

  void FreeAllocatorBuffer() noexcept {
    if (buffer_ != nullptr && allocator_ != nullptr && api_ != nullptr) {
      // Best-effort free; release any returned status without throwing, since this function is noexcept. The default
      // allocator is owned by ORT and must not be released here.
      Ort::Status free_status{api_->AllocatorFree(allocator_, buffer_)};
      static_cast<void>(free_status);
    }
    api_ = nullptr;
    allocator_ = nullptr;
    buffer_ = nullptr;
    buffer_size_ = 0;
  }

  // Takes ownership of `buffer` (allocated from `allocator`, freed via `api`), replacing anything already held.
  // Used by the read entry point to adopt the callback buffer and the file-fallback buffer through one path.
  void Adopt(const OrtApi& api, OrtAllocator* allocator, void* buffer, size_t buffer_size) noexcept {
    FreeAllocatorBuffer();
    api_ = &api;
    allocator_ = allocator;
    buffer_ = buffer;
    buffer_size_ = buffer_size;
  }

  void MoveFrom(EpContextData& other) noexcept {
    api_ = other.api_;
    allocator_ = other.allocator_;
    buffer_ = other.buffer_;
    buffer_size_ = other.buffer_size_;
    other.api_ = nullptr;
    other.allocator_ = nullptr;
    other.buffer_ = nullptr;
    other.buffer_size_ = 0;
  }

  const OrtApi* api_ = nullptr;        // Frees buffer_ via allocator_; null when there is no owned buffer.
  OrtAllocator* allocator_ = nullptr;  // Allocator that owns buffer_ (callback or file path); null when none.
  void* buffer_ = nullptr;             // Owned bytes (callback buffer or allocator-backed file read); null when empty.
  size_t buffer_size_ = 0;             // Size of buffer_ in bytes.
};

// Zero-copy read: reads EPContext binary data named `file_name` into `out` (reset first). If `read_func` is non-null
// it is invoked and the buffer it allocates is adopted by `out` (no copy); otherwise the data is read from the file
// fallback into a buffer allocated the same way. `allocator` is used for the output buffer on BOTH paths (the
// callback path hands it to the callback; the file path allocates the read buffer from it); pass nullptr to use ORT's
// default allocator. Whatever allocator allocates the buffer is stored in `out` and used for the matching free, so a
// caller may supply its own (e.g. arena/pinned) allocator and it will be honored consistently.
//
// Allocator ownership: this function does NOT take ownership of `allocator`. The caller must keep the OrtAllocator
// alive at least as long as the resulting `out`, because EpContextData frees its buffer via that allocator on
// destruction. Callers using the C++ API can make this borrowing relationship explicit with a non-owning wrapper such
// as Ort::UnownedAllocator (passing its underlying OrtAllocator* here).
//
// See ReadEpContextDataFromFile() for how `graph` governs name resolution. This low-level overload takes the callback
// directly so tests can inject one; production EPs use the OrtEpContextConfig overload.
inline OrtStatus* ReadEpContextData(const OrtApi& api, OrtReadNamedBufferFunc read_func, void* read_state,
                                    const char* file_name, const OrtGraph* graph, EpContextData& out,
                                    OrtAllocator* allocator) {
  out.Reset();

  if (file_name == nullptr || file_name[0] == '\0') {
    return api.CreateStatus(ORT_INVALID_ARGUMENT, "EPContext data file name must not be empty");
  }

  // Use the caller-provided allocator if any; otherwise ORT's default allocator. Whatever allocates the output buffer
  // is also what frees it (stored in `out` for the matching free), so a caller-supplied allocator is honored on both
  // the callback and file paths. Prefer the C allocator API over Ort::AllocatorWithDefaultOptions, whose constructor
  // throws on failure, so an allocator error is reported through the OrtStatus* return like every other failure here.
  // This is about the error-reporting style, not a no-throw guarantee: allocation done elsewhere (paths, strings,
  // streams) can still throw, and making the function truly no-throw would need try/catch that is not worth the
  // complexity. The default allocator is owned by ORT and must not be released here.
  OrtAllocator* effective_allocator = allocator;
  if (effective_allocator == nullptr) {
    RETURN_IF_ERROR(api.GetAllocatorWithDefaultOptions(&effective_allocator));
  }

  if (read_func == nullptr) {
    // No callback: read the file fallback into a buffer allocated from `effective_allocator` (so a caller-supplied
    // allocator is honored here too), then transfer ownership to `out`. The helper frees its own buffer and leaves
    // the outputs empty on failure, so `out` stays empty (it was reset above).
    void* file_buffer = nullptr;
    size_t file_buffer_size = 0;
    RETURN_IF_ERROR(ReadEpContextDataFromFileWithAllocator(api, file_name, graph, effective_allocator, &file_buffer,
                                                           &file_buffer_size));
    out.Adopt(api, effective_allocator, file_buffer, file_buffer_size);
    return nullptr;
  }

  void* ep_context_data = nullptr;
  size_t ep_context_data_size = 0;
  OrtStatus* status = read_func(read_state, file_name, effective_allocator, &ep_context_data, &ep_context_data_size);

  // Hold any callback-allocated buffer in a local RAII guard so it is freed via the same allocator on every error
  // path below, while `out` stays empty (it was reset above). Ownership is transferred to `out` only on success,
  // matching the reset-first / bytes-on-success contract.
  auto buffer_deleter = [&api, effective_allocator](void* buffer_to_free) {
    if (buffer_to_free != nullptr) {
      // Best-effort free; release any returned status without throwing, keeping errors on the OrtStatus* path.
      Ort::Status free_status{api.AllocatorFree(effective_allocator, buffer_to_free)};
      static_cast<void>(free_status);
    }
  };
  std::unique_ptr<void, decltype(buffer_deleter)> buffer_guard(ep_context_data, buffer_deleter);

  if (status != nullptr) {
    return status;
  }

  if (ep_context_data_size != 0 && ep_context_data == nullptr) {
    return api.CreateStatus(ORT_FAIL, "OrtReadNamedBufferFunc returned a null buffer for non-empty EPContext data");
  }

  // Success: transfer ownership of the callback buffer to `out` (no copy); `out` frees it via the same allocator.
  out.Adopt(api, effective_allocator, buffer_guard.release(), ep_context_data_size);

  return nullptr;
}

/**
 * \brief Read EPContext binary data into an owning, zero-copy EpContextData buffer (recommended read entry point).
 *
 * Reads the EPContext data named `file_name`. If `ep_context_config` carries an application-supplied
 * OrtReadNamedBufferFunc, that callback is invoked and the buffer it allocates is adopted by `out` without an extra
 * copy; otherwise the data is read from the file fallback into an allocator-backed buffer.
 *
 * \param api The OrtApi.
 * \param ep_context_config EPContext config carrying the optional read callback; may be null (uses the file fallback).
 * \param file_name Logical name of the EPContext data: a callback-namespace key, or a file name for the fallback.
 * \param graph Governs name resolution on the FILE-FALLBACK path only. When non-null it is the untrusted EPContext
 *              model graph, and `file_name` must be relative and must resolve inside the model directory; null
 *              denotes a trusted caller supplying a physical path. When a read callback is configured, `graph` is
 *              not consulted at all: `file_name` is forwarded to the callback verbatim as an opaque namespace key,
 *              so the callback must treat it as untrusted input and must not use it as a filesystem path without
 *              applying its own validation.
 * \param out Reset first; receives the bytes on success and is left empty on failure. Access via out.data()/out.size().
 * \param allocator Optional allocator used for the output buffer on both the callback and file paths; null uses ORT's
 *                  default allocator. Not owned: it must outlive `out` (see the low-level overload for details).
 * \return nullptr on success, or an OrtStatus* error owned by the caller.
 */
inline OrtStatus* ReadEpContextData(const OrtApi& api, const OrtEpContextConfig* ep_context_config,
                                    const char* file_name, const OrtGraph* graph, EpContextData& out,
                                    OrtAllocator* allocator = nullptr) {
  // Reset up front so the documented "empty on failure" contract also holds for the early returns below, which are
  // reached before the low-level overload (which does its own reset) is ever called.
  out.Reset();

  OrtReadNamedBufferFunc read_func = nullptr;
  void* read_state = nullptr;
  if (ep_context_config != nullptr) {
    auto get_read_func =
        Ort::Experimental::Get_OrtEpApi_EpContextConfig_GetEpContextDataReadFunc_SinceV28_Fn(&api);
    if (get_read_func == nullptr) {
      return api.CreateStatus(ORT_NOT_IMPLEMENTED,
                              "OrtEpApi_EpContextConfig_GetEpContextDataReadFunc is not available");
    }
    RETURN_IF_ERROR(get_read_func(ep_context_config, &read_func, &read_state));
  }
  return ReadEpContextData(api, read_func, read_state, file_name, graph, out, allocator);
}

// Low-level overload that takes the write callback and its opaque state directly. Production EPs should use the
// overloads below that take an OrtEpContextConfig; this overload exists so unit tests can inject a callback without
// constructing an OrtEpContextConfig. When `write_func` is null the data is written to the file fallback.
inline OrtStatus* WriteEpContextDataWithFileFallback(
    const OrtApi& api,
    OrtWriteNamedBufferFunc write_func, void* write_state,
    const char* file_name, const char* fallback_file_name,
    const OrtGraph* graph,
    const void* buffer, size_t buffer_size) {
  if (file_name == nullptr || file_name[0] == '\0') {
    return api.CreateStatus(ORT_INVALID_ARGUMENT, "EPContext data file name must not be empty");
  }

  if (buffer == nullptr && buffer_size != 0) {
    return api.CreateStatus(ORT_INVALID_ARGUMENT, "EPContext data buffer must not be null for non-empty data");
  }

  // `file_name` is a logical name in the write callback's own namespace, so it is passed through unmodified and is
  // never validated as a filesystem path. Only `fallback_file_name` below is mapped onto the filesystem.
  if (write_func != nullptr) {
    return write_func(write_state, file_name, buffer, buffer_size);
  }

  if (fallback_file_name == nullptr || fallback_file_name[0] == '\0') {
    return api.CreateStatus(ORT_INVALID_ARGUMENT, "EPContext data fallback file name must not be empty");
  }

  std::filesystem::path data_path;
  RETURN_IF_ERROR(ResolveEpContextDataPath(api, fallback_file_name, graph, data_path));
  return WriteEpContextDataToResolvedFile(api, data_path, buffer, buffer_size);
}

/**
 * \brief Write EPContext binary data via an application callback or the file fallback (recommended write entry point).
 *
 * If `ep_context_config` carries an application-supplied OrtWriteNamedBufferFunc, it is invoked and `file_name` is
 * passed through unmodified as the logical name. Otherwise the data is written to a file at `fallback_file_name`.
 *
 * \param api The OrtApi.
 * \param ep_context_config EPContext config carrying the optional write callback; may be null (uses the file fallback).
 * \param file_name Logical name written into the model / passed to the callback.
 * \param fallback_file_name File path used only when no write callback is configured; resolved against the model
 *                           directory when `graph` is non-null (and rejected if absolute or rooted in that case).
 * \param graph When non-null, resolves `fallback_file_name` against the model directory. `graph == nullptr` denotes a
 *              trusted caller that may supply an absolute physical path.
 * \param buffer The bytes to write; may be null only when `buffer_size` is 0.
 * \param buffer_size Number of bytes to write.
 * \return nullptr on success, or an OrtStatus* error owned by the caller.
 */
inline OrtStatus* WriteEpContextDataWithFileFallback(
    const OrtApi& api,
    const OrtEpContextConfig* ep_context_config,
    const char* file_name, const char* fallback_file_name,
    const OrtGraph* graph,
    const void* buffer, size_t buffer_size) {
  OrtWriteNamedBufferFunc write_func = nullptr;
  void* write_state = nullptr;
  if (ep_context_config != nullptr) {
    auto get_write_func =
        Ort::Experimental::Get_OrtEpApi_EpContextConfig_GetEpContextDataWriteFunc_SinceV28_Fn(&api);
    if (get_write_func == nullptr) {
      return api.CreateStatus(ORT_NOT_IMPLEMENTED,
                              "OrtEpApi_EpContextConfig_GetEpContextDataWriteFunc is not available");
    }
    RETURN_IF_ERROR(get_write_func(ep_context_config, &write_func, &write_state));
  }
  return WriteEpContextDataWithFileFallback(api, write_func, write_state, file_name, fallback_file_name, graph, buffer,
                                            buffer_size);
}

/**
 * \brief Convenience write overload that uses `file_name` as both the logical callback name and the file-fallback path.
 *
 * Because `file_name` doubles as the fallback path, when no callback is configured it is resolved by
 * ResolveEpContextDataPath(): with `graph` non-null it must be relative and must stay within the model directory;
 * with `graph == nullptr` a trusted caller may supply an absolute physical path. To use a different physical target
 * than the logical name, use the overload above that takes a separate `fallback_file_name`.
 *
 * \param api The OrtApi.
 * \param ep_context_config EPContext config carrying the optional write callback; may be null (uses the file fallback).
 * \param file_name Logical name and, on the file-fallback path, the file name to write.
 * \param graph When non-null, resolves `file_name` against the model directory; null denotes a trusted caller.
 * \param buffer The bytes to write; may be null only when `buffer_size` is 0.
 * \param buffer_size Number of bytes to write.
 * \return nullptr on success, or an OrtStatus* error owned by the caller.
 */
inline OrtStatus* WriteEpContextDataWithFileFallback(
    const OrtApi& api,
    const OrtEpContextConfig* ep_context_config,
    const char* file_name, const OrtGraph* graph,
    const void* buffer, size_t buffer_size) {
  return WriteEpContextDataWithFileFallback(api, ep_context_config, file_name, file_name, graph, buffer, buffer_size);
}

}  // namespace ep_context_data_utils

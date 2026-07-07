// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if USE_FPA_INTB_GEMM

#include "contrib_ops/cuda/llm/gemm_tactic_cache.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <utility>

#include "core/platform/env_var_utils.h"
#include "onnxruntime_config.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace onnxruntime::llm::gemm_cache {

namespace {

using onnxruntime::llm::cutlass_extensions::ClusterShape;
using onnxruntime::llm::cutlass_extensions::CutlassTileConfig;
using onnxruntime::llm::cutlass_extensions::CutlassTileConfigSM100;
using onnxruntime::llm::cutlass_extensions::CutlassTileConfigSM120;
using onnxruntime::llm::cutlass_extensions::CutlassTileConfigSM90;
using onnxruntime::llm::cutlass_extensions::EpilogueScheduleType;
using onnxruntime::llm::cutlass_extensions::MainloopScheduleType;
using onnxruntime::llm::cutlass_extensions::SplitKStyle;

std::vector<std::string> SplitTabs(const std::string& line) {
  std::vector<std::string> out;
  std::string field;
  std::istringstream ss(line);
  while (std::getline(ss, field, '\t')) {
    out.push_back(field);
  }
  // std::getline drops a trailing empty field; recover it so column counts stay stable.
  if (!line.empty() && line.back() == '\t') {
    out.emplace_back();
  }
  return out;
}

std::string JoinTabs(const std::vector<std::string>& fields) {
  std::string out;
  for (size_t i = 0; i < fields.size(); ++i) {
    if (i != 0) {
      out.push_back('\t');
    }
    out += fields[i];
  }
  return out;
}

bool ParseInt(const std::string& s, int& value) {
  if (s.empty()) {
    return false;
  }
  errno = 0;
  char* end = nullptr;
  long parsed = std::strtol(s.c_str(), &end, 10);
  if (end == s.c_str() || *end != '\0' || errno != 0) {
    return false;
  }
  value = static_cast<int>(parsed);
  return true;
}

// Extracts the "git-commit-id=..." token from ORT_BUILD_INFO, if present.
std::string ExtractGitCommit(const std::string& build_info) {
  const std::string key = "git-commit-id=";
  auto pos = build_info.find(key);
  if (pos == std::string::npos) {
    return "unknown";
  }
  pos += key.size();
  auto end = build_info.find_first_of(", ", pos);
  if (end == std::string::npos) {
    end = build_info.size();
  }
  std::string commit = build_info.substr(pos, end - pos);
  return commit.empty() ? "unknown" : commit;
}

std::string CurrentOrtVersion() {
#ifdef ORT_VERSION
  return std::string(ORT_VERSION);
#else
  return "unknown";
#endif
}

std::string CurrentOrtBuildInfo() {
#ifdef ORT_BUILD_INFO
  return std::string(ORT_BUILD_INFO);
#else
  return std::string();
#endif
}

std::string CurrentBuildConfig() {
#ifdef NDEBUG
  return "Release";
#else
  return "Debug";
#endif
}

// RAII cross-process advisory lock on "<path>.lock".
class ScopedFileLock {
 public:
  explicit ScopedFileLock(const std::string& path) : lock_path_(path + ".lock") {
#if defined(_WIN32)
    handle_ = ::CreateFileA(lock_path_.c_str(), GENERIC_READ | GENERIC_WRITE,
                            FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                            OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle_ != INVALID_HANDLE_VALUE) {
      OVERLAPPED ov{};
      locked_ = ::LockFileEx(handle_, LOCKFILE_EXCLUSIVE_LOCK, 0, MAXDWORD, MAXDWORD, &ov) != 0;
    }
#else
    fd_ = ::open(lock_path_.c_str(), O_CREAT | O_RDWR, 0644);
    if (fd_ != -1) {
      locked_ = ::flock(fd_, LOCK_EX) == 0;
    }
#endif
  }

  ~ScopedFileLock() {
#if defined(_WIN32)
    if (handle_ != INVALID_HANDLE_VALUE) {
      if (locked_) {
        OVERLAPPED ov{};
        ::UnlockFileEx(handle_, 0, MAXDWORD, MAXDWORD, &ov);
      }
      ::CloseHandle(handle_);
    }
#else
    if (fd_ != -1) {
      if (locked_) {
        ::flock(fd_, LOCK_UN);
      }
      ::close(fd_);
    }
#endif
  }

  bool locked() const { return locked_; }

  ScopedFileLock(const ScopedFileLock&) = delete;
  ScopedFileLock& operator=(const ScopedFileLock&) = delete;

 private:
  std::string lock_path_;
  bool locked_ = false;
#if defined(_WIN32)
  HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
  int fd_ = -1;
#endif
};

}  // namespace

std::string TsvEncode(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    switch (c) {
      case '%':
        out += "%25";
        break;
      case '\t':
        out += "%09";
        break;
      case '\n':
        out += "%0A";
        break;
      case '\r':
        out += "%0D";
        break;
      default:
        out.push_back(c);
        break;
    }
  }
  return out;
}

std::string TsvDecode(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '%' && i + 2 < s.size()) {
      auto hex = s.substr(i + 1, 2);
      char* end = nullptr;
      long code = std::strtol(hex.c_str(), &end, 16);
      if (end == hex.c_str() + 2) {
        out.push_back(static_cast<char>(code));
        i += 2;
        continue;
      }
    }
    out.push_back(s[i]);
  }
  return out;
}

HardwareSignature HardwareSignature::Compute() {
  HardwareSignature sig;

  int device = 0;
  if (cudaGetDevice(&device) == cudaSuccess) {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
      sig.device_name = prop.name;
      sig.sm = prop.major * 10 + prop.minor;
      sig.multiprocessor_count = prop.multiProcessorCount;
    }
  }

  sig.cuda_runtime = CUDART_VERSION;
  int driver = 0;
  if (cudaDriverGetVersion(&driver) == cudaSuccess) {
    sig.cuda_driver = driver;
  }

  sig.ort_version = CurrentOrtVersion();
  sig.ort_git_commit = ExtractGitCommit(CurrentOrtBuildInfo());
  sig.ort_build_config = CurrentBuildConfig();
  return sig;
}

bool HardwareSignature::StrictMatches(const HardwareSignature& other) const {
  return device_name == other.device_name &&
         sm == other.sm &&
         cuda_runtime == other.cuda_runtime &&
         ort_version == other.ort_version &&
         ort_git_commit == other.ort_git_commit &&
         ort_build_config == other.ort_build_config;
}

std::string HardwareSignature::FilePrefixToken() const {
  std::string token = device_name.empty() ? "unknown_gpu" : device_name;
  for (char& c : token) {
    if (!std::isalnum(static_cast<unsigned char>(c))) {
      c = '_';
    }
  }
  token += "_sm" + std::to_string(sm);
  return token;
}

void AppendConfigColumns(std::vector<std::string>& row, const std::optional<CutlassGemmConfig>& config) {
  if (!config.has_value()) {
    // valid_config=0; remaining columns are placeholders (ignored on parse).
    // Order: valid_config sm_version tile80 tile90 tile100 tile120
    //        split_k_style split_k stages cluster mainloop epilogue tma enable_cuda_kernel
    const char* placeholders[kNumConfigColumns] = {
        "0", "0", "0", "0", "0", "0", "0", "-1", "-1", "0", "0", "0", "0", "0"};
    for (const char* p : placeholders) {
      row.emplace_back(p);
    }
    return;
  }

  const CutlassGemmConfig& c = *config;
  row.emplace_back("1");                                                    // valid_config
  row.emplace_back(std::to_string(c.sm_version));                           // sm_version
  row.emplace_back(std::to_string(static_cast<int>(c.tile_config_sm80)));   // tile80
  row.emplace_back(std::to_string(static_cast<int>(c.tile_config_sm90)));   // tile90
  row.emplace_back(std::to_string(static_cast<int>(c.tile_config_sm100)));  // tile100
  row.emplace_back(std::to_string(static_cast<int>(c.tile_config_sm120)));  // tile120
  row.emplace_back(std::to_string(static_cast<int>(c.split_k_style)));      // split_k_style
  row.emplace_back(std::to_string(c.split_k_factor));                       // split_k
  row.emplace_back(std::to_string(c.stages));                               // stages
  row.emplace_back(std::to_string(static_cast<int>(c.cluster_shape)));      // cluster
  row.emplace_back(std::to_string(static_cast<int>(c.mainloop_schedule)));  // mainloop
  row.emplace_back(std::to_string(static_cast<int>(c.epilogue_schedule)));  // epilogue
  row.emplace_back(c.is_tma_warp_specialized ? "1" : "0");                  // tma
  row.emplace_back(c.enableCudaKernel ? "1" : "0");                         // enable_cuda_kernel
}

std::optional<std::optional<CutlassGemmConfig>> ParseConfigColumns(
    const std::vector<std::string>& columns, size_t begin) {
  if (begin + kNumConfigColumns > columns.size()) {
    return std::nullopt;
  }

  int vals[kNumConfigColumns];
  for (int i = 0; i < kNumConfigColumns; ++i) {
    if (!ParseInt(columns[begin + i], vals[i])) {
      return std::nullopt;
    }
  }

  const int valid_config = vals[0];
  if (valid_config == 0) {
    // A profiled bucket with no valid tactic.
    return std::optional<CutlassGemmConfig>{std::nullopt};
  }

  CutlassGemmConfig c;
  c.sm_version = vals[1];
  c.tile_config_sm80 = static_cast<CutlassTileConfig>(vals[2]);
  c.tile_config_sm90 = static_cast<CutlassTileConfigSM90>(vals[3]);
  c.tile_config_sm100 = static_cast<CutlassTileConfigSM100>(vals[4]);
  c.tile_config_sm120 = static_cast<CutlassTileConfigSM120>(vals[5]);
  c.split_k_style = static_cast<SplitKStyle>(vals[6]);
  c.split_k_factor = vals[7];
  c.stages = vals[8];
  c.cluster_shape = static_cast<ClusterShape>(vals[9]);
  c.mainloop_schedule = static_cast<MainloopScheduleType>(vals[10]);
  c.epilogue_schedule = static_cast<EpilogueScheduleType>(vals[11]);
  c.is_tma_warp_specialized = vals[12] != 0;
  c.enableCudaKernel = vals[13] != 0;
  return std::optional<CutlassGemmConfig>{c};
}

bool MatMulNBitsKey::operator==(const MatMulNBitsKey& o) const {
  return n_16b == o.n_16b && k == o.k &&
         activation_dtype == o.activation_dtype && weight_type == o.weight_type &&
         bits == o.bits && block_size == o.block_size &&
         has_zero_points == o.has_zero_points && zero_point_dtype == o.zero_point_dtype &&
         gemv_enabled == o.gemv_enabled && packing_sm == o.packing_sm;
}

std::size_t MatMulNBitsKeyHash::operator()(const MatMulNBitsKey& k) const {
  std::size_t h = std::hash<int>{}(k.n_16b);
  auto mix = [&h](std::size_t v) {
    h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  };
  mix(std::hash<int>{}(k.k));
  mix(std::hash<std::string>{}(k.activation_dtype));
  mix(std::hash<std::string>{}(k.weight_type));
  mix(std::hash<int>{}(k.bits));
  mix(std::hash<int>{}(k.block_size));
  mix(std::hash<bool>{}(k.has_zero_points));
  mix(std::hash<std::string>{}(k.zero_point_dtype));
  mix(std::hash<bool>{}(k.gemv_enabled));
  mix(std::hash<int>{}(k.packing_sm));
  return h;
}

namespace {

// Column names for the MatMulNBits data rows, in write order.
const std::vector<std::string>& MatMulNBitsColumnNames() {
  static const std::vector<std::string> names = {
      "schema_version", "n_16b", "k", "activation_dtype", "weight_type", "bits",
      "block_size", "has_zero_points", "zero_point_dtype", "gemv_enabled", "packing_sm",
      "m_bucket", "valid_config", "sm_version", "tile80", "tile90", "tile100", "tile120",
      "split_k_style", "split_k", "stages", "cluster", "mainloop", "epilogue", "tma",
      "enable_cuda_kernel"};
  return names;
}

}  // namespace

MatMulNBitsTacticCache::MatMulNBitsTacticCache(std::string file_path, HardwareSignature signature)
    : file_path_(std::move(file_path)), signature_(std::move(signature)) {}

std::shared_ptr<MatMulNBitsTacticCache> MatMulNBitsTacticCache::MaybeCreate(
    const std::string& config_dir, const std::string& config_prefix) {
  // Session-config values take precedence; fall back to the environment variables.
  std::string prefix = config_prefix;
  if (prefix.empty()) {
    prefix = ParseEnvironmentVariableWithDefault<std::string>(kEnvCachePrefix, "");
  }
  std::string dir = config_dir;
  if (dir.empty()) {
    dir = ParseEnvironmentVariableWithDefault<std::string>(kEnvCacheDir, "");
  }

  HardwareSignature signature = HardwareSignature::Compute();

  std::string file_path;
  const std::string suffix = std::string(".") + kTableMatMulNBits + ".tsv";
  if (!prefix.empty()) {
    file_path = prefix + suffix;
  } else if (!dir.empty()) {
    file_path = dir + "/" + signature.FilePrefixToken() + suffix;
  } else {
    // Persistence disabled: keep today's in-process-only behavior.
    return nullptr;
  }

  auto cache = std::make_shared<MatMulNBitsTacticCache>(file_path, std::move(signature));
  auto status = cache->Load();
  ORT_UNUSED_PARAMETER(status);  // A missing/mismatched file simply yields an empty cache.
  return cache;
}

std::optional<std::optional<CutlassGemmConfig>> MatMulNBitsTacticCache::Get(
    const MatMulNBitsKey& key, int m_bucket) const {
  std::lock_guard<std::mutex> guard(mutex_);
  auto it = table_.find(key);
  if (it == table_.end()) {
    return std::nullopt;
  }
  auto bucket_it = it->second.find(m_bucket);
  if (bucket_it == it->second.end()) {
    return std::nullopt;
  }
  return bucket_it->second;
}

MatMulNBitsTacticCache::BucketMap MatMulNBitsTacticCache::GetAll(const MatMulNBitsKey& key) const {
  std::lock_guard<std::mutex> guard(mutex_);
  auto it = table_.find(key);
  if (it == table_.end()) {
    return {};
  }
  return it->second;
}

void MatMulNBitsTacticCache::Put(const MatMulNBitsKey& key, int m_bucket,
                                 const std::optional<CutlassGemmConfig>& config) {
  std::lock_guard<std::mutex> guard(mutex_);
  table_[key][m_bucket] = config;
  dirty_ = true;
}

onnxruntime::common::Status MatMulNBitsTacticCache::Load() {
  std::ifstream in(file_path_);
  if (!in.is_open()) {
    return onnxruntime::common::Status::OK();
  }

  HardwareSignature file_sig;
  file_sig.ort_build_config.clear();
  std::string magic_ok_version;
  std::string table_name;
  std::vector<std::string> header_columns;
  std::unordered_map<MatMulNBitsKey, BucketMap, MatMulNBitsKeyHash> loaded;

  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (line.empty()) {
      continue;
    }

    if (line[0] == '#') {
      auto fields = SplitTabs(line.substr(1));
      // Strip a leading space after '#'.
      if (!fields.empty()) {
        auto& first = fields[0];
        size_t start = first.find_first_not_of(' ');
        first = (start == std::string::npos) ? "" : first.substr(start);
      }
      if (fields.size() >= 2) {
        const std::string& k = fields[0];
        const std::string& v = fields[1];
        if (k == kCacheMagic) {
          magic_ok_version = v;
        } else if (k == "table") {
          table_name = v;
        } else if (k == "device_name") {
          file_sig.device_name = TsvDecode(v);
        } else if (k == "sm") {
          ParseInt(v, file_sig.sm);
        } else if (k == "multiprocessor_count") {
          ParseInt(v, file_sig.multiprocessor_count);
        } else if (k == "cuda_runtime") {
          ParseInt(v, file_sig.cuda_runtime);
        } else if (k == "cuda_driver") {
          ParseInt(v, file_sig.cuda_driver);
        } else if (k == "ort_version") {
          file_sig.ort_version = v;
        } else if (k == "ort_git_commit") {
          file_sig.ort_git_commit = v;
        } else if (k == "ort_build_config") {
          file_sig.ort_build_config = v;
        }
      }
      continue;
    }

    if (header_columns.empty()) {
      header_columns = SplitTabs(line);
      continue;
    }

    // Data row: map by column name so appended columns are tolerated.
    auto values = SplitTabs(line);
    auto field = [&](const char* name) -> std::string {
      for (size_t i = 0; i < header_columns.size() && i < values.size(); ++i) {
        if (header_columns[i] == name) {
          return values[i];
        }
      }
      return std::string();
    };

    MatMulNBitsKey key;
    if (!ParseInt(field("n_16b"), key.n_16b) || !ParseInt(field("k"), key.k) ||
        !ParseInt(field("bits"), key.bits) || !ParseInt(field("block_size"), key.block_size) ||
        !ParseInt(field("packing_sm"), key.packing_sm)) {
      continue;
    }
    key.activation_dtype = TsvDecode(field("activation_dtype"));
    key.weight_type = TsvDecode(field("weight_type"));
    int has_zp = 0;
    ParseInt(field("has_zero_points"), has_zp);
    key.has_zero_points = has_zp != 0;
    key.zero_point_dtype = TsvDecode(field("zero_point_dtype"));
    int gemv = 0;
    ParseInt(field("gemv_enabled"), gemv);
    key.gemv_enabled = gemv != 0;

    int m_bucket = 0;
    if (!ParseInt(field("m_bucket"), m_bucket)) {
      continue;
    }

    // Locate the start of the config columns by name so appended key columns do not shift them.
    size_t valid_config_index = header_columns.size();
    for (size_t i = 0; i < header_columns.size(); ++i) {
      if (header_columns[i] == "valid_config") {
        valid_config_index = i;
        break;
      }
    }
    auto parsed = ParseConfigColumns(values, valid_config_index);
    if (!parsed.has_value()) {
      continue;
    }
    loaded[key][m_bucket] = *parsed;
  }

  // Reject the file if the format, table, or hardware/build signature does not match.
  if (magic_ok_version != kCacheFormatVersion || table_name != kTableMatMulNBits ||
      !signature_.StrictMatches(file_sig)) {
    return onnxruntime::common::Status::OK();
  }

  std::lock_guard<std::mutex> guard(mutex_);
  for (auto& [key, buckets] : loaded) {
    auto& dest = table_[key];
    for (auto& [m, cfg] : buckets) {
      dest.emplace(m, cfg);  // Do not clobber entries already set in-process.
    }
  }
  return onnxruntime::common::Status::OK();
}

onnxruntime::common::Status MatMulNBitsTacticCache::WriteAllLocked(
    const std::unordered_map<MatMulNBitsKey, BucketMap, MatMulNBitsKeyHash>& table) const {
  const std::string tmp_path = file_path_ + ".tmp";
  {
    std::ofstream out(tmp_path, std::ios::trunc);
    if (!out.is_open()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to open gemm tactic cache temp file for writing: ", tmp_path);
    }

    // Signature / format header lines.
    out << "# " << kCacheMagic << '\t' << kCacheFormatVersion << '\n';
    out << "# table\t" << kTableMatMulNBits << '\n';
    out << "# device_name\t" << TsvEncode(signature_.device_name) << '\n';
    out << "# sm\t" << signature_.sm << '\n';
    out << "# multiprocessor_count\t" << signature_.multiprocessor_count << '\n';
    out << "# cuda_runtime\t" << signature_.cuda_runtime << '\n';
    out << "# cuda_driver\t" << signature_.cuda_driver << '\n';
    out << "# ort_version\t" << signature_.ort_version << '\n';
    out << "# ort_git_commit\t" << signature_.ort_git_commit << '\n';
    out << "# ort_build_config\t" << signature_.ort_build_config << '\n';

    // Column header.
    out << JoinTabs(MatMulNBitsColumnNames()) << '\n';

    // Data rows (sorted by bucket within each key for stable, diffable output).
    for (const auto& [key, buckets] : table) {
      for (const auto& [m_bucket, cfg] : buckets) {
        std::vector<std::string> row;
        row.reserve(MatMulNBitsColumnNames().size());
        row.emplace_back(std::to_string(kSchemaVersion));
        row.emplace_back(std::to_string(key.n_16b));
        row.emplace_back(std::to_string(key.k));
        row.emplace_back(TsvEncode(key.activation_dtype));
        row.emplace_back(TsvEncode(key.weight_type));
        row.emplace_back(std::to_string(key.bits));
        row.emplace_back(std::to_string(key.block_size));
        row.emplace_back(key.has_zero_points ? "1" : "0");
        row.emplace_back(TsvEncode(key.zero_point_dtype));
        row.emplace_back(key.gemv_enabled ? "1" : "0");
        row.emplace_back(std::to_string(key.packing_sm));
        row.emplace_back(std::to_string(m_bucket));
        AppendConfigColumns(row, cfg);
        out << JoinTabs(row) << '\n';
      }
    }

    out.flush();
    if (!out.good()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed while writing gemm tactic cache temp file: ", tmp_path);
    }
  }

  if (std::rename(tmp_path.c_str(), file_path_.c_str()) != 0) {
    std::remove(tmp_path.c_str());
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to atomically replace gemm tactic cache file: ", file_path_);
  }
  return onnxruntime::common::Status::OK();
}

onnxruntime::common::Status MatMulNBitsTacticCache::Flush() {
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!dirty_) {
      return onnxruntime::common::Status::OK();
    }
  }

  ScopedFileLock lock(file_path_);
  // Even if the OS lock could not be acquired, proceed best-effort; atomic rename
  // still guarantees a consistent file, only lost updates become possible.

  // Reload the current on-disk file so concurrently-written rows are not dropped,
  // then overlay the in-memory table (in-memory wins on conflict).
  MatMulNBitsTacticCache disk(file_path_, signature_);
  ORT_RETURN_IF_ERROR(disk.Load());

  std::unordered_map<MatMulNBitsKey, BucketMap, MatMulNBitsKeyHash> merged;
  {
    std::lock_guard<std::mutex> disk_guard(disk.mutex_);
    merged = disk.table_;
  }
  {
    std::lock_guard<std::mutex> guard(mutex_);
    for (const auto& [key, buckets] : table_) {
      auto& dest = merged[key];
      for (const auto& [m, cfg] : buckets) {
        dest[m] = cfg;
      }
    }
  }

  ORT_RETURN_IF_ERROR(WriteAllLocked(merged));

  {
    std::lock_guard<std::mutex> guard(mutex_);
    dirty_ = false;
  }
  return onnxruntime::common::Status::OK();
}

}  // namespace onnxruntime::llm::gemm_cache

#endif  // USE_FPA_INTB_GEMM

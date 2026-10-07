// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Unit tests for the fpA_intB / MatMulNBits GEMM tactic cache utilities.
// GemmTacticCacheTest uses synthetic signatures and needs no GPU.
// GemmTacticCacheCudaTest additionally exercises the profiler's CUDA events.
//
// Built into onnxruntime_providers_cuda_ut (onnxruntime_ENABLE_CUDA_EP_INTERNAL_TESTS=ON); run like:
//  ./onnxruntime_provider_test --gtest_filter=CUDA_EP_Unittest.All
// Plugin builds register these directly: --gtest_filter=GemmTacticCache*.*
#if USE_FPA_INTB_GEMM
#include <gtest/gtest.h>

#include <cstdio>
#include <barrier>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#endif

#include "contrib_ops/cuda/llm/gemm_tactic_cache.h"
#include "contrib_ops/cuda/llm/gemm_profiler.h"
#include "test/util/include/scoped_env_vars.h"

namespace onnxruntime {
namespace test {

namespace gc = onnxruntime::llm::gemm_cache;
using onnxruntime::llm::cutlass_extensions::ClusterShape;
using onnxruntime::llm::cutlass_extensions::CutlassGemmConfig;
using onnxruntime::llm::cutlass_extensions::CutlassTileConfig;
using onnxruntime::llm::cutlass_extensions::CutlassTileConfigSM90;
using onnxruntime::llm::cutlass_extensions::MainloopScheduleType;
using onnxruntime::llm::cutlass_extensions::SplitKStyle;

namespace {

gc::HardwareSignature MakeSignature(const std::string& device_name = "TEST GPU 4090", int sm = 80) {
  gc::HardwareSignature sig;
  sig.device_name = device_name;
  sig.sm = sm;
  sig.multiprocessor_count = 108;
  sig.cuda_runtime = 13000;
  sig.cuda_driver = 13000;
  sig.ort_version = "1.28.0";
  sig.ort_git_commit = "deadbeef";
  sig.ort_build_config = "Release";
  return sig;
}

gc::MatMulNBitsKey MakeKey(int n_16b = 3072, int k = 4096) {
  gc::MatMulNBitsKey key;
  key.n_16b = n_16b;
  key.k = k;
  key.activation_dtype = "half";
  key.weight_type = "uint4b_t";
  key.bits = 4;
  key.block_size = 64;
  key.has_zero_points = true;
  key.zero_point_dtype = "uint4b_t";
  key.gemv_enabled = true;
  key.packing_sm = 80;
  return key;
}

CutlassGemmConfig MakeSm80Config() {
  CutlassGemmConfig c;
  c.sm_version = 80;
  c.tile_config_sm80 = CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64;
  c.split_k_style = SplitKStyle::SPLIT_K_SERIAL;
  c.split_k_factor = 4;
  c.stages = 3;
  c.is_tma_warp_specialized = false;
  c.enableCudaKernel = false;
  return c;
}

CutlassGemmConfig MakeSm90Config() {
  CutlassGemmConfig c;
  c.sm_version = 90;
  c.tile_config_sm90 = CutlassTileConfigSM90::CtaShape128x128x128B;
  c.mainloop_schedule = MainloopScheduleType::COOPERATIVE;
  c.cluster_shape = ClusterShape::ClusterShape_2x1x1;
  c.is_tma_warp_specialized = true;
  c.enableCudaKernel = false;
  return c;
}

void ExpectConfigEqual(const CutlassGemmConfig& a, const CutlassGemmConfig& b) {
  EXPECT_EQ(a.sm_version, b.sm_version);
  EXPECT_EQ(static_cast<int>(a.tile_config_sm80), static_cast<int>(b.tile_config_sm80));
  EXPECT_EQ(static_cast<int>(a.tile_config_sm90), static_cast<int>(b.tile_config_sm90));
  EXPECT_EQ(static_cast<int>(a.tile_config_sm100), static_cast<int>(b.tile_config_sm100));
  EXPECT_EQ(static_cast<int>(a.tile_config_sm120), static_cast<int>(b.tile_config_sm120));
  EXPECT_EQ(static_cast<int>(a.split_k_style), static_cast<int>(b.split_k_style));
  EXPECT_EQ(a.split_k_factor, b.split_k_factor);
  EXPECT_EQ(a.stages, b.stages);
  EXPECT_EQ(static_cast<int>(a.cluster_shape), static_cast<int>(b.cluster_shape));
  EXPECT_EQ(static_cast<int>(a.mainloop_schedule), static_cast<int>(b.mainloop_schedule));
  EXPECT_EQ(static_cast<int>(a.epilogue_schedule), static_cast<int>(b.epilogue_schedule));
  EXPECT_EQ(a.is_tma_warp_specialized, b.is_tma_warp_specialized);
  EXPECT_EQ(a.enableCudaKernel, b.enableCudaKernel);
}

// Returns a unique temp file prefix and removes any leftover files from a prior run.
std::string UniqueTempPrefix(const std::string& tag) {
  auto dir = std::filesystem::temp_directory_path();
  std::string prefix = (dir / ("ort_gemm_tactic_cache_test_" + tag)).string();
  std::string file = prefix + ".matmulnbits_fpa_intb.tsv";
  std::error_code ec;
  std::filesystem::remove(file, ec);
  std::filesystem::remove(file + ".lock", ec);
  std::filesystem::remove(file + ".tmp", ec);
  return prefix;
}

void CleanUp(const std::string& file) {
  std::error_code ec;
  std::filesystem::remove(file, ec);
  std::filesystem::remove(file + ".lock", ec);
  std::filesystem::remove(file + ".tmp", ec);
}

}  // namespace

TEST(GemmTacticCacheTest, TsvEncodeDecodeRoundTrip) {
  const std::vector<std::string> samples = {
      "NVIDIA A100-SXM4-80GB",
      "has\ttab",
      "has\nnewline",
      "has%percent",
      "mix\t%\n\rall",
      "",
  };
  for (const auto& s : samples) {
    std::string encoded = gc::TsvEncode(s);
    EXPECT_EQ(encoded.find('\t'), std::string::npos) << "encoded still has tab: " << s;
    EXPECT_EQ(encoded.find('\n'), std::string::npos) << "encoded still has newline: " << s;
    EXPECT_EQ(gc::TsvDecode(encoded), s) << "round-trip failed for: " << s;
  }
}

TEST(GemmTacticCacheTest, TsvDecodeKeepsMalformedEscapes) {
  EXPECT_EQ(gc::TsvDecode("%-1"), "%-1");
  EXPECT_EQ(gc::TsvDecode("% 9"), "% 9");
  EXPECT_EQ(gc::TsvDecode("%4"), "%4");
  EXPECT_EQ(gc::TsvDecode("a%41b"), "aAb");
}

TEST(GemmTacticCacheTest, ResolveFilePathSessionConfig) {
  ScopedEnvironmentVariables scoped_env_vars{
      {{gc::kEnvCachePrefix, "/tmp/env_model"}, {gc::kEnvCacheDir, "/tmp/env_dir"}}};
  const auto sig = MakeSignature("NVIDIA H200", 90);
  const std::string suffix = ".matmulnbits_fpa_intb.tsv";
  EXPECT_EQ(gc::MatMulNBitsTacticCache::ResolveFilePath("/tmp/dir", "", sig),
            "/tmp/dir/NVIDIA_H200_sm90" + suffix);
  EXPECT_EQ(gc::MatMulNBitsTacticCache::ResolveFilePath("", "/tmp/model", sig), "/tmp/model" + suffix);
  // A prefix wins over a directory.
  EXPECT_EQ(gc::MatMulNBitsTacticCache::ResolveFilePath("/tmp/dir", "/tmp/model", sig), "/tmp/model" + suffix);
}

TEST(GemmTacticCacheTest, ResolveFilePathEnvironmentDisabled) {
  const auto sig = MakeSignature();
  ScopedEnvironmentVariables scoped_env_vars{
      {{gc::kEnvCachePrefix, std::nullopt}, {gc::kEnvCacheDir, std::nullopt}}};
  EXPECT_TRUE(gc::MatMulNBitsTacticCache::ResolveFilePath("", "", sig).empty());

  ScopedEnvironmentVariables empty_env_vars{
      {{gc::kEnvCachePrefix, ""}, {gc::kEnvCacheDir, ""}}};
  EXPECT_TRUE(gc::MatMulNBitsTacticCache::ResolveFilePath("", "", sig).empty());
}

TEST(GemmTacticCacheTest, ResolveFilePathEnvironmentFallback) {
  const auto sig = MakeSignature("NVIDIA H200", 90);
  const std::string suffix = ".matmulnbits_fpa_intb.tsv";
  const std::string dir = "/tmp/cache \xE4\xB8\xAD";
  const std::string prefix = "/tmp/model \xE4\xB8\xAD";
  ScopedEnvironmentVariables scoped_env_vars{
      {{gc::kEnvCachePrefix, std::nullopt}, {gc::kEnvCacheDir, dir}}};
  EXPECT_EQ(gc::MatMulNBitsTacticCache::ResolveFilePath("", "", sig),
            dir + "/NVIDIA_H200_sm90" + suffix);

  {
    ScopedEnvironmentVariables prefix_env_var{{{gc::kEnvCachePrefix, prefix}}};
    EXPECT_EQ(gc::MatMulNBitsTacticCache::ResolveFilePath("", "", sig), prefix + suffix);
  }

  {
    ScopedEnvironmentVariables prefix_only_env_vars{
        {{gc::kEnvCachePrefix, prefix}, {gc::kEnvCacheDir, std::nullopt}}};
    EXPECT_EQ(gc::MatMulNBitsTacticCache::ResolveFilePath("", "", sig), prefix + suffix);
  }
}

TEST(GemmTacticCacheTest, ConfigColumnsRoundTripSm80) {
  const CutlassGemmConfig original = MakeSm80Config();
  std::vector<std::string> row;
  gc::AppendConfigColumns(row, original);
  ASSERT_EQ(row.size(), static_cast<size_t>(gc::kNumConfigColumns));

  auto parsed = gc::ParseConfigColumns(row, 0);
  ASSERT_TRUE(parsed.has_value());
  ASSERT_TRUE(parsed->has_value());
  ExpectConfigEqual(original, **parsed);
}

TEST(GemmTacticCacheTest, ConfigColumnsRoundTripSm90) {
  const CutlassGemmConfig original = MakeSm90Config();
  std::vector<std::string> row;
  gc::AppendConfigColumns(row, original);
  auto parsed = gc::ParseConfigColumns(row, 0);
  ASSERT_TRUE(parsed.has_value());
  ASSERT_TRUE(parsed->has_value());
  ExpectConfigEqual(original, **parsed);
}

TEST(GemmTacticCacheTest, ConfigColumnsNullTacticRoundTrip) {
  std::vector<std::string> row;
  gc::AppendConfigColumns(row, std::nullopt);  // profiled bucket with no valid tactic
  auto parsed = gc::ParseConfigColumns(row, 0);
  ASSERT_TRUE(parsed.has_value());    // outer: the columns parsed
  EXPECT_FALSE(parsed->has_value());  // inner: no valid tactic
}

TEST(GemmTacticCacheTest, StoreLoadRoundTrip) {
  const std::string prefix = UniqueTempPrefix("roundtrip");
  const std::string file = prefix + ".matmulnbits_fpa_intb.tsv";
  const gc::HardwareSignature sig = MakeSignature();
  const gc::MatMulNBitsKey key = MakeKey();

  {
    gc::MatMulNBitsTacticCache cache(file, sig);
    cache.Put(key, 1, MakeSm80Config());
    cache.Put(key, 64, MakeSm90Config());
    cache.Put(key, 128, std::nullopt);  // negative result
    ASSERT_TRUE(cache.Flush().IsOK());
  }

  gc::MatMulNBitsTacticCache reloaded(file, sig);
  ASSERT_TRUE(reloaded.Load().IsOK());

  auto c1 = reloaded.Get(key, 1);
  ASSERT_TRUE(c1.has_value());
  ASSERT_TRUE(c1->has_value());
  ExpectConfigEqual(MakeSm80Config(), **c1);

  auto c64 = reloaded.Get(key, 64);
  ASSERT_TRUE(c64.has_value());
  ASSERT_TRUE(c64->has_value());
  ExpectConfigEqual(MakeSm90Config(), **c64);

  auto c128 = reloaded.Get(key, 128);
  EXPECT_FALSE(c128.has_value());  // Temporary profiling failures are never persisted.

  EXPECT_FALSE(reloaded.Get(key, 999).has_value());  // never profiled

  CleanUp(file);
}

TEST(GemmTacticCacheTest, SignatureMismatchRejected) {
  const std::string prefix = UniqueTempPrefix("sigmismatch");
  const std::string file = prefix + ".matmulnbits_fpa_intb.tsv";
  const gc::MatMulNBitsKey key = MakeKey();

  {
    gc::MatMulNBitsTacticCache cache(file, MakeSignature("NVIDIA RTX 4090", 89));
    cache.Put(key, 1, MakeSm80Config());
    ASSERT_TRUE(cache.Flush().IsOK());
  }

  // Different device name -> file must be rejected on load.
  gc::MatMulNBitsTacticCache other(file, MakeSignature("NVIDIA RTX 4060", 89));
  ASSERT_TRUE(other.Load().IsOK());
  EXPECT_FALSE(other.Get(key, 1).has_value());

  // Same signature -> accepted.
  gc::MatMulNBitsTacticCache same(file, MakeSignature("NVIDIA RTX 4090", 89));
  ASSERT_TRUE(same.Load().IsOK());
  EXPECT_TRUE(same.Get(key, 1).has_value());

  CleanUp(file);
}

TEST(GemmTacticCacheTest, GitCommitAndBuildConfigNotStrict) {
  const std::string prefix = UniqueTempPrefix("nonstrict");
  const std::string file = prefix + ".matmulnbits_fpa_intb.tsv";
  const gc::MatMulNBitsKey key = MakeKey();

  // Write with one git commit / build config.
  gc::HardwareSignature writer = MakeSignature();
  writer.ort_git_commit = "commit_aaaa";
  writer.ort_build_config = "Release";
  {
    gc::MatMulNBitsTacticCache cache(file, writer);
    cache.Put(key, 1, MakeSm80Config());
    ASSERT_TRUE(cache.Flush().IsOK());
  }

  // A reader with a DIFFERENT git commit and build config, but the same device/sm/cuda_runtime/
  // ort_version, must still accept the cache: those two fields are diagnostic-only.
  gc::HardwareSignature reader = MakeSignature();
  reader.ort_git_commit = "commit_bbbb";
  reader.ort_build_config = "Debug";
  gc::MatMulNBitsTacticCache reloaded(file, reader);
  ASSERT_TRUE(reloaded.Load().IsOK());
  EXPECT_TRUE(reloaded.Get(key, 1).has_value());

  // A different ort_version is still rejected (it remains part of the strict guard).
  gc::HardwareSignature other_version = MakeSignature();
  other_version.ort_version = "9.9.9";
  gc::MatMulNBitsTacticCache rejected(file, other_version);
  ASSERT_TRUE(rejected.Load().IsOK());
  EXPECT_FALSE(rejected.Get(key, 1).has_value());

  CleanUp(file);
}

TEST(GemmTacticCacheTest, AppendedColumnTolerated) {
  const std::string prefix = UniqueTempPrefix("appendcol");
  const std::string file = prefix + ".matmulnbits_fpa_intb.tsv";
  const gc::HardwareSignature sig = MakeSignature();
  const gc::MatMulNBitsKey key = MakeKey();

  {
    gc::MatMulNBitsTacticCache cache(file, sig);
    cache.Put(key, 8, MakeSm80Config());
    ASSERT_TRUE(cache.Flush().IsOK());
  }

  // Simulate a future writer that appended an extra trailing column to the header and rows.
  std::vector<std::string> lines;
  {
    std::ifstream in(file);
    std::string line;
    bool header_seen = false;
    while (std::getline(in, line)) {
      if (!line.empty() && line[0] != '#') {
        line += (header_seen ? "\t999" : "\tfuture_col");
        header_seen = true;
      }
      lines.push_back(line);
    }
  }
  {
    std::ofstream out(file, std::ios::trunc);
    for (const auto& l : lines) {
      out << l << '\n';
    }
  }

  // Reader must map by column name and ignore the unknown appended column.
  gc::MatMulNBitsTacticCache reloaded(file, sig);
  ASSERT_TRUE(reloaded.Load().IsOK());
  auto c8 = reloaded.Get(key, 8);
  ASSERT_TRUE(c8.has_value());
  ASSERT_TRUE(c8->has_value());
  ExpectConfigEqual(MakeSm80Config(), **c8);

  CleanUp(file);
}

TEST(GemmTacticCacheTest, FlushMergesSequentialRows) {
  const std::string prefix = UniqueTempPrefix("merge");
  const std::string file = prefix + ".matmulnbits_fpa_intb.tsv";
  const gc::HardwareSignature sig = MakeSignature();
  const gc::MatMulNBitsKey key = MakeKey();

  // First writer persists bucket 1.
  {
    gc::MatMulNBitsTacticCache a(file, sig);
    a.Put(key, 1, MakeSm80Config());
    ASSERT_TRUE(a.Flush().IsOK());
  }

  // Second writer (independent instance) persists bucket 64; Flush must merge, not clobber.
  {
    gc::MatMulNBitsTacticCache b(file, sig);
    b.Put(key, 64, MakeSm90Config());
    ASSERT_TRUE(b.Flush().IsOK());
  }

  gc::MatMulNBitsTacticCache reloaded(file, sig);
  ASSERT_TRUE(reloaded.Load().IsOK());
  EXPECT_TRUE(reloaded.Get(key, 1).has_value());
  EXPECT_TRUE(reloaded.Get(key, 64).has_value());

  CleanUp(file);
}

TEST(GemmTacticCacheTest, SelectionVersionMismatchRejected) {
  const std::string prefix = UniqueTempPrefix("selversion");
  const std::string file = prefix + ".matmulnbits_fpa_intb.tsv";
  const gc::HardwareSignature sig = MakeSignature();
  const gc::MatMulNBitsKey key = MakeKey();
  {
    gc::MatMulNBitsTacticCache cache(file, sig);
    cache.Put(key, 1, MakeSm80Config());
    ASSERT_TRUE(cache.Flush().IsOK());
  }

  std::vector<std::string> lines;
  {
    std::ifstream in(file);
    std::string line;
    while (std::getline(in, line)) {
      if (line.rfind("# tactic_selection_version\t", 0) == 0) {
        line = "# tactic_selection_version\tstale";
      }
      lines.push_back(line);
    }
  }
  {
    std::ofstream out(file, std::ios::trunc);
    for (const auto& l : lines) {
      out << l << '\n';
    }
  }

  gc::MatMulNBitsTacticCache reloaded(file, sig);
  ASSERT_TRUE(reloaded.Load().IsOK());
  EXPECT_FALSE(reloaded.Get(key, 1).has_value());

  CleanUp(file);
}

TEST(GemmTacticCacheTest, FlushCreatesMissingDirectory) {
  const auto dir = std::filesystem::temp_directory_path() / "ort_gemm_tactic_cache_test_newdir";
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
  const std::string file = (dir / "nested" / "cache.matmulnbits_fpa_intb.tsv").string();
  const gc::HardwareSignature sig = MakeSignature();
  const gc::MatMulNBitsKey key = MakeKey();
  {
    gc::MatMulNBitsTacticCache cache(file, sig);
    cache.Put(key, 1, MakeSm80Config());
    ASSERT_TRUE(cache.Flush().IsOK());
  }

  gc::MatMulNBitsTacticCache reloaded(file, sig);
  ASSERT_TRUE(reloaded.Load().IsOK());
  EXPECT_TRUE(reloaded.Get(key, 1).has_value());

  // Only the cache and its lock file remain; per-writer temp files are renamed away.
  size_t entries = 0;
  for (const auto& entry : std::filesystem::directory_iterator(dir / "nested")) {
    static_cast<void>(entry);
    ++entries;
  }
  EXPECT_EQ(entries, 2u);
  std::filesystem::remove_all(dir, ec);
}

// The process-global profile map must not share tactics across variants that the persistent key separates.
TEST(GemmTacticCacheTest, GemmIdCoreSeparatesQuantVariants) {
  using onnxruntime::llm::kernels::weight_only::GemmIdCore;
  using onnxruntime::llm::kernels::weight_only::GemmIdCoreHash;
  const auto dtype = onnxruntime::llm::nvinfer::DataType::kHALF;
  const GemmIdCore base(1024, 4096, dtype, 80, 4, 64, false, true);

  EXPECT_EQ(base, GemmIdCore(1024, 4096, dtype, 80, 4, 64, false, true));
  EXPECT_EQ(GemmIdCoreHash{}(base), GemmIdCoreHash{}(GemmIdCore(1024, 4096, dtype, 80, 4, 64, false, true)));
  EXPECT_FALSE(base == GemmIdCore(1024, 4096, dtype, 80, 8, 64, false, true));
  EXPECT_FALSE(base == GemmIdCore(1024, 4096, dtype, 80, 4, 128, false, true));
  EXPECT_FALSE(base == GemmIdCore(1024, 4096, dtype, 80, 4, 64, true, true));
  EXPECT_FALSE(base == GemmIdCore(1024, 4096, dtype, 80, 4, 64, false, false));
  EXPECT_FALSE(base == GemmIdCore(1024, 4096, dtype, 80, 4, 64, false, true, true));
  EXPECT_FALSE(base == GemmIdCore(1024, 4096, dtype, 80, 4, 64, false, true, false, "NVIDIA RTX 4090"));

  // Same packing SM, different GPU: the process-global map must not mix their tactics.
  const GemmIdCore rtx4090(1024, 4096, dtype, 80, 4, 64, false, true, false, "NVIDIA RTX 4090");
  const GemmIdCore rtx4060(1024, 4096, dtype, 80, 4, 64, false, true, false, "NVIDIA RTX 4060");
  EXPECT_FALSE(rtx4090 == rtx4060);
  EXPECT_EQ(rtx4090, GemmIdCore(1024, 4096, dtype, 80, 4, 64, false, true, false, "NVIDIA RTX 4090"));
}

TEST(GemmTacticCacheTest, BiasIsPartOfPersistentKey) {
  const std::string prefix = UniqueTempPrefix("bias");
  const std::string file = prefix + ".matmulnbits_fpa_intb.tsv";
  const gc::HardwareSignature sig = MakeSignature();
  gc::MatMulNBitsKey no_bias = MakeKey();
  gc::MatMulNBitsKey with_bias = MakeKey();
  with_bias.has_bias = true;

  {
    gc::MatMulNBitsTacticCache cache(file, sig);
    cache.Put(no_bias, 1, MakeSm80Config());
    cache.Put(with_bias, 1, MakeSm90Config());
    ASSERT_TRUE(cache.Flush().IsOK());
  }

  gc::MatMulNBitsTacticCache reloaded(file, sig);
  ASSERT_TRUE(reloaded.Load().IsOK());
  auto biasless = reloaded.Get(no_bias, 1);
  auto biasful = reloaded.Get(with_bias, 1);
  ASSERT_TRUE(biasless.has_value() && biasless->has_value());
  ASSERT_TRUE(biasful.has_value() && biasful->has_value());
  ExpectConfigEqual(MakeSm80Config(), **biasless);
  ExpectConfigEqual(MakeSm90Config(), **biasful);

  CleanUp(file);
}

TEST(GemmTacticCacheTest, FlushFailsWhenFileLockUnavailable) {
  const std::string prefix = UniqueTempPrefix("lockfail");
  const std::string file = prefix + ".matmulnbits_fpa_intb.tsv";
  const gc::HardwareSignature sig = MakeSignature();

  // A directory at the lock path makes lock acquisition fail on every platform.
  std::error_code ec;
  ASSERT_TRUE(std::filesystem::create_directory(file + ".lock", ec)) << ec.message();

  gc::MatMulNBitsTacticCache cache(file, sig);
  cache.Put(MakeKey(), 1, MakeSm80Config());
  EXPECT_FALSE(cache.Flush().IsOK());
  EXPECT_FALSE(std::filesystem::exists(file));

  CleanUp(file);
}

TEST(GemmTacticCacheTest, StaleReaderDoesNotOverwriteRetunedBucket) {
  const auto file = UniqueTempPrefix("stale") + ".matmulnbits_fpa_intb.tsv";
  const auto key = MakeKey();
  gc::MatMulNBitsTacticCache first(file, MakeSignature());
  first.Put(key, 1, MakeSm80Config());
  ASSERT_TRUE(first.Flush().IsOK());
  gc::MatMulNBitsTacticCache stale(file, MakeSignature());
  ASSERT_TRUE(stale.Load().IsOK());
  first.Put(key, 1, MakeSm90Config());
  ASSERT_TRUE(first.Flush().IsOK());
  stale.Put(key, 64, MakeSm80Config());
  ASSERT_TRUE(stale.Flush().IsOK());
  gc::MatMulNBitsTacticCache result(file, MakeSignature());
  ASSERT_TRUE(result.Load().IsOK());
  ASSERT_TRUE(result.Get(key, 1).has_value());
  ExpectConfigEqual(**result.Get(key, 1), MakeSm90Config());
  EXPECT_TRUE(result.Get(key, 64).has_value());
  CleanUp(file);
}

TEST(GemmTacticCacheTest, ConcurrentWritersPreserveAllBuckets) {
  const auto file = UniqueTempPrefix("concurrent") + ".matmulnbits_fpa_intb.tsv";
  constexpr int writers = 8;
  std::barrier start(writers);
  std::vector<std::thread> threads;
  for (int i = 1; i <= writers; ++i) {
    threads.emplace_back([&, i] {
      gc::MatMulNBitsTacticCache cache(file, MakeSignature());
      cache.Put(MakeKey(), i, MakeSm80Config());
      start.arrive_and_wait();
      EXPECT_TRUE(cache.Flush().IsOK());
    });
  }
  for (auto& thread : threads) thread.join();
  gc::MatMulNBitsTacticCache result(file, MakeSignature());
  ASSERT_TRUE(result.Load().IsOK());
  EXPECT_EQ(result.GetAll(MakeKey()).size(), writers);
  CleanUp(file);
}

TEST(GemmTacticCacheTest, ProfilingFailureIsRetriedAfterReload) {
  const auto file = UniqueTempPrefix("retry") + ".matmulnbits_fpa_intb.tsv";
  gc::MatMulNBitsTacticCache failed(file, MakeSignature());
  failed.Put(MakeKey(), 1, std::nullopt);
  ASSERT_TRUE(failed.Flush().IsOK());
  gc::MatMulNBitsTacticCache healthy(file, MakeSignature());
  ASSERT_TRUE(healthy.Load().IsOK());
  ASSERT_FALSE(healthy.Get(MakeKey(), 1).has_value());
  healthy.Put(MakeKey(), 1, MakeSm80Config());
  ASSERT_TRUE(healthy.Flush().IsOK());
  gc::MatMulNBitsTacticCache result(file, MakeSignature());
  ASSERT_TRUE(result.Load().IsOK());
  ASSERT_TRUE(result.Get(MakeKey(), 1).has_value());
  CleanUp(file);
}

TEST(GemmTacticCacheTest, PartialReadDoesNotAcceptRows) {
  const auto file = UniqueTempPrefix("partial") + ".matmulnbits_fpa_intb.tsv";
  gc::MatMulNBitsTacticCache writer(file, MakeSignature());
  writer.Put(MakeKey(), 1, MakeSm80Config());
  ASSERT_TRUE(writer.Flush().IsOK());
  std::ifstream input(file);
  std::string contents((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
  input.close();
  class FailingBuffer : public std::streambuf {
   public:
    explicit FailingBuffer(std::string& data) { setg(data.data(), data.data(), data.data() + data.size()); }
    int_type underflow() override { throw std::runtime_error("injected read error after a complete row"); }
  } buffer(contents);
  std::istream broken(&buffer);
  gc::MatMulNBitsTacticCache reader(file, MakeSignature());
  EXPECT_FALSE(reader.Load(broken).IsOK());
  EXPECT_TRUE(reader.GetAll(MakeKey()).empty());
  CleanUp(file);
}

TEST(GemmTacticCacheTest, LegacyNegativeRowsAreIgnored) {
  const auto file = UniqueTempPrefix("legacy_negative") + ".matmulnbits_fpa_intb.tsv";
  gc::MatMulNBitsTacticCache writer(file, MakeSignature());
  writer.Put(MakeKey(), 1, MakeSm80Config());
  ASSERT_TRUE(writer.Flush().IsOK());
  std::ifstream input(file);
  std::string contents((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
  input.close();
  auto columns_text = [](const std::optional<CutlassGemmConfig>& config) {
    std::vector<std::string> columns;
    gc::AppendConfigColumns(columns, config);
    std::string text;
    for (const auto& column : columns) text += "\t" + column;
    return text;
  };
  const auto successful = columns_text(MakeSm80Config());
  const auto position = contents.find(successful);
  ASSERT_NE(position, std::string::npos);
  contents.replace(position, successful.size(), columns_text(std::nullopt));
  std::istringstream legacy(contents);
  gc::MatMulNBitsTacticCache reader(file, MakeSignature());
  ASSERT_TRUE(reader.Load(legacy).IsOK());
  EXPECT_FALSE(reader.Get(MakeKey(), 1).has_value());
  CleanUp(file);
}

#ifdef _WIN32
TEST(GemmTacticCacheTest, UnreadableSnapshotIsNotReplaced) {
  const auto file = UniqueTempPrefix("unreadable") + ".matmulnbits_fpa_intb.tsv";
  gc::MatMulNBitsTacticCache cache(file, MakeSignature());
  cache.Put(MakeKey(), 1, MakeSm80Config());
  ASSERT_TRUE(cache.Flush().IsOK());
  // Deny reads while still permitting replacement. Flush must fail at the read, not the rename.
  HANDLE handle = CreateFileA(file.c_str(), GENERIC_WRITE, FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                              nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  ASSERT_NE(handle, INVALID_HANDLE_VALUE);
  cache.Put(MakeKey(), 64, MakeSm90Config());
  EXPECT_FALSE(cache.Load().IsOK());
  EXPECT_FALSE(cache.Flush().IsOK());
  CloseHandle(handle);
  gc::MatMulNBitsTacticCache reader(file, MakeSignature());
  ASSERT_TRUE(reader.Load().IsOK());
  EXPECT_TRUE(reader.Get(MakeKey(), 1).has_value());
  EXPECT_FALSE(reader.Get(MakeKey(), 64).has_value());
  EXPECT_TRUE(cache.Flush().IsOK());  // Failed writes retain their dirty rows.
  CleanUp(file);
}

TEST(GemmTacticCacheTest, FailedReplacementRemovesTemporaryFile) {
  const auto file = UniqueTempPrefix("replacefail") + ".matmulnbits_fpa_intb.tsv";
  gc::MatMulNBitsTacticCache cache(file, MakeSignature());
  cache.Put(MakeKey(), 1, MakeSm80Config());
  ASSERT_TRUE(cache.Flush().IsOK());
  HANDLE handle = CreateFileA(file.c_str(), GENERIC_READ, FILE_SHARE_READ,
                              nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  ASSERT_NE(handle, INVALID_HANDLE_VALUE);
  cache.Put(MakeKey(), 64, MakeSm90Config());
  EXPECT_FALSE(cache.Flush().IsOK());
  CloseHandle(handle);
  const auto path = std::filesystem::path(file);
  for (const auto& entry : std::filesystem::directory_iterator(path.parent_path())) {
    EXPECT_NE(entry.path().filename().string().find(path.filename().string() + ".tmp."), 0u);
  }
  gc::MatMulNBitsTacticCache reader(file, MakeSignature());
  ASSERT_TRUE(reader.Load().IsOK());
  EXPECT_FALSE(reader.Get(MakeKey(), 64).has_value());
  EXPECT_TRUE(cache.Flush().IsOK());
  CleanUp(file);
}
#endif

TEST(GemmTacticCacheTest, SharedExactAndRoundedHitsReachSeparatePrefixes) {
  using namespace onnxruntime::llm::kernels::weight_only;
  class Profiler : public GemmPluginProfiler<CutlassGemmConfig, std::shared_ptr<int>, GemmIdCore, GemmIdCoreHash> {
   public:
    std::shared_ptr<gc::MatMulNBitsTacticCache> cache;
    int stages = 0;

   protected:
    void runTactic(int, int, int, const CutlassGemmConfig&, char*, const cudaStream_t&) override {}
    size_t computeTmpSize(size_t, size_t, size_t) override { return 0; }
    std::vector<CutlassGemmConfig> getTactics(int, int, int) const override { return {}; }
    void stagePersistentCache(const GemmIdCore&, const MProfileMap& map, bool) override {
      ++stages;
      if (cache) {
        for (const auto& [m, config] : map) cache->Put(MakeKey(), m, config);
      }
    }
  };
  const auto file_a = UniqueTempPrefix("shared_a") + ".matmulnbits_fpa_intb.tsv";
  const auto file_b = UniqueTempPrefix("shared_b") + ".matmulnbits_fpa_intb.tsv";
  for (bool persist_a : {false, true}) {
    auto shared = std::make_shared<Profiler::MNKProfileMap>();
    GemmIdCore id(3072, 4096, onnxruntime::llm::nvinfer::DataType::kHALF);
    shared->createMProfileMap(id);
    Profiler a, b;
    a.setSelectionTactics(shared);
    b.setSelectionTactics(shared);
    if (persist_a) a.cache = std::make_shared<gc::MatMulNBitsTacticCache>(file_a, MakeSignature());
    b.cache = std::make_shared<gc::MatMulNBitsTacticCache>(file_b, MakeSignature());
    (*shared->getMProfileMap(id))[64] = MakeSm80Config();
    ASSERT_TRUE(a.getBestConfigOrProfile(64, id).has_value());
    ASSERT_TRUE(b.getBestConfigOrProfile(64, id).has_value());
    (*shared->getMProfileMap(id))[128] = MakeSm90Config();
    ASSERT_TRUE(a.getBestConfigOrProfile(128, id).has_value());
    ASSERT_TRUE(b.getBestConfigOrProfile(65, id).has_value());
    ASSERT_TRUE(b.getBestConfigOrProfile(65, id).has_value());
    EXPECT_EQ(b.stages, 2);  // Repeated inference performs no staging work.
    ASSERT_TRUE(b.cache->Flush().IsOK());
    gc::MatMulNBitsTacticCache result(file_b, MakeSignature());
    ASSERT_TRUE(result.Load().IsOK());
    EXPECT_EQ(result.GetAll(MakeKey()).size(), 2u);
    CleanUp(file_a);
    CleanUp(file_b);
  }
}

TEST(GemmTacticCacheCudaTest, TemporaryRunnerFailureDoesNotPoisonNextProfiler) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device required for profiler events";
  }
  using namespace onnxruntime::llm::kernels::weight_only;
  class Profiler : public GemmPluginProfiler<CutlassGemmConfig, std::shared_ptr<int>, GemmIdCore, GemmIdCoreHash> {
   public:
    std::shared_ptr<gc::MatMulNBitsTacticCache> cache;
    bool fail = false;
    int launches = 0;

   protected:
    void runTactic(int, int, int, const CutlassGemmConfig&, char*, const cudaStream_t&) override {
      ++launches;
      if (fail) throw std::runtime_error("temporary runner failure");
    }
    size_t computeTmpSize(size_t, size_t, size_t) override { return 1; }
    std::vector<CutlassGemmConfig> getTactics(int, int, int) const override { return {MakeSm80Config()}; }
    void loadPersistentCache(const GemmIdCore&, MProfileMap& map, bool) override {
      for (const auto& entry : cache->GetAll(MakeKey())) map.insert(entry);
    }
    void stagePersistentCache(const GemmIdCore&, const MProfileMap& map, bool) override {
      for (const auto& [m, config] : map) cache->Put(MakeKey(), m, config);
    }
  };
  const auto file = UniqueTempPrefix("runner_failure") + ".matmulnbits_fpa_intb.tsv";
  const auto dtype = onnxruntime::llm::nvinfer::DataType::kHALF;
  const GemmIdCore id(3072, 4096, dtype);
  for (bool fail : {true, false}) {
    Profiler profiler;
    profiler.fail = fail;
    profiler.cache = std::make_shared<gc::MatMulNBitsTacticCache>(file, MakeSignature());
    ASSERT_TRUE(profiler.cache->Load().IsOK());
    profiler.setAllocator(std::make_shared<CPUAllocator>());  // Fake runner never dereferences workspace.
    profiler.profileTactics(std::make_shared<int>(0), dtype, GemmDims(1, 1, id.n, id.k), id);
    EXPECT_GT(profiler.launches, 0);
    EXPECT_EQ(profiler.getBestConfig(1, id).has_value(), !fail);
    ASSERT_TRUE(profiler.cache->Flush().IsOK());
  }
  gc::MatMulNBitsTacticCache result(file, MakeSignature());
  ASSERT_TRUE(result.Load().IsOK());
  EXPECT_TRUE(result.Get(MakeKey(), 1).has_value());
  CleanUp(file);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // USE_FPA_INTB_GEMM

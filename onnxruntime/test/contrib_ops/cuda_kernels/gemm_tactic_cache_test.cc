// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Unit tests for the fpA_intB / MatMulNBits GEMM tactic cache utilities.
// These are pure-logic tests (no GPU required): HardwareSignature is constructed
// manually and only the (de)serialization, TSV parsing, and file round-trip paths
// are exercised.
//
// Built into onnxruntime_providers_cuda_ut (onnxruntime_ENABLE_CUDA_EP_INTERNAL_TESTS=ON); run like:
//  ./onnxruntime_provider_test --gtest_filter=CUDA_EP_Unittest.All
#if USE_FPA_INTB_GEMM
#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

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
  ASSERT_TRUE(c128.has_value());    // present in the cache
  EXPECT_FALSE(c128->has_value());  // but no valid tactic

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

TEST(GemmTacticCacheTest, FlushMergesConcurrentRows) {
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

}  // namespace test
}  // namespace onnxruntime

#endif  // USE_FPA_INTB_GEMM

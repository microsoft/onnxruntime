// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/// \file test_commit.cc
/// \brief Commit, vacuum, and validate tests.

#include "model_package.h"
#include "model_package_api.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <thread>

namespace fs = std::filesystem;

namespace {

int g_failed = 0;
int g_passed = 0;
const char* g_current = "<none>";

#define CHECK(cond)                                                                       \
  do {                                                                                    \
    if (!(cond)) {                                                                        \
      std::fprintf(stderr, "[FAIL] %s line %d: CHECK(%s)\n", g_current, __LINE__, #cond); \
      return false;                                                                       \
    }                                                                                     \
  } while (0)

#define CHECK_OK(status)                                                                              \
  do {                                                                                                \
    ModelPackageStatus* _s = (status);                                                                \
    if (_s != nullptr) {                                                                              \
      std::fprintf(stderr, "[FAIL] %s line %d: expected OK, got: %s\n",                               \
                   g_current, __LINE__, ModelPackageStatus_Message(_s));                              \
      ModelPackageStatus_Release(_s);                                                                 \
      return false;                                                                                   \
    }                                                                                                 \
  } while (0)

#define CHECK_ERR(status, expected_code)                                                              \
  do {                                                                                                \
    ModelPackageStatus* _s = (status);                                                                \
    if (_s == nullptr) {                                                                              \
      std::fprintf(stderr, "[FAIL] %s line %d: expected error %d, got OK\n",                          \
                   g_current, __LINE__, (int)(expected_code));                                        \
      return false;                                                                                   \
    }                                                                                                 \
    ModelPackageErrorCode _c = ModelPackageStatus_Code(_s);                                           \
    if (_c != (expected_code)) {                                                                      \
      std::fprintf(stderr, "[FAIL] %s line %d: expected error %d, got %d (%s)\n",                     \
                   g_current, __LINE__, (int)(expected_code), (int)_c,                                \
                   ModelPackageStatus_Message(_s));                                                   \
      ModelPackageStatus_Release(_s);                                                                 \
      return false;                                                                                   \
    }                                                                                                 \
    ModelPackageStatus_Release(_s);                                                                   \
  } while (0)

class Sandbox {
 public:
  Sandbox() {
    std::random_device rd;
    std::mt19937_64 g(rd());
    char buf[32];
    std::snprintf(buf, sizeof(buf), "mp_commit_%016lx", static_cast<unsigned long>(g()));
    root_ = fs::temp_directory_path() / buf;
    fs::create_directories(root_);
  }
  ~Sandbox() { std::error_code ec; fs::remove_all(root_, ec); }
  Sandbox(const Sandbox&) = delete;
  Sandbox& operator=(const Sandbox&) = delete;
  const fs::path& root() const { return root_; }
  fs::path path(const std::string& rel) const { return root_ / rel; }
  void Write(const std::string& rel, const std::string& contents) {
    fs::path full = root_ / rel;
    fs::create_directories(full.parent_path());
    std::ofstream f(full, std::ios::binary);
    f << contents;
  }
 private:
  fs::path root_;
};

class PkgHandle {
 public:
  explicit PkgHandle(ModelPackage* p) : p_(p) {}
  ~PkgHandle() { ModelPackage_Close(p_); }
  PkgHandle(const PkgHandle&) = delete;
  PkgHandle& operator=(const PkgHandle&) = delete;
  ModelPackage* get() const { return p_; }
  ModelPackage** outparam() { return &p_; }
 private:
  ModelPackage* p_;
};

// Open a freshly-created in-memory package ready to commit at `root`.
// `root` must be empty/nonexistent for the subsequent dest_root commit.
PkgHandle MakeAuthoredPkgAt(const fs::path& /*root*/,
                            const std::string& layout = "portable") {
  ModelPackage* raw = nullptr;
  ModelPackage_New(&raw);
  if (layout != "portable") ModelPackage_SetLayout(raw, layout.c_str());
  ModelPackage_SetComponentInline(raw, "encoder", R"({"variants": {}})");
  ModelPackage_SetVariant(raw, "encoder", "v1", R"({"ep": "CPU"})");
  return PkgHandle(raw);
}

// ─────────────────────────────────────────────────────────────────────────────
// Commit (in-place, PRESERVE)
// ─────────────────────────────────────────────────────────────────────────────

bool test_commit_inplace_basic_roundtrip() {
  Sandbox s;
  PkgHandle p = MakeAuthoredPkgAt(s.path("pkg"));
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("pkg").c_str(), MODEL_PACKAGE_WRITE_PRESERVE));
  // manifest.json exists.
  CHECK(fs::is_regular_file(s.path("pkg") / "manifest.json"));

  // Reopen and confirm.
  ModelPackage* re = nullptr;
  CHECK_OK(ModelPackage_Open(s.path("pkg").c_str(), nullptr, &re));
  PkgHandle rep(re);
  CHECK(ModelPackage_Info(rep.get())->num_components == 1);
  const ModelPackageInfo* info = ModelPackage_Info(rep.get());
  const ModelComponentInfo* c = ModelPackage_FindComponent(info, "encoder");
  CHECK(c != nullptr);
  CHECK(c->num_variants == 1);
  const ModelVariantInfo* v = ModelComponentInfo_FindVariant(c, "v1");
  CHECK(std::string(v->ep) == "CPU");
  return true;
}

bool test_commit_requires_package_root() {
  ModelPackage* raw = nullptr;
  CHECK_OK(ModelPackage_New(&raw));
  PkgHandle p(raw);
  CHECK_ERR(ModelPackage_Commit(p.get(), nullptr, MODEL_PACKAGE_WRITE_PRESERVE),
            MODEL_PACKAGE_ERR_STATE);
  return true;
}

bool test_commit_external_component_writes_file() {
  Sandbox s;
  // Author an inline package committed to disk first.
  PkgHandle p = MakeAuthoredPkgAt(s.path("pkg"));
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("pkg").c_str(), MODEL_PACKAGE_WRITE_PRESERVE));

  // Reopen, add an external component pointing at a file that doesn't exist yet.
  ModelPackage* re = nullptr;
  CHECK_OK(ModelPackage_Open(s.path("pkg").c_str(), nullptr, &re));
  PkgHandle rep(re);
  CHECK_OK(ModelPackage_SetComponentExternal(rep.get(), "decoder", "decoder.json"));
  CHECK_OK(ModelPackage_Commit(rep.get(), nullptr, MODEL_PACKAGE_WRITE_PRESERVE));
  CHECK(fs::is_regular_file(s.path("pkg") / "decoder.json"));
  CHECK(fs::is_regular_file(s.path("pkg") / "manifest.json"));

  // Reopen yet again and verify external component round-trips.
  ModelPackage* re2 = nullptr;
  CHECK_OK(ModelPackage_Open(s.path("pkg").c_str(), nullptr, &re2));
  PkgHandle rep2(re2);
  CHECK(ModelPackage_FindComponent(ModelPackage_Info(rep2.get()), "decoder") != nullptr);
  return true;
}

bool test_commit_pending_shared_asset_copy_in() {
  Sandbox s;
  s.Write("src_asset/m.onnx", "hello world");
  PkgHandle p = MakeAuthoredPkgAt(s.path("pkg"));

  const char* uri = nullptr;
  CHECK_OK(ModelPackage_AddSharedAsset(p.get(), s.path("src_asset").c_str(),
                                       nullptr, /*copy_in=*/true, &uri));
  std::string uri_copy(uri);
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("pkg").c_str(),
                               MODEL_PACKAGE_WRITE_PRESERVE));
  std::string hex = uri_copy.substr(7);
  fs::path landed = s.path("pkg") / "shared_assets" / ("sha256-" + hex);
  CHECK(fs::is_directory(landed));
  CHECK(fs::is_regular_file(landed / "m.onnx"));
  return true;
}

bool test_commit_dense_inlines_external_component() {
  Sandbox s;
  PkgHandle p = MakeAuthoredPkgAt(s.path("pkg"));
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("pkg").c_str(), MODEL_PACKAGE_WRITE_PRESERVE));
  CHECK_OK(ModelPackage_SetComponentExternal(p.get(), "decoder", "decoder.json"));
  CHECK_OK(ModelPackage_Commit(p.get(), nullptr, MODEL_PACKAGE_WRITE_DENSE));
  // The dense commit should NOT have written decoder.json (component became inline).
  CHECK(!fs::exists(s.path("pkg") / "decoder.json"));
  // Manifest contains decoder as an inline object.
  std::ifstream f(s.path("pkg") / "manifest.json");
  std::ostringstream oss; oss << f.rdbuf();
  std::string m = oss.str();
  CHECK(m.find("\"decoder\"") != std::string::npos);
  CHECK(m.find("\"variants\"") != std::string::npos);
  return true;
}

bool test_commit_dense_rejects_external_executor_info() {
  Sandbox s;
  PkgHandle p = MakeAuthoredPkgAt(s.path("pkg"));
  CHECK_OK(ModelPackage_SetVariantExecutorInfoExternal(
      p.get(), "encoder", "v1", "ort", "encoder/ort.json"));
  CHECK_ERR(ModelPackage_Commit(p.get(), s.path("pkg").c_str(), MODEL_PACKAGE_WRITE_DENSE),
            MODEL_PACKAGE_ERR_STATE);
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Commit (dest_root "save as")
// ─────────────────────────────────────────────────────────────────────────────

bool test_commit_dest_root_self_contained() {
  Sandbox s;
  s.Write("src_asset/m.onnx", "alpha");
  PkgHandle p = MakeAuthoredPkgAt(s.path("orig"));
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("orig").c_str(),
                               MODEL_PACKAGE_WRITE_PRESERVE));

  // Add an asset and commit as.
  const char* uri = nullptr;
  CHECK_OK(ModelPackage_AddSharedAsset(p.get(), s.path("src_asset").c_str(),
                                       nullptr, /*copy_in=*/true, &uri));
  std::string uri_copy(uri);
  fs::path saved = s.path("saved");
  CHECK_OK(ModelPackage_Commit(p.get(), saved.c_str(), MODEL_PACKAGE_WRITE_PRESERVE));
  CHECK(fs::is_regular_file(saved / "manifest.json"));
  std::string hex = uri_copy.substr(7);
  CHECK(fs::is_directory(saved / "shared_assets" / ("sha256-" + hex)));

  // After dest_root commit, in-memory state reflects the new root.
  // (We can verify by mutating + committing in-place again.)
  CHECK_OK(ModelPackage_SetMetadata(p.get(), "savedpkg", "1.0", nullptr));
  CHECK_OK(ModelPackage_Commit(p.get(), nullptr, MODEL_PACKAGE_WRITE_PRESERVE));
  // The most recent in-place commit should have landed at `saved`, not `orig`.
  std::ifstream f(saved / "manifest.json");
  std::ostringstream oss; oss << f.rdbuf();
  CHECK(oss.str().find("savedpkg") != std::string::npos);
  return true;
}

bool test_commit_dest_root_must_be_empty() {
  Sandbox s;
  PkgHandle p = MakeAuthoredPkgAt(s.path("pkg"));
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("pkg").c_str(),
                               MODEL_PACKAGE_WRITE_PRESERVE));
  s.Write("dest/something", "x");
  // Try to commit to non-empty dest.
  CHECK_ERR(ModelPackage_Commit(p.get(), s.path("dest").c_str(),
                                MODEL_PACKAGE_WRITE_PRESERVE),
            MODEL_PACKAGE_ERR_STATE);
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Vacuum
// ─────────────────────────────────────────────────────────────────────────────

bool test_vacuum_skips_within_grace_period() {
  Sandbox s;
  PkgHandle p = MakeAuthoredPkgAt(s.path("pkg"));
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("pkg").c_str(),
                               MODEL_PACKAGE_WRITE_PRESERVE));

  // Manually plant an orphan asset dir (fresh mtime).
  fs::path orphan = s.path("pkg") / "shared_assets" /
                    ("sha256-" + std::string(64, 'a'));
  fs::create_directories(orphan);
  CHECK(fs::is_directory(orphan));
  CHECK_OK(ModelPackage_Vacuum(p.get()));
  // Within grace period -> still there.
  CHECK(fs::is_directory(orphan));
  return true;
}

bool test_vacuum_removes_old_orphans() {
  Sandbox s;
  PkgHandle p = MakeAuthoredPkgAt(s.path("pkg"));
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("pkg").c_str(),
                               MODEL_PACKAGE_WRITE_PRESERVE));

  fs::path orphan = s.path("pkg") / "shared_assets" /
                    ("sha256-" + std::string(64, 'b'));
  fs::create_directories(orphan);
  // Backdate mtime to >60s ago.
  auto old = fs::file_time_type::clock::now() - std::chrono::seconds(120);
  std::error_code ec;
  fs::last_write_time(orphan, old, ec);
  CHECK(!ec);
  CHECK_OK(ModelPackage_Vacuum(p.get()));
  CHECK(!fs::exists(orphan));
  return true;
}

bool test_vacuum_removes_stale_staging_dirs() {
  Sandbox s;
  PkgHandle p = MakeAuthoredPkgAt(s.path("pkg"));
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("pkg").c_str(),
                               MODEL_PACKAGE_WRITE_PRESERVE));

  fs::path stage = s.path("pkg") / "shared_assets" /
                   ("sha256-" + std::string(64, 'c') + ".tmp.abcdef0123");
  fs::create_directories(stage);
  auto old = fs::file_time_type::clock::now() - std::chrono::seconds(120);
  std::error_code ec; fs::last_write_time(stage, old, ec);
  CHECK_OK(ModelPackage_Vacuum(p.get()));
  CHECK(!fs::exists(stage));
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Validate
// ─────────────────────────────────────────────────────────────────────────────

bool test_validate_all_clean_package() {
  Sandbox s;
  PkgHandle p = MakeAuthoredPkgAt(s.path("pkg"));
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("pkg").c_str(),
                               MODEL_PACKAGE_WRITE_PRESERVE));
  const char* report = nullptr;
  CHECK_OK(ModelPackage_Validate(p.get(), MODEL_PACKAGE_VALIDATE_ALL, &report));
  CHECK(report != nullptr);
  CHECK(std::string(report).find("\"errors\": []") != std::string::npos);
  return true;
}

bool test_validate_asset_reach_flags_unknown_uri() {
  Sandbox s;
  PkgHandle p = MakeAuthoredPkgAt(s.path("pkg"));
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("pkg").c_str(),
                               MODEL_PACKAGE_WRITE_PRESERVE));
  // Add a uses_assets URI but no matching shared asset.
  std::string fake_uri = "sha256:" + std::string(64, '0');
  std::string variant = R"({"variant_directory": "encoder", "uses_assets": [")" +
                        fake_uri + R"("]})";
  CHECK_OK(ModelPackage_SetVariant(p.get(), "encoder", "v1", variant.c_str()));
  const char* report = nullptr;
  CHECK_ERR(ModelPackage_Validate(p.get(), MODEL_PACKAGE_VALIDATE_ASSET_REACH, &report),
            MODEL_PACKAGE_ERR_STATE);
  CHECK(std::string(report).find("ASSET_REACH") != std::string::npos);
  return true;
}

bool test_validate_paths_flags_missing_external() {
  Sandbox s;
  PkgHandle p = MakeAuthoredPkgAt(s.path("pkg"));
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("pkg").c_str(),
                               MODEL_PACKAGE_WRITE_PRESERVE));
  // Register an external component then delete the file behind the library's back.
  CHECK_OK(ModelPackage_SetComponentExternal(p.get(), "decoder", "decoder.json"));
  CHECK_OK(ModelPackage_Commit(p.get(), nullptr, MODEL_PACKAGE_WRITE_PRESERVE));
  std::error_code ec;
  fs::remove(s.path("pkg") / "decoder.json", ec);
  const char* report = nullptr;
  CHECK_OK(ModelPackage_Validate(p.get(), MODEL_PACKAGE_VALIDATE_PATHS, &report));
  // PATHS findings are warnings, not errors -> OK status, but warning surfaces.
  CHECK(std::string(report).find("PATHS") != std::string::npos);
  return true;
}

bool test_validate_asset_rehash_detects_mutation() {
  Sandbox s;
  s.Write("src_asset/m.onnx", "alpha");
  PkgHandle p = MakeAuthoredPkgAt(s.path("pkg"));
  const char* uri = nullptr;
  CHECK_OK(ModelPackage_AddSharedAsset(p.get(), s.path("src_asset").c_str(),
                                       nullptr, /*copy_in=*/true, &uri));
  std::string uri_copy(uri);
  // Reference the asset from the variant so it surfaces in shared_assets[].
  std::string variant = R"({"uses_assets": [")" + uri_copy + R"("], "ep": "CPU"})";
  CHECK_OK(ModelPackage_SetVariant(p.get(), "encoder", "v1", variant.c_str()));
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("pkg").c_str(),
                               MODEL_PACKAGE_WRITE_PRESERVE));
  // Mutate the on-disk shared asset directly.
  std::string hex = uri_copy.substr(7);
  fs::path landed = s.path("pkg") / "shared_assets" / ("sha256-" + hex) / "m.onnx";
  CHECK(fs::is_regular_file(landed));
  { std::ofstream f(landed, std::ios::binary); f << "MUTATED"; }
  const char* report = nullptr;
  CHECK_ERR(ModelPackage_Validate(p.get(), MODEL_PACKAGE_VALIDATE_ASSET_REHASH, &report),
            MODEL_PACKAGE_ERR_STATE);
  CHECK(std::string(report).find("ASSET_REHASH") != std::string::npos);
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Atomicity hint: no stray .tmp.* under <pkg_root> after successful commit
// ─────────────────────────────────────────────────────────────────────────────

bool test_commit_leaves_no_temp_files() {
  Sandbox s;
  s.Write("src_asset/m.onnx", "alpha");
  PkgHandle p = MakeAuthoredPkgAt(s.path("pkg"));
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("pkg").c_str(),
                               MODEL_PACKAGE_WRITE_PRESERVE));
  const char* uri = nullptr;
  CHECK_OK(ModelPackage_AddSharedAsset(p.get(), s.path("src_asset").c_str(),
                                       nullptr, true, &uri));
  CHECK_OK(ModelPackage_SetComponentExternal(p.get(), "decoder", "decoder.json"));
  CHECK_OK(ModelPackage_Commit(p.get(), nullptr,
                               MODEL_PACKAGE_WRITE_PRESERVE));
  std::error_code ec;
  for (auto& e : fs::recursive_directory_iterator(s.path("pkg"), ec)) {
    if (e.path().filename().string().find(".tmp.") != std::string::npos) {
      std::fprintf(stderr, "  stray temp file: %s\n", e.path().c_str());
      return false;
    }
  }
  return true;
}

struct Test { const char* name; bool (*fn)(); };

const Test kTests[] = {
    {"commit_inplace_basic_roundtrip", test_commit_inplace_basic_roundtrip},
    {"commit_requires_package_root", test_commit_requires_package_root},
    {"commit_external_component_writes_file", test_commit_external_component_writes_file},
    {"commit_pending_shared_asset_copy_in", test_commit_pending_shared_asset_copy_in},
    {"commit_dense_inlines_external_component", test_commit_dense_inlines_external_component},
    {"commit_dense_rejects_external_executor_info", test_commit_dense_rejects_external_executor_info},
    {"commit_dest_root_self_contained", test_commit_dest_root_self_contained},
    {"commit_dest_root_must_be_empty", test_commit_dest_root_must_be_empty},
    {"vacuum_skips_within_grace_period", test_vacuum_skips_within_grace_period},
    {"vacuum_removes_old_orphans", test_vacuum_removes_old_orphans},
    {"vacuum_removes_stale_staging_dirs", test_vacuum_removes_stale_staging_dirs},
    {"validate_all_clean_package", test_validate_all_clean_package},
    {"validate_asset_reach_flags_unknown_uri", test_validate_asset_reach_flags_unknown_uri},
    {"validate_paths_flags_missing_external", test_validate_paths_flags_missing_external},
    {"validate_asset_rehash_detects_mutation", test_validate_asset_rehash_detects_mutation},
    {"commit_leaves_no_temp_files", test_commit_leaves_no_temp_files},
};

}  // namespace

int main() {
  for (const auto& t : kTests) {
    g_current = t.name;
    bool ok = t.fn();
    if (ok) { std::printf("[PASS] %s\n", t.name); g_passed++; }
    else    { g_failed++; }
  }
  std::printf("\n=== %d passed, %d failed ===\n", g_passed, g_failed);
  return g_failed == 0 ? 0 : 1;
}

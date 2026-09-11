// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/// \file test_commit.cc
/// \brief Commit, prune, and validate tests.

#include "model_package.h"
#include "model_package_api.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
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

#define CHECK_OK(status)                                                 \
  do {                                                                   \
    ModelPackageStatus* _s = (status);                                   \
    if (_s != nullptr) {                                                 \
      std::fprintf(stderr, "[FAIL] %s line %d: expected OK, got: %s\n",  \
                   g_current, __LINE__, ModelPackageStatus_Message(_s)); \
      ModelPackageStatus_Release(_s);                                    \
      return false;                                                      \
    }                                                                    \
  } while (0)

#define CHECK_ERR(status, expected_code)                                          \
  do {                                                                            \
    ModelPackageStatus* _s = (status);                                            \
    if (_s == nullptr) {                                                          \
      std::fprintf(stderr, "[FAIL] %s line %d: expected error %d, got OK\n",      \
                   g_current, __LINE__, (int)(expected_code));                    \
      return false;                                                               \
    }                                                                             \
    ModelPackageErrorCode _c = ModelPackageStatus_Code(_s);                       \
    if (_c != (expected_code)) {                                                  \
      std::fprintf(stderr, "[FAIL] %s line %d: expected error %d, got %d (%s)\n", \
                   g_current, __LINE__, (int)(expected_code), (int)_c,            \
                   ModelPackageStatus_Message(_s));                               \
      ModelPackageStatus_Release(_s);                                             \
      return false;                                                               \
    }                                                                             \
    ModelPackageStatus_Release(_s);                                               \
  } while (0)

class Sandbox {
 public:
  Sandbox() {
    std::random_device rd;
    std::mt19937_64 g(rd());
    char buf[32];
    std::snprintf(buf, sizeof(buf), "mp_commit_%016llx", static_cast<unsigned long long>(g()));
    root_ = fs::temp_directory_path() / buf;
    fs::create_directories(root_);
  }
  ~Sandbox() {
    std::error_code ec;
    fs::remove_all(root_, ec);
  }
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
  CHECK((ModelPackage_Info(rep.get()))->num_components == 1);
  const ModelPackageInfo* info = ModelPackage_Info(rep.get());
  const ModelComponentInfo* c = ModelPackage_FindComponent(info, "encoder");
  CHECK(c != nullptr);
  CHECK((c)->num_variants == 1);
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
  std::ostringstream oss;
  oss << f.rdbuf();
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
  std::ostringstream oss;
  oss << f.rdbuf();
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
// Prune
// ─────────────────────────────────────────────────────────────────────────────

bool test_commit_dest_root_rehashes_existing_asset() {
  Sandbox s;
  s.Write("src_asset/m.onnx", "alpha");
  PkgHandle p = MakeAuthoredPkgAt(s.path("orig"));
  const char* uri = nullptr;
  CHECK_OK(ModelPackage_AddSharedAsset(p.get(), s.path("src_asset").c_str(),
                                       nullptr, /*copy_in=*/true, &uri));
  std::string uri_copy(uri);
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("orig").c_str(),
                               MODEL_PACKAGE_WRITE_PRESERVE));

  // Tamper with the landed sha256-<hex>/ dir under the existing package root.
  std::string hex = uri_copy.substr(7);
  fs::path landed = s.path("orig") / "shared_assets" / ("sha256-" + hex) / "m.onnx";
  {
    std::ofstream f(landed, std::ios::binary);
    f << "TAMPERED";
  }

  // CommitToDestRoot must rehash the source and refuse the mismatch.
  CHECK_ERR(ModelPackage_Commit(p.get(), s.path("saved").c_str(),
                                MODEL_PACKAGE_WRITE_PRESERVE),
            MODEL_PACKAGE_ERR_STATE);
  return true;
}

bool test_prune_never_touches_shared_assets() {
  // Shared assets are content-addressed and only removed via explicit
  // RemoveSharedAsset. Even an obviously orphan sha256-<hex>/ directory that
  // matches no manifest entry must survive Prune.
  Sandbox s;
  PkgHandle p = MakeAuthoredPkgAt(s.path("pkg"));
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("pkg").c_str(),
                               MODEL_PACKAGE_WRITE_PRESERVE));

  fs::path planted = s.path("pkg") / "shared_assets" /
                     ("sha256-" + std::string(64, 'a'));
  fs::create_directories(planted);
  // Backdate mtime to past grace window to make sure it isn't grace-protected.
  auto old = fs::file_time_type::clock::now() - std::chrono::seconds(120);
  std::error_code ec;
  fs::last_write_time(planted, old, ec);
  CHECK_OK(ModelPackage_Prune(p.get()));
  CHECK(fs::is_directory(planted));
  return true;
}

bool test_prune_reclaims_tracked_orphan_variant_dirs() {
  Sandbox s;
  PkgHandle p = MakeAuthoredPkgAt(s.path("pkg"));
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("pkg").c_str(),
                               MODEL_PACKAGE_WRITE_PRESERVE));
  // Now that package_root is anchored, materialize an on-disk variant dir and
  // register it so subsequent removal records a tracked orphan.
  fs::path victim = s.path("pkg") / "encoder" / "v1";
  fs::create_directories(victim);
  CHECK_OK(ModelPackage_SetVariant(p.get(), "encoder", "v1",
                                   R"({"ep":"CPU","variant_directory":"encoder/v1"})"));
  CHECK_OK(ModelPackage_Commit(p.get(), nullptr, MODEL_PACKAGE_WRITE_PRESERVE));
  CHECK(fs::is_directory(victim));
  CHECK_OK(ModelPackage_RemoveVariant(p.get(), "encoder", "v1"));
  CHECK_OK(ModelPackage_Commit(p.get(), nullptr, MODEL_PACKAGE_WRITE_PRESERVE));
  CHECK_OK(ModelPackage_Prune(p.get()));
  CHECK(!fs::exists(victim));
  return true;
}

bool test_prune_removes_stale_staging_dirs() {
  Sandbox s;
  PkgHandle p = MakeAuthoredPkgAt(s.path("pkg"));
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("pkg").c_str(),
                               MODEL_PACKAGE_WRITE_PRESERVE));

  fs::path stage = s.path("pkg") / "shared_assets" /
                   ("sha256-" + std::string(64, 'c') + ".tmp.abcdef0123");
  fs::create_directories(stage);
  auto old = fs::file_time_type::clock::now() - std::chrono::seconds(120);
  std::error_code ec;
  fs::last_write_time(stage, old, ec);
  CHECK_OK(ModelPackage_Prune(p.get()));
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
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("pkg").c_str(),
                               MODEL_PACKAGE_WRITE_PRESERVE));
  // Mutate the on-disk shared asset directly.
  std::string hex = uri_copy.substr(7);
  fs::path landed = s.path("pkg") / "shared_assets" / ("sha256-" + hex) / "m.onnx";
  CHECK(fs::is_regular_file(landed));
  {
    std::ofstream f(landed, std::ios::binary);
    f << "MUTATED";
  }
  const char* report = nullptr;
  CHECK_ERR(ModelPackage_Validate(p.get(), MODEL_PACKAGE_VALIDATE_ASSET_REHASH, &report),
            MODEL_PACKAGE_ERR_STATE);
  CHECK(std::string(report).find("ASSET_REHASH") != std::string::npos);
  return true;
}

bool test_commit_accepts_unreferenced_shared_asset() {
  // Shared assets no longer require an in-manifest reference: AddSharedAsset
  // signals the user's intent to ship the asset, period. Commit materializes
  // it under shared_assets/ at the default-convention path.
  Sandbox s;
  s.Write("src_asset/m.onnx", "alpha");
  PkgHandle p = MakeAuthoredPkgAt(s.path("pkg"));
  const char* uri = nullptr;
  CHECK_OK(ModelPackage_AddSharedAsset(p.get(), s.path("src_asset").c_str(),
                                       nullptr, /*copy_in=*/true, &uri));
  std::string uri_copy(uri);
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("pkg").c_str(),
                               MODEL_PACKAGE_WRITE_PRESERVE));
  std::string hex = uri_copy.substr(7);
  CHECK(fs::is_directory(s.path("pkg") / "shared_assets" / ("sha256-" + hex)));
  // Same on dest_root path.
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("saved").c_str(),
                               MODEL_PACKAGE_WRITE_PRESERVE));
  CHECK(fs::is_directory(s.path("saved") / "shared_assets" / ("sha256-" + hex)));
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
  (void)uri;
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

struct Test {
  const char* name;
  bool (*fn)();
};

const Test kTests[] = {
    {"commit_inplace_basic_roundtrip", test_commit_inplace_basic_roundtrip},
    {"commit_requires_package_root", test_commit_requires_package_root},
    {"commit_external_component_writes_file", test_commit_external_component_writes_file},
    {"commit_pending_shared_asset_copy_in", test_commit_pending_shared_asset_copy_in},
    {"commit_dense_inlines_external_component", test_commit_dense_inlines_external_component},
    {"commit_dense_rejects_external_executor_info", test_commit_dense_rejects_external_executor_info},
    {"commit_dest_root_self_contained", test_commit_dest_root_self_contained},
    {"commit_dest_root_must_be_empty", test_commit_dest_root_must_be_empty},
    {"commit_dest_root_rehashes_existing_asset", test_commit_dest_root_rehashes_existing_asset},
    {"prune_never_touches_shared_assets", test_prune_never_touches_shared_assets},
    {"prune_reclaims_tracked_orphan_variant_dirs", test_prune_reclaims_tracked_orphan_variant_dirs},
    {"prune_removes_stale_staging_dirs", test_prune_removes_stale_staging_dirs},
    {"validate_all_clean_package", test_validate_all_clean_package},
    {"validate_paths_flags_missing_external", test_validate_paths_flags_missing_external},
    {"validate_asset_rehash_detects_mutation", test_validate_asset_rehash_detects_mutation},
    {"commit_accepts_unreferenced_shared_asset", test_commit_accepts_unreferenced_shared_asset},
    {"commit_leaves_no_temp_files", test_commit_leaves_no_temp_files},
};

}  // namespace

int main() {
  for (const auto& t : kTests) {
    g_current = t.name;
    bool ok = t.fn();
    if (ok) {
      std::printf("[PASS] %s\n", t.name);
      g_passed++;
    } else {
      g_failed++;
    }
  }
  std::printf("\n=== %d passed, %d failed ===\n", g_passed, g_failed);
  return g_failed == 0 ? 0 : 1;
}

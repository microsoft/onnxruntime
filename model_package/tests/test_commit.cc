// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/// \file test_commit.cc
/// \brief Commit, prune, and validate tests.

#include "model_package.h"
#include "model_package_api.h"
#include "model_package_impl.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>

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
  fs::path path(const std::string& rel) const { return root_ / fs::u8path(rel); }
  void Write(const std::string& rel, const std::string& contents) {
    fs::path full = root_ / fs::u8path(rel);
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

class ScopedCurrentPath {
 public:
  explicit ScopedCurrentPath(const fs::path& path) : original_(fs::current_path()) {
    fs::current_path(path);
  }
  ~ScopedCurrentPath() {
    std::error_code ec;
    fs::current_path(original_, ec);
  }
  ScopedCurrentPath(const ScopedCurrentPath&) = delete;
  ScopedCurrentPath& operator=(const ScopedCurrentPath&) = delete;

 private:
  fs::path original_;
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
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("pkg").u8string().c_str(), MODEL_PACKAGE_WRITE_PRESERVE));
  // manifest.json exists.
  CHECK(fs::is_regular_file(s.path("pkg") / "manifest.json"));

  // Reopen and confirm.
  ModelPackage* re = nullptr;
  CHECK_OK(ModelPackage_Open(s.path("pkg").u8string().c_str(), nullptr, &re));
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
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("pkg").u8string().c_str(), MODEL_PACKAGE_WRITE_PRESERVE));

  // Reopen, add an external component pointing at a file that doesn't exist yet.
  ModelPackage* re = nullptr;
  CHECK_OK(ModelPackage_Open(s.path("pkg").u8string().c_str(), nullptr, &re));
  PkgHandle rep(re);
  CHECK_OK(ModelPackage_SetComponentExternal(rep.get(), "decoder", "decoder.json"));
  CHECK_OK(ModelPackage_Commit(rep.get(), nullptr, MODEL_PACKAGE_WRITE_PRESERVE));
  CHECK(fs::is_regular_file(s.path("pkg") / "decoder.json"));
  CHECK(fs::is_regular_file(s.path("pkg") / "manifest.json"));

  // Reopen yet again and verify external component round-trips.
  ModelPackage* re2 = nullptr;
  CHECK_OK(ModelPackage_Open(s.path("pkg").u8string().c_str(), nullptr, &re2));
  PkgHandle rep2(re2);
  CHECK(ModelPackage_FindComponent(ModelPackage_Info(rep2.get()), "decoder") != nullptr);
  return true;
}

bool test_commit_pending_shared_asset_copy_in() {
  Sandbox s;
  s.Write("src_asset/m.onnx", "hello world");
  PkgHandle p = MakeAuthoredPkgAt(s.path("pkg"));

  const char* uri = nullptr;
  CHECK_OK(ModelPackage_AddSharedAsset(p.get(), s.path("src_asset").u8string().c_str(),
                                       nullptr, /*copy_in=*/true, &uri));
  std::string uri_copy(uri);
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("pkg").u8string().c_str(),
                               MODEL_PACKAGE_WRITE_PRESERVE));
  std::string hex = uri_copy.substr(7);
  fs::path landed = s.path("pkg") / "shared_assets" / ("sha256-" + hex);
  CHECK(fs::is_directory(landed));
  CHECK(fs::is_regular_file(landed / "m.onnx"));
  return true;
}

bool test_commit_shared_asset_copy_in_nested_directory() {
  Sandbox s;
  s.Write("src_asset/m.onnx", "hello world");
  s.Write("src_asset/weights/w.bin", "weights");
  PkgHandle p = MakeAuthoredPkgAt(s.path("pkg"));

  const char* uri = nullptr;
  CHECK_OK(ModelPackage_AddSharedAsset(p.get(), s.path("src_asset").u8string().c_str(),
                                       nullptr, /*copy_in=*/true, &uri));
  std::string uri_copy(uri);
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("pkg").u8string().c_str(),
                               MODEL_PACKAGE_WRITE_PRESERVE));
  fs::path landed = s.path("pkg") / "shared_assets" / ("sha256-" + uri_copy.substr(7));
  CHECK(fs::is_regular_file(landed / "weights" / "w.bin"));
  return true;
}

bool test_commit_reports_unreadable_copy_in_subdirectory() {
  Sandbox s;
  s.Write("src_asset/m.onnx", "hello world");
  s.Write("src_asset/weights/w.bin", "weights");
  PkgHandle p = MakeAuthoredPkgAt(s.path("pkg"));

  const char* uri = nullptr;
  CHECK_OK(ModelPackage_AddSharedAsset(p.get(), s.path("src_asset").u8string().c_str(),
                                       nullptr, /*copy_in=*/true, &uri));
  const fs::path locked = s.path("src_asset") / "weights";
  std::error_code ec;
  fs::permissions(locked, fs::perms::none, ec);
  fs::directory_iterator probe(locked, ec);
  if (!ec) {
    fs::permissions(locked, fs::perms::owner_all, ec);
    std::printf("[SKIP] %s (directory permissions not enforced)\n", g_current);
    return true;
  }
  ModelPackageStatus* st = ModelPackage_Commit(p.get(), s.path("pkg").u8string().c_str(),
                                               MODEL_PACKAGE_WRITE_PRESERVE);
  fs::permissions(locked, fs::perms::owner_all, ec);
  CHECK_ERR(st, MODEL_PACKAGE_ERR_IO);
  return true;
}

bool test_commit_dense_inlines_external_component() {
  Sandbox s;
  PkgHandle p = MakeAuthoredPkgAt(s.path("pkg"));
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("pkg").u8string().c_str(), MODEL_PACKAGE_WRITE_PRESERVE));
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
  CHECK_ERR(ModelPackage_Commit(p.get(), s.path("pkg").u8string().c_str(), MODEL_PACKAGE_WRITE_DENSE),
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
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("orig").u8string().c_str(),
                               MODEL_PACKAGE_WRITE_PRESERVE));

  // Add an asset and commit as.
  const char* uri = nullptr;
  CHECK_OK(ModelPackage_AddSharedAsset(p.get(), s.path("src_asset").u8string().c_str(),
                                       nullptr, /*copy_in=*/true, &uri));
  std::string uri_copy(uri);
  fs::path saved = s.path("saved");
  CHECK_OK(ModelPackage_Commit(p.get(), saved.u8string().c_str(), MODEL_PACKAGE_WRITE_PRESERVE));
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
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("pkg").u8string().c_str(),
                               MODEL_PACKAGE_WRITE_PRESERVE));
  s.Write("dest/something", "x");
  // Try to commit to non-empty dest.
  CHECK_ERR(ModelPackage_Commit(p.get(), s.path("dest").u8string().c_str(),
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
  CHECK_OK(ModelPackage_AddSharedAsset(p.get(), s.path("src_asset").u8string().c_str(),
                                       nullptr, /*copy_in=*/true, &uri));
  std::string uri_copy(uri);
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("orig").u8string().c_str(),
                               MODEL_PACKAGE_WRITE_PRESERVE));

  // Tamper with the landed sha256-<hex>/ dir under the existing package root.
  std::string hex = uri_copy.substr(7);
  fs::path landed = s.path("orig") / "shared_assets" / ("sha256-" + hex) / "m.onnx";
  {
    std::ofstream f(landed, std::ios::binary);
    f << "TAMPERED";
  }

  // CommitToDestRoot must rehash the source and refuse the mismatch.
  CHECK_ERR(ModelPackage_Commit(p.get(), s.path("saved").u8string().c_str(),
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
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("pkg").u8string().c_str(),
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
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("pkg").u8string().c_str(),
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
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("pkg").u8string().c_str(),
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

bool test_prune_preserves_linked_shared_assets_entry() {
  Sandbox s;
  PkgHandle p = MakeAuthoredPkgAt(s.path("pkg"));
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("pkg").u8string().c_str(),
                               MODEL_PACKAGE_WRITE_PRESERVE));

  fs::path loop = s.path("pkg") / "shared_assets" / "loop";
  std::error_code ec;
  fs::create_directories(loop.parent_path(), ec);
  fs::create_directory_symlink("loop", loop, ec);
  if (ec) {
    std::printf("[SKIP] %s (symlink unsupported)\n", g_current);
    return true;
  }
  CHECK_OK(ModelPackage_Prune(p.get()));
  CHECK(fs::is_symlink(loop));
  return true;
}

#ifndef _WIN32
bool test_prune_reports_unreadable_shared_assets_directory() {
  Sandbox s;
  PkgHandle p = MakeAuthoredPkgAt(s.path("pkg"));
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("pkg").u8string().c_str(),
                               MODEL_PACKAGE_WRITE_PRESERVE));

  const fs::path assets_root = s.path("pkg") / "shared_assets";
  std::error_code ec;
  fs::create_directories(assets_root, ec);
  CHECK(!ec);
  fs::permissions(assets_root, fs::perms::none, ec);
  if (ec) {
    std::printf("[SKIP] %s (cannot remove directory permissions)\n", g_current);
    return true;
  }
  std::error_code iter_ec;
  fs::directory_iterator probe(assets_root, iter_ec);
  if (!iter_ec) {
    fs::permissions(assets_root, fs::perms::owner_all, ec);
    std::printf("[SKIP] %s (directory remains readable)\n", g_current);
    return true;
  }
  ModelPackageStatus* status = ModelPackage_Prune(p.get());
  fs::permissions(assets_root, fs::perms::owner_all, ec);
  CHECK(status != nullptr);
  CHECK(ModelPackageStatus_Code(status) == MODEL_PACKAGE_ERR_IO);
  ModelPackageStatus_Release(status);
  return true;
}
#endif

bool test_prune_skips_stale_staging_links() {
  Sandbox s;
  PkgHandle p = MakeAuthoredPkgAt(s.path("pkg"));
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("pkg").u8string().c_str(),
                               MODEL_PACKAGE_WRITE_PRESERVE));

  const fs::path assets_root = s.path("pkg") / "shared_assets";
  const fs::path linked_stage = assets_root /
                                ("sha256-" + std::string(64, 'a') + ".tmp.link");
  const fs::path regular_stage = assets_root /
                                 ("sha256-" + std::string(64, 'b') + ".tmp.regular");
  fs::create_directories(s.path("outside"));
  fs::create_directories(regular_stage);
  std::error_code ec;
  fs::create_directory_symlink(s.path("outside"), linked_stage, ec);
  if (ec) {
    std::printf("[SKIP] %s (symlink unsupported)\n", g_current);
    return true;
  }
  fs::last_write_time(regular_stage,
                      fs::file_time_type::clock::now() - std::chrono::seconds(120), ec);

  CHECK_OK(ModelPackage_Prune(p.get()));
  CHECK(fs::is_symlink(linked_stage));
  CHECK(!fs::exists(regular_stage));
  return true;
}

#ifdef _WIN32
bool test_windows_junction_rejected_by_public_apis() {
  Sandbox s;
  const fs::path target = s.path("junction_target");
  const fs::path junction = s.path("junction");
  s.Write("junction_target/model.onnx", "asset");
  fs::create_directories(target);
  const std::wstring command =
      L"cmd.exe /d /c mklink /J \"" + junction.wstring() + L"\" \"" +
      target.wstring() + L"\" >nul";
  if (::_wsystem(command.c_str()) != 0) {
    std::printf("[SKIP] %s (junction creation unsupported)\n", g_current);
    return true;
  }

  PkgHandle p = MakeAuthoredPkgAt(s.path("pkg"));
  const char* uri = nullptr;
  CHECK_ERR(ModelPackage_AddSharedAsset(p.get(), junction.u8string().c_str(),
                                        nullptr, true, &uri),
            MODEL_PACKAGE_ERR_SCHEMA);

  CHECK_OK(ModelPackage_Commit(p.get(), s.path("pkg").u8string().c_str(),
                               MODEL_PACKAGE_WRITE_PRESERVE));
  fs::remove_all(s.path("pkg") / "shared_assets");
  const fs::path assets_target = s.path("assets_target");
  fs::create_directories(assets_target);
  const fs::path assets_junction = s.path("pkg") / "shared_assets";
  const std::wstring assets_command =
      L"cmd.exe /d /c mklink /J \"" + assets_junction.wstring() + L"\" \"" +
      assets_target.wstring() + L"\" >nul";
  if (::_wsystem(assets_command.c_str()) != 0) {
    std::printf("[SKIP] %s (shared_assets junction creation unsupported)\n", g_current);
    return true;
  }
  ModelPackage* opened = nullptr;
  CHECK_ERR(ModelPackage_Open(s.path("pkg").u8string().c_str(), nullptr, &opened),
            MODEL_PACKAGE_ERR_PATH_CONFINEMENT);
  CHECK(opened == nullptr);
  return true;
}
#endif

bool test_open_rejects_linked_shared_asset_paths() {
  Sandbox s;
  PkgHandle p = MakeAuthoredPkgAt(s.path("pkg"));
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("pkg").u8string().c_str(),
                               MODEL_PACKAGE_WRITE_PRESERVE));

  fs::create_directories(s.path("outside"));
  std::error_code ec;
  fs::create_directory_symlink(s.path("outside"), s.path("pkg") / "shared_assets", ec);
  if (ec) {
    std::printf("[SKIP] %s (symlink unsupported)\n", g_current);
    return true;
  }

  ModelPackage* opened = nullptr;
  CHECK_ERR(ModelPackage_Open(s.path("pkg").u8string().c_str(), nullptr, &opened),
            MODEL_PACKAGE_ERR_PATH_CONFINEMENT);
  CHECK(opened == nullptr);

  fs::remove(s.path("pkg") / "shared_assets", ec);
  fs::create_directories(s.path("pkg") / "shared_assets");
  const fs::path linked_asset = s.path("pkg") / "shared_assets" /
                                ("sha256-" + std::string(64, 'a'));
  fs::create_directory_symlink(s.path("outside"), linked_asset, ec);
  CHECK(!ec);

  CHECK_ERR(ModelPackage_Open(s.path("pkg").u8string().c_str(), nullptr, &opened),
            MODEL_PACKAGE_ERR_PATH_CONFINEMENT);
  CHECK(opened == nullptr);
  return true;
}

bool test_shared_asset_mutations_reject_replaced_parent() {
  Sandbox s;
  s.Write("source/model.onnx", "asset");
  PkgHandle p = MakeAuthoredPkgAt(s.path("pkg"));
  const char* uri = nullptr;
  CHECK_OK(ModelPackage_AddSharedAsset(p.get(), s.path("source").u8string().c_str(),
                                       nullptr, true, &uri));
  const std::string asset_uri(uri);
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("pkg").u8string().c_str(),
                               MODEL_PACKAGE_WRITE_PRESERVE));

  std::error_code ec;
  fs::remove_all(s.path("pkg") / "shared_assets", ec);
  s.Write("outside/marker.txt", "keep");
  fs::create_directory_symlink(s.path("outside"), s.path("pkg") / "shared_assets", ec);
  if (ec) {
    std::printf("[SKIP] %s (symlink unsupported)\n", g_current);
    return true;
  }

  CHECK_ERR(ModelPackage_RemoveSharedAsset(p.get(), asset_uri.c_str()),
            MODEL_PACKAGE_ERR_PATH_CONFINEMENT);
  CHECK(fs::is_regular_file(s.path("outside/marker.txt")));
  const ModelPackageInfo* info = ModelPackage_Info(p.get());
  CHECK(info->num_shared_assets == 1);
  CHECK(std::string(info->shared_assets[0].uri) == asset_uri);
  CHECK_ERR(ModelPackage_Prune(p.get()), MODEL_PACKAGE_ERR_PATH_CONFINEMENT);
  CHECK(fs::is_regular_file(s.path("outside/marker.txt")));

  s.Write("second_source/model.onnx", "different asset");
  const char* const unchanged = "unchanged";
  const char* second_uri = unchanged;
  CHECK_ERR(ModelPackage_AddSharedAsset(p.get(), s.path("second_source").u8string().c_str(),
                                        nullptr, true, &second_uri),
            MODEL_PACKAGE_ERR_PATH_CONFINEMENT);
  CHECK(second_uri == unchanged);
  info = ModelPackage_Info(p.get());
  CHECK(info->num_shared_assets == 1);
  CHECK(std::string(info->shared_assets[0].uri) == asset_uri);
  CHECK(fs::is_regular_file(s.path("outside/marker.txt")));
  return true;
}

bool test_dest_commit_rejects_replaced_asset_link() {
  Sandbox s;
  s.Write("source/model.onnx", "asset");
  PkgHandle p = MakeAuthoredPkgAt(s.path("pkg"));
  const char* uri = nullptr;
  CHECK_OK(ModelPackage_AddSharedAsset(p.get(), s.path("source").u8string().c_str(),
                                       nullptr, true, &uri));
  const std::string asset_dir_name = "sha256-" + std::string(uri).substr(7);
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("pkg").u8string().c_str(),
                               MODEL_PACKAGE_WRITE_PRESERVE));

  fs::path asset_dir = s.path("pkg") / "shared_assets" / asset_dir_name;
  std::error_code ec;
  fs::remove_all(asset_dir, ec);
  s.Write("outside/marker.txt", "keep");
  fs::create_directory_symlink(s.path("outside"), asset_dir, ec);
  if (ec) {
    std::printf("[SKIP] %s (symlink unsupported)\n", g_current);
    return true;
  }

  CHECK_ERR(ModelPackage_Commit(p.get(), s.path("saved").u8string().c_str(),
                                MODEL_PACKAGE_WRITE_PRESERVE),
            MODEL_PACKAGE_ERR_PATH_CONFINEMENT);
  CHECK(fs::is_regular_file(s.path("outside/marker.txt")));
  CHECK(!fs::exists(s.path("saved") / "shared_assets" / asset_dir_name / "marker.txt"));
  return true;
}

bool test_linked_source_root_behavior() {
  Sandbox s;
  s.Write("outside/marker.txt", "keep");
  std::error_code ec;
  fs::create_directory_symlink(s.path("outside"), s.path("source"), ec);
  if (ec) {
    std::printf("[SKIP] %s (symlink unsupported)\n", g_current);
    return true;
  }

  const char* uri = nullptr;
  std::string expected_uri;
  for (const char* suffix : {"/", "/."}) {
    const std::string linked_source = s.path("source").u8string() + suffix;
    CHECK_OK(ModelPackage_ComputeDirectoryHash(linked_source.c_str(), &uri));
    if (expected_uri.empty()) {
      expected_uri = uri;
    } else {
      CHECK(std::string(uri) == expected_uri);
    }
  }

  PkgHandle installed = MakeAuthoredPkgAt(s.path("installed"), "installed");
  CHECK_OK(ModelPackage_AddSharedAsset(installed.get(), s.path("source").u8string().c_str(),
                                       expected_uri.c_str(), false, &uri));
  CHECK((ModelPackage_Info(installed.get()))->num_shared_assets == 1);

  PkgHandle p = MakeAuthoredPkgAt(s.path("pkg"));
  const char* const unchanged = "unchanged";
  uri = unchanged;
  CHECK_ERR(ModelPackage_AddSharedAsset(p.get(), s.path("source").u8string().c_str(),
                                        nullptr, true, &uri),
            MODEL_PACKAGE_ERR_SCHEMA);
  CHECK(uri == unchanged);
  CHECK((ModelPackage_Info(p.get()))->num_shared_assets == 0);

  fs::remove(s.path("source"), ec);
  s.Write("source/model.onnx", "asset");
  const std::string source_with_dot = s.path("source").u8string() + "/.";
  CHECK_OK(ModelPackage_AddSharedAsset(p.get(), source_with_dot.c_str(),
                                       nullptr, true, &uri));

  fs::remove_all(s.path("source"), ec);
  fs::create_directory_symlink(s.path("outside"), s.path("source"), ec);
  CHECK(!ec);
  CHECK_ERR(ModelPackage_Commit(p.get(), s.path("pkg").u8string().c_str(),
                                MODEL_PACKAGE_WRITE_PRESERVE),
            MODEL_PACKAGE_ERR_SCHEMA);
  CHECK(fs::is_regular_file(s.path("outside/marker.txt")));
  return true;
}

bool test_source_path_preserves_link_ancestor_parent_traversal() {
  Sandbox s;
  s.Write("outside/asset/model.onnx", "outside");
  s.Write("work/asset/model.onnx", "work");
  fs::create_directories(s.path("outside/subdir"));
  std::error_code ec;
  fs::create_directory_symlink(s.path("outside/subdir"), s.path("work/alias"), ec);
  if (ec) {
    std::printf("[SKIP] %s (symlink unsupported)\n", g_current);
    return true;
  }

  const fs::path source = s.path("work/alias") / ".." / "asset";
  const char* expected_uri = nullptr;
  CHECK_OK(ModelPackage_ComputeDirectoryHash(s.path("outside/asset").u8string().c_str(),
                                             &expected_uri));
  const std::string expected_uri_copy(expected_uri);
  const char* wrong_uri = nullptr;
  CHECK_OK(ModelPackage_ComputeDirectoryHash(s.path("work/asset").u8string().c_str(),
                                             &wrong_uri));
  CHECK(expected_uri_copy != wrong_uri);

  const char* source_uri = nullptr;
  CHECK_OK(ModelPackage_ComputeDirectoryHash(source.u8string().c_str(), &source_uri));
  CHECK(std::string(source_uri) == expected_uri_copy);

  PkgHandle p = MakeAuthoredPkgAt(s.path("pkg"));
  CHECK_OK(ModelPackage_AddSharedAsset(p.get(), source.u8string().c_str(),
                                       expected_uri_copy.c_str(), true, &source_uri));
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("pkg").u8string().c_str(),
                               MODEL_PACKAGE_WRITE_PRESERVE));
  const fs::path copied = s.path("pkg") / "shared_assets" /
                          ("sha256-" + expected_uri_copy.substr(7)) / "model.onnx";
  std::ifstream copied_file(copied, std::ios::binary);
  std::string copied_contents;
  copied_file >> copied_contents;
  CHECK(copied_contents == "outside");
  return true;
}

bool test_failed_shared_asset_refresh_preserves_live_records() {
  Sandbox s;
  s.Write("source_a/model.onnx", "asset a");
  s.Write("source_c/model.onnx", "asset c");
  PkgHandle p = MakeAuthoredPkgAt(s.path("pkg"));
  const char* uri_a = nullptr;
  CHECK_OK(ModelPackage_AddSharedAsset(p.get(), s.path("source_a").u8string().c_str(),
                                       nullptr, true, &uri_a));
  const std::string uri_a_copy(uri_a);
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("pkg").u8string().c_str(),
                               MODEL_PACKAGE_WRITE_PRESERVE));

  fs::create_directories(s.path("outside"));
  const fs::path linked_entry = s.path("pkg") / "shared_assets" /
                                ("sha256-" + std::string(64, 'b'));
  std::error_code ec;
  fs::create_directory_symlink(s.path("outside"), linked_entry, ec);
  if (ec) {
    std::printf("[SKIP] %s (symlink unsupported)\n", g_current);
    return true;
  }

  const char* const unchanged = "unchanged";
  const char* uri_c = unchanged;
  CHECK_ERR(ModelPackage_AddSharedAsset(p.get(), s.path("source_c").u8string().c_str(),
                                        nullptr, true, &uri_c),
            MODEL_PACKAGE_ERR_PATH_CONFINEMENT);
  CHECK(uri_c == unchanged);
  const ModelPackageInfo* info = ModelPackage_Info(p.get());
  CHECK(info->num_shared_assets == 1);
  CHECK(std::string(info->shared_assets[0].uri) == uri_a_copy);
  return true;
}

bool test_open_rejects_linked_manifest_override() {
  Sandbox s;
  s.Write("source/model.onnx", "asset");
  PkgHandle p = MakeAuthoredPkgAt(s.path("pkg"));
  const char* uri = nullptr;
  CHECK_OK(ModelPackage_AddSharedAsset(p.get(), s.path("source").u8string().c_str(),
                                       nullptr, true, &uri));
  const std::string asset_uri(uri);
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("pkg").u8string().c_str(),
                               MODEL_PACKAGE_WRITE_PRESERVE));

  s.Write("pkg/assets/target/model.onnx", "asset");
  std::error_code ec;
  fs::create_directory_symlink("target", s.path("pkg/assets/alias"), ec);
  if (ec) {
    std::printf("[SKIP] %s (symlink unsupported)\n", g_current);
    return true;
  }

  const fs::path manifest_path = s.path("pkg/manifest.json");
  nlohmann::ordered_json manifest;
  {
    std::ifstream input(manifest_path);
    input >> manifest;
  }
  manifest["shared_assets"][asset_uri] = "assets/alias";
  {
    std::ofstream output(manifest_path, std::ios::trunc);
    output << manifest.dump(2) << "\n";
  }

  ModelPackage* opened = nullptr;
  CHECK_ERR(ModelPackage_Open(s.path("pkg").u8string().c_str(), nullptr, &opened),
            MODEL_PACKAGE_ERR_PATH_CONFINEMENT);
  CHECK(opened == nullptr);
  return true;
}

bool test_commit_accepts_relative_destination_roots() {
  Sandbox s;
  s.Write("source/model.onnx", "asset");

  PkgHandle saved_package = MakeAuthoredPkgAt(s.path("saved"));
  const char* saved_uri = nullptr;
  CHECK_OK(ModelPackage_AddSharedAsset(saved_package.get(), s.path("source").u8string().c_str(),
                                       nullptr, true, &saved_uri));
  const std::string saved_dir_name = "sha256-" + std::string(saved_uri).substr(7);
  fs::create_directories(s.path("saved_work"));
  {
    ScopedCurrentPath current_path(s.path("saved_work"));
    CHECK_OK(ModelPackage_Commit(saved_package.get(), "saved/",
                                 MODEL_PACKAGE_WRITE_PRESERVE));
    CHECK(fs::is_directory(fs::path("saved/shared_assets") / saved_dir_name));
  }

  PkgHandle dot_package = MakeAuthoredPkgAt(s.path("dot"));
  const char* dot_uri = nullptr;
  CHECK_OK(ModelPackage_AddSharedAsset(dot_package.get(), s.path("source").u8string().c_str(),
                                       nullptr, true, &dot_uri));
  const std::string dot_dir_name = "sha256-" + std::string(dot_uri).substr(7);
  fs::create_directories(s.path("dot"));
  {
    ScopedCurrentPath current_path(s.path("dot"));
    CHECK_OK(ModelPackage_Commit(dot_package.get(), ".",
                                 MODEL_PACKAGE_WRITE_PRESERVE));
    CHECK(fs::is_directory(fs::path("shared_assets") / dot_dir_name));
  }
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
  CHECK_OK(ModelPackage_AddSharedAsset(p.get(), s.path("src_asset").u8string().c_str(),
                                       nullptr, /*copy_in=*/true, &uri));
  std::string uri_copy(uri);
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("pkg").u8string().c_str(),
                               MODEL_PACKAGE_WRITE_PRESERVE));
  std::string hex = uri_copy.substr(7);
  CHECK(fs::is_directory(s.path("pkg") / "shared_assets" / ("sha256-" + hex)));
  // Same on dest_root path.
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("saved").u8string().c_str(),
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
  CHECK_OK(ModelPackage_Commit(p.get(), s.path("pkg").u8string().c_str(),
                               MODEL_PACKAGE_WRITE_PRESERVE));
  const char* uri = nullptr;
  CHECK_OK(ModelPackage_AddSharedAsset(p.get(), s.path("src_asset").u8string().c_str(),
                                       nullptr, true, &uri));
  (void)uri;
  CHECK_OK(ModelPackage_SetComponentExternal(p.get(), "decoder", "decoder.json"));
  CHECK_OK(ModelPackage_Commit(p.get(), nullptr,
                               MODEL_PACKAGE_WRITE_PRESERVE));
  std::error_code ec;
  for (auto& e : fs::recursive_directory_iterator(s.path("pkg"), ec)) {
    if (e.path().filename().u8string().find(".tmp.") != std::string::npos) {
      std::fprintf(stderr, "  stray temp file: %s\n", e.path().u8string().c_str());
      return false;
    }
  }
  return true;
}

bool test_commit_unicode_paths_and_assets() {
  Sandbox s;
  const std::string directory = u8"\u6a21\u578b_\U0001F9EA";
  const std::string asset_directory = u8"\u6743\u91cd";
  const std::string filename = "weights.bin";
  s.Write(asset_directory + "/" + filename, "weights");
  const auto root = s.path(directory);
  PkgHandle package = MakeAuthoredPkgAt(root);
  const char* uri = nullptr;
  CHECK_OK(ModelPackage_AddSharedAsset(package.get(), s.path(asset_directory).u8string().c_str(),
                                       nullptr, true, &uri));
  const std::string asset_uri(uri);
  CHECK_OK(ModelPackage_Commit(package.get(), root.u8string().c_str(), MODEL_PACKAGE_WRITE_PRESERVE));
  ModelPackage* reopened = nullptr;
  CHECK_OK(ModelPackage_Open(root.u8string().c_str(), nullptr, &reopened));
  PkgHandle loaded(reopened);
  const char* resolved = nullptr;
  const std::string reference = asset_uri + "/" + filename;
  CHECK_OK(ModelPackage_ResolveStringRef(loaded.get(), nullptr, reference.c_str(), true, &resolved));
  CHECK(fs::is_regular_file(fs::u8path(resolved)));
  const auto* info = ModelPackage_Info(loaded.get());
  CHECK(info->num_shared_assets == 1);
  const char* rehashed = nullptr;
  CHECK_OK(ModelPackage_ComputeDirectoryHash(info->shared_assets[0].resolved_path, &rehashed));
  CHECK(std::string(rehashed) == asset_uri);
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
    {"commit_shared_asset_copy_in_nested_directory", test_commit_shared_asset_copy_in_nested_directory},
    {"commit_reports_unreadable_copy_in_subdirectory", test_commit_reports_unreadable_copy_in_subdirectory},
    {"commit_dense_inlines_external_component", test_commit_dense_inlines_external_component},
    {"commit_dense_rejects_external_executor_info", test_commit_dense_rejects_external_executor_info},
    {"commit_dest_root_self_contained", test_commit_dest_root_self_contained},
    {"commit_dest_root_must_be_empty", test_commit_dest_root_must_be_empty},
    {"commit_dest_root_rehashes_existing_asset", test_commit_dest_root_rehashes_existing_asset},
    {"prune_never_touches_shared_assets", test_prune_never_touches_shared_assets},
    {"prune_reclaims_tracked_orphan_variant_dirs", test_prune_reclaims_tracked_orphan_variant_dirs},
    {"prune_removes_stale_staging_dirs", test_prune_removes_stale_staging_dirs},
    {"prune_preserves_linked_shared_assets_entry", test_prune_preserves_linked_shared_assets_entry},
#ifndef _WIN32
    {"prune_reports_unreadable_shared_assets_directory", test_prune_reports_unreadable_shared_assets_directory},
#endif
    {"prune_skips_stale_staging_links", test_prune_skips_stale_staging_links},
#ifdef _WIN32
    {"windows_junction_rejected_by_public_apis", test_windows_junction_rejected_by_public_apis},
#endif
    {"open_rejects_linked_shared_asset_paths", test_open_rejects_linked_shared_asset_paths},
    {"shared_asset_mutations_reject_replaced_parent", test_shared_asset_mutations_reject_replaced_parent},
    {"dest_commit_rejects_replaced_asset_link", test_dest_commit_rejects_replaced_asset_link},
    {"linked_source_root_behavior", test_linked_source_root_behavior},
    {"source_path_preserves_link_ancestor_parent_traversal", test_source_path_preserves_link_ancestor_parent_traversal},
    {"failed_shared_asset_refresh_preserves_live_records", test_failed_shared_asset_refresh_preserves_live_records},
    {"open_rejects_linked_manifest_override", test_open_rejects_linked_manifest_override},
    {"commit_accepts_relative_destination_roots", test_commit_accepts_relative_destination_roots},
    {"commit_accepts_unreferenced_shared_asset", test_commit_accepts_unreferenced_shared_asset},
    {"commit_leaves_no_temp_files", test_commit_leaves_no_temp_files},
    {"commit_unicode_paths_and_assets", test_commit_unicode_paths_and_assets},
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

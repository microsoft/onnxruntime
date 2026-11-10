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
#include <iterator>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#ifndef _WIN32
#include <sys/stat.h>
#endif

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

bool test_chain_commit_modes_and_reference_failures() {
  using Json = nlohmann::ordered_json;
  for (const char* layout : {"portable", "installed"}) {
    for (const auto mode : {MODEL_PACKAGE_WRITE_PRESERVE, MODEL_PACKAGE_WRITE_DENSE}) {
      Sandbox s;
      Json manifest = Json::parse(R"({
        "schema_version":"1.2",
        "components":{
          "decoder_prefill":"prefill",
          "decoder_decode":{"variants":{"cpu":{"variant_directory":"decode/cpu","executor_info":{
            "ort":{"model_file":"model2.onnx"}
          }}}}
        },
        "future_manifest":{"keep":true},
        "chains":{"generate":{
          "future_chain":1,
          "steps":[
            {"id":"prefill","component":"decoder_prefill","executor_info":{"ort":{
              "kind":"session_loop","loop":{"exit_conditions":[{"kind":"end_of_inputs"}]}
            }}},
            {"id":"decode","component":"decoder_decode","future_step":2,"executor_info":{"ort":{
              "kind":"session_loop","loop":{"generation":{"initial_logits":{
                "kind":"model_output","step":"prefill","output":"logits"
              }},"sampler":{"algorithm":"greedy"}}
            }}}
          ],
          "executor_info":{"foreign":{"keep":[true,null,42]}}
        }}
      })");
      manifest["layout"] = layout;
      const Json component = {
          {"variants",
           {{"cpu",
             {{"variant_directory", "cpu"},
              {"executor_info",
               {{"ort",
                 mode == MODEL_PACKAGE_WRITE_PRESERVE ? Json("ort.json") : Json{{"model_file", "model1.onnx"}}}}}}}}}};
      s.Write("source/manifest.json", manifest.dump());
      s.Write("source/prefill/component.json", component.dump());
      s.Write("source/prefill/cpu/model1.onnx", "prefill model fixture");
      s.Write("source/prefill/cpu/ort.json", R"({"model_file":"model1.onnx"})");
      s.Write("source/prefill/cpu/nested/weights.bin", std::string("a\0b", 3));
      fs::create_directories(s.path("source/prefill/cpu/empty"));
      s.Write("source/decode/cpu/model2.onnx", "decode model fixture");
      ModelPackage* raw = nullptr;
      CHECK_OK(ModelPackage_Open(s.path("source").u8string().c_str(), nullptr, &raw));
      PkgHandle p(raw);
      const std::string destination = s.path("destination").u8string();
      CHECK_OK(ModelPackage_Commit(raw, destination.c_str(), mode));
      fs::rename(s.path("source"), s.path("unavailable"));
      CHECK_OK(ModelPackage_SetMetadata(raw, "updated", nullptr, nullptr));
      CHECK_OK(ModelPackage_Commit(raw, nullptr, mode));
      ModelPackage* reopened = nullptr;
      CHECK_OK(ModelPackage_Open(destination.c_str(), nullptr, &reopened));
      PkgHandle other(reopened);
      const char* json = nullptr;
      CHECK_OK(ModelPackage_GetChainsJson(reopened, &json));
      CHECK(Json::parse(json) == manifest["chains"]);
      CHECK(ModelPackage_Info(reopened)->schema_version_minor == 2);
      CHECK_ERR(ModelPackage_RemoveComponent(reopened, "decoder_prefill"), MODEL_PACKAGE_ERR_SCHEMA);
      CHECK_OK(ModelPackage_Prune(reopened));
      const auto* info = ModelPackage_Info(reopened);
      const auto* prefill = ModelComponentInfo_FindVariant(ModelPackage_FindComponent(info, "decoder_prefill"), "cpu");
      const auto* decode = ModelComponentInfo_FindVariant(ModelPackage_FindComponent(info, "decoder_decode"), "cpu");
      CHECK(fs::exists(fs::u8path(prefill->variant_directory) / "model1.onnx"));
      CHECK(fs::exists(fs::u8path(decode->variant_directory) / "model2.onnx"));
      CHECK(fs::is_directory(fs::u8path(prefill->variant_directory) / "empty"));
      std::ifstream weights(fs::u8path(prefill->variant_directory) / "nested/weights.bin", std::ios::binary);
      CHECK(std::string(std::istreambuf_iterator<char>(weights), {}) == std::string("a\0b", 3));
      std::ifstream committed(s.path("destination/manifest.json"));
      Json saved;
      committed >> saved;
      CHECK(saved["future_manifest"] == manifest["future_manifest"]);
      CHECK_OK(ModelPackage_RemoveChain(reopened, "generate"));
      CHECK_OK(ModelPackage_Commit(reopened, nullptr, mode));
      std::ifstream removed(s.path("destination/manifest.json"));
      removed >> saved;
      CHECK(!saved.contains("chains"));
      CHECK(saved["schema_version"] == "1.2");
    }
  }
  return true;
}

bool test_export_installed_external_files_and_reexport() {
  using Json = nlohmann::ordered_json;
  Sandbox s;
  const Json payload = {{"model_file", "nested/model.onnx"}, {"foreign", {1, true, nullptr}}};
  s.Write("external/models/nested/model.onnx", std::string("model\0bytes", 11));
  s.Write("external/models/nested/weights.data", "external tensor bytes");
  fs::create_directories(s.path("external/models/empty"));
  s.Write("external/config.json", payload.dump());
  const Json component = {{"variants",
                           {{"cpu",
                             {{"variant_directory", s.path("external/models").u8string()},
                              {"executor_info",
                               {{"remove_first", s.path("external/config.json").u8string()},
                                {"ort", s.path("external/config.json").u8string()}}}}}}}};
  s.Write("external/component.json", component.dump());
  const Json manifest = {{"schema_version", "1.0"},
                         {"layout", "installed"},
                         {"components", {{"decoder", s.path("external/component.json").u8string()}}}};
  s.Write("source/manifest.json", manifest.dump());
  ModelPackage* raw = nullptr;
  CHECK_OK(ModelPackage_Open(s.path("source").u8string().c_str(), nullptr, &raw));
  PkgHandle p(raw);
  CHECK_OK(ModelPackage_Commit(raw, s.path("first").u8string().c_str(), MODEL_PACKAGE_WRITE_PRESERVE));
  CHECK_OK(ModelPackage_RemoveVariantExecutorInfo(raw, "decoder", "cpu", "remove_first"));
  CHECK_OK(ModelPackage_Commit(raw, s.path("second").u8string().c_str(), MODEL_PACKAGE_WRITE_PRESERVE));
  fs::rename(s.path("source"), s.path("source_unavailable"));
  fs::rename(s.path("external"), s.path("external_unavailable"));
  fs::rename(s.path("first"), s.path("first_unavailable"));
  ModelPackage* reopened = nullptr;
  CHECK_OK(ModelPackage_Open(s.path("second").u8string().c_str(), nullptr, &reopened));
  PkgHandle other(reopened);
  const auto* info = ModelPackage_Info(reopened);
  CHECK(info->schema_version_minor == 0);
  const auto* variant = ModelComponentInfo_FindVariant(ModelPackage_FindComponent(info, "decoder"), "cpu");
  const auto* entry = ModelVariantInfo_FindExecutorInfo(variant, "ort");
  CHECK(entry != nullptr);
  CHECK(Json::parse(entry->json) == payload);
  const fs::path directory = fs::u8path(variant->variant_directory);
  CHECK(fs::is_directory(directory / "empty"));
  std::ifstream model(directory / "nested/model.onnx", std::ios::binary);
  CHECK(std::string(std::istreambuf_iterator<char>(model), {}) == std::string("model\0bytes", 11));
  std::ifstream data(directory / "nested/weights.data");
  CHECK(std::string(std::istreambuf_iterator<char>(data), {}) == "external tensor bytes");
  CHECK_OK(ModelPackage_Commit(reopened, nullptr, MODEL_PACKAGE_WRITE_PRESERVE));
  return true;
}

bool test_dense_inplace_keeps_variant_anchor() {
  using Json = nlohmann::ordered_json;
  for (const char* layout : {"portable", "installed"}) {
    Sandbox s;
    const bool installed = std::string(layout) == "installed";
    s.Write(installed ? "external/model.onnx" : "source/decoder/cpu/model.onnx", "model bytes");
    const Json component = {{"variants",
                             {{"cpu",
                               {{"variant_directory", installed ? s.path("external").u8string() : "cpu"},
                                {"executor_info", {{"ort", {{"model_file", "model.onnx"}}}}}}}}}};
    s.Write("source/decoder/component.json", component.dump());
    const Json manifest = {{"schema_version", "1.0"}, {"layout", layout}, {"components", {{"decoder", "decoder"}}}};
    s.Write("source/manifest.json", manifest.dump());
    ModelPackage* raw = nullptr;
    CHECK_OK(ModelPackage_Open(s.path("source").u8string().c_str(), nullptr, &raw));
    PkgHandle p(raw);
    CHECK_OK(ModelPackage_Commit(raw, nullptr, MODEL_PACKAGE_WRITE_DENSE));
    ModelPackage* reopened = nullptr;
    CHECK_OK(ModelPackage_Open(s.path("source").u8string().c_str(), nullptr, &reopened));
    PkgHandle other(reopened);
    const auto* variant =
        ModelComponentInfo_FindVariant(ModelPackage_FindComponent(ModelPackage_Info(reopened), "decoder"), "cpu");
    CHECK(fs::exists(fs::u8path(variant->variant_directory) / "model.onnx"));
    CHECK_OK(ModelPackage_Commit(raw, nullptr, MODEL_PACKAGE_WRITE_DENSE));
  }
  return true;
}

bool test_export_rejects_external_source_overlap() {
  using Json = nlohmann::ordered_json;
  Sandbox s;
  s.Write("external/model.onnx", "model bytes");
  const Json manifest = {
      {"schema_version", "1.0"},
      {"layout", "installed"},
      {"components", {{"decoder", {{"variants", {{"cpu", {{"variant_directory", s.path("external").u8string()}}}}}}}}}};
  s.Write("source/manifest.json", manifest.dump());
  ModelPackage* raw = nullptr;
  CHECK_OK(ModelPackage_Open(s.path("source").u8string().c_str(), nullptr, &raw));
  PkgHandle p(raw);
  CHECK_ERR(ModelPackage_Commit(raw, s.path("external/destination").u8string().c_str(), MODEL_PACKAGE_WRITE_PRESERVE),
            MODEL_PACKAGE_ERR_PATH_CONFINEMENT);
  CHECK(!fs::exists(s.path("external/destination")));
  ModelPackage* authored = nullptr;
  CHECK_OK(ModelPackage_New(&authored));
  PkgHandle other(authored);
  const char* uri = nullptr;
  CHECK_OK(ModelPackage_AddSharedAsset(authored, s.path("external").u8string().c_str(), nullptr, true, &uri));
  CHECK_ERR(
      ModelPackage_Commit(authored, s.path("external/destination").u8string().c_str(), MODEL_PACKAGE_WRITE_PRESERVE),
      MODEL_PACKAGE_ERR_PATH_CONFINEMENT);
  CHECK(!fs::exists(s.path("external/destination")));
  return true;
}

bool test_export_failures_preserve_live_state() {
  using Json = nlohmann::ordered_json;
  const std::vector<std::string> cases = {
      "missing",
      "not_directory",
      "metadata_missing",
      "metadata_collision",
      "overlap",
      "io",
#ifndef _WIN32
      "symlink",
      "root_symlink",
      "fifo",
#endif
  };
  for (const auto& name : cases) {
    Sandbox s;
    s.Write("source/cpu/model.onnx", "source bytes");
    s.Write("source/cpu/ort.json", R"({"model_file":"model.onnx"})");
    s.Write("source/manifest.json", R"({
      "schema_version":"1.1",
      "components":{"decoder":{"variants":{"cpu":{
        "variant_directory":"cpu","executor_info":{"ort":"ort.json"}
      }}}},
      "chains":{"generate":{"steps":[{"id":"run","component":"decoder"}]}}
    })");
#ifndef _WIN32
    if (name == "root_symlink") {
      fs::rename(s.path("source/cpu"), s.path("source/original_cpu"));
      fs::create_directory_symlink(s.path("source/original_cpu"), s.path("source/cpu"));
    }
#endif
    ModelPackage* raw = nullptr;
    CHECK_OK(ModelPackage_Open(s.path("source").u8string().c_str(), nullptr, &raw));
    PkgHandle p(raw);
    auto destination = s.path("destination");
    auto code = MODEL_PACKAGE_ERR_SCHEMA;
    if (name == "missing" || name == "not_directory") {
      fs::rename(s.path("source/cpu"), s.path("removed_cpu"));
      if (name == "not_directory") s.Write("source/cpu", "not a directory");
      code = MODEL_PACKAGE_ERR_NOT_FOUND;
    } else if (name == "metadata_missing") {
      CHECK_OK(ModelPackage_SetVariantExecutorInfoExternal(raw, "decoder", "cpu", "ort", "missing.json"));
      code = MODEL_PACKAGE_ERR_NOT_FOUND;
    } else if (name == "metadata_collision") {
      s.Write("source/cpu/.model_package_executor_info_0.json", "do not overwrite");
      s.Write("external_info.json", "{}");
      CHECK_OK(ModelPackage_SetLayout(raw, "installed"));
      CHECK_OK(ModelPackage_SetVariantExecutorInfoExternal(raw, "decoder", "cpu", "ort",
                                                           s.path("external_info.json").u8string().c_str()));
      code = MODEL_PACKAGE_ERR_STATE;
    } else if (name == "overlap") {
      destination = s.path("source/nested_destination");
      code = MODEL_PACKAGE_ERR_PATH_CONFINEMENT;
    } else if (name == "io") {
      s.Write("blocker", "not a directory");
      destination = s.path("blocker/destination");
      code = MODEL_PACKAGE_ERR_IO;
    }
#ifndef _WIN32
    else if (name == "symlink") {
      fs::create_symlink(s.path("source/cpu/model.onnx"), s.path("source/cpu/link"));
    } else if (name == "fifo") {
      CHECK(::mkfifo(s.path("source/cpu/pipe").c_str(), 0600) == 0);
    }
#endif
    const auto* info = ModelPackage_Info(raw);
    const char* json = nullptr;
    CHECK_OK(ModelPackage_GetChainsJson(raw, &json));
    const std::string snapshot(json);
    std::ifstream before_manifest(s.path("source/manifest.json"), std::ios::binary);
    const std::string source_manifest(std::istreambuf_iterator<char>(before_manifest), {});
    CHECK_ERR(ModelPackage_Commit(raw, destination.u8string().c_str(), MODEL_PACKAGE_WRITE_PRESERVE), code);
    std::ifstream after_manifest(s.path("source/manifest.json"), std::ios::binary);
    CHECK(std::string(std::istreambuf_iterator<char>(after_manifest), {}) == source_manifest);
    CHECK(ModelPackage_Info(raw) == info);
    const char* after = nullptr;
    CHECK_OK(ModelPackage_GetChainsJson(raw, &after));
    CHECK(after == json);
    CHECK(snapshot == after);
    const char* path = nullptr;
    CHECK_OK(ModelPackage_ResolveStringRef(raw, nullptr, "manifest.json", true, &path));
    CHECK(fs::u8path(path) == s.path("source/manifest.json"));
    if (name == "overlap") CHECK(!fs::exists(destination));
  }
  return true;
}

struct Test {
  const char* name;
  bool (*fn)();
};

const Test kTests[] = {
    {"export_rejects_external_source_overlap", test_export_rejects_external_source_overlap},
    {"export_installed_external_files_and_reexport", test_export_installed_external_files_and_reexport},
    {"dense_inplace_keeps_variant_anchor", test_dense_inplace_keeps_variant_anchor},
    {"export_failures_preserve_live_state", test_export_failures_preserve_live_state},
    {"chain_commit_modes_and_reference_failures", test_chain_commit_modes_and_reference_failures},
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

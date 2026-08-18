// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/// \file test_authoring.cc
/// \brief Authoring (mutation) API tests.

#include "model_package.h"
#include "model_package_api.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>

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
    std::snprintf(buf, sizeof(buf), "mp_auth_%016llx", static_cast<unsigned long long>(g()));
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

 private:
  ModelPackage* p_;
};

// ─────────────────────────────────────────────────────────────────────────────
// ModelPackage_New
// ─────────────────────────────────────────────────────────────────────────────

bool test_new_creates_empty_package() {
  ModelPackage* raw = nullptr;
  CHECK_OK(ModelPackage_New(&raw));
  CHECK(raw != nullptr);
  PkgHandle p(raw);
  const ModelPackageInfo* info = ModelPackage_Info(p.get());
  CHECK(info != nullptr);
  CHECK(info->schema_version_major == 0);
  CHECK(info->schema_version_minor == 0);
  CHECK((info)->num_components == 0);
  CHECK((info)->num_shared_assets == 0);
  CHECK(std::string(info->layout) == "portable");
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Component operations
// ─────────────────────────────────────────────────────────────────────────────

bool test_set_component_inline_basic() {
  ModelPackage* raw = nullptr;
  CHECK_OK(ModelPackage_New(&raw));
  PkgHandle p(raw);

  CHECK_OK(ModelPackage_SetComponentInline(p.get(), "encoder",
                                           R"({"variants": {}})"));
  CHECK((ModelPackage_Info(p.get()))->num_components == 1);
  const ModelComponentInfo* c = ModelPackage_FindComponent(ModelPackage_Info(p.get()), "encoder");
  CHECK(c != nullptr);
  CHECK(std::string(c->name) == "encoder");
  CHECK((c)->num_variants == 0);
  return true;
}

bool test_set_component_inline_replaces_existing() {
  ModelPackage* raw = nullptr;
  CHECK_OK(ModelPackage_New(&raw));
  PkgHandle p(raw);

  CHECK_OK(ModelPackage_SetComponentInline(p.get(), "c", R"({"variants": {}})"));
  CHECK_OK(ModelPackage_SetComponentInline(p.get(), "c",
                                           R"({"variants": {"v1": {"variant_directory": "."}}})"));
  CHECK((ModelPackage_Info(p.get()))->num_components == 1);
  const ModelComponentInfo* c = ModelPackage_FindComponent(ModelPackage_Info(p.get()), "c");
  CHECK((c)->num_variants == 1);
  return true;
}

bool test_set_component_inline_rejects_unknown_field() {
  ModelPackage* raw = nullptr;
  CHECK_OK(ModelPackage_New(&raw));
  PkgHandle p(raw);
  CHECK_ERR(ModelPackage_SetComponentInline(p.get(), "c",
                                            R"({"variants": {}, "typo_field": 1})"),
            MODEL_PACKAGE_ERR_SCHEMA);
  CHECK((ModelPackage_Info(p.get()))->num_components == 0);
  return true;
}

bool test_set_component_inline_rejects_bad_json() {
  ModelPackage* raw = nullptr;
  CHECK_OK(ModelPackage_New(&raw));
  PkgHandle p(raw);
  CHECK_ERR(ModelPackage_SetComponentInline(p.get(), "c", "not-json"),
            MODEL_PACKAGE_ERR_SCHEMA);
  return true;
}

bool test_remove_component() {
  ModelPackage* raw = nullptr;
  CHECK_OK(ModelPackage_New(&raw));
  PkgHandle p(raw);
  CHECK_OK(ModelPackage_SetComponentInline(p.get(), "a", R"({"variants": {}})"));
  CHECK_OK(ModelPackage_SetComponentInline(p.get(), "b", R"({"variants": {}})"));
  CHECK((ModelPackage_Info(p.get()))->num_components == 2);
  CHECK_OK(ModelPackage_RemoveComponent(p.get(), "a"));
  CHECK((ModelPackage_Info(p.get()))->num_components == 1);
  const ModelPackageInfo* info = ModelPackage_Info(p.get());
  CHECK(ModelPackage_FindComponent(info, "a") == nullptr);
  CHECK(ModelPackage_FindComponent(info, "b") != nullptr);
  return true;
}

bool test_remove_missing_component_is_noop() {
  ModelPackage* raw = nullptr;
  CHECK_OK(ModelPackage_New(&raw));
  PkgHandle p(raw);
  CHECK_OK(ModelPackage_RemoveComponent(p.get(), "nope"));
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Variant operations
// ─────────────────────────────────────────────────────────────────────────────

bool test_set_variant_upsert() {
  ModelPackage* raw = nullptr;
  CHECK_OK(ModelPackage_New(&raw));
  PkgHandle p(raw);
  CHECK_OK(ModelPackage_SetComponentInline(p.get(), "c", R"({"variants": {}})"));

  CHECK_OK(ModelPackage_SetVariant(p.get(), "c", "v1",
                                   R"({"variant_directory": ".", "ep": "CPU"})"));
  const ModelComponentInfo* c = ModelPackage_FindComponent(ModelPackage_Info(p.get()), "c");
  CHECK((c)->num_variants == 1);
  const ModelVariantInfo* v = ModelComponentInfo_FindVariant(c, "v1");
  CHECK(v != nullptr);
  CHECK(std::string(v->ep) == "CPU");

  // Upsert: change ep.
  CHECK_OK(ModelPackage_SetVariant(p.get(), "c", "v1",
                                   R"({"variant_directory": ".", "ep": "CUDA"})"));
  c = ModelPackage_FindComponent(ModelPackage_Info(p.get()), "c");
  CHECK((c)->num_variants == 1);
  v = ModelComponentInfo_FindVariant(c, "v1");
  CHECK(std::string(v->ep) == "CUDA");
  return true;
}

bool test_set_variant_unknown_component_errors() {
  ModelPackage* raw = nullptr;
  CHECK_OK(ModelPackage_New(&raw));
  PkgHandle p(raw);
  CHECK_ERR(ModelPackage_SetVariant(p.get(), "nope", "v1", R"({"variant_directory": "."})"),
            MODEL_PACKAGE_ERR_NOT_FOUND);
  return true;
}

bool test_remove_variant() {
  ModelPackage* raw = nullptr;
  CHECK_OK(ModelPackage_New(&raw));
  PkgHandle p(raw);
  CHECK_OK(ModelPackage_SetComponentInline(p.get(), "c", R"({"variants": {}})"));
  CHECK_OK(ModelPackage_SetVariant(p.get(), "c", "v1", R"({"variant_directory": "."})"));
  CHECK_OK(ModelPackage_RemoveVariant(p.get(), "c", "v1"));
  const ModelComponentInfo* c = ModelPackage_FindComponent(ModelPackage_Info(p.get()), "c");
  CHECK((c)->num_variants == 0);
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Variant executor_info
// ─────────────────────────────────────────────────────────────────────────────

bool test_set_executor_info_inline_and_remove() {
  ModelPackage* raw = nullptr;
  CHECK_OK(ModelPackage_New(&raw));
  PkgHandle p(raw);
  CHECK_OK(ModelPackage_SetComponentInline(p.get(), "c", R"({"variants": {}})"));
  CHECK_OK(ModelPackage_SetVariant(p.get(), "c", "v1", R"({"variant_directory": "."})"));

  CHECK_OK(ModelPackage_SetVariantExecutorInfoInline(p.get(), "c", "v1", "ort",
                                                     R"({"model": "m.onnx"})"));
  const ModelVariantInfo* v = ModelComponentInfo_FindVariant(
      ModelPackage_FindComponent(ModelPackage_Info(p.get()), "c"), "v1");
  const char* ej = nullptr;
  const ModelExecutorInfoEntry* ei = ModelVariantInfo_FindExecutorInfo(v, "ort");
  ej = ei ? ei->json : nullptr;
  CHECK(ej != nullptr);
  CHECK(std::strstr(ej, "\"model\"") != nullptr);

  CHECK_OK(ModelPackage_RemoveVariantExecutorInfo(p.get(), "c", "v1", "ort"));
  v = ModelComponentInfo_FindVariant(ModelPackage_FindComponent(ModelPackage_Info(p.get()), "c"), "v1");
  ei = ModelVariantInfo_FindExecutorInfo(v, "ort");
  ej = ei ? ei->json : nullptr;
  CHECK(ei == nullptr);
  CHECK(ej == nullptr);
  return true;
}

bool test_set_executor_info_external_records_path() {
  ModelPackage* raw = nullptr;
  CHECK_OK(ModelPackage_New(&raw));
  PkgHandle p(raw);
  CHECK_OK(ModelPackage_SetComponentInline(p.get(), "c", R"({"variants": {}})"));
  CHECK_OK(ModelPackage_SetVariant(p.get(), "c", "v1", R"({"variant_directory": "."})"));
  CHECK_OK(ModelPackage_SetVariantExecutorInfoExternal(p.get(), "c", "v1", "ort",
                                                       "ort_info.json"));
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Package metadata
// ─────────────────────────────────────────────────────────────────────────────

bool test_set_metadata() {
  ModelPackage* raw = nullptr;
  CHECK_OK(ModelPackage_New(&raw));
  PkgHandle p(raw);
  CHECK_OK(ModelPackage_SetMetadata(p.get(), "mypkg", "1.0.0", "desc"));
  const ModelPackageInfo* info = ModelPackage_Info(p.get());
  CHECK(std::string(info->package_name) == "mypkg");
  CHECK(std::string(info->package_version) == "1.0.0");
  CHECK(std::string(info->description) == "desc");

  // Empty string clears.
  CHECK_OK(ModelPackage_SetMetadata(p.get(), nullptr, "", nullptr));
  info = ModelPackage_Info(p.get());
  CHECK(info->package_version == nullptr);
  CHECK(std::string(info->package_name) == "mypkg");
  return true;
}

bool test_set_layout() {
  ModelPackage* raw = nullptr;
  CHECK_OK(ModelPackage_New(&raw));
  PkgHandle p(raw);
  CHECK_OK(ModelPackage_SetLayout(p.get(), "installed"));
  CHECK(std::string(ModelPackage_Info(p.get())->layout) == "installed");
  CHECK_ERR(ModelPackage_SetLayout(p.get(), "weird"), MODEL_PACKAGE_ERR_SCHEMA);
  return true;
}

bool test_set_additional_metadata_manifest_scope() {
  ModelPackage* raw = nullptr;
  CHECK_OK(ModelPackage_New(&raw));
  PkgHandle p(raw);
  CHECK_OK(ModelPackage_SetAdditionalMetadataJson(p.get(), "manifest", nullptr, nullptr,
                                                  R"({"author":"jambayk"})"));
  const ModelPackageInfo* info = ModelPackage_Info(p.get());
  CHECK(info->additional_metadata_json != nullptr);
  CHECK(std::string(info->additional_metadata_json).find("jambayk") != std::string::npos);

  // Clear.
  CHECK_OK(ModelPackage_SetAdditionalMetadataJson(p.get(), "manifest", nullptr, nullptr, nullptr));
  info = ModelPackage_Info(p.get());
  CHECK(info->additional_metadata_json == nullptr);
  return true;
}

bool test_set_additional_metadata_variant_scope() {
  ModelPackage* raw = nullptr;
  CHECK_OK(ModelPackage_New(&raw));
  PkgHandle p(raw);
  CHECK_OK(ModelPackage_SetComponentInline(p.get(), "c", R"({"variants": {}})"));
  CHECK_OK(ModelPackage_SetVariant(p.get(), "c", "v1", R"({"variant_directory": "."})"));
  CHECK_OK(ModelPackage_SetAdditionalMetadataJson(p.get(), "variant", "c", "v1",
                                                  R"({"foo":"bar"})"));
  const ModelVariantInfo* v = ModelComponentInfo_FindVariant(
      ModelPackage_FindComponent(ModelPackage_Info(p.get()), "c"), "v1");
  CHECK(v != nullptr);
  const char* md = v->additional_metadata_json;
  CHECK(md != nullptr);
  CHECK(std::string(md).find("foo") != std::string::npos);
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared assets — authoring
// ─────────────────────────────────────────────────────────────────────────────

bool test_add_shared_asset_copy_in_true_portable_ok() {
  Sandbox s;
  s.Write("src/a.txt", "alpha");

  ModelPackage* raw = nullptr;
  CHECK_OK(ModelPackage_New(&raw));
  PkgHandle p(raw);
  const char* uri = nullptr;
  CHECK_OK(ModelPackage_AddSharedAsset(p.get(), (s.root() / "src").c_str(),
                                       nullptr, /*copy_in=*/true, &uri));
  CHECK(uri != nullptr);
  CHECK(std::string(uri).substr(0, 7) == "sha256:");
  return true;
}

bool test_add_shared_asset_copy_in_false_portable_rejected() {
  Sandbox s;
  s.Write("src/a.txt", "alpha");

  ModelPackage* raw = nullptr;
  CHECK_OK(ModelPackage_New(&raw));
  PkgHandle p(raw);
  const char* uri = nullptr;
  CHECK_ERR(ModelPackage_AddSharedAsset(p.get(), (s.root() / "src").c_str(),
                                        nullptr, /*copy_in=*/false, &uri),
            MODEL_PACKAGE_ERR_STATE);
  return true;
}

bool test_add_shared_asset_copy_in_false_installed_ok() {
  Sandbox s;
  s.Write("src/a.txt", "alpha");

  ModelPackage* raw = nullptr;
  CHECK_OK(ModelPackage_New(&raw));
  PkgHandle p(raw);
  CHECK_OK(ModelPackage_SetLayout(p.get(), "installed"));
  const char* uri = nullptr;
  CHECK_OK(ModelPackage_AddSharedAsset(p.get(), (s.root() / "src").c_str(),
                                       nullptr, /*copy_in=*/false, &uri));
  CHECK(uri != nullptr);
  // Surfaced as a manifest override -> shared_assets count should be 1.
  CHECK((ModelPackage_Info(p.get()))->num_shared_assets == 1);
  return true;
}

bool test_add_shared_asset_expected_uri_mismatch_errors() {
  Sandbox s;
  s.Write("src/a.txt", "alpha");

  ModelPackage* raw = nullptr;
  CHECK_OK(ModelPackage_New(&raw));
  PkgHandle p(raw);
  CHECK_OK(ModelPackage_SetLayout(p.get(), "installed"));
  const char* uri = nullptr;
  std::string bogus = "sha256:" + std::string(64, '0');
  CHECK_ERR(ModelPackage_AddSharedAsset(p.get(), (s.root() / "src").c_str(),
                                        bogus.c_str(), /*copy_in=*/false, &uri),
            MODEL_PACKAGE_ERR_STATE);
  return true;
}

bool test_remove_shared_asset() {
  Sandbox s;
  s.Write("src/a.txt", "alpha");

  ModelPackage* raw = nullptr;
  CHECK_OK(ModelPackage_New(&raw));
  PkgHandle p(raw);
  CHECK_OK(ModelPackage_SetLayout(p.get(), "installed"));
  const char* uri = nullptr;
  CHECK_OK(ModelPackage_AddSharedAsset(p.get(), (s.root() / "src").c_str(),
                                       nullptr, /*copy_in=*/false, &uri));
  std::string uri_copy(uri);
  CHECK((ModelPackage_Info(p.get()))->num_shared_assets == 1);
  CHECK_OK(ModelPackage_RemoveSharedAsset(p.get(), uri_copy.c_str()));
  CHECK((ModelPackage_Info(p.get()))->num_shared_assets == 0);
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Round-trip through GetComponentJson / GetVariantJson
// ─────────────────────────────────────────────────────────────────────────────

bool test_round_trip_component_json() {
  ModelPackage* raw = nullptr;
  CHECK_OK(ModelPackage_New(&raw));
  PkgHandle p(raw);
  CHECK_OK(ModelPackage_SetComponentInline(p.get(), "c",
                                           R"({"variants": {"v1": {"variant_directory": ".", "ep": "CPU"}}})"));
  const char* j = nullptr;
  CHECK_OK(ModelPackage_GetComponentJson(p.get(), "c", &j));
  CHECK(j != nullptr);
  std::string s(j);
  CHECK(s.find("\"variants\"") != std::string::npos);
  CHECK(s.find("\"v1\"") != std::string::npos);
  CHECK(s.find("\"CPU\"") != std::string::npos);
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// View cache invalidation after mutation
// ─────────────────────────────────────────────────────────────────────────────

bool test_view_cache_drops_on_remove() {
  ModelPackage* raw = nullptr;
  CHECK_OK(ModelPackage_New(&raw));
  PkgHandle p(raw);
  CHECK_OK(ModelPackage_SetComponentInline(p.get(), "a", R"({"variants": {}})"));
  CHECK_OK(ModelPackage_SetComponentInline(p.get(), "b", R"({"variants": {}})"));
  const ModelComponentInfo* a = ModelPackage_FindComponent(ModelPackage_Info(p.get()), "a");
  CHECK(a != nullptr);
  CHECK_OK(ModelPackage_RemoveComponent(p.get(), "a"));
  // Old pointer was invalidated by the mutation; re-fetch and 'a' must now be gone.
  const ModelPackageInfo* info = ModelPackage_Info(p.get());
  CHECK(ModelPackage_FindComponent(info, "a") == nullptr);
  CHECK(ModelPackage_FindComponent(info, "b") != nullptr);
  return true;
}

struct Test {
  const char* name;
  bool (*fn)();
};

const Test kTests[] = {
    {"new_creates_empty_package", test_new_creates_empty_package},
    {"set_component_inline_basic", test_set_component_inline_basic},
    {"set_component_inline_replaces_existing", test_set_component_inline_replaces_existing},
    {"set_component_inline_rejects_unknown_field", test_set_component_inline_rejects_unknown_field},
    {"set_component_inline_rejects_bad_json", test_set_component_inline_rejects_bad_json},
    {"remove_component", test_remove_component},
    {"remove_missing_component_is_noop", test_remove_missing_component_is_noop},
    {"set_variant_upsert", test_set_variant_upsert},
    {"set_variant_unknown_component_errors", test_set_variant_unknown_component_errors},
    {"remove_variant", test_remove_variant},
    {"set_executor_info_inline_and_remove", test_set_executor_info_inline_and_remove},
    {"set_executor_info_external_records_path", test_set_executor_info_external_records_path},
    {"set_metadata", test_set_metadata},
    {"set_layout", test_set_layout},
    {"set_additional_metadata_manifest_scope", test_set_additional_metadata_manifest_scope},
    {"set_additional_metadata_variant_scope", test_set_additional_metadata_variant_scope},
    {"add_shared_asset_copy_in_true_portable_ok", test_add_shared_asset_copy_in_true_portable_ok},
    {"add_shared_asset_copy_in_false_portable_rejected", test_add_shared_asset_copy_in_false_portable_rejected},
    {"add_shared_asset_copy_in_false_installed_ok", test_add_shared_asset_copy_in_false_installed_ok},
    {"add_shared_asset_expected_uri_mismatch_errors", test_add_shared_asset_expected_uri_mismatch_errors},
    {"remove_shared_asset", test_remove_shared_asset},
    {"round_trip_component_json", test_round_trip_component_json},
    {"view_cache_drops_on_remove", test_view_cache_drops_on_remove},
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

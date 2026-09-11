// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/// \file test_inspection.cc
/// \brief Tests for the read-only inspection API (model_package.h).

#include "model_package.h"
#include "model_package_api.h"

#include <cstdio>
#include <cstdlib>
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

#define CHECK_ERR(status, expected_code)                                         \
  do {                                                                           \
    ModelPackageStatus* _s = (status);                                           \
    if (_s == nullptr) {                                                         \
      std::fprintf(stderr, "[FAIL] %s line %d: expected error %d, got OK\n",     \
                   g_current, __LINE__, (int)(expected_code));                   \
      return false;                                                              \
    }                                                                            \
    ModelPackageErrorCode _c = ModelPackageStatus_Code(_s);                      \
    if (_c != (expected_code)) {                                                 \
      std::fprintf(stderr, "[FAIL] %s line %d: expected error %d, got %d: %s\n", \
                   g_current, __LINE__, (int)(expected_code), (int)_c,           \
                   ModelPackageStatus_Message(_s));                              \
      ModelPackageStatus_Release(_s);                                            \
      return false;                                                              \
    }                                                                            \
    ModelPackageStatus_Release(_s);                                              \
  } while (0)

class Sandbox {
 public:
  Sandbox() {
    std::random_device rd;
    std::mt19937_64 g(rd());
    char buf[32];
    std::snprintf(buf, sizeof(buf), "mp_inspect_%016llx", static_cast<unsigned long long>(g()));
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

  void Write(const std::string& relpath, const std::string& contents) {
    fs::path full = root_ / relpath;
    fs::create_directories(full.parent_path());
    std::ofstream f(full, std::ios::binary);
    f << contents;
  }

  void Touch(const std::string& relpath) { Write(relpath, ""); }

 private:
  fs::path root_;
};

bool test_open_minimal_inline() {
  Sandbox s;
  s.Write("manifest.json", R"({
    "schema_version": 1,
    "package_name": "test",
    "components": {
      "alpha": {
        "variants": {
          "cpu": {}
        }
      }
    }
  })");

  ModelPackage* pkg = nullptr;
  CHECK_OK(ModelPackage_Open(s.root().c_str(), nullptr, &pkg));
  CHECK(pkg != nullptr);

  const ModelPackageInfo* info = ModelPackage_Info(pkg);
  CHECK(info != nullptr);
  CHECK(info->schema_version_major == 1);
  CHECK(info->schema_version_minor == 0);
  CHECK(std::string(info->package_name) == "test");
  CHECK(std::string(info->layout) == "portable");
  CHECK((info)->num_components == 1);
  CHECK((info)->num_shared_assets == 0);
  CHECK(info->additional_metadata_json == nullptr);

  const ModelComponentInfo* c = &(info)->components[0];
  CHECK(c != nullptr);
  CHECK(std::string(c->name) == "alpha");
  CHECK((c)->num_variants == 1);

  const ModelVariantInfo* v = &(c)->variants[0];
  CHECK(v != nullptr);
  CHECK(std::string(v->name) == "cpu");
  CHECK(v->ep == nullptr);
  CHECK(v->device == nullptr);
  CHECK(v->compatibility_string == nullptr);

  ModelPackage_Close(pkg);
  return true;
}

bool test_open_full_inline_with_metadata() {
  Sandbox s;
  s.Write("manifest.json", R"({
    "schema_version": 1,
    "package_name": "phi-4",
    "package_version": "1.2.3",
    "description": "demo",
    "layout": "portable",
    "additional_metadata": {"author": "team"},
    "components": {
      "decoder": {
        "additional_metadata": {"size": "small"},
        "variants": {
          "cuda_fp16": {
            "variant_directory": "decoder/cuda_fp16",
            "ep": "CUDAExecutionProvider",
            "device": "gpu",
            "compatibility_string": "sm_80",
            "additional_metadata": {"notes": "quantized"}
          }
        }
      }
    }
  })");
  fs::create_directories(s.root() / "decoder" / "cuda_fp16");

  ModelPackage* pkg = nullptr;
  CHECK_OK(ModelPackage_Open(s.root().c_str(), nullptr, &pkg));
  const ModelPackageInfo* info = ModelPackage_Info(pkg);
  CHECK(std::string(info->package_name) == "phi-4");
  CHECK(std::string(info->package_version) == "1.2.3");
  CHECK(std::string(info->description) == "demo");
  CHECK(info->additional_metadata_json != nullptr);
  CHECK(std::string(info->additional_metadata_json).find("\"author\":\"team\"") != std::string::npos);

  const ModelComponentInfo* c = ModelPackage_FindComponent(info, "decoder");
  CHECK(c != nullptr);
  const char* comp_meta = c->additional_metadata_json;
  CHECK(comp_meta != nullptr);
  CHECK(std::string(comp_meta).find("\"size\":\"small\"") != std::string::npos);

  const ModelVariantInfo* v = ModelComponentInfo_FindVariant(c, "cuda_fp16");
  CHECK(v != nullptr);
  CHECK(std::string(v->ep) == "CUDAExecutionProvider");
  CHECK(std::string(v->device) == "gpu");
  CHECK(std::string(v->compatibility_string) == "sm_80");
  const char* var_meta = v->additional_metadata_json;
  CHECK(var_meta != nullptr);
  CHECK(std::string(var_meta).find("\"notes\":\"quantized\"") != std::string::npos);

  const char* resolved = v->variant_directory;
  CHECK(resolved != nullptr);
  CHECK(std::string(resolved).find("decoder/cuda_fp16") != std::string::npos);

  ModelPackage_Close(pkg);
  return true;
}

bool test_external_component_file() {
  Sandbox s;
  s.Write("manifest.json", R"({
    "schema_version": 1,
    "components": { "decoder": "components/decoder.json" }
  })");
  s.Write("components/decoder.json", R"({
    "variants": { "cpu": {} }
  })");
  ModelPackage* pkg = nullptr;
  CHECK_OK(ModelPackage_Open(s.root().c_str(), nullptr, &pkg));
  const ModelComponentInfo* c = ModelPackage_FindComponent(ModelPackage_Info(pkg), "decoder");
  CHECK(c != nullptr);
  CHECK((c)->num_variants == 1);
  ModelPackage_Close(pkg);
  return true;
}

bool test_external_component_directory() {
  Sandbox s;
  s.Write("manifest.json", R"({
    "schema_version": 1,
    "components": { "decoder": "components/decoder" }
  })");
  s.Write("components/decoder/component.json", R"({
    "variants": { "cpu": {} }
  })");
  ModelPackage* pkg = nullptr;
  CHECK_OK(ModelPackage_Open(s.root().c_str(), nullptr, &pkg));
  CHECK((ModelPackage_Info(pkg))->num_components == 1);
  ModelPackage_Close(pkg);
  return true;
}

bool test_executor_info_inline_and_external() {
  Sandbox s;
  s.Write("manifest.json", R"({
    "schema_version": 1,
    "components": {
      "decoder": {
        "variants": {
          "cuda": {
            "variant_directory": "v",
            "executor_info": {
              "ort":   "ort_info.json",
              "other": {"x": 1}
            }
          }
        }
      }
    }
  })");
  fs::create_directories(s.root() / "v");
  s.Write("v/ort_info.json", R"({"model_file":"model.onnx"})");

  ModelPackage* pkg = nullptr;
  CHECK_OK(ModelPackage_Open(s.root().c_str(), nullptr, &pkg));
  const ModelPackageInfo* info = ModelPackage_Info(pkg);
  const ModelVariantInfo* v =
      ModelComponentInfo_FindVariant(ModelPackage_FindComponent(info, "decoder"), "cuda");
  CHECK(v != nullptr);

  const ModelExecutorInfoEntry* ort_ei = ModelVariantInfo_FindExecutorInfo(v, "ort");
  const char* ort_json = ort_ei ? ort_ei->json : nullptr;
  CHECK(ort_json != nullptr);
  CHECK(std::string(ort_json).find("model.onnx") != std::string::npos);

  const ModelExecutorInfoEntry* other_ei = ModelVariantInfo_FindExecutorInfo(v, "other");
  const char* other_json = other_ei ? other_ei->json : nullptr;
  CHECK(other_json != nullptr);
  CHECK(std::string(other_json).find("\"x\":1") != std::string::npos);

  const ModelExecutorInfoEntry* missing_ei = ModelVariantInfo_FindExecutorInfo(v, "absent");
  const char* missing = missing_ei ? missing_ei->json : nullptr;
  CHECK(missing_ei == nullptr);
  CHECK(missing == nullptr);

  ModelPackage_Close(pkg);
  return true;
}

bool test_inline_executor_info_without_directory_accepted() {
  // Library no longer requires variant_directory to exist for inline
  // executor_info. Executors interpret their own payload.
  Sandbox s;
  s.Write("manifest.json", R"({
    "schema_version": 1,
    "components": {
      "decoder": {
        "variants": {
          "cuda": {
            "executor_info": { "other": {"x": 1} }
          }
        }
      }
    }
  })");
  ModelPackage* pkg = nullptr;
  CHECK_OK(ModelPackage_Open(s.root().c_str(), nullptr, &pkg));
  ModelPackage_Close(pkg);
  return true;
}

bool test_path_confinement_rejects_external_paths() {
  Sandbox s;
  s.Write("manifest.json", R"({
    "schema_version": 1,
    "components": { "x": "../escape.json" }
  })");
  ModelPackage* pkg = nullptr;
  CHECK_ERR(ModelPackage_Open(s.root().c_str(), nullptr, &pkg), MODEL_PACKAGE_ERR_PATH_CONFINEMENT);
  return true;
}

bool test_installed_layout_allows_absolute() {
  // Build a package whose component lives outside its root.
  Sandbox external;
  external.Write("decoder.json", R"({"variants": {"cpu": {}}})");

  Sandbox s;
  std::string abs_comp = (external.root() / "decoder.json").string();
  // Escape backslashes for any platform that uses them — POSIX is fine as-is.
  s.Write("manifest.json", std::string(R"({
    "schema_version": 1,
    "layout": "installed",
    "components": {"decoder": ")") +
                               abs_comp + R"("}
  })");

  ModelPackage* pkg = nullptr;
  CHECK_OK(ModelPackage_Open(s.root().c_str(), nullptr, &pkg));
  CHECK((ModelPackage_Info(pkg))->num_components == 1);
  ModelPackage_Close(pkg);
  return true;
}

bool test_shared_assets_resolve() {
  Sandbox s;
  s.Write("manifest.json", R"({
    "schema_version": 1,
    "shared_assets": {
      "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa": "assets/a"
    },
    "components": {
      "x": {
        "variants": {
          "cpu": {}
        }
      }
    }
  })");
  fs::create_directories(s.root() / "assets" / "a");
  // Discovery: an on-disk sha256-<hex> dir without an override entry must
  // surface alongside the explicit override.
  fs::create_directories(
      s.root() / "shared_assets" /
      "sha256-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb");

  ModelPackage* pkg = nullptr;
  CHECK_OK(ModelPackage_Open(s.root().c_str(), nullptr, &pkg));
  CHECK((ModelPackage_Info(pkg))->num_shared_assets == 2);

  const ModelSharedAssetInfo* a = &(ModelPackage_Info(pkg))->shared_assets[0];
  CHECK(a != nullptr);
  CHECK(std::string(a->uri).find("aaaa") != std::string::npos);
  CHECK(std::string(a->resolved_path).find("assets/a") != std::string::npos);

  const ModelSharedAssetInfo* b = &(ModelPackage_Info(pkg))->shared_assets[1];
  CHECK(b != nullptr);
  CHECK(std::string(b->uri).find("bbbb") != std::string::npos);
  // Default convention path: shared_assets/sha256-<hex>
  CHECK(std::string(b->resolved_path).find("shared_assets/sha256-bb") != std::string::npos);

  // Resolve via API.
  const char* path = nullptr;
  CHECK_OK(ModelPackage_ResolveAssetUri(pkg,
                                        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                                        &path));
  CHECK(std::string(path).find("assets/a") != std::string::npos);

  CHECK_ERR(ModelPackage_ResolveAssetUri(pkg, "sha256:not_a_known_one", &path),
            MODEL_PACKAGE_ERR_ASSET_MISSING);

  ModelPackage_Close(pkg);
  return true;
}

bool test_unknown_field_rejected_strict() {
  Sandbox s;
  s.Write("manifest.json", R"({
    "schema_version": 1,
    "components": { "x": {"variants": {"cpu": {"typo_field": 1}}} }
  })");
  ModelPackage* pkg = nullptr;
  CHECK_ERR(ModelPackage_Open(s.root().c_str(), nullptr, &pkg), MODEL_PACKAGE_ERR_SCHEMA);
  return true;
}

bool test_unknown_field_tolerated_lenient() {
  Sandbox s;
  s.Write("manifest.json", R"({
    "schema_version": 1,
    "components": { "x": {"variants": {"cpu": {"typo_field": 1}}} }
  })");
  ModelPackageOpenOptions opts{};
  opts.strict_unknown_fields = false;
  opts.follow_symlinks = true;
  ModelPackage* pkg = nullptr;
  CHECK_OK(ModelPackage_Open(s.root().c_str(), &opts, &pkg));
  ModelPackage_Close(pkg);
  return true;
}

bool test_round_trip_getters_preserve_order() {
  Sandbox s;
  s.Write("manifest.json", R"({
    "schema_version": 1,
    "components": { "decoder": {"variants": {"cuda": {"ep":"CUDAExecutionProvider","device":"gpu"}}} }
  })");
  ModelPackage* pkg = nullptr;
  CHECK_OK(ModelPackage_Open(s.root().c_str(), nullptr, &pkg));
  const char* comp_json = nullptr;
  CHECK_OK(ModelPackage_GetComponentJson(pkg, "decoder", &comp_json));
  CHECK(comp_json != nullptr);
  CHECK(std::string(comp_json).find("\"variants\":") != std::string::npos);

  const char* var_json = nullptr;
  CHECK_OK(ModelPackage_GetVariantJson(pkg, "decoder", "cuda", &var_json));
  CHECK(var_json != nullptr);
  // "ep" must appear before "device" — ordered_json preserves declaration order.
  size_t ep_pos = std::string(var_json).find("\"ep\"");
  size_t dev_pos = std::string(var_json).find("\"device\"");
  CHECK(ep_pos != std::string::npos && dev_pos != std::string::npos && ep_pos < dev_pos);
  ModelPackage_Close(pkg);
  return true;
}

bool test_round_trip_preserves_unknown_fields_lenient() {
  Sandbox s;
  s.Write("manifest.json", R"({
    "schema_version": 1,
    "components": { "x": {"variants": {"cpu": {"future_field":"keepme"}}} }
  })");
  ModelPackageOpenOptions opts{};
  opts.strict_unknown_fields = false;
  opts.follow_symlinks = true;
  ModelPackage* pkg = nullptr;
  CHECK_OK(ModelPackage_Open(s.root().c_str(), &opts, &pkg));
  const char* var_json = nullptr;
  CHECK_OK(ModelPackage_GetVariantJson(pkg, "x", "cpu", &var_json));
  CHECK(std::string(var_json).find("future_field") != std::string::npos);
  ModelPackage_Close(pkg);
  return true;
}

bool test_missing_manifest() {
  Sandbox s;
  ModelPackage* pkg = nullptr;
  CHECK_ERR(ModelPackage_Open(s.root().c_str(), nullptr, &pkg), MODEL_PACKAGE_ERR_IO);
  return true;
}

bool test_unsupported_schema_version() {
  Sandbox s;
  s.Write("manifest.json", R"({"schema_version": 99, "components": {}})");
  ModelPackage* pkg = nullptr;
  CHECK_ERR(ModelPackage_Open(s.root().c_str(), nullptr, &pkg), MODEL_PACKAGE_ERR_VERSION);
  return true;
}

bool test_schema_version_string_and_minor() {
  // "<major>.<minor>" string parses into the split fields.
  {
    Sandbox s;
    s.Write("manifest.json",
            R"({"schema_version": "1.0", "components": {"a": {"variants": {"cpu": {}}}}})");
    ModelPackage* pkg = nullptr;
    CHECK_OK(ModelPackage_Open(s.root().c_str(), nullptr, &pkg));
    const ModelPackageInfo* info = ModelPackage_Info(pkg);
    CHECK(info->schema_version_major == 1);
    CHECK(info->schema_version_minor == 0);
    ModelPackage_Close(pkg);
  }

  // A newer minor than this build knows is accepted, and its unknown additive fields are
  // tolerated rather than rejected even under the default strict mode.
  {
    Sandbox s;
    s.Write("manifest.json",
            R"({"schema_version": "1.7", "some_future_field": true,
                "components": {"a": {"variants": {"cpu": {}}}}})");
    ModelPackage* pkg = nullptr;
    CHECK_OK(ModelPackage_Open(s.root().c_str(), nullptr, &pkg));
    const ModelPackageInfo* info = ModelPackage_Info(pkg);
    CHECK(info->schema_version_major == 1);
    CHECK(info->schema_version_minor == 7);
    ModelPackage_Close(pkg);
  }

  // An unsupported major is rejected regardless of minor.
  {
    Sandbox s;
    s.Write("manifest.json", R"({"schema_version": "2.0", "components": {}})");
    ModelPackage* pkg = nullptr;
    CHECK_ERR(ModelPackage_Open(s.root().c_str(), nullptr, &pkg), MODEL_PACKAGE_ERR_VERSION);
  }

  // A malformed schema_version string is a schema error.
  {
    Sandbox s;
    s.Write("manifest.json", R"({"schema_version": "1.x", "components": {}})");
    ModelPackage* pkg = nullptr;
    CHECK_ERR(ModelPackage_Open(s.root().c_str(), nullptr, &pkg), MODEL_PACKAGE_ERR_SCHEMA);
  }
  return true;
}

bool test_invalid_sha256_uri_rejected() {
  Sandbox s;
  s.Write("manifest.json", R"({
    "schema_version": 1,
    "shared_assets": { "sha256:notenough": "assets/a" },
    "components": {"x": {"variants": {"cpu": {}}}}
  })");
  ModelPackage* pkg = nullptr;
  CHECK_ERR(ModelPackage_Open(s.root().c_str(), nullptr, &pkg), MODEL_PACKAGE_ERR_SCHEMA);
  return true;
}

bool test_find_returns_null_on_missing() {
  Sandbox s;
  s.Write("manifest.json", R"({"schema_version":1,"components":{"a":{"variants":{"cpu":{}}}}})");
  ModelPackage* pkg = nullptr;
  CHECK_OK(ModelPackage_Open(s.root().c_str(), nullptr, &pkg));
  const ModelPackageInfo* info = ModelPackage_Info(pkg);
  CHECK(ModelPackage_FindComponent(info, "missing") == nullptr);
  CHECK(ModelComponentInfo_FindVariant(ModelPackage_FindComponent(info, "a"), "missing") == nullptr);
  ModelPackage_Close(pkg);
  return true;
}

struct Test {
  const char* name;
  bool (*fn)();
};

const Test kTests[] = {
    {"open_minimal_inline", test_open_minimal_inline},
    {"open_full_inline_with_metadata", test_open_full_inline_with_metadata},
    {"external_component_file", test_external_component_file},
    {"external_component_directory", test_external_component_directory},
    {"executor_info_inline_and_external", test_executor_info_inline_and_external},
    {"inline_executor_info_without_directory_accepted",
     test_inline_executor_info_without_directory_accepted},
    {"path_confinement_rejects_external_paths", test_path_confinement_rejects_external_paths},
    {"installed_layout_allows_absolute", test_installed_layout_allows_absolute},
    {"shared_assets_resolve", test_shared_assets_resolve},
    {"unknown_field_rejected_strict", test_unknown_field_rejected_strict},
    {"unknown_field_tolerated_lenient", test_unknown_field_tolerated_lenient},
    {"round_trip_getters_preserve_order", test_round_trip_getters_preserve_order},
    {"round_trip_preserves_unknown_fields_lenient",
     test_round_trip_preserves_unknown_fields_lenient},
    {"missing_manifest", test_missing_manifest},
    {"unsupported_schema_version", test_unsupported_schema_version},
    {"schema_version_string_and_minor", test_schema_version_string_and_minor},
    {"invalid_sha256_uri_rejected", test_invalid_sha256_uri_rejected},
    {"find_returns_null_on_missing", test_find_returns_null_on_missing},
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

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/// \file test_ort_json.cc
/// \brief Standalone unit tests for the OrtJson_* opaque-handle DOM API.
///
/// No external test framework: each test is a plain function that returns
/// true on success. main() runs the suite and exits non-zero on any failure.

#include "ort_json.h"
#include "model_package_api.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

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
                   g_current, __LINE__, ModelPackage_GetErrorMessage(_s));                            \
      ModelPackage_ReleaseStatus(_s);                                                                 \
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
    ModelPackageErrorCode _c = ModelPackage_GetErrorCode(_s);                                         \
    ModelPackage_ReleaseStatus(_s);                                                                   \
    if (_c != (expected_code)) {                                                                      \
      std::fprintf(stderr, "[FAIL] %s line %d: expected error %d, got %d\n",                          \
                   g_current, __LINE__, (int)(expected_code), (int)_c);                               \
      return false;                                                                                   \
    }                                                                                                 \
  } while (0)

bool test_parse_basic_types() {
  const char* doc = R"({"n": null, "b": true, "i": 42, "f": 3.5, "s": "hello"})";
  OrtJsonValue* root = nullptr;
  CHECK_OK(OrtJson_Parse(doc, std::strlen(doc), &root));
  CHECK(root != nullptr);
  CHECK(OrtJson_TypeOf(root) == ORT_JSON_OBJECT);
  CHECK(OrtJson_ObjectSize(root) == 5);

  const OrtJsonValue* vn = OrtJson_GetKey(root, "n");
  CHECK(OrtJson_TypeOf(vn) == ORT_JSON_NULL);

  const OrtJsonValue* vb = OrtJson_GetKey(root, "b");
  bool b = false;
  CHECK_OK(OrtJson_AsBool(vb, &b));
  CHECK(b == true);

  const OrtJsonValue* vi = OrtJson_GetKey(root, "i");
  int64_t i = 0;
  CHECK_OK(OrtJson_AsInt(vi, &i));
  CHECK(i == 42);

  const OrtJsonValue* vf = OrtJson_GetKey(root, "f");
  double d = 0;
  CHECK_OK(OrtJson_AsDouble(vf, &d));
  CHECK(d == 3.5);

  const OrtJsonValue* vs = OrtJson_GetKey(root, "s");
  const char* s = nullptr;
  CHECK_OK(OrtJson_AsString(vs, &s));
  CHECK(std::string(s) == "hello");

  OrtJson_Release(root);
  return true;
}

bool test_object_key_order_preserved() {
  const char* doc = R"({"zebra": 1, "alpha": 2, "mango": 3})";
  OrtJsonValue* root = nullptr;
  CHECK_OK(OrtJson_Parse(doc, std::strlen(doc), &root));
  CHECK(OrtJson_ObjectSize(root) == 3);
  CHECK(std::string(OrtJson_ObjectKeyAt(root, 0)) == "zebra");
  CHECK(std::string(OrtJson_ObjectKeyAt(root, 1)) == "alpha");
  CHECK(std::string(OrtJson_ObjectKeyAt(root, 2)) == "mango");
  CHECK(OrtJson_ObjectKeyAt(root, 3) == nullptr);
  OrtJson_Release(root);
  return true;
}

bool test_round_trip_preserves_order() {
  const char* doc = R"({"zebra":1,"alpha":2,"mango":3})";
  OrtJsonValue* root = nullptr;
  CHECK_OK(OrtJson_Parse(doc, std::strlen(doc), &root));
  const char* out = nullptr;
  CHECK_OK(OrtJson_Serialize(root, false, &out));
  CHECK(std::string(out) == doc);
  OrtJson_Release(root);
  return true;
}

bool test_array_navigation() {
  const char* doc = R"([10, 20, "thirty", false])";
  OrtJsonValue* root = nullptr;
  CHECK_OK(OrtJson_Parse(doc, std::strlen(doc), &root));
  CHECK(OrtJson_TypeOf(root) == ORT_JSON_ARRAY);
  CHECK(OrtJson_ArraySize(root) == 4);

  int64_t i = 0;
  CHECK_OK(OrtJson_AsInt(OrtJson_ArrayAt(root, 0), &i));
  CHECK(i == 10);
  CHECK_OK(OrtJson_AsInt(OrtJson_ArrayAt(root, 1), &i));
  CHECK(i == 20);

  const char* s = nullptr;
  CHECK_OK(OrtJson_AsString(OrtJson_ArrayAt(root, 2), &s));
  CHECK(std::string(s) == "thirty");

  bool b = true;
  CHECK_OK(OrtJson_AsBool(OrtJson_ArrayAt(root, 3), &b));
  CHECK(b == false);

  CHECK(OrtJson_ArrayAt(root, 4) == nullptr);

  OrtJson_Release(root);
  return true;
}

bool test_build_from_scratch() {
  OrtJsonValue* root = OrtJson_NewObject();
  CHECK(root != nullptr);

  CHECK_OK(OrtJson_ObjectSet(root, "name", OrtJson_NewString("x")));

  OrtJsonValue* args = OrtJson_NewArray();
  CHECK_OK(OrtJson_ArrayAppend(args, OrtJson_NewInt(1)));
  CHECK_OK(OrtJson_ArrayAppend(args, OrtJson_NewInt(2)));
  CHECK_OK(OrtJson_ObjectSet(root, "args", args));

  OrtJsonValue* meta = OrtJson_NewObject();
  CHECK_OK(OrtJson_ObjectSet(meta, "ok", OrtJson_NewBool(true)));
  CHECK_OK(OrtJson_ObjectSet(root, "meta", meta));

  const char* out = nullptr;
  CHECK_OK(OrtJson_Serialize(root, false, &out));
  CHECK(std::string(out) == R"({"name":"x","args":[1,2],"meta":{"ok":true}})");

  OrtJson_Release(root);
  return true;
}

bool test_object_remove_and_set_overwrite() {
  OrtJsonValue* root = OrtJson_NewObject();
  CHECK_OK(OrtJson_ObjectSet(root, "a", OrtJson_NewInt(1)));
  CHECK_OK(OrtJson_ObjectSet(root, "b", OrtJson_NewInt(2)));
  CHECK_OK(OrtJson_ObjectSet(root, "a", OrtJson_NewInt(99)));
  CHECK(OrtJson_ObjectSize(root) == 2);
  int64_t i = 0;
  CHECK_OK(OrtJson_AsInt(OrtJson_GetKey(root, "a"), &i));
  CHECK(i == 99);

  CHECK_OK(OrtJson_ObjectRemove(root, "b"));
  CHECK(OrtJson_ObjectSize(root) == 1);
  CHECK(!OrtJson_HasKey(root, "b"));
  CHECK(OrtJson_GetKey(root, "b") == nullptr);

  OrtJson_Release(root);
  return true;
}

bool test_type_mismatch_errors() {
  OrtJsonValue* root = OrtJson_NewString("hello");
  bool b = false;
  CHECK_ERR(OrtJson_AsBool(root, &b), MODEL_PACKAGE_ERR_SCHEMA);
  int64_t i = 0;
  CHECK_ERR(OrtJson_AsInt(root, &i), MODEL_PACKAGE_ERR_SCHEMA);
  double d = 0;
  CHECK_ERR(OrtJson_AsDouble(root, &d), MODEL_PACKAGE_ERR_SCHEMA);
  OrtJson_Release(root);

  OrtJsonValue* num = OrtJson_NewDouble(3.14);
  CHECK_ERR(OrtJson_AsInt(num, &i), MODEL_PACKAGE_ERR_SCHEMA);
  CHECK_OK(OrtJson_AsDouble(num, &d));
  CHECK(d == 3.14);
  OrtJson_Release(num);
  return true;
}

bool test_null_arg_errors() {
  OrtJsonValue* out = nullptr;
  CHECK_ERR(OrtJson_Parse(nullptr, 0, &out), MODEL_PACKAGE_ERR_INVALID_ARG);

  OrtJsonValue* root = OrtJson_NewObject();
  OrtJsonValue* leaked = OrtJson_NewInt(1);  // released below on failure
  CHECK_ERR(OrtJson_ObjectSet(root, nullptr, leaked), MODEL_PACKAGE_ERR_INVALID_ARG);
  OrtJson_Release(leaked);  // on failure, ownership stays with the caller
  OrtJson_Release(root);
  return true;
}

bool test_parse_error_returns_schema() {
  OrtJsonValue* out = nullptr;
  CHECK_ERR(OrtJson_Parse("{not json", 9, &out), MODEL_PACKAGE_ERR_SCHEMA);
  CHECK(out == nullptr);
  return true;
}

bool test_object_set_view_rejected() {
  OrtJsonValue* root = OrtJson_NewObject();
  CHECK_OK(OrtJson_ObjectSet(root, "x", OrtJson_NewInt(1)));
  const OrtJsonValue* view = OrtJson_GetKey(root, "x");
  CHECK(view != nullptr);
  OrtJsonValue* dest = OrtJson_NewObject();
  CHECK_ERR(OrtJson_ObjectSet(dest, "y", const_cast<OrtJsonValue*>(view)),
            MODEL_PACKAGE_ERR_INVALID_ARG);
  OrtJson_Release(dest);
  OrtJson_Release(root);
  return true;
}

bool test_pretty_vs_compact_serialize() {
  OrtJsonValue* root = OrtJson_NewObject();
  CHECK_OK(OrtJson_ObjectSet(root, "k", OrtJson_NewInt(1)));
  const char* compact = nullptr;
  CHECK_OK(OrtJson_Serialize(root, false, &compact));
  CHECK(std::string(compact) == R"({"k":1})");
  const char* pretty = nullptr;
  CHECK_OK(OrtJson_Serialize(root, true, &pretty));
  CHECK(std::string(compact) == R"({"k":1})");  // earlier pointer still valid
  CHECK(std::strstr(pretty, "\n") != nullptr);
  CHECK(std::strstr(pretty, "  \"k\": 1") != nullptr);
  OrtJson_Release(root);
  return true;
}

bool test_navigation_returns_cached_view() {
  OrtJsonValue* root = OrtJson_NewObject();
  CHECK_OK(OrtJson_ObjectSet(root, "x", OrtJson_NewInt(7)));
  const OrtJsonValue* a = OrtJson_GetKey(root, "x");
  const OrtJsonValue* b = OrtJson_GetKey(root, "x");
  CHECK(a == b);
  OrtJson_Release(root);
  return true;
}

bool test_parse_file() {
  std::string path = "/tmp/ort_json_test_input.json";
  {
    std::ofstream f(path);
    f << R"({"hello":"world"})";
  }
  OrtJsonValue* root = nullptr;
  CHECK_OK(OrtJson_ParseFile(path.c_str(), &root));
  const char* s = nullptr;
  CHECK_OK(OrtJson_AsString(OrtJson_GetKey(root, "hello"), &s));
  CHECK(std::string(s) == "world");
  OrtJson_Release(root);
  std::remove(path.c_str());

  OrtJsonValue* out = nullptr;
  CHECK_ERR(OrtJson_ParseFile("/tmp/does_not_exist_xyzzy.json", &out),
            MODEL_PACKAGE_ERR_IO);
  return true;
}

bool test_uint64_overflow_rejected() {
  const char* doc = "9223372036854775808";  // 2^63
  OrtJsonValue* root = nullptr;
  CHECK_OK(OrtJson_Parse(doc, std::strlen(doc), &root));
  int64_t i = 0;
  CHECK_ERR(OrtJson_AsInt(root, &i), MODEL_PACKAGE_ERR_SCHEMA);
  double d = 0;
  CHECK_OK(OrtJson_AsDouble(root, &d));
  OrtJson_Release(root);
  return true;
}

bool test_unicode_string_passthrough() {
  const char* doc = "{\"k\":\"\xc3\xa9\"}";  // "é" U+00E9
  OrtJsonValue* root = nullptr;
  CHECK_OK(OrtJson_Parse(doc, std::strlen(doc), &root));
  const char* s = nullptr;
  CHECK_OK(OrtJson_AsString(OrtJson_GetKey(root, "k"), &s));
  CHECK(std::string(s) == "\xc3\xa9");
  OrtJson_Release(root);
  return true;
}

struct Test {
  const char* name;
  bool (*fn)();
};

const Test kTests[] = {
    {"parse_basic_types", test_parse_basic_types},
    {"object_key_order_preserved", test_object_key_order_preserved},
    {"round_trip_preserves_order", test_round_trip_preserves_order},
    {"array_navigation", test_array_navigation},
    {"build_from_scratch", test_build_from_scratch},
    {"object_remove_and_set_overwrite", test_object_remove_and_set_overwrite},
    {"type_mismatch_errors", test_type_mismatch_errors},
    {"null_arg_errors", test_null_arg_errors},
    {"parse_error_returns_schema", test_parse_error_returns_schema},
    {"object_set_view_rejected", test_object_set_view_rejected},
    {"pretty_vs_compact_serialize", test_pretty_vs_compact_serialize},
    {"navigation_returns_cached_view", test_navigation_returns_cached_view},
    {"parse_file", test_parse_file},
    {"uint64_overflow_rejected", test_uint64_overflow_rejected},
    {"unicode_string_passthrough", test_unicode_string_passthrough},
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

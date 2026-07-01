// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "gtest/gtest.h"
#include "nlohmann/json.hpp"

#include "core/session/model_package/model_package_context.h"
#include "core/session/onnxruntime_experimental_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/abi_devices.h"
#include "test/autoep/test_autoep_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/api_asserts.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {
namespace {

// Typed function pointers for every OrtModelPackageApi_* experimental entry,
// resolved once via the experimental name-based lookup.
struct ModelPackageFns {
  OrtExperimental_OrtModelPackageApi_CreateModelPackageOptionsFromSessionOptions_SinceV28_Fn
      CreateModelPackageOptionsFromSessionOptions{nullptr};
  OrtExperimental_OrtModelPackageApi_ReleaseModelPackageOptions_SinceV28_Fn
      ReleaseModelPackageOptions{nullptr};
  OrtExperimental_OrtModelPackageApi_CreateModelPackageContext_SinceV28_Fn
      CreateModelPackageContext{nullptr};
  OrtExperimental_OrtModelPackageApi_ReleaseModelPackageContext_SinceV28_Fn
      ReleaseModelPackageContext{nullptr};
  OrtExperimental_OrtModelPackageApi_ModelPackage_GetSchemaVersion_SinceV28_Fn
      ModelPackage_GetSchemaVersion{nullptr};
  OrtExperimental_OrtModelPackageApi_ModelPackage_GetComponentCount_SinceV28_Fn
      ModelPackage_GetComponentCount{nullptr};
  OrtExperimental_OrtModelPackageApi_ModelPackage_GetComponentNames_SinceV28_Fn
      ModelPackage_GetComponentNames{nullptr};
  OrtExperimental_OrtModelPackageApi_ModelPackage_GetVariantCount_SinceV28_Fn
      ModelPackage_GetVariantCount{nullptr};
  OrtExperimental_OrtModelPackageApi_ModelPackage_GetVariantNames_SinceV28_Fn
      ModelPackage_GetVariantNames{nullptr};
  OrtExperimental_OrtModelPackageApi_ModelPackage_GetVariantEpName_SinceV28_Fn
      ModelPackage_GetVariantEpName{nullptr};
  OrtExperimental_OrtModelPackageApi_ModelPackage_ResolveStringRef_SinceV28_Fn
      ModelPackage_ResolveStringRef{nullptr};
  OrtExperimental_OrtModelPackageApi_SelectComponent_SinceV28_Fn
      SelectComponent{nullptr};
  OrtExperimental_OrtModelPackageApi_ReleaseModelPackageComponentContext_SinceV28_Fn
      ReleaseModelPackageComponentContext{nullptr};
  OrtExperimental_OrtModelPackageApi_ModelPackageComponent_GetSelectedVariantName_SinceV28_Fn
      ModelPackageComponent_GetSelectedVariantName{nullptr};
  OrtExperimental_OrtModelPackageApi_ModelPackageComponent_GetSelectedVariantFolderPath_SinceV28_Fn
      ModelPackageComponent_GetSelectedVariantFolderPath{nullptr};
  OrtExperimental_OrtModelPackageApi_CreateSession_SinceV28_Fn
      CreateSession{nullptr};
};

inline const ModelPackageFns& GetModelPackageFns() {
  static const ModelPackageFns fns = []() {
    const OrtApi* api = &Ort::GetApi();
    namespace Exp = Ort::Experimental;
    ModelPackageFns f;
    f.CreateModelPackageOptionsFromSessionOptions =
        Exp::Get_OrtModelPackageApi_CreateModelPackageOptionsFromSessionOptions_SinceV28_FnOrThrow(api);
    f.ReleaseModelPackageOptions =
        Exp::Get_OrtModelPackageApi_ReleaseModelPackageOptions_SinceV28_FnOrThrow(api);
    f.CreateModelPackageContext =
        Exp::Get_OrtModelPackageApi_CreateModelPackageContext_SinceV28_FnOrThrow(api);
    f.ReleaseModelPackageContext =
        Exp::Get_OrtModelPackageApi_ReleaseModelPackageContext_SinceV28_FnOrThrow(api);
    f.ModelPackage_GetSchemaVersion =
        Exp::Get_OrtModelPackageApi_ModelPackage_GetSchemaVersion_SinceV28_FnOrThrow(api);
    f.ModelPackage_GetComponentCount =
        Exp::Get_OrtModelPackageApi_ModelPackage_GetComponentCount_SinceV28_FnOrThrow(api);
    f.ModelPackage_GetComponentNames =
        Exp::Get_OrtModelPackageApi_ModelPackage_GetComponentNames_SinceV28_FnOrThrow(api);
    f.ModelPackage_GetVariantCount =
        Exp::Get_OrtModelPackageApi_ModelPackage_GetVariantCount_SinceV28_FnOrThrow(api);
    f.ModelPackage_GetVariantNames =
        Exp::Get_OrtModelPackageApi_ModelPackage_GetVariantNames_SinceV28_FnOrThrow(api);
    f.ModelPackage_GetVariantEpName =
        Exp::Get_OrtModelPackageApi_ModelPackage_GetVariantEpName_SinceV28_FnOrThrow(api);
    f.ModelPackage_ResolveStringRef =
        Exp::Get_OrtModelPackageApi_ModelPackage_ResolveStringRef_SinceV28_FnOrThrow(api);
    f.SelectComponent =
        Exp::Get_OrtModelPackageApi_SelectComponent_SinceV28_FnOrThrow(api);
    f.ReleaseModelPackageComponentContext =
        Exp::Get_OrtModelPackageApi_ReleaseModelPackageComponentContext_SinceV28_FnOrThrow(api);
    f.ModelPackageComponent_GetSelectedVariantName =
        Exp::Get_OrtModelPackageApi_ModelPackageComponent_GetSelectedVariantName_SinceV28_FnOrThrow(api);
    f.ModelPackageComponent_GetSelectedVariantFolderPath =
        Exp::Get_OrtModelPackageApi_ModelPackageComponent_GetSelectedVariantFolderPath_SinceV28_FnOrThrow(api);
    f.CreateSession =
        Exp::Get_OrtModelPackageApi_CreateSession_SinceV28_FnOrThrow(api);
    return f;
  }();
  return fns;
}

// ────────────────────────────────────────────────────────────────────────────
// Fixture helpers for building model packages on disk.
// Every package is a single manifest.json at the package root that declares
// components/variants/executor_info inline. Variant directories live at
// `<package_root>/<component>/<variant>/` and contain the model file.
// ────────────────────────────────────────────────────────────────────────────

struct VariantSpec {
  std::string variant_name;
  std::string ep;                      // empty => omit
  std::string device;                  // empty => omit
  std::string compatibility_string;    // empty => omit
  std::filesystem::path source_model;  // empty => no executor_info
  std::optional<std::unordered_map<std::string, std::string>> session_options;
  std::optional<std::unordered_map<std::string, std::string>> provider_options;
};

// Build a single-component new-schema package on disk and return its root.
// `package_root` is wiped before writing.
std::filesystem::path BuildPackage(const std::filesystem::path& package_root,
                                   const std::string& component_name,
                                   const std::vector<VariantSpec>& variants) {
  std::error_code ec;
  std::filesystem::remove_all(package_root, ec);
  std::filesystem::create_directories(package_root);

  using ojson = nlohmann::ordered_json;
  ojson variants_obj = ojson::object();
  for (const auto& v : variants) {
    const std::string variant_dir_rel = component_name + "/" + v.variant_name;
    const auto variant_dir_abs = package_root / component_name / v.variant_name;
    std::filesystem::create_directories(variant_dir_abs);

    ojson variant_obj = ojson::object();
    variant_obj["variant_directory"] = variant_dir_rel;
    if (!v.ep.empty()) variant_obj["ep"] = v.ep;
    if (!v.device.empty()) variant_obj["device"] = v.device;
    if (!v.compatibility_string.empty()) variant_obj["compatibility_string"] = v.compatibility_string;

    if (!v.source_model.empty()) {
      const std::string model_filename = v.source_model.filename().string();
      std::filesystem::copy_file(v.source_model, variant_dir_abs / model_filename,
                                 std::filesystem::copy_options::overwrite_existing, ec);

      ojson ort_info = ojson::object();
      ort_info["model_file"] = model_filename;
      if (v.session_options.has_value()) {
        ojson so = ojson::object();
        for (const auto& kv : *v.session_options) so[kv.first] = kv.second;
        ort_info["session_options"] = std::move(so);
      }
      if (v.provider_options.has_value()) {
        ojson po = ojson::object();
        for (const auto& kv : *v.provider_options) po[kv.first] = kv.second;
        ort_info["provider_options"] = std::move(po);
      }
      ojson executor_info = ojson::object();
      executor_info["ort"] = std::move(ort_info);
      variant_obj["executor_info"] = std::move(executor_info);
    }

    variants_obj[v.variant_name] = std::move(variant_obj);
  }

  ojson component_obj = ojson::object();
  component_obj["variants"] = std::move(variants_obj);

  ojson components_obj = ojson::object();
  components_obj[component_name] = std::move(component_obj);

  ojson manifest = ojson::object();
  manifest["schema_version"] = "1.0";
  manifest["components"] = std::move(components_obj);

  std::ofstream os(package_root / "manifest.json", std::ios::binary);
  os << manifest.dump(2);
  return package_root;
}

// Convenience: most tests use the same two-variant shape backed by mul_1.onnx /
// mul_16.onnx. `compat_1` and `compat_2` default to empty (no compatibility string).
std::filesystem::path BuildTwoVariantPackage(const std::filesystem::path& package_root,
                                             std::string_view variant_name_1,
                                             std::string_view device_1,
                                             std::string_view compat_1,
                                             const std::filesystem::path& model_1,
                                             std::string_view variant_name_2,
                                             std::string_view device_2,
                                             std::string_view compat_2,
                                             const std::filesystem::path& model_2,
                                             std::string_view ep_name = "example_ep") {
  std::vector<VariantSpec> variants;
  variants.push_back(VariantSpec{std::string(variant_name_1), std::string(ep_name), std::string(device_1), std::string(compat_1), model_1, {}, {}});
  variants.push_back(VariantSpec{std::string(variant_name_2), std::string(ep_name), std::string(device_2), std::string(compat_2), model_2, {}, {}});
  return BuildPackage(package_root, "model_1", variants);
}

}  // namespace

// ────────────────────────────────────────────────────────────────────────────
// Model Package API tests
// ────────────────────────────────────────────────────────────────────────────
TEST(ModelPackageApiTest, PackageContextQueries) {
  const auto package_root = std::filesystem::temp_directory_path() / "ort_model_package_api_test";
  BuildTwoVariantPackage(package_root,
                         "variant_1", "cpu",
                         "example_ep;version=0.1.0;ort_api_version=25;hardware_architecture=arch1",
                         "testdata/mul_1.onnx",
                         "variant_2", "npu",
                         "example_ep;version=0.1.0;ort_api_version=25;hardware_architecture=arch2",
                         "testdata/mul_16.onnx");

  const auto& pkg_api = GetModelPackageFns();
  ASSERT_NE(pkg_api.CreateModelPackageContext, nullptr) << "Model package experimental API is not available";

  auto context_deleter = [&pkg_api](OrtModelPackageContext* p) {
    if (p) pkg_api.ReleaseModelPackageContext(p);
  };
  std::unique_ptr<OrtModelPackageContext, decltype(context_deleter)> model_pkg_context(nullptr, context_deleter);

  OrtModelPackageContext* raw_context = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api.CreateModelPackageContext(package_root.c_str(), &raw_context));
  model_pkg_context.reset(raw_context);

  size_t component_count = 0;
  ASSERT_ORTSTATUS_OK(pkg_api.ModelPackage_GetComponentCount(model_pkg_context.get(), &component_count));
  ASSERT_EQ(component_count, 1u);

  const char* const* component_names = nullptr;
  size_t component_name_count = 0;
  ASSERT_ORTSTATUS_OK(pkg_api.ModelPackage_GetComponentNames(
      model_pkg_context.get(), &component_names, &component_name_count));
  ASSERT_EQ(component_name_count, 1u);
  ASSERT_NE(component_names, nullptr);
  ASSERT_NE(component_names[0], nullptr);
  EXPECT_STREQ(component_names[0], "model_1");

  size_t variant_count = 0;
  ASSERT_ORTSTATUS_OK(pkg_api.ModelPackage_GetVariantCount(
      model_pkg_context.get(), "model_1", &variant_count));
  ASSERT_EQ(variant_count, 2u);

  const char* const* variant_names = nullptr;
  size_t variant_name_count = 0;
  ASSERT_ORTSTATUS_OK(pkg_api.ModelPackage_GetVariantNames(
      model_pkg_context.get(), "model_1", &variant_names, &variant_name_count));
  ASSERT_EQ(variant_name_count, 2u);

  std::unordered_set<std::string> variant_name_set;
  for (size_t i = 0; i < variant_name_count; ++i) {
    ASSERT_NE(variant_names[i], nullptr);
    variant_name_set.insert(variant_names[i]);
  }
  EXPECT_EQ(variant_name_set.count("variant_1"), 1u);
  EXPECT_EQ(variant_name_set.count("variant_2"), 1u);

  std::error_code ec;
  std::filesystem::remove_all(package_root, ec);
}

TEST(ModelPackageApiTest, ResolveStringRef) {
  const auto package_root = std::filesystem::temp_directory_path() / "ort_model_package_resolve_test";
  std::vector<VariantSpec> variants;
  variants.push_back(VariantSpec{"variant_1", "example_ep", "cpu", "", "testdata/mul_1.onnx", {}, {}});
  BuildPackage(package_root, "model_1", variants);

  // A content-addressed shared asset, discovered by convention at shared_assets/sha256-<hex>/.
  const std::string digest(64, 'a');
  const auto asset_dir = package_root / "shared_assets" / ("sha256-" + digest);
  std::filesystem::create_directories(asset_dir);
  {
    std::ofstream os(asset_dir / "asset.txt", std::ios::binary);
    os << "hello";
  }

  const auto& pkg_api = GetModelPackageFns();
  ASSERT_NE(pkg_api.ModelPackage_ResolveStringRef, nullptr) << "Model package experimental API is not available";

  auto context_deleter = [&pkg_api](OrtModelPackageContext* p) {
    if (p) pkg_api.ReleaseModelPackageContext(p);
  };
  std::unique_ptr<OrtModelPackageContext, decltype(context_deleter)> ctx(nullptr, context_deleter);
  OrtModelPackageContext* raw_context = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api.CreateModelPackageContext(package_root.c_str(), &raw_context));
  ctx.reset(raw_context);

  const char* resolved = nullptr;

  // "sha256:<hex>" resolves to the shared asset directory (override/discovery aware).
  ASSERT_ORTSTATUS_OK(pkg_api.ModelPackage_ResolveStringRef(
      ctx.get(), nullptr, ("sha256:" + digest).c_str(), /*must_exist=*/1, &resolved));
  ASSERT_NE(resolved, nullptr);
  EXPECT_EQ(std::filesystem::canonical(resolved), std::filesystem::canonical(asset_dir));

  // "sha256:<hex>/<tail>" resolves the confined tail under the asset directory.
  ASSERT_ORTSTATUS_OK(pkg_api.ModelPackage_ResolveStringRef(
      ctx.get(), nullptr, ("sha256:" + digest + "/asset.txt").c_str(), /*must_exist=*/1, &resolved));
  ASSERT_NE(resolved, nullptr);
  EXPECT_EQ(std::filesystem::canonical(resolved), std::filesystem::canonical(asset_dir / "asset.txt"));

  // A plain relative path resolves against base_dir.
  const auto variant_dir = package_root / "model_1" / "variant_1";
  ASSERT_ORTSTATUS_OK(pkg_api.ModelPackage_ResolveStringRef(
      ctx.get(), variant_dir.string().c_str(), "mul_1.onnx", /*must_exist=*/1, &resolved));
  ASSERT_NE(resolved, nullptr);
  EXPECT_EQ(std::filesystem::canonical(resolved), std::filesystem::canonical(variant_dir / "mul_1.onnx"));

  // An undeclared sha256 asset is rejected even when must_exist is false.
  const std::string missing_digest(64, 'b');
  OrtStatus* status = pkg_api.ModelPackage_ResolveStringRef(
      ctx.get(), nullptr, ("sha256:" + missing_digest).c_str(), /*must_exist=*/0, &resolved);
  EXPECT_NE(status, nullptr);
  if (status != nullptr) Ort::GetApi().ReleaseStatus(status);

  std::error_code ec;
  std::filesystem::remove_all(package_root, ec);
}

TEST(ModelPackageApiTest, SingleFileVariantInComponent_SelectComponentAndCreateSession) {
  const auto package_root = std::filesystem::temp_directory_path() / "ort_model_package_api_test";
  std::vector<VariantSpec> variants;
  variants.push_back(VariantSpec{
      "variant_1", "example_ep", "cpu",
      "example_ep;version=0.1.0;ort_api_version=25;hardware_architecture=arch1",
      "testdata/mul_1.onnx",
      std::unordered_map<std::string, std::string>{
          {"session.disable_prepacking", "1"},
          {"session.intra_op.allow_spinning", "0"},
      },
      std::unordered_map<std::string, std::string>{
          {"backend_path", "example_backend"},
          {"enable_htp", "1"},
      }});
  variants.push_back(VariantSpec{
      "variant_2", "example_ep", "npu", "example_ep;version=0.1.0;ort_api_version=25;hardware_architecture=arch2", "testdata/mul_16.onnx", {}, {}});
  BuildPackage(package_root, "model_1", variants);

  RegisteredEpDeviceUniquePtr example_ep;
  ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_info, example_ep));
  Ort::ConstEpDevice plugin_ep_device(example_ep.get());

  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> ep_options;
  session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);

  const auto& pkg_api = GetModelPackageFns();
  ASSERT_NE(pkg_api.CreateModelPackageContext, nullptr) << "Model package experimental API is not available";

  auto options_deleter = [&pkg_api](OrtModelPackageOptions* p) {
    if (p) pkg_api.ReleaseModelPackageOptions(p);
  };
  auto context_deleter = [&pkg_api](OrtModelPackageContext* p) {
    if (p) pkg_api.ReleaseModelPackageContext(p);
  };
  auto component_context_deleter = [&pkg_api](OrtModelPackageComponentContext* p) {
    if (p) pkg_api.ReleaseModelPackageComponentContext(p);
  };

  std::unique_ptr<OrtModelPackageOptions, decltype(options_deleter)> model_pkg_options(nullptr, options_deleter);
  std::unique_ptr<OrtModelPackageContext, decltype(context_deleter)> model_pkg_context(nullptr, context_deleter);
  std::unique_ptr<OrtModelPackageComponentContext, decltype(component_context_deleter)> component_context(nullptr, component_context_deleter);

  OrtModelPackageOptions* raw_options = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api.CreateModelPackageOptionsFromSessionOptions(*ort_env, session_options, &raw_options));
  model_pkg_options.reset(raw_options);

  OrtModelPackageContext* raw_context = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api.CreateModelPackageContext(package_root.c_str(), &raw_context));
  model_pkg_context.reset(raw_context);

  OrtModelPackageComponentContext* raw_component_context = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api.SelectComponent(model_pkg_context.get(),
                                              "model_1",
                                              model_pkg_options.get(),
                                              &raw_component_context));
  component_context.reset(raw_component_context);

  OrtSession* raw_session = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api.CreateSession(*ort_env,
                                            component_context.get(),
                                            session_options,
                                            &raw_session));
  Ort::Session session(raw_session);

  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::vector<int64_t> shape = {3, 2};
  std::vector<float> input_data = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  Ort::Value input = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_data.size(),
                                                     shape.data(), shape.size());
  const char* input_names[] = {"X"};
  const char* output_names[] = {"Y"};
  std::vector<Ort::Value> inputs;
  inputs.push_back(std::move(input));

  auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, inputs.data(), inputs.size(), output_names, 1);
  ASSERT_EQ(outputs.size(), 1u);
  const float* out = outputs[0].GetTensorData<float>();
  gsl::span<const float> out_span(out, input_data.size());
  EXPECT_THAT(out_span, ::testing::ElementsAre(1.f, 4.f, 9.f, 16.f, 25.f, 36.f));

  std::error_code ec;
  std::filesystem::remove_all(package_root, ec);
}

TEST(ModelPackageTest, LoadModelPackageAndRunInference_PluginEp_AppendV2) {
  // package_root is a new-schema model package directory with one component and two variants.
  // ORT parses the manifest, selects the variant whose device matches the registered EP (cpu),
  // and loads/runs it successfully.
  const auto package_root = std::filesystem::temp_directory_path() / "ort_model_package_test";
  BuildTwoVariantPackage(package_root,
                         "variant_1", "cpu", "",
                         "testdata/mul_1.onnx",
                         "variant_2", "npu", "",
                         "testdata/mul_16.onnx");

  RegisteredEpDeviceUniquePtr example_ep;
  ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_info, example_ep));
  Ort::ConstEpDevice plugin_ep_device(example_ep.get());

  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> ep_options;
  session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);

  Ort::Session session(*ort_env, package_root.c_str(), session_options);

  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::vector<int64_t> shape = {3, 2};
  std::vector<float> input_data = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  Ort::Value input = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_data.size(),
                                                     shape.data(), shape.size());
  const char* input_names[] = {"X"};
  const char* output_names[] = {"Y"};
  std::vector<Ort::Value> inputs;
  inputs.push_back(std::move(input));

  auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, inputs.data(), inputs.size(),
                             output_names, 1);
  ASSERT_EQ(outputs.size(), 1u);
  const float* out = outputs[0].GetTensorData<float>();
  gsl::span<const float> out_span(out, input_data.size());
  EXPECT_THAT(out_span, ::testing::ElementsAre(1.f, 4.f, 9.f, 16.f, 25.f, 36.f));

  std::error_code ec;
  std::filesystem::remove_all(package_root, ec);
}

TEST(ModelPackageTest, LoadModelPackageAndRunInference_PreferCpu) {
  const auto package_root = std::filesystem::temp_directory_path() / "ort_model_package_test";
  BuildTwoVariantPackage(package_root,
                         "variant_1", "cpu", "",
                         "testdata/mul_1.onnx",
                         "variant_2", "npu", "",
                         "testdata/mul_16.onnx");

  RegisteredEpDeviceUniquePtr example_ep;
  ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_info, example_ep));
  Ort::ConstEpDevice plugin_ep_device(example_ep.get());

  Ort::SessionOptions session_options;
  session_options.SetEpSelectionPolicy(OrtExecutionProviderDevicePolicy_PREFER_CPU);

  Ort::Session session(*ort_env, package_root.c_str(), session_options);

  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::vector<int64_t> shape = {3, 2};
  std::vector<float> input_data = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  Ort::Value input = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_data.size(),
                                                     shape.data(), shape.size());
  const char* input_names[] = {"X"};
  const char* output_names[] = {"Y"};
  std::vector<Ort::Value> inputs;
  inputs.push_back(std::move(input));

  auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, inputs.data(), inputs.size(),
                             output_names, 1);
  ASSERT_EQ(outputs.size(), 1u);
  const float* out = outputs[0].GetTensorData<float>();
  gsl::span<const float> out_span(out, input_data.size());
  EXPECT_THAT(out_span, ::testing::ElementsAre(1.f, 4.f, 9.f, 16.f, 25.f, 36.f));

  std::error_code ec;
  std::filesystem::remove_all(package_root, ec);
}

TEST(ModelPackageTest, CheckCompiledModelCompatibilityInfo) {
  RegisteredEpDeviceUniquePtr example_ep;
  ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_info, example_ep));
  Ort::ConstEpDevice plugin_ep_device(example_ep.get());

  const ORTCHAR_T* input_model_file = ORT_TSTR("testdata/mul_1.onnx");
  const ORTCHAR_T* output_model_file = ORT_TSTR("plugin_ep_compat_test.onnx");
  std::filesystem::remove(output_model_file);

  {
    Ort::SessionOptions session_options;
    std::unordered_map<std::string, std::string> ep_options;
    session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);

    Ort::ModelCompilationOptions compile_options(*ort_env, session_options);
    compile_options.SetFlags(OrtCompileApiFlags_ERROR_IF_NO_NODES_COMPILED);
    compile_options.SetInputModelPath(input_model_file);
    compile_options.SetOutputModelPath(output_model_file);

    // Embed the EPContext binary data inside the compiled model so the model is self-contained.
    // This test copies only the compiled .onnx into the model package, so it must not rely on a
    // separate sidecar EPContext data file (which non-embedded mode would produce).
    compile_options.SetEpContextEmbedMode(true);

    ASSERT_CXX_ORTSTATUS_OK(Ort::CompileModel(*ort_env, compile_options));
    ASSERT_TRUE(std::filesystem::exists(output_model_file));
  }

  // Build compat strings dynamically against current ORT_API_VERSION so the EP's ORT-version check
  // doesn't short-circuit to PREFER_RECOMPILATION for both variants. With matching ORT versions the
  // hardware_architecture field differentiates: arch1 -> OPTIMAL, arch2 -> PREFER_RECOMPILATION, so
  // variant_1 (mul_1) must win over variant_2 (mul_16). If variant_2 was picked, session init would
  // fail with "No Op registered for Mul16".
  const std::string ort_api_version_str = std::to_string(ORT_API_VERSION);
  const std::string compat_arch2 =
      "example_ep;version=0.1.0;ort_api_version=" + ort_api_version_str + ";hardware_architecture=arch2";
  const std::string compat_arch1 =
      "example_ep;version=0.1.0;ort_api_version=" + ort_api_version_str + ";hardware_architecture=arch1";

  const auto package_root = std::filesystem::temp_directory_path() / "ort_model_package_test";
  BuildTwoVariantPackage(package_root,
                         "variant_2", "cpu", compat_arch2,
                         std::filesystem::path{"testdata/mul_16.onnx"},
                         "variant_1", "cpu", compat_arch1,
                         std::filesystem::path{"plugin_ep_compat_test.onnx"});

  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> ep_options;
  session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);

  Ort::Session session(*ort_env, package_root.c_str(), session_options);

  std::error_code ec;
  std::filesystem::remove_all(package_root, ec);
}

TEST(ModelPackageTest, ParseVariantsFromPackageRoot) {
  const auto package_root = std::filesystem::temp_directory_path() / "ort_model_package_parse_from_package_root";
  BuildTwoVariantPackage(package_root,
                         "variant_1", "cpu", "",
                         std::filesystem::path{"testdata/mul_1.onnx"},
                         "variant_2", "npu", "",
                         std::filesystem::path{"testdata/mul_16.onnx"});

  ModelPackageContext ctx(package_root);
  const auto& variants = ctx.GetVariantInfos();

  ASSERT_EQ(variants.size(), 2u);

  std::unordered_map<std::string, const VariantInfo*> by_file;
  for (const auto& v : variants) {
    ASSERT_TRUE(v.file.has_value());
    by_file.emplace(v.file->model_file_path.filename().string(), &v);
  }

  ASSERT_EQ(by_file.count("mul_1.onnx"), 1u);
  ASSERT_EQ(by_file.count("mul_16.onnx"), 1u);

  const auto* v1 = by_file.at("mul_1.onnx");
  EXPECT_EQ(v1->ep_compatibility.ep.value_or(""), "example_ep");
  EXPECT_EQ(v1->ep_compatibility.device.value_or(""), "cpu");

  const auto* v2 = by_file.at("mul_16.onnx");
  EXPECT_EQ(v2->ep_compatibility.ep.value_or(""), "example_ep");
  EXPECT_EQ(v2->ep_compatibility.device.value_or(""), "npu");

  std::error_code ec;
  std::filesystem::remove_all(package_root, ec);
}

TEST(ModelPackageApiTest, GetVariantEpName_ReturnsSingleEp) {
  const auto package_root = std::filesystem::temp_directory_path() / "ort_mp_pre_selection_ep_name";
  std::vector<VariantSpec> variants;
  variants.push_back(VariantSpec{"variant_1", "example_ep", "cpu", "", "testdata/mul_1.onnx", {}, {}});
  variants.push_back(VariantSpec{"variant_2", "other_ep", "npu", "", "testdata/mul_1.onnx", {}, {}});
  BuildPackage(package_root, "model_1", variants);

  const auto& pkg_api = GetModelPackageFns();
  ASSERT_NE(pkg_api.CreateModelPackageContext, nullptr) << "Model package experimental API is not available";

  auto context_deleter = [&pkg_api](OrtModelPackageContext* p) {
    if (p) pkg_api.ReleaseModelPackageContext(p);
  };
  std::unique_ptr<OrtModelPackageContext, decltype(context_deleter)> ctx(nullptr, context_deleter);
  OrtModelPackageContext* raw_ctx = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api.CreateModelPackageContext(package_root.c_str(), &raw_ctx));
  ctx.reset(raw_ctx);

  const char* ep1 = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api.ModelPackage_GetVariantEpName(
      ctx.get(), "model_1", "variant_1", &ep1));
  ASSERT_NE(ep1, nullptr);
  EXPECT_STREQ(ep1, "example_ep");

  const char* ep2 = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api.ModelPackage_GetVariantEpName(
      ctx.get(), "model_1", "variant_2", &ep2));
  ASSERT_NE(ep2, nullptr);
  EXPECT_STREQ(ep2, "other_ep");

  // Optional out-parameter: callers can pass NULL.
  ASSERT_ORTSTATUS_OK(pkg_api.ModelPackage_GetVariantEpName(
      ctx.get(), "model_1", "variant_1", nullptr));

  std::error_code ec;
  std::filesystem::remove_all(package_root, ec);
}

TEST(ModelPackageTest, VariantSelector_TieBreakIsDeterministic) {
  // Both variants point at the *same* model file (mul_1.onnx) so whichever wins works at runtime.
  // They advertise identical EP/device pairs and empty compatibility_string so the EP returns the
  // same score (NOT_APPLICABLE) for both: ties must resolve deterministically across runs.
  RegisteredEpDeviceUniquePtr example_ep;
  ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_info, example_ep));
  Ort::ConstEpDevice plugin_ep_device(example_ep.get());

  std::string first_selected_variant;

  for (int iter = 0; iter < 5; ++iter) {
    const auto package_root = std::filesystem::temp_directory_path() / "ort_mp_tie_break";
    BuildTwoVariantPackage(package_root,
                           "variant_a", "cpu", "",
                           std::filesystem::path{"testdata/mul_1.onnx"},
                           "variant_b", "cpu", "",
                           std::filesystem::path{"testdata/mul_1.onnx"});

    Ort::SessionOptions session_options;
    std::unordered_map<std::string, std::string> ep_options;
    session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);

    const auto& pkg_api = GetModelPackageFns();
    ASSERT_NE(pkg_api.CreateModelPackageContext, nullptr) << "Model package experimental API is not available";

    auto options_deleter = [&pkg_api](OrtModelPackageOptions* p) { if (p) pkg_api.ReleaseModelPackageOptions(p); };
    auto context_deleter = [&pkg_api](OrtModelPackageContext* p) { if (p) pkg_api.ReleaseModelPackageContext(p); };
    auto component_context_deleter = [&pkg_api](OrtModelPackageComponentContext* p) {
      if (p) pkg_api.ReleaseModelPackageComponentContext(p);
    };
    std::unique_ptr<OrtModelPackageOptions, decltype(options_deleter)> mp_opts(nullptr, options_deleter);
    std::unique_ptr<OrtModelPackageContext, decltype(context_deleter)> ctx(nullptr, context_deleter);
    std::unique_ptr<OrtModelPackageComponentContext, decltype(component_context_deleter)> comp_ctx(nullptr, component_context_deleter);

    OrtModelPackageOptions* raw_mp_opts = nullptr;
    ASSERT_ORTSTATUS_OK(pkg_api.CreateModelPackageOptionsFromSessionOptions(*ort_env, session_options, &raw_mp_opts));
    mp_opts.reset(raw_mp_opts);

    OrtModelPackageContext* raw_ctx = nullptr;
    ASSERT_ORTSTATUS_OK(pkg_api.CreateModelPackageContext(package_root.c_str(), &raw_ctx));
    ctx.reset(raw_ctx);

    OrtModelPackageComponentContext* raw_comp_ctx = nullptr;
    ASSERT_ORTSTATUS_OK(pkg_api.SelectComponent(ctx.get(), "model_1", mp_opts.get(), &raw_comp_ctx));
    comp_ctx.reset(raw_comp_ctx);

    const ORTCHAR_T* selected_folder = nullptr;
    ASSERT_ORTSTATUS_OK(pkg_api.ModelPackageComponent_GetSelectedVariantFolderPath(comp_ctx.get(), &selected_folder));
    ASSERT_NE(selected_folder, nullptr);

    // Variant directories live at <root>/model_1/<variant_x>; the leaf name is the variant.
    const auto selected_variant_dir = std::filesystem::path(selected_folder).filename().string();
    ASSERT_TRUE(selected_variant_dir == "variant_a" || selected_variant_dir == "variant_b")
        << "unexpected variant dir: " << selected_variant_dir;

    if (iter == 0) {
      first_selected_variant = selected_variant_dir;
    } else {
      EXPECT_EQ(selected_variant_dir, first_selected_variant)
          << "tie-break selection drifted across runs (iter " << iter << ")";
    }

    std::error_code ec;
    std::filesystem::remove_all(package_root, ec);
  }
}

TEST(ModelPackageTest, VariantSessionOptions_DispatchedThroughAddSessionConfigEntry) {
  // Per-variant session_options assigns a typed key (session.intra_op_num_threads) a value that
  // is not a valid integer. Routing this through OrtApis::AddSessionConfigEntry must reject it.
  const auto package_root = std::filesystem::temp_directory_path() / "ort_mp_session_options_dispatch";
  std::vector<VariantSpec> variants;
  variants.push_back(VariantSpec{
      "variant_1", "example_ep", "cpu", "", "testdata/mul_1.onnx", std::unordered_map<std::string, std::string>{
                                                                       {"session.intra_op_num_threads", "not_an_int"},
                                                                   },
      {}});
  BuildPackage(package_root, "model_1", variants);

  RegisteredEpDeviceUniquePtr example_ep;
  ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_info, example_ep));
  Ort::ConstEpDevice plugin_ep_device(example_ep.get());

  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> ep_options;
  session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);

  const auto& pkg_api = GetModelPackageFns();
  ASSERT_NE(pkg_api.CreateModelPackageContext, nullptr) << "Model package experimental API is not available";

  auto options_deleter = [&pkg_api](OrtModelPackageOptions* p) { if (p) pkg_api.ReleaseModelPackageOptions(p); };
  auto context_deleter = [&pkg_api](OrtModelPackageContext* p) { if (p) pkg_api.ReleaseModelPackageContext(p); };
  auto component_context_deleter = [&pkg_api](OrtModelPackageComponentContext* p) {
    if (p) pkg_api.ReleaseModelPackageComponentContext(p);
  };
  std::unique_ptr<OrtModelPackageOptions, decltype(options_deleter)> mp_opts(nullptr, options_deleter);
  std::unique_ptr<OrtModelPackageContext, decltype(context_deleter)> ctx(nullptr, context_deleter);
  std::unique_ptr<OrtModelPackageComponentContext, decltype(component_context_deleter)> comp_ctx(nullptr, component_context_deleter);

  OrtModelPackageOptions* raw_mp_opts = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api.CreateModelPackageOptionsFromSessionOptions(*ort_env, session_options, &raw_mp_opts));
  mp_opts.reset(raw_mp_opts);

  OrtModelPackageContext* raw_ctx = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api.CreateModelPackageContext(package_root.c_str(), &raw_ctx));
  ctx.reset(raw_ctx);

  OrtModelPackageComponentContext* raw_comp_ctx = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api.SelectComponent(ctx.get(), "model_1", mp_opts.get(), &raw_comp_ctx));
  comp_ctx.reset(raw_comp_ctx);

  // Pass nullptr for session_options so the metadata-merge path runs.
  OrtSession* raw_session = nullptr;
  OrtStatus* st = pkg_api.CreateSession(*ort_env, comp_ctx.get(), /*session_options=*/nullptr, &raw_session);
  if (raw_session != nullptr) {
    Ort::GetApi().ReleaseSession(raw_session);
    raw_session = nullptr;
  }
  ASSERT_NE(st, nullptr) << "CreateSession unexpectedly succeeded with malformed session.intra_op_num_threads";
  const std::string err_msg = Ort::GetApi().GetErrorMessage(st);
  Ort::GetApi().ReleaseStatus(st);

  const bool mentions_dispatch =
      err_msg.find("AddSessionConfigEntry") != std::string::npos ||
      err_msg.find("base-10 int32") != std::string::npos ||
      err_msg.find("intra_op_num_threads") != std::string::npos;
  EXPECT_TRUE(mentions_dispatch) << "error did not mention typed dispatch: " << err_msg;

  std::error_code ec;
  std::filesystem::remove_all(package_root, ec);
}

// A variant's path-valued session option (the external initializers folder) is resolved against
// the package (relative -> absolute) at parse time and applied, so the model's external data can
// live outside the model's own directory.
TEST(ModelPackageTest, VariantSessionOption_ResolvesExternalInitializersFolder) {
  const auto package_root = std::filesystem::temp_directory_path() / "ort_mp_ext_ini_resolve";
  std::vector<VariantSpec> variants;
  variants.push_back(VariantSpec{
      "variant_1", "example_ep", "cpu", "", "testdata/conv_qdq_external_ini.onnx", std::unordered_map<std::string, std::string>{
                                                                                       {kOrtSessionOptionsModelExternalInitializersFileFolderPath, "weights"},
                                                                                   },
      {}});
  BuildPackage(package_root, "model_1", variants);

  // Put the external data file in a subfolder of the variant dir. It can only be found if
  // "weights" is resolved to <variant_dir>/weights and used to override the model directory.
  const auto weights_dir = package_root / "model_1" / "variant_1" / "weights";
  std::error_code ec;
  std::filesystem::create_directories(weights_dir, ec);
  ASSERT_FALSE(ec);
  std::filesystem::copy_file("testdata/conv_qdq_external_ini.bin",
                             weights_dir / "conv_qdq_external_ini.bin",
                             std::filesystem::copy_options::overwrite_existing, ec);
  ASSERT_FALSE(ec);

  RegisteredEpDeviceUniquePtr example_ep;
  ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_info, example_ep));
  Ort::ConstEpDevice plugin_ep_device(example_ep.get());

  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> ep_options;
  session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);

  const auto& pkg_api = GetModelPackageFns();
  ASSERT_NE(pkg_api.CreateModelPackageContext, nullptr) << "Model package experimental API is not available";

  auto options_deleter = [&pkg_api](OrtModelPackageOptions* p) { if (p) pkg_api.ReleaseModelPackageOptions(p); };
  auto context_deleter = [&pkg_api](OrtModelPackageContext* p) { if (p) pkg_api.ReleaseModelPackageContext(p); };
  auto component_context_deleter = [&pkg_api](OrtModelPackageComponentContext* p) {
    if (p) pkg_api.ReleaseModelPackageComponentContext(p);
  };
  std::unique_ptr<OrtModelPackageOptions, decltype(options_deleter)> mp_opts(nullptr, options_deleter);
  std::unique_ptr<OrtModelPackageContext, decltype(context_deleter)> ctx(nullptr, context_deleter);
  std::unique_ptr<OrtModelPackageComponentContext, decltype(component_context_deleter)> comp_ctx(nullptr, component_context_deleter);

  OrtModelPackageOptions* raw_mp_opts = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api.CreateModelPackageOptionsFromSessionOptions(*ort_env, session_options, &raw_mp_opts));
  mp_opts.reset(raw_mp_opts);

  OrtModelPackageContext* raw_ctx = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api.CreateModelPackageContext(package_root.c_str(), &raw_ctx));
  ctx.reset(raw_ctx);

  OrtModelPackageComponentContext* raw_comp_ctx = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api.SelectComponent(ctx.get(), "model_1", mp_opts.get(), &raw_comp_ctx));
  comp_ctx.reset(raw_comp_ctx);

  // nullptr session_options -> metadata-merge (default) path applies the resolved folder option.
  // Session creation loads the external initializers during Initialize; it succeeds only if the
  // relative "weights" was resolved and used to override the model directory.
  OrtSession* raw_session = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api.CreateSession(*ort_env, comp_ctx.get(), /*session_options=*/nullptr, &raw_session));
  ASSERT_NE(raw_session, nullptr);
  Ort::Session session(raw_session);

  // Advanced path: pass the caller's own session options. The variant's resolved folder option is
  // still carried over (the caller did not set it), so external initializers still load.
  OrtSession* raw_session_adv = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api.CreateSession(*ort_env, comp_ctx.get(), session_options, &raw_session_adv));
  ASSERT_NE(raw_session_adv, nullptr);
  Ort::Session session_adv(raw_session_adv);

  std::filesystem::remove_all(package_root, ec);
}

// GetSelectedVariantFolderPath returns the correct path even when the variant
// declares no executor_info (i.e., no `file` descriptor for the variant).
TEST(ModelPackageApiTest, FolderPath_ReturnsCorrectPath_WhenExecutorInfoAbsent) {
  const auto package_root = std::filesystem::temp_directory_path() / "ort_mp_folder_path_no_executor_info";
  std::vector<VariantSpec> variants;
  // No source_model => no executor_info is emitted for this variant.
  VariantSpec only{"variant_1", "example_ep", "cpu", "", {}, {}, {}};
  variants.push_back(only);
  BuildPackage(package_root, "model_1", variants);

  // Drop a model file in the variant directory so the package looks plausible on disk.
  std::error_code ec;
  std::filesystem::copy_file("testdata/mul_1.onnx",
                             package_root / "model_1" / "variant_1" / "mul_1.onnx",
                             std::filesystem::copy_options::overwrite_existing, ec);

  RegisteredEpDeviceUniquePtr example_ep;
  ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_info, example_ep));
  Ort::ConstEpDevice plugin_ep_device(example_ep.get());

  Ort::SessionOptions so;
  std::unordered_map<std::string, std::string> ep_options;
  so.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);

  const auto& pkg_api = GetModelPackageFns();
  ASSERT_NE(pkg_api.CreateModelPackageContext, nullptr) << "Model package experimental API is not available";

  OrtModelPackageOptions* raw_mp_opts = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api.CreateModelPackageOptionsFromSessionOptions(*ort_env, so, &raw_mp_opts));
  auto options_deleter = [&pkg_api](OrtModelPackageOptions* p) { if (p) pkg_api.ReleaseModelPackageOptions(p); };
  std::unique_ptr<OrtModelPackageOptions, decltype(options_deleter)> mp_opts(raw_mp_opts, options_deleter);

  OrtModelPackageContext* raw_ctx = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api.CreateModelPackageContext(package_root.c_str(), &raw_ctx));
  auto context_deleter = [&pkg_api](OrtModelPackageContext* p) { if (p) pkg_api.ReleaseModelPackageContext(p); };
  std::unique_ptr<OrtModelPackageContext, decltype(context_deleter)> ctx(raw_ctx, context_deleter);

  OrtModelPackageComponentContext* raw_comp_ctx = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api.SelectComponent(ctx.get(), "model_1", mp_opts.get(), &raw_comp_ctx));
  auto component_context_deleter = [&pkg_api](OrtModelPackageComponentContext* p) {
    if (p) pkg_api.ReleaseModelPackageComponentContext(p);
  };
  std::unique_ptr<OrtModelPackageComponentContext, decltype(component_context_deleter)> comp_ctx(raw_comp_ctx, component_context_deleter);

  const ORTCHAR_T* selected_folder = nullptr;
  ASSERT_ORTSTATUS_OK(pkg_api.ModelPackageComponent_GetSelectedVariantFolderPath(comp_ctx.get(), &selected_folder));
  ASSERT_NE(selected_folder, nullptr);

  const auto result_path = std::filesystem::path(selected_folder);
  EXPECT_FALSE(result_path.empty());
  EXPECT_EQ(result_path.filename().string(), "variant_1");

  std::filesystem::remove_all(package_root, ec);
}

}  // namespace test
}  // namespace onnxruntime

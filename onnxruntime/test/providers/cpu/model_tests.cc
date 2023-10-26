// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iterator>
#include <gtest/gtest.h>

#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/ort_apis.h"
#include "core/session/inference_session.h"
#include "core/session/ort_env.h"
#include "core/providers/tensorrt/tensorrt_provider_options.h"
#include "asserts.h"
#include <core/platform/path_lib.h>
#include "default_providers.h"
#include "test/onnx/DisabledTests.h"
#include <string>
#include <codecvt>
#include <locale>

#ifdef USE_DNNL
#include "core/providers/dnnl/dnnl_provider_factory.h"
#endif

#ifdef USE_NNAPI
#include "core/providers/nnapi/nnapi_provider_factory.h"
#endif

#ifdef USE_RKNPU
#include "core/providers/rknpu/rknpu_provider_factory.h"
#endif

#ifdef USE_ACL
#include "core/providers/acl/acl_provider_factory.h"
#endif

#ifdef USE_ARMNN
#include "core/providers/armnn/armnn_provider_factory.h"
#endif

// test infrastructure
#include "test/onnx/testenv.h"
#include "test/onnx/TestCase.h"
#include "test/compare_ortvalue.h"
#include "test/onnx/heap_buffer.h"
#include "test/onnx/onnx_model_info.h"
#include "test/onnx/callback.h"
#include "test/onnx/testcase_request.h"

extern std::unique_ptr<Ort::Env> ort_env;

// asserts that the OrtStatus* result of `status_expr` does not indicate an error
// note: this takes ownership of the OrtStatus* result
#define ASSERT_ORT_STATUS_OK(status_expr)                                           \
  do {                                                                              \
    if (OrtStatus* _status = (status_expr); _status != nullptr) {                   \
      std::unique_ptr<OrtStatus, decltype(&OrtApis::ReleaseStatus)> _rel_status{    \
          _status, &OrtApis::ReleaseStatus};                                        \
      FAIL() << "OrtStatus error: " << OrtApis::GetErrorMessage(_rel_status.get()); \
    }                                                                               \
  } while (false)

using namespace onnxruntime::common;

namespace onnxruntime {
namespace test {
// parameter is provider_name + "_" + model_path
class ModelTest : public testing::TestWithParam<std::basic_string<ORTCHAR_T>> {};

#ifdef GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ModelTest);
#endif

void SkipTest(const std::string& reason = "") {
  GTEST_SKIP() << "Skipping single test " << reason;
}

TEST_P(ModelTest, Run) {
  std::basic_string<ORTCHAR_T> param = GetParam();
  size_t pos = param.find(ORT_TSTR("_"));
  ASSERT_NE(pos, std::string::npos);
  std::string provider_name = ToUTF8String(param.substr(0, pos));
  std::basic_string<ORTCHAR_T> model_path = param.substr(pos + 1);
  double per_sample_tolerance = 1e-3;
  double relative_per_sample_tolerance = 1e-3;

  // when cuda or openvino is enabled, set it to a larger value for resolving random MNIST test failure
  if (model_path.find(ORT_TSTR("_MNIST")) > 0) {
    if (provider_name == "cuda" || provider_name == "openvino") {
      relative_per_sample_tolerance = 1e-2;
    }
  }

  std::unique_ptr<OnnxModelInfo> model_info = std::make_unique<OnnxModelInfo>(model_path.c_str());

  if (model_info->HasDomain(ONNX_NAMESPACE::AI_ONNX_TRAINING_DOMAIN) ||
      model_info->HasDomain(ONNX_NAMESPACE::AI_ONNX_PREVIEW_TRAINING_DOMAIN)) {
    SkipTest("it has the training domain. No pipeline should need to run these tests.");
    return;
  }

  auto broken_tests = GetBrokenTests(provider_name);
  auto broken_tests_keyword_set = GetBrokenTestsKeyWordSet(provider_name);
  std::basic_string<ORTCHAR_T> model_dir;
  (void)GetDirNameFromFilePath(model_path, model_dir);
  std::basic_string<PATH_CHAR_TYPE> test_case_name = GetLastComponent(model_dir);
  if (test_case_name.compare(0, 5, ORT_TSTR("test_")) == 0)
    test_case_name = test_case_name.substr(5);
  {
    BrokenTest t = {ToUTF8String(test_case_name), ""};
    auto iter = broken_tests->find(t);
    auto opset_version = model_info->GetNominalOpsetVersion();
    if (iter != broken_tests->end() &&
        (opset_version == TestModelInfo::unknown_version || iter->broken_opset_versions_.empty() ||
         iter->broken_opset_versions_.find(opset_version) != iter->broken_opset_versions_.end())) {
      SkipTest("It's in broken_tests");
      return;
    }

    for (auto iter2 = broken_tests_keyword_set->begin(); iter2 != broken_tests_keyword_set->end(); ++iter2) {
      std::string keyword = *iter2;
      if (ToUTF8String(test_case_name).find(keyword) != std::string::npos) {
        SkipTest("It's in broken_tests_keyword");
        return;
      }
    }
  }

  // TODO(leca): move the parallel run test list to a config file and load it in GetParameterStrings() to make the load process run only once
  std::set<std::string> tests_run_parallel = {"test_resnet18v2",
                                              "test_resnet34v2",
                                              "test_resnet50",
                                              "test_resnet50v2",
                                              "test_resnet101v2",
                                              "test_resnet152v2",
                                              "keras_lotus_resnet3D",
                                              "coreml_Resnet50_ImageNet",
                                              "mlperf_mobilenet",
                                              "mlperf_resnet",
                                              "mlperf_ssd_mobilenet_300",
                                              "mlperf_ssd_resnet34_1200"};
  bool is_single_node = !model_info->GetNodeName().empty();
  std::vector<ExecutionMode> execution_modes = {ExecutionMode::ORT_SEQUENTIAL};
  if (provider_name == "cpu" && !is_single_node)
    execution_modes.push_back(ExecutionMode::ORT_PARALLEL);

  std::vector<bool> use_single_thread{false};
  // Test the model with intra op threadpool disabled
  if (provider_name == "cpu" && is_single_node)
    use_single_thread.push_back(true);

  std::unique_ptr<ITestCase> l = CreateOnnxTestCase(ToUTF8String(test_case_name), std::move(model_info),
                                                    per_sample_tolerance, relative_per_sample_tolerance);

#ifndef USE_DNNL
  auto tp = TestEnv::CreateThreadPool(Env::Default());
#endif

  for (bool is_single_thread : use_single_thread) {
    for (ExecutionMode execution_mode : execution_modes) {
      Ort::SessionOptions ortso{};
      if (!is_single_thread) {
        ortso.DisablePerSessionThreads();
      } else {
        ortso.SetIntraOpNumThreads(1);
      }
      ortso.SetExecutionMode(execution_mode);
      ortso.SetLogId(ToUTF8String(test_case_name).c_str());
      ortso.SetLogSeverityLevel(ORT_LOGGING_LEVEL_ERROR);
      if (provider_name == "cuda") {
        OrtCUDAProviderOptionsV2* cuda_options = nullptr;
        ASSERT_ORT_STATUS_OK(OrtApis::CreateCUDAProviderOptions(&cuda_options));
        std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(&OrtApis::ReleaseCUDAProviderOptions)> rel_cuda_options(
            cuda_options, &OrtApis::ReleaseCUDAProviderOptions);
        std::vector<const char*> keys{"device_id"};

        std::vector<const char*> values;
        std::string device_id = Env::Default().GetEnvironmentVar("ONNXRUNTIME_TEST_GPU_DEVICE_ID");
        values.push_back(device_id.empty() ? "0" : device_id.c_str());
        ASSERT_ORT_STATUS_OK(OrtApis::UpdateCUDAProviderOptions(cuda_options, keys.data(), values.data(), 1));
        ortso.AppendExecutionProvider_CUDA_V2(*cuda_options);
      } else if (provider_name == "rocm") {
        OrtROCMProviderOptions ep_options;
        ortso.AppendExecutionProvider_ROCM(ep_options);
      }
#ifdef USE_DNNL
      else if (provider_name == "dnnl") {
        OrtDnnlProviderOptions* ep_option;
        ASSERT_ORT_STATUS_OK(OrtApis::CreateDnnlProviderOptions(&ep_option));
        std::unique_ptr<OrtDnnlProviderOptions, decltype(&OrtApis::ReleaseDnnlProviderOptions)>
            rel_dnnl_options(ep_option, &OrtApis::ReleaseDnnlProviderOptions);
        ep_option->use_arena = 0;
        ASSERT_ORT_STATUS_OK(OrtApis::SessionOptionsAppendExecutionProvider_Dnnl(ortso, ep_option));
      }
#endif
      else if (provider_name == "tensorrt") {
        if (test_case_name.find(ORT_TSTR("FLOAT16")) != std::string::npos) {
          OrtTensorRTProviderOptionsV2 params;
          ortso.AppendExecutionProvider_TensorRT_V2(params);
        } else {
          OrtTensorRTProviderOptionsV2* ep_option = nullptr;
          ASSERT_ORT_STATUS_OK(OrtApis::CreateTensorRTProviderOptions(&ep_option));
          std::unique_ptr<OrtTensorRTProviderOptionsV2, decltype(&OrtApis::ReleaseTensorRTProviderOptions)>
              rel_cuda_options(ep_option, &OrtApis::ReleaseTensorRTProviderOptions);
          ortso.AppendExecutionProvider_TensorRT_V2(*ep_option);
        }
        // Enable CUDA fallback
        OrtCUDAProviderOptionsV2* cuda_options = nullptr;
        ASSERT_ORT_STATUS_OK(OrtApis::CreateCUDAProviderOptions(&cuda_options));
        std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(&OrtApis::ReleaseCUDAProviderOptions)> rel_cuda_options(
            cuda_options, &OrtApis::ReleaseCUDAProviderOptions);
        ortso.AppendExecutionProvider_CUDA_V2(*cuda_options);
      } else if (provider_name == "migraphx") {
        OrtMIGraphXProviderOptions ep_options;
        ortso.AppendExecutionProvider_MIGraphX(ep_options);
      } else if (provider_name == "openvino") {
        OrtOpenVINOProviderOptions ep_options;
        ortso.AppendExecutionProvider_OpenVINO(ep_options);
      }
#ifdef USE_NNAPI
      else if (provider_name == "nnapi") {
        ASSERT_ORT_STATUS_OK(OrtSessionOptionsAppendExecutionProvider_Nnapi(ortso, 0));
      }
#endif
#ifdef USE_RKNPU
      else if (provider_name == "rknpu") {
        ASSERT_ORT_STATUS_OK(OrtSessionOptionsAppendExecutionProvider_Rknpu(ortso));
      }
#endif
#ifdef USE_ACL
      else if (provider_name == "acl") {
        ASSERT_ORT_STATUS_OK(OrtSessionOptionsAppendExecutionProvider_ACL(ortso, 0));
      }
#endif
#ifdef USE_ARMNN
      else if (provider_name == "armnn") {
        ASSERT_ORT_STATUS_OK(OrtSessionOptionsAppendExecutionProvider_ArmNN(ortso));
      }
#endif
      OrtSession* ort_session;
      OrtStatus* ort_st = OrtApis::CreateSession(*ort_env, model_path.c_str(), ortso, &ort_session);
      if (ort_st != nullptr) {
        OrtErrorCode error_code = OrtApis::GetErrorCode(ort_st);
        if (error_code == ORT_NOT_IMPLEMENTED) {
          OrtApis::ReleaseStatus(ort_st);
          continue;
        }
        FAIL() << OrtApis::GetErrorMessage(ort_st);
      }
      std::unique_ptr<OrtSession, decltype(&OrtApis::ReleaseSession)> rel_ort_session(ort_session,
                                                                                      &OrtApis::ReleaseSession);
      const size_t data_count = l->GetDataCount();
#ifndef USE_DNNL  // potential crash for DNNL pipeline
      if (data_count > 1 && tests_run_parallel.find(l->GetTestCaseName()) != tests_run_parallel.end()) {
        LOGS_DEFAULT(ERROR) << "Parallel test for " << l->GetTestCaseName();  // TODO(leca): change level to INFO or even delete the log once verified parallel test working
        std::shared_ptr<TestCaseResult> results = TestCaseRequestContext::Run(tp.get(), *l, *ort_env, ortso, data_count, 1 /*repeat_count*/);
        for (EXECUTE_RESULT res : results->GetExcutionResult()) {
          EXPECT_EQ(res, EXECUTE_RESULT::SUCCESS) << "is_single_thread:" << is_single_thread << ", execution_mode:" << execution_mode << ", provider_name:"
                                                  << provider_name << ", test name:" << results->GetName() << ", result: " << res;
        }
        continue;
      }
#endif  // !USE_DNNL
      // TODO(leca): leverage TestCaseRequestContext::Run() to make it short
      auto default_allocator = std::make_unique<MockedOrtAllocator>();

      for (size_t task_id = 0; task_id != data_count; ++task_id) {
        onnxruntime::test::HeapBuffer holder;
        std::unordered_map<std::string, Ort::Value> feeds;
        l->LoadTestData(task_id, holder, feeds, true);
        size_t output_count;
        ASSERT_ORT_STATUS_OK(OrtApis::SessionGetOutputCount(ort_session, &output_count));
        // Create output feed
        std::vector<char*> output_names(output_count);
        for (size_t i = 0; i != output_count; ++i) {
          ASSERT_ORT_STATUS_OK(
              OrtApis::SessionGetOutputName(ort_session, i, default_allocator.get(), &output_names[i]));
        }

        std::vector<const char*> input_names;
        std::vector<OrtValue*> input_values;
        std::vector<OrtValue*> output_values(output_count);
        {
          for (auto& p : feeds) {
            input_names.push_back(p.first.c_str());
            input_values.push_back(p.second);
          }
          ort_st = OrtApis::Run(ort_session, nullptr, input_names.data(), input_values.data(), input_values.size(),
                                output_names.data(), output_names.size(), output_values.data());
          if (ort_st != nullptr) {
            OrtErrorCode error_code = OrtApis::GetErrorCode(ort_st);
            if (error_code == ORT_NOT_IMPLEMENTED) {
              OrtApis::ReleaseStatus(ort_st);
              for (char* p : output_names) {
                default_allocator->Free(p);
              }
              for (OrtValue* v : output_values) {
                OrtApis::ReleaseValue(v);
              }
            }
            FAIL() << OrtApis::GetErrorMessage(ort_st);
          }
        }

        bool post_procesing = false;
        Status status;
        l->GetPerSampleTolerance(&per_sample_tolerance);
        l->GetRelativePerSampleTolerance(&relative_per_sample_tolerance);
        l->GetPostProcessing(&post_procesing);

        // TODO: if there are no output value files, just skip the validation
        std::unordered_map<std::string, Ort::Value> expected_output_values;
        l->LoadTestData(task_id, holder, expected_output_values, false);

        std::unordered_map<std::string, OrtValue*> name_fetch_output_map;
        std::unordered_map<std::string, const ONNX_NAMESPACE::ValueInfoProto*> name_output_value_info_proto;
        size_t i = 0;
        for (auto& output_name : output_names) {
          // p_fetches is filled in the order of output_names.
          name_fetch_output_map[output_name] = output_values[i];
          const ONNX_NAMESPACE::ValueInfoProto* infoProto = l->GetOutputInfoFromModel(i);
          if (infoProto != nullptr)
            name_output_value_info_proto.insert(std::make_pair(infoProto->name(), infoProto));
          i++;
        }

        for (auto& output : expected_output_values) {
          const OrtValue* expected_output_value = output.second;
          const std::string& output_name = output.first;
          auto iter = name_fetch_output_map.find(output_name);
          ASSERT_NE(iter, name_fetch_output_map.end());

          OrtValue* actual_output_value = iter->second;
          std::pair<COMPARE_RESULT, std::string> ret =
              CompareOrtValue(*actual_output_value, *expected_output_value, per_sample_tolerance,
                              relative_per_sample_tolerance, post_procesing);
          COMPARE_RESULT compare_result = ret.first;
          ASSERT_EQ(COMPARE_RESULT::SUCCESS, ret.first) << ret.second;

          const ONNX_NAMESPACE::ValueInfoProto* v = name_output_value_info_proto[output_name];
          if (v == nullptr)
            continue;
          ret = VerifyValueInfo(*v, actual_output_value);
          compare_result = ret.first;
          ASSERT_EQ(COMPARE_RESULT::SUCCESS, ret.first) << ret.second;

          if (compare_result != COMPARE_RESULT::SUCCESS) {
            break;
          }
        }
        for (char* p : output_names) {
          default_allocator->Free(p);
        }
        for (OrtValue* v : output_values) {
          OrtApis::ReleaseValue(v);
        }
      }
    }
  }
}

::std::vector<::std::basic_string<ORTCHAR_T>> GetParameterStrings() {
  // Map key is provider name(CPU, CUDA, etc). Value is the ONNX node tests' opsets to run.
  std::map<ORT_STRING_VIEW, std::vector<ORT_STRING_VIEW>> provider_names;
  // The default CPU provider always supports all opsets, and must maintain backwards compatibility.
  provider_names[provider_name_cpu] = {opset7, opset8, opset9, opset10, opset11, opset12, opset13, opset14, opset15, opset16, opset17, opset18};
  // The other EPs can choose which opsets to test.
  // If an EP doesn't have any CI build pipeline, then there is no need to specify any opset.
#ifdef USE_TENSORRT
  // tensorrt: only enable opset 14 to 17 of onnx tests
  provider_names[provider_name_tensorrt] = {opset14, opset15, opset16, opset17};
#endif
#ifdef USE_MIGRAPHX
  provider_names[provider_name_migraphx] = {opset7, opset8, opset9, opset10, opset11, opset12, opset13, opset14, opset15, opset16, opset17, opset18};
#endif
#ifdef USE_OPENVINO
  provider_names[provider_name_openvino] = {};
#endif
#ifdef USE_CUDA
  provider_names[provider_name_cuda] = {opset7, opset8, opset9, opset10, opset11, opset12, opset13, opset14, opset15, opset16, opset17, opset18};
#endif
#ifdef USE_ROCM
  provider_names[provider_name_rocm] = {opset7, opset8, opset9, opset10, opset11, opset12, opset13, opset14, opset15, opset16, opset17, opset18};
#endif
#ifdef USE_DNNL
  provider_names[provider_name_dnnl] = {opset10};
#endif
// For any non-Android system, NNAPI will only be used for ort model converter
#if defined(USE_NNAPI) && defined(__ANDROID__)
  provider_names[provider_name_nnapi] = {opset7, opset8, opset9, opset10, opset11, opset12, opset13, opset14, opset15, opset16, opset17, opset18};
#endif
#ifdef USE_RKNPU
  provider_names[provider_name_rknpu] = {};
#endif
#ifdef USE_ACL
  provider_names[provider_name_acl] = {};
#endif
#ifdef USE_ARMNN
  provider_names[provider_name_armnn] = {};
#endif
#ifdef USE_DML
  provider_names[provider_name_dml] = {opset7, opset8, opset9, opset10, opset11, opset12, opset13, opset14, opset15, opset16, opset17, opset18};
#endif
  std::vector<std::basic_string<ORTCHAR_T>> v;

  std::vector<std::basic_string<ORTCHAR_T>> paths;

  for (std::pair<ORT_STRING_VIEW, std::vector<ORT_STRING_VIEW>> kvp : provider_names) {
    // Setup ONNX node tests. The test data is preloaded on our CI build machines.
#if !defined(_WIN32)
    ORT_STRING_VIEW node_test_root_path = ORT_TSTR("/data/onnx");
#else
    ORT_STRING_VIEW node_test_root_path = ORT_TSTR("c:\\local\\data\\onnx");
#endif
    for (auto p : kvp.second) {
      paths.push_back(ConcatPathComponent(node_test_root_path, p));
    }

    // Same as the above, except this one is for large models
#if defined(NDEBUG) || defined(RUN_MODELTEST_IN_DEBUG_MODE)
#ifdef _WIN32
    ORT_STRING_VIEW model_test_root_path = ORT_TSTR("..\\models");
    // thus, only the root path should be mounted.
    ORT_STRING_VIEW model_zoo_path = ORT_TSTR("..\\models\\zoo");
#else
    ORT_STRING_VIEW model_test_root_path = ORT_TSTR("../models");
    ORT_STRING_VIEW model_zoo_path = ORT_TSTR("../models/zoo");
#endif
    for (auto p : kvp.second) {
      paths.push_back(ConcatPathComponent(model_test_root_path, p));
      paths.push_back(ConcatPathComponent(model_zoo_path, p));
    }
#endif

    ORT_STRING_VIEW provider_name = kvp.first;
    std::unordered_set<std::basic_string<ORTCHAR_T>> all_disabled_tests = GetAllDisabledTests(provider_name);

        while (!paths.empty()) {
      std::basic_string<ORTCHAR_T> node_data_root_path = paths.back();
      paths.pop_back();
      std::basic_string<ORTCHAR_T> my_dir_name = GetLastComponent(node_data_root_path);
      ORT_TRY {
        LoopDir(node_data_root_path, [&](const ORTCHAR_T* filename, OrtFileType f_type) -> bool {
          if (filename[0] == ORT_TSTR('.'))
            return true;
          if (f_type == OrtFileType::TYPE_DIR) {
            std::basic_string<PATH_CHAR_TYPE> p = ConcatPathComponent(node_data_root_path, filename);
            paths.push_back(p);
            return true;
          }
          std::basic_string<PATH_CHAR_TYPE> filename_str = filename;
          if (!HasExtensionOf(filename_str, ORT_TSTR("onnx")))
            return true;
          std::basic_string<PATH_CHAR_TYPE> test_case_name = my_dir_name;
          if (test_case_name.compare(0, 5, ORT_TSTR("test_")) == 0)
            test_case_name = test_case_name.substr(5);
          if (all_disabled_tests.find(test_case_name) != all_disabled_tests.end())
            return true;

#ifdef DISABLE_ML_OPS
          auto starts_with = [](const std::basic_string<PATH_CHAR_TYPE>& find_in,
                                const std::basic_string<PATH_CHAR_TYPE>& find_what) {
            return find_in.compare(0, find_what.size(), find_what) == 0;
          };
          if (starts_with(test_case_name, ORT_TSTR("XGBoost_")) || starts_with(test_case_name, ORT_TSTR("coreml_")) ||
              starts_with(test_case_name, ORT_TSTR("scikit_")) || starts_with(test_case_name, ORT_TSTR("libsvm_"))) {
            return true;
          }
#endif
          std::basic_ostringstream<PATH_CHAR_TYPE> oss;
          oss << provider_name << ORT_TSTR("_") << ConcatPathComponent(node_data_root_path, filename_str);
          v.emplace_back(oss.str());
          return true;
        });
      }
      ORT_CATCH(const std::exception&) {
      }  // ignore non-exist dir
    }
  }
  return v;
}

auto ExpandModelName = [](const ::testing::TestParamInfo<ModelTest::ParamType>& info) {
  // use info.param here to generate the test suffix
  std::basic_string<ORTCHAR_T> name = info.param;

  // the original name here is the combination of provider name and model path name
  // remove the trailing 'xxxxxxx/model.onnx' of name
  if (name.size() > 11 && name.substr(name.size() - 11) == ORT_TSTR("/model.onnx")) {
    name = name.substr(0, info.param.size() - 11);
  }
  // remove the trailing 'xxxxxx.onnx' of name
  else if (name.size() > 5 && name.substr(name.size() - 5) == ORT_TSTR(".onnx")) {
    name = name.substr(0, info.param.size() - 5);
  }

  // Note: test name only accepts '_' and alphanumeric
  // replace '/' or '\' with '_'
  std::replace(name.begin(), name.end(), '/', '_');
  std::replace(name.begin(), name.end(), '\\', '_');

  // in case there's whitespace in directory name
  std::replace(name.begin(), name.end(), ' ', '_');

  // Note: test name only accepts '_' and alphanumeric
  // remove '.', '-', ':'
  char chars[] = ".-:()";
  for (unsigned int i = 0; i < strlen(chars); ++i) {
    name.erase(std::remove(name.begin(), name.end(), chars[i]), name.end());
  }
#ifdef _WIN32
  // Note: The return value of INSTANTIATE_TEST_SUITE_P accepts std::basic_string<char...>.
  // Need conversion of wchar_t to char.
  return std::wstring_convert<std::codecvt_utf8<wchar_t>>().to_bytes(name);
#else
  return name;
#endif
};

// The optional last argument is a function or functor that generates custom test name suffixes based on the test
// parameters. Specify the last argument to make test name more meaningful and clear instead of just the sequential
// number.
INSTANTIATE_TEST_SUITE_P(ModelTests, ModelTest, testing::ValuesIn(GetParameterStrings()), ExpandModelName);

}  // namespace test
}  // namespace onnxruntime

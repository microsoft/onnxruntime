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
#include "test/onnx/TestCase.h"
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

using ORT_STRING_VIEW = std::basic_string_view<ORTCHAR_T>;
static ORT_STRING_VIEW opset7 = ORT_TSTR("opset7");
static ORT_STRING_VIEW opset8 = ORT_TSTR("opset8");
static ORT_STRING_VIEW opset9 = ORT_TSTR("opset9");
static ORT_STRING_VIEW opset10 = ORT_TSTR("opset10");
static ORT_STRING_VIEW opset11 = ORT_TSTR("opset11");
static ORT_STRING_VIEW opset12 = ORT_TSTR("opset12");
static ORT_STRING_VIEW opset13 = ORT_TSTR("opset13");
static ORT_STRING_VIEW opset14 = ORT_TSTR("opset14");
static ORT_STRING_VIEW opset15 = ORT_TSTR("opset15");
static ORT_STRING_VIEW opset16 = ORT_TSTR("opset16");
static ORT_STRING_VIEW opset17 = ORT_TSTR("opset17");
static ORT_STRING_VIEW opset18 = ORT_TSTR("opset18");
// TODO: enable opset19 tests
// static ORT_STRING_VIEW opset19 = ORT_TSTR("opset19");

static ORT_STRING_VIEW provider_name_cpu = ORT_TSTR("cpu");
static ORT_STRING_VIEW provider_name_tensorrt = ORT_TSTR("tensorrt");
#ifdef USE_MIGRAPHX
static ORT_STRING_VIEW provider_name_migraphx = ORT_TSTR("migraphx");
#endif
static ORT_STRING_VIEW provider_name_openvino = ORT_TSTR("openvino");
static ORT_STRING_VIEW provider_name_cuda = ORT_TSTR("cuda");
#ifdef USE_ROCM
static ORT_STRING_VIEW provider_name_rocm = ORT_TSTR("rocm");
#endif
static ORT_STRING_VIEW provider_name_dnnl = ORT_TSTR("dnnl");
// For any non-Android system, NNAPI will only be used for ort model converter
#if defined(USE_NNAPI) && defined(__ANDROID__)
static ORT_STRING_VIEW provider_name_nnapi = ORT_TSTR("nnapi");
#endif
#ifdef USE_RKNPU
static ORT_STRING_VIEW provider_name_rknpu = ORT_TSTR("rknpu");
#endif
#ifdef USE_ACL
static ORT_STRING_VIEW provider_name_acl = ORT_TSTR("acl");
#endif
#ifdef USE_ARMNN
static ORT_STRING_VIEW provider_name_armnn = ORT_TSTR("armnn");
#endif
static ORT_STRING_VIEW provider_name_dml = ORT_TSTR("dml");

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

#if defined(ENABLE_TRAINING_CORE) && defined(USE_CUDA)
  // Removing the CPU EP tests from CUDA build for training as these tests are already run in the CPU pipelines.
  // Note: These are inference tests, we run these in training builds as an extra check. Therefore reducing
  // the number of times these are run to reduce the CI time.
  provider_names.erase(provider_name_cpu);
#endif
  std::vector<std::basic_string<ORTCHAR_T>> v;
  // Permanently exclude following tests because ORT support only opset starting from 7,
  // Please make no more changes to the list
  static const ORTCHAR_T* immutable_broken_tests[] = {
      ORT_TSTR("AvgPool1d"),
      ORT_TSTR("AvgPool1d_stride"),
      ORT_TSTR("AvgPool2d"),
      ORT_TSTR("AvgPool2d_stride"),
      ORT_TSTR("AvgPool3d"),
      ORT_TSTR("AvgPool3d_stride"),
      ORT_TSTR("AvgPool3d_stride1_pad0_gpu_input"),
      ORT_TSTR("BatchNorm1d_3d_input_eval"),
      ORT_TSTR("BatchNorm2d_eval"),
      ORT_TSTR("BatchNorm2d_momentum_eval"),
      ORT_TSTR("BatchNorm3d_eval"),
      ORT_TSTR("BatchNorm3d_momentum_eval"),
      ORT_TSTR("GLU"),
      ORT_TSTR("GLU_dim"),
      ORT_TSTR("Linear"),
      ORT_TSTR("PReLU_1d"),
      ORT_TSTR("PReLU_1d_multiparam"),
      ORT_TSTR("PReLU_2d"),
      ORT_TSTR("PReLU_2d_multiparam"),
      ORT_TSTR("PReLU_3d"),
      ORT_TSTR("PReLU_3d_multiparam"),
      ORT_TSTR("PoissonNLLLLoss_no_reduce"),
      ORT_TSTR("Softsign"),
      ORT_TSTR("operator_add_broadcast"),
      ORT_TSTR("operator_add_size1_broadcast"),
      ORT_TSTR("operator_add_size1_right_broadcast"),
      ORT_TSTR("operator_add_size1_singleton_broadcast"),
      ORT_TSTR("operator_addconstant"),
      ORT_TSTR("operator_addmm"),
      ORT_TSTR("operator_basic"),
      ORT_TSTR("operator_mm"),
      ORT_TSTR("operator_non_float_params"),
      ORT_TSTR("operator_params"),
      ORT_TSTR("operator_pow"),
  };

  static const ORTCHAR_T* cuda_flaky_tests[] = {ORT_TSTR("fp16_inception_v1"),
                                                ORT_TSTR("fp16_shufflenet"),
                                                ORT_TSTR("fp16_tiny_yolov2"),
                                                ORT_TSTR("candy"),
                                                ORT_TSTR("tinyyolov3"),
                                                ORT_TSTR("mlperf_ssd_mobilenet_300"),
                                                ORT_TSTR("mlperf_ssd_resnet34_1200"),
                                                ORT_TSTR("tf_inception_v1"),
                                                ORT_TSTR("faster_rcnn"),
                                                ORT_TSTR("split_zero_size_splits"),
                                                ORT_TSTR("convtranspose_3d"),
                                                ORT_TSTR("fp16_test_tiny_yolov2-Candy"),
                                                ORT_TSTR("fp16_coreml_FNS-Candy"),
                                                ORT_TSTR("fp16_test_tiny_yolov2"),
                                                ORT_TSTR("fp16_test_shufflenet"),
                                                ORT_TSTR("keras2coreml_SimpleRNN_ImageNet")};
  static const ORTCHAR_T* openvino_disabled_tests[] = {
      ORT_TSTR("tf_mobilenet_v1_1.0_224"),
      ORT_TSTR("bertsquad"),
      ORT_TSTR("yolov3"),
      ORT_TSTR("LSTM_Seq_lens_unpacked"),
      ORT_TSTR("tinyyolov3"),
      ORT_TSTR("faster_rcnn"),
      ORT_TSTR("mask_rcnn"),
      ORT_TSTR("coreml_FNS-Candy_ImageNet"),
      ORT_TSTR("tf_mobilenet_v2_1.0_224"),
      ORT_TSTR("tf_mobilenet_v2_1.4_224"),
      ORT_TSTR("operator_permute2"),
      ORT_TSTR("operator_repeat"),
      ORT_TSTR("operator_repeat_dim_overflow"),
      ORT_TSTR("mlperf_ssd_resnet34_1200"),
      ORT_TSTR("candy"),
      ORT_TSTR("cntk_simple_seg"),
      ORT_TSTR("GPT2_LM_HEAD"),
      ORT_TSTR("mlperf_ssd_mobilenet_300"),
      ORT_TSTR("fp16_coreml_FNS-Candy"),
      ORT_TSTR("fp16_test_tiny_yolov2"),
      ORT_TSTR("negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_mean_weight"),
      ORT_TSTR("negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_mean_weight_expanded"),
      ORT_TSTR("negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_none_no_weight"),
      ORT_TSTR("negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_none_no_weight_expanded"),
      ORT_TSTR("softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight"),
      ORT_TSTR("softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_expanded"),
      ORT_TSTR("softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_log_prob"),
      ORT_TSTR("softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_log_prob_expanded"),
      ORT_TSTR("softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight"),
      ORT_TSTR("softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_expanded"),
      ORT_TSTR("softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_log_prob"),
      ORT_TSTR("softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_log_prob_expanded"),
      // models from model zoo
      ORT_TSTR("Tiny YOLOv3"),
      ORT_TSTR("BERT-Squad"),
      ORT_TSTR("YOLOv3"),
      ORT_TSTR("Candy"),
      ORT_TSTR("SSD"),
      ORT_TSTR("ResNet101_DUC_HDC-12"),
      ORT_TSTR("YOLOv3-12")};
  static const ORTCHAR_T* dml_disabled_tests[] = {ORT_TSTR("mlperf_ssd_resnet34_1200"),
                                                  ORT_TSTR("mlperf_ssd_mobilenet_300"),
                                                  ORT_TSTR("mask_rcnn"),
                                                  ORT_TSTR("faster_rcnn"),
                                                  ORT_TSTR("tf_pnasnet_large"),
                                                  ORT_TSTR("zfnet512"),
                                                  ORT_TSTR("keras2coreml_Dense_ImageNet")};
  static const ORTCHAR_T* dnnl_disabled_tests[] = {ORT_TSTR("densenet121"),
                                                   ORT_TSTR("resnet18v2"),
                                                   ORT_TSTR("resnet34v2"),
                                                   ORT_TSTR("resnet50v2"),
                                                   ORT_TSTR("resnet101v2"),
                                                   ORT_TSTR("resnet101v2"),
                                                   ORT_TSTR("vgg19"),
                                                   ORT_TSTR("tf_inception_resnet_v2"),
                                                   ORT_TSTR("tf_inception_v1"),
                                                   ORT_TSTR("tf_inception_v3"),
                                                   ORT_TSTR("tf_inception_v4"),
                                                   ORT_TSTR("tf_mobilenet_v1_1.0_224"),
                                                   ORT_TSTR("tf_mobilenet_v2_1.0_224"),
                                                   ORT_TSTR("tf_mobilenet_v2_1.4_224"),
                                                   ORT_TSTR("tf_nasnet_large"),
                                                   ORT_TSTR("tf_pnasnet_large"),
                                                   ORT_TSTR("tf_resnet_v1_50"),
                                                   ORT_TSTR("tf_resnet_v1_101"),
                                                   ORT_TSTR("tf_resnet_v1_101"),
                                                   ORT_TSTR("tf_resnet_v2_101"),
                                                   ORT_TSTR("tf_resnet_v2_152"),
                                                   ORT_TSTR("batchnorm_example_training_mode"),
                                                   ORT_TSTR("batchnorm_epsilon_training_mode"),
                                                   ORT_TSTR("mobilenetv2-1.0"),
                                                   ORT_TSTR("shufflenet"),
                                                   ORT_TSTR("candy"),
                                                   ORT_TSTR("range_float_type_positive_delta_expanded"),
                                                   ORT_TSTR("range_int32_type_negative_delta_expanded"),
                                                   ORT_TSTR("averagepool_2d_ceil"),
                                                   ORT_TSTR("maxpool_2d_ceil"),
                                                   ORT_TSTR("maxpool_2d_dilations"),
                                                   ORT_TSTR("mlperf_ssd_resnet34_1200"),
                                                   ORT_TSTR("convtranspose_1d"),
                                                   ORT_TSTR("convtranspose_3d"),
                                                   ORT_TSTR("maxpool_2d_uint8"),
                                                   ORT_TSTR("mul_uint8"),
                                                   ORT_TSTR("div_uint8")};
  static const ORTCHAR_T* tensorrt_disabled_tests[] = {
      ORT_TSTR("size")  // INVALID_ARGUMENT: Cannot find binding of given name: x
  };
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
#if defined(NDEBUG) || defined(RUN_MODELTEST_IN_DEBUG_MODE) || defined(USE_TENSORRT)
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
    std::unordered_set<std::basic_string<ORTCHAR_T>> all_disabled_tests(std::begin(immutable_broken_tests),
                                                                        std::end(immutable_broken_tests));
    if (provider_name == provider_name_cuda) {
      all_disabled_tests.insert(std::begin(cuda_flaky_tests), std::end(cuda_flaky_tests));
    } else if (provider_name == provider_name_dml) {
      all_disabled_tests.insert(std::begin(dml_disabled_tests), std::end(dml_disabled_tests));
    } else if (provider_name == provider_name_dnnl) {
      // these models run but disabled tests to keep memory utilization low
      // This will be removed after LRU implementation
      all_disabled_tests.insert(std::begin(dnnl_disabled_tests), std::end(dnnl_disabled_tests));
    } else if (provider_name == provider_name_tensorrt) {
      // these models run but disabled tests to keep memory utilization low
      // This will be removed after LRU implementation
      all_disabled_tests.insert(std::begin(tensorrt_disabled_tests), std::end(tensorrt_disabled_tests));
    } else if (provider_name == provider_name_openvino) {
      // these models run but disabled tests to keep memory utilization low
      // This will be removed after LRU implementation
      all_disabled_tests.insert(std::begin(openvino_disabled_tests), std::end(openvino_disabled_tests));
    }

#if !defined(__amd64__) && !defined(_M_AMD64)
    // out of memory
    static const ORTCHAR_T* x86_disabled_tests[] = {ORT_TSTR("BERT_Squad"),
                                                    ORT_TSTR("bvlc_alexnet"),
                                                    ORT_TSTR("bvlc_reference_caffenet"),
                                                    ORT_TSTR("coreml_VGG16_ImageNet"),
                                                    ORT_TSTR("VGG 16-fp32"),
                                                    ORT_TSTR("VGG 19-caffe2"),
                                                    ORT_TSTR("VGG 19-bn"),
                                                    ORT_TSTR("VGG 16-bn"),
                                                    ORT_TSTR("VGG 19"),
                                                    ORT_TSTR("VGG 16"),
                                                    ORT_TSTR("faster_rcnn"),
                                                    ORT_TSTR("GPT2"),
                                                    ORT_TSTR("GPT2_LM_HEAD"),
                                                    ORT_TSTR("keras_lotus_resnet3D"),
                                                    ORT_TSTR("mlperf_ssd_resnet34_1200"),
                                                    ORT_TSTR("mask_rcnn_keras"),
                                                    ORT_TSTR("mask_rcnn"),
                                                    ORT_TSTR("ssd"),
                                                    ORT_TSTR("vgg19"),
                                                    ORT_TSTR("zfnet512"),
                                                    ORT_TSTR("ResNet101_DUC_HDC"),
                                                    ORT_TSTR("ResNet101_DUC_HDC-12"),
                                                    ORT_TSTR("FCN ResNet-101"),
                                                    ORT_TSTR("SSD")};
    all_disabled_tests.insert(std::begin(x86_disabled_tests), std::end(x86_disabled_tests));
#endif
    // fp16 models have different outputs with different kinds of hardware. We need to disable all fp16 models
    all_disabled_tests.insert(ORT_TSTR("fp16_shufflenet"));
    all_disabled_tests.insert(ORT_TSTR("fp16_inception_v1"));
    all_disabled_tests.insert(ORT_TSTR("fp16_tiny_yolov2"));

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

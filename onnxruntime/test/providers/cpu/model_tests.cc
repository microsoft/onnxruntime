// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/inference_session.h"
#include "core/session/ort_env.h"
#include "asserts.h"
#include <iterator>
#include "gtest/gtest.h"
#include <core/platform/path_lib.h>
#include "default_providers.h"

// test infrastructure
#include "test/onnx/TestCase.h"
#include "test/compare_ortvalue.h"
#include "test/onnx/heap_buffer.h"
#include "test/onnx/onnx_model_info.h"
#include "test/onnx/callback.h"
#include "test/onnx/test_filters.h"

extern std::unique_ptr<Ort::Env> ort_env;

using namespace onnxruntime::common;

namespace onnxruntime {
namespace test {
// parameter is provider_name + "_" + model_path
class ModelTest : public testing::TestWithParam<std::basic_string<ORTCHAR_T>> {};

TEST_P(ModelTest, Run) {
  std::basic_string<ORTCHAR_T> param = GetParam();
  size_t pos = param.find(ORT_TSTR("_"));
  ASSERT_NE(pos, std::string::npos);
  std::string provider_name = ToMBString(param.substr(0, pos));
  std::basic_string<ORTCHAR_T> model_path = param.substr(pos + 1);
  double per_sample_tolerance = 1e-3;
  // when cuda is enabled, set it to a larger value for resolving random MNIST test failure
  // when openvino is enabled, set it to a larger value for resolving MNIST accuracy mismatch
  double relative_per_sample_tolerance = 1e-3;
  if (provider_name == "openvino") {
    relative_per_sample_tolerance = 0.009;
  }

  std::unique_ptr<OnnxModelInfo> model_info = onnxruntime::make_unique<OnnxModelInfo>(model_path.c_str());
  if (model_info->GetONNXOpSetVersion() != 8 && provider_name == "tensorrt") {
    // TensorRT can run most of the model tests, but only part of
    // them is enabled here to save CI build time.
    return;
  }
  if (model_info->GetONNXOpSetVersion() == 10 && provider_name == "dnnl") {
    // DNNL can run most of the model tests, but only part of
    // them is enabled here to save CI build time.
    return;
  }
#ifndef ENABLE_TRAINING
  if (model_info->HasDomain(ONNX_NAMESPACE::AI_ONNX_TRAINING_DOMAIN) ||
      model_info->HasDomain(ONNX_NAMESPACE::AI_ONNX_PREVIEW_TRAINING_DOMAIN)) {
    return;
  }
#endif
  std::set<BrokenTest> broken_tests = GetBrokenTestsForProvider(provider_name);
  std::basic_string<ORTCHAR_T> model_dir;
  (void)GetDirNameFromFilePath(model_path, model_dir);
  std::basic_string<PATH_CHAR_TYPE> test_case_name = GetLastComponent(model_dir);
  if (test_case_name.compare(0, 5, ORT_TSTR("test_")) == 0)
    test_case_name = test_case_name.substr(5);
  {
    BrokenTest t = {ToMBString(test_case_name), ""};
    auto iter = broken_tests.find(t);
    auto model_version = model_info->GetModelVersion();
    if (iter != broken_tests.end() &&
        (model_version == TestModelInfo::unknown_version || iter->broken_versions_.empty() ||
         iter->broken_versions_.find(model_version) != iter->broken_versions_.end())) {
      return;
    }
  }
  bool is_single_node = !model_info->GetNodeName().empty();
  std::vector<ExecutionMode> execution_modes = {ExecutionMode::ORT_SEQUENTIAL};
  if (provider_name == "cpu" && !is_single_node)
    execution_modes.push_back(ExecutionMode::ORT_PARALLEL);

  std::vector<bool> use_single_thread{false};
  // Test the model with intra op threadpool disabled
  if (provider_name == "cpu" && !is_single_node)
    use_single_thread.push_back(true);

  std::unique_ptr<ITestCase> l = CreateOnnxTestCase(ToMBString(test_case_name), std::move(model_info),
                                                    per_sample_tolerance, relative_per_sample_tolerance);
  for (bool is_single_thread : use_single_thread) {
    for (ExecutionMode execution_mode : execution_modes) {
      SessionOptions so;
      if (!is_single_thread)
        so.use_per_session_threads = false;
      else
        so.intra_op_param.thread_pool_size = 1;  // Disable intra op thread pool
      so.execution_mode = execution_mode;
      so.session_logid = ToMBString(test_case_name);
      so.session_log_severity_level = (int)logging::Severity::kERROR;
      InferenceSession session_object(so, (**ort_env).GetEnvironment());
      if (provider_name == "cuda") {
        ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(DefaultCudaExecutionProvider()));
      } else if (provider_name == "dnnl") {
        ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(DefaultDnnlExecutionProvider()));
      } else if (provider_name == "ngraph") {
        ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(DefaultNGraphExecutionProvider()));
      } else if (provider_name == "nuphar") {
        ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(DefaultNupharExecutionProvider()));
      } else if (provider_name == "tensorrt") {
        ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(DefaultTensorrtExecutionProvider()));
      } else if (provider_name == "migraphx") {
        ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(DefaultMIGraphXExecutionProvider()));
      } else if (provider_name == "openvino") {
        ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(DefaultOpenVINOExecutionProvider()));
      } else if (provider_name == "nnapi") {
        ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(DefaultNnapiExecutionProvider()));
      } else if (provider_name == "rknpu") {
        ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(DefaultRknpuExecutionProvider()));
      } else if (provider_name == "acl") {
        ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(DefaultAclExecutionProvider()));
      }
      if (provider_name == "armnn") {
        ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(DefaultArmNNExecutionProvider()));
      }

      ASSERT_STATUS_OK(session_object.Load(model_path));
      auto st = session_object.Initialize();
      if (st.Code() == NOT_IMPLEMENTED)
        return;
      ASSERT_TRUE(st.IsOK()) << st.ErrorMessage();
      const size_t data_count = l->GetDataCount();
      for (size_t task_id = 0; task_id != data_count; ++task_id) {
        onnxruntime::test::HeapBuffer holder;
        std::unordered_map<std::string, Ort::Value> feeds;
        l->LoadTestData(task_id, holder, feeds, true);

        std::pair<common::Status, const OutputDefList*> output_meta_data = session_object.GetModelOutputs();
        ASSERT_STATUS_OK(output_meta_data.first);
        // Create output feed
        size_t output_count = output_meta_data.second->size();
        std::vector<std::string> output_names(output_count);
        for (size_t i = 0; i != output_count; ++i) {
          output_names[i] = (*output_meta_data.second)[i]->Name();
        }

        std::vector<OrtValue> output_values(output_count);
        {
          std::unordered_map<std::string, OrtValue> input;
          for (auto& p : feeds) {
            const OrtValue* v = p.second;
            input.emplace(p.first, *v);
          }
          ASSERT_STATUS_OK(session_object.Run(input, output_names, &output_values));
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
          name_fetch_output_map[output_name] = &output_values[i];
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
          ret = VerifyValueInfo(*v, Ort::Unowned<Ort::Value>{actual_output_value});
          compare_result = ret.first;
          ASSERT_EQ(COMPARE_RESULT::SUCCESS, ret.first) << ret.second;

          if (compare_result != COMPARE_RESULT::SUCCESS) {
            break;
          }
        }
      }
    }
  }
}

// TODO: all providers
::std::vector<::std::basic_string<ORTCHAR_T>> GetParameterStrings() {
  std::vector<const ORTCHAR_T*> provider_names;
  provider_names.push_back(ORT_TSTR("cpu"));
#ifdef USE_TENSORRT
  provider_names.push_back(ORT_TSTR("tensorrt"));
#endif
#ifdef USE_MIGRAPHX
  provider_names.push_back(ORT_TSTR("migraphx"));
#endif
#ifdef USE_OPENVINO
  provider_names.push_back(ORT_TSTR("openvino"));
#endif
#ifdef USE_CUDA
  provider_names.push_back(ORT_TSTR("cuda"));
#endif
#ifdef USE_DNNL
  provider_names.push_back(ORT_TSTR("dnnl"));
#endif
#ifdef USE_NGRAPH
  provider_names.push_back(ORT_TSTR("ngraph"));
#endif
#ifdef USE_NUPHAR
  provider_names.push_back(ORT_TSTR("nuphar"));
#endif
#ifdef USE_NNAPI
  provider_names.push_back(ORT_TSTR("nnapi"));
#endif
#ifdef USE_RKNPU
  provider_names.push_back(ORT_TSTR("rknpu"));
#endif
#ifdef USE_ACL
  provider_names.push_back(ORT_TSTR("acl"));
#endif
#ifdef USE_ARMNN
  provider_names.push_back(ORT_TSTR("armnn"));
#endif
  std::vector<std::basic_string<ORTCHAR_T>> v;
  for (const ORTCHAR_T* provider_name : provider_names) {
    std::unordered_set<std::basic_string<ORTCHAR_T>> all_disabled_tests(std::begin(immutable_broken_tests),
                                                                        std::end(immutable_broken_tests));
    if (CompareCString(provider_name, ORT_TSTR("cuda")) == 0) {
      all_disabled_tests.insert(std::begin(cuda_flaky_tests), std::end(cuda_flaky_tests));
    } else if (CompareCString(provider_name, ORT_TSTR("dml")) == 0) {
      all_disabled_tests.insert(std::begin(dml_disabled_tests), std::end(dml_disabled_tests));
    } else if (CompareCString(provider_name, ORT_TSTR("dnnl")) == 0) {
      // these models run but disabled tests to keep memory utilization low
      // This will be removed after LRU implementation
      all_disabled_tests.insert(std::begin(dnnl_disabled_tests), std::end(dnnl_disabled_tests));
    } else if (CompareCString(provider_name, ORT_TSTR("tensorrt")) == 0) {
      // these models run but disabled tests to keep memory utilization low
      // This will be removed after LRU implementation
      all_disabled_tests.insert(std::begin(tensorrt_disabled_tests), std::end(tensorrt_disabled_tests));
    } else if (CompareCString(provider_name, ORT_TSTR("openvino")) == 0) {
      // these models run but disabled tests to keep memory utilization low
      // This will be removed after LRU implementation
      all_disabled_tests.insert(std::begin(openvino_disabled_tests), std::end(openvino_disabled_tests));
    }

#if !defined(__amd64__) && !defined(_M_AMD64)
    all_disabled_tests.insert(std::begin(x86_disabled_tests), std::end(x86_disabled_tests));
#endif

    std::vector<std::basic_string<ORTCHAR_T>> paths;
#if defined(NDEBUG) || defined(RUN_MODELTEST_IN_DEBUG_MODE)
#ifdef _WIN32
    paths.push_back(ORT_TSTR("..\\models"));
#else
    paths.push_back(ORT_TSTR("../models"));
#endif
#endif

// TENSORRT has too many test failures in the single node tests
#if !defined(_WIN32) && !defined(USE_TENSORRT)
    paths.push_back("/data/onnx");
#endif
    while (!paths.empty()) {
      std::basic_string<ORTCHAR_T> node_data_root_path = paths.back();
      paths.pop_back();
      std::basic_string<ORTCHAR_T> my_dir_name = GetLastComponent(node_data_root_path);
      ORT_TRY {
        LoopDir(node_data_root_path, [&](const ORTCHAR_T* filename, OrtFileType f_type) -> bool {
          if (filename[0] == ORT_TSTR('.'))
            return true;
          if (f_type == OrtFileType::TYPE_DIR) {
            std::basic_string<PATH_CHAR_TYPE> p = ConcatPathComponent<PATH_CHAR_TYPE>(node_data_root_path, filename);
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
          std::basic_string<PATH_CHAR_TYPE> p = ConcatPathComponent<PATH_CHAR_TYPE>(node_data_root_path, filename_str);
          std::basic_string<PATH_CHAR_TYPE> r = provider_name;
          r.append(ORT_TSTR("_")).append(p);
          v.emplace_back(r);
          return true;
        });
      }
      ORT_CATCH(const std::exception&) {
      }  // ignore non-exist dir
    }
  }
  return v;
}

INSTANTIATE_TEST_SUITE_P(ModelTests, ModelTest, testing::ValuesIn(GetParameterStrings()));

}  // namespace test
}  // namespace onnxruntime

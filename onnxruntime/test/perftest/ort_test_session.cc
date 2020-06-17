// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ort_test_session.h"
#include <core/session/onnxruntime_c_api.h>
#include <assert.h>
#include "providers.h"
#include "TestCase.h"
#include "utils.h"

#ifdef _WIN32
#define strdup _strdup
#endif

namespace onnxruntime {
namespace perftest {

const std::string& SampleLoader::Name() const {
  return test_case_->GetTestCaseName();
}
size_t SampleLoader::TotalSampleCount() {
  return test_case_->GetDataCount();
}
size_t SampleLoader::PerformanceSampleCount() {
  return test_case_->GetDataCount();
}
void SampleLoader::LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex>& samples) {
  // TODO:In the MultiStream scenarios, Samples will appear more than once.
  std::unordered_map<std::string, OrtValue*> feeds;
  for (const mlperf::QuerySampleIndex& test_data_id : samples) {
    OrtValue** input_list = inputs_.data() + test_data_id * input_length_;
    if (input_list[0] == nullptr) {
      test_case_->LoadTestData(test_data_id /* id */, b_, feeds, true);
      // Discard the names in feeds
      for (size_t i = 0; i != input_length_; ++i) {

        auto iter = feeds.find(input_names_[i]);
        if (iter == feeds.end()) {
          std::ostringstream oss;
          oss << "there is no test input data for input " << input_names_[i] << " and model "
              << test_case_->GetTestCaseName() << std::endl;
          throw std::runtime_error(oss.str());
        }
        input_list[i] = iter->second;
      }
      feeds.clear();
    }
  }
}

void SampleLoader::UnloadSamplesFromRam(const std::vector<mlperf::QuerySampleIndex>& samples) {
  for (const mlperf::QuerySampleIndex& test_data_id : samples) {
    OrtValue** input_list = inputs_.data() + test_data_id * input_length_;
    for (size_t i = 0; i != input_length_; ++i) {
      c_api->ReleaseValue(input_list[i]);
      input_list[i] = nullptr;
    }
  }
}

SampleLoader::SampleLoader(OrtSession* sess, ITestCase* test_case) : test_case_(test_case) {
  ThrowOnError(c_api->SessionGetInputCount(sess, &input_length_));
  OrtAllocator* alloc;
  ThrowOnError(c_api->GetAllocatorWithDefaultOptions(&alloc));
  input_names_.resize(input_length_);
  for (size_t i = 0; i != input_length_; ++i) {
    char* input_name;
    ThrowOnError(c_api->SessionGetInputName(sess, i, alloc, &input_name));
    assert(input_name != nullptr);
    input_names_[i] = input_name;
    alloc->Free(alloc, input_name);
  }

  inputs_.resize(test_case_->GetDataCount() * input_length_);
}

OnnxRuntimeTestSession::OnnxRuntimeTestSession(OrtSession* sess, SampleLoader* sample_loader, std::random_device& rd, size_t concurrent_session_runs)
    : sess_(sess), sample_loader_(sample_loader), rand_engine_(rd()) {
  ThrowOnError(c_api->SessionGetInputCount(sess, &input_length_));
  OrtAllocator* alloc;
  ThrowOnError(c_api->GetAllocatorWithDefaultOptions(&alloc));
  input_names_.resize(input_length_);
  for (size_t i = 0; i != input_length_; ++i) {
    char* input_name;
    ThrowOnError(c_api->SessionGetInputName(sess, i, alloc, &input_name));
    assert(input_name != nullptr);
    input_names_[i] = strdup(input_name);
    alloc->Free(alloc, input_name);
  }

  size_t output_count;
  ThrowOnError(c_api->SessionGetOutputCount(sess, &output_count));
  output_names_.resize(output_count);
  for (size_t i = 0; i != output_count; ++i) {
    char* output_name;
    ThrowOnError(c_api->SessionGetOutputName(sess, i, alloc, &output_name));
    assert(output_name != nullptr);
    output_names_[i] = output_name;
    alloc->Free(alloc, output_name);
  }
  output_names_raw_ptr.resize(output_count);
  for (size_t i = 0; i != output_count; ++i) {
    output_names_raw_ptr[i] = output_names_[i].c_str();
  }
  eigen_threadpool_ = std::make_unique<onnxruntime::ThreadPoolTempl<onnxruntime::Env>>(ORT_TSTR("perftest"),static_cast<int>(concurrent_session_runs),false,onnxruntime::Env::Default(),thread_options_);
}

void OnnxRuntimeTestSession::IssueQuery(const std::vector<mlperf::QuerySample>& samples) {
  if (samples.size() == 1) {
    mlperf::QuerySampleResponse res;
    size_t output_count = output_names_.size();    
    std::vector<OrtValue*> outputs(output_count);    
    const mlperf::QuerySample& s = samples[0];
    ThrowOnError(c_api->Run(sess_, nullptr, input_names_.data(), sample_loader_->GetInput(s.index),
                                 input_names_.size(), output_names_raw_ptr.data(), output_names_raw_ptr.size(),
                                 outputs.data()));
    for (size_t i = 0; i != output_names_.size(); ++i) {
      c_api->ReleaseValue(outputs[i]);
    }
    res.id = s.id;
    res.data = 0;
    res.size = 0;
    mlperf::QuerySamplesComplete(&res, 1);
  } else {
    size_t output_count = output_names_.size();
    //It is possible to group the samples in batches to get better performance, but it is highly model dependent.
    for(const mlperf::QuerySample& s: samples) {
      eigen_threadpool_->Schedule([output_count, this, &s]() {
        //OrtValue* outputs[output_count];
        //memset(outputs, 0, output_count * sizeof(OrtValue*));
        std::vector<OrtValue*> outputs(output_count);
        ThrowOnError(c_api->Run(sess_, nullptr, input_names_.data(), sample_loader_->GetInput(s.index),
                                input_names_.size(), output_names_raw_ptr.data(), output_names_raw_ptr.size(),
                                outputs.data()));
        for (size_t i = 0; i != output_names_.size(); ++i) {
          c_api->ReleaseValue(outputs[i]);
        }
        mlperf::QuerySampleResponse res;
        res.id = s.id;
        res.data = 0;
        res.size = 0;
        mlperf::QuerySamplesComplete(&res, 1);
      });
    }
  }
}

OrtSession* CreateOrtSession(OrtEnv* env, const PerformanceTestConfig& performance_test_config) {
  OrtSessionOptions* session_options;
  ThrowOnError(c_api->CreateSessionOptions(&session_options));
  const std::string& provider_name = performance_test_config.machine_config.provider_type_name;
  if (provider_name == onnxruntime::kDnnlExecutionProvider) {
#ifdef USE_DNNL
    ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Dnnl(
        session_options, performance_test_config.run_config.enable_cpu_mem_arena ? 1 : 0));
#else
    ORT_THROW("DNNL is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kNGraphExecutionProvider) {
#ifdef USE_NGRAPH
    ThrowOnError(OrtSessionOptionsAppendExecutionProvider_NGraph(session_options, "CPU"));
#else
    ORT_THROW("nGraph is not supported in this build");
#endif
  } else if (provider_name == onnxruntime::kCudaExecutionProvider) {
#ifdef USE_CUDA
    ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
#else
    ORT_THROW("CUDA is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kNupharExecutionProvider) {
#ifdef USE_NUPHAR
    ThrowOnError(
        OrtSessionOptionsAppendExecutionProvider_Nuphar(session_options, /*allow_unaligned_buffers*/ 1, ""));
#else
    ORT_THROW("Nuphar is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kTensorrtExecutionProvider) {
#ifdef USE_TENSORRT
    ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_options, 0));
    ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
#else
    ORT_THROW("TensorRT is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kOpenVINOExecutionProvider) {
#ifdef USE_OPENVINO
    ThrowOnError(OrtSessionOptionsAppendExecutionProvider_OpenVINO(session_options, ""));
#else
    ORT_THROW("OpenVINO is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kNnapiExecutionProvider) {
#ifdef USE_NNAPI
    ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(session_options));
#else
    ORT_THROW("NNAPI is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kDmlExecutionProvider) {
#ifdef USE_DML
    ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(session_options, 0));
#else
    ORT_THROW("DirectML is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kAclExecutionProvider) {
#ifdef USE_ACL
    ThrowOnError(OrtSessionOptionsAppendExecutionProvider_ACL(
        session_options, performance_test_config.run_config.enable_cpu_mem_arena ? 1 : 0));
#else
    ORT_THROW("Acl is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kMIGraphXExecutionProvider) {
#ifdef USE_MIGRAPHX
    ThrowOnError(OrtSessionOptionsAppendExecutionProvider_MIGraphX(session_options, 0));
#else
    ORT_THROW("MIGraphX is not supported in this build\n");
#endif
  } else if (!provider_name.empty() && provider_name != onnxruntime::kCpuExecutionProvider) {
    ORT_THROW("This backend is not included in perf test runner.\n");
  }

  if (performance_test_config.run_config.enable_cpu_mem_arena)
    ThrowOnError(c_api->EnableCpuMemArena(session_options));
  else
    ThrowOnError(c_api->DisableCpuMemArena(session_options));

  if (performance_test_config.run_config.enable_memory_pattern &&
      performance_test_config.run_config.execution_mode == ExecutionMode::ORT_SEQUENTIAL)
    ThrowOnError(c_api->EnableMemPattern(session_options));
  else
    ThrowOnError(c_api->DisableMemPattern(session_options));

  ThrowOnError(c_api->SetSessionExecutionMode(session_options, performance_test_config.run_config.execution_mode));

  if (performance_test_config.run_config.intra_op_num_threads > 0) {
    fprintf(stdout, "Setting intra_op_num_threads to %d\n", performance_test_config.run_config.intra_op_num_threads);
    // TODO: If ORT depends on openmp, we should call omp_set_num_threads instead
    ThrowOnError(
        c_api->SetIntraOpNumThreads(session_options, performance_test_config.run_config.intra_op_num_threads));
  }

  if (performance_test_config.run_config.execution_mode == ExecutionMode::ORT_PARALLEL &&
      performance_test_config.run_config.inter_op_num_threads > 0) {
    fprintf(stdout, "Setting inter_op_num_threads to %d\n", performance_test_config.run_config.inter_op_num_threads);
    ThrowOnError(
        c_api->SetInterOpNumThreads(session_options, performance_test_config.run_config.inter_op_num_threads));
  }

  // Set optimization level.
  ThrowOnError(
      c_api->SetSessionGraphOptimizationLevel(session_options, performance_test_config.run_config.optimization_level));
  if (!performance_test_config.run_config.profile_file.empty())
    ThrowOnError(c_api->EnableProfiling(session_options, performance_test_config.run_config.profile_file.c_str()));
  if (!performance_test_config.run_config.optimized_model_path.empty())
    ThrowOnError(c_api->SetOptimizedModelFilePath(
        session_options, performance_test_config.run_config.optimized_model_path.c_str()));
  OrtSession* ret;
  ThrowOnError(
      c_api->CreateSession(env, performance_test_config.model_info.model_file_path.c_str(), session_options, &ret));
  c_api->ReleaseSessionOptions(session_options);
  return ret;
}

}  // namespace perftest
}  // namespace onnxruntime

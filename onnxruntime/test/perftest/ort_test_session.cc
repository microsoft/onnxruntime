#include "ort_test_session.h"
#include <core/session/onnxruntime_cxx_api.h>
#include <assert.h>
#include "providers.h"
#include "TestCase.h"

#ifdef _WIN32
#define strdup _strdup
#endif

namespace onnxruntime {
namespace perftest {

std::chrono::duration<double> OnnxRuntimeTestSession::Run() {
  //Randomly pick one OrtValueArray from test_inputs_. (NOT ThreadSafe)
  const std::uniform_int_distribution<int>::param_type p(0, static_cast<int>(test_inputs_.size() - 1));
  const size_t id = static_cast<size_t>(dist_(rand_engine_, p));
  OrtValueArray* const input = test_inputs_.at(id);
  auto start = std::chrono::high_resolution_clock::now();
  ORT_THROW_ON_ERROR(OrtRun(session_object_, nullptr, input_names_.data(), input->Data(), input_names_.size(),
                            output_names_raw_ptr.data(), output_names_raw_ptr.size(), output_values_.data()));
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration_seconds = end - start;
  for (size_t i = 0; i != output_values_.size(); ++i) {
    OrtReleaseValue(output_values_[i]);
    output_values_[i] = nullptr;
  }
  return duration_seconds;
}

OnnxRuntimeTestSession::OnnxRuntimeTestSession(OrtEnv* env, std::random_device& rd,
                                               const PerformanceTestConfig& performance_test_config,
                                               const TestModelInfo* m)
    : rand_engine_(rd()), input_names_(m->GetInputCount()), input_length_(m->GetInputCount()) {
  SessionOptionsWrapper sf(env);
  const bool enable_cpu_mem_arena = true;
  const std::string& provider_name = performance_test_config.machine_config.provider_type_name;
  if (provider_name == onnxruntime::kMklDnnExecutionProvider) {
#ifdef USE_MKLDNN
    ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_Mkldnn(sf, enable_cpu_mem_arena ? 1 : 0));
#else
    ORT_THROW("MKL-DNN is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kNGraphExecutionProvider) {
#ifdef USE_NGRAPH
    ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_NGraph(sf, "CPU"));
#else
    ORT_THROW("nGraph is not supported in this build");
#endif
  } else if (provider_name == onnxruntime::kCudaExecutionProvider) {
#ifdef USE_CUDA
    ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_CUDA(sf, 0));
#else
    ORT_THROW("CUDA is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kNupharExecutionProvider) {
#ifdef USE_NUPHAR
    ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_Nuphar(sf, 0, ""));
#else
    ORT_THROW("Nuphar is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kTensorrtExecutionProvider) {
#ifdef USE_TENSORRT
    ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_Tensorrt(sf));
    ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_CUDA(sf, 0));
#else
    ORT_THROW("TensorRT is not supported in this build\n");
#endif
  } else if (!provider_name.empty() && provider_name != onnxruntime::kCpuExecutionProvider) {
    ORT_THROW("This backend is not included in perf test runner.\n");
  }

  if (enable_cpu_mem_arena)
    sf.EnableCpuMemArena();
  else
    sf.DisableCpuMemArena();
  if (performance_test_config.run_config.enable_sequential_execution)
    sf.EnableSequentialExecution();
  else
    sf.DisableSequentialExecution();
  fprintf(stdout, "Setting thread pool size to %d\n", performance_test_config.run_config.session_thread_pool_size);
  sf.SetSessionThreadPoolSize(performance_test_config.run_config.session_thread_pool_size);
  // Set optimization level.
  sf.SetSessionGraphOptimizationLevel(performance_test_config.run_config.optimization_level);
  if (!performance_test_config.run_config.profile_file.empty())
    sf.EnableProfiling(performance_test_config.run_config.profile_file.c_str());
  session_object_ = sf.OrtCreateSession(performance_test_config.model_info.model_file_path.c_str());

  size_t output_count;
  ORT_THROW_ON_ERROR(OrtSessionGetOutputCount(session_object_, &output_count));
  output_names_.resize(output_count);
  OrtAllocator* a;
  ORT_THROW_ON_ERROR(OrtCreateDefaultAllocator(&a));
  for (size_t i = 0; i != output_count; ++i) {
    char* output_name = nullptr;
    ORT_THROW_ON_ERROR(OrtSessionGetOutputName(session_object_, i, a, &output_name));
    assert(output_name != nullptr);
    output_names_[i] = output_name;
    a->Free(a, output_name);
  }
  output_names_raw_ptr.resize(output_count);
  for (size_t i = 0; i != output_count; ++i) {
    output_names_raw_ptr[i] = output_names_[i].c_str();
  }
  OrtReleaseAllocator(a);
  output_values_.resize(output_count);

  size_t input_count = static_cast<size_t>(m->GetInputCount());
  for (size_t i = 0; i != input_count; ++i) {
    input_names_[i] = strdup(m->GetInputName(i).c_str());
  }
}

}  // namespace perftest
}  // namespace onnxruntime
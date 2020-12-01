// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cxxopts.hpp"
#include "core/framework/bfc_arena.h"
#include "core/platform/path_lib.h"
#include "core/providers/providers.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/ort_env.h"
#ifdef USE_CUDA
#include "core/providers/cuda/cuda_allocator.h"
#endif
#include "orttraining/core/framework/distributed_run_context.h"
#include "orttraining/core/framework/mpi_context.h"
#include "orttraining/core/optimizer/megatron_transformer.h"

// test infrastructure
#include "test/onnx/TestCase.h"
#include "test/compare_ortvalue.h"
#include "test/onnx/heap_buffer.h"
#include "test/onnx/onnx_model_info.h"
#include "test/onnx/callback.h"

namespace onnxruntime {
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_CUDA(OrtDevice::DeviceId device_id,
                                                                               OrtCudnnConvAlgoSearch cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::EXHAUSTIVE,
                                                                               size_t cuda_mem_limit = std::numeric_limits<size_t>::max(),
                                                                               onnxruntime::ArenaExtendStrategy arena_extend_strategy = ArenaExtendStrategy::kNextPowerOfTwo,
                                                                               bool do_copy_in_default_stream = true);
}

using namespace onnxruntime;
using namespace onnxruntime::common;
using namespace onnxruntime::training;
using namespace onnxruntime::test;

struct InferenceParams{
  std::string model_path;
  std::unordered_map<std::string, std::shared_ptr<IExecutionProviderFactory>> providers;
  float cuda_mem_limit_in_gb = -1;
  // Allocator to use for allocating inputs from the dataset (optional).
  AllocatorPtr input_allocator;
  int horizontal_parallel_size = 1;
};

Status ParseArguments(int argc, char* argv[], InferenceParams* params){
  cxxopts::Options options("Distributed Inference", "Inference a model across multiple GPUs");
  options.add_options()
  ("f, model_path", "Path to the model and input/output data", cxxopts::value<std::string>())
  ("cuda_mem_limit_in_gb", "Max cuda memory ort can use, in GB", cxxopts::value<float>()->default_value("-1.0"))
  ("horizontal_parallel_size", "Horizontal model parallel group size.", cxxopts::value<int>()->default_value("1"));

  try {
    auto flags = options.parse(argc, argv);
    params->model_path = flags["model_path"].as<std::string>();
    params->cuda_mem_limit_in_gb = flags["cuda_mem_limit_in_gb"].as<float>();
    params->horizontal_parallel_size = flags["horizontal_parallel_size"].as<int>();
  } catch (const std::exception& e) {
    const std::string msg = "Failed to parse the command line arguments";
    std::cerr << msg << ": " << e.what() << "\n"
              << options.help() << "\n";
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, msg);
  }

  return Status::OK();
}

int main(int argc, char* argv[]) {
/*  bool loop = true;
  while(loop){
    loop=true;
  }*/
  InferenceParams params;
  ParseArguments(argc, argv, &params);  
  std::basic_string<ORTCHAR_T> model_path = ToPathString(params.model_path);
  double per_sample_tolerance = 1e-3;
  double relative_per_sample_tolerance = 1e-3;
  std::unique_ptr<OnnxModelInfo> model_info = onnxruntime::make_unique<OnnxModelInfo>(model_path.c_str());

  std::basic_string<ORTCHAR_T> model_dir;
  (void)GetDirNameFromFilePath(model_path, model_dir);
  std::basic_string<PATH_CHAR_TYPE> test_case_name = GetLastComponent(model_dir);
  if (test_case_name.compare(0, 5, ORT_TSTR("test_")) == 0){
    test_case_name = test_case_name.substr(5);
  }

  #ifdef USE_CUDA
  OrtDevice::DeviceId device_id = static_cast<OrtDevice::DeviceId>(MPIContext::GetInstance().GetLocalRank());
  size_t cuda_mem_limit = std::numeric_limits<size_t>::max();
  if (params.cuda_mem_limit_in_gb > 0)
    cuda_mem_limit = static_cast<size_t>(params.cuda_mem_limit_in_gb * 1024 * 1024 * 1024);
  params.providers.emplace(kCudaExecutionProvider, CreateExecutionProviderFactory_CUDA(device_id, OrtCudnnConvAlgoSearch::EXHAUSTIVE,
                                                                                       cuda_mem_limit));
  params.input_allocator = std::make_shared<CUDAPinnedAllocator>(device_id, CUDA_PINNED);
  #endif

  SessionOptions so;
  so.session_logid = ToMBString(test_case_name);
  so.session_log_severity_level = (int)logging::Severity::kERROR;
  //std::unique_ptr<Environment> env;
  //Environment::Create(nullptr, env);
  std::unique_ptr<Ort::Env> ort_env;
  OrtThreadingOptions tpo;
  ort_env.reset(new Ort::Env(&tpo, ORT_LOGGING_LEVEL_WARNING, "Default"));
  InferenceSession session_object(so, (**ort_env).GetEnvironment());
  std::shared_ptr<IExecutionProviderFactory> cuda_provider_factory = params.providers[kCudaExecutionProvider];
  session_object.RegisterExecutionProvider(std::move(cuda_provider_factory->CreateProvider()));
  session_object.Load(model_path);
  std::unordered_map<std::string, std::string> updated_weight_names;
  std::unordered_set<std::string> weights_to_train;
  if (params.horizontal_parallel_size > 1) {
    LOGS_DEFAULT(WARNING) << params.horizontal_parallel_size << "-way horizontal model parallel is enabled";
    session_object.RegisterGraphTransformer(onnxruntime::make_unique<MegatronTransformer>(
      training::DistributedRunContext::RankInGroup(training::WorkerGroupType::HorizontalParallel),
      params.horizontal_parallel_size, updated_weight_names, weights_to_train));
  }
  session_object.Initialize();
  std::unique_ptr<ITestCase> l = CreateOnnxTestCase(ToMBString(test_case_name), std::move(model_info),
                                                    per_sample_tolerance, relative_per_sample_tolerance);
  const size_t data_count = l->GetDataCount();
  for (size_t task_id = 0; task_id != data_count; ++task_id) {
    onnxruntime::test::HeapBuffer holder;
    std::unordered_map<std::string, Ort::Value> feeds;
    l->LoadTestData(task_id, holder, feeds, true);

    std::pair<common::Status, const OutputDefList*> output_meta_data = session_object.GetModelOutputs();
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
      session_object.Run(input, output_names, &output_values);
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
      if (infoProto != nullptr){
        name_output_value_info_proto.insert(std::make_pair(infoProto->name(), infoProto));
      }
      i++;
    }

    for (auto& output : expected_output_values) {
      const OrtValue* expected_output_value = output.second;
      const std::string& output_name = output.first;
      auto iter = name_fetch_output_map.find(output_name);

      OrtValue* actual_output_value = iter->second;
      std::pair<COMPARE_RESULT, std::string> ret =
          CompareOrtValue(*actual_output_value, *expected_output_value, per_sample_tolerance,
                            relative_per_sample_tolerance, post_procesing);
      COMPARE_RESULT compare_result = ret.first;

      const ONNX_NAMESPACE::ValueInfoProto* v = name_output_value_info_proto[output_name];
      if (v == nullptr)  continue;
      ret = VerifyValueInfo(*v, Ort::Unowned<Ort::Value>{actual_output_value});
      compare_result = ret.first;

      if (compare_result != COMPARE_RESULT::SUCCESS) {
        std::cout<<"results mismatch" << std::endl;
        return 1;
      }
    }
  }
  std::cout<<"model executed successfully!" << std::endl;
  
  #if defined(USE_MPI)
  #ifdef _WIN32
  // https://docs.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-best-practices
  // shutdown_mpi() is not called within MPIContext destructor because of DllMain's restriction
  // call shutdown_mpi() here instead.
  MPIContext::shutdown_mpi();
  #endif
  #endif

  return 0;
}

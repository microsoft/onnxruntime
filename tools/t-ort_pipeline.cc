// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.

// Compile
// g++ -std=c++11 -o t-ort_pipeline t-ort_pipeline.cc -I /bert_ort/pranav/onnxruntime/include/onnxruntime/core/session -I /bert_ort/pranav/onnxruntime/include/onnxruntime/core/providers/cuda/ -lonnxruntime -lpthread -ltbb -L build/Linux/Debug/ -Wl,-rpath,/bert_ort/pranav/onnxruntime/build/Linux/Debug/

#include "bits/stdc++.h"
#include <assert.h>
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#include "/bert_ort/pranav/onnxruntime/include/onnxruntime/core/providers/cuda/cuda_provider_factory.h"
#include "tbb/pipeline.h"
#include "json.hpp"

using json = nlohmann::json;

/*
* This is just a prototype to demonstrate the usage of Intel TBB's parallel_pipeline to implement
* pipeline parallelism. See main() for usage.
* It runs 2 models on 2 separate GPU devices in a pipeline.
* The output of first model is allocated on GPU-0 and fed to the next model running on GPU-1.
* The cross-device copy (from GPU-0 to GPU-1) is done by ORT as part of Run().
*/

// TODO error handling

const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

// helper function to check for status
void CheckStatus(OrtStatus* status) {
  if (status != NULL) {
    const char* msg = g_ort->GetErrorMessage(status);
    fprintf(stderr, "%s\n", msg);
    g_ort->ReleaseStatus(status);
    exit(1);
  }
}

struct Ensemble {
  struct ModelConfig {
    std::string model_name;
    std::string model_file_path;
    std::vector<std::string> input_names;                                          // TODO can be obtained from model
    std::vector<std::string> output_names;                                         // TODO can be obtained from model
    std::unordered_map<std::string, std::vector<int64_t>> output_names_shape_map;  // output name -> shape; TODO shape can be obtained from model
    std::unordered_map<std::string, std::string> output_input_map;                 // maps output of this step to input of the next step
    int device_id;
  };

  std::vector<ModelConfig> models;
  int degree_of_parallelism = 10;  // TODO
};

struct PipelineSession {
  // TODO add batch API
  std::shared_ptr<OrtStatus> Run(const std::vector<std::string>& input_names,
                                 std::vector<Ort::Value>& input_values,  // TODO should be const
                                 const std::vector<std::string>& output_names,
                                 std::vector<Ort::Value>& output_values) {
    using namespace tbb;
    int i = 0;
    int batch_size = 1;

    auto input_stage_fn = [&](flow_control& fc) {
      if (i++ < batch_size) {
        std::vector<Ort::Value> dummy;
        return std::shared_ptr<Token>(new Token{std::string(), input_names, std::move(input_values)});
      } else {
        fc.stop();
        return std::shared_ptr<Token>(nullptr);
      }
    };
    auto input_stage_filter = make_filter<void, std::shared_ptr<Token>>(filter::serial, input_stage_fn);

    auto output_stage_fn = [&](std::shared_ptr<Token> token) {
      for (auto& elem : token->input_values) {
        output_values.push_back(std::move(elem));
      }
    };
    auto output_stage_filter = make_filter<std::shared_ptr<Token>, void>(filter::serial, output_stage_fn);

    auto model_exec_fn = [](int stage_id, Token& token, const std::string& model_name, PipelineSession& psess) {
      std::cout << "Executing model: " << model_name << "\n";

      auto& model_config = psess.ens.models[psess.model_configs.at(model_name)];
      auto& run_config = psess.run_configs.at(model_name);

      auto& ort_sess = run_config.session;
      auto* device_allocator = run_config.device_allocator.get();

      std::vector<const char*> input_names;
      for (const auto& elem : token.input_names) {
        input_names.push_back(elem.c_str());
      }

      std::vector<const char*> output_names;
      for (const auto& elem : model_config.output_names) {
        output_names.push_back(elem.c_str());
      }

      auto token_ptr = std::make_shared<Token>();
      if (stage_id == psess.ens.models.size() - 1) {
        // TODO use the Ort::Value vector provided by the user when she called Run
        // this code will copy the output to the CPU
        std::vector<Ort::Value> output_values = ort_sess.Run(Ort::RunOptions{nullptr}, input_names.data(),
                                                             token.input_values.data(), token.input_values.size(),
                                                             output_names.data(), output_names.size());
        // now populate token
        for (int i = 0; i < output_names.size(); ++i) {
          // get input name from the map
          token_ptr->input_names.push_back(model_config.output_input_map[output_names[i]]);
          token_ptr->input_values.push_back(std::move(output_values[i]));
        }
      } else {
        // TODO what if you don't know the shape of the output? use iobinding here
        std::vector<Ort::Value> output_values;
        for (const auto& output_name : model_config.output_names) {
          const auto& output_shape = model_config.output_names_shape_map[output_name];
          auto output_tensor = Ort::Value::CreateTensor<float>(device_allocator, output_shape.data(),
                                                               output_shape.size());
          output_values.push_back(std::move(output_tensor));
        }

        ort_sess.Run(Ort::RunOptions{nullptr}, input_names.data(), token.input_values.data(), token.input_values.size(),
                     output_names.data(), output_values.data(), output_names.size());

        // now populate token
        for (int i = 0; i < output_names.size(); ++i) {
          // get input name from the map
          token_ptr->input_names.push_back(model_config.output_input_map[output_names[i]]);
          token_ptr->input_values.push_back(std::move(output_values[i]));
        }
      }

      return token_ptr;
    };

    // create filter based on first model
    auto model_exec_filter_chain =
        make_filter<std::shared_ptr<Token>,
                    std::shared_ptr<Token>>(filter::parallel,
                                            [this, &model_exec_fn, &model_name = ens.models[0].model_name](std::shared_ptr<Token> token_ptr) {
                                              return model_exec_fn(0, *token_ptr, model_name, *this);
                                            });

    // join filters from other models
    for (int i = 1; i < ens.models.size(); ++i) {
      model_exec_filter_chain =
          model_exec_filter_chain &
          make_filter<std::shared_ptr<Token>,
                      std::shared_ptr<Token>>(filter::parallel,
                                              [this, i, &model_exec_fn, &model_name = ens.models[i].model_name](std::shared_ptr<Token> token_ptr) {
                                                return model_exec_fn(i, *token_ptr, model_name, *this);
                                              });
    }

    // create and run the pipeline
    parallel_pipeline(ens.degree_of_parallelism,
                      input_stage_filter & model_exec_filter_chain & output_stage_filter);
  }

  PipelineSession(const std::string& ensemble_json_file, Ort::Env& env) {
    Ensemble ens;
    std::ifstream ifs(ensemble_json_file);
    if (!ifs.good()) {
      throw std::runtime_error("File error");
    }

    auto j = json::parse(ifs);
    ens.degree_of_parallelism = j["degree_of_parallelism"];
    for (const auto& m : j["ensemble"]) {
      Ensemble::ModelConfig cfg;
      cfg.model_name = m["model_name"];
      cfg.model_file_path = m["model_file_path"];
      cfg.device_id = m["device_id"];
      for (const auto& elem : m["input_names"]) {
        cfg.input_names.push_back(elem);
      }
      for (const auto& elem : m["output_names"]) {
        cfg.output_names.push_back(elem);
      }
      const auto& j_shape_map = m["output_shape_map"];
      for (int i = 0; i < cfg.output_names.size(); ++i) {
        for (auto elem : j_shape_map[i]) {
          cfg.output_names_shape_map[cfg.output_names[i]].push_back(elem);
        }
      }
      const auto& j_oi_map = m["output_input_map"];
      for (const auto& elem : j_oi_map) {
        cfg.output_input_map[elem[0]] = elem[1];
      }
      ens.models.push_back(std::move(cfg));
    }

    Init(ens, env);
  }

  PipelineSession(const Ensemble& ens, Ort::Env& env) {
    Init(ens, env);
  }

  void Init(const Ensemble& ens0, Ort::Env& env) {
    ens = ens0;  // TODO can avoid copy of ensemble config
    for (int i = 0; i < ens.models.size(); ++i) {
      const auto& name = ens.models[i].model_name;
      model_configs[name] = i;
    }

    for (const auto& mcfg : ens.models) {
      // create session
      Ort::SessionOptions session_options;  // TODO should we accept session options from the config?
      CheckStatus(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, mcfg.device_id));
      Ort::Session session(env, mcfg.model_file_path.c_str(), session_options);

      // create device allocator
      OrtAllocator* device_allocator = nullptr;
      Ort::MemoryInfo mem_info{
          "Cuda",
          OrtArenaAllocator,
          mcfg.device_id,
          OrtMemTypeDefault,
      };
      CheckStatus(g_ort->CreateAllocator(session, mem_info, &device_allocator));
      std::unique_ptr<OrtAllocator, decltype(g_ort->ReleaseAllocator)> u_device_allocator(device_allocator, g_ort->ReleaseAllocator);
      RunConfig rcfg{std::move(session), std::move(u_device_allocator)};
      run_configs.emplace(mcfg.model_name, std::move(rcfg));
    }
  }

  // data members
  struct Token {
    std::string error_msg;
    std::vector<std::string> input_names;
    std::vector<Ort::Value> input_values;
  };

  struct RunConfig {
    using OrtAllocatorUptr = std::unique_ptr<OrtAllocator, decltype(g_ort->ReleaseAllocator)>;
    Ort::Session session;
    OrtAllocatorUptr device_allocator;
  };

  std::unordered_map<std::string, RunConfig> run_configs;
  std::unordered_map<std::string, int> model_configs;
  Ensemble ens;
};

// void SetupEnsemble(Ensemble& ens) {
//   std::string model_name1 = "mul_1";
//   std::string model_name2 = "mul_2";
//   std::string input_name = "X";
//   std::string output_name = "Y";
//   std::string model_path = "/bert_ort/pranav/onnxruntime/onnxruntime/test/testdata/mul_1.onnx";
//   Ensemble::ModelConfig m1{model_name1,
//                            model_path,
//                            {input_name},
//                            {output_name},
//                            {{output_name, {3, 2}}},
//                            {{output_name, input_name}},
//                            2};

//   Ensemble::ModelConfig m2{model_name2,
//                            model_path,
//                            {input_name},
//                            {output_name},
//                            {{output_name, {3, 2}}},
//                            {{output_name, input_name}},
//                            3};

//   ens.models.push_back(std::move(m1));
//   ens.models.push_back(std::move(m2));
// }

int main(int argc, char* argv[]) {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

  // prepare inputs
  size_t input_tensor_size = 3 * 2;
  std::vector<int64_t> input_node_dims{3, 2};
  std::vector<std::string> input_node_names{"X"};
  std::vector<float> input_tensor_values(input_tensor_size);
  std::vector<std::string> output_node_names = {"Y"};

  int c = 1;
  for (unsigned int i = 0; i < input_tensor_size; i++) {
    input_tensor_values[i] = (float)c++;
  }
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size,
                                                      input_node_dims.data(), input_node_dims.size());
  assert(input_tensor.IsTensor());
  std::vector<Ort::Value> ort_inputs;
  ort_inputs.push_back(std::move(input_tensor));

  // read ensemble file name
  std::string ensemble_file_name = "model_ensemble.json";
  if (argc < 2) {
    std::cout << "Using ensemble file: model_ensemble.json\n";
  } else {
    ensemble_file_name = argv[1];
  }

  // setup the pipeline session
  PipelineSession pipeline_session(ensemble_file_name, env);

  // Run the pipeline
  std::vector<Ort::Value> output_values;
  pipeline_session.Run(input_node_names, ort_inputs, output_node_names, output_values);

  // print output
  auto* data_ptr = output_values[0].GetTensorData<float>();
  std::cout << "Printing output " << std::endl;
  for (int i = 0; i < 3 * 2; ++i) {
    std::cout << data_ptr[i] << " ";
  }
  std::cout << std::endl;

  printf("Done!\n");
  return 0;
}

// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.

// Compile
// g++ -g -std=c++14 -o ~/t-ort_pipeline ~/t-ort_pipeline.cc -I /bert_ort/pranav/onnxruntime/include/onnxruntime/core/session -I /bert_ort/pranav/onnxruntime/include/onnxruntime/core/providers/cuda/ -lonnxruntime -ltbb -L  /bert_ort/pranav/onnxruntime/build/Linux/Debug/ -Wl,-rpath,/bert_ort/pranav/onnxruntime/build/Linux/Debug/ -I/usr/local/cuda/include -L/usr/local/cuda/lib -lcuda -lcudart

#include "bits/stdc++.h"
#include <assert.h>
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#include "cuda_provider_factory.h"
#include "tbb/pipeline.h"
#include "json.hpp"
#include <cuda_runtime_api.h>
#include <cuda.h>

using json = nlohmann::json;

/*
* This is just a prototype to demonstrate the usage of Intel TBB's parallel_pipeline to implement
* pipeline parallelism. See main() for usage.
* It runs 2 models on 2 separate GPU devices in a pipeline.
* The output of first model is allocated on GPU-0 and fed to the next model running on GPU-1.
* The cross-device copy (from GPU-0 to GPU-1) is done by ORT as part of Run().
*/

// TODO error handling, replace asserts

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

std::pair<bool, int> Exists(const std::vector<std::string>& vec, const std::string& to_find) {
  auto it = std::find(std::begin(vec), std::end(vec), to_find);
  if (it != std::end(vec)) {
    return {true, it - std::begin(vec)};
  } else {
    return {false, -1};
  }
}

struct Ensemble {
  struct ModelConfig {
    std::string model_name;
    std::string model_file_path;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::unordered_map<std::string, std::string> output_input_map;  // maps output of this step to input of the next step
    // state_input_names and state_output_names should have 1-1 correspondence
    // TODO validate their sizes are same
    std::vector<std::string> state_input_names;   // names of inputs whose values come from the previous output
    std::vector<std::string> state_output_names;  // names of outputs that feed the next inputs
    int device_id;
  };

  std::vector<ModelConfig> models;
  std::unordered_map<std::string, int> model_idx_map;  // maps model name to index in models vector
  int degree_of_parallelism = 10;
};

void CopyOrtValues(std::vector<Ort::Value>& input_values, std::vector<Ort::Value>& input_values_copy) {
  for (auto& val : input_values) {
    void* data_ptr = nullptr;
    CheckStatus(g_ort->GetTensorMutableData(val, &data_ptr));
    auto type_shape_info = val.GetTensorTypeAndShapeInfo();
    int byte_count = sizeof(int64_t) * type_shape_info.GetElementCount();  // TODO
    auto shape = type_shape_info.GetShape();
    auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    input_values_copy.push_back(Ort::Value::CreateTensor(mem_info, data_ptr, byte_count,
                                                         shape.data(), shape.size(), type_shape_info.GetElementType()));
  }
}

struct PipelineSession {
  // TODO do we need to add a looping batch API?
  // TODO return error status code
  void Run(const std::vector<std::string>& input_names,
           std::vector<Ort::Value>& input_values,  // TODO should be const
           const std::vector<std::string>& output_names,
           std::vector<Ort::Value>& output_values,  // TODO does the user want to pre-allocate the outputs? how does she know the shape?
           int max_steps) {
    using namespace tbb;

    int step_id = 0;

    // input stage
    auto input_stage_fn = [&](flow_control& fc) {
      std::shared_ptr<Token> token_ptr;
      if (step_id < max_steps) {
        // HACK HACK TODO
        // you shouldn't need to copy the input values
        // need to figure out how to run a user-defined function on some of the inputs for step 2 onwards
        std::vector<Ort::Value> input_values_copy;
        CopyOrtValues(input_values, input_values_copy);
        // for the first step, the input should come from the user
        if (step_id == 0) {  // first step
          token_ptr = std::shared_ptr<Token>(new Token{step_id, std::string(), input_names, std::move(input_values_copy)});
        } else {
          // for the second step input_ids and position_ids are derived from the output of the first step
          // and the rest of the inputs come from state_output_names
          std::vector<Ort::Value> ort_inputs;
          ort_inputs.push_back(std::move(input_values_copy[0]));  // input_ids
          ort_inputs.push_back(std::move(input_values_copy[1]));  // position_ids
          std::vector<std::string> input_names{"input_ids", "position_ids"};
          token_ptr = std::shared_ptr<Token>(new Token{step_id, std::string(), input_names, std::move(ort_inputs)});
        }
      } else {
        fc.stop();
        token_ptr = std::shared_ptr<Token>(nullptr);
      }
      ++step_id;
      return token_ptr;
    };
    auto input_stage_filter = make_filter<void, std::shared_ptr<Token>>(filter::serial, input_stage_fn);

    // output stage
    auto output_stage_fn = [&](std::shared_ptr<Token> token) {
      output_values.clear();
      output_values.reserve(output_names.size());
      // only give those outputs to the user that she has requested
      for (const auto& oname : output_names) {
        // first check if this is a valid output name
        auto num_models = ens.models.size();
        auto rc = Exists(ens.models[num_models - 1].output_names, oname);
        if (!rc.first) {
          output_values.push_back(Ort::Value(nullptr));
          continue;  // TODO invalid output name requested; throw error here?
        }
        output_values.push_back(std::move(token->input_values[rc.second]));
      }
    };
    auto output_stage_filter = make_filter<std::shared_ptr<Token>, void>(filter::serial, output_stage_fn);

    // model execution
    auto model_exec_fn = [](int stage_id, Token& token, const std::string& model_name, PipelineSession& psess) {
      std::cout << "Executing step " << token.step_id << " stage " << stage_id << " model: " << model_name << std::endl;

      auto model_idx = psess.ens.model_idx_map.at(model_name);
      auto& model_config = psess.ens.models.at(model_idx);
      auto& session_state = psess.model_session_state_vec.at(model_idx);
      auto& run_state = psess.model_run_state_vec.at(model_idx);

      auto& ort_sess = session_state.session;

      CheckStatus(g_ort->SetCurrentGpuDeviceId(model_config.device_id));

      std::vector<const char*> model_output_names;
      for (const auto& elem : model_config.output_names) {
        model_output_names.push_back(elem.c_str());
      }

      auto token_ptr = std::make_shared<Token>();
      token_ptr->step_id = token.step_id;

      session_state.io_binding.ClearBoundInputs();
      session_state.io_binding.ClearBoundOutputs();

      // inputs
      // go through all the inputs from the config and for each one if you find it in token.input_names
      // use the value from there.
      // else search this input name inside state_input_names. If found, get the corresponding output name from
      // state_output_names and the OrtValue associated with it.
      // for the first stage and the first step, all inputs should come from the user
      if (stage_id == 0 && token.step_id == 0) {
        assert(token.input_names.size() == model_config.input_names.size());  // TODO throw an error
      }
      for (const auto& iname : model_config.input_names) {
        auto rc = Exists(token.input_names, iname);
        if (rc.first) {
          session_state.io_binding.BindInput(token.input_names[rc.second].c_str(), token.input_values[rc.second]);
          continue;
        }

        // for step 0/stage 1 how to get the initial past inputs?
        rc = Exists(model_config.state_input_names, iname);
        if (rc.first) {
          const auto& mapped_oname = model_config.state_output_names[rc.second];
          session_state.io_binding.BindInput(iname.c_str(),
                                             run_state.output_val_map.at(mapped_oname));
        }
      }

      // allocate outputs
      // TODO optimize - no need to allocate every single time
      for (const auto& output_name : model_config.output_names) {
        session_state.io_binding.BindOutput(output_name.c_str(), session_state.cuda_ort_mem_info);
      }

      // run
      ort_sess.Run(session_state.run_options, session_state.io_binding);

      // now populate token and save state from this run
      auto vec_out_vals = session_state.io_binding.GetOutputValues();
      for (int i = 0; i < model_output_names.size(); ++i) {
        const auto& oname = model_output_names[i];

        // Assume that the same output name is not present in both the state that needs to be kept
        // and that needs to be passed on to the next layer.
        auto rc = Exists(model_config.state_output_names, oname);
        assert(rc.first && model_config.output_input_map.count(oname));

        // if this output is present in state_output_names, store it in model_run_state_vec
        // because we don't want to store all outputs
        if (rc.first) {
          run_state.output_val_map.emplace(oname, std::move(vec_out_vals[i]));
          continue;
        }

        // only pass those outputs to the next layer for which there is a config in the ensemble
        // other outputs are states to be used in the next run
        if (model_config.output_input_map.count(oname)) {
          token_ptr->input_names.push_back(model_config.output_input_map[oname]);
          token_ptr->input_values.push_back(std::move(vec_out_vals[i]));
        }
      }
      std::cout << "Done Executing step " << token.step_id << " stage " << stage_id << " model: " << model_name << std::endl;
      return token_ptr;
    };

    // create filter based on first model
    auto model_exec_filter_chain =
        make_filter<std::shared_ptr<Token>,
                    std::shared_ptr<Token>>(filter::serial,
                                            [this, &model_exec_fn, &model_name = ens.models[0].model_name](std::shared_ptr<Token> token_ptr) {
                                              return model_exec_fn(0, *token_ptr, model_name, *this);
                                            });

    // join filters from other models
    for (int i = 1; i < ens.models.size(); ++i) {
      model_exec_filter_chain =
          model_exec_filter_chain &
          make_filter<std::shared_ptr<Token>,
                      std::shared_ptr<Token>>(filter::serial,
                                              [this, i, &model_exec_fn, &model_name = ens.models[i].model_name](std::shared_ptr<Token> token_ptr) {
                                                return model_exec_fn(i, *token_ptr, model_name, *this);
                                              });
    }

    // create and run the pipeline
    parallel_pipeline(ens.degree_of_parallelism,
                      input_stage_filter & model_exec_filter_chain & output_stage_filter);
  }

  void ParseEnsembleJsonFile(const std::string& ensemble_json_file, Ensemble& ens) {
    std::ifstream ifs(ensemble_json_file);
    if (!ifs.good()) {
      throw std::runtime_error("File error");
    }

    auto j = json::parse(ifs);
    ens.degree_of_parallelism = j["degree_of_parallelism"];
    int idx = 0;
    for (const auto& m : j["ensemble"]) {
      Ensemble::ModelConfig cfg;
      std::string model_name = m["model_name"];
      cfg.model_name = model_name;
      cfg.model_file_path = m["model_file_path"];
      cfg.device_id = m["device_id"];

      const char* key = "output_input_map";
      if (m.find(key) != m.end()) {
        const auto& j_oi_map = m[key];
        for (const auto& elem : j_oi_map) {
          cfg.output_input_map[elem[0]] = elem[1];
        }
      }

      key = "state_input_names";
      if (m.find(key) != m.end()) {
        const auto& si_names = m[key];
        for (const auto& elem : si_names) {
          cfg.state_input_names.push_back(elem);
        }
      }

      key = "state_output_names";
      if (m.find(key) != m.end()) {
        const auto& so_names = m[key];
        for (const auto& elem : so_names) {
          cfg.state_output_names.push_back(elem);
        }
      }

      ens.models.push_back(std::move(cfg));
      ens.model_idx_map[model_name] = idx;
      ++idx;
    }
  }

  PipelineSession(const std::string& ensemble_json_file, Ort::Env& env) {
    Ensemble ens;
    ParseEnsembleJsonFile(ensemble_json_file, ens);
    Init(ens, env);
  }

  PipelineSession(const Ensemble& ens, Ort::Env& env) {
    Init(ens, env);
  }

  void Init(const Ensemble& ens0, Ort::Env& env) {
    ens = ens0;  // TODO can avoid copy of ensemble config
    model_session_state_vec.reserve(ens.models.size());
    model_run_state_vec.resize(ens.models.size());

    Ort::AllocatorWithDefaultOptions ort_allocator;
    for (auto& mcfg : ens.models) {
      Ort::SessionOptions session_options;  // TODO should we accept session options from the config?
      CheckStatus(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, mcfg.device_id));
      Ort::Session session(env, mcfg.model_file_path.c_str(), session_options);

      // fill output names
      int output_count = session.GetOutputCount();
      mcfg.output_names.reserve(output_count);
      for (int i = 0; i < output_count; ++i) {
        auto name_ptr = std::unique_ptr<char>(session.GetOutputName(i, ort_allocator));
        mcfg.output_names.push_back(std::string(name_ptr.get()));
      }

      // fill input names
      int input_count = session.GetInputCount();
      mcfg.input_names.reserve(input_count);
      for (int i = 0; i < input_count; ++i) {
        auto name_ptr = std::unique_ptr<char>(session.GetInputName(i, ort_allocator));
        mcfg.input_names.push_back(std::string(name_ptr.get()));
      }

      Ort::IoBinding io_binding(session);
      Ort::MemoryInfo mem_info("Cuda", OrtDeviceAllocator, mcfg.device_id, OrtMemTypeDefault);

      SessionState ses_state{std::move(session), std::move(io_binding), std::move(mem_info)};
      model_session_state_vec.push_back(std::move(ses_state));
    }
  }

  struct Token {
    int step_id;
    std::string error_msg;
    std::vector<std::string> input_names;
    std::vector<Ort::Value> input_values;
  };

  struct SessionState {
    Ort::Session session;
    Ort::IoBinding io_binding;
    Ort::MemoryInfo cuda_ort_mem_info;
    Ort::RunOptions run_options;
  };

  struct RunState {
    std::unordered_map<std::string, Ort::Value> output_val_map;
  };

  std::vector<SessionState> model_session_state_vec;
  std::vector<RunState> model_run_state_vec;
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
//                            {},
//                            {},
//                            {{output_name, input_name}},
//                            {},
//                            {},
//                            2};

//   Ensemble::ModelConfig m2{model_name2,
//                            model_path,
//                            {},
//                            {},
//                            {{output_name, input_name}},
//                            {},
//                            {},
//                            3};

//   ens.models.push_back(std::move(m1));
//   ens.models.push_back(std::move(m2));

//   for (int i = 0; i < ens.models.size(); ++i) {
//     const auto& name = ens.models[i].model_name;
//     ens.model_idx_map[name] = i;
//   }
// }

int main(int argc, char* argv[]) {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
  // CheckStatus(g_ort->SetCurrentGpuDeviceId(3));
  // int current_device_id;
  // CheckStatus(g_ort->GetCurrentGpuDeviceId(&current_device_id));
  // std::cout << "current device id: " << current_device_id << std::endl;
  // exit(0);
  // prepare inputs
  int batch_size = 1;
  int seq_len = 5;
  size_t input_tensor_size = batch_size * seq_len;
  std::vector<int64_t> input_node_dims{batch_size, seq_len};
  std::vector<std::string> input_node_names{"input_ids", "position_ids"};
  for (int i = 0; i < 24; ++i) {
    std::ostringstream ostr;
    ostr << "past_" << i;
    input_node_names.push_back(ostr.str());
  }
  std::vector<int64_t> input_ids(input_tensor_size);
  std::vector<int64_t> posn_ids(input_tensor_size);
  std::vector<std::string> output_node_names = {"logits"};

  int c = 1;
  for (unsigned int i = 0; i < input_tensor_size; ++i, ++c) {
    input_ids[i] = static_cast<int64_t>(c);
    posn_ids[i] = static_cast<int64_t>(c);
  }

  std::vector<Ort::Value> ort_inputs;
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto input_ids_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, input_ids.data(), input_tensor_size,
                                                            input_node_dims.data(), input_node_dims.size());
  auto posn_ids_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, posn_ids.data(), input_tensor_size,
                                                           input_node_dims.data(), input_node_dims.size());
  ort_inputs.push_back(std::move(input_ids_tensor));
  ort_inputs.push_back(std::move(posn_ids_tensor));

  // all past inputs
  std::vector<int64_t> past_dims{2, batch_size, 32, 0, 128};
  for (int i = 0; i < 24; ++i) {
    Ort::Float16_t values[] = {15360};
    auto past_tensor = Ort::Value::CreateTensor<Ort::Float16_t>(memory_info, values, sizeof(values) / sizeof(values[0]),
                                                                past_dims.data(), past_dims.size());
    ort_inputs.push_back(std::move(past_tensor));
  }

  // read ensemble file name
  std::string ensemble_file_name = "turing_model_ensemble.json";
  if (argc > 1) {
    ensemble_file_name = argv[1];
  } else {
    std::cout << "Using ensemble file: model_ensemble.json\n";
  }

  int max_steps = 1;
  if (argc > 2) {
    max_steps = atoi(argv[2]);
  } else {
    std::cout << "Using max_steps = 1\n";
  }

  // setup the pipeline session
  PipelineSession pipeline_session(ensemble_file_name, env);
  // Ensemble ens;
  // SetupEnsemble(ens);
  // PipelineSession pipeline_session(ens, env);

  // Run the pipeline
  std::vector<Ort::Value> output_values;
  std::vector<std::string> output_names{"logits"};
  pipeline_session.Run(input_node_names, ort_inputs, output_names, output_values, max_steps);
  assert(output_values.size() == output_names.size());

  // print output
  // auto* data_ptr = output_values[0].GetTensorData<float>();
  // std::cout << "Printing output " << std::endl;
  // std::array<float, 6> dst;
  // auto err = cudaMemcpy(dst.data(), data_ptr, 6 * sizeof(float), cudaMemcpyDefault);
  // if (err == cudaSuccess) {
  //   for (int i = 0; i < 3 * 2; ++i) {
  //     std::cout << dst[i] << " ";
  //   }
  // } else {
  //   std::cout << "output copy failed from cuda\n";
  // }
  // std::cout << std::endl;

  printf("Done!\n");
  return 0;
}

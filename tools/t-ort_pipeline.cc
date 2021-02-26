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
*/

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

static std::pair<bool, int> Contains(const std::vector<std::string>& vec, const std::string& to_find) {
  auto it = std::find(std::begin(vec), std::end(vec), to_find);
  if (it != std::end(vec)) {
    return {true, it - std::begin(vec)};
  } else {
    return {false, -1};
  }
}

static std::vector<int64_t> GetShape(const Ort::Session& sess,
                                     const std::vector<std::string>& io_names,
                                     const std::string& io_name,
                                     bool is_input) {
  auto rc = Contains(io_names, io_name);
  if (!rc.first) {
    return {};
  }
  std::vector<int64_t> retval;
  if (is_input) {
    retval = sess.GetInputTypeInfo(rc.second).GetTensorTypeAndShapeInfo().GetShape();
  } else {
    retval = sess.GetOutputTypeInfo(rc.second).GetTensorTypeAndShapeInfo().GetShape();
  }
  return retval;
}

struct PipelineConfig {
  struct ModelConfig {
    std::string model_name;
    std::string model_file_path;
    std::vector<std::string> input_names;                           // same order as model
    std::vector<std::string> output_names;                          // same order as model
    std::unordered_map<std::string, std::string> inter_stage_output_input_map;  // maps output of this step to input of the next step
    // past_input_names and present_output_names should have 1-1 correspondence
    std::vector<std::string> past_input_names;   // names of inputs whose values come from the previous output
    std::vector<std::string> present_output_names;  // names of outputs that feed the next inputs
    int device_id;
    int batch_dim_index_in_state;
    int seq_len_dim_index_in_state;
  };

  int max_seq_len;
  std::vector<ModelConfig> model_config_vec;
  std::unordered_map<std::string, int> model_idx_map;  // maps model name to index in models vector
};

void CopyOrtValues(std::vector<Ort::Value>& input_values, std::vector<Ort::Value>& input_values_copy) {
  for (auto& val : input_values) {
    void* data_ptr = nullptr;
    CheckStatus(g_ort->GetTensorMutableData(val, &data_ptr));
    auto type_shape_info = val.GetTensorTypeAndShapeInfo();
    int byte_count = sizeof(int64_t) * type_shape_info.GetElementCount();
    auto shape = type_shape_info.GetShape();
    auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    input_values_copy.push_back(Ort::Value::CreateTensor(mem_info, data_ptr, byte_count,
                                                         shape.data(), shape.size(), type_shape_info.GetElementType()));
  }
}

struct PipelineSession {
  // TODO do we need to add a looping batch API?
  // TODO return error status code, decide how to do error handling in stages
  // TODO stop execution when an error is detected
  // TODO - adjust output shape for states
  // TODO - change position_ids for step > 0
  OrtStatus* Run(const std::vector<std::string>& input_names,
                 std::vector<Ort::Value>& input_values,  // TODO should be const
                 const std::vector<std::string>& output_names,
                 // TODO honor the user's choice of where she wants the output to be; basically use this OrtValue for BindOutput in the final stage
                 std::vector<Ort::Value>& output_values,
                 int max_steps,
                 int batch_size,
                 int degree_of_parallelism) {
    ExecutionFrame exec_frame(*this, batch_size);
    int step_id = 0;

    // input stage
    auto input_stage_fn = [&](tbb::flow_control& fc) {
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
          // and the rest of the inputs come from present_output_names
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
    auto input_stage_filter = tbb::make_filter<void, std::shared_ptr<Token>>(tbb::filter::serial, input_stage_fn);

    // output stage
    auto output_stage_fn = [&](const std::shared_ptr<Token> token) {
      output_values.clear();
      output_values.reserve(output_names.size());
      // only give those outputs to the user that she has requested
      for (const auto& oname : output_names) {
        auto rc = Contains(token->ort_value_names, oname);
        if (!rc.first) {
          output_values.push_back(Ort::Value(nullptr));
          continue;  // TODO invalid output name requested; throw error here?
        }
        output_values.push_back(std::move(token->ort_values[rc.second]));
      }
    };
    auto output_stage_filter = tbb::make_filter<std::shared_ptr<Token>, void>(tbb::filter::serial, output_stage_fn);

    // model execution
    auto model_exec_fn = [](int stage_id, const Token& token, const std::string& model_name,
                            PipelineSession& psess /* passed by ref or else sess.Run won't work as Run is non-const */,
                            ExecutionFrame& exec_frame /* intentional as we'll update the state */) {
      std::cout << "Executing " << stage_id << "/" << token.step_id << " model: " << model_name << std::endl;

      // TODO psess will be shared across stages, do we need a mutex?
      const auto model_idx = psess.ens.model_idx_map.at(model_name);
      const auto& model_config = psess.ens.model_config_vec[model_idx];
      auto& session_state = psess.model_session_state_vec[model_idx];
      auto& ort_sess = session_state.session;

      // TODO exec_frame is shared across stages, do we need a mutex?
      ExecutionFrame::RunState& run_state = exec_frame.model_run_state_vec[model_idx];

      CheckStatus(g_ort->SetCurrentGpuDeviceId(model_config.device_id));

      std::vector<const char*> model_output_names;
      for (const auto& elem : model_config.output_names) {
        model_output_names.push_back(elem.c_str());
      }

      auto token_ptr = std::make_shared<Token>();
      token_ptr->step_id = token.step_id;

      auto& io_binding = *run_state.io_binding;
      io_binding.ClearBoundInputs();
      io_binding.ClearBoundOutputs();

      // inputs
      // go through all the inputs from the config and for each one if you find it in token.input_names
      // use the value from there.
      // else search this input name inside past_input_names. If found, get the corresponding output name from
      // present_output_names and the OrtValue associated with it.
      for (const auto& iname : model_config.input_names) {
        auto rc = Contains(token.ort_value_names, iname);
        if (rc.first) {
          // std::cout << stage_id << "/" << token.step_id << " binding input " << token.ort_value_names[rc.second] << std::endl;
          io_binding.BindInput(token.ort_value_names[rc.second].c_str(), token.ort_values[rc.second]);
          continue;
        }

        rc = Contains(model_config.past_input_names, iname);
        if (rc.first) {
          const auto& mapped_oname = model_config.present_output_names[rc.second];
          // std::cout << stage_id << "/" << token.step_id << " state_binding " << iname << " with value of " << mapped_oname << std::endl;
          io_binding.BindInput(iname.c_str(),
                               run_state.output_val_map.at(mapped_oname));
        }
      }

      // allocate outputs
      // output seq len = current input seq len + past seq len (which is 0 the first time)
      // if output is a state, use the pre-allocated buffer to create an OrtValue and bind it.
      // if output is not a state, bind using just cuda_mem_info.
      // TODO optimize - pre-allocate buffers for states
      // TODO output length of states needs to be corrected; it should be input_len + past_seq_len
      for (const auto& oname : model_config.output_names) {
        if (Contains(model_config.present_output_names, oname).first) {
          auto& mem_allocation = token.step_id % 2 == 0
                                     ? run_state.state_buffer_2_map[oname]
                                     : run_state.state_buffer_1_map[oname];
          auto output_ort_val = Ort::Value::CreateTensor(
              session_state.cuda_mem_info, mem_allocation.get(), mem_allocation.size(),
              /* shape, shape_len, element_type */);
          io_binding.BindOutput(oname.c_str(), output_ort_val);
          )
        } else {
          io_binding.BindOutput(oname.c_str(), session_state.cuda_mem_info);
        }
      }

      // run
      ort_sess.Run({}, io_binding);

      // now populate token and save state from this run
      auto vec_out_vals = io_binding.GetOutputValues();
      for (int i = 0; i < model_output_names.size(); ++i) {
        const auto& oname = model_output_names[i];

        // Assume that the same output name is not present in both the state that needs to be kept
        // and that needs to be passed on to the next layer.
        auto rc = Contains(model_config.present_output_names, oname);
        assert(!(rc.first && model_config.inter_stage_output_input_map.count(oname)));

        // if this output is present in present_output_names, store it in model_run_state_vec
        // because we don't want to store all outputs
        if (rc.first) {
          // std::cout << stage_id << "/" << token.step_id << " saving state " << oname << std::endl;
          if (stage_id == 0 && i == 1)
            // std::cout << "output seq len: " << vec_out_vals[i].GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape()[3] << std::endl;
            run_state.output_val_map.emplace(oname, std::move(vec_out_vals[i]));
          continue;
        }

        // only pass those outputs to the next layer for which there is a config in the ensemble
        // other outputs are states to be used in the next run
        if (model_config.inter_stage_output_input_map.count(oname)) {
          // std::cout << stage_id << "/" << token.step_id << " mapping output " << oname << " to input of next stage "
          //           << model_config.inter_stage_output_input_map[oname] << std::endl;
          token_ptr->ort_value_names.push_back(model_config.inter_stage_output_input_map.at(oname));
          token_ptr->ort_values.push_back(std::move(vec_out_vals[i]));
        }
      }
      std::cout << "Done Executing step " << token.step_id << " stage " << stage_id << " model: " << model_name << std::endl;
      return token_ptr;
    };

    // create filter based on first model
    tbb::filter_t<std::shared_ptr<Token>, std::shared_ptr<Token>> model_exec_filter_chain;
    // auto model_exec_filter_chain =
    //     tbb::make_filter<std::shared_ptr<Token>,
    //                      std::shared_ptr<Token>>(tbb::filter::serial,
    //                                              [this, &exec_frame, &model_exec_fn, &model_name = ens.model_config_vec[0].model_name](
    //                                                  const std::shared_ptr<Token> token_ptr) {
    //                                                return model_exec_fn(0, *token_ptr, model_name, *this, exec_frame);
    //                                              });

    // join filters from other models
    for (int i = 1; i < ens.model_config_vec.size(); ++i) {
      model_exec_filter_chain =
          model_exec_filter_chain &
          tbb::make_filter<std::shared_ptr<Token>,
                           std::shared_ptr<Token>>(tbb::filter::serial,
                                                   [this, i, &exec_frame, &model_exec_fn, &model_name = ens.model_config_vec[i].model_name](
                                                       const std::shared_ptr<Token> token_ptr) {
                                                     return model_exec_fn(i, *token_ptr, model_name, *this, exec_frame);
                                                   });
    }

    // create and run the pipeline
    tbb::parallel_pipeline(degree_of_parallelism,
                           input_stage_filter & model_exec_filter_chain & output_stage_filter);
    return nullptr;
  }

  void ParseEnsembleJsonFile(const std::string& ensemble_json_file, PipelineConfig& ens) {
    std::ifstream ifs(ensemble_json_file);
    if (!ifs.good()) {
      throw std::runtime_error("File error");
    }

    auto j = json::parse(ifs, nullptr, true, true);
    ens.max_seq_len = j["max_seq_len"];
    int idx = 0;
    for (const auto& m : j["ensemble"]) {
      PipelineConfig::ModelConfig cfg;
      std::string model_name = m["model_name"];
      cfg.model_name = model_name;
      cfg.model_file_path = m["model_file_path"];
      cfg.device_id = m["device_id"];  // TODO validate device id

      const char* key = "inter_stage_output_input_map";  // TODO validate entries of this map
      if (m.find(key) != m.end()) {
        const auto& j_oi_map = m[key];
        for (const auto& elem : j_oi_map) {
          cfg.inter_stage_output_input_map[elem[0]] = elem[1];
        }
      }

      key = "past_input_names";
      if (m.find(key) != m.end()) {
        const auto& si_names = m[key];
        for (const auto& elem : si_names) {
          cfg.past_input_names.push_back(elem);
        }
      }

      key = "present_output_names";
      if (m.find(key) != m.end()) {
        const auto& so_names = m[key];
        for (const auto& elem : so_names) {
          cfg.present_output_names.push_back(elem);
        }
      }

      // TODO validate sizes of past_input_names and present_output_names is same

      ens.model_config_vec.push_back(std::move(cfg));
      ens.model_idx_map[model_name] = idx;
      ++idx;
    }
  }

  PipelineSession(const std::string& ensemble_json_file, Ort::Env& env) {
    PipelineConfig ens;
    ParseEnsembleJsonFile(ensemble_json_file, ens);
    Init(ens, env);
  }

  PipelineSession(const PipelineConfig& ens, Ort::Env& env) {
    Init(ens, env);
  }

  void Init(const PipelineConfig& ens0, Ort::Env& env) {
    ens = ens0;  // TODO can avoid copy of ensemble config

    Ort::AllocatorWithDefaultOptions ort_allocator;
    for (auto& mcfg : ens.model_config_vec) {
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

      Ort::MemoryInfo cuda_mem_info("Cuda", OrtDeviceAllocator, mcfg.device_id, OrtMemTypeDefault);
      Ort::Allocator cuda_allocator(session, cuda_mem_info);
      SessionState sess_state{std::move(session), std::move(cuda_allocator), std::move(cuda_mem_info)};
      model_session_state_vec.push_back(std::move(sess_state));
    }
  }

  struct Token {
    int step_id;
    std::string error_msg;
    std::vector<std::string> ort_value_names;
    std::vector<Ort::Value> ort_values;
  };

  struct ExecutionFrame {
    ExecutionFrame(const PipelineSession& psess, int batch_size) {
      model_run_state_vec.reserve(psess.ens.model_config_vec.size());
      int idx = 0;
      for (const auto& mcfg : psess.ens.model_config_vec) {
        // HACK HACK TODO how do i know the shape of the past_* inputs given they've symbolic dims?
        // pre-allocate past input for the first step for both stages
        RunState rs;
        auto& cuda_allocator = psess.model_session_state_vec[idx].cuda_allocator;

        // BEGIN determine size to pre-allocate
        const auto& session = psess.model_session_state_vec[idx].session;
        auto output_state_shape = GetShape(session, mcfg.present_output_names, mcfg.present_output_names[0], false);
        // override batch and seq dims with batch_size and maximum seq len
        output_state_shape[mcfg.batch_dim_index_in_state] = batch_size;
        output_state_shape[mcfg.seq_len_dim_index_in_state] = psess.ens.max_seq_len;
        auto num_elements = std::accumulate(std::begin(output_state_shape), std::end(output_state_shape), 1, std::multiplies<int>());
        int size_to_allocate = sizeof(Ort::Float16_t) * num_elements;
        // END determine size to pre-allocate

        rs.state_buffer_1 = cuda_allocator.GetAllocation(size_to_allocate);
        rs.state_buffer_2 = cuda_allocator.GetAllocation(size_to_allocate);

        auto io_binding = std::make_unique<Ort::IoBinding>(psess.model_session_state_vec[idx].session);
        model_run_state_vec.push_back(std::move(rs));

        ++idx;
      }
    }

    struct RunState {
      // needs to be stored per model since it's associated with a session
      std::unique_ptr<Ort::IoBinding> io_binding;
      std::unordered_map<std::string, Ort::Value> output_val_map;
      std::unordered_map<std::string, Ort::MemoryAllocation> state_buffer_1_map;  // pre-allocated on cuda
      std::unordered_map<std::string, Ort::MemoryAllocation> state_buffer_2_map;  // pre-allocated on cuda
    };
    std::vector<RunState> model_run_state_vec;
  };

  struct SessionState {
    Ort::Session session;
    // needs to be stored per model since it's associated with a session
    Ort::Allocator cuda_allocator;
    // needs to be stored per model since it's associated with device id
    Ort::MemoryInfo cuda_mem_info;
  };

  std::vector<SessionState> model_session_state_vec;
  PipelineConfig ens;
};

// void SetupEnsemble(PipelineConfig& ens) {
//   std::string model_name1 = "mul_1";
//   std::string model_name2 = "mul_2";
//   std::string input_name = "X";
//   std::string output_name = "Y";
//   std::string model_path = "/bert_ort/pranav/onnxruntime/onnxruntime/test/testdata/mul_1.onnx";
//   PipelineConfig::ModelConfig m1{model_name1,
//                            model_path,
//                            {},
//                            {},
//                            {{output_name, input_name}},
//                            {},
//                            {},
//                            2};

//   PipelineConfig::ModelConfig m2{model_name2,
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
  auto cpu_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto input_ids_tensor = Ort::Value::CreateTensor<int64_t>(cpu_memory_info, input_ids.data(), input_tensor_size,
                                                            input_node_dims.data(), input_node_dims.size());
  auto posn_ids_tensor = Ort::Value::CreateTensor<int64_t>(cpu_memory_info, posn_ids.data(), input_tensor_size,
                                                           input_node_dims.data(), input_node_dims.size());
  ort_inputs.push_back(std::move(input_ids_tensor));
  ort_inputs.push_back(std::move(posn_ids_tensor));

  // all past inputs
  std::vector<int64_t> past_dims{2, batch_size, 32, 0, 128};
  for (int i = 0; i < 24; ++i) {
    Ort::Float16_t values[] = {15360};
    auto past_tensor = Ort::Value::CreateTensor<Ort::Float16_t>(cpu_memory_info, values, sizeof(values) / sizeof(values[0]),
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
  // PipelineConfig ens;
  // SetupEnsemble(ens);
  // PipelineSession pipeline_session(ens, env);

  // Run the pipeline
  std::vector<Ort::Value> output_values;
  std::vector<std::string> output_names{"logits"};
  pipeline_session.Run(input_node_names, ort_inputs, output_names, output_values,
                       max_steps, batch_size, 10);
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
}

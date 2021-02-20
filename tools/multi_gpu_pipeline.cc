// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.

// Compile
// g++ -g -std=c++14 -o ~/multi_gpu_pipeline onnxruntime/tools/multi_gpu_pipeline.cc -I /bert_ort/pranav/onnxruntime/include/onnxruntime/core/session -I /bert_ort/pranav/onnxruntime/include/onnxruntime/core/providers/cuda/ -I /bert_ort/pranav/onnxruntime/tools/ -lonnxruntime -L  /bert_ort/pranav/onnxruntime/build/Linux/Debug/ -Wl,-rpath,/bert_ort/pranav/onnxruntime/build/Linux/Debug/ -I/usr/local/cuda/include -L/usr/local/cuda/lib -lcuda -lcudart -lpthread

#include "bits/stdc++.h"
#include <assert.h>
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#include "cuda_provider_factory.h"
// #include "tbb/pipeline.h"
#include "multi_gpu_pipeline.h"
#include "json.hpp"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "task_thread_pool.h"
#include "response_queue.h"

using json = nlohmann::json;
using namespace std;

/*
Prototype for pipeline parallelism for the 10B turing model.
*/

// TODO replace all asserts with proper error checks

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

static std::pair<bool, int> Exists(const std::vector<std::string>& vec, const std::string& to_find) {
  auto it = std::find(std::begin(vec), std::end(vec), to_find);
  if (it != std::end(vec)) {
    return {true, it - std::begin(vec)};
  } else {
    return {false, -1};
  }
}

static std::vector<int64_t> GetShape(Ort::Session& sess,
                                     int io_idx,
                                     bool is_input) {
  std::vector<int64_t> retval;
  if (is_input) {
    retval = sess.GetInputTypeInfo(io_idx).GetTensorTypeAndShapeInfo().GetShape();
  } else {
    retval = sess.GetOutputTypeInfo(io_idx).GetTensorTypeAndShapeInfo().GetShape();
  }

  return retval;
}

RequestExecutionFrame::RequestExecutionFrame(/*const*/ PipelineSession& psess
                                             /*passing by non-const ref since i need to call GetAllocation on the allocator*/,
                                             int req_idx0,
                                             ReqId req_id0,
                                             int batch_size0,
                                             int orig_input_seq_len0,
                                             int stage_id0)
    : req_index(req_idx0),
      req_id(req_id0),
      batch_size(batch_size0),
      orig_input_seq_len(orig_input_seq_len0),
      stage_id(stage_id0) {
  model_run_state_vec.reserve(psess.ens.num_stages);
  int idx = 0;
  for (const auto& mcfg : psess.ens.model_config_vec) {
    RunState rs;
    const auto& cuda_mem_info = psess.model_session_state_vec[idx].cuda_mem_info;
    auto& cuda_allocator = psess.model_session_state_vec[idx].cuda_allocator;
    auto& session = psess.model_session_state_vec[idx].session;

    // Pre-allocate memory for both input and output states
    // Calcuate the amount of memory to allocate
    // For now assume all state inputs/outputs have the same shape and the same indices for batch and seq dimension
    // This allows us to calculate the shape only once.
    auto rc = Exists(mcfg.input_names, mcfg.state_input_names[0]);
    assert(rc.first);
    auto io_idx = rc.second;
    auto state_shape = GetShape(session, io_idx, true);
    // override batch and seq dims with batch_size and maximum seq len
    state_shape[mcfg.batch_dim_index_in_state] = batch_size;
    state_shape[mcfg.seq_len_dim_index_in_state] = psess.ens.max_seq_len;
    auto num_elements = std::accumulate(std::begin(state_shape), std::end(state_shape), 1, std::multiplies<int>());
    int size_to_allocate = sizeof(Ort::Float16_t) * num_elements;  // TODO don't hardcode type

    // pre-allocate buffers for input and output states
    for (const auto& name : mcfg.state_input_names) {
      rs.state_buffer_1_vec.push_back(cuda_allocator.GetAllocation(size_to_allocate));
      rs.state_buffer_2_vec.push_back(cuda_allocator.GetAllocation(size_to_allocate));
    }

    // initialize the output states
    // intentionally 0 since when the model is run the first time, there's no past state to feed.
    state_shape[mcfg.seq_len_dim_index_in_state] = 0;
    for (int idx = 0; idx < mcfg.state_output_names.size(); ++idx) {
      const auto& oname = mcfg.state_output_names[idx];
      auto& mem_allocation = rs.state_buffer_1_vec[idx];
      auto ort_val = Ort::Value::CreateTensor(
          cuda_mem_info, mem_allocation.get(), mem_allocation.size(),
          state_shape.data(), state_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);  // TODO remove hardcoded type
      rs.output_val_map.emplace(oname, std::move(ort_val));
    }

    rs.io_binding = std::make_unique<Ort::IoBinding>(psess.model_session_state_vec[idx].session);
    model_run_state_vec.push_back(std::move(rs));

    ++idx;
  }
}

ReqId CreateRequestId() {
  using namespace std::chrono;
  return duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
}

struct RequestProcessor {
  std::shared_ptr<Token> ProcessRequest(const Token& token,
                                        /*const*/ PipelineSession& psess /* passed by ref or else sess.Run won't work as Run is non-const */,
                                        RequestExecutionFrame& exec_frame /* intentional as we'll update the state */) {
    std::cout << "Executing req_id(" << token.req_id << ")/step(" << token.step_id << ")/stage(" << exec_frame.stage_id << ")" << std::endl;

    const auto model_idx = exec_frame.stage_id;
    // const auto model_idx = psess.ens.model_idx_map.at(model_name);
    const auto& model_config = psess.ens.model_config_vec[model_idx];
    auto& session_state = psess.model_session_state_vec[model_idx];
    auto& ort_sess = session_state.session;

    RequestExecutionFrame::RunState& run_state = exec_frame.model_run_state_vec[model_idx];

    // CheckStatus(g_ort->SetCurrentGpuDeviceId(model_config.device_id));

    std::vector<const char*> model_output_names;
    for (const auto& elem : model_config.output_names) {
      model_output_names.push_back(elem.c_str());
    }

    auto token_ptr = std::make_shared<Token>();
    token_ptr->step_id = token.step_id;
    token_ptr->req_id = token.req_id;

    auto& io_binding = *run_state.io_binding;
    io_binding.ClearBoundInputs();
    io_binding.ClearBoundOutputs();

    // inputs
    // go through all the inputs from the config and for each one if you find it in token.input_names
    // use the value from there.
    // else search this input name inside state_input_names. If found, get the corresponding output name from
    // state_output_names and the OrtValue associated with it.
    for (const auto& iname : model_config.input_names) {
      auto rc = Exists(token.ort_value_names, iname);
      if (rc.first) {
        // std::cout << stage_id << "/" << token.step_id << " binding input " << token.ort_value_names[rc.second] << std::endl;
        io_binding.BindInput(token.ort_value_names[rc.second].c_str(), token.ort_values[rc.second]);
        continue;
      }

      rc = Exists(model_config.state_input_names, iname);
      if (rc.first) {
        const auto& mapped_oname = model_config.state_output_names[rc.second];
        // std::cout << stage_id << "/" << token.step_id << " state_binding " << iname << " with value of " << mapped_oname << std::endl;
        io_binding.BindInput(iname.c_str(),
                             run_state.output_val_map.at(mapped_oname));
      }
    }

    // allocate outputs
    // output seq len = current input seq len + past seq len (which is 0 the first time)
    // if output is a state, use the pre-allocated buffer to create an OrtValue and bind it.
    // if output is not a state, bind using just cuda_mem_info.

    // get seq len of input_ids
    auto rc = Exists(token.ort_value_names, model_config.input_to_use_for_seq_len);
    assert(rc.first);  // TODO
    auto& input_ort_value = token.ort_values[rc.second];
    int input_seq_len = input_ort_value.GetTensorTypeAndShapeInfo().GetShape()[model_config.seq_len_dim_index_in_input];

    // get past seq len
    // assume past_seq_len is same for all states
    // int past_seq_len = 0;
    // // for the very first run there's no past.
    // if (exec_frame.stage_id == 0 && token.step_id == 0) {
    //   past_seq_len = 0;
    // } else {
    //   past_seq_len = run_state.output_val_map.at(model_config.state_output_names[0])
    //                      .GetTensorTypeAndShapeInfo()
    //                      .GetShape()[model_config.seq_len_dim_index_in_state];
    // }
    int past_seq_len = run_state.output_val_map.at(model_config.state_output_names[0])
                           .GetTensorTypeAndShapeInfo()
                           .GetShape()[model_config.seq_len_dim_index_in_state];

    // new seq len for state output = seq len of input_ids + past_seq_len
    int new_seq_len = input_seq_len + past_seq_len;

    // populate shape for state outputs
    // assume same shape for all outputs
    auto rc2 = Exists(model_config.output_names, model_config.state_output_names[0]);
    assert(rc2.first);
    auto out_idx = rc2.second;
    auto state_shape = GetShape(ort_sess, out_idx, false /*output*/);
    state_shape[model_config.batch_dim_index_in_state] = exec_frame.batch_size;
    state_shape[model_config.seq_len_dim_index_in_state] = new_seq_len;

    // assume types are same for all outputs
    auto out_type = ort_sess.GetOutputTypeInfo(out_idx).GetTensorTypeAndShapeInfo().GetElementType();

    for (const auto& oname : model_config.output_names) {
      auto rc = Exists(model_config.state_output_names, oname);
      if (rc.first) {
        auto& mem_allocation = token.step_id % 2 == 0  // even: use buffer1 for input and buffer2 for output
                                   ? run_state.state_buffer_2_vec[rc.second]
                                   : run_state.state_buffer_1_vec[rc.second];
        // cout << "mem allocation size: " << mem_allocation.size() << "\n";
        auto output_ort_val = Ort::Value::CreateTensor(
            session_state.cuda_mem_info, mem_allocation.get(), mem_allocation.size(),
            state_shape.data(), state_shape.size(), out_type);
        // std::cout << "step(" << token.step_id << ") / stage(" << exec_frame.stage_id << ")"
        //           << " created tensor for " << oname << "\n";
        io_binding.BindOutput(oname.c_str(), output_ort_val);
      } else {
        io_binding.BindOutput(oname.c_str(), session_state.cuda_mem_info);  // hidden_states
      }
    }

    // run
    // std::cout << "step(" << token.step_id << ") / stage(" << exec_frame.stage_id << ")"
    //           << " just before run\n";
    ort_sess.Run({}, io_binding);
    // std::cout << "step(" << token.step_id << ") / stage(" << exec_frame.stage_id << ")"
    //           << " Done with run\n";
    // now populate token and save state from this run
    auto vec_out_vals = io_binding.GetOutputValues();
    for (int i = 0; i < model_output_names.size(); ++i) {
      const auto& oname = model_output_names[i];

      // Assume that the same output name is not present in both the state that needs to be kept
      // and that needs to be passed on to the next layer.
      auto is_loop_back_state_output = Exists(model_config.state_output_names, oname);
      assert(!(is_loop_back_state_output.first && model_config.output_input_map.count(oname)));

      // if this output is present in state_output_names, store it in model_run_state_vec
      // because we don't want to store all outputs
      if (is_loop_back_state_output.first) {
        // std::cout << "step(" << token.step_id << ") / stage(" << exec_frame.stage_id << ")"
        //           << " saving state " << oname << std::endl;
        run_state.output_val_map.emplace(oname, std::move(vec_out_vals[i]));
        continue;
      }

      // only pass those outputs to the next layer for which there is a config in the ensemble
      // other outputs are states to be used in the next run
      if (model_config.output_input_map.count(oname)) {
        // std::cout << stage_id << "/" << token.step_id << " mapping output " << oname << " to input of next stage "
        //           << model_config.output_input_map[oname] << std::endl;
        token_ptr->ort_value_names.push_back(model_config.output_input_map.at(oname));  // input_hidden_states
        token_ptr->ort_values.push_back(std::move(vec_out_vals[i]));
      }
    }
    std::cout << "Done executing req_id(" << token.req_id << ")/step(" << token.step_id << ")/stage(" << exec_frame.stage_id << ")" << std::endl;
    return token_ptr;
  };
};

// TODO proper error handling
// TODO - change position_ids for step > 0
// TODO - token memory can be optimized
OrtStatus* PipelineSession::Run(/*const*/ std::vector<OrtReq>& req_list, std::vector<OrtResp>& resp_list, int max_steps) {
  ResponseQueue<std::shared_ptr<Token>> rq;
  std::unordered_map<ReqId, RequestExecutionFrame> req_frame_map;
  std::unordered_map<ReqId, std::future<void>> req_future_map;

  int nreqs = req_list.size();

  // Enqueue the first step and first stage processing for all the requests
  RequestProcessor rp;
  auto cpu_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  for (int req_idx = 0; req_idx < nreqs; ++req_idx) {
    ReqId req_id = CreateRequestId();
    std::cout << "creating req_id: " << req_id << std::endl;
    auto& one_req = req_list[req_idx];
    int stage_id = 0;

    // store batch size and input seq len to change position_ids for step > 0
    auto rc = Exists(one_req.input_names, ens.model_config_vec[0].input_to_use_for_seq_len);
    assert(rc.first);
    const auto& shape = one_req.input_values[rc.second]
                            .GetTensorTypeAndShapeInfo()
                            .GetShape();
    int orig_seq_len = shape[ens.model_config_vec[0].seq_len_dim_index_in_input];

    RequestExecutionFrame ref(*this,
                              req_idx,
                              req_id,
                              one_req.batch_size,
                              orig_seq_len,
                              stage_id);
    req_frame_map.emplace(req_id, std::move(ref));

    int step_id = 0;
    // TODO this moves the input names and values
    auto in_token_ptr = std::shared_ptr<Token>(new Token{req_id, step_id, std::move(one_req.input_names),
                                                         std::move(one_req.input_values)});
    auto lambda = [this, &rq, &rp, &in_token_ptr, &req_frame_map, req_id]() {
      auto out_token_ptr = rp.ProcessRequest(*in_token_ptr, *this, req_frame_map.at(req_id));
      rq.Put(out_token_ptr);
    };
    std::packaged_task<void()> task(lambda);
    req_future_map[req_id] = std::move(task.get_future());
    tp.RunTask(std::move(task));
    // task();
    try {
      req_future_map.at(req_id).get();
    } catch (const std::exception& e) {
      throw;
    }
  }

  // now read the response queue and enqueue further steps/stages for processing
  // passing the output of one stage to the next one
  int req_processed = 0;
  while (req_processed < nreqs) {
    auto token_ptr = rq.Get();
    // TODO call get on the future to catch exceptions
    ReqId req_id = token_ptr->req_id;
    int step_id = token_ptr->step_id;
    auto& exec_frame = req_frame_map.at(req_id);
    exec_frame.stage_id = (exec_frame.stage_id + 1) % ens.num_stages;

    if (exec_frame.stage_id == 0) {  // this means we've reached step > 0
      ++step_id;
      if (step_id == max_steps) {
        // look for the requested output_names in the token
        // fetch the corresponding Ort::Value and copy it to OrtResp
        int req_index = exec_frame.req_index;
        for (const auto& oname : resp_list[req_index].output_names) {
          auto ex = Exists(token_ptr->ort_value_names, oname);
          if (ex.first) {
            resp_list[req_index].output_values.push_back(std::move(token_ptr->ort_values[ex.second]));
          } else {
            // case when the user requested for an invalid output
            resp_list[req_index].output_values.push_back(Ort::Value{nullptr});
          }
        }
        ++req_processed;
        continue;
      } else {
        token_ptr->step_id = step_id;
        // TODO adjust input_ids and position_ids as per the inter-model feedback loop
        token_ptr->ort_value_names.clear();
        token_ptr->ort_values.clear();

        int batch_size = exec_frame.batch_size;
        int new_seq_len = exec_frame.orig_input_seq_len + step_id - 1;

        // update input_ids
        // HACK HACK placeholder - should post process logits to get the new input_ids
        // but logits will be on the GPU device, so we'll have to copy it to CPU to post-process it
        std::vector<int64_t> input_ids(batch_size, new_seq_len);
        std::vector<int64_t> input_node_dims{batch_size, 1};  // TODO get shape from API
        // assume shape is same for both input_ids and position_ids
        auto input_ids_tensor = Ort::Value::CreateTensor<int64_t>(cpu_memory_info, input_ids.data(), input_ids.size(),
                                                                  input_node_dims.data(), input_node_dims.size());  // TODO don't hardcode type
        // update position ids
        std::vector<int64_t> posn_ids(batch_size, new_seq_len);
        auto posn_ids_tensor = Ort::Value::CreateTensor<int64_t>(cpu_memory_info, posn_ids.data(), posn_ids.size(),
                                                                 input_node_dims.data(), input_node_dims.size());  // TODO don't hardcode type

        token_ptr->ort_value_names = {"input_ids", "position_ids"};  // TODO don't hardcode
        token_ptr->ort_values.push_back(std::move(input_ids_tensor));
        token_ptr->ort_values.push_back(std::move(posn_ids_tensor));
      }
    } else {
      token_ptr->req_id = req_id;
      token_ptr->step_id = step_id;
    }

    auto lambda = [this, &rq, &rp, &token_ptr, &req_frame_map, req_id]() {
      auto out_token_ptr = rp.ProcessRequest(*token_ptr, *this, req_frame_map.at(req_id));
      rq.Put(out_token_ptr);
    };
    std::packaged_task<void()> task(lambda);
    req_future_map[req_id] = std::move(task.get_future());
    tp.RunTask(std::move(task));
    // task();
    try {
      req_future_map.at(req_id).get();
    } catch (const std::exception& e) {
      throw;
    }
  }

  return nullptr;
}

void PipelineSession::ParseEnsembleJsonFile(const std::string& ensemble_json_file, Ensemble& ens) {
  std::ifstream ifs(ensemble_json_file);
  if (!ifs.good()) {
    throw std::runtime_error("File error");
  }

  auto j = json::parse(ifs, nullptr, true, true);
  ens.max_seq_len = j["max_seq_len"];
  int idx = 0;
  for (const auto& m : j["ensemble"]) {
    Ensemble::ModelConfig cfg;
    std::string model_name = m["model_name"];
    cfg.model_name = model_name;
    cfg.model_file_path = m["model_file_path"];
    cfg.input_to_use_for_seq_len = m["input_to_use_for_seq_len"];
    cfg.seq_len_dim_index_in_input = m["seq_len_dim_index_in_input"];
    cfg.batch_dim_index_in_input = m["batch_dim_index_in_input"];
    cfg.batch_dim_index_in_state = m["batch_dim_index_in_state"];
    cfg.seq_len_dim_index_in_state = m["seq_len_dim_index_in_state"];
    cfg.device_id = m["device_id"];  // TODO validate device id

    const char* key = "output_input_map";  // TODO validate entries of this map
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

    // TODO validate sizes of state_input_names and state_output_names is same

    ens.model_config_vec.push_back(std::move(cfg));
    ens.model_idx_map[model_name] = idx;
    ++idx;
  }

  ens.num_stages = ens.model_config_vec.size();
}

PipelineSession::PipelineSession(const std::string& ensemble_json_file, Ort::Env& env) : tp(10) {
  ParseEnsembleJsonFile(ensemble_json_file, ens);
  Init(ens, env);
}

PipelineSession::PipelineSession(const Ensemble& ens0, Ort::Env& env) : ens(ens0), tp(10) {
  Init(ens, env);
}

void PipelineSession::Init(Ensemble& ens, Ort::Env& env) {
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

int main(int argc, char* argv[]) {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

  // read ensemble file name
  std::string ensemble_file_name = "/bert_ort/pranav/onnxruntime/tools/turing_model_ensemble.json";
  if (argc > 1) {
    ensemble_file_name = argv[1];
  }
  std::cout << "Using ensemble file: " << ensemble_file_name << "\n";

  int max_steps = 1;
  if (argc > 2) {
    max_steps = atoi(argv[2]);
  }
  std::cout << "Using max_steps = " << max_steps << "\n";

  int max_num_reqs = 1;
  if (argc > 3) {
    max_num_reqs = atoi(argv[3]);
  }
  std::cout << "Using max_num_reqs = " << max_num_reqs << "\n";

  // setup the pipeline session
  PipelineSession pipeline_session(ensemble_file_name, env);

  // prepare inputs
  int batch_size = 1;
  int seq_len = 1;
  size_t input_tensor_size = batch_size * seq_len;
  std::vector<int64_t> input_node_dims{batch_size, seq_len};
  std::vector<std::string> input_node_names{"input_ids", "position_ids"};
  std::vector<int64_t> input_ids(input_tensor_size, 1);
  std::vector<int64_t> posn_ids(input_tensor_size, 0);
  std::vector<std::string> output_node_names = {"logits"};

  int c = 1;
  for (unsigned int i = 0; i < input_tensor_size; ++i, ++c) {
    input_ids[i] = static_cast<int64_t>(c);
    posn_ids[i] = static_cast<int64_t>(c);
  }

  std::vector<OrtReq> req_list;
  for (int i = 0; i < max_num_reqs; ++i) {
    std::vector<Ort::Value> ort_inputs;
    auto cpu_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_ids_tensor = Ort::Value::CreateTensor<int64_t>(cpu_memory_info, input_ids.data(), input_tensor_size,
                                                              input_node_dims.data(), input_node_dims.size());
    auto posn_ids_tensor = Ort::Value::CreateTensor<int64_t>(cpu_memory_info, posn_ids.data(), input_tensor_size,
                                                             input_node_dims.data(), input_node_dims.size());
    ort_inputs.push_back(std::move(input_ids_tensor));
    ort_inputs.push_back(std::move(posn_ids_tensor));
    OrtReq one_req{input_node_names, std::move(ort_inputs), batch_size};
    req_list.push_back(std::move(one_req));
  }

  std::vector<OrtResp> resp_list;
  std::vector<std::string> output_names{"logits"};
  for (int i = 0; i < max_num_reqs; ++i) {
    OrtResp one_resp{output_names};
    resp_list.push_back(std::move(one_resp));
  }

  // Run the pipeline
  pipeline_session.Run(req_list, resp_list, max_steps);
  // assert(output_values.size() == output_names.size());

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

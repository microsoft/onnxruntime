// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.

// Compile
// g++ -g -std=c++14 -o ~/multi_gpu_pipeline onnxruntime/tools/multi_gpu_pipeline.cc -I \
/bert_ort/pranav/onnxruntime/include/onnxruntime/core/session -I /bert_ort/pranav/onnxruntime/include/onnxruntime/core/providers/cuda/\
 -I /bert_ort/pranav/onnxruntime/tools/ -lonnxruntime -L  /bert_ort/pranav/onnxruntime/build/Linux/Debug/ \
 -Wl,-rpath,/bert_ort/pranav/onnxruntime/build/Linux/Debug/ -I/usr/local/cuda/include -I /bert_ort/pranav/eigen-d10b2/ \
 -L/usr/local/cuda/lib -lcuda -lcudart -lpthread

// g++ -g -std=c++14 -o ~/multi_gpu_pipeline onnxruntime/tools/multi_gpu_pipeline.cc -I \
 /bert_ort/pranav/onnxruntime-linux-x64-gpu-1.6.0/include -I /bert_ort/pranav/onnxruntime/tools/ -lonnxruntime \
 -L /bert_ort/pranav/onnxruntime-linux-x64-gpu-1.6.0/lib/ -Wl,-rpath,/bert_ort/pranav/onnxruntime-linux-x64-gpu-1.6.0/lib/ \
 -I/usr/local/cuda/include -I /bert_ort/pranav/eigen-d10b2/ -L/usr/local/cuda/lib -lcuda -lcudart -lpthread

#include "bits/stdc++.h"
#include <assert.h>
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#include "cuda_provider_factory.h"
#include "multi_gpu_pipeline.h"
#include "json.hpp"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "task_thread_pool.h"
#include "response_queue.h"
#include "Eigen/Core"
#include "Eigen/src/Core/arch/Default/Half.h"

using json = nlohmann::json;

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

struct Timer {
  // using namespace std::chrono;
  using Clock = std::chrono::high_resolution_clock;
  Timer(const char* msg0) : msg(msg0), start(Clock::now()) {
  }
  ~Timer() {
    auto stop = Clock::now();
    std::cout << msg << " took " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << " microseconds\n";
  }
  const char* msg;
  std::chrono::time_point<Clock> start;
};

// returns a pair p; p.first is true if the elem is found in which case p.second is the index of the elem in the container
static std::pair<bool, int> Contains(const std::vector<std::string>& vec, const std::string& to_find) {
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
                                             int stage_id0,
                                             OrtResp& ort_resp0)
    : req_index(req_idx0),
      req_id(req_id0),
      batch_size(batch_size0),
      orig_input_seq_len(orig_input_seq_len0),
      stage_id(stage_id0),
      ort_resp(ort_resp0) {
  model_run_state_vec.reserve(psess.pcfg.num_stages);
  int idx = 0;
  for (const auto& mcfg : psess.pcfg.model_config_vec) {
    RunState rs;
    const auto& cuda_mem_info = psess.model_session_state_vec[idx].cuda_mem_info;
    auto& session = psess.model_session_state_vec[idx].session;
    auto cuda_allocator = std::make_unique<Ort::Allocator>(session, cuda_mem_info);

    // Pre-allocate memory for both input and output states
    // Calcuate the amount of memory to allocate
    // For now assume all state inputs/outputs have the same shape and the same indices for batch and seq dimension
    // This allows us to calculate the shape only once.
    auto rc = Contains(mcfg.input_names, mcfg.state_input_names[0]);
    assert(rc.first);
    auto io_idx = rc.second;
    auto state_shape = GetShape(session, io_idx, true);
    // override batch and seq dims with batch_size and maximum seq len
    state_shape[mcfg.batch_dim_index_in_state] = batch_size;
    state_shape[mcfg.seq_len_dim_index_in_state] = psess.pcfg.max_seq_len;
    auto num_elements = std::accumulate(std::begin(state_shape), std::end(state_shape), 1, std::multiplies<int>());
    int size_to_allocate = sizeof(Ort::Float16_t) * num_elements;  // TODO don't hardcode type

    // pre-allocate buffers for input and output states
    for (const auto& name : mcfg.state_input_names) {
      rs.state_buffer_1_vec.push_back(cuda_allocator->GetAllocation(size_to_allocate));
      rs.state_buffer_2_vec.push_back(cuda_allocator->GetAllocation(size_to_allocate));
    }

    // initialize the output states
    // intentionally 0 since when the model is run the first time, there's no past state to feed.
    state_shape[mcfg.seq_len_dim_index_in_state] = 0;
    for (int idx = 0; idx < mcfg.state_output_names.size(); ++idx) {
      const auto& oname = mcfg.state_output_names[idx];
      auto& mem_allocation = rs.state_buffer_1_vec[idx];  // careful, use buffer1 here
      auto ort_val = Ort::Value::CreateTensor(
          cuda_mem_info, mem_allocation.get(), mem_allocation.size(),
          state_shape.data(), state_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);  // TODO remove hardcoded type
      rs.output_val_map.emplace(oname, std::move(ort_val));
    }

    rs.io_binding = std::make_unique<Ort::IoBinding>(psess.model_session_state_vec[idx].session);
    rs.cuda_allocator = std::move(cuda_allocator);
    model_run_state_vec.push_back(std::move(rs));

    ++idx;
  }
}

static ReqId CreateRequestId() {
  using namespace std::chrono;
  return duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
}

struct RequestProcessor {
  std::shared_ptr<Token> ProcessRequest(const Token& token,
                                        /*const*/ PipelineSession& psess /* passed by ref or else sess.Run won't work as Run is non-const */,
                                        RequestExecutionFrame& exec_frame /* pass by non-const ref intentional as we'll update the state */) {
    std::cout << "Executing req_id(" << token.req_id << ")/step(" << token.step_id << ")/stage(" << exec_frame.stage_id << ")" << std::endl;

    const auto model_idx = exec_frame.stage_id;
    // const auto model_idx = psess.pcfg.model_idx_map.at(model_name);
    const auto& model_config = psess.pcfg.model_config_vec[model_idx];
    auto& session_state = psess.model_session_state_vec[model_idx];
    auto& ort_sess = session_state.session;

    RequestExecutionFrame::RunState& run_state = exec_frame.model_run_state_vec[model_idx];

    CheckStatus(g_ort->SetCurrentGpuDeviceId(model_config.device_id));

    // convert output_names from std::string to const char*
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
      auto rc = Contains(token.ort_value_names, iname);
      if (rc.first) {
        // std::cout << stage_id << "/" << token.step_id << " binding input " << token.ort_value_names[rc.second] << std::endl;
        io_binding.BindInput(iname.c_str(), token.ort_values[rc.second]);
        continue;
      }

      rc = Contains(model_config.state_input_names, iname);
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
    auto rc = Contains(token.ort_value_names, model_config.input_to_use_for_seq_len);
    assert(rc.first);  // TODO
    auto& input_ort_value = token.ort_values[rc.second];
    int input_seq_len = input_ort_value.GetTensorTypeAndShapeInfo().GetShape()[model_config.seq_len_dim_index_in_input];

    // get past seq len
    // assume past_seq_len is same for all states
    int past_seq_len = run_state.output_val_map.at(model_config.state_output_names[0])
                           .GetTensorTypeAndShapeInfo()
                           .GetShape()[model_config.seq_len_dim_index_in_state];

    // new seq len for state output = seq len of input_ids + past_seq_len
    int new_seq_len = input_seq_len + past_seq_len;

    // populate shape for state outputs
    // assume same shape for all outputs
    auto rc2 = Contains(model_config.output_names, model_config.state_output_names[0]);
    assert(rc2.first);
    auto out_idx = rc2.second;
    auto state_shape = GetShape(ort_sess, out_idx, false /*output*/);
    state_shape[model_config.batch_dim_index_in_state] = exec_frame.batch_size;
    state_shape[model_config.seq_len_dim_index_in_state] = new_seq_len;

    // assume types are same for all outputs
    auto out_type = ort_sess.GetOutputTypeInfo(out_idx).GetTensorTypeAndShapeInfo().GetElementType();

    for (const auto& oname : model_config.output_names) {
      auto rc = Contains(model_config.state_output_names, oname);
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
        // if oname is present in OrtResp::output_names, bind the corresponding OrtValue from OrtResp
        // we check in OrtResp::output_names because we want to use the info provided by the user to tell us
        // where the output should go.
        auto rc = Contains(exec_frame.ort_resp.output_names, oname);
        if (rc.first) {  // logits
          // get the corresponding ortval from OrtResp
          auto* mem_info = exec_frame.ort_resp.output_meminfo[rc.second];
          // if user provided mem_info, use that for binding
          if (mem_info) {
            CheckStatus(g_ort->BindOutputToDevice(io_binding, oname.c_str(), mem_info));
          } else {  // bind the pre-allocated OrtVal
            auto& ort_val = exec_frame.ort_resp.output_values[rc.second];
            io_binding.BindOutput(oname.c_str(), ort_val);
          }
        } else {
          io_binding.BindOutput(oname.c_str(), session_state.cuda_mem_info);  // hidden_states
        }
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
      auto is_loop_back_state_output = Contains(model_config.state_output_names, oname);
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
        std::cout << "Copying output req_id(" << token.req_id << ")/step(" << token.step_id << ")/stage(" << exec_frame.stage_id << ") "
                  << model_config.output_input_map.at(oname) << std::endl;
        token_ptr->ort_value_names.push_back(model_config.output_input_map.at(oname));  // input_hidden_states
        token_ptr->ort_values.push_back(std::move(vec_out_vals[i]));
      }
    }
    std::cout << "Done executing req_id(" << token.req_id << ")/step(" << token.step_id << ")/stage(" << exec_frame.stage_id << ")" << std::endl;
    return token_ptr;
  };
};

// PostProcessLogits
static float HalfToFloat(uint16_t h) {
  return Eigen::half_impl::half_to_float(Eigen::half_impl::raw_uint16_to_half(h));
}

static void PostProcessLogits(int batch_size,
                              const Ort::Value& logits,
                              const std::vector<int64_t> logits_shape,
                              std::vector<int64_t>& input_ids,
                              std::vector<int64_t>& input_ids_shape) {
  input_ids.reserve(batch_size);
  input_ids_shape = std::vector<int64_t>{batch_size, 1};
  const auto* logits_data = logits.GetTensorData<Ort::Float16_t>();

  // first we need to skip the second (seq len) dimension
  std::vector<Ort::Float16_t> tmp;
  int new_size = logits_shape[0] * logits_shape[2];
  tmp.reserve(new_size);
  int num_elems = logits_shape[0] * logits_shape[1] * logits_shape[2];
  int ltwo = logits_shape[1] * logits_shape[2];
  for (int batch_id = 0; batch_id < num_elems; batch_id += ltwo) {
    for (int k = 0; k < logits_shape[2]; ++k) {
      tmp.push_back(logits_data[batch_id + ((logits_shape[1] - 1) * logits_shape[2]) + k]);
    }
  }

  // now find the max per onnx batch
  for (int batch_id = 0; batch_id < new_size; batch_id += logits_shape[2]) {
    // float max = HalfToFloat(tmp[batch_id]);
    auto max = tmp[batch_id];
    int max_idx = 0;
    for (int j = 1; j < logits_shape[2]; ++j) {
      // auto elem = HalfToFloat(tmp[batch_id + j]);
      auto elem = tmp[batch_id + j];
      if (elem > max) {
        max = elem;
        max_idx = j;
      }
    }
    input_ids.push_back(max_idx);
  }
}

// TODO proper error handling
// TODO - token memory can be optimized
// For simplicity even if one req in the batch fails, we consider the full batch to have failed.
OrtStatus* PipelineSession::Run(/*const*/ std::vector<OrtReq>& req_list, std::vector<OrtResp>& resp_list, int num_steps) {
  ResponseQueue<std::shared_ptr<Token>> rq;
  std::unordered_map<ReqId, RequestExecutionFrame> req_frame_map;

  // code that will run by the threads in the threadpool
  auto lambda_helper = [](ResponseQueue<std::shared_ptr<Token>>& rq,
                          RequestProcessor& rp,
                          const Token& token,
                          PipelineSession& psess,
                          RequestExecutionFrame& ref) {
    std::shared_ptr<Token> out_token_ptr;
    try {
      out_token_ptr = rp.ProcessRequest(token, psess, ref);
    } catch (const std::exception& e) {
      std::ostringstream error;
      error << "Error in processing request id: " << token.req_id << " with exception: " << e.what();
      out_token_ptr = std::make_shared<Token>();
      out_token_ptr->req_id = token.req_id;
      out_token_ptr->step_id = token.step_id;
      out_token_ptr->error_msg = error.str();
    } catch (...) {
      std::ostringstream error;
      error << "Error in processing request id: " << token.req_id << " with unknown exception";
      out_token_ptr = std::make_shared<Token>();
      out_token_ptr->req_id = token.req_id;
      out_token_ptr->step_id = token.step_id;
      out_token_ptr->error_msg = error.str();
    }
    rq.Put(out_token_ptr);
  };

  // First enqueue the first step and first stage processing for all the requests
  RequestProcessor rp;
  auto cpu_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  int num_reqs = req_list.size();

  for (int req_idx = 0; req_idx < num_reqs; ++req_idx) {
    ReqId req_id = CreateRequestId();
    std::cout << "creating req_id: " << req_id << std::endl;
    auto& one_req = req_list[req_idx];
    auto& one_resp = resp_list[req_idx];
    int stage_id = 0;

    // validate resp vector
    auto& ovalues = resp_list[req_idx].output_values;
    const auto& onames = resp_list[req_idx].output_names;
    assert(ovalues.size() == onames.size());

    // store batch size and input seq len to change position_ids for step > 0
    auto rc = Contains(one_req.input_names, pcfg.model_config_vec[0].input_to_use_for_seq_len);
    assert(rc.first);
    const auto& shape = one_req.input_values[rc.second]
                            .GetTensorTypeAndShapeInfo()
                            .GetShape();
    int orig_seq_len = shape[pcfg.model_config_vec[0].seq_len_dim_index_in_input];
    int batch_size = shape[pcfg.model_config_vec[0].batch_dim_index_in_input];

    RequestExecutionFrame ref(*this, req_idx, req_id, batch_size, orig_seq_len, stage_id, one_resp);
    req_frame_map.emplace(req_id, std::move(ref));

    int step_id = 0;
    // TODO this moves the input names and values
    auto in_token_ptr = std::shared_ptr<Token>(new Token{req_id, step_id, one_req.input_names,
                                                         std::move(one_req.input_values)});
    auto lambda = [this, &lambda_helper, &rq, &rp, in_token_ptr, &req_frame_map, req_id]() {
      lambda_helper(rq, rp, *in_token_ptr, *this, req_frame_map.at(req_id));
    };
    std::function<void()> task(lambda);
    tp.RunTask(std::move(task));
  }

  // get logits index
  auto f = Contains(pcfg.model_config_vec[pcfg.num_stages - 1].output_names, pcfg.logits_name);
  assert(f.first);
  int logits_idx = f.second;

  // now read the response queue and enqueue further steps/stages for processing
  // passing the output of one stage to the next one
  int req_processed = 0;
  while (req_processed < num_reqs) {
    auto token_ptr = rq.Get();
    ReqId req_id = token_ptr->req_id;
    int step_id = token_ptr->step_id;
    auto& exec_frame = req_frame_map.at(req_id);

    // fail the whole batch if even one req fails
    if (!token_ptr->error_msg.empty()) {
      return g_ort->CreateStatus(ORT_FAIL, token_ptr->error_msg.c_str());
    }

    exec_frame.stage_id = (exec_frame.stage_id + 1) % pcfg.num_stages;
    if (exec_frame.stage_id == 0) {  // this means we've reached step > 0
      ++step_id;
      if (step_id == num_steps) {  // we're done with all steps of this request, move the output
        // look for the requested output_names in the token
        // fetch the corresponding Ort::Value and copy it to OrtResp
        int req_index = exec_frame.req_index;
        int resp_index = 0;
        for (const auto& oname : resp_list[req_index].output_names) {
          auto ex = Contains(token_ptr->ort_value_names, oname);
          if (ex.first) {
            resp_list[req_index].output_values[resp_index] = std::move(token_ptr->ort_values[ex.second]);
          } else {
            // case when the user requested output was not present in the final output
            std::ostringstream ostr;
            ostr << "Error: Output " << oname << " not getting output by the final stage\n";
            return g_ort->CreateStatus(ORT_FAIL, ostr.str().c_str());
          }
          ++resp_index;
        }
        ++req_processed;
        continue;
      } else {  // done with one step; now continue with the next step
        int batch_size = exec_frame.batch_size;

        // update input_ids
        // TODO but logits will be on the GPU device, so we'll have to copy it to CPU to post-process it
        // get index of 'logits' output
        auto rc = Contains(token_ptr->ort_value_names, pcfg.logits_name);
        if (!rc.first) {
          return g_ort->CreateStatus(ORT_FAIL, "Did not get logits in the output");
        }
        std::vector<int64_t> input_ids;
        std::vector<int64_t> input_ids_shape;
        auto logits_shape = model_session_state_vec[pcfg.num_stages - 1]  // only the last stage has logits as output
                                .session.GetOutputTypeInfo(logits_idx)
                                .GetTensorTypeAndShapeInfo()
                                .GetShape();
        // replace batch size dim since shape returned has -1
        // we don't care about seq len since we'll throw it away
        logits_shape[0] = batch_size;
        PostProcessLogits(batch_size, token_ptr->ort_values[rc.second], logits_shape, input_ids, input_ids_shape);

        // assume shape is same for both input_ids and position_ids
        auto input_ids_tensor = Ort::Value::CreateTensor<int64_t>(cpu_memory_info, input_ids.data(), input_ids.size(),
                                                                  input_ids_shape.data(), input_ids_shape.size());  // TODO don't hardcode type

        // update position ids
        // assume shape of position ids is same as input_ids
        int new_seq_len = exec_frame.orig_input_seq_len + step_id - 1;
        std::vector<int64_t> posn_ids(batch_size, new_seq_len);
        auto posn_ids_tensor = Ort::Value::CreateTensor<int64_t>(cpu_memory_info, posn_ids.data(), posn_ids.size(),
                                                                 input_ids_shape.data(), input_ids_shape.size());  // TODO don't hardcode type

        token_ptr->Clear();
        token_ptr->req_id = req_id;
        token_ptr->step_id = step_id;
        token_ptr->ort_value_names = {pcfg.input_ids_name, pcfg.position_ids_name};
        token_ptr->ort_values.push_back(std::move(input_ids_tensor));
        token_ptr->ort_values.push_back(std::move(posn_ids_tensor));
      }
    } else {  // continue executing the next stage
      token_ptr->req_id = req_id;
      token_ptr->step_id = step_id;
    }

    auto lambda = [this, &lambda_helper, &rq, &rp, token_ptr, &req_frame_map, req_id]() {
      lambda_helper(rq, rp, *token_ptr, *this, req_frame_map.at(req_id));
    };
    std::function<void()> task(lambda);
    tp.RunTask(std::move(task));
  }

  return nullptr;
}

void PipelineSession::ParseEnsembleJsonFile(const std::string& ensemble_json_file, PipelineConfig& pcfg) {
  std::ifstream ifs(ensemble_json_file);
  if (!ifs.good()) {
    throw std::runtime_error(std::string("Error reading file ") + ensemble_json_file);
  }

  auto j = json::parse(ifs, nullptr, true, true);

  pcfg.input_ids_name = j["input_ids_name"];
  pcfg.position_ids_name = j["position_ids_name"];
  pcfg.logits_name = j["logits_name"];
  pcfg.max_seq_len = j["max_seq_len"];
  int idx = 0;
  for (const auto& m : j["ensemble"]) {
    PipelineConfig::ModelConfig cfg;
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

    pcfg.model_config_vec.push_back(std::move(cfg));
    pcfg.model_idx_map[model_name] = idx;
    ++idx;
  }

  pcfg.num_stages = pcfg.model_config_vec.size();
}

bool PipelineSession::Validate(const PipelineConfig& pcfg) {
  // TODO validate
  return true;
}

PipelineSession::PipelineSession(const std::string& ensemble_json_file, Ort::Env& env) : tp(10) {
  ParseEnsembleJsonFile(ensemble_json_file, pcfg);
  auto rc = Validate(pcfg);
  assert(rc);
  Init(pcfg, env);
}

PipelineSession::PipelineSession(const PipelineConfig& ens0, Ort::Env& env) : pcfg(ens0), tp(10) {
  auto rc = Validate(pcfg);
  assert(rc);
  Init(pcfg, env);
}

void PipelineSession::Init(PipelineConfig& pcfg, Ort::Env& env) {
  Ort::AllocatorWithDefaultOptions ort_allocator;
  for (auto& mcfg : pcfg.model_config_vec) {
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
    SessionState sess_state{std::move(session), std::move(cuda_mem_info)};
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

  int num_steps = 1;
  if (argc > 2) {
    num_steps = atoi(argv[2]);
  }
  std::cout << "Using num_steps = " << num_steps << "\n";

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
  std::vector<std::string> input_node_names{pipeline_session.pcfg.input_ids_name,
                                            pipeline_session.pcfg.position_ids_name};
  std::vector<int64_t> input_ids(input_tensor_size, 1);
  std::vector<int64_t> posn_ids(input_tensor_size, 0);
  std::vector<std::string> output_node_names = {pipeline_session.pcfg.logits_name};

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
    OrtReq one_req{input_node_names, std::move(ort_inputs)};
    req_list.push_back(std::move(one_req));
  }

  auto cpu_mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  std::vector<OrtResp> resp_list;
  std::vector<std::string> output_names{pipeline_session.pcfg.logits_name};
  for (int i = 0; i < max_num_reqs; ++i) {
    OrtResp one_resp;
    for (const auto& oname : output_names) {
      one_resp.output_names.push_back(oname);
      one_resp.output_values.push_back(Ort::Value{nullptr});
      one_resp.output_meminfo.push_back(cpu_mem_info);
    }
    resp_list.push_back(std::move(one_resp));
  }

  // Run the pipeline
  OrtStatus* status = pipeline_session.Run(req_list, resp_list, num_steps);
  std::unique_ptr<OrtStatus, decltype(g_ort->ReleaseStatus)> status_deleter(status, g_ort->ReleaseStatus);
  if (status) {
    std::cout << "Execution failed with error " << g_ort->GetErrorMessage(status) << std::endl;
    return -1;
  }

  for (int idx = 0; idx < resp_list.size(); ++idx) {
    assert(resp_list[idx].output_names[0] == pipeline_session.pcfg.logits_name);
    auto retval = std::move(resp_list[idx].output_values[0]);
    assert(retval.IsTensor());
    assert(retval.GetTensorData<Ort::Float16_t>());
    assert(resp_list[0].output_values.size() == resp_list[idx].output_names.size());

    // print output
    auto* data_ptr = retval.GetTensorData<Ort::Float16_t>();
    auto num_elems = retval.GetTensorTypeAndShapeInfo().GetElementCount();
    std::cout << "Printing output " << std::endl;
    for (int i = 0; i < num_elems; i += 10000) {
      std::cout << "elem: " << data_ptr[i] << "\n";
    }
    std::cout << "\n";
  }
  printf("Done!\n");
  return 0;
}

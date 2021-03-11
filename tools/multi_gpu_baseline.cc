// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.

// Compile
// g++ -g -std=c++14 -o ~/multi_gpu_pipeline /bert_ort/pranav/onnxruntime/tools/multi_gpu_pipeline.cc -I \
/bert_ort/pranav/onnxruntime/include/onnxruntime/core/session -I /bert_ort/pranav/onnxruntime/include/onnxruntime/core/providers/cuda/ \
-I /bert_ort/pranav/onnxruntime/tools/ -lonnxruntime -L  /bert_ort/pranav/onnxruntime/build/Linux/Debug/ \
-Wl,-rpath,/bert_ort/pranav/onnxruntime/build/Linux/Debug/ -I/usr/local/cuda/include -I /bert_ort/pranav/eigen-d10b2/ \
-L/usr/local/cuda/lib -lcuda -lcudart -lpthread

#include "bits/stdc++.h"
#include <assert.h>
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#include "cuda_provider_factory.h"
#include "multi_gpu_baseline.h"
#include "json.hpp"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "task_thread_pool.h"
#include "queue.h"
#include "Eigen/Core"
#include "Eigen/src/Core/arch/Default/Half.h"
#include "cxxopts.hpp"

namespace onnxruntime {
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
  using Clock = std::chrono::high_resolution_clock;
  Timer(const char* msg0) : msg(msg0), start(Clock::now()) {
  }
  ~Timer() {
    auto stop = Clock::now();
    std::cout << "TIMER: " << msg << " took " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << " microseconds\n";
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

RequestExecutionFrame::RequestExecutionFrame(PipelineSession& psess,  // passing by non-const exec_frame to create iobinding
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

    // Pre-allocate memory for both present and past states
    // Calcuate the amount of memory to allocate
    // For now assume all present and past states have the same shape and the same indices for batch and seq dimension
    // This allows us to calculate the shape only once.
    auto rc = Contains(mcfg.input_names, mcfg.past_input_names[0]);
    assert(rc.first);
    auto io_idx = rc.second;
    auto past_present_state_shape = GetShape(session, io_idx, true);
    // override batch and seq dims with batch_size and maximum seq len
    past_present_state_shape[mcfg.batch_dim_index_in_state] = batch_size;
    past_present_state_shape[mcfg.seq_len_dim_index_in_state] = psess.pcfg.max_seq_len;
    auto num_elements = std::accumulate(std::begin(past_present_state_shape), std::end(past_present_state_shape), 1, std::multiplies<int>());
    int size_to_allocate = sizeof(Ort::Float16_t) * num_elements;  // TODO don't hardcode type

    // pre-allocate buffers for input and output states
    for (const auto& name : mcfg.past_input_names) {
      rs.present_past_prealloc_buffer_1_vec.push_back(cuda_allocator->GetAllocation(size_to_allocate));
      rs.present_past_prealloc_buffer_2_vec.push_back(cuda_allocator->GetAllocation(size_to_allocate));
    }

    // initialize the output states
    // intentionally 0 since when the model is run the first time, there's no past state to feed.
    past_present_state_shape[mcfg.seq_len_dim_index_in_state] = 0;
    for (int j = 0, end = mcfg.present_output_names.size(); j < end; ++j) {
      const auto& oname = mcfg.present_output_names[j];
      auto& mem_allocation = rs.present_past_prealloc_buffer_1_vec[j];  // careful, use buffer1 here
      auto ort_val = Ort::Value::CreateTensor(
          cuda_mem_info, mem_allocation.get(), mem_allocation.size(),
          past_present_state_shape.data(), past_present_state_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);  // TODO remove hardcoded type
      rs.output_val_map[oname] = OrtValueHandle(ort_val.release());
    }

    // it's inefficient to allocate memory for the inter stage outputs for every step
    // pre-allocate buffers for inter stage outputs except the last stage
    if (idx < psess.pcfg.num_stages - 1) {
      for (const auto& elem_pair : mcfg.inter_stage_output_input_map) {
        // get the shape of the output name
        const auto& oname = elem_pair.first;
        auto rc = Contains(mcfg.output_names, oname);
        assert(rc.first);
        auto output_shape = GetShape(session, rc.second, false /*output*/);

        // replace seq_len dim with max_seq_len
        output_shape[mcfg.batch_dim_in_inter_stage_output] = batch_size;
        output_shape[mcfg.seq_len_dim_in_inter_stage_output] = psess.pcfg.max_seq_len;

        // get the total number of bytes to allocate
        auto num_elements = std::accumulate(std::begin(output_shape), std::end(output_shape), 1, std::multiplies<int>());
        int size_to_allocate = sizeof(Ort::Float16_t) * num_elements;  // TODO don't hardcode type
        // std::cout << "inter stage output num_elements " << num_elements << "\n";
        // allocate and store in map
        auto rcx = rs.inter_stage_output_prealloc_buffer_map.emplace(oname, cuda_allocator->GetAllocation(size_to_allocate));
        assert(rcx.second);
      }
    }

    rs.io_binding = std::make_unique<Ort::IoBinding>(psess.model_session_state_vec[idx].session);
    rs.cuda_allocator = std::move(cuda_allocator);
    model_run_state_vec.push_back(std::move(rs));

    ++idx;
  }
}

static ReqId CreateRequestId() {
  static int req_id;
  return ++req_id;
  // using namespace std::chrono;
  // return duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
}

static float HalfToFloat(uint16_t h) {
  return Eigen::half_impl::half_to_float(Eigen::half_impl::raw_uint16_to_half(h));
}

struct RequestExecutor {
  Token* ExecuteRequest(Token& token,
                        const PipelineConfig::ModelConfig& mcfg,
                        PipelineSession::SessionState& session_state,
                        RequestExecutionFrame& exec_frame /* pass by non-const exec_frame intentional as we'll update the state */) {
    // std::ostringstream ostr;
    // ostr << "Executing req_id(" << token.req_id << ")/step(" << token.step_id << ")/stage(" << exec_frame.stage_id << ")";
    // const std::string& str = ostr.str();
    // std::cout << str << "\n";
    // Timer t(str.c_str());

    int model_idx = exec_frame.stage_id;
    RequestExecutionFrame::RunState& run_state = exec_frame.model_run_state_vec[model_idx];

    // set the GPU device id for this thread
    CheckStatus(g_ort->SetCurrentGpuDeviceId(mcfg.device_id));

    // reuse the token; move the things out of this token since we'll overwrite them
    auto* out_token_ptr = &token;
    auto in_token_ort_value_names = token.ort_value_names;
    std::vector<OrtValueHandle> in_token_ort_values;
    in_token_ort_values.reserve(token.ort_values.size());
    for (auto& v : token.ort_values) {
      in_token_ort_values.push_back(std::move(v));
    }
    token.ort_value_names.clear();
    token.ort_values.clear();

    // if (token.step_id > 0 && exec_frame.stage_id == 0) {
    //   auto* posn_ids_data = in_token_ort_values[1].GetTensorData<int64_t>();
    //   for (int i = 0; i < exec_frame.batch_size; ++i) {
    //     std::cout << "step_id: " << token.step_id << " posn: " << posn_ids_data[i] << "\n";
    //   }
    // }

    auto& io_binding_obj = *run_state.io_binding;
    auto* io_binding = static_cast<OrtIoBinding*>(io_binding_obj);
    io_binding_obj.ClearBoundInputs();
    io_binding_obj.ClearBoundOutputs();

    // inputs
    // go through all the inputs from the config and for each one if you find it in token.input_names
    // use the value from there.
    // else search this input name inside past_input_names. If found, get the corresponding output name from
    // present_output_names and the OrtValue associated with it.
    for (const auto& iname : mcfg.input_names) {
      auto rc = Contains(in_token_ort_value_names, iname);
      if (rc.first) {
        // std::cout << stage_id << "/" << token.step_id << " binding input " << token.ort_value_names[rc.second] << "\n";
        CheckStatus(g_ort->BindInput(io_binding, iname.c_str(), in_token_ort_values[rc.second]));
        continue;
      }

      rc = Contains(mcfg.past_input_names, iname);
      if (rc.first) {
        const auto& mapped_oname = mcfg.present_output_names[rc.second];
        // std::cout << stage_id << "/" << token.step_id << " state_binding " << iname << " with value of " << mapped_oname << "\n";
        CheckStatus(g_ort->BindInput(io_binding, iname.c_str(), run_state.output_val_map.at(mapped_oname)));
      }
    }

    // allocate outputs
    // output seq len = current input seq len + past seq len (which is 0 the first time)
    // if output is a state, use the pre-allocated buffer to create an OrtValue and bind it.
    // if output is not a state, bind using just cuda_mem_info.

    // get seq len of input_ids (stage 0) or input_hidden_states (stage 1)
    auto rc = Contains(in_token_ort_value_names, mcfg.input_to_use_for_seq_len);
    assert(rc.first);  // TODO
    const auto& input_ort_value = in_token_ort_values[rc.second];
    int input_seq_len = input_ort_value.GetTensorTypeAndShapeInfo().GetShape()[mcfg.seq_len_dim_index_in_input];

    // get past seq len
    // assume past_seq_len is same for all states
    int past_seq_len = run_state.output_val_map.at(mcfg.present_output_names[0])
                           .GetTensorTypeAndShapeInfo()
                           .GetShape()[mcfg.seq_len_dim_index_in_state];
    // std::cout << "input_seq_len: " << input_seq_len << "\n";
    // std::cout << "past_seq_len: " << past_seq_len << "\n";
    // new seq len for state output = seq len of input_ids + past_seq_len
    int new_seq_len = input_seq_len + past_seq_len;

    auto& ort_sess = session_state.session;

    // populate shape for state outputs
    // assume same shape for all outputs
    auto rc2 = Contains(mcfg.output_names, mcfg.present_output_names[0]);
    assert(rc2.first);
    auto out_idx = rc2.second;
    auto past_present_state_shape = GetShape(ort_sess, out_idx, false /*output*/);
    past_present_state_shape[mcfg.batch_dim_index_in_state] = exec_frame.batch_size;
    past_present_state_shape[mcfg.seq_len_dim_index_in_state] = new_seq_len;

    // assume types are same for all states
    auto past_present_type = ort_sess.GetOutputTypeInfo(out_idx).GetTensorTypeAndShapeInfo().GetElementType();

    for (const auto& oname : mcfg.output_names) {
      auto rc = Contains(mcfg.present_output_names, oname);
      if (rc.first) {
        auto& mem_allocation = token.step_id % 2 == 0  // even: use buffer1 for input and buffer2 for output
                                   ? run_state.present_past_prealloc_buffer_2_vec[rc.second]
                                   : run_state.present_past_prealloc_buffer_1_vec[rc.second];
        // std::cout << "mem allocation size: " << mem_allocation.size() << "\n";
        auto output_ort_val = Ort::Value::CreateTensor(
            session_state.cuda_mem_info, mem_allocation.get(), mem_allocation.size(),
            past_present_state_shape.data(), past_present_state_shape.size(), past_present_type);
        // std::cout << "step(" << token.step_id << ") / stage(" << exec_frame.stage_id << ")"
        //           << " created tensor for " << oname << "\n";
        CheckStatus(g_ort->BindOutput(io_binding, oname.c_str(), output_ort_val));
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
            const auto& ort_val = exec_frame.ort_resp.output_values[rc.second];
            CheckStatus(g_ort->BindOutput(io_binding, oname.c_str(), ort_val));
          }
        } else {  // inter stage outputs (e.g. hidden_states)
          // get shape of oname
          auto found = Contains(mcfg.output_names, oname);
          assert(found.first);
          auto inter_stage_output_shape = GetShape(ort_sess, found.second, false /*output*/);

          // replace batch_size and seq_len
          inter_stage_output_shape[mcfg.batch_dim_in_inter_stage_output] = exec_frame.batch_size;
          inter_stage_output_shape[mcfg.seq_len_dim_in_inter_stage_output] = input_seq_len;

          auto& mem_allocation = run_state.inter_stage_output_prealloc_buffer_map.at(oname);
          auto inter_stage_ort_val = Ort::Value::CreateTensor(
              session_state.cuda_mem_info, mem_allocation.get(), mem_allocation.size(),
              inter_stage_output_shape.data(), inter_stage_output_shape.size(), past_present_type);
          CheckStatus(g_ort->BindOutput(io_binding, oname.c_str(), inter_stage_ort_val));
        }
      }
    }

    // run
    // std::cout << "step(" << token.step_id << ") / stage(" << exec_frame.stage_id << ")"
    //           << " just before run\n";
    {
      // std::string run_timer_str = "Run: " + str;
      // Timer t2(run_timer_str.c_str());
      ort_sess.Run({}, io_binding_obj);
    }
    // std::cout << "step(" << token.step_id << ") / stage(" << exec_frame.stage_id << ")"
    //           << " Done with run\n";
    // now populate token and save state from this run
    auto vec_out_vals = io_binding_obj.GetOutputValues();
    for (int i = 0, end = mcfg.output_names.size(); i < end; ++i) {
      const auto& oname = mcfg.output_names[i];

      // Assume that the same output name is not present in both the state that needs to be kept
      // and that needs to be passed on to the next layer.
      auto is_loop_back_state_output = Contains(mcfg.present_output_names, oname);
      assert(!(is_loop_back_state_output.first && mcfg.inter_stage_output_input_map.count(oname)));

      // if this output is present in present_output_names, store it in model_run_state_vec
      // because we don't want to store all outputs
      if (is_loop_back_state_output.first) {
        // std::cout << "step(" << token.step_id << ") / stage(" << exec_frame.stage_id << ")"
        //           << " saving state " << oname << "\n";
        assert(vec_out_vals[i].GetTensorData<Ort::Float16_t>());
        run_state.output_val_map[oname] = OrtValueHandle(vec_out_vals[i].release());
        continue;
      }

      // only pass those outputs to the next layer for which there is a config in the ensemble
      // other outputs are states to be used in the next run
      if (mcfg.inter_stage_output_input_map.count(oname)) {
        // std::cout << "Copying output req_id(" << token.req_id << ")/step(" << token.step_id << ")/stage(" << exec_frame.stage_id << ") "
        // << mcfg.inter_stage_output_input_map.at(oname) << "\n";
        out_token_ptr->ort_value_names.push_back(mcfg.inter_stage_output_input_map.at(oname));  // input_hidden_states
        assert(vec_out_vals[i].GetTensorData<Ort::Float16_t>());
        // if (oname == "hidden_states") {
        //   auto* data = vec_out_vals[i].GetTensorData<Ort::Float16_t>();
        //   auto typeshape = vec_out_vals[i].GetTensorTypeAndShapeInfo();
        //   auto slen = typeshape.GetShape()[1];
        //   // std::cout << "slen: " << slen << " bsize: " << typeshape.GetShape()[0] << "\n";
        //   std::vector<Ort::Float16_t> v(exec_frame.batch_size * slen * 4096);
        //   cudaMemcpy(v.data(), data, sizeof(Ort::Float16_t) * exec_frame.batch_size * slen * 4096, cudaMemcpyDefault);
        //   for (int b = 0; b < exec_frame.batch_size * 94 * 4096; b += 94 * 4096) {
        //     std::cout << "b = " << b << "\n";
        //     for (int k = 0; k < 10; ++k) {
        //       printf("output hidden state %f\n", HalfToFloat(v[b + k]));
        //     }
        //   }
        // }
        out_token_ptr->ort_values.push_back(OrtValueHandle(vec_out_vals[i].release()));
      }
    }

    // std::cout << "Done executing req_id(" << token.req_id << ")/step(" << token.step_id << ")/stage(" << exec_frame.stage_id << ")"
    //           << "\n";
    return out_token_ptr;
  };
};

// GetNewInputIdsFromLogits
static void GetNewInputIdsFromLogits(int batch_size,
                                     const OrtValueHandle& logits,
                                     const std::vector<int64_t>& logits_shape,
                                     std::vector<int64_t>& input_ids,
                                     std::vector<int64_t>& input_ids_shape) {
  // Timer t("GetNewInputIdsFromLogits");
  input_ids.clear();
  input_ids_shape.clear();

  input_ids.reserve(batch_size);
  input_ids_shape = std::vector<int64_t>{batch_size, 1};
  const auto* logits_data = logits.GetTensorData<Ort::Float16_t>();

  // for (int x = 0; x < 10; ++x) {
  //   printf("logits uint %hu, float val: %f\n", logits_data[x].value, HalfToFloat(logits_data[x]));
  // }

  int num_elems = logits_shape[0] * logits_shape[1] * logits_shape[2];
  int ltwo = logits_shape[1] * logits_shape[2];
  int skip = (logits_shape[1] - 1) * logits_shape[2];

  // now find the max per onnx batch
  for (int batch_id = 0; batch_id < num_elems; batch_id += ltwo) {  // TODO parallelize on batches
                                                                    // if (batch_id == 0)
    // std::cout << "batch_id " << batch_id << " first: " << logits_data[batch_id + skip] << "\n";
    auto tmp = std::max_element(logits_data + batch_id + skip,
                                logits_data + batch_id + skip + logits_shape[2],
                                [](const Ort::Float16_t& a, const Ort::Float16_t& b) { return HalfToFloat(a) < HalfToFloat(b); });
    int64_t max_idx = std::distance(logits_data + batch_id + skip, tmp);
    // if (batch_id == 0)
    // std::cout << "batch_id: " << batch_id << " next token: " << max_idx << "\n";
    input_ids.push_back(max_idx);
  }
}

void GetNewPosnIds(int batch_size, int orig_input_seq_len, int step_id, std::vector<int64_t>& posn_ids) {
  int new_posn_id = orig_input_seq_len + step_id - 1;
  // std::cout << "new posn id: " << new_posn_id << "\n";
  posn_ids.assign(batch_size, new_posn_id);
}

// TODO proper error handling
// TODO - replace all cout with LOG
// For simplicity even if one req in the batch fails, we consider the full batch to have failed.
OrtStatus* PipelineSession::Run(const std::vector<OrtReq>& req_list, std::vector<OrtResp>& resp_list, int num_steps) {
  std::unordered_map<ReqId, RequestExecutionFrame> req_frame_map;
  std::unordered_map<int, ReqId> req_idx_req_id_map;

  // First enqueue the first step and first stage processing for all the requests
  RequestExecutor rp;
  auto cpu_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  int num_reqs = req_list.size();

  for (int req_idx = 0; req_idx < num_reqs; ++req_idx) {
    ReqId req_id = CreateRequestId();
    req_idx_req_id_map[req_idx] = req_id;
    // std::cout << "creating req_id: " << req_id << "\n";
    auto& one_req = req_list[req_idx];
    auto& one_resp = resp_list[req_idx];

    // validate resp vector
    auto& ovalues = resp_list[req_idx].output_values;
    const auto& onames = resp_list[req_idx].output_names;
    assert(ovalues.size() == onames.size());

    // store batch size and input seq len to change position_ids for step > 0
    auto rc = Contains(one_req.input_names, pcfg.model_config_vec[0].input_to_use_for_seq_len);
    assert(rc.first);
    std::vector<OrtValueHandle> i_values;
    i_values.reserve(one_req.input_values.size());
    for (auto& v : one_req.input_values) {
      i_values.push_back(OrtValueHandle(v, false));  // don't own the OrtValues supplied by the user
    }
    const auto& shape = i_values[rc.second]
                            .GetTensorTypeAndShapeInfo()
                            .GetShape();
    int orig_seq_len = shape[pcfg.model_config_vec[0].seq_len_dim_index_in_input];
    int batch_size = shape[pcfg.model_config_vec[0].batch_dim_index_in_input];

    // create and store RequestExecutionFrame
    int stage_id = 0;
    RequestExecutionFrame tmp_exec_frame(*this, req_idx, req_id, batch_size, orig_seq_len, stage_id, one_resp);
    auto rcx = req_frame_map.emplace(req_id, std::move(tmp_exec_frame));
    assert(rcx.second);
  }

  // get logits index
  auto f = Contains(pcfg.model_config_vec[pcfg.num_stages - 1].output_names, pcfg.logits_name);
  assert(f.first);
  int logits_idx = f.second;

  int num_stages = pipeline_stages.size();
  for (int req_idx = 0; req_idx < num_reqs; ++req_idx) {
    int req_id = req_idx_req_id_map[req_idx];
    auto& exec_frame = req_frame_map.at(req_id);
    auto token_ptr = &exec_frame.token;
    auto& one_req = req_list[req_idx];
    auto& one_resp = resp_list[req_idx];
    token_ptr->req_id = exec_frame.req_id;

    for (int step_id = 0; step_id < num_steps; ++step_id) {
      token_ptr->step_id = step_id;

      for (int stage_id = 0; stage_id < num_stages; ++stage_id) {
        exec_frame.stage_id = stage_id;
        const auto& model_config = pcfg.model_config_vec[stage_id];
        auto& session_state = model_session_state_vec[stage_id];

        if (step_id == 0 && stage_id == 0) {
          std::vector<OrtValueHandle> i_values;
          i_values.reserve(one_req.input_values.size());
          for (auto& v : one_req.input_values) {
            i_values.push_back(OrtValueHandle(v, false));  // don't own the OrtValues supplied by the user
          }
          token_ptr->Init(exec_frame.req_id, step_id, one_req.input_names, std::move(i_values));
        }

        token_ptr = rp.ExecuteRequest(*token_ptr, model_config, session_state, exec_frame);
      }  // done with one step/all stages

      // modify input_ids and posn_ids here
      int batch_size = exec_frame.batch_size;

      // update input_ids
      // get index of 'logits' output
      auto rc = Contains(token_ptr->ort_value_names, pcfg.logits_name);
      if (!rc.first) {
        return g_ort->CreateStatus(ORT_FAIL, "Did not get logits in the output");
      }
      auto& input_ids = exec_frame.next_step_input_buffer_map[pcfg.input_ids_name].data;
      auto& input_ids_shape = exec_frame.next_step_input_buffer_map[pcfg.input_ids_name].shape;
      const auto& logits_ort_val = token_ptr->ort_values[rc.second];
      auto logits_shape = logits_ort_val.GetTensorTypeAndShapeInfo().GetShape();
      GetNewInputIdsFromLogits(batch_size, logits_ort_val, logits_shape, input_ids, input_ids_shape);

      // assume shape is same for both input_ids and position_ids
      auto input_ids_tensor = Ort::Value::CreateTensor<int64_t>(cpu_memory_info, input_ids.data(), input_ids.size(),
                                                                input_ids_shape.data(), input_ids_shape.size());  // TODO don't hardcode type

      // update position ids
      // assume shape of position ids is same as input_ids
      auto& posn_ids = exec_frame.next_step_input_buffer_map[pcfg.logits_name].data;
      GetNewPosnIds(batch_size, exec_frame.orig_input_seq_len, step_id + 1, posn_ids);

      auto posn_ids_tensor = Ort::Value::CreateTensor<int64_t>(cpu_memory_info, posn_ids.data(), posn_ids.size(),
                                                               input_ids_shape.data(), input_ids_shape.size());  // TODO don't hardcode type

      // clear and fill Token for the next step for this request
      if (step_id < num_steps - 1) {
        token_ptr->ort_values.clear();
        token_ptr->ort_value_names = {pcfg.input_ids_name, pcfg.position_ids_name};
        token_ptr->ort_values.push_back(OrtValueHandle(input_ids_tensor.release()));
        token_ptr->ort_values.push_back(OrtValueHandle(posn_ids_tensor.release()));
      }
    }  // done with one request

    // copy output of one request here
    int req_index = exec_frame.req_index;
    int resp_index = 0;
    for (const auto& oname : resp_list[req_index].output_names) {
      auto ex = Contains(token_ptr->ort_value_names, oname);
      if (ex.first) {
        resp_list[req_index].output_values[resp_index] = token_ptr->ort_values[ex.second].release();
      } else {
        // case when the user requested output was not present in the final output
        std::ostringstream ostr;
        ostr << "Error: Output " << oname << " is not produced by the final stage\n";
        return g_ort->CreateStatus(ORT_FAIL, ostr.str().c_str());
      }
      ++resp_index;
    }
    req_frame_map.erase(exec_frame.req_id);
  }

  return nullptr;
}

void PipelineSession::ParseEnsembleJsonFile(const std::string& ensemble_json_file, PipelineConfig& pcfg) {
  std::ifstream ifs(ensemble_json_file);
  if (!ifs.good()) {
    throw std::runtime_error(std::string("Error reading file ") + ensemble_json_file);
  }

  auto j = json::parse(ifs, nullptr, true);

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
    cfg.seq_len_dim_in_inter_stage_output = m["seq_len_dim_in_inter_stage_output"];
    cfg.batch_dim_in_inter_stage_output = m["batch_dim_in_inter_stage_output"];
    cfg.device_id = m["device_id"];

    const char* key = "inter_stage_output_input_map";
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

PipelineSession::PipelineSession(const std::string& ensemble_json_file, Ort::Env& env) {
  ParseEnsembleJsonFile(ensemble_json_file, pcfg);
  auto rc = Validate(pcfg);
  assert(rc);
  Init(pcfg, env);
}

PipelineSession::PipelineSession(const PipelineConfig& ens0, Ort::Env& env) : pcfg(ens0) {
  auto rc = Validate(pcfg);
  assert(rc);
  Init(pcfg, env);
}

void PipelineSession::Init(PipelineConfig& pcfg, Ort::Env& env) {
  Ort::AllocatorWithDefaultOptions ort_allocator;
  pipeline_stages.reserve(pcfg.model_config_vec.size());

  for (auto& mcfg : pcfg.model_config_vec) {
    Ort::SessionOptions session_options;
    session_options.DisablePerSessionThreads();
    CheckStatus(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, mcfg.device_id));
    Ort::Session session{nullptr};
    {
      std::string session_time_msg = mcfg.model_name;
      session_time_msg.append(" session creation");
      Timer t(session_time_msg.c_str());
      Ort::Session sess_tmp(env, mcfg.model_file_path.c_str(), session_options);
      session = std::move(sess_tmp);
    }

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

    // create session state
    Ort::MemoryInfo cuda_mem_info("Cuda", OrtDeviceAllocator, mcfg.device_id, OrtMemTypeDefault);
    SessionState sess_state{std::move(session), std::move(cuda_mem_info)};
    model_session_state_vec.push_back(std::move(sess_state));

    // create stages
    pipeline_stages.push_back(std::make_unique<PipelineStage>(mcfg.device_id, 1 /*thread pool size per stage*/));
  }
}
}  // namespace onnxruntime

int main(int argc, char* argv[]) {
  using namespace onnxruntime;
  OrtThreadingOptions* t_options = nullptr;
  CheckStatus(g_ort->CreateThreadingOptions(&t_options));
  Ort::Env env(t_options, ORT_LOGGING_LEVEL_WARNING, "test");

  cxxopts::Options options("test", "A brief description");
  // clang-format off
  options.add_options()
  ("f,config_file", "config file", cxxopts::value<std::string>()->default_value("/bert_ort/pranav/turing_model_ensemble.json"))
  ("b,batch_size", "batch_size", cxxopts::value<int>()->default_value("1"))
  ("s,num_steps", "num_steps", cxxopts::value<int>()->default_value("1"))
  ("n,num_reqs", "num_requests", cxxopts::value<int>()->default_value("1"))
  ("r,num_iters", "num_iters", cxxopts::value<int>()->default_value("1"))
  ("h,help", "Print usage");
  // clang-format on

  auto result = options.parse(argc, argv);

  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    exit(0);
  }

  // read ensemble file name
  std::string ensemble_file_name = result["config_file"].as<std::string>();
  std::cout << "Using ensemble file: " << ensemble_file_name << "\n";

  int num_steps = result["num_steps"].as<int>();
  std::cout << "Using num_steps = " << num_steps << "\n";

  int max_num_reqs = result["num_reqs"].as<int>();
  std::cout << "Using max_num_reqs = " << max_num_reqs << "\n";

  int num_iters = result["num_iters"].as<int>();
  std::cout << "Using num_iters = " << num_iters << "\n";

  int batch_size = result["batch_size"].as<int>();
  std::cout << "Using batch_size = " << batch_size << "\n";

  // prepare inputs
  const int seq_len = 94;
  size_t input_tensor_size = batch_size * seq_len;
  std::vector<int64_t> input_node_dims{batch_size, seq_len};
  std::vector<std::string> input_node_names{"input_ids",
                                            "position_ids"};
  std::vector<int64_t> input_ids;
  input_ids.reserve(input_tensor_size);
  std::vector<int64_t> input_ids_tmp{50264, 5211, 345, 760, 546, 326, 30, 5211, 345, 760,
                                     546, 326, 30, 5211, 345, 760, 546, 326, 30, 5211,
                                     345, 760, 546, 326, 30, 5211, 345, 760, 546, 326,
                                     30, 5211, 345, 760, 546, 326, 30, 50265, 19693, 19693,
                                     19693, 19693, 19693, 19693, 50264, 5211, 345, 760, 546, 39849,
                                     312, 12, 1129, 5211, 345, 760, 546, 39849, 312, 12,
                                     1129, 5211, 345, 760, 546, 39849, 312, 12, 1129, 5211,
                                     345, 760, 546, 39849, 312, 12, 1129, 5211, 345, 760,
                                     546, 39849, 312, 12, 1129, 5211, 345, 760, 546, 39849,
                                     312, 12, 1129, 50258};
  for (int i = 0; i < batch_size; ++i) {
    input_ids.insert(input_ids.end(), input_ids_tmp.begin(), input_ids_tmp.end());
  }
  std::vector<int64_t> posn_ids_tmp(seq_len);
  std::iota(std::begin(posn_ids_tmp), std::end(posn_ids_tmp), 0);
  std::vector<int64_t> posn_ids;
  for (int i = 0; i < batch_size; ++i) {
    posn_ids.insert(posn_ids.end(), posn_ids_tmp.begin(), posn_ids_tmp.end());
  }

  std::vector<std::string> output_node_names = {"logits"};
  auto cpu_mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  std::vector<OrtReq> req_list;
  std::vector<OrtValueHandle> input_val_deleters;
  for (int i = 0; i < max_num_reqs; ++i) {
    std::vector<OrtValue*> ort_inputs;
    auto input_ids_tensor = Ort::Value::CreateTensor<int64_t>(cpu_mem_info, input_ids.data(), input_tensor_size,
                                                              input_node_dims.data(), input_node_dims.size());
    auto posn_ids_tensor = Ort::Value::CreateTensor<int64_t>(cpu_mem_info, posn_ids.data(), input_tensor_size,
                                                             input_node_dims.data(), input_node_dims.size());
    auto* val = input_ids_tensor.release();
    input_val_deleters.push_back(OrtValueHandle(val));
    ort_inputs.push_back(val);

    val = posn_ids_tensor.release();
    input_val_deleters.push_back(OrtValueHandle(val));
    ort_inputs.push_back(val);

    OrtReq one_req{input_node_names, std::move(ort_inputs)};
    req_list.push_back(std::move(one_req));
  }

  std::vector<OrtResp> resp_list;
  std::vector<std::string> output_names{"logits"};
  for (int i = 0; i < max_num_reqs; ++i) {
    OrtResp one_resp;
    for (const auto& oname : output_names) {
      one_resp.output_names.push_back(oname);
      one_resp.output_values.push_back(nullptr);
      one_resp.output_meminfo.push_back(cpu_mem_info);
    }
    resp_list.push_back(std::move(one_resp));
  }

  // setup the pipeline session
  std::unique_ptr<PipelineSession> pipeline_session_ptr = nullptr;
  {
    Timer t("Creating PipelineSession");
    pipeline_session_ptr = std::make_unique<PipelineSession>(ensemble_file_name, env);
  }
  PipelineSession& pipeline_session = *pipeline_session_ptr;

  // Run the pipeline
  OrtStatus* status;
  {
    Timer t("PipelineSession::Run Warmup");
    status = pipeline_session.Run(req_list, resp_list, num_steps);
    std::unique_ptr<OrtStatus, decltype(g_ort->ReleaseStatus)> status_deleter(status, g_ort->ReleaseStatus);
    if (status) {
      std::cout << "Execution failed with error " << g_ort->GetErrorMessage(status) << "\n";
      return -1;
    }
  }

  int64_t total_time_us = 0;
  for (int i = 0; i < num_iters; ++i) {
    // Timer t("PipelineSession::Run");
    auto start = std::chrono::high_resolution_clock::now();
    status = pipeline_session.Run(req_list, resp_list, num_steps);
    std::unique_ptr<OrtStatus, decltype(g_ort->ReleaseStatus)> status_deleter(status, g_ort->ReleaseStatus);
    if (status) {
      std::cout << "Execution failed with error " << g_ort->GetErrorMessage(status) << "\n";
      return -1;
    }
    auto end = std::chrono::high_resolution_clock::now();
    total_time_us += (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
  }
  std::cout << "Average time taken for " << num_iters << " iterations: " << (float)total_time_us / num_iters << " microseconds\n";

  // for num_steps = 10 and batch_size = 1
  std::vector<Ort::Float16_t> valid_results{16464, 15168, 48600, 46534, 48945, 49080};
  // valid new input ids generated after call to GetNewInputIds
  // tensor([[35528, 35528, 20174, 14430, 42092, 36466,  1825,  1825, 35528, 42760]]
  for (int idx = 0; idx < resp_list.size(); ++idx) {
    assert(resp_list[idx].output_names[0] == pipeline_session.pcfg.logits_name);
    auto retval = OrtValueHandle(resp_list[idx].output_values[0]);
    assert(retval.GetTensorData<Ort::Float16_t>());
    assert(resp_list[0].output_values.size() == resp_list[idx].output_names.size());

    // print output
    auto* data_ptr = retval.GetTensorData<Ort::Float16_t>();
    auto type_shape = retval.GetTensorTypeAndShapeInfo();
    auto o_shape = type_shape.GetShape();
    auto num_elems = type_shape.GetElementCount();
    // std::cout << "num_elems: " << num_elems << "\n";
    for (int bsz = 0; bsz < num_elems; bsz += (o_shape[1] * o_shape[2])) {
      std::vector<Ort::Float16_t> v;
      for (int j = 0; j < o_shape[2]; j += 10000) {
        // std::cout << "bsz: " << bsz << " result: " << data_ptr[bsz + j] << "\n";
        v.push_back(data_ptr[bsz + j]);
      }
      if (num_steps == 10 && batch_size == 1) {
        // std::cout << "validating output for batch " << bsz << "\n";
        assert(v == valid_results);
      }
    }
  }

  printf("Done!\n");
  return 0;
}

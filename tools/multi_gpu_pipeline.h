#include "bits/stdc++.h"
#include <assert.h>
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#include "cuda_provider_factory.h"
// #include "tbb/pipeline.h"
#include "json.hpp"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "task_thread_pool.h"
#include "response_queue.h"

using ReqId = int64_t;

struct PipelineConfig {
  struct ModelConfig {
    std::string model_name;
    std::string model_file_path;
    std::vector<std::string> input_names;                           // same order as model
    std::vector<std::string> output_names;                          // same order as model
    std::unordered_map<std::string, std::string> output_input_map;  // maps output of this step to input of the next step in the pipeline
    // state_input_names and state_output_names should have 1-1 correspondence
    std::vector<std::string> state_input_names;   // names of inputs whose values come from the previous output
    std::vector<std::string> state_output_names;  // names of outputs that feed the next inputs
    int device_id;
    int batch_dim_index_in_state;
    int batch_dim_index_in_input;
    int seq_len_dim_index_in_state;
    std::string input_to_use_for_seq_len;
    int seq_len_dim_index_in_input;
  };

  int max_seq_len;
  int num_stages;  // = model_config_vec.size()
  std::string input_ids_name;
  std::string position_ids_name;
  std::string logits_name;
  std::vector<ModelConfig> model_config_vec;
  std::unordered_map<std::string, int> model_idx_map;  // maps model name to index in models vector
};

struct OrtReq {
  std::vector<std::string> input_names;
  std::vector<Ort::Value> input_values;
};

struct OrtResp {
  std::vector<std::string> output_names;
  std::vector<Ort::Value> output_values;
  std::vector<OrtMemoryInfo*> output_meminfo;  // specify location of outputs or null for preallocated
};

struct PipelineSession;
struct RequestExecutionFrame {
  RequestExecutionFrame(PipelineSession& psess,
                        int req_idx0,
                        ReqId req_id0,
                        int batch_size0,
                        int orig_input_seq_len0,
                        int stage_id0,
                        OrtResp& ort_resp);

  struct RunState {
    // needs to be stored per model since it's associated with a session
    std::unique_ptr<Ort::IoBinding> io_binding;
    // needs to be stored per model since it's associated with a session
    // storing it here as opposed to PipelineConfig since it may not be thread-safe when multiple requests
    // are getting executed in parallel
    std::unique_ptr<Ort::Allocator> cuda_allocator;
    std::unordered_map<std::string, Ort::Value> output_val_map;  // output generated after running a stage
    // pre-allocated on cuda; order should be same as ModelConfig::state_output_names/state_input_names
    std::vector<Ort::MemoryAllocation> state_buffer_1_vec;
    std::vector<Ort::MemoryAllocation> state_buffer_2_vec;
  };

  const int req_index;
  const int64_t req_id;  // request to be executed
  const int batch_size;
  const int orig_input_seq_len;
  int stage_id;  // stage to be executed; stage_id can be used as an index into model_run_state_vec
  std::vector<RunState> model_run_state_vec;
  OrtResp& ort_resp;
};

struct Token {
  void Clear() {
    ort_value_names.clear();
    ort_values.clear();
    error_msg.clear();
    req_id = -1;
    step_id = -1;
  }

  int64_t req_id;
  int step_id;
  std::vector<std::string> ort_value_names;
  std::vector<Ort::Value> ort_values;
  std::string error_msg;
};

struct PipelineSession {
  // TODO return error status code, decide how to do error handling in stages
  // TODO stop execution when an error is detected
  // TODO - adjust output shape for states
  // TODO - change position_ids for step > 0
  OrtStatus* Run(std::vector<OrtReq>& req_vec, std::vector<OrtResp>& resp_vec, int max_steps);
  void ParseEnsembleJsonFile(const std::string& ensemble_json_file, PipelineConfig& ens);
  PipelineSession(const std::string& ensemble_json_file, Ort::Env& env);
  PipelineSession(const PipelineConfig& ens, Ort::Env& env);
  void Init(PipelineConfig& ens0, Ort::Env& env);
  bool Validate(const PipelineConfig& ens);

  struct SessionState {
    Ort::Session session;
    // needs to be stored per model since it's associated with device id
    Ort::MemoryInfo cuda_mem_info;
  };

  // TODO later we might store this keyed by device id if we allow the same model to have sessions on multiple GPUs for
  // better load balancing
  std::vector<SessionState> model_session_state_vec;
  PipelineConfig pcfg;
  TaskThreadPool tp;
};

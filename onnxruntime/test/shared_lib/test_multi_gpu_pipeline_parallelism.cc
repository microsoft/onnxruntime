// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <atomic>
#include <mutex>
#include <algorithm>

#include <gtest/gtest.h>

#include "core/common/common.h"
#include "core/common/make_unique.h"
#include "core/graph/constants.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "providers.h"
#include "test_allocator.h"
#include "test_fixture.h"
#include "utils.h"
#include "core/session/pipeline_parallelism/multi_gpu_pipeline.h"

extern std::unique_ptr<Ort::Env> ort_env;

// TODO add more tests

TEST(CApiTest, positive_test) {
  const OrtApi* g_ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  const OrtExperimentalApi* g_ort_exp_api = OrtGetApiBase()->GetExperimentalApi();

  int num_steps = 10;
  int num_reqs = 4;
  int batch_size = 1;

  // prepare inputs
  const int seq_len = 94;
  size_t input_tensor_size = batch_size * seq_len;
  std::vector<int64_t> input_node_dims{batch_size, seq_len};
  std::vector<const char*> input_node_names{"input_ids",
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

  OrtRequestBatch* req_batch = nullptr;
  std::unique_ptr<OrtStatus, decltype(OrtApi::ReleaseStatus)> st_ptr(nullptr, g_ort_api->ReleaseStatus);
  st_ptr.reset(g_ort_exp_api->CreateOrtRequestBatch(&req_batch));
  std::unique_ptr<OrtRequestBatch, decltype(g_ort_exp_api->ReleaseRequestBatch)> d(req_batch,
                                                                                   g_ort_exp_api->ReleaseRequestBatch);
  ASSERT_EQ(st_ptr.get(), nullptr);

  using namespace onnxruntime::experimental;
  std::vector<OrtValueHandle> ort_val_deleters;

  std::vector<std::vector<OrtValue*>> ort_inputs(num_reqs);
  auto cpu_mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  for (int i = 0; i < num_reqs; ++i) {
    auto input_ids_tensor = Ort::Value::CreateTensor<int64_t>(cpu_mem_info, input_ids.data(), input_tensor_size,
                                                              input_node_dims.data(), input_node_dims.size());
    auto posn_ids_tensor = Ort::Value::CreateTensor<int64_t>(cpu_mem_info, posn_ids.data(), input_tensor_size,
                                                             input_node_dims.data(), input_node_dims.size());
    auto* val = input_ids_tensor.release();
    ort_val_deleters.push_back(OrtValueHandle(val));
    ort_inputs[i].push_back(val);

    val = posn_ids_tensor.release();
    ort_val_deleters.push_back(OrtValueHandle(val));
    ort_inputs[i].push_back(val);

    ASSERT_EQ(g_ort_exp_api->AddRequestToBatch(req_batch, ort_inputs[i].size(), input_node_names.data(), ort_inputs[i].data()),
              nullptr);
  }

  OrtResponseBatch* resp_batch = nullptr;
  st_ptr.reset(g_ort_exp_api->CreateOrtResponseBatch(&resp_batch));
  std::unique_ptr<OrtResponseBatch, decltype(g_ort_exp_api->ReleaseResponseBatch)> d2(resp_batch,
                                                                                      g_ort_exp_api->ReleaseResponseBatch);
  ASSERT_EQ(st_ptr.get(), nullptr);

  struct OneResp {
    std::vector<const char*> output_names;
    std::vector<OrtValue*> ort_vals;
    std::vector<const OrtMemoryInfo*> ort_mem_infos;
  };
  std::vector<OneResp> ort_outputs(num_reqs);
  for (int i = 0; i < num_reqs; ++i) {
    ort_outputs[i].output_names.push_back("logits");
    ort_outputs[i].ort_vals.push_back(nullptr);
    ort_outputs[i].ort_mem_infos.push_back(cpu_mem_info);
    ASSERT_EQ(g_ort_exp_api->AddResponseToBatch(resp_batch, ort_outputs[i].output_names.size(),
                                                ort_outputs[i].output_names.data(),
                                                ort_outputs[i].ort_vals.data(),
                                                ort_outputs[i].ort_mem_infos.data()),
              nullptr);
  }

  // create session
  OrtPipelineSession* pipeline_session_ptr = nullptr;
  st_ptr.reset(g_ort_exp_api->CreatePipelineSession(*ort_env.get(),
                                                    "testdata/gpt2_model_ensemble.json",
                                                    &pipeline_session_ptr));
  std::unique_ptr<OrtPipelineSession, decltype(g_ort_exp_api->ReleasePipelineSession)> sdel(pipeline_session_ptr,
                                                                                            g_ort_exp_api->ReleasePipelineSession);
  ASSERT_EQ(st_ptr.get(), nullptr) << g_ort_api->GetErrorMessage(st_ptr.get());

  // run session
  st_ptr.reset(g_ort_exp_api->Run(pipeline_session_ptr, req_batch, resp_batch, num_steps));
  ASSERT_EQ(st_ptr.get(), nullptr) << g_ort_api->GetErrorMessage(st_ptr.get());

  // validate output
  // for num_steps = 10 and batch_size = 1
  std::vector<Ort::Float16_t> valid_results{16464, 15168, 48600, 46534, 48945, 49080};
  // valid new input ids generated after call to GetNewInputIds
  // tensor([[35528, 35528, 20174, 14430, 42092, 36466,  1825,  1825, 35528, 42760]]

  Ort::AllocatorWithDefaultOptions default_allocator;

  for (int req_num = 0; req_num < num_reqs; ++req_num) {
    OrtValue** outputs = nullptr;
    size_t output_count = 0;
    st_ptr.reset(g_ort_exp_api->GetOutputValues(resp_batch, req_num, default_allocator, &outputs, &output_count));
    std::unique_ptr<OrtValue*, std::function<void(OrtValue**)>> array_deleter(outputs, [&default_allocator](OrtValue** oval_array) {
      if (!oval_array) {
        return;
      }
      default_allocator.Free(oval_array);
    });
    ASSERT_EQ(st_ptr.get(), nullptr) << g_ort_api->GetErrorMessage(st_ptr.get());

    for (size_t idx = 0; idx < output_count; ++idx) {
      // assert(resp_list[idx].output_names[0] == pipeline_session.model_ensemble_cfg.logits_name);
      auto* val = outputs[idx];
      ort_val_deleters.push_back(OrtValueHandle(val));
      auto& retval = ort_val_deleters.back();
      ASSERT_NE(retval.GetTensorData<Ort::Float16_t>(), nullptr);
      //   assert(resp_list[0].output_values.size() == resp_list[idx].output_names.size());

      // print output
      auto* data_ptr = retval.GetTensorData<Ort::Float16_t>();
      auto type_shape = retval.GetTensorTypeAndShapeInfo();
      auto o_shape = type_shape.GetShape();
      auto num_elems = type_shape.GetElementCount();
      // std::cout << "num_elems: " << num_elems << "\n";
      for (size_t bsz = 0; bsz < num_elems; bsz += (o_shape[1] * o_shape[2])) {
        std::vector<Ort::Float16_t> v;
        for (int j = 0; j < o_shape[2]; j += 10000) {
          // std::cout << "bsz: " << bsz << " result: " << data_ptr[bsz + j] << "\n";
          v.push_back(data_ptr[bsz + j]);
        }
        if (num_steps == 10 && batch_size == 1) {
          // std::cout << "validating output for batch " << bsz << "\n";
          ASSERT_EQ(v, valid_results);
          // std::cout << "validation successful\n";
        }
      }
    }
  }

  g_ort_exp_api->ClearRequestBatch(req_batch);
  g_ort_exp_api->ClearResponseBatch(resp_batch);
}
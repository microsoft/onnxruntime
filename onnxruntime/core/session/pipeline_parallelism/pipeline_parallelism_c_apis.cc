// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/error_code_helper.h"
#include "core/session/ort_apis.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/ort_experimental_apis.h"
#include "multi_gpu_pipeline.h"
#include "core/framework/ml_value.h"

#define DEFINE_RELEASE_ORT_OBJECT_FUNCTION_EXPERIMENTAL(INPUT_TYPE, REAL_TYPE)                      \
  ORT_API(void, OrtExperimentalApis::Release##INPUT_TYPE, _Frees_ptr_opt_ Ort##INPUT_TYPE* value) { \
    delete reinterpret_cast<REAL_TYPE*>(value);                                                     \
  }

DEFINE_RELEASE_ORT_OBJECT_FUNCTION_EXPERIMENTAL(PipelineSession, ::onnxruntime::experimental::PipelineSession)
DEFINE_RELEASE_ORT_OBJECT_FUNCTION_EXPERIMENTAL(RequestBatch, std::vector<::onnxruntime::experimental::OrtReq>)
DEFINE_RELEASE_ORT_OBJECT_FUNCTION_EXPERIMENTAL(ResponseBatch, std::vector<::onnxruntime::experimental::OrtResp>)

namespace OrtExperimentalApis {
ORT_API_STATUS_IMPL(CreatePipelineSession, _In_ const OrtEnv* env, _In_ const char* ensemble_config_file_path,
                    _Outptr_ OrtPipelineSession** out) {
#ifndef USE_CUDA
  ORT_UNUSED_PARAMETER(env);
  ORT_UNUSED_PARAMETER(ensemble_config_file_path);
  ORT_UNUSED_PARAMETER(out);
  return CreateStatus(ORT_FAIL, "CUDA execution provider is not enabled.");
#endif

  API_IMPL_BEGIN
  *out = nullptr;

  auto sess = onnxruntime::make_unique<onnxruntime::experimental::PipelineSession>(ensemble_config_file_path, *env);
  *out = reinterpret_cast<OrtPipelineSession*>(sess.release());
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(Run, _Inout_ OrtPipelineSession* sess, _In_ const OrtRequestBatch* req_batch,
                    _Out_ OrtResponseBatch* resp_batch, int num_steps) {
#ifndef USE_CUDA
  ORT_UNUSED_PARAMETER(sess);
  ORT_UNUSED_PARAMETER(req_batch);
  ORT_UNUSED_PARAMETER(resp_batch);
  ORT_UNUSED_PARAMETER(num_steps);
  return CreateStatus(ORT_FAIL, "CUDA execution provider is not enabled.");
#endif

  API_IMPL_BEGIN
  auto* pipeline_session = reinterpret_cast<onnxruntime::experimental::PipelineSession*>(sess);
  const auto* req_vec = reinterpret_cast<const std::vector<onnxruntime::experimental::OrtReq>*>(req_batch);
  auto* resp_vec = reinterpret_cast<std::vector<onnxruntime::experimental::OrtResp>*>(resp_batch);
  auto* status = pipeline_session->Run(*req_vec, *resp_vec, num_steps);
  return status;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(CreateOrtRequestBatch, _Outptr_ OrtRequestBatch** req_batch) {
#ifndef USE_CUDA
  ORT_UNUSED_PARAMETER(req_batch);
  return CreateStatus(ORT_FAIL, "CUDA execution provider is not enabled.");
#endif

  API_IMPL_BEGIN
  auto batch = onnxruntime::make_unique<std::vector<onnxruntime::experimental::OrtReq>>();
  *req_batch = reinterpret_cast<OrtRequestBatch*>(batch.release());
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(CreateOrtResponseBatch, _Outptr_ OrtResponseBatch** resp_batch) {
#ifndef USE_CUDA
  ORT_UNUSED_PARAMETER(resp_batch);
  return CreateStatus(ORT_FAIL, "CUDA execution provider is not enabled.");
#endif

  API_IMPL_BEGIN
  auto batch = onnxruntime::make_unique<std::vector<onnxruntime::experimental::OrtResp>>();
  *resp_batch = reinterpret_cast<OrtResponseBatch*>(batch.release());
  return nullptr;
  API_IMPL_END
}

ORT_API(void, ClearRequestBatch, _Inout_ OrtRequestBatch* req_batch) {
  auto* req_vec = reinterpret_cast<std::vector<onnxruntime::experimental::OrtReq>*>(req_batch);
  req_vec->clear();
}

ORT_API(void, ClearResponseBatch, _Inout_ OrtResponseBatch* resp_batch) {
  auto* resp_vec = reinterpret_cast<std::vector<onnxruntime::experimental::OrtResp>*>(resp_batch);
  resp_vec->clear();
}

ORT_API_STATUS_IMPL(AddRequestToBatch, _Inout_ OrtRequestBatch* req_batch, size_t input_len, _In_reads_(input_len) const char* const* input_names,
                    _In_reads_(input_len) const OrtValue* const* input) {
#ifndef USE_CUDA
  ORT_UNUSED_PARAMETER(req_batch);
  ORT_UNUSED_PARAMETER(input_len);
  ORT_UNUSED_PARAMETER(input_names);
  ORT_UNUSED_PARAMETER(input);
  return CreateStatus(ORT_FAIL, "CUDA execution provider is not enabled.");
#endif

  API_IMPL_BEGIN
  auto* req_vec = reinterpret_cast<std::vector<onnxruntime::experimental::OrtReq>*>(req_batch);
  onnxruntime::experimental::OrtReq ort_req;
  for (size_t i = 0; i < input_len; ++i) {
    ort_req.input_names.push_back(input_names[i]);
    ort_req.input_values.push_back(const_cast<OrtValue*>(input[i]));
  }
  req_vec->push_back(std::move(ort_req));
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(AddResponseToBatch, _Inout_ OrtResponseBatch* resp_batch, size_t output_names_len,
                    _In_reads_(output_names_len) const char* const* output_names,
                    _In_reads_(output_names_len) OrtValue** output, _In_reads_(output_names_len) const OrtMemoryInfo** info) {
#ifndef USE_CUDA
  ORT_UNUSED_PARAMETER(resp_batch);
  ORT_UNUSED_PARAMETER(output_names_len);
  ORT_UNUSED_PARAMETER(output_names);
  ORT_UNUSED_PARAMETER(output);
  ORT_UNUSED_PARAMETER(info);
  return CreateStatus(ORT_FAIL, "CUDA execution provider is not enabled.");
#endif

  API_IMPL_BEGIN
  auto* resp_vec = reinterpret_cast<std::vector<onnxruntime::experimental::OrtResp>*>(resp_batch);
  onnxruntime::experimental::OrtResp ort_resp;
  for (size_t i = 0; i < output_names_len; ++i) {
    ort_resp.output_names.push_back(output_names[i]);

    // both output[i] and info[i] cannot be null
    if (output[i] == nullptr && info[i] == nullptr) {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Both OrtValue and OrtMemoryInfo cannot be nullptr");
    }
    ort_resp.output_values.push_back(output[i]);
    ort_resp.output_meminfo.push_back(info[i]);
  }
  resp_vec->push_back(std::move(ort_resp));
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(GetOutputValues, _In_ const OrtResponseBatch* resp_batch, _In_ size_t batch_idx, _In_ OrtAllocator* allocator,
                    _Out_writes_all_(output_count) OrtValue*** output, _Out_ size_t* output_count) {
#ifndef USE_CUDA
  ORT_UNUSED_PARAMETER(resp_batch);
  ORT_UNUSED_PARAMETER(batch_idx);
  ORT_UNUSED_PARAMETER(allocator);
  ORT_UNUSED_PARAMETER(output);
  ORT_UNUSED_PARAMETER(output_count);
  return CreateStatus(ORT_FAIL, "CUDA execution provider is not enabled.");
#endif

  API_IMPL_BEGIN
  const auto* resp_vec_ptr = reinterpret_cast<const std::vector<onnxruntime::experimental::OrtResp>*>(resp_batch);
  const auto& resp_vec = *resp_vec_ptr;
  if (resp_vec.empty()) {
    *output = nullptr;
    *output_count = 0U;
    return nullptr;
  }

  if (batch_idx >= resp_vec.size()) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Invalid batch_idx supplied; it's bigger than the number of elements in the resp batch");
  }

  // Used to destroy and de-allocate on exception
  size_t created = 0;
  using IAllocatorDeleter = std::unique_ptr<OrtValue*, std::function<void(OrtValue**)>>;
  IAllocatorDeleter ortvalues_alloc(reinterpret_cast<OrtValue**>(
                                        allocator->Alloc(allocator,
                                                         resp_vec[batch_idx].output_values.size() * sizeof(OrtValue*))),
                                    [&created, allocator](OrtValue** buffer) {
                                      if (buffer) {
                                        while (created > 0) {
                                          auto p = buffer + --created;
                                          delete (*p);
                                        }
                                        allocator->Free(allocator, buffer);
                                      }
                                    });

  if (!ortvalues_alloc) {
    return OrtApis::CreateStatus(ORT_FAIL, "Output buffer allocation failed");
  }

  OrtValue** out_ptr = ortvalues_alloc.get();
  for (const auto* out_value : resp_vec[batch_idx].output_values) {
    *out_ptr = new OrtValue(*out_value);
    ++out_ptr;
    ++created;
  }

  assert(created == resp_vec[batch_idx].output_values.size());

  *output = ortvalues_alloc.release();
  *output_count = created;
  return nullptr;
  API_IMPL_END
}
}  // namespace OrtExperimentalApis
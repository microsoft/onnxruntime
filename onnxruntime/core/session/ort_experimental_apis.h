// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace OrtExperimentalApis {

ORT_API(const OrtExperimentalApi*, GetExperimentalApi);

ORT_API(void, ReleasePipelineSession, _Frees_ptr_opt_ OrtPipelineSession*);
ORT_API(void, ReleaseRequestBatch, _Frees_ptr_opt_ OrtRequestBatch*);
ORT_API(void, ReleaseResponseBatch, _Frees_ptr_opt_ OrtResponseBatch*);

ORT_API_STATUS_IMPL(CreatePipelineSession, _In_ const OrtEnv* env, _In_ const char* ensemble_config_file_path,
                    _Outptr_ OrtPipelineSession** out);
ORT_API_STATUS_IMPL(Run, _Inout_ OrtPipelineSession* sess, _In_ const OrtRequestBatch* req_batch, _Out_ OrtResponseBatch* resp_batch, int num_steps);
ORT_API_STATUS_IMPL(CreateOrtRequestBatch, _Outptr_ OrtRequestBatch** req_batch);
ORT_API_STATUS_IMPL(CreateOrtResponseBatch, _Outptr_ OrtResponseBatch** resp_batch);
ORT_API_STATUS_IMPL(AddRequestToBatch, _Inout_ OrtRequestBatch* req_batch, size_t input_len, _In_reads_(input_len) const char* const* input_names,
                    _In_reads_(input_len) const OrtValue* const* input);
ORT_API_STATUS_IMPL(AddResponseToBatch, _Inout_ OrtResponseBatch* resp_batch, size_t output_names_len,
                    _In_reads_(output_names_len) const char* const* output_names,
                    _In_reads_(output_names_len) OrtValue** output, _In_reads_(output_names_len) const OrtMemoryInfo** info);
ORT_API_STATUS_IMPL(GetOutputValues, _In_ const OrtResponseBatch* resp_batch, _In_ size_t batch_idx, _In_ OrtAllocator* allocator,
                    _Out_writes_all_(output_count) OrtValue*** output, _Out_ size_t* output_count);
ORT_API(void, ClearRequestBatch, _Inout_ OrtRequestBatch* req_batch);
ORT_API(void, ClearResponseBatch, _Inout_ OrtResponseBatch* resp_batch);
}  // namespace OrtExperimentalApis

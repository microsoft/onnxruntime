// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>
#include <memory>
#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <thread>
#include <mutex>
#include <iostream>

#include <openenclave/enclave.h>

// Using the C interface (instead of C++) as it makes error handling
// more convenient in the enclave interface functions.
#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "test/openenclave/session_enclave/shared/ortvalue_to_tensorproto_converter.h"
#include "session_t.h"  // generated from session.edl
#include "threading.h"

#include "test/onnx/tensorprotoutils.h"
#include "test/onnx/callback.h"
#include "test/onnx/mem_buffer.h"

using namespace onnxruntime::openenclave;

OE_SET_ENCLAVE_SGX(
    1,      /* ProductID */
    1,      /* SecurityVersion */
    true,   /* AllowDebug */
    609600, /* HeapPageCount */
    32768,  /* StackPageCount */
    8);     /* TCSCount */

const OrtApi* ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
OrtEnv* ort_env = nullptr;
OrtAllocator* ort_allocator = nullptr;
OrtSession* ort_session = nullptr;
OrtRunOptions* ort_run_options = nullptr;

std::vector<std::string> ort_input_names;
std::vector<std::string> ort_output_names;

#define ORT_RETURN_ON_ERROR(expr)                                      \
  do {                                                                 \
    OrtStatus* onnx_status = (expr);                                   \
    if (onnx_status != nullptr) {                                      \
      std::string ort_error_message = ort->GetErrorMessage(onnx_status); \
      OrtErrorCode error_code = ort->GetErrorCode(onnx_status);          \
      ort->ReleaseStatus(onnx_status);                                   \
      *error_msg = oe_host_strndup(ort_error_message.c_str(), 256);    \
      return error_code;                                               \
    }                                                                  \
  } while (0);

#define ORT_RETURN_ERROR(status, msg)                   \
  do {                                                  \
    *error_msg = oe_host_strndup(msg, strlen(msg) + 1); \
    return status;                                      \
  } while (0);

void CleanupGlobalState() {
  if (ort_session) {
    ort->ReleaseSession(ort_session);
    ort_session = nullptr;
  }
  if (ort_env) {
    ort->ReleaseEnv(ort_env);
    ort_env = nullptr;
  }
  if (ort_run_options) {
    ort->ReleaseRunOptions(ort_run_options);
    ort_run_options = nullptr;
  }
}

extern "C" int EnclaveCreateSession(char** error_msg, size_t* output_count,
                                    const char* model_buf, size_t model_buf_len,
                                    int logging_level,
                                    int enable_sequential_execution,
                                    int intra_op_num_threads,
                                    int inter_op_num_threads,
                                    uint32_t optimization_level) {
  if (ort_session) {
    ORT_RETURN_ERROR(ORT_FAIL, "Session already created");
  }

  InitializeOpenEnclavePThreads();

  ORT_RETURN_ON_ERROR(ort->GetAllocatorWithDefaultOptions(&ort_allocator));
  ORT_RETURN_ON_ERROR(ort->CreateEnv(static_cast<OrtLoggingLevel>(logging_level), "Default", &ort_env));

  OrtSessionOptions* ort_session_options = nullptr;
  ORT_RETURN_ON_ERROR(ort->CreateSessionOptions(&ort_session_options));
  ORT_RETURN_ON_ERROR(ort->CreateRunOptions(&ort_run_options));

  ORT_RETURN_ON_ERROR(ort->EnableCpuMemArena(ort_session_options));
  if (enable_sequential_execution) {
    ORT_RETURN_ON_ERROR(ort->SetSessionExecutionMode(ort_session_options, ORT_SEQUENTIAL));
  } else {
    ORT_RETURN_ON_ERROR(ort->SetSessionExecutionMode(ort_session_options, ORT_PARALLEL));
  }

  ORT_RETURN_ON_ERROR(ort->SetInterOpNumThreads(ort_session_options, inter_op_num_threads));
  ORT_RETURN_ON_ERROR(ort->SetIntraOpNumThreads(ort_session_options, intra_op_num_threads));
  ORT_RETURN_ON_ERROR(ort->SetSessionGraphOptimizationLevel(ort_session_options, static_cast<GraphOptimizationLevel>(optimization_level)));

  char* tmp = nullptr;

  ORT_RETURN_ON_ERROR(ort->CreateSessionFromArray(ort_env,
                                                  tmp != nullptr ? tmp : model_buf, model_buf_len, ort_session_options, &ort_session));
  ort->ReleaseSessionOptions(ort_session_options);

  if (tmp != nullptr) {
    free(tmp);
  }

  size_t input_count;
  ORT_RETURN_ON_ERROR(ort->SessionGetInputCount(ort_session, &input_count));
  ORT_RETURN_ON_ERROR(ort->SessionGetOutputCount(ort_session, output_count));

  ort_input_names.clear();
  for (size_t i = 0; i < input_count; i++) {
    char* input_name;
    ORT_RETURN_ON_ERROR(ort->SessionGetInputName(ort_session, i, ort_allocator, &input_name));
    ort_input_names.emplace_back(input_name);
    ORT_RETURN_ON_ERROR(ort->AllocatorFree(ort_allocator, input_name));
  }

  ort_output_names.clear();
  for (size_t i = 0; i < *output_count; i++) {
    char* output_name;
    ORT_RETURN_ON_ERROR(ort->SessionGetOutputName(ort_session, i, ort_allocator, &output_name));
    ort_output_names.emplace_back(output_name);
    ORT_RETURN_ON_ERROR(ort->AllocatorFree(ort_allocator, output_name));
  }

  return 0;
}

extern "C" int EnclaveRunInference(char** error_msg,
                                   char** input_bufs, size_t* input_sizes, size_t input_count,
                                   char** output_bufs, size_t* output_sizes, size_t output_count, size_t output_max_size) {
  if (input_count != ort_input_names.size()) {
    ORT_RETURN_ERROR(ORT_INVALID_ARGUMENT, "input count does not match model inputs");
  }
  // 0 = don't return outputs
  if (output_count != 0 && output_count != ort_output_names.size()) {
    ORT_RETURN_ERROR(ORT_INVALID_ARGUMENT, "output count does not match model outputs");
  }

  OrtValue* inputs[input_count];
  onnxruntime::test::OrtCallback deleters[input_count];
  void* buffers[input_count];
  for (size_t i = 0; i < input_count; i++) {
    size_t input_len = input_sizes[i];
    char* input_buf = input_bufs[i];

    onnx::TensorProto tensorproto{};
    tensorproto.ParseFromArray(input_buf, input_len);

    size_t tensor_memsize;
    auto status = onnxruntime::test::GetSizeInBytesFromTensorProto<0>(tensorproto, &tensor_memsize);
    if (!status.IsOK()) {
      std::string msg = status.ToString();
      ORT_RETURN_ERROR(ORT_FAIL, msg.c_str());
    }

    ORT_RETURN_ON_ERROR(ort->AllocatorAlloc(ort_allocator, tensor_memsize, &buffers[i]));

    Ort::Value input{nullptr};
    deleters[i].f = nullptr;
    const onnxruntime::test::MemBuffer mbuf(buffers[i], tensor_memsize, *ort_allocator->Info(ort_allocator));
    status = onnxruntime::test::TensorProtoToMLValue(tensorproto, mbuf, input, deleters[i]);
    if (!status.IsOK()) {
      std::string msg = status.ToString();
      ORT_RETURN_ERROR(ORT_FAIL, msg.c_str());
    }
    inputs[i] = input.release();
  }

  const char* input_names_c[ort_input_names.size()];
  const char* output_names_c[ort_output_names.size()];
  for (size_t i = 0; i < ort_input_names.size(); i++) {
    input_names_c[i] = ort_input_names[i].data();
  }
  for (size_t i = 0; i < ort_output_names.size(); i++) {
    output_names_c[i] = ort_output_names[i].data();
  }

  std::vector<OrtValue*> outputs(ort_output_names.size());
  ORT_RETURN_ON_ERROR(ort->Run(ort_session, ort_run_options,
                             input_names_c, inputs, ort_input_names.size(),
                             output_names_c, ort_output_names.size(), outputs.data()));

  for (size_t i = 0; i < input_count; i++) {
    ort->ReleaseValue(inputs[i]);
    if (deleters[i].f != nullptr) {
      deleters[i].f(deleters[i].param);
    }
    ORT_RETURN_ON_ERROR(ort->AllocatorFree(ort_allocator, buffers[i]));
  }

  for (size_t i = 0; i < output_count; i++) {
    void* output_buf = nullptr;
    size_t output_size = 0;
    ORT_RETURN_ON_ERROR(OrtValueToTensorProto_C(outputs.at(i), &output_buf, &output_size));

    output_sizes[i] = output_size;

    if (output_size > output_max_size) {
      ORT_RETURN_ERROR(ORT_FAIL, "output buffer not big enough");
    }
    std::memcpy(output_bufs[i], output_buf, output_size);
    
    free(output_buf);

    ort->ReleaseValue(outputs[i]);
  }

  return 0;
}

extern "C" void EnclaveDestroySession() {
  CleanupGlobalState();
}

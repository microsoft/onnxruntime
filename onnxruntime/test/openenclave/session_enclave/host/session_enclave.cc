// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <iomanip>
#include <thread>
#include <mutex>

#include <core/session/onnxruntime_cxx_api.h>
#include <core/common/status.h>
#include "test/openenclave/session_enclave/shared/ortvalue_to_tensorproto_converter.h"
#include "session_u.h"  // generated from session.edl
#include "session_enclave.h"

#include "test/onnx/tensorprotoutils.h"
#include "test/onnx/callback.h"
#include "test/onnx/mem_buffer.h"

namespace onnxruntime {
namespace openenclave {

using onnxruntime::common::StatusCategory;

#define OE_THROW_IF_ERROR(expr)                                                     \
  do {                                                                              \
    oe_result_t oe_result = (expr);                                                 \
    if (oe_result != OE_OK) {                                                       \
      throw onnxruntime::OnnxRuntimeException(ORT_WHERE, oe_result_str(oe_result)); \
    }                                                                               \
  } while (0);

#define OE_ORT_THROW_IF_ERROR(status, error_msg)                        \
  do {                                                                  \
    OrtErrorCode ort_err_code = static_cast<OrtErrorCode>(status);      \
    if (ort_err_code != ORT_OK) {                                       \
      Status ort_status(StatusCategory::ONNXRUNTIME, ort_err_code, error_msg);          \
      free(error_msg);                                                  \
      ORT_THROW_IF_ERROR(ort_status);                                   \
    }                                                                   \
  } while (0);

#define OE_ORT_THROW_IF_ERROR_S(status, error_msg)                      \
  do {                                                                  \
    OrtErrorCode ort_err_code = static_cast<OrtErrorCode>(status);      \
    if (ort_err_code != ORT_OK) {                                       \
      Status ort_status(StatusCategory::ONNXRUNTIME, ort_err_code, error_msg);          \
      ORT_THROW_IF_ERROR(ort_status);                                   \
    }                                                                   \
  } while (0);

constexpr const size_t MAX_OUTPUT_SIZE = 1024 * 1024 * 10;  // 10 MiB

SessionEnclave::SessionEnclave(const std::string& enclave_path, bool debug, bool simulate) {
  uint32_t enclave_flags = 0;
  if (debug) {
    enclave_flags |= OE_ENCLAVE_FLAG_DEBUG;
  }
  if (simulate) {
    enclave_flags |= OE_ENCLAVE_FLAG_SIMULATE;
  }

  std::cout << "Enclave starting..." << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  OE_THROW_IF_ERROR(oe_create_session_enclave(enclave_path.c_str(),
                                              OE_ENCLAVE_TYPE_SGX, enclave_flags, NULL, 0, &enclave_));

  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Enclave start-up time: " << ((std::chrono::duration<double>)(end - start)).count() << " sec" << std::endl;
}

SessionEnclave::~SessionEnclave() {
  DestroySession();
  // Cannot use OE_THROW_IF_ERROR as we are in a destructor.
  oe_result_t result = oe_terminate_enclave(enclave_);
  if (result != OE_OK) {
    std::cerr << "Error terminating enclave: " << oe_result_str(result) << std::endl;
    abort();
  }
}

void SessionEnclave::CreateSession(const std::string& model_path, OrtLoggingLevel log_level,
                                   bool enable_sequential_execution,
                                   int intra_op_num_threads,
                                   int inter_op_num_threads,
                                   uint32_t optimization_level) {
  size_t model_len;
  ORT_THROW_IF_ERROR(Env::Default().GetFileLength(model_path.c_str(), model_len));
  std::vector<char> model_data(model_len);
  auto model_data_span = gsl::make_span(model_data);
  auto st = Env::Default().ReadFileIntoBuffer(model_path.c_str(),
                                              0, model_len, model_data_span);
  if (!st.IsOK()) {
    ORT_THROW("read file ", model_path, " failed:", st.ErrorMessage());
  }

  int status;
  char* error_msg = nullptr;
  OE_THROW_IF_ERROR(EnclaveCreateSession(enclave_, &status, &error_msg, &output_count_,
                                         model_data.data(), model_data.size(),
                                         log_level,
                                         enable_sequential_execution,
                                         intra_op_num_threads, inter_op_num_threads,
                                         optimization_level));
  OE_ORT_THROW_IF_ERROR(status, error_msg);
}

void SessionEnclave::DestroySession() {
  EnclaveDestroySession(enclave_);
}

std::vector<Ort::Value> SessionEnclave::Run(const std::vector<Ort::Value>& inputs, bool return_outputs,
                                            OrtInferenceTimestamps* timestamps) const {
  std::vector<char*> input_bufs(inputs.size());
  std::vector<size_t> input_sizes(inputs.size());
  for (size_t i = 0; i < inputs.size(); i++) {
    const Ort::Value& ort_value = inputs.at(i);
    void* input_buf = nullptr;
    size_t input_size = 0;
    ORT_THROW_IF_ERROR(OrtValueToTensorProto(ort_value, &input_buf, &input_size));
    input_bufs[i] = reinterpret_cast<char*>(input_buf);
    input_sizes[i] = input_size;
  }

  if (timestamps) {
    timestamps->start = std::chrono::high_resolution_clock::now();
  }

  int status;
  char* error_msg = nullptr;
  size_t output_count = return_outputs ? output_count_ : 0;
  std::vector<char*> output_bufs(output_count);
  std::vector<size_t> output_sizes(output_count);
  for (size_t i = 0; i < output_count; i++) {
    output_bufs[i] = reinterpret_cast<char*>(malloc(sizeof(char) * MAX_OUTPUT_SIZE));
  }

  OE_THROW_IF_ERROR(EnclaveRunInference(enclave_, &status, &error_msg,
                                        input_bufs.data(), input_sizes.data(), inputs.size(),
                                        output_bufs.data(), output_sizes.data(), output_count, MAX_OUTPUT_SIZE));
  OE_ORT_THROW_IF_ERROR(status, error_msg);

  if (timestamps) {
    timestamps->end = std::chrono::high_resolution_clock::now();
  }

  for (size_t i = 0; i < inputs.size(); i++) {
    free(input_bufs[i]);
  }

  std::vector<Ort::Value> outputs;
  if (return_outputs) {
    for (size_t i = 0; i < output_count; i++) {
      size_t output_len = output_sizes[i];
      char* output_buf = output_bufs[i];

      onnx::TensorProto tensorproto{};
      tensorproto.ParseFromArray(output_buf, output_len);

      size_t tensor_memsize;
      auto status = onnxruntime::test::GetSizeInBytesFromTensorProto<0>(tensorproto, &tensor_memsize);
      if (!status.IsOK()) {
        ORT_THROW(status.ToString());
      }

      Ort::AllocatorWithDefaultOptions allocator{};
      void* output_buf_tensor = allocator.Alloc(tensor_memsize);
      Ort::Value output{nullptr};
      onnxruntime::test::OrtCallback deleter;

      status = onnxruntime::test::TensorProtoToMLValue(
          tensorproto,
          onnxruntime::test::MemBuffer(output_buf_tensor, tensor_memsize, *allocator.GetInfo()),
          output, deleter);
      if (!status.IsOK()) {
        ORT_THROW(status.ToString());
      }

      // Note: output_buf_tensor is owned by output (Ort::Value) and will be freed automatically.

      outputs.emplace_back(std::move(output));
    }
  }

  for (size_t i = 0; i < output_count; i++) {
    free(output_bufs[i]);
  }

  return outputs;
}

}  // namespace openenclave
}  // namespace onnxruntime
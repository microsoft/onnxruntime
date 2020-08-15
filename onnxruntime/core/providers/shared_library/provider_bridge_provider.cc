// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This is the provider DLL side of the provider API to let providers be built as a DLL

#include "provider_api.h"
#include <assert.h>
#include <mutex>
#include <iostream>  // For std::cout used in a stub

#define PROVIDER_NOT_IMPLEMENTED ORT_THROW("Unimplemented shared library provider method");

extern "C" {
void* Provider_GetHost();
}

namespace onnxruntime {

ProviderHost* g_host = reinterpret_cast<ProviderHost*>(Provider_GetHost());

static std::unique_ptr<std::vector<std::function<void()>>> s_run_on_unload_;

void RunOnUnload(std::function<void()> function) {
  static std::mutex mutex;
  std::lock_guard<std::mutex> guard{mutex};
  if (!s_run_on_unload_)
    s_run_on_unload_ = onnxruntime::make_unique<std::vector<std::function<void()>>>();
  s_run_on_unload_->push_back(std::move(function));
}

// This object is destroyed as part of the DLL unloading code and handles running all of the RunOnLoad functions
struct OnUnload {
  ~OnUnload() {
    if (!s_run_on_unload_)
      return;

    for (auto& function : *s_run_on_unload_)
      function();

    s_run_on_unload_.reset();
  }

} g_on_unload;

}  // namespace onnxruntime

// Override default new/delete so that we match the host's allocator
void* operator new(size_t n) { return onnxruntime::g_host->HeapAllocate(n); }
void operator delete(void* p) { return onnxruntime::g_host->HeapFree(p); }
void operator delete(void* p, size_t /*size*/) { return onnxruntime::g_host->HeapFree(p); }

namespace onnxruntime {

Provider_AllocatorPtr CreateAllocator(const Provider_DeviceAllocatorRegistrationInfo& info, int16_t device_id,
                                      bool use_arena) {
  return g_host->CreateAllocator(info, device_id, use_arena);
}

std::unique_ptr<Provider_OrtMemoryInfo> Provider_OrtMemoryInfo::Create(
    const char* name_, OrtAllocatorType type_, Provider_OrtDevice* device_, int id_, OrtMemType mem_type_) {
  return g_host->OrtMemoryInfo_Create(name_, type_, device_, id_, mem_type_);
}

template <>
MLDataType DataTypeImpl::GetType<float>() {
  return g_host->DataTypeImpl_GetType_float();
}

template <>
MLDataType DataTypeImpl::GetTensorType<float>() {
  return g_host->DataTypeImpl_GetTensorType_float();
}

const std::vector<MLDataType>& DataTypeImpl::AllFixedSizeTensorTypes() {
  return g_host->DataTypeImpl_AllFixedSizeTensorTypes();
}

TensorShape::TensorShape(const int64_t* dimension_sizes, size_t dimension_count)
    : std::vector<int64_t>(dimension_count) {
  for (size_t i = 0; i < dimension_count; ++i) {
    (*this)[i] = dimension_sizes[i];
  }
}

TensorShape::TensorShape(const std::vector<int64_t>& dims, size_t start, size_t end) {
  assign(dims.begin() + start, dims.begin() + end);
}

int64_t TensorShape::Size() const {
  size_t arraySize = size();
  int64_t size = SizeHelper(0, arraySize);
  //should we cache the size? as multiple operation may be expensive.
  return size;
}

int64_t TensorShape::SizeHelper(size_t start, size_t end) const {
  // Must return 1 for an empty sequence
  int64_t size = 1;
  for (size_t i = start; i < end; i++) {
    if ((*this)[i] < 0) return -1;
    size *= (*this)[i];
  }
  return size;
}

TensorShape TensorShape::Slice(size_t dimstart, size_t dimend) const {
  assert(dimstart <= dimend && dimend <= size());  // "Invalid tensor shape slice argument."
  return TensorShape(*this, dimstart, dimend);
}

TensorShape TensorShape::Slice(size_t dimstart) const {
  return Slice(dimstart, size());
}

std::string TensorShape::ToString() const {
  std::string result;

  result.append("{");
  bool first = true;
  for (auto dim : (*this)) {
    if (!first) {
      result.append(",");
    }

    result.append(std::to_string(dim));
    first = false;
  }
  result.append("}");

  return result;
}

CPUIDInfo g_info;

const CPUIDInfo& CPUIDInfo::GetCPUIDInfo() {
  return g_info;
}

bool CPUIDInfo::HasAVX2() const {
  return g_host->CPU_HasAVX2();
}

bool CPUIDInfo::HasAVX512f() const {
  return g_host->CPU_HasAVX512f();
}

Provider_AllocatorPtr CreateAllocator(Provider_DeviceAllocatorRegistrationInfo info, int16_t device_id) {
  return g_host->CreateAllocator(info, device_id);
}

std::unique_ptr<Provider_IDeviceAllocator> Provider_CreateCPUAllocator(std::unique_ptr<Provider_OrtMemoryInfo> info) {
  return g_host->CreateCPUAllocator(std::move(info));
}

#ifdef USE_TENSORRT
std::unique_ptr<Provider_IDeviceAllocator> Provider_CreateCUDAAllocator(int16_t device_id, const char* name) {
  return g_host->CreateCUDAAllocator(device_id, name);
}

std::unique_ptr<Provider_IDeviceAllocator> Provider_CreateCUDAPinnedAllocator(int16_t device_id, const char* name) {
  return g_host->CreateCUDAPinnedAllocator(device_id, name);
}

std::unique_ptr<Provider_IDataTransfer> Provider_CreateGPUDataTransfer() {
  return g_host->CreateGPUDataTransfer();
}
#endif

std::string GetEnvironmentVar(const std::string& var_name) {
  return g_host->GetEnvironmentVar(var_name);
}

Provider_IExecutionProvider::Provider_IExecutionProvider(const std::string& type) {
  p_ = g_host->Create_IExecutionProvider_Router(this, type).release();
}

namespace logging {

bool Logger::OutputIsEnabled(Severity severity, DataType data_type) const noexcept {
  ORT_UNUSED_PARAMETER(severity);
  ORT_UNUSED_PARAMETER(data_type);
  return false;
  // TODO: Logging not essential to make it work initially, do later
}

static Logger g_default_logger;

const Logger& LoggingManager::DefaultLogger() {
  return g_default_logger;
}

Capture::Capture(const Logger& logger, logging::Severity severity, const char* category,
                 logging::DataType dataType, const CodeLocation& location) {
  PROVIDER_NOT_IMPLEMENTED
  ORT_UNUSED_PARAMETER(logger);
  ORT_UNUSED_PARAMETER(severity);
  ORT_UNUSED_PARAMETER(category);
  ORT_UNUSED_PARAMETER(dataType);
  ORT_UNUSED_PARAMETER(location);
}

std::ostream& Capture::Stream() noexcept {
  // PROVIDER_NOT_IMPLEMENTED
  return std::cout;
}

const char* Category::onnxruntime = "onnxruntime";

}  // namespace logging

namespace common {

Status::Status(StatusCategory category, int code, const std::string& msg) {
  // state_ will be allocated here causing the status to be treated as a failure
  ORT_ENFORCE(code != static_cast<int>(common::OK));

  state_ = onnxruntime::make_unique<State>(category, code, msg);
}

Status::Status(StatusCategory category, int code, const char* msg) {
  // state_ will be allocated here causing the status to be treated as a failure
  ORT_ENFORCE(code != static_cast<int>(common::OK));

  state_ = onnxruntime::make_unique<State>(category, code, msg);
}

int Status::Code() const noexcept {
  return IsOK() ? static_cast<int>(common::OK) : state_->code;
}

const std::string& Status::ErrorMessage() const noexcept {
  return IsOK() ? EmptyString() : state_->msg;
}

std::string Status::ToString() const {
  if (state_ == nullptr) {
    return std::string("OK");
  }

  std::string result;

  if (common::SYSTEM == state_->category) {
    result += "SystemError";
    result += " : ";
    result += std::to_string(errno);
  } else if (common::ONNXRUNTIME == state_->category) {
    result += "[ONNXRuntimeError]";
    result += " : ";
    result += std::to_string(Code());
    result += " : ";
    result += StatusCodeToString(static_cast<StatusCode>(Code()));
    result += " : ";
    result += state_->msg;
  }

  return result;
}

const std::string& Status::EmptyString() noexcept {
  static std::string s_empty;
  return s_empty;
}

}  // namespace common

std::vector<std::string> GetStackTrace() {
  // PROVIDER_NOT_IMPLEMENTED
  return {};
}

void LogRuntimeError(uint32_t session_id, const common::Status& status,
                     const char* file, const char* function, uint32_t line) {
  return g_host->LogRuntimeError(session_id, status, file, function, line);
}

}  // namespace onnxruntime

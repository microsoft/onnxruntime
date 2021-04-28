// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This is the provider DLL side of the provider API to let providers be built as a DLL

#include "provider_api.h"
#include <assert.h>
#include <mutex>
#include "core/providers/shared/common.h"

#ifndef _Ret_notnull_
#define _Ret_notnull_
#endif


#ifndef _Post_writable_byte_size_
#define _Post_writable_byte_size_(n)
#endif

// Override default new/delete so that we match the host's allocator
_Ret_notnull_ _Post_writable_byte_size_(n) void* operator new(size_t n) { return Provider_GetHost()->HeapAllocate(n); }
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wimplicit-exception-spec-mismatch"
void operator delete(void* p) { return Provider_GetHost()->HeapFree(p); }
#pragma clang diagnostic pop
void operator delete(void* p, size_t /*size*/) { return Provider_GetHost()->HeapFree(p); }

namespace onnxruntime {

ProviderHost* g_host = Provider_GetHost();

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

AllocatorPtr CreateAllocator(const AllocatorCreationInfo& info) {
  return g_host->CreateAllocator(info);
}

void AllocatorManager__InsertAllocator(AllocatorManager* p, AllocatorPtr allocator) {
  return g_host->AllocatorManager__InsertAllocator(p, allocator);
}

AllocatorPtr AllocatorManager__GetAllocator(AllocatorManager* p, int id, OrtMemType mem_type) {
  return g_host->AllocatorManager__GetAllocator(p, id, mem_type);
}

template <>
MLDataType DataTypeImpl::GetType<float>() {
  return g_host->DataTypeImpl_GetType_float();
}

template <>
MLDataType DataTypeImpl::GetTensorType<float>() {
  return g_host->DataTypeImpl_GetTensorType_float();
}

Status IDataTransfer::CopyTensor(const Tensor& src, Tensor& dst) const {
  return g_host->IDataTransfer__CopyTensor(this, src, dst);
}

Status IDataTransfer::CopyTensors(const std::vector<SrcDstPair>& src_dst_pairs) const {
  return g_host->IDataTransfer__CopyTensors(this, src_dst_pairs);
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
  return g_host->TensorShape__SizeHelper(this, start, end);
}

TensorShape TensorShape::Slice(size_t dimstart, size_t dimend) const {
  assert(dimstart <= dimend && dimend <= size());  // "Invalid tensor shape slice argument."
  return TensorShape(*this, dimstart, dimend);
}

TensorShape TensorShape::Slice(size_t dimstart) const {
  return Slice(dimstart, size());
}

std::string TensorShape::ToString() const {
  return g_host->TensorShape__ToString(this);
}

AllocatorPtr CreateAllocator(AllocatorCreationInfo info) {
  return g_host->CreateAllocator(info);
}

std::unique_ptr<IAllocator> CreateCPUAllocator(const OrtMemoryInfo& info) {
  return g_host->CreateCPUAllocator(info);
}

bool IAllocator::CalcMemSizeForArrayWithAlignment(size_t nmemb, size_t size, size_t alignment, size_t* out) noexcept {
  return g_host->IAllocator__CalcMemSizeForArrayWithAlignment(nmemb, size, alignment, out);
}

AllocatorPtr IExecutionProvider::GetAllocator(int id, OrtMemType mem_type) const {
  return g_host->IExecutionProvider__GetAllocator(this, id, mem_type);
}

void IExecutionProvider::InsertAllocator(AllocatorPtr allocator) {
  g_host->IExecutionProvider__InsertAllocator(this, allocator);
}

void IExecutionProvider::TryInsertAllocator(AllocatorPtr allocator) {
  g_host->IExecutionProvider__TryInsertAllocator(this, allocator);
}

std::vector<std::unique_ptr<ComputeCapability>> IExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                                                                  const std::vector<const KernelRegistry*>& kernel_registries) const {
  return g_host->IExecutionProvider__GetCapability(this, graph_viewer, kernel_registries);
}

common::Status IExecutionProvider::Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                                           std::vector<NodeComputeInfo>& node_compute_funcs) {
  return g_host->IExecutionProvider__Compile(this, fused_nodes, node_compute_funcs);
}

common::Status IExecutionProvider::Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                                           std::string& dll_path) {
  return g_host->IExecutionProvider__Compile(this, fused_nodes, dll_path);
}

common::Status IExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                           std::vector<NodeComputeInfo>& node_compute_funcs) {
  return g_host->IExecutionProvider__Compile(this, fused_nodes_and_graphs, node_compute_funcs);
}

int IExecutionProvider::GenerateMetaDefId(const onnxruntime::GraphViewer& graph_viewer, uint64_t& model_hash) const {
  return g_host->IExecutionProvider__GenerateMetaDefId(this, graph_viewer, model_hash);
}

void IExecutionProvider::RegisterAllocator(std::shared_ptr<AllocatorManager> allocator_manager) {
  return g_host->IExecutionProvider__RegisterAllocator(this, allocator_manager);
}

#ifdef USE_TENSORRT
std::unique_ptr<IAllocator> CreateCUDAAllocator(int16_t device_id, const char* name) {
  return g_host->CreateCUDAAllocator(device_id, name);
}

std::unique_ptr<IAllocator> CreateCUDAPinnedAllocator(int16_t device_id, const char* name) {
  return g_host->CreateCUDAPinnedAllocator(device_id, name);
}

std::unique_ptr<IDataTransfer> CreateGPUDataTransfer(void* stream) {
  return g_host->CreateGPUDataTransfer(stream);
}
#endif

std::string GetEnvironmentVar(const std::string& var_name) {
  return g_host->GetEnvironmentVar(var_name);
}

namespace logging {

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

Status::Status(StatusCategory category, int code) : Status(category, code, "") {
}

int Status::Code() const noexcept {
  return IsOK() ? static_cast<int>(common::OK) : state_->code;
}

const std::string& Status::ErrorMessage() const noexcept {
  return IsOK() ? EmptyString() : state_->msg;
}

std::string Status::ToString() const { return g_host->Status__ToString(this); }

const std::string& Status::EmptyString() noexcept {
  static std::string s_empty;
  return s_empty;
}

}  // namespace common

std::vector<std::string> GetStackTrace() { return g_host->GetStackTrace(); }

void LogRuntimeError(uint32_t session_id, const common::Status& status,
                     const char* file, const char* function, uint32_t line) {
  return g_host->LogRuntimeError(session_id, status, file, function, line);
}

std::unique_ptr<OpKernelInfo> CopyOpKernelInfo(const OpKernelInfo& info) {
  return g_host->CopyOpKernelInfo(info);
}

}  // namespace onnxruntime

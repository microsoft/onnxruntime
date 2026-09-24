// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <windows.h>

#include "core/graph/graph.h"
#include "core/providers/cuda/cuda_provider_factory.h"
#include "core/providers/providers.h"
#include "test/providers/cuda/internal_testing/cuda_internal_test_api.h"
#include "test/providers/cuda/internal_testing/cuda_internal_test_helpers.h"

namespace onnxruntime {
ProviderInfo_CUDA& GetProviderInfo_CUDA_Test();

namespace test {
namespace {
const CudaInternalTestApi& Api() {
  static const auto* api = [] {
    // The normal provider loader sets ProviderHost before loading the module.
    // Linking its import library would run provider static initializers too early.
    GetProviderInfo_CUDA_Test();
    const auto module = GetModuleHandleW(L"onnxruntime_providers_cuda_ut.dll");
    ORT_ENFORCE(module != nullptr, "CUDA internal-test module was not loaded");
    const auto get_api = reinterpret_cast<const CudaInternalTestApi* (*)()>(
        GetProcAddress(module, "GetCudaInternalTestApi"));
    ORT_ENFORCE(get_api != nullptr, "CUDA internal-test API was not exported");
    return get_api();
  }();
  return *api;
}
}  // namespace

std::shared_ptr<CUDAExecutionProvider> CreateCudaInternalTestExecutionProvider(const CUDAExecutionProviderInfo& info) {
  // Kernel instances and the runtime probes must come from the same provider module.
  auto factory = GetProviderInfo_CUDA_Test().CreateExecutionProviderFactory(info);
  std::shared_ptr<IExecutionProvider> provider = factory->CreateProvider();
  return std::static_pointer_cast<CUDAExecutionProvider>(provider);
}

std::unique_ptr<IExternalDataLoader> CreateCudaInternalTestExternalDataLoader(
    int device, size_t readers, cuda::ExternalDataLoader::AllocatePinnedBufferFn allocate,
    cuda::ExternalDataLoader::CreateStreamFn create_stream) {
  return Api().create_external_data_loader(device, readers, allocate, create_stream);
}

#if !defined(USE_CUDA_MINIMAL) && !defined(DISABLE_CONTRIB_OPS) && defined(USE_FPA_INTB_GEMM) && USE_FPA_INTB_GEMM
size_t GetMatMulNBitsLastComputeWorkspaceBytes(const OpKernel* kernel) {
  return Api().matmul_last_workspace_bytes(kernel);
}

bool GetMatMulNBitsLastComputeUsedPreallocatedWorkspace(const OpKernel* kernel) {
  return Api().matmul_last_used_preallocated(kernel);
}
#endif
}  // namespace test

#if !defined(USE_CUDA_MINIMAL)
void AttentionKernelOptions::InitializeOnce(int kernel, bool use_build_flag, bool check_cudnn_version) {
  test::Api().initialize_attention_options(*this, kernel, use_build_flag, check_cudnn_version);
}

namespace contrib::cuda {
#if !defined(DISABLE_CONTRIB_OPS)
PackedAttentionProblemResult<PackedAttentionProblem> BuildPackedAttentionProblem(
    const PackedAttentionInputShapes& inputs) noexcept {
  return test::Api().build_packed_problem(inputs);
}

PackedAttentionProblemResult<PackedMultiHeadAttentionProblem> BuildPackedMultiHeadAttentionProblem(
    const PackedMultiHeadAttentionInputShapes& inputs) noexcept {
  return test::Api().build_packed_mha_problem(inputs);
}

PackedAttentionWorkspaceResult GetPackedAttentionWorkspaceRecipe(const PackedAttentionProblem& problem) noexcept {
  return test::Api().packed_recipe(problem);
}

PackedAttentionWorkspaceResult GetPackedMultiHeadAttentionWorkspaceRecipe(
    const PackedMultiHeadAttentionProblem& problem) noexcept {
  return test::Api().packed_mha_recipe(problem);
}

PackedAttentionWorkspaceAggregate GetPackedAttentionWorkspaceAggregateForBounds(
    const PackedAttentionProblem& problem, PackedAttentionBackendMask backends) noexcept {
  return test::Api().packed_aggregate(problem, backends);
}

PackedAttentionWorkspaceAggregate GetPackedMultiHeadAttentionWorkspaceAggregateForBounds(
    const PackedMultiHeadAttentionProblem& problem, PackedAttentionBackendMask backends) noexcept {
  return test::Api().packed_mha_aggregate(problem, backends);
}

PackedAttentionBackendMask GetPackedAttentionReachableBackendsForBounds(
    const PackedAttentionProblem& problem, PackedAttentionHeadSizeDomain domain,
    const cudaDeviceProp& device, const AttentionKernelOptions& options) {
  return test::Api().packed_reachable_backends(problem, domain, device, options);
}

PackedAttentionBackendMask GetPackedMultiHeadAttentionReachableBackendsForBounds(
    const PackedMultiHeadAttentionProblem& problem, const cudaDeviceProp& device, const AttentionKernelOptions& options) {
  return test::Api().packed_mha_reachable_backends(problem, device, options);
}

std::optional<PackedAttentionWorkspaceAggregate> EstimatePackedAttentionWorkspace(
    const PackedAttentionWorkspaceEstimateConfig& config, gsl::span<const WorkspaceInputShape> shapes,
    const cudaDeviceProp& device, const AttentionKernelOptions& options) {
  return test::Api().estimate_packed_config(config, shapes, device, options);
}

std::optional<PackedAttentionWorkspaceAggregate> EstimatePackedAttentionWorkspace(
    const Node& node, gsl::span<const WorkspaceInputShape> shapes,
    const cudaDeviceProp& device, const AttentionKernelOptions& options) {
  return test::Api().estimate_packed_node(node, shapes, device, options);
}

void SetPackedAttentionWorkspaceRequirements(
    const PackedAttentionWorkspaceAggregate& estimate, InlinedVector<WorkspaceRequirement>& requirements) {
  test::Api().set_packed_requirements(estimate, requirements);
}

GQAFlashWorkspaceResult GetGQAFlashWorkspaceRecipe(
    const GQAWorkspaceProblem& problem, const GQAFlashConfig& config) noexcept {
  return test::Api().gqa_flash_recipe(problem, config);
}

GQACompleteWorkspaceResult GetGQACompleteWorkspaceRecipe(
    const GQAWorkspaceProblem& problem, const GQAConcreteRoute& route) noexcept {
  return test::Api().gqa_complete_recipe(problem, route);
}

GQAWorkspaceAggregate GetGQAWorkspaceAggregateForBounds(const GQAWorkspaceBounds& bounds) noexcept {
  return test::Api().gqa_aggregate(bounds);
}

std::optional<GQAWorkspaceAggregate> EstimateGroupQueryAttentionWorkspace(
    const GQAWorkspaceEstimateConfig& config, gsl::span<const WorkspaceInputShape> shapes,
    const cudaDeviceProp& device, const AttentionKernelOptions& options) {
  return test::Api().estimate_gqa_config(config, shapes, device, options);
}

std::optional<GQAWorkspaceAggregate> EstimateGroupQueryAttentionWorkspace(
    const Node& node, gsl::span<const WorkspaceInputShape> shapes,
    const cudaDeviceProp& device, const AttentionKernelOptions& options, bool constant_head_sink) {
  return test::Api().estimate_gqa_node(node, shapes, device, options, constant_head_sink);
}

void SetGroupQueryAttentionWorkspaceRequirements(
    const GQAWorkspaceAggregate& estimate, InlinedVector<WorkspaceRequirement>& requirements) {
  test::Api().set_gqa_requirements(estimate, requirements);
}

void SetGroupQueryAttentionLevel1MemoryEstimate(const GQAWorkspaceAggregate& workspace, Level1MemoryEstimate& estimate) {
  test::Api().set_gqa_level1_estimate(workspace, estimate);
}

#if defined(USE_FPA_INTB_GEMM) && USE_FPA_INTB_GEMM
std::optional<int64_t> ComputeMatMulNBitsLeadingDimProduct(gsl::span<const int64_t> shape) {
  return test::Api().matmul_leading_dim_product(shape);
}

std::optional<Level1MemoryEstimate> EstimateMatMulNBitsMemory(
    const Node& node, const cudaDeviceProp& device, MatMulNBitsMemoryEstimateOptions options) {
  return test::Api().estimate_matmul_node(node, device, options);
}

std::optional<Level1MemoryEstimate> EstimateMatMulNBitsMemory(
    const Node& node, gsl::span<const int64_t> shape, const cudaDeviceProp& device, MatMulNBitsMemoryEstimateOptions options) {
  return test::Api().estimate_matmul_shape(node, shape, device, options);
}

std::optional<size_t> EstimateMatMulNBitsWorkspace(const Node& node, const cudaDeviceProp& device) {
  return test::Api().matmul_workspace_node(node, device);
}

std::optional<size_t> EstimateMatMulNBitsWorkspace(
    const Node& node, gsl::span<const int64_t> shape, const cudaDeviceProp& device) {
  return test::Api().matmul_workspace_shape(node, shape, device);
}
#endif
#endif
}  // namespace contrib::cuda
#endif
}  // namespace onnxruntime

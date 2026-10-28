// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef USE_CUDA_MINIMAL

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cudnn_frontend.h>
#include <gtest/gtest.h>

namespace onnxruntime::test {
namespace {

namespace fe = cudnn_frontend;

TEST(CUDA_EP_Unittest, CudnnFrontendPre128LibraryAPIsAreUnsupported) {
#if CUDART_VERSION < 12080
  EXPECT_EQ(fe::experimental::detail::cuda_library_load_data(
                nullptr, nullptr, nullptr, nullptr, 0, nullptr, nullptr, 0),
            cudaErrorNotSupported);
  EXPECT_EQ(fe::experimental::detail::cuda_library_get_kernel(nullptr, nullptr, ""), cudaErrorNotSupported);
  EXPECT_EQ(fe::experimental::detail::cuda_library_unload(nullptr), cudaErrorNotSupported);
  EXPECT_EQ(fe::experimental::detail::cuda_kernel_set_attribute_for_device(nullptr, 0, 0, 0),
            cudaErrorNotSupported);
  EXPECT_EQ(fe::experimental::detail::cu_tensor_map_encode_tiled(
                nullptr, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 0, nullptr, nullptr, nullptr, nullptr, nullptr,
                CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
            CUDA_ERROR_NOT_SUPPORTED);
#else
  GTEST_SKIP() << "Requires CUDA headers older than 12.8.";
#endif
}

TEST(CUDA_EP_Unittest, CudnnFrontendOssEngineCapability) {
  fe::graph::Execution_plan_list plans;
  const auto sdpa = plans.check_oss_sdpa_engine_support(90);
  const auto norm = plans.check_oss_rms_norm_silu_support(100);
  EXPECT_EQ(sdpa.code, fe::error_code_t::GRAPH_NOT_SUPPORTED);
  EXPECT_EQ(norm.code, fe::error_code_t::GRAPH_NOT_SUPPORTED);
#if CUDART_VERSION < 12080
  EXPECT_NE(sdpa.err_msg.find("CUDA 12.8"), std::string::npos);
  EXPECT_NE(norm.err_msg.find("CUDA 12.8"), std::string::npos);
#else
  EXPECT_EQ(sdpa.err_msg, "No OSS engine registered");
  EXPECT_EQ(norm.err_msg, "No RmsNorm+SiLU OSS engine registered");
#endif
}

void TestCudnnBackendAttention(bool bf16) {
  int device = 0;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  cudaDeviceProp properties{};
  ASSERT_EQ(cudaGetDeviceProperties(&properties, device), cudaSuccess);
  if (properties.major < 8 || cudnnGetVersion() < 90000) {
    GTEST_SKIP() << "Requires SM80+ and cuDNN 9+.";
  }

  cudnnHandle_t raw_handle = nullptr;
  ASSERT_EQ(cudnnCreate(&raw_handle), CUDNN_STATUS_SUCCESS);
  auto destroy_handle = [](cudnnHandle_t handle) { EXPECT_EQ(cudnnDestroy(handle), CUDNN_STATUS_SUCCESS); };
  std::unique_ptr<std::remove_pointer_t<cudnnHandle_t>, decltype(destroy_handle)> handle(raw_handle, destroy_handle);
  auto free_buffer = [](void* buffer) { EXPECT_EQ(cudaFree(buffer), cudaSuccess); };
  auto allocate = [&](size_t bytes) {
    void* buffer = nullptr;
    const auto status = cudaMalloc(&buffer, bytes);
    EXPECT_EQ(status, cudaSuccess);
    return std::unique_ptr<void, decltype(free_buffer)>(buffer, free_buffer);
  };

  constexpr size_t elements = 2 * 32 * 64;
  constexpr size_t bytes = elements * sizeof(uint16_t);
  auto q_buffer = allocate(bytes);
  auto k_buffer = allocate(bytes);
  auto v_buffer = allocate(bytes);
  auto o_buffer = allocate(bytes);
  ASSERT_NE(q_buffer, nullptr);
  ASSERT_NE(k_buffer, nullptr);
  ASSERT_NE(v_buffer, nullptr);
  ASSERT_NE(o_buffer, nullptr);
  ASSERT_EQ(cudaMemset(q_buffer.get(), 0, bytes), cudaSuccess);
  ASSERT_EQ(cudaMemset(k_buffer.get(), 0, bytes), cudaSuccess);
  // With zero Q/K, attention over a constant V must return that same constant.
  const uint16_t one = bf16 ? 0x3f80 : 0x3c00;
  std::array<uint16_t, elements> expected;
  expected.fill(one);
  ASSERT_EQ(cudaMemcpy(v_buffer.get(), expected.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);

  fe::graph::Graph graph;
  graph.set_io_data_type(bf16 ? fe::DataType_t::BFLOAT16 : fe::DataType_t::HALF)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT);
  auto tensor = [&](int64_t uid) {
    return graph.tensor(fe::graph::Tensor_attributes()
                            .set_uid(uid)
                            .set_dim({1, 2, 32, 64})
                            .set_stride({4096, 2048, 64, 1}));
  };
  auto q = tensor(1);
  auto k = tensor(2);
  auto v = tensor(3);
  auto [output, stats] = graph.sdpa(q, k, v, fe::graph::SDPA_attributes().set_generate_stats(false).set_attn_scale(0.125f));
  output->set_output(true).set_uid(4).set_dim({1, 2, 32, 64}).set_stride({4096, 2048, 64, 1});
  const auto status = graph.build(handle.get(), {fe::HeurMode_t::A});
  ASSERT_TRUE(status.is_good()) << status.err_msg;
  const auto workspace_bytes = graph.get_workspace_size();
  ASSERT_GE(workspace_bytes, 0);
  auto workspace = allocate(std::max<size_t>(1, static_cast<size_t>(workspace_bytes)));
  ASSERT_NE(workspace, nullptr);
  std::unordered_map<int64_t, void*> variant_pack{{q->get_uid(), q_buffer.get()},
                                                  {k->get_uid(), k_buffer.get()},
                                                  {v->get_uid(), v_buffer.get()},
                                                  {output->get_uid(), o_buffer.get()}};
  const auto executed = graph.execute(handle.get(), variant_pack, workspace.get());
  ASSERT_TRUE(executed.is_good()) << executed.err_msg;
  std::array<uint16_t, elements> actual{};
  ASSERT_EQ(cudaMemcpy(actual.data(), o_buffer.get(), bytes, cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(actual, expected);
}

TEST(CUDA_EP_Unittest, CudnnFrontendBackendAttentionFloat16) {
  TestCudnnBackendAttention(false);
}

TEST(CUDA_EP_Unittest, CudnnFrontendBackendAttentionBFloat16) {
  TestCudnnBackendAttention(true);
}

}  // namespace
}  // namespace onnxruntime::test
#endif

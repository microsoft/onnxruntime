// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <vector>
#include "gtest/gtest.h"
#include "contrib_ops/webgpu/moe/gate_1token.h"
#include "core/providers/webgpu/allocator.h"
#include "core/providers/webgpu/compute_context.h"
#include "test/providers/webgpu/internal_test_kernel.h"
#include "core/common/float16.h"
#include "test/util/include/asserts.h"

#if !defined(DISABLE_CONTRIB_OPS) && !defined(ENABLE_TRAINING)
namespace onnxruntime::test {
static void CheckSingleTokenGateIndices(const std::vector<float>& logits, int k,
                                        const std::vector<uint32_t>& expected, bool use_fp16) {
  WebGpuTestKernel test_kernel;
  auto& webgpu_ep = test_kernel.Ep();
  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  auto& recording = webgpu_ep.Recording();
  const auto cols = static_cast<uint32_t>(logits.size());
  ASSERT_LE(cols, context.DeviceLimits().maxComputeWorkgroupSizeX);
  ASSERT_LE(cols, context.DeviceLimits().maxComputeInvocationsPerWorkgroup);
  ASSERT_EQ(expected.size(), static_cast<size_t>(k));

  webgpu::ComputeContextBase compute_context(context, webgpu_ep, test_kernel.Kernel());

  std::vector<MLFloat16> fp16_logits;
  fp16_logits.reserve(logits.size());
  for (float value : logits) fp16_logits.emplace_back(value);
  auto fp32_logits = logits;
  const size_t element_size = use_fp16 ? sizeof(MLFloat16) : sizeof(float);
  const size_t buffer_size = (cols * element_size + 3) & ~size_t{3};
  wgpu::BufferDescriptor logits_desc{};
  logits_desc.size = buffer_size;
  logits_desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
  auto logits_buffer = context.Device().CreateBuffer(&logits_desc);
  ASSERT_NE(logits_buffer.Get(), nullptr);
  webgpu_ep.BufferManager().Upload(recording, use_fp16 ? static_cast<void*>(fp16_logits.data()) : static_cast<void*>(fp32_logits.data()),
                                   logits_buffer.Get(), cols * element_size);
  wgpu::BufferDescriptor weights_desc{};
  weights_desc.size = buffer_size;
  weights_desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
  auto weights_buffer = context.Device().CreateBuffer(&weights_desc);
  wgpu::BufferDescriptor indices_desc{};
  indices_desc.size = (k * sizeof(uint32_t) + 15) & ~size_t{15};
  indices_desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
  auto indices_buffer = context.Device().CreateBuffer(&indices_desc);
  ASSERT_NE(weights_buffer.Get(), nullptr);
  ASSERT_NE(indices_buffer.Get(), nullptr);

  const auto dtype = use_fp16 ? DataTypeImpl::GetType<MLFloat16>() : DataTypeImpl::GetType<float>();
  const auto memory_info = OrtMemoryInfo(WEBGPU_BUFFER, OrtDeviceAllocator, webgpu::WebGpuDevice(0), OrtMemTypeDefault);
  Tensor router_logits(dtype, TensorShape{1, cols}, logits_buffer.Get(), memory_info);
  Tensor topk_values(dtype, TensorShape{1, cols}, weights_buffer.Get(), memory_info);
  Tensor indirect_experts(DataTypeImpl::GetType<uint32_t>(), TensorShape{k}, indices_buffer.Get(), memory_info);
  contrib::webgpu::Gate1TokenProgram gate{k, use_fp16, false, true};
  gate.AddInputs({{&router_logits, webgpu::ProgramTensorMetadataDependency::Type}})
      .AddOutput({&topk_values, webgpu::ProgramTensorMetadataDependency::None})
      .AddOutput({&indirect_experts, webgpu::ProgramTensorMetadataDependency::None})
      .SetWorkgroupSize(cols)
      .SetDispatchGroupSize(1)
      .AddUniformVariables({1u, cols});
  ASSERT_STATUS_OK(compute_context.RunProgram(gate));

  std::vector<uint32_t> actual(k);
  webgpu_ep.BufferManager().Download(recording, indices_buffer.Get(), actual.data(), actual.size() * sizeof(uint32_t));
  EXPECT_EQ(actual, expected);
}

TEST(MoETest, QMoETest_WebGPU_SingleTokenGateIndices) {
  CheckSingleTokenGateIndices({0.0f, -INFINITY, -INFINITY}, 2, {0, 1}, true);
  CheckSingleTokenGateIndices({0.0f, -INFINITY, -INFINITY}, 2, {0, 1}, false);
  CheckSingleTokenGateIndices({10.0f, 10.0f, 10.0f, 10.0f}, 4, {0, 1, 2, 3}, true);
  CheckSingleTokenGateIndices({10.0f, 10.0f, 10.0f, 10.0f}, 4, {0, 1, 2, 3}, false);
}

}  // namespace onnxruntime::test
#endif

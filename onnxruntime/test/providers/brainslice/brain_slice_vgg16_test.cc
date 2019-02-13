// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/brainslice/brain_slice_execution_provider.h"
#include "core/providers/brainslice/fpga_handle.h"
#include "gtest/gtest.h"
#include "3rdparty/half.hpp"
#include "test/providers/brainslice/brain_slice_test_util.h"
#include "bond_request.h"
#include "bond_response.h"

namespace onnxruntime {
namespace test {

using float16type = half_float::half;

// VGG-16 constants (input: [1,224,224,3]  output: [1,7,7,512]).
const int kPadding = 3;
const int kInputSize = (kPadding + 224 + kPadding) * (kPadding + 224 + kPadding) * 3;
const int kOutputSize = 196 * 128;

static size_t AddPaddingToPayload(const BrainSlice_Parameters& bsParameters, size_t payloadSize) {
  const uint32_t block_dim = bsParameters.NATIVE_DIM;
  return ((payloadSize + block_dim - 1) / block_dim) * block_dim;
}

static Status LoadMatrix(fpga::FPGAHandle& handle, uint32_t addr, uint32_t blocks, ISA_Mem memType) {
  const BrainSlice_Parameters& bsParameters = handle.GetParameters();
  const uint32_t block_dim = bsParameters.NATIVE_DIM;
  return handle.LoadMatrix(std::vector<float16type>(blocks * block_dim * block_dim, float16type(0.0f)), blocks * block_dim, block_dim, addr, true, memType);
}

static Status LoadVector(fpga::FPGAHandle& handle, uint32_t addr, uint32_t blocks, ISA_Mem memType) {
  const BrainSlice_Parameters& bsParameters = handle.GetParameters();
  const uint32_t block_dim = bsParameters.NATIVE_DIM;
  return handle.LoadVector(std::vector<float16type>(blocks * block_dim), addr, memType);
}

// Hardware capability validation - replicated from vgg16.c firmware code.
bool CheckVGG16Compatibility(const BrainSlice_Parameters& bsParameters) {
  if (bsParameters.NATIVE_DIM != 128) return false;
  if (bsParameters.MFUS < 2) return false;
  if (bsParameters.INITIAL_VRF_SIZE < 9204) return false;
  if (bsParameters.MVM_MATRIX_RF_SIZE < 128) return false;
  if (bsParameters.ADDSUB_VRF_0_SIZE < 4) return false;
  if (bsParameters.ADDSUB_VRF_1_SIZE < 3808) return false;
  //if (bsParameters.MULTIPLY_VRF_SIZE < 0) return false;
  if (bsParameters.USE_DRAM == false) return false;
  if (bsParameters.VECTOR_MEM_SIZE < 260) return false;

  return true;
}

TEST(BrainSliceVGG16Test, LoadFirmware) {
  fpga::FPGAInfo info = {0, true, "testdata/firmwares/vgg16/instructions.bin", "testdata/firmwares/vgg16/data.bin", "testdata/firmwares/vgg16/schema.bin"};
  fpga::FPGAHandle handle(info);

  BS_Capabilities capacity;
  ASSERT_TRUE(handle.GetCapabilities(&capacity).IsOK());
  ASSERT_EQ(capacity.m_appId, 1700u);
  ASSERT_EQ(capacity.m_appMajorVersion, 2u);
  ASSERT_EQ(capacity.m_appMinorVersion, 0u);
}

TEST(BrainSliceVGG16Test, Execute_ZeroWeights) {
  fpga::FPGAInfo info = {0, true, "testdata/firmwares/vgg16/instructions.bin", "testdata/firmwares/vgg16/data.bin", "testdata/firmwares/vgg16/schema.bin"};
  fpga::FPGAHandle handle(info);

  const BrainSlice_Parameters& bsParameters = handle.GetParameters();
  if (!CheckVGG16Compatibility(bsParameters))
    return;

  // Load weights/biases (all zero).
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 0, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());      // conv1_1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 2, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());      // conv1_2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 11, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());     // pool1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 12, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());     // conv2_1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 21, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());     // conv2_2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 30, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());     // pool2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 31, /*blocks:*/ 18, ISA_Mem_Dram).IsOK());    // conv3_1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 49, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());    // conv3_2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 85, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());    // conv3_3_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 121, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // pool3_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 122, /*blocks:*/ 72, ISA_Mem_Dram).IsOK());   // conv4_1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 194, /*blocks:*/ 144, ISA_Mem_Dram).IsOK());  // conv4_2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 338, /*blocks:*/ 144, ISA_Mem_Dram).IsOK());  // conv4_3_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 482, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // pool4_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 483, /*blocks:*/ 144, ISA_Mem_Dram).IsOK());  // conv5_1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 627, /*blocks:*/ 144, ISA_Mem_Dram).IsOK());  // conv5_2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 771, /*blocks:*/ 144, ISA_Mem_Dram).IsOK());  // conv5_3_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 915, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // block5_pool_MaxPool_MRF:0

  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 226, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv1_1_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 227, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv1_2_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 228, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv2_1_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 229, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv2_2_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 230, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());  // conv3_1_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 232, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());  // conv3_2_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 234, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());  // conv3_3_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 236, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());  // conv4_1_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 240, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());  // conv4_2_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 244, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());  // conv4_3_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 248, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());  // conv5_1_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 252, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());  // conv5_2_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 256, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());  // conv5_3_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 0, /*blocks:*/ 226, ISA_Mem_Dram).IsOK());  // zeros:0

  // Initialization of VGG-16 model.
  bond_util::BondStruct runtime_argument = CreateRuntimeArguments(0, {});
  auto status = handle.SendSync(
      [&](void* buffer, size_t* size) {
        void* payloadPtr;
        size_t payloadSize = 0;
        return BrainSlice_Request(&bsParameters, &runtime_argument, InitFunctionId, payloadSize, &payloadPtr, buffer, size);
      },
      [&](void* buffer, size_t size) {
        const void* payloadPtr;
        size_t payloadSize = 0;
        return BrainSlice_Response(&bsParameters, buffer, size, &payloadPtr, &payloadSize);
      });
  ASSERT_TRUE(status.IsOK());

  std::vector<float16type> inputs(kInputSize, float16type(1.0f));
  std::vector<float16type> outputs(kOutputSize, float16type(1.0f));

  // Execution of VGG-16 model.
  status = handle.SendSync(
      [&](void* buffer, size_t* size) {
        void* payloadPtr;
        size_t payloadSize = AddPaddingToPayload(bsParameters, inputs.size() * sizeof(float16type));
        auto status = BrainSlice_Request(&bsParameters, &runtime_argument, ExecuteFunctionId, payloadSize, &payloadPtr, buffer, size);
        if (status)
          return status;
        memcpy(payloadPtr, inputs.data(), payloadSize);
        return 0;
      },
      [&](void* buffer, size_t size) {
        const void* payloadPtr;
        size_t payloadSize = outputs.size() * sizeof(float16type);
        auto status = BrainSlice_Response(&bsParameters, buffer, size, &payloadPtr, &payloadSize);
        if (status)
          return status;
        memcpy(outputs.data(), payloadPtr, payloadSize);
        return 0;
      });

  ASSERT_TRUE(status.IsOK());
  for (size_t i = 0; i < outputs.size(); i++)
    ASSERT_EQ(float(outputs[i]), 0.0f);  // All features should be zero if all weights/biases are zero.
}

TEST(BrainSliceVGG16Test, Execute_InvalidInputSize) {
  fpga::FPGAInfo info = {0, true, "testdata/firmwares/vgg16/instructions.bin", "testdata/firmwares/vgg16/data.bin", "testdata/firmwares/vgg16/schema.bin"};
  fpga::FPGAHandle handle(info);

  const BrainSlice_Parameters& bsParameters = handle.GetParameters();
  if (!CheckVGG16Compatibility(bsParameters))
    return;

  // Execution of VGG-16 model with input too small.
  bond_util::BondStruct runtime_argument = CreateRuntimeArguments(0, {});
  auto status = handle.SendSync(
      [&](void* buffer, size_t* size) {
        void* payloadPtr;
        size_t payloadSize = (kInputSize / 2) * sizeof(float16type);
        return BrainSlice_Request(&bsParameters, &runtime_argument, ExecuteFunctionId, payloadSize, &payloadPtr, buffer, size);
      },
      [&](void* buffer, size_t size) {
        const void* payloadPtr;
        size_t payloadSize = 0;
        return BrainSlice_Response(&bsParameters, buffer, size, &payloadPtr, &payloadSize);
      });
  ASSERT_FALSE(status.IsOK());

  // Execution of VGG-16 model with input too large.
  status = handle.SendSync(
      [&](void* buffer, size_t* size) {
        void* payloadPtr;
        size_t payloadSize = (kInputSize * 2) * sizeof(float16type);
        return BrainSlice_Request(&bsParameters, &runtime_argument, ExecuteFunctionId, payloadSize, &payloadPtr, buffer, size);
      },
      [&](void* buffer, size_t size) {
        const void* payloadPtr;
        size_t payloadSize = 0;
        return BrainSlice_Response(&bsParameters, buffer, size, &payloadPtr, &payloadSize);
      });
  ASSERT_FALSE(status.IsOK());
}

}  // namespace test
}  // namespace onnxruntime

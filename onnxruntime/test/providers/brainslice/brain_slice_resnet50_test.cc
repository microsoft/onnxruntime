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

// ResNet-50 constants (input: [1,224,224,3]  output: [1,7,7,2048]).
const int kPadding = 3;
const int kInputSize = (kPadding + 224 + kPadding) * (kPadding + 224 + kPadding) * 3;
const int kOutputSize = 784 * 128;

static size_t AddPaddingToPayload(const BrainSlice_Parameters& bsParameters, size_t payloadSize) {
  const uint32_t block_dim = bsParameters.NATIVE_DIM;
  return ((payloadSize + block_dim - 1) / block_dim) * block_dim;
}

// Hardware capability validation - replicated from resnet50.c firmware code.
bool CheckResNet50Compatibility(const BrainSlice_Parameters& bsParameters) {
  if (bsParameters.NATIVE_DIM != 128) return false;
  if (bsParameters.MFUS < 2) return false;
  if (bsParameters.INITIAL_VRF_SIZE < 9075) return false;
  if (bsParameters.MVM_MATRIX_RF_SIZE < 128) return false;
  if (bsParameters.ADDSUB_VRF_0_SIZE < 16) return false;
  if (bsParameters.ADDSUB_VRF_1_SIZE < 12100) return false;
  if (bsParameters.MULTIPLY_VRF_SIZE < 16) return false;
  if (bsParameters.USE_DRAM == false) return false;
  if (bsParameters.VECTOR_MEM_SIZE < 422) return false;

  return true;
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

TEST(BrainSliceResNet50Test, LoadFirmware) {
  fpga::FPGAInfo info = {0, true, "testdata/firmwares/resnet50/instructions.bin", "testdata/firmwares/resnet50/data.bin", "testdata/firmwares/resnet50/schema.bin"};
  fpga::FPGAHandle handle(info);

  BS_Capabilities capacity;
  ASSERT_TRUE(handle.GetCapabilities(&capacity).IsOK());
  ASSERT_EQ(capacity.m_appId, 1700u);
  ASSERT_EQ(capacity.m_appMajorVersion, 2u);
  ASSERT_EQ(capacity.m_appMinorVersion, 0u);
}

TEST(BrainSliceResNet50Test, Execute_ZeroWeights) {
  fpga::FPGAInfo info = {0, true, "testdata/firmwares/resnet50/instructions.bin", "testdata/firmwares/resnet50/data.bin", "testdata/firmwares/resnet50/schema.bin"};
  fpga::FPGAHandle handle(info);

  const BrainSlice_Parameters& bsParameters = handle.GetParameters();
  if (!CheckResNet50Compatibility(bsParameters))
    return;

  // Load weights/biases (all zero).
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 0, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());       // conv1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 2, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());       // pool1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 3, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());       // res2a_branch1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 5, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());       // res2a_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 6, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());       // res2a_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 15, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());      // res2a_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 17, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());      // res2b_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 19, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());      // res2b_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 28, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());      // res2b_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 30, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());      // res2c_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 32, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());      // res2c_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 41, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());      // res2c_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 43, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());      // res3a_branch1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 51, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());      // res3a_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 53, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());      // res3a_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 62, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());      // res3a_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 66, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());      // res3b_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 70, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());      // res3b_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 79, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());      // res3b_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 83, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());      // res3c_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 87, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());      // res3c_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 96, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());      // res3c_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 100, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());     // res3d_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 104, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());     // res3d_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 113, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());     // res3d_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 117, /*blocks:*/ 32, ISA_Mem_Dram).IsOK());    // res4a_branch1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 149, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());     // res4a_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 157, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());    // res4a_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 193, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());    // res4a_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 209, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());    // res4b_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 225, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());    // res4b_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 261, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());    // res4b_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 277, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());    // res4c_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 293, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());    // res4c_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 329, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());    // res4c_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 345, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());    // res4d_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 361, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());    // res4d_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 397, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());    // res4d_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 413, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());    // res4e_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 429, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());    // res4e_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 465, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());    // res4e_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 481, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());    // res4f_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 497, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());    // res4f_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 533, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());    // res4f_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 549, /*blocks:*/ 128, ISA_Mem_Dram).IsOK());   // res5a_branch1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 677, /*blocks:*/ 32, ISA_Mem_Dram).IsOK());    // res5a_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 709, /*blocks:*/ 144, ISA_Mem_Dram).IsOK());   // res5a_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 853, /*blocks:*/ 64, ISA_Mem_Dram).IsOK());    // res5a_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 917, /*blocks:*/ 64, ISA_Mem_Dram).IsOK());    // res5b_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 981, /*blocks:*/ 144, ISA_Mem_Dram).IsOK());   // res5b_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1125, /*blocks:*/ 64, ISA_Mem_Dram).IsOK());   // res5b_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1189, /*blocks:*/ 64, ISA_Mem_Dram).IsOK());   // res5c_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1253, /*blocks:*/ 144, ISA_Mem_Dram).IsOK());  // res5c_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1397, /*blocks:*/ 64, ISA_Mem_Dram).IsOK());   // res5c_branch2c_MRF:0

  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 0, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());     // bn_conv1_scale__vv_mul__scale_conv1_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 1, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());     // bn_conv1_bias__vv_mul__scale_conv1_scale__vv_add__scale_conv1_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 2, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());     // bn2a_branch1_scale__vv_mul__scale2a_branch1_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 4, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());     // bn2a_branch1_bias__vv_mul__scale2a_branch1_scale__vv_add__scale2a_branch1_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 6, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());     // bn2a_branch2a_scale__vv_mul__scale2a_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 7, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());     // bn2a_branch2a_bias__vv_mul__scale2a_branch2a_scale__vv_add__scale2a_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 8, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());     // bn2a_branch2b_scale__vv_mul__scale2a_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 9, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());     // bn2a_branch2b_bias__vv_mul__scale2a_branch2b_scale__vv_add__scale2a_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 10, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn2a_branch2c_scale__vv_mul__scale2a_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 12, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn2a_branch2c_bias__vv_mul__scale2a_branch2c_scale__vv_add__scale2a_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 14, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // bn2b_branch2a_scale__vv_mul__scale2b_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 15, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // bn2b_branch2a_bias__vv_mul__scale2b_branch2a_scale__vv_add__scale2b_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 16, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // bn2b_branch2b_scale__vv_mul__scale2b_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 17, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // bn2b_branch2b_bias__vv_mul__scale2b_branch2b_scale__vv_add__scale2b_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 18, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn2b_branch2c_scale__vv_mul__scale2b_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 20, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn2b_branch2c_bias__vv_mul__scale2b_branch2c_scale__vv_add__scale2b_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 22, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // bn2c_branch2a_scale__vv_mul__scale2c_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 23, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // bn2c_branch2a_bias__vv_mul__scale2c_branch2a_scale__vv_add__scale2c_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 24, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // bn2c_branch2b_scale__vv_mul__scale2c_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 25, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // bn2c_branch2b_bias__vv_mul__scale2c_branch2b_scale__vv_add__scale2c_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 26, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn2c_branch2c_scale__vv_mul__scale2c_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 28, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn2c_branch2c_bias__vv_mul__scale2c_branch2c_scale__vv_add__scale2c_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 30, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());    // bn3a_branch1_scale__vv_mul__scale3a_branch1_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 34, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());    // bn3a_branch1_bias__vv_mul__scale3a_branch1_scale__vv_add__scale3a_branch1_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 38, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // bn3a_branch2a_scale__vv_mul__scale3a_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 39, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // bn3a_branch2a_bias__vv_mul__scale3a_branch2a_scale__vv_add__scale3a_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 40, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // bn3a_branch2b_scale__vv_mul__scale3a_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 41, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // bn3a_branch2b_bias__vv_mul__scale3a_branch2b_scale__vv_add__scale3a_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 42, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());    // bn3a_branch2c_scale__vv_mul__scale3a_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 46, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());    // bn3a_branch2c_bias__vv_mul__scale3a_branch2c_scale__vv_add__scale3a_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 50, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // bn3b_branch2a_scale__vv_mul__scale3b_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 51, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // bn3b_branch2a_bias__vv_mul__scale3b_branch2a_scale__vv_add__scale3b_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 52, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // bn3b_branch2b_scale__vv_mul__scale3b_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 53, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // bn3b_branch2b_bias__vv_mul__scale3b_branch2b_scale__vv_add__scale3b_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 54, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());    // bn3b_branch2c_scale__vv_mul__scale3b_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 58, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());    // bn3b_branch2c_bias__vv_mul__scale3b_branch2c_scale__vv_add__scale3b_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 62, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // bn3c_branch2a_scale__vv_mul__scale3c_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 63, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // bn3c_branch2a_bias__vv_mul__scale3c_branch2a_scale__vv_add__scale3c_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 64, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // bn3c_branch2b_scale__vv_mul__scale3c_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 65, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // bn3c_branch2b_bias__vv_mul__scale3c_branch2b_scale__vv_add__scale3c_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 66, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());    // bn3c_branch2c_scale__vv_mul__scale3c_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 70, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());    // bn3c_branch2c_bias__vv_mul__scale3c_branch2c_scale__vv_add__scale3c_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 74, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // bn3d_branch2a_scale__vv_mul__scale3d_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 75, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // bn3d_branch2a_bias__vv_mul__scale3d_branch2a_scale__vv_add__scale3d_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 76, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // bn3d_branch2b_scale__vv_mul__scale3d_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 77, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // bn3d_branch2b_bias__vv_mul__scale3d_branch2b_scale__vv_add__scale3d_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 78, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());    // bn3d_branch2c_scale__vv_mul__scale3d_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 82, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());    // bn3d_branch2c_bias__vv_mul__scale3d_branch2c_scale__vv_add__scale3d_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 86, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4a_branch1_scale__vv_mul__scale4a_branch1_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 94, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4a_branch1_bias__vv_mul__scale4a_branch1_scale__vv_add__scale4a_branch1_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 102, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // bn4a_branch2a_scale__vv_mul__scale4a_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 104, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // bn4a_branch2a_bias__vv_mul__scale4a_branch2a_scale__vv_add__scale4a_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 106, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // bn4a_branch2b_scale__vv_mul__scale4a_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 108, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // bn4a_branch2b_bias__vv_mul__scale4a_branch2b_scale__vv_add__scale4a_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 110, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());   // bn4a_branch2c_scale__vv_mul__scale4a_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 118, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());   // bn4a_branch2c_bias__vv_mul__scale4a_branch2c_scale__vv_add__scale4a_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 126, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // bn4b_branch2a_scale__vv_mul__scale4b_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 128, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // bn4b_branch2a_bias__vv_mul__scale4b_branch2a_scale__vv_add__scale4b_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 130, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // bn4b_branch2b_scale__vv_mul__scale4b_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 132, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // bn4b_branch2b_bias__vv_mul__scale4b_branch2b_scale__vv_add__scale4b_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 134, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());   // bn4b_branch2c_scale__vv_mul__scale4b_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 142, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());   // bn4b_branch2c_bias__vv_mul__scale4b_branch2c_scale__vv_add__scale4b_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 150, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // bn4c_branch2a_scale__vv_mul__scale4c_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 152, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // bn4c_branch2a_bias__vv_mul__scale4c_branch2a_scale__vv_add__scale4c_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 154, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // bn4c_branch2b_scale__vv_mul__scale4c_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 156, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // bn4c_branch2b_bias__vv_mul__scale4c_branch2b_scale__vv_add__scale4c_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 158, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());   // bn4c_branch2c_scale__vv_mul__scale4c_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 166, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());   // bn4c_branch2c_bias__vv_mul__scale4c_branch2c_scale__vv_add__scale4c_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 174, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // bn4d_branch2a_scale__vv_mul__scale4d_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 176, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // bn4d_branch2a_bias__vv_mul__scale4d_branch2a_scale__vv_add__scale4d_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 178, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // bn4d_branch2b_scale__vv_mul__scale4d_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 180, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // bn4d_branch2b_bias__vv_mul__scale4d_branch2b_scale__vv_add__scale4d_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 182, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());   // bn4d_branch2c_scale__vv_mul__scale4d_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 190, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());   // bn4d_branch2c_bias__vv_mul__scale4d_branch2c_scale__vv_add__scale4d_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 198, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // bn4e_branch2a_scale__vv_mul__scale4e_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 200, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // bn4e_branch2a_bias__vv_mul__scale4e_branch2a_scale__vv_add__scale4e_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 202, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // bn4e_branch2b_scale__vv_mul__scale4e_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 204, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // bn4e_branch2b_bias__vv_mul__scale4e_branch2b_scale__vv_add__scale4e_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 206, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());   // bn4e_branch2c_scale__vv_mul__scale4e_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 214, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());   // bn4e_branch2c_bias__vv_mul__scale4e_branch2c_scale__vv_add__scale4e_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 222, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // bn4f_branch2a_scale__vv_mul__scale4f_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 224, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // bn4f_branch2a_bias__vv_mul__scale4f_branch2a_scale__vv_add__scale4f_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 226, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // bn4f_branch2b_scale__vv_mul__scale4f_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 228, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // bn4f_branch2b_bias__vv_mul__scale4f_branch2b_scale__vv_add__scale4f_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 230, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());   // bn4f_branch2c_scale__vv_mul__scale4f_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 238, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());   // bn4f_branch2c_bias__vv_mul__scale4f_branch2c_scale__vv_add__scale4f_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 246, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());  // bn5a_branch1_scale__vv_mul__scale5a_branch1_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 262, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());  // bn5a_branch1_bias__vv_mul__scale5a_branch1_scale__vv_add__scale5a_branch1_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 278, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());   // bn5a_branch2a_scale__vv_mul__scale5a_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 282, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());   // bn5a_branch2a_bias__vv_mul__scale5a_branch2a_scale__vv_add__scale5a_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 286, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());   // bn5a_branch2b_scale__vv_mul__scale5a_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 290, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());   // bn5a_branch2b_bias__vv_mul__scale5a_branch2b_scale__vv_add__scale5a_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 294, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());  // bn5a_branch2c_scale__vv_mul__scale5a_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 310, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());  // bn5a_branch2c_bias__vv_mul__scale5a_branch2c_scale__vv_add__scale5a_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 326, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());   // bn5b_branch2a_scale__vv_mul__scale5b_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 330, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());   // bn5b_branch2a_bias__vv_mul__scale5b_branch2a_scale__vv_add__scale5b_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 334, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());   // bn5b_branch2b_scale__vv_mul__scale5b_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 338, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());   // bn5b_branch2b_bias__vv_mul__scale5b_branch2b_scale__vv_add__scale5b_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 342, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());  // bn5b_branch2c_scale__vv_mul__scale5b_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 358, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());  // bn5b_branch2c_bias__vv_mul__scale5b_branch2c_scale__vv_add__scale5b_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 374, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());   // bn5c_branch2a_scale__vv_mul__scale5c_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 378, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());   // bn5c_branch2a_bias__vv_mul__scale5c_branch2a_scale__vv_add__scale5c_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 382, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());   // bn5c_branch2b_scale__vv_mul__scale5c_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 386, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());   // bn5c_branch2b_bias__vv_mul__scale5c_branch2b_scale__vv_add__scale5c_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 390, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());  // bn5c_branch2c_scale__vv_mul__scale5c_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 406, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());  // bn5c_branch2c_bias__vv_mul__scale5c_branch2c_scale__vv_add__scale5c_branch2c_bias:0

  // Initialization of ResNet-50 model.
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

  // Execution of ResNet-50 model.
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

TEST(BrainSliceResNet50Test, Execute_InvalidInputSize) {
  fpga::FPGAInfo info = {0, true, "testdata/firmwares/resnet50/instructions.bin", "testdata/firmwares/resnet50/data.bin", "testdata/firmwares/resnet50/schema.bin"};
  fpga::FPGAHandle handle(info);

  const BrainSlice_Parameters& bsParameters = handle.GetParameters();
  if (!CheckResNet50Compatibility(bsParameters))
    return;

  // Execution of ResNet-50 model with input too small.
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

  // Execution of ResNet-50 model with input too large.
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

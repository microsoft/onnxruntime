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

// ResNet-152 constants (input: [1,224,224,3]  output: [1,7,7,2048]).
const int kPadding = 3;
const int kInputSize = (kPadding + 224 + kPadding) * (kPadding + 224 + kPadding) * 3;
const int kOutputSize = 784 * 128;

static size_t AddPaddingToPayload(const BrainSlice_Parameters& bsParameters, size_t payloadSize) {
  const uint32_t block_dim = bsParameters.NATIVE_DIM;
  return ((payloadSize + block_dim - 1) / block_dim) * block_dim;
}

// Hardware capability validation - replicated from resnet152.c firmware code.
bool CheckResNet152Compatibility(const BrainSlice_Parameters& bsParameters) {
  if (bsParameters.NATIVE_DIM != 128) return false;
  if (bsParameters.MFUS < 2) return false;
  if (bsParameters.INITIAL_VRF_SIZE < 9075) return false;
  if (bsParameters.MVM_MATRIX_RF_SIZE < 128) return false;
  if (bsParameters.ADDSUB_VRF_0_SIZE < 16) return false;
  if (bsParameters.ADDSUB_VRF_1_SIZE < 12100) return false;
  if (bsParameters.MULTIPLY_VRF_SIZE < 16) return false;
  if (bsParameters.USE_DRAM == false) return false;
  if (bsParameters.VECTOR_MEM_SIZE < 1190) return false;

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

TEST(BrainSliceResNet152Test, LoadFirmware) {
  fpga::FPGAInfo info = {0, true, "testdata/firmwares/resnet152/instructions.bin", "testdata/firmwares/resnet152/data.bin", "testdata/firmwares/resnet152/schema.bin"};
  fpga::FPGAHandle handle(info);

  BS_Capabilities capacity;
  ASSERT_TRUE(handle.GetCapabilities(&capacity).IsOK());
  ASSERT_EQ(capacity.m_appId, 1700u);
  ASSERT_EQ(capacity.m_appMajorVersion, 2u);
  ASSERT_EQ(capacity.m_appMinorVersion, 0u);
}

TEST(BrainSliceResNet152Test, Execute_ZeroWeights) {
  fpga::FPGAInfo info = {0, true, "testdata/firmwares/resnet152/instructions.bin", "testdata/firmwares/resnet152/data.bin", "testdata/firmwares/resnet152/schema.bin"};
  fpga::FPGAHandle handle(info);

  const BrainSlice_Parameters& bsParameters = handle.GetParameters();
  if (!CheckResNet152Compatibility(bsParameters))
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
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 66, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());      // res3b1_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 70, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());      // res3b1_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 79, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());      // res3b1_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 83, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());      // res3b2_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 87, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());      // res3b2_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 96, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());      // res3b2_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 100, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());     // res3b3_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 104, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());     // res3b3_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 113, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());     // res3b3_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 117, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());     // res3b4_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 121, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());     // res3b4_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 130, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());     // res3b4_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 134, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());     // res3b5_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 138, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());     // res3b5_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 147, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());     // res3b5_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 151, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());     // res3b6_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 155, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());     // res3b6_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 164, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());     // res3b6_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 168, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());     // res3b7_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 172, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());     // res3b7_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 181, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());     // res3b7_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 185, /*blocks:*/ 32, ISA_Mem_Dram).IsOK());    // res4a_branch1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 217, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());     // res4a_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 225, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());    // res4a_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 261, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());    // res4a_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 277, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());    // res4b1_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 293, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());    // res4b1_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 329, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());    // res4b1_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 345, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());    // res4b2_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 361, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());    // res4b2_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 397, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());    // res4b2_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 413, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());    // res4b3_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 429, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());    // res4b3_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 465, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());    // res4b3_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 481, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());    // res4b4_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 497, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());    // res4b4_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 533, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());    // res4b4_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 549, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());    // res4b5_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 565, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());    // res4b5_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 601, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());    // res4b5_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 617, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());    // res4b6_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 633, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());    // res4b6_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 669, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());    // res4b6_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 685, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());    // res4b7_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 701, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());    // res4b7_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 737, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());    // res4b7_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 753, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());    // res4b8_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 769, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());    // res4b8_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 805, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());    // res4b8_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 821, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());    // res4b9_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 837, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());    // res4b9_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 873, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());    // res4b9_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 889, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());    // res4b10_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 905, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());    // res4b10_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 941, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());    // res4b10_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 957, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());    // res4b11_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 973, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());    // res4b11_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1009, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b11_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1025, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b12_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1041, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());   // res4b12_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1077, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b12_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1093, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b13_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1109, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());   // res4b13_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1145, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b13_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1161, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b14_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1177, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());   // res4b14_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1213, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b14_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1229, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b15_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1245, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());   // res4b15_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1281, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b15_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1297, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b16_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1313, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());   // res4b16_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1349, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b16_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1365, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b17_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1381, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());   // res4b17_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1417, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b17_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1433, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b18_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1449, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());   // res4b18_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1485, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b18_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1501, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b19_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1517, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());   // res4b19_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1553, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b19_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1569, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b20_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1585, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());   // res4b20_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1621, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b20_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1637, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b21_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1653, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());   // res4b21_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1689, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b21_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1705, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b22_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1721, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());   // res4b22_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1757, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b22_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1773, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b23_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1789, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());   // res4b23_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1825, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b23_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1841, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b24_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1857, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());   // res4b24_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1893, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b24_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1909, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b25_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1925, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());   // res4b25_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1961, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b25_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1977, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b26_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 1993, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());   // res4b26_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 2029, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b26_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 2045, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b27_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 2061, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());   // res4b27_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 2097, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b27_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 2113, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b28_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 2129, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());   // res4b28_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 2165, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b28_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 2181, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b29_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 2197, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());   // res4b29_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 2233, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b29_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 2249, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b30_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 2265, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());   // res4b30_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 2301, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b30_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 2317, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b31_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 2333, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());   // res4b31_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 2369, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b31_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 2385, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b32_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 2401, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());   // res4b32_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 2437, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b32_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 2453, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b33_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 2469, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());   // res4b33_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 2505, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b33_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 2521, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b34_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 2537, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());   // res4b34_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 2573, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b34_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 2589, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b35_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 2605, /*blocks:*/ 36, ISA_Mem_Dram).IsOK());   // res4b35_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 2641, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());   // res4b35_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 2657, /*blocks:*/ 128, ISA_Mem_Dram).IsOK());  // res5a_branch1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 2785, /*blocks:*/ 32, ISA_Mem_Dram).IsOK());   // res5a_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 2817, /*blocks:*/ 144, ISA_Mem_Dram).IsOK());  // res5a_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 2961, /*blocks:*/ 64, ISA_Mem_Dram).IsOK());   // res5a_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 3025, /*blocks:*/ 64, ISA_Mem_Dram).IsOK());   // res5b_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 3089, /*blocks:*/ 144, ISA_Mem_Dram).IsOK());  // res5b_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 3233, /*blocks:*/ 64, ISA_Mem_Dram).IsOK());   // res5b_branch2c_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 3297, /*blocks:*/ 64, ISA_Mem_Dram).IsOK());   // res5c_branch2a_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 3361, /*blocks:*/ 144, ISA_Mem_Dram).IsOK());  // res5c_branch2b_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 3505, /*blocks:*/ 64, ISA_Mem_Dram).IsOK());   // res5c_branch2c_MRF:0

  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 0, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());      // bn_conv1_scale__vv_mul__scale_conv1_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 1, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());      // bn_conv1_bias__vv_mul__scale_conv1_scale__vv_add__scale_conv1_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 2, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());      // bn2a_branch1_scale__vv_mul__scale2a_branch1_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 4, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());      // bn2a_branch1_bias__vv_mul__scale2a_branch1_scale__vv_add__scale2a_branch1_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 6, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());      // bn2a_branch2a_scale__vv_mul__scale2a_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 7, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());      // bn2a_branch2a_bias__vv_mul__scale2a_branch2a_scale__vv_add__scale2a_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 8, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());      // bn2a_branch2b_scale__vv_mul__scale2a_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 9, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());      // bn2a_branch2b_bias__vv_mul__scale2a_branch2b_scale__vv_add__scale2a_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 10, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());     // bn2a_branch2c_scale__vv_mul__scale2a_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 12, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());     // bn2a_branch2c_bias__vv_mul__scale2a_branch2c_scale__vv_add__scale2a_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 14, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());     // bn2b_branch2a_scale__vv_mul__scale2b_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 15, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());     // bn2b_branch2a_bias__vv_mul__scale2b_branch2a_scale__vv_add__scale2b_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 16, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());     // bn2b_branch2b_scale__vv_mul__scale2b_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 17, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());     // bn2b_branch2b_bias__vv_mul__scale2b_branch2b_scale__vv_add__scale2b_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 18, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());     // bn2b_branch2c_scale__vv_mul__scale2b_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 20, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());     // bn2b_branch2c_bias__vv_mul__scale2b_branch2c_scale__vv_add__scale2b_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 22, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());     // bn2c_branch2a_scale__vv_mul__scale2c_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 23, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());     // bn2c_branch2a_bias__vv_mul__scale2c_branch2a_scale__vv_add__scale2c_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 24, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());     // bn2c_branch2b_scale__vv_mul__scale2c_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 25, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());     // bn2c_branch2b_bias__vv_mul__scale2c_branch2b_scale__vv_add__scale2c_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 26, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());     // bn2c_branch2c_scale__vv_mul__scale2c_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 28, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());     // bn2c_branch2c_bias__vv_mul__scale2c_branch2c_scale__vv_add__scale2c_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 30, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());     // bn3a_branch1_scale__vv_mul__scale3a_branch1_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 34, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());     // bn3a_branch1_bias__vv_mul__scale3a_branch1_scale__vv_add__scale3a_branch1_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 38, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());     // bn3a_branch2a_scale__vv_mul__scale3a_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 39, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());     // bn3a_branch2a_bias__vv_mul__scale3a_branch2a_scale__vv_add__scale3a_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 40, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());     // bn3a_branch2b_scale__vv_mul__scale3a_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 41, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());     // bn3a_branch2b_bias__vv_mul__scale3a_branch2b_scale__vv_add__scale3a_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 42, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());     // bn3a_branch2c_scale__vv_mul__scale3a_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 46, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());     // bn3a_branch2c_bias__vv_mul__scale3a_branch2c_scale__vv_add__scale3a_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 50, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());     // bn3b1_branch2a_scale__vv_mul__scale3b1_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 51, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());     // bn3b1_branch2a_bias__vv_mul__scale3b1_branch2a_scale__vv_add__scale3b1_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 52, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());     // bn3b1_branch2b_scale__vv_mul__scale3b1_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 53, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());     // bn3b1_branch2b_bias__vv_mul__scale3b1_branch2b_scale__vv_add__scale3b1_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 54, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());     // bn3b1_branch2c_scale__vv_mul__scale3b1_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 58, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());     // bn3b1_branch2c_bias__vv_mul__scale3b1_branch2c_scale__vv_add__scale3b1_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 62, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());     // bn3b2_branch2a_scale__vv_mul__scale3b2_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 63, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());     // bn3b2_branch2a_bias__vv_mul__scale3b2_branch2a_scale__vv_add__scale3b2_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 64, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());     // bn3b2_branch2b_scale__vv_mul__scale3b2_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 65, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());     // bn3b2_branch2b_bias__vv_mul__scale3b2_branch2b_scale__vv_add__scale3b2_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 66, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());     // bn3b2_branch2c_scale__vv_mul__scale3b2_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 70, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());     // bn3b2_branch2c_bias__vv_mul__scale3b2_branch2c_scale__vv_add__scale3b2_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 74, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());     // bn3b3_branch2a_scale__vv_mul__scale3b3_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 75, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());     // bn3b3_branch2a_bias__vv_mul__scale3b3_branch2a_scale__vv_add__scale3b3_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 76, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());     // bn3b3_branch2b_scale__vv_mul__scale3b3_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 77, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());     // bn3b3_branch2b_bias__vv_mul__scale3b3_branch2b_scale__vv_add__scale3b3_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 78, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());     // bn3b3_branch2c_scale__vv_mul__scale3b3_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 82, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());     // bn3b3_branch2c_bias__vv_mul__scale3b3_branch2c_scale__vv_add__scale3b3_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 86, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());     // bn3b4_branch2a_scale__vv_mul__scale3b4_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 87, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());     // bn3b4_branch2a_bias__vv_mul__scale3b4_branch2a_scale__vv_add__scale3b4_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 88, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());     // bn3b4_branch2b_scale__vv_mul__scale3b4_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 89, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());     // bn3b4_branch2b_bias__vv_mul__scale3b4_branch2b_scale__vv_add__scale3b4_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 90, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());     // bn3b4_branch2c_scale__vv_mul__scale3b4_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 94, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());     // bn3b4_branch2c_bias__vv_mul__scale3b4_branch2c_scale__vv_add__scale3b4_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 98, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());     // bn3b5_branch2a_scale__vv_mul__scale3b5_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 99, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());     // bn3b5_branch2a_bias__vv_mul__scale3b5_branch2a_scale__vv_add__scale3b5_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 100, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // bn3b5_branch2b_scale__vv_mul__scale3b5_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 101, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // bn3b5_branch2b_bias__vv_mul__scale3b5_branch2b_scale__vv_add__scale3b5_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 102, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());    // bn3b5_branch2c_scale__vv_mul__scale3b5_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 106, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());    // bn3b5_branch2c_bias__vv_mul__scale3b5_branch2c_scale__vv_add__scale3b5_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 110, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // bn3b6_branch2a_scale__vv_mul__scale3b6_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 111, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // bn3b6_branch2a_bias__vv_mul__scale3b6_branch2a_scale__vv_add__scale3b6_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 112, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // bn3b6_branch2b_scale__vv_mul__scale3b6_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 113, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // bn3b6_branch2b_bias__vv_mul__scale3b6_branch2b_scale__vv_add__scale3b6_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 114, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());    // bn3b6_branch2c_scale__vv_mul__scale3b6_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 118, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());    // bn3b6_branch2c_bias__vv_mul__scale3b6_branch2c_scale__vv_add__scale3b6_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 122, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // bn3b7_branch2a_scale__vv_mul__scale3b7_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 123, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // bn3b7_branch2a_bias__vv_mul__scale3b7_branch2a_scale__vv_add__scale3b7_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 124, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // bn3b7_branch2b_scale__vv_mul__scale3b7_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 125, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // bn3b7_branch2b_bias__vv_mul__scale3b7_branch2b_scale__vv_add__scale3b7_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 126, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());    // bn3b7_branch2c_scale__vv_mul__scale3b7_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 130, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());    // bn3b7_branch2c_bias__vv_mul__scale3b7_branch2c_scale__vv_add__scale3b7_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 134, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4a_branch1_scale__vv_mul__scale4a_branch1_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 142, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4a_branch1_bias__vv_mul__scale4a_branch1_scale__vv_add__scale4a_branch1_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 150, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4a_branch2a_scale__vv_mul__scale4a_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 152, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4a_branch2a_bias__vv_mul__scale4a_branch2a_scale__vv_add__scale4a_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 154, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4a_branch2b_scale__vv_mul__scale4a_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 156, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4a_branch2b_bias__vv_mul__scale4a_branch2b_scale__vv_add__scale4a_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 158, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4a_branch2c_scale__vv_mul__scale4a_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 166, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4a_branch2c_bias__vv_mul__scale4a_branch2c_scale__vv_add__scale4a_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 174, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b1_branch2a_scale__vv_mul__scale4b1_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 176, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b1_branch2a_bias__vv_mul__scale4b1_branch2a_scale__vv_add__scale4b1_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 178, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b1_branch2b_scale__vv_mul__scale4b1_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 180, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b1_branch2b_bias__vv_mul__scale4b1_branch2b_scale__vv_add__scale4b1_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 182, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b1_branch2c_scale__vv_mul__scale4b1_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 190, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b1_branch2c_bias__vv_mul__scale4b1_branch2c_scale__vv_add__scale4b1_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 198, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b2_branch2a_scale__vv_mul__scale4b2_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 200, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b2_branch2a_bias__vv_mul__scale4b2_branch2a_scale__vv_add__scale4b2_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 202, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b2_branch2b_scale__vv_mul__scale4b2_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 204, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b2_branch2b_bias__vv_mul__scale4b2_branch2b_scale__vv_add__scale4b2_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 206, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b2_branch2c_scale__vv_mul__scale4b2_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 214, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b2_branch2c_bias__vv_mul__scale4b2_branch2c_scale__vv_add__scale4b2_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 222, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b3_branch2a_scale__vv_mul__scale4b3_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 224, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b3_branch2a_bias__vv_mul__scale4b3_branch2a_scale__vv_add__scale4b3_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 226, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b3_branch2b_scale__vv_mul__scale4b3_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 228, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b3_branch2b_bias__vv_mul__scale4b3_branch2b_scale__vv_add__scale4b3_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 230, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b3_branch2c_scale__vv_mul__scale4b3_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 238, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b3_branch2c_bias__vv_mul__scale4b3_branch2c_scale__vv_add__scale4b3_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 246, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b4_branch2a_scale__vv_mul__scale4b4_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 248, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b4_branch2a_bias__vv_mul__scale4b4_branch2a_scale__vv_add__scale4b4_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 250, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b4_branch2b_scale__vv_mul__scale4b4_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 252, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b4_branch2b_bias__vv_mul__scale4b4_branch2b_scale__vv_add__scale4b4_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 254, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b4_branch2c_scale__vv_mul__scale4b4_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 262, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b4_branch2c_bias__vv_mul__scale4b4_branch2c_scale__vv_add__scale4b4_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 270, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b5_branch2a_scale__vv_mul__scale4b5_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 272, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b5_branch2a_bias__vv_mul__scale4b5_branch2a_scale__vv_add__scale4b5_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 274, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b5_branch2b_scale__vv_mul__scale4b5_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 276, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b5_branch2b_bias__vv_mul__scale4b5_branch2b_scale__vv_add__scale4b5_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 278, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b5_branch2c_scale__vv_mul__scale4b5_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 286, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b5_branch2c_bias__vv_mul__scale4b5_branch2c_scale__vv_add__scale4b5_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 294, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b6_branch2a_scale__vv_mul__scale4b6_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 296, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b6_branch2a_bias__vv_mul__scale4b6_branch2a_scale__vv_add__scale4b6_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 298, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b6_branch2b_scale__vv_mul__scale4b6_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 300, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b6_branch2b_bias__vv_mul__scale4b6_branch2b_scale__vv_add__scale4b6_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 302, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b6_branch2c_scale__vv_mul__scale4b6_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 310, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b6_branch2c_bias__vv_mul__scale4b6_branch2c_scale__vv_add__scale4b6_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 318, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b7_branch2a_scale__vv_mul__scale4b7_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 320, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b7_branch2a_bias__vv_mul__scale4b7_branch2a_scale__vv_add__scale4b7_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 322, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b7_branch2b_scale__vv_mul__scale4b7_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 324, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b7_branch2b_bias__vv_mul__scale4b7_branch2b_scale__vv_add__scale4b7_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 326, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b7_branch2c_scale__vv_mul__scale4b7_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 334, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b7_branch2c_bias__vv_mul__scale4b7_branch2c_scale__vv_add__scale4b7_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 342, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b8_branch2a_scale__vv_mul__scale4b8_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 344, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b8_branch2a_bias__vv_mul__scale4b8_branch2a_scale__vv_add__scale4b8_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 346, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b8_branch2b_scale__vv_mul__scale4b8_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 348, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b8_branch2b_bias__vv_mul__scale4b8_branch2b_scale__vv_add__scale4b8_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 350, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b8_branch2c_scale__vv_mul__scale4b8_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 358, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b8_branch2c_bias__vv_mul__scale4b8_branch2c_scale__vv_add__scale4b8_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 366, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b9_branch2a_scale__vv_mul__scale4b9_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 368, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b9_branch2a_bias__vv_mul__scale4b9_branch2a_scale__vv_add__scale4b9_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 370, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b9_branch2b_scale__vv_mul__scale4b9_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 372, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b9_branch2b_bias__vv_mul__scale4b9_branch2b_scale__vv_add__scale4b9_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 374, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b9_branch2c_scale__vv_mul__scale4b9_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 382, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b9_branch2c_bias__vv_mul__scale4b9_branch2c_scale__vv_add__scale4b9_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 390, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b10_branch2a_scale__vv_mul__scale4b10_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 392, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b10_branch2a_bias__vv_mul__scale4b10_branch2a_scale__vv_add__scale4b10_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 394, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b10_branch2b_scale__vv_mul__scale4b10_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 396, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b10_branch2b_bias__vv_mul__scale4b10_branch2b_scale__vv_add__scale4b10_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 398, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b10_branch2c_scale__vv_mul__scale4b10_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 406, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b10_branch2c_bias__vv_mul__scale4b10_branch2c_scale__vv_add__scale4b10_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 414, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b11_branch2a_scale__vv_mul__scale4b11_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 416, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b11_branch2a_bias__vv_mul__scale4b11_branch2a_scale__vv_add__scale4b11_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 418, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b11_branch2b_scale__vv_mul__scale4b11_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 420, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b11_branch2b_bias__vv_mul__scale4b11_branch2b_scale__vv_add__scale4b11_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 422, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b11_branch2c_scale__vv_mul__scale4b11_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 430, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b11_branch2c_bias__vv_mul__scale4b11_branch2c_scale__vv_add__scale4b11_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 438, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b12_branch2a_scale__vv_mul__scale4b12_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 440, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b12_branch2a_bias__vv_mul__scale4b12_branch2a_scale__vv_add__scale4b12_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 442, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b12_branch2b_scale__vv_mul__scale4b12_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 444, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b12_branch2b_bias__vv_mul__scale4b12_branch2b_scale__vv_add__scale4b12_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 446, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b12_branch2c_scale__vv_mul__scale4b12_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 454, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b12_branch2c_bias__vv_mul__scale4b12_branch2c_scale__vv_add__scale4b12_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 462, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b13_branch2a_scale__vv_mul__scale4b13_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 464, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b13_branch2a_bias__vv_mul__scale4b13_branch2a_scale__vv_add__scale4b13_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 466, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b13_branch2b_scale__vv_mul__scale4b13_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 468, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b13_branch2b_bias__vv_mul__scale4b13_branch2b_scale__vv_add__scale4b13_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 470, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b13_branch2c_scale__vv_mul__scale4b13_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 478, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b13_branch2c_bias__vv_mul__scale4b13_branch2c_scale__vv_add__scale4b13_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 486, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b14_branch2a_scale__vv_mul__scale4b14_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 488, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b14_branch2a_bias__vv_mul__scale4b14_branch2a_scale__vv_add__scale4b14_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 490, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b14_branch2b_scale__vv_mul__scale4b14_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 492, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b14_branch2b_bias__vv_mul__scale4b14_branch2b_scale__vv_add__scale4b14_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 494, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b14_branch2c_scale__vv_mul__scale4b14_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 502, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b14_branch2c_bias__vv_mul__scale4b14_branch2c_scale__vv_add__scale4b14_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 510, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b15_branch2a_scale__vv_mul__scale4b15_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 512, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b15_branch2a_bias__vv_mul__scale4b15_branch2a_scale__vv_add__scale4b15_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 514, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b15_branch2b_scale__vv_mul__scale4b15_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 516, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b15_branch2b_bias__vv_mul__scale4b15_branch2b_scale__vv_add__scale4b15_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 518, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b15_branch2c_scale__vv_mul__scale4b15_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 526, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b15_branch2c_bias__vv_mul__scale4b15_branch2c_scale__vv_add__scale4b15_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 534, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b16_branch2a_scale__vv_mul__scale4b16_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 536, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b16_branch2a_bias__vv_mul__scale4b16_branch2a_scale__vv_add__scale4b16_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 538, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b16_branch2b_scale__vv_mul__scale4b16_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 540, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b16_branch2b_bias__vv_mul__scale4b16_branch2b_scale__vv_add__scale4b16_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 542, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b16_branch2c_scale__vv_mul__scale4b16_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 550, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b16_branch2c_bias__vv_mul__scale4b16_branch2c_scale__vv_add__scale4b16_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 558, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b17_branch2a_scale__vv_mul__scale4b17_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 560, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b17_branch2a_bias__vv_mul__scale4b17_branch2a_scale__vv_add__scale4b17_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 562, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b17_branch2b_scale__vv_mul__scale4b17_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 564, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b17_branch2b_bias__vv_mul__scale4b17_branch2b_scale__vv_add__scale4b17_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 566, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b17_branch2c_scale__vv_mul__scale4b17_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 574, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b17_branch2c_bias__vv_mul__scale4b17_branch2c_scale__vv_add__scale4b17_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 582, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b18_branch2a_scale__vv_mul__scale4b18_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 584, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b18_branch2a_bias__vv_mul__scale4b18_branch2a_scale__vv_add__scale4b18_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 586, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b18_branch2b_scale__vv_mul__scale4b18_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 588, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b18_branch2b_bias__vv_mul__scale4b18_branch2b_scale__vv_add__scale4b18_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 590, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b18_branch2c_scale__vv_mul__scale4b18_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 598, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b18_branch2c_bias__vv_mul__scale4b18_branch2c_scale__vv_add__scale4b18_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 606, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b19_branch2a_scale__vv_mul__scale4b19_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 608, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b19_branch2a_bias__vv_mul__scale4b19_branch2a_scale__vv_add__scale4b19_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 610, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b19_branch2b_scale__vv_mul__scale4b19_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 612, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b19_branch2b_bias__vv_mul__scale4b19_branch2b_scale__vv_add__scale4b19_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 614, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b19_branch2c_scale__vv_mul__scale4b19_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 622, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b19_branch2c_bias__vv_mul__scale4b19_branch2c_scale__vv_add__scale4b19_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 630, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b20_branch2a_scale__vv_mul__scale4b20_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 632, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b20_branch2a_bias__vv_mul__scale4b20_branch2a_scale__vv_add__scale4b20_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 634, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b20_branch2b_scale__vv_mul__scale4b20_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 636, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b20_branch2b_bias__vv_mul__scale4b20_branch2b_scale__vv_add__scale4b20_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 638, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b20_branch2c_scale__vv_mul__scale4b20_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 646, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b20_branch2c_bias__vv_mul__scale4b20_branch2c_scale__vv_add__scale4b20_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 654, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b21_branch2a_scale__vv_mul__scale4b21_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 656, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b21_branch2a_bias__vv_mul__scale4b21_branch2a_scale__vv_add__scale4b21_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 658, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b21_branch2b_scale__vv_mul__scale4b21_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 660, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b21_branch2b_bias__vv_mul__scale4b21_branch2b_scale__vv_add__scale4b21_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 662, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b21_branch2c_scale__vv_mul__scale4b21_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 670, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b21_branch2c_bias__vv_mul__scale4b21_branch2c_scale__vv_add__scale4b21_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 678, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b22_branch2a_scale__vv_mul__scale4b22_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 680, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b22_branch2a_bias__vv_mul__scale4b22_branch2a_scale__vv_add__scale4b22_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 682, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b22_branch2b_scale__vv_mul__scale4b22_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 684, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b22_branch2b_bias__vv_mul__scale4b22_branch2b_scale__vv_add__scale4b22_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 686, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b22_branch2c_scale__vv_mul__scale4b22_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 694, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b22_branch2c_bias__vv_mul__scale4b22_branch2c_scale__vv_add__scale4b22_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 702, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b23_branch2a_scale__vv_mul__scale4b23_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 704, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b23_branch2a_bias__vv_mul__scale4b23_branch2a_scale__vv_add__scale4b23_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 706, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b23_branch2b_scale__vv_mul__scale4b23_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 708, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b23_branch2b_bias__vv_mul__scale4b23_branch2b_scale__vv_add__scale4b23_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 710, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b23_branch2c_scale__vv_mul__scale4b23_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 718, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b23_branch2c_bias__vv_mul__scale4b23_branch2c_scale__vv_add__scale4b23_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 726, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b24_branch2a_scale__vv_mul__scale4b24_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 728, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b24_branch2a_bias__vv_mul__scale4b24_branch2a_scale__vv_add__scale4b24_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 730, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b24_branch2b_scale__vv_mul__scale4b24_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 732, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b24_branch2b_bias__vv_mul__scale4b24_branch2b_scale__vv_add__scale4b24_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 734, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b24_branch2c_scale__vv_mul__scale4b24_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 742, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b24_branch2c_bias__vv_mul__scale4b24_branch2c_scale__vv_add__scale4b24_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 750, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b25_branch2a_scale__vv_mul__scale4b25_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 752, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b25_branch2a_bias__vv_mul__scale4b25_branch2a_scale__vv_add__scale4b25_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 754, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b25_branch2b_scale__vv_mul__scale4b25_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 756, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b25_branch2b_bias__vv_mul__scale4b25_branch2b_scale__vv_add__scale4b25_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 758, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b25_branch2c_scale__vv_mul__scale4b25_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 766, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b25_branch2c_bias__vv_mul__scale4b25_branch2c_scale__vv_add__scale4b25_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 774, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b26_branch2a_scale__vv_mul__scale4b26_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 776, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b26_branch2a_bias__vv_mul__scale4b26_branch2a_scale__vv_add__scale4b26_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 778, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b26_branch2b_scale__vv_mul__scale4b26_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 780, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b26_branch2b_bias__vv_mul__scale4b26_branch2b_scale__vv_add__scale4b26_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 782, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b26_branch2c_scale__vv_mul__scale4b26_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 790, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b26_branch2c_bias__vv_mul__scale4b26_branch2c_scale__vv_add__scale4b26_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 798, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b27_branch2a_scale__vv_mul__scale4b27_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 800, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b27_branch2a_bias__vv_mul__scale4b27_branch2a_scale__vv_add__scale4b27_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 802, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b27_branch2b_scale__vv_mul__scale4b27_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 804, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b27_branch2b_bias__vv_mul__scale4b27_branch2b_scale__vv_add__scale4b27_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 806, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b27_branch2c_scale__vv_mul__scale4b27_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 814, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b27_branch2c_bias__vv_mul__scale4b27_branch2c_scale__vv_add__scale4b27_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 822, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b28_branch2a_scale__vv_mul__scale4b28_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 824, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b28_branch2a_bias__vv_mul__scale4b28_branch2a_scale__vv_add__scale4b28_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 826, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b28_branch2b_scale__vv_mul__scale4b28_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 828, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b28_branch2b_bias__vv_mul__scale4b28_branch2b_scale__vv_add__scale4b28_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 830, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b28_branch2c_scale__vv_mul__scale4b28_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 838, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b28_branch2c_bias__vv_mul__scale4b28_branch2c_scale__vv_add__scale4b28_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 846, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b29_branch2a_scale__vv_mul__scale4b29_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 848, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b29_branch2a_bias__vv_mul__scale4b29_branch2a_scale__vv_add__scale4b29_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 850, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b29_branch2b_scale__vv_mul__scale4b29_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 852, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b29_branch2b_bias__vv_mul__scale4b29_branch2b_scale__vv_add__scale4b29_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 854, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b29_branch2c_scale__vv_mul__scale4b29_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 862, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b29_branch2c_bias__vv_mul__scale4b29_branch2c_scale__vv_add__scale4b29_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 870, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b30_branch2a_scale__vv_mul__scale4b30_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 872, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b30_branch2a_bias__vv_mul__scale4b30_branch2a_scale__vv_add__scale4b30_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 874, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b30_branch2b_scale__vv_mul__scale4b30_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 876, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b30_branch2b_bias__vv_mul__scale4b30_branch2b_scale__vv_add__scale4b30_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 878, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b30_branch2c_scale__vv_mul__scale4b30_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 886, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b30_branch2c_bias__vv_mul__scale4b30_branch2c_scale__vv_add__scale4b30_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 894, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b31_branch2a_scale__vv_mul__scale4b31_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 896, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b31_branch2a_bias__vv_mul__scale4b31_branch2a_scale__vv_add__scale4b31_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 898, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b31_branch2b_scale__vv_mul__scale4b31_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 900, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b31_branch2b_bias__vv_mul__scale4b31_branch2b_scale__vv_add__scale4b31_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 902, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b31_branch2c_scale__vv_mul__scale4b31_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 910, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b31_branch2c_bias__vv_mul__scale4b31_branch2c_scale__vv_add__scale4b31_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 918, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b32_branch2a_scale__vv_mul__scale4b32_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 920, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b32_branch2a_bias__vv_mul__scale4b32_branch2a_scale__vv_add__scale4b32_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 922, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b32_branch2b_scale__vv_mul__scale4b32_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 924, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b32_branch2b_bias__vv_mul__scale4b32_branch2b_scale__vv_add__scale4b32_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 926, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b32_branch2c_scale__vv_mul__scale4b32_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 934, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b32_branch2c_bias__vv_mul__scale4b32_branch2c_scale__vv_add__scale4b32_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 942, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b33_branch2a_scale__vv_mul__scale4b33_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 944, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b33_branch2a_bias__vv_mul__scale4b33_branch2a_scale__vv_add__scale4b33_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 946, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b33_branch2b_scale__vv_mul__scale4b33_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 948, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b33_branch2b_bias__vv_mul__scale4b33_branch2b_scale__vv_add__scale4b33_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 950, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b33_branch2c_scale__vv_mul__scale4b33_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 958, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b33_branch2c_bias__vv_mul__scale4b33_branch2c_scale__vv_add__scale4b33_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 966, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b34_branch2a_scale__vv_mul__scale4b34_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 968, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b34_branch2a_bias__vv_mul__scale4b34_branch2a_scale__vv_add__scale4b34_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 970, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b34_branch2b_scale__vv_mul__scale4b34_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 972, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b34_branch2b_bias__vv_mul__scale4b34_branch2b_scale__vv_add__scale4b34_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 974, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b34_branch2c_scale__vv_mul__scale4b34_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 982, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b34_branch2c_bias__vv_mul__scale4b34_branch2c_scale__vv_add__scale4b34_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 990, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b35_branch2a_scale__vv_mul__scale4b35_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 992, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b35_branch2a_bias__vv_mul__scale4b35_branch2a_scale__vv_add__scale4b35_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 994, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b35_branch2b_scale__vv_mul__scale4b35_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 996, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // bn4b35_branch2b_bias__vv_mul__scale4b35_branch2b_scale__vv_add__scale4b35_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 998, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());    // bn4b35_branch2c_scale__vv_mul__scale4b35_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 1006, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());   // bn4b35_branch2c_bias__vv_mul__scale4b35_branch2c_scale__vv_add__scale4b35_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 1014, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());  // bn5a_branch1_scale__vv_mul__scale5a_branch1_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 1030, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());  // bn5a_branch1_bias__vv_mul__scale5a_branch1_scale__vv_add__scale5a_branch1_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 1046, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());   // bn5a_branch2a_scale__vv_mul__scale5a_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 1050, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());   // bn5a_branch2a_bias__vv_mul__scale5a_branch2a_scale__vv_add__scale5a_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 1054, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());   // bn5a_branch2b_scale__vv_mul__scale5a_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 1058, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());   // bn5a_branch2b_bias__vv_mul__scale5a_branch2b_scale__vv_add__scale5a_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 1062, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());  // bn5a_branch2c_scale__vv_mul__scale5a_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 1078, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());  // bn5a_branch2c_bias__vv_mul__scale5a_branch2c_scale__vv_add__scale5a_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 1094, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());   // bn5b_branch2a_scale__vv_mul__scale5b_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 1098, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());   // bn5b_branch2a_bias__vv_mul__scale5b_branch2a_scale__vv_add__scale5b_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 1102, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());   // bn5b_branch2b_scale__vv_mul__scale5b_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 1106, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());   // bn5b_branch2b_bias__vv_mul__scale5b_branch2b_scale__vv_add__scale5b_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 1110, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());  // bn5b_branch2c_scale__vv_mul__scale5b_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 1126, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());  // bn5b_branch2c_bias__vv_mul__scale5b_branch2c_scale__vv_add__scale5b_branch2c_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 1142, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());   // bn5c_branch2a_scale__vv_mul__scale5c_branch2a_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 1146, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());   // bn5c_branch2a_bias__vv_mul__scale5c_branch2a_scale__vv_add__scale5c_branch2a_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 1150, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());   // bn5c_branch2b_scale__vv_mul__scale5c_branch2b_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 1154, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());   // bn5c_branch2b_bias__vv_mul__scale5c_branch2b_scale__vv_add__scale5c_branch2b_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 1158, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());  // bn5c_branch2c_scale__vv_mul__scale5c_branch2c_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 1174, /*blocks:*/ 16, ISA_Mem_Dram).IsOK());  // bn5c_branch2c_bias__vv_mul__scale5c_branch2c_scale__vv_add__scale5c_branch2c_bias:0

  // Initialization of ResNet-152 model.
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

  // Execution of ResNet-152 model.
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

TEST(BrainSliceResNet152Test, Execute_InvalidInputSize) {
  fpga::FPGAInfo info = {0, true, "testdata/firmwares/resnet152/instructions.bin", "testdata/firmwares/resnet152/data.bin", "testdata/firmwares/resnet152/schema.bin"};
  fpga::FPGAHandle handle(info);

  const BrainSlice_Parameters& bsParameters = handle.GetParameters();
  if (!CheckResNet152Compatibility(bsParameters))
    return;

  // Execution of ResNet-152 model with input too small.
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

  // Execution of ResNet-152 model with input too large.
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

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

// DenseNet-121 constants (input: [1,224,224,3]  output: [1,6,6,1024]).
const int kPadding = 3;
const int kInputSize = (kPadding + 224 + kPadding) * (kPadding + 224 + kPadding) * 3;
const int kOutputSize = 288 * 128;

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

// Hardware capability validation - replicated from densenet121.c firmware code.
bool CheckDenseNet121Compatibility(const BrainSlice_Parameters& bsParameters) {
  if (bsParameters.NATIVE_DIM != 128) return false;
  if (bsParameters.MFUS < 2) return false;
  if (bsParameters.INITIAL_VRF_SIZE < 12100) return false;
  if (bsParameters.MVM_MATRIX_RF_SIZE < 128) return false;
  if (bsParameters.ADDSUB_VRF_0_SIZE < 8) return false;
  if (bsParameters.ADDSUB_VRF_1_SIZE < 3135) return false;
  if (bsParameters.MULTIPLY_VRF_SIZE < 8) return false;
  if (bsParameters.USE_DRAM == false) return false;
  if (bsParameters.VECTOR_MEM_SIZE < 705) return false;

  return true;
}

TEST(BrainSliceDenseNet121Test, LoadFirmware) {
  fpga::FPGAInfo info = {0, true, "testdata/firmwares/densenet121/instructions.bin", "testdata/firmwares/densenet121/data.bin", "testdata/firmwares/densenet121/schema.bin"};
  fpga::FPGAHandle handle(info);

  BS_Capabilities capacity;
  ASSERT_TRUE(handle.GetCapabilities(&capacity).IsOK());
  ASSERT_EQ(capacity.m_appId, 1700u);
  ASSERT_EQ(capacity.m_appMajorVersion, 2u);
  ASSERT_EQ(capacity.m_appMinorVersion, 0u);
}

TEST(BrainSliceDenseNet121Test, Execute_ZeroWeights) {
  fpga::FPGAInfo info = {0, true, "testdata/firmwares/densenet121/instructions.bin", "testdata/firmwares/densenet121/data.bin", "testdata/firmwares/densenet121/schema.bin"};
  fpga::FPGAHandle handle(info);

  const BrainSlice_Parameters& bsParameters = handle.GetParameters();
  if (!CheckDenseNet121Compatibility(bsParameters))
    return;

  // Load weights/biases (all zero).
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 0, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());     // conv1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 2, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());     // pool1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 3, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());     // dummy_conv_conv2_1_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 4, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());     // conv2_1_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 5, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());     // conv2_1_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 14, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // concat_2_1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 15, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // dummy_conv_conv2_2_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 16, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // conv2_2_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 17, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());    // conv2_2_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 26, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // concat_2_2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 27, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // dummy_conv_conv2_3_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 28, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // conv2_3_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 29, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());    // conv2_3_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 38, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // concat_2_3_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 39, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // dummy_conv_conv2_4_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 40, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // conv2_4_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 42, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());    // conv2_4_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 51, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // concat_2_4_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 52, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // dummy_conv_conv2_5_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 53, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // conv2_5_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 55, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());    // conv2_5_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 64, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // concat_2_5_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 65, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // dummy_conv_conv2_6_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 66, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // conv2_6_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 68, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());    // conv2_6_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 77, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // concat_2_6_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 78, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // dummy_conv_conv2_blk_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 79, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // conv2_blk_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 81, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // pool2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 82, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // dummy_conv_conv3_1_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 83, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // conv3_1_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 84, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());    // conv3_1_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 93, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // concat_3_1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 94, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // dummy_conv_conv3_2_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 95, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());    // conv3_2_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 97, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());    // conv3_2_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 106, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_3_2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 107, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv3_3_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 108, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // conv3_3_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 110, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv3_3_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 119, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_3_3_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 120, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv3_4_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 121, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // conv3_4_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 123, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv3_4_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 132, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_3_4_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 133, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv3_5_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 134, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // conv3_5_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 136, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv3_5_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 145, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_3_5_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 146, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv3_6_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 147, /*blocks:*/ 3, ISA_Mem_Dram).IsOK());   // conv3_6_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 150, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv3_6_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 159, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_3_6_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 160, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv3_7_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 161, /*blocks:*/ 3, ISA_Mem_Dram).IsOK());   // conv3_7_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 164, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv3_7_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 173, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_3_7_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 174, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv3_8_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 175, /*blocks:*/ 3, ISA_Mem_Dram).IsOK());   // conv3_8_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 178, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv3_8_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 187, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_3_8_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 188, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv3_9_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 189, /*blocks:*/ 3, ISA_Mem_Dram).IsOK());   // conv3_9_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 192, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv3_9_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 201, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_3_9_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 202, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv3_10_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 203, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());   // conv3_10_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 207, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv3_10_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 216, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_3_10_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 217, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv3_11_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 218, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());   // conv3_11_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 222, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv3_11_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 231, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_3_11_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 232, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv3_12_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 233, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());   // conv3_12_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 237, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv3_12_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 246, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_3_12_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 247, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv3_blk_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 248, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());   // conv3_blk_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 256, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // pool3_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 257, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv4_1_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 258, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // conv4_1_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 260, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv4_1_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 269, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_4_1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 270, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv4_2_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 271, /*blocks:*/ 3, ISA_Mem_Dram).IsOK());   // conv4_2_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 274, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv4_2_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 283, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_4_2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 284, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv4_3_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 285, /*blocks:*/ 3, ISA_Mem_Dram).IsOK());   // conv4_3_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 288, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv4_3_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 297, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_4_3_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 298, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv4_4_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 299, /*blocks:*/ 3, ISA_Mem_Dram).IsOK());   // conv4_4_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 302, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv4_4_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 311, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_4_4_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 312, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv4_5_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 313, /*blocks:*/ 3, ISA_Mem_Dram).IsOK());   // conv4_5_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 316, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv4_5_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 325, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_4_5_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 326, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv4_6_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 327, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());   // conv4_6_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 331, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv4_6_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 340, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_4_6_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 341, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv4_7_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 342, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());   // conv4_7_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 346, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv4_7_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 355, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_4_7_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 356, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv4_8_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 357, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());   // conv4_8_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 361, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv4_8_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 370, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_4_8_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 371, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv4_9_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 372, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());   // conv4_9_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 376, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv4_9_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 385, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_4_9_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 386, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv4_10_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 387, /*blocks:*/ 5, ISA_Mem_Dram).IsOK());   // conv4_10_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 392, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv4_10_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 401, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_4_10_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 402, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv4_11_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 403, /*blocks:*/ 5, ISA_Mem_Dram).IsOK());   // conv4_11_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 408, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv4_11_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 417, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_4_11_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 418, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv4_12_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 419, /*blocks:*/ 5, ISA_Mem_Dram).IsOK());   // conv4_12_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 424, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv4_12_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 433, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_4_12_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 434, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv4_13_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 435, /*blocks:*/ 5, ISA_Mem_Dram).IsOK());   // conv4_13_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 440, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv4_13_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 449, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_4_13_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 450, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv4_14_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 451, /*blocks:*/ 6, ISA_Mem_Dram).IsOK());   // conv4_14_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 457, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv4_14_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 466, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_4_14_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 467, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv4_15_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 468, /*blocks:*/ 6, ISA_Mem_Dram).IsOK());   // conv4_15_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 474, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv4_15_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 483, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_4_15_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 484, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv4_16_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 485, /*blocks:*/ 6, ISA_Mem_Dram).IsOK());   // conv4_16_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 491, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv4_16_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 500, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_4_16_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 501, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv4_17_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 502, /*blocks:*/ 6, ISA_Mem_Dram).IsOK());   // conv4_17_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 508, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv4_17_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 517, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_4_17_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 518, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv4_18_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 519, /*blocks:*/ 7, ISA_Mem_Dram).IsOK());   // conv4_18_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 526, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv4_18_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 535, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_4_18_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 536, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv4_19_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 537, /*blocks:*/ 7, ISA_Mem_Dram).IsOK());   // conv4_19_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 544, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv4_19_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 553, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_4_19_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 554, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv4_20_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 555, /*blocks:*/ 7, ISA_Mem_Dram).IsOK());   // conv4_20_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 562, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv4_20_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 571, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_4_20_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 572, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv4_21_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 573, /*blocks:*/ 7, ISA_Mem_Dram).IsOK());   // conv4_21_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 580, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv4_21_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 589, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_4_21_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 590, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv4_22_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 591, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());   // conv4_22_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 599, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv4_22_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 608, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_4_22_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 609, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv4_23_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 610, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());   // conv4_23_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 618, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv4_23_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 627, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_4_23_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 628, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv4_24_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 629, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());   // conv4_24_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 637, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv4_24_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 646, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_4_24_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 647, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv4_blk_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 648, /*blocks:*/ 32, ISA_Mem_Dram).IsOK());  // conv4_blk_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 680, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // pool4_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 681, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv5_1_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 682, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());   // conv5_1_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 686, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv5_1_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 695, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_5_1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 696, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv5_2_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 697, /*blocks:*/ 5, ISA_Mem_Dram).IsOK());   // conv5_2_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 702, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv5_2_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 711, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_5_2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 712, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv5_3_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 713, /*blocks:*/ 5, ISA_Mem_Dram).IsOK());   // conv5_3_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 718, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv5_3_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 727, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_5_3_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 728, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv5_4_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 729, /*blocks:*/ 5, ISA_Mem_Dram).IsOK());   // conv5_4_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 734, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv5_4_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 743, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_5_4_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 744, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv5_5_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 745, /*blocks:*/ 5, ISA_Mem_Dram).IsOK());   // conv5_5_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 750, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv5_5_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 759, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_5_5_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 760, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv5_6_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 761, /*blocks:*/ 6, ISA_Mem_Dram).IsOK());   // conv5_6_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 767, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv5_6_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 776, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_5_6_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 777, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv5_7_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 778, /*blocks:*/ 6, ISA_Mem_Dram).IsOK());   // conv5_7_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 784, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv5_7_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 793, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_5_7_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 794, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv5_8_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 795, /*blocks:*/ 6, ISA_Mem_Dram).IsOK());   // conv5_8_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 801, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv5_8_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 810, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_5_8_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 811, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv5_9_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 812, /*blocks:*/ 6, ISA_Mem_Dram).IsOK());   // conv5_9_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 818, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv5_9_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 827, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_5_9_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 828, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv5_10_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 829, /*blocks:*/ 7, ISA_Mem_Dram).IsOK());   // conv5_10_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 836, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv5_10_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 845, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_5_10_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 846, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv5_11_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 847, /*blocks:*/ 7, ISA_Mem_Dram).IsOK());   // conv5_11_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 854, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv5_11_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 863, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_5_11_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 864, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv5_12_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 865, /*blocks:*/ 7, ISA_Mem_Dram).IsOK());   // conv5_12_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 872, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv5_12_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 881, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_5_12_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 882, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv5_13_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 883, /*blocks:*/ 7, ISA_Mem_Dram).IsOK());   // conv5_13_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 890, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv5_13_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 899, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_5_13_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 900, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv5_14_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 901, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());   // conv5_14_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 909, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv5_14_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 918, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_5_14_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 919, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv5_15_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 920, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());   // conv5_15_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 928, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv5_15_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 937, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_5_15_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 938, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv5_16_x1_bn_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 939, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());   // conv5_16_x1_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 947, /*blocks:*/ 9, ISA_Mem_Dram).IsOK());   // conv5_16_x2_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 956, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // concat_5_16_MRF:0
  ASSERT_TRUE(LoadMatrix(handle, /*addr:*/ 957, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // dummy_conv_conv5_blk_bn_MRF:0

  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 36, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // pool2_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 135, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());  // pool3_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 453, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());  // pool4_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 0, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // conv1_bn_scale__vv_mul__conv1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 1, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // conv1_bn_bias__vv_mul__conv1_scale_scale__vv_add__conv1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 2, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // conv2_1_x1_bn_scale__vv_mul__conv2_1_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 3, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // conv2_1_x1_bn_bias__vv_mul__conv2_1_x1_scale_scale__vv_add__conv2_1_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 4, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // conv2_1_x2_bn_scale__vv_mul__conv2_1_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 5, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // conv2_1_x2_bn_bias__vv_mul__conv2_1_x2_scale_scale__vv_add__conv2_1_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 6, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // conv2_2_x1_bn_scale__vv_mul__conv2_2_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 7, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // conv2_2_x1_bn_bias__vv_mul__conv2_2_x1_scale_scale__vv_add__conv2_2_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 8, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // conv2_2_x2_bn_scale__vv_mul__conv2_2_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 9, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());    // conv2_2_x2_bn_bias__vv_mul__conv2_2_x2_scale_scale__vv_add__conv2_2_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 10, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // conv2_3_x1_bn_scale__vv_mul__conv2_3_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 11, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // conv2_3_x1_bn_bias__vv_mul__conv2_3_x1_scale_scale__vv_add__conv2_3_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 12, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // conv2_3_x2_bn_scale__vv_mul__conv2_3_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 13, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // conv2_3_x2_bn_bias__vv_mul__conv2_3_x2_scale_scale__vv_add__conv2_3_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 14, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // conv2_4_x1_bn_scale__vv_mul__conv2_4_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 16, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // conv2_4_x1_bn_bias__vv_mul__conv2_4_x1_scale_scale__vv_add__conv2_4_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 18, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // conv2_4_x2_bn_scale__vv_mul__conv2_4_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 19, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // conv2_4_x2_bn_bias__vv_mul__conv2_4_x2_scale_scale__vv_add__conv2_4_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 20, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // conv2_5_x1_bn_scale__vv_mul__conv2_5_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 22, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // conv2_5_x1_bn_bias__vv_mul__conv2_5_x1_scale_scale__vv_add__conv2_5_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 24, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // conv2_5_x2_bn_scale__vv_mul__conv2_5_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 25, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // conv2_5_x2_bn_bias__vv_mul__conv2_5_x2_scale_scale__vv_add__conv2_5_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 26, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // conv2_6_x1_bn_scale__vv_mul__conv2_6_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 28, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // conv2_6_x1_bn_bias__vv_mul__conv2_6_x1_scale_scale__vv_add__conv2_6_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 30, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // conv2_6_x2_bn_scale__vv_mul__conv2_6_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 31, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // conv2_6_x2_bn_bias__vv_mul__conv2_6_x2_scale_scale__vv_add__conv2_6_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 32, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // conv2_blk_bn_scale__vv_mul__conv2_blk_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 34, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // conv2_blk_bn_bias__vv_mul__conv2_blk_scale_scale__vv_add__conv2_blk_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 37, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // conv3_1_x1_bn_scale__vv_mul__conv3_1_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 38, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // conv3_1_x1_bn_bias__vv_mul__conv3_1_x1_scale_scale__vv_add__conv3_1_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 39, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // conv3_1_x2_bn_scale__vv_mul__conv3_1_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 40, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // conv3_1_x2_bn_bias__vv_mul__conv3_1_x2_scale_scale__vv_add__conv3_1_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 41, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // conv3_2_x1_bn_scale__vv_mul__conv3_2_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 43, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // conv3_2_x1_bn_bias__vv_mul__conv3_2_x1_scale_scale__vv_add__conv3_2_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 45, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // conv3_2_x2_bn_scale__vv_mul__conv3_2_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 46, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // conv3_2_x2_bn_bias__vv_mul__conv3_2_x2_scale_scale__vv_add__conv3_2_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 47, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // conv3_3_x1_bn_scale__vv_mul__conv3_3_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 49, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // conv3_3_x1_bn_bias__vv_mul__conv3_3_x1_scale_scale__vv_add__conv3_3_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 51, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // conv3_3_x2_bn_scale__vv_mul__conv3_3_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 52, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // conv3_3_x2_bn_bias__vv_mul__conv3_3_x2_scale_scale__vv_add__conv3_3_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 53, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // conv3_4_x1_bn_scale__vv_mul__conv3_4_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 55, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // conv3_4_x1_bn_bias__vv_mul__conv3_4_x1_scale_scale__vv_add__conv3_4_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 57, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // conv3_4_x2_bn_scale__vv_mul__conv3_4_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 58, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // conv3_4_x2_bn_bias__vv_mul__conv3_4_x2_scale_scale__vv_add__conv3_4_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 59, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // conv3_5_x1_bn_scale__vv_mul__conv3_5_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 61, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());   // conv3_5_x1_bn_bias__vv_mul__conv3_5_x1_scale_scale__vv_add__conv3_5_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 63, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // conv3_5_x2_bn_scale__vv_mul__conv3_5_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 64, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // conv3_5_x2_bn_bias__vv_mul__conv3_5_x2_scale_scale__vv_add__conv3_5_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 65, /*blocks:*/ 3, ISA_Mem_Dram).IsOK());   // conv3_6_x1_bn_scale__vv_mul__conv3_6_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 68, /*blocks:*/ 3, ISA_Mem_Dram).IsOK());   // conv3_6_x1_bn_bias__vv_mul__conv3_6_x1_scale_scale__vv_add__conv3_6_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 71, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // conv3_6_x2_bn_scale__vv_mul__conv3_6_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 72, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // conv3_6_x2_bn_bias__vv_mul__conv3_6_x2_scale_scale__vv_add__conv3_6_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 73, /*blocks:*/ 3, ISA_Mem_Dram).IsOK());   // conv3_7_x1_bn_scale__vv_mul__conv3_7_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 76, /*blocks:*/ 3, ISA_Mem_Dram).IsOK());   // conv3_7_x1_bn_bias__vv_mul__conv3_7_x1_scale_scale__vv_add__conv3_7_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 79, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // conv3_7_x2_bn_scale__vv_mul__conv3_7_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 80, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // conv3_7_x2_bn_bias__vv_mul__conv3_7_x2_scale_scale__vv_add__conv3_7_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 81, /*blocks:*/ 3, ISA_Mem_Dram).IsOK());   // conv3_8_x1_bn_scale__vv_mul__conv3_8_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 84, /*blocks:*/ 3, ISA_Mem_Dram).IsOK());   // conv3_8_x1_bn_bias__vv_mul__conv3_8_x1_scale_scale__vv_add__conv3_8_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 87, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // conv3_8_x2_bn_scale__vv_mul__conv3_8_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 88, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // conv3_8_x2_bn_bias__vv_mul__conv3_8_x2_scale_scale__vv_add__conv3_8_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 89, /*blocks:*/ 3, ISA_Mem_Dram).IsOK());   // conv3_9_x1_bn_scale__vv_mul__conv3_9_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 92, /*blocks:*/ 3, ISA_Mem_Dram).IsOK());   // conv3_9_x1_bn_bias__vv_mul__conv3_9_x1_scale_scale__vv_add__conv3_9_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 95, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // conv3_9_x2_bn_scale__vv_mul__conv3_9_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 96, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());   // conv3_9_x2_bn_bias__vv_mul__conv3_9_x2_scale_scale__vv_add__conv3_9_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 97, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());   // conv3_10_x1_bn_scale__vv_mul__conv3_10_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 101, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());  // conv3_10_x1_bn_bias__vv_mul__conv3_10_x1_scale_scale__vv_add__conv3_10_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 105, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv3_10_x2_bn_scale__vv_mul__conv3_10_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 106, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv3_10_x2_bn_bias__vv_mul__conv3_10_x2_scale_scale__vv_add__conv3_10_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 107, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());  // conv3_11_x1_bn_scale__vv_mul__conv3_11_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 111, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());  // conv3_11_x1_bn_bias__vv_mul__conv3_11_x1_scale_scale__vv_add__conv3_11_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 115, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv3_11_x2_bn_scale__vv_mul__conv3_11_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 116, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv3_11_x2_bn_bias__vv_mul__conv3_11_x2_scale_scale__vv_add__conv3_11_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 117, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());  // conv3_12_x1_bn_scale__vv_mul__conv3_12_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 121, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());  // conv3_12_x1_bn_bias__vv_mul__conv3_12_x1_scale_scale__vv_add__conv3_12_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 125, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv3_12_x2_bn_scale__vv_mul__conv3_12_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 126, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv3_12_x2_bn_bias__vv_mul__conv3_12_x2_scale_scale__vv_add__conv3_12_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 127, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());  // conv3_blk_bn_scale__vv_mul__conv3_blk_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 131, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());  // conv3_blk_bn_bias__vv_mul__conv3_blk_scale_scale__vv_add__conv3_blk_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 137, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());  // conv4_1_x1_bn_scale__vv_mul__conv4_1_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 139, /*blocks:*/ 2, ISA_Mem_Dram).IsOK());  // conv4_1_x1_bn_bias__vv_mul__conv4_1_x1_scale_scale__vv_add__conv4_1_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 141, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_1_x2_bn_scale__vv_mul__conv4_1_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 142, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_1_x2_bn_bias__vv_mul__conv4_1_x2_scale_scale__vv_add__conv4_1_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 143, /*blocks:*/ 3, ISA_Mem_Dram).IsOK());  // conv4_2_x1_bn_scale__vv_mul__conv4_2_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 146, /*blocks:*/ 3, ISA_Mem_Dram).IsOK());  // conv4_2_x1_bn_bias__vv_mul__conv4_2_x1_scale_scale__vv_add__conv4_2_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 149, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_2_x2_bn_scale__vv_mul__conv4_2_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 150, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_2_x2_bn_bias__vv_mul__conv4_2_x2_scale_scale__vv_add__conv4_2_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 151, /*blocks:*/ 3, ISA_Mem_Dram).IsOK());  // conv4_3_x1_bn_scale__vv_mul__conv4_3_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 154, /*blocks:*/ 3, ISA_Mem_Dram).IsOK());  // conv4_3_x1_bn_bias__vv_mul__conv4_3_x1_scale_scale__vv_add__conv4_3_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 157, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_3_x2_bn_scale__vv_mul__conv4_3_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 158, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_3_x2_bn_bias__vv_mul__conv4_3_x2_scale_scale__vv_add__conv4_3_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 159, /*blocks:*/ 3, ISA_Mem_Dram).IsOK());  // conv4_4_x1_bn_scale__vv_mul__conv4_4_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 162, /*blocks:*/ 3, ISA_Mem_Dram).IsOK());  // conv4_4_x1_bn_bias__vv_mul__conv4_4_x1_scale_scale__vv_add__conv4_4_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 165, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_4_x2_bn_scale__vv_mul__conv4_4_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 166, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_4_x2_bn_bias__vv_mul__conv4_4_x2_scale_scale__vv_add__conv4_4_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 167, /*blocks:*/ 3, ISA_Mem_Dram).IsOK());  // conv4_5_x1_bn_scale__vv_mul__conv4_5_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 170, /*blocks:*/ 3, ISA_Mem_Dram).IsOK());  // conv4_5_x1_bn_bias__vv_mul__conv4_5_x1_scale_scale__vv_add__conv4_5_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 173, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_5_x2_bn_scale__vv_mul__conv4_5_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 174, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_5_x2_bn_bias__vv_mul__conv4_5_x2_scale_scale__vv_add__conv4_5_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 175, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());  // conv4_6_x1_bn_scale__vv_mul__conv4_6_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 179, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());  // conv4_6_x1_bn_bias__vv_mul__conv4_6_x1_scale_scale__vv_add__conv4_6_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 183, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_6_x2_bn_scale__vv_mul__conv4_6_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 184, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_6_x2_bn_bias__vv_mul__conv4_6_x2_scale_scale__vv_add__conv4_6_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 185, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());  // conv4_7_x1_bn_scale__vv_mul__conv4_7_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 189, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());  // conv4_7_x1_bn_bias__vv_mul__conv4_7_x1_scale_scale__vv_add__conv4_7_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 193, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_7_x2_bn_scale__vv_mul__conv4_7_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 194, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_7_x2_bn_bias__vv_mul__conv4_7_x2_scale_scale__vv_add__conv4_7_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 195, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());  // conv4_8_x1_bn_scale__vv_mul__conv4_8_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 199, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());  // conv4_8_x1_bn_bias__vv_mul__conv4_8_x1_scale_scale__vv_add__conv4_8_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 203, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_8_x2_bn_scale__vv_mul__conv4_8_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 204, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_8_x2_bn_bias__vv_mul__conv4_8_x2_scale_scale__vv_add__conv4_8_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 205, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());  // conv4_9_x1_bn_scale__vv_mul__conv4_9_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 209, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());  // conv4_9_x1_bn_bias__vv_mul__conv4_9_x1_scale_scale__vv_add__conv4_9_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 213, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_9_x2_bn_scale__vv_mul__conv4_9_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 214, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_9_x2_bn_bias__vv_mul__conv4_9_x2_scale_scale__vv_add__conv4_9_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 215, /*blocks:*/ 5, ISA_Mem_Dram).IsOK());  // conv4_10_x1_bn_scale__vv_mul__conv4_10_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 220, /*blocks:*/ 5, ISA_Mem_Dram).IsOK());  // conv4_10_x1_bn_bias__vv_mul__conv4_10_x1_scale_scale__vv_add__conv4_10_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 225, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_10_x2_bn_scale__vv_mul__conv4_10_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 226, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_10_x2_bn_bias__vv_mul__conv4_10_x2_scale_scale__vv_add__conv4_10_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 227, /*blocks:*/ 5, ISA_Mem_Dram).IsOK());  // conv4_11_x1_bn_scale__vv_mul__conv4_11_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 232, /*blocks:*/ 5, ISA_Mem_Dram).IsOK());  // conv4_11_x1_bn_bias__vv_mul__conv4_11_x1_scale_scale__vv_add__conv4_11_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 237, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_11_x2_bn_scale__vv_mul__conv4_11_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 238, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_11_x2_bn_bias__vv_mul__conv4_11_x2_scale_scale__vv_add__conv4_11_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 239, /*blocks:*/ 5, ISA_Mem_Dram).IsOK());  // conv4_12_x1_bn_scale__vv_mul__conv4_12_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 244, /*blocks:*/ 5, ISA_Mem_Dram).IsOK());  // conv4_12_x1_bn_bias__vv_mul__conv4_12_x1_scale_scale__vv_add__conv4_12_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 249, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_12_x2_bn_scale__vv_mul__conv4_12_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 250, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_12_x2_bn_bias__vv_mul__conv4_12_x2_scale_scale__vv_add__conv4_12_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 251, /*blocks:*/ 5, ISA_Mem_Dram).IsOK());  // conv4_13_x1_bn_scale__vv_mul__conv4_13_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 256, /*blocks:*/ 5, ISA_Mem_Dram).IsOK());  // conv4_13_x1_bn_bias__vv_mul__conv4_13_x1_scale_scale__vv_add__conv4_13_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 261, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_13_x2_bn_scale__vv_mul__conv4_13_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 262, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_13_x2_bn_bias__vv_mul__conv4_13_x2_scale_scale__vv_add__conv4_13_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 263, /*blocks:*/ 6, ISA_Mem_Dram).IsOK());  // conv4_14_x1_bn_scale__vv_mul__conv4_14_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 269, /*blocks:*/ 6, ISA_Mem_Dram).IsOK());  // conv4_14_x1_bn_bias__vv_mul__conv4_14_x1_scale_scale__vv_add__conv4_14_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 275, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_14_x2_bn_scale__vv_mul__conv4_14_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 276, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_14_x2_bn_bias__vv_mul__conv4_14_x2_scale_scale__vv_add__conv4_14_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 277, /*blocks:*/ 6, ISA_Mem_Dram).IsOK());  // conv4_15_x1_bn_scale__vv_mul__conv4_15_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 283, /*blocks:*/ 6, ISA_Mem_Dram).IsOK());  // conv4_15_x1_bn_bias__vv_mul__conv4_15_x1_scale_scale__vv_add__conv4_15_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 289, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_15_x2_bn_scale__vv_mul__conv4_15_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 290, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_15_x2_bn_bias__vv_mul__conv4_15_x2_scale_scale__vv_add__conv4_15_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 291, /*blocks:*/ 6, ISA_Mem_Dram).IsOK());  // conv4_16_x1_bn_scale__vv_mul__conv4_16_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 297, /*blocks:*/ 6, ISA_Mem_Dram).IsOK());  // conv4_16_x1_bn_bias__vv_mul__conv4_16_x1_scale_scale__vv_add__conv4_16_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 303, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_16_x2_bn_scale__vv_mul__conv4_16_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 304, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_16_x2_bn_bias__vv_mul__conv4_16_x2_scale_scale__vv_add__conv4_16_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 305, /*blocks:*/ 6, ISA_Mem_Dram).IsOK());  // conv4_17_x1_bn_scale__vv_mul__conv4_17_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 311, /*blocks:*/ 6, ISA_Mem_Dram).IsOK());  // conv4_17_x1_bn_bias__vv_mul__conv4_17_x1_scale_scale__vv_add__conv4_17_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 317, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_17_x2_bn_scale__vv_mul__conv4_17_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 318, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_17_x2_bn_bias__vv_mul__conv4_17_x2_scale_scale__vv_add__conv4_17_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 319, /*blocks:*/ 7, ISA_Mem_Dram).IsOK());  // conv4_18_x1_bn_scale__vv_mul__conv4_18_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 326, /*blocks:*/ 7, ISA_Mem_Dram).IsOK());  // conv4_18_x1_bn_bias__vv_mul__conv4_18_x1_scale_scale__vv_add__conv4_18_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 333, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_18_x2_bn_scale__vv_mul__conv4_18_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 334, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_18_x2_bn_bias__vv_mul__conv4_18_x2_scale_scale__vv_add__conv4_18_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 335, /*blocks:*/ 7, ISA_Mem_Dram).IsOK());  // conv4_19_x1_bn_scale__vv_mul__conv4_19_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 342, /*blocks:*/ 7, ISA_Mem_Dram).IsOK());  // conv4_19_x1_bn_bias__vv_mul__conv4_19_x1_scale_scale__vv_add__conv4_19_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 349, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_19_x2_bn_scale__vv_mul__conv4_19_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 350, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_19_x2_bn_bias__vv_mul__conv4_19_x2_scale_scale__vv_add__conv4_19_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 351, /*blocks:*/ 7, ISA_Mem_Dram).IsOK());  // conv4_20_x1_bn_scale__vv_mul__conv4_20_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 358, /*blocks:*/ 7, ISA_Mem_Dram).IsOK());  // conv4_20_x1_bn_bias__vv_mul__conv4_20_x1_scale_scale__vv_add__conv4_20_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 365, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_20_x2_bn_scale__vv_mul__conv4_20_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 366, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_20_x2_bn_bias__vv_mul__conv4_20_x2_scale_scale__vv_add__conv4_20_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 367, /*blocks:*/ 7, ISA_Mem_Dram).IsOK());  // conv4_21_x1_bn_scale__vv_mul__conv4_21_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 374, /*blocks:*/ 7, ISA_Mem_Dram).IsOK());  // conv4_21_x1_bn_bias__vv_mul__conv4_21_x1_scale_scale__vv_add__conv4_21_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 381, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_21_x2_bn_scale__vv_mul__conv4_21_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 382, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_21_x2_bn_bias__vv_mul__conv4_21_x2_scale_scale__vv_add__conv4_21_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 383, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());  // conv4_22_x1_bn_scale__vv_mul__conv4_22_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 391, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());  // conv4_22_x1_bn_bias__vv_mul__conv4_22_x1_scale_scale__vv_add__conv4_22_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 399, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_22_x2_bn_scale__vv_mul__conv4_22_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 400, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_22_x2_bn_bias__vv_mul__conv4_22_x2_scale_scale__vv_add__conv4_22_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 401, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());  // conv4_23_x1_bn_scale__vv_mul__conv4_23_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 409, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());  // conv4_23_x1_bn_bias__vv_mul__conv4_23_x1_scale_scale__vv_add__conv4_23_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 417, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_23_x2_bn_scale__vv_mul__conv4_23_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 418, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_23_x2_bn_bias__vv_mul__conv4_23_x2_scale_scale__vv_add__conv4_23_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 419, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());  // conv4_24_x1_bn_scale__vv_mul__conv4_24_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 427, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());  // conv4_24_x1_bn_bias__vv_mul__conv4_24_x1_scale_scale__vv_add__conv4_24_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 435, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_24_x2_bn_scale__vv_mul__conv4_24_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 436, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv4_24_x2_bn_bias__vv_mul__conv4_24_x2_scale_scale__vv_add__conv4_24_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 437, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());  // conv4_blk_bn_scale__vv_mul__conv4_blk_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 445, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());  // conv4_blk_bn_bias__vv_mul__conv4_blk_scale_scale__vv_add__conv4_blk_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 457, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());  // conv5_1_x1_bn_scale__vv_mul__conv5_1_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 461, /*blocks:*/ 4, ISA_Mem_Dram).IsOK());  // conv5_1_x1_bn_bias__vv_mul__conv5_1_x1_scale_scale__vv_add__conv5_1_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 465, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv5_1_x2_bn_scale__vv_mul__conv5_1_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 466, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv5_1_x2_bn_bias__vv_mul__conv5_1_x2_scale_scale__vv_add__conv5_1_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 467, /*blocks:*/ 5, ISA_Mem_Dram).IsOK());  // conv5_2_x1_bn_scale__vv_mul__conv5_2_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 472, /*blocks:*/ 5, ISA_Mem_Dram).IsOK());  // conv5_2_x1_bn_bias__vv_mul__conv5_2_x1_scale_scale__vv_add__conv5_2_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 477, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv5_2_x2_bn_scale__vv_mul__conv5_2_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 478, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv5_2_x2_bn_bias__vv_mul__conv5_2_x2_scale_scale__vv_add__conv5_2_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 479, /*blocks:*/ 5, ISA_Mem_Dram).IsOK());  // conv5_3_x1_bn_scale__vv_mul__conv5_3_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 484, /*blocks:*/ 5, ISA_Mem_Dram).IsOK());  // conv5_3_x1_bn_bias__vv_mul__conv5_3_x1_scale_scale__vv_add__conv5_3_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 489, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv5_3_x2_bn_scale__vv_mul__conv5_3_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 490, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv5_3_x2_bn_bias__vv_mul__conv5_3_x2_scale_scale__vv_add__conv5_3_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 491, /*blocks:*/ 5, ISA_Mem_Dram).IsOK());  // conv5_4_x1_bn_scale__vv_mul__conv5_4_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 496, /*blocks:*/ 5, ISA_Mem_Dram).IsOK());  // conv5_4_x1_bn_bias__vv_mul__conv5_4_x1_scale_scale__vv_add__conv5_4_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 501, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv5_4_x2_bn_scale__vv_mul__conv5_4_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 502, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv5_4_x2_bn_bias__vv_mul__conv5_4_x2_scale_scale__vv_add__conv5_4_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 503, /*blocks:*/ 5, ISA_Mem_Dram).IsOK());  // conv5_5_x1_bn_scale__vv_mul__conv5_5_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 508, /*blocks:*/ 5, ISA_Mem_Dram).IsOK());  // conv5_5_x1_bn_bias__vv_mul__conv5_5_x1_scale_scale__vv_add__conv5_5_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 513, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv5_5_x2_bn_scale__vv_mul__conv5_5_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 514, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv5_5_x2_bn_bias__vv_mul__conv5_5_x2_scale_scale__vv_add__conv5_5_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 515, /*blocks:*/ 6, ISA_Mem_Dram).IsOK());  // conv5_6_x1_bn_scale__vv_mul__conv5_6_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 521, /*blocks:*/ 6, ISA_Mem_Dram).IsOK());  // conv5_6_x1_bn_bias__vv_mul__conv5_6_x1_scale_scale__vv_add__conv5_6_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 527, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv5_6_x2_bn_scale__vv_mul__conv5_6_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 528, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv5_6_x2_bn_bias__vv_mul__conv5_6_x2_scale_scale__vv_add__conv5_6_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 529, /*blocks:*/ 6, ISA_Mem_Dram).IsOK());  // conv5_7_x1_bn_scale__vv_mul__conv5_7_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 535, /*blocks:*/ 6, ISA_Mem_Dram).IsOK());  // conv5_7_x1_bn_bias__vv_mul__conv5_7_x1_scale_scale__vv_add__conv5_7_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 541, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv5_7_x2_bn_scale__vv_mul__conv5_7_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 542, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv5_7_x2_bn_bias__vv_mul__conv5_7_x2_scale_scale__vv_add__conv5_7_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 543, /*blocks:*/ 6, ISA_Mem_Dram).IsOK());  // conv5_8_x1_bn_scale__vv_mul__conv5_8_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 549, /*blocks:*/ 6, ISA_Mem_Dram).IsOK());  // conv5_8_x1_bn_bias__vv_mul__conv5_8_x1_scale_scale__vv_add__conv5_8_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 555, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv5_8_x2_bn_scale__vv_mul__conv5_8_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 556, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv5_8_x2_bn_bias__vv_mul__conv5_8_x2_scale_scale__vv_add__conv5_8_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 557, /*blocks:*/ 6, ISA_Mem_Dram).IsOK());  // conv5_9_x1_bn_scale__vv_mul__conv5_9_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 563, /*blocks:*/ 6, ISA_Mem_Dram).IsOK());  // conv5_9_x1_bn_bias__vv_mul__conv5_9_x1_scale_scale__vv_add__conv5_9_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 569, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv5_9_x2_bn_scale__vv_mul__conv5_9_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 570, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv5_9_x2_bn_bias__vv_mul__conv5_9_x2_scale_scale__vv_add__conv5_9_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 571, /*blocks:*/ 7, ISA_Mem_Dram).IsOK());  // conv5_10_x1_bn_scale__vv_mul__conv5_10_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 578, /*blocks:*/ 7, ISA_Mem_Dram).IsOK());  // conv5_10_x1_bn_bias__vv_mul__conv5_10_x1_scale_scale__vv_add__conv5_10_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 585, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv5_10_x2_bn_scale__vv_mul__conv5_10_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 586, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv5_10_x2_bn_bias__vv_mul__conv5_10_x2_scale_scale__vv_add__conv5_10_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 587, /*blocks:*/ 7, ISA_Mem_Dram).IsOK());  // conv5_11_x1_bn_scale__vv_mul__conv5_11_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 594, /*blocks:*/ 7, ISA_Mem_Dram).IsOK());  // conv5_11_x1_bn_bias__vv_mul__conv5_11_x1_scale_scale__vv_add__conv5_11_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 601, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv5_11_x2_bn_scale__vv_mul__conv5_11_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 602, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv5_11_x2_bn_bias__vv_mul__conv5_11_x2_scale_scale__vv_add__conv5_11_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 603, /*blocks:*/ 7, ISA_Mem_Dram).IsOK());  // conv5_12_x1_bn_scale__vv_mul__conv5_12_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 610, /*blocks:*/ 7, ISA_Mem_Dram).IsOK());  // conv5_12_x1_bn_bias__vv_mul__conv5_12_x1_scale_scale__vv_add__conv5_12_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 617, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv5_12_x2_bn_scale__vv_mul__conv5_12_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 618, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv5_12_x2_bn_bias__vv_mul__conv5_12_x2_scale_scale__vv_add__conv5_12_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 619, /*blocks:*/ 7, ISA_Mem_Dram).IsOK());  // conv5_13_x1_bn_scale__vv_mul__conv5_13_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 626, /*blocks:*/ 7, ISA_Mem_Dram).IsOK());  // conv5_13_x1_bn_bias__vv_mul__conv5_13_x1_scale_scale__vv_add__conv5_13_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 633, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv5_13_x2_bn_scale__vv_mul__conv5_13_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 634, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv5_13_x2_bn_bias__vv_mul__conv5_13_x2_scale_scale__vv_add__conv5_13_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 635, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());  // conv5_14_x1_bn_scale__vv_mul__conv5_14_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 643, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());  // conv5_14_x1_bn_bias__vv_mul__conv5_14_x1_scale_scale__vv_add__conv5_14_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 651, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv5_14_x2_bn_scale__vv_mul__conv5_14_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 652, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv5_14_x2_bn_bias__vv_mul__conv5_14_x2_scale_scale__vv_add__conv5_14_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 653, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());  // conv5_15_x1_bn_scale__vv_mul__conv5_15_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 661, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());  // conv5_15_x1_bn_bias__vv_mul__conv5_15_x1_scale_scale__vv_add__conv5_15_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 669, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv5_15_x2_bn_scale__vv_mul__conv5_15_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 670, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv5_15_x2_bn_bias__vv_mul__conv5_15_x2_scale_scale__vv_add__conv5_15_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 671, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());  // conv5_16_x1_bn_scale__vv_mul__conv5_16_x1_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 679, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());  // conv5_16_x1_bn_bias__vv_mul__conv5_16_x1_scale_scale__vv_add__conv5_16_x1_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 687, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv5_16_x2_bn_scale__vv_mul__conv5_16_x2_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 688, /*blocks:*/ 1, ISA_Mem_Dram).IsOK());  // conv5_16_x2_bn_bias__vv_mul__conv5_16_x2_scale_scale__vv_add__conv5_16_x2_scale_bias:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 689, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());  // conv5_blk_bn_scale__vv_mul__conv5_blk_scale_scale:0
  ASSERT_TRUE(LoadVector(handle, /*addr:*/ 697, /*blocks:*/ 8, ISA_Mem_Dram).IsOK());  // conv5_blk_bn_bias__vv_mul__conv5_blk_scale_scale__vv_add__conv5_blk_scale_bias:0

  // Initialization of DenseNet-121 model.
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

  // Execution of DenseNet-121 model.
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

TEST(BrainSliceDenseNet121Test, Execute_InvalidInputSize) {
  fpga::FPGAInfo info = {0, true, "testdata/firmwares/densenet121/instructions.bin", "testdata/firmwares/densenet121/data.bin", "testdata/firmwares/densenet121/schema.bin"};
  fpga::FPGAHandle handle(info);

  const BrainSlice_Parameters& bsParameters = handle.GetParameters();
  if (!CheckDenseNet121Compatibility(bsParameters))
    return;

  // Execution of DenseNet-121 model with input too small.
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

  // Execution of DenseNet-121 model with input too large.
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

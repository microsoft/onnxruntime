// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/brainslice/brain_slice_execution_provider.h"
#include "core/providers/brainslice/fpga_handle.h"
#include "gtest/gtest.h"
#include "3rdparty/half.hpp"
#include "loopback_client.h"
#include "Lstm_client.h"

namespace onnxruntime {
namespace test {

TEST(BrainSliceBasicTest, MvMulTest) {
  fpga::FPGAInfo info = {0, true, "testdata/firmwares/onnx_rnns/instructions.bin", "testdata/firmwares/onnx_rnns/data.bin", "testdata/firmwares/onnx_rnns/schema.bin"};
  fpga::FPGAHandle handle(info);

  const BrainSlice_Parameters& bsParameters = handle.GetParameters();

  //1. prepare an 400 * 200 matrix
  int row = 400, col = 200;
  typedef half_float::half float16type;
  std::vector<float16type> half_m(row * col, float16type(1.0f));
  auto status = handle.LoadMatrix(half_m, row, col, 0, true, ISA_Mem_MatrixRf);
  EXPECT_TRUE(status.IsOK());

  std::vector<float16type> half_x(col, float16type(1.0f));

  BS_MVMultiplyParams param;
  param.numCols = col;
  param.numRows = row;
  param.startMaddr = 0;
  param.useDram = false;

  std::vector<float> result;

  status = handle.SendSync(
      [&](void* buffer, size_t* size) {
        return BS_CommonFunctions_MatrixVectorMultiply_Request_Float16(&bsParameters, &param, half_x.data(), buffer, size);
      },
      [&](void* buffer, size_t size) {
        const void* output;
        size_t n_output;
        auto status = BS_CommonFunctions_MatrixVectorMultiply_Response_Float16(&bsParameters, &param, buffer, size, &output, &n_output);
        if (status)
          return status;
        const float16type* out = static_cast<const float16type*>(output);
        for (size_t i = 0; i < n_output; i++)
          result.push_back(float(out[i]));
        return status;
      });
  EXPECT_TRUE(status.IsOK());
  EXPECT_EQ(result.size(), 400);
  for (auto f : result)
    EXPECT_EQ(f, 200.f);
}

TEST(BrainSliceBasicTest, LoopBackTest) {
  fpga::FPGAInfo info = {0, true, "testdata/firmwares/loopback/instructions.bin", "testdata/firmwares/loopback/data.bin", "testdata/firmwares/loopback/schema.bin"};
  fpga::FPGAHandle handle(info);

  const BrainSlice_Parameters& bsParameters = handle.GetParameters();

  using float16type = half_float::half;
  size_t native_dim = bsParameters.NATIVE_DIM;
  std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float16type> half_x;
  size_t expect_dim = (x.size() + native_dim - 1) / native_dim;
  for (auto f : x)
    half_x.push_back(float16type(f));
  for (size_t i = half_x.size(); i < (expect_dim * native_dim); i++)
    half_x.push_back(float16type());
  void* input_ptr = &half_x[0];

  Example_Param param = {true, 6};
  Example_Result result;
  size_t result_size = sizeof(Example_Result);
  std::vector<float> outputs;
  auto status = handle.SendSync(
      [&](void* buffer, size_t* size) { return Example_Model_Loopback_Request_Float16(&bsParameters, &param, input_ptr, buffer, size); },
      [&](void* buffer, size_t size) {
        const void* output;
        size_t count = 0;
        auto status = Example_Model_Loopback_Response_Float16(&bsParameters, &param, buffer, size, &result, &result_size, &output, &count);
        if (status)
          return status;
        const float16type* val = static_cast<const float16type*>(output);
        for (size_t i = 0; i < count; i++) {
          outputs.push_back(float(val[i]));
        }
        return status;
      });
  EXPECT_TRUE(status.IsOK());
  EXPECT_EQ(outputs.size(), x.size());
  for (size_t i = 0; i < outputs.size(); i++)
    EXPECT_EQ(outputs[i], x[i]);
}

}  // namespace test
}  // namespace onnxruntime

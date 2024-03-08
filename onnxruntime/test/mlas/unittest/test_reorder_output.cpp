// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_util.h"

class MlasReorderOutputTest : public MlasTestBase {
 private:
  const size_t BlockSize = MlasNchwcGetBlockSize();

  MatrixGuardBuffer<float> BufferInput;
  MatrixGuardBuffer<float> BufferOutput;
  MatrixGuardBuffer<float> BufferOutput2;
  MatrixGuardBuffer<float> BufferOutputReference;

  void Test(size_t BatchCount, size_t Channels, size_t Height, size_t Width) {
    size_t NchwcChannels = (Channels + BlockSize - 1) & ~(BlockSize - 1);

    size_t InputBufferElements = BatchCount * NchwcChannels * Height * Width;
    size_t OutputBufferElements = BatchCount * Channels * Height * Width;

    const float* Input = BufferInput.GetBuffer(InputBufferElements);
    float* Output = BufferOutput.GetBuffer(OutputBufferElements);
    float* OutputReference = BufferOutputReference.GetBuffer(OutputBufferElements);

    int64_t NchwOutputShape[] = {int64_t(BatchCount), int64_t(Channels), int64_t(Height), int64_t(Width)};

    std::fill_n(Output, OutputBufferElements, -0.5f);
    std::fill_n(OutputReference, OutputBufferElements, -0.5f);

    MlasReorderOutputNchw(NchwOutputShape, Input, Output, GetMlasThreadPool());
    ReferenceReorderOutput(BatchCount, Channels, Height, Width, Input, OutputReference, false);
    ASSERT_EQ(memcmp(Output, OutputReference, OutputBufferElements * sizeof(float)), 0)
        << " [Nchw] batch=" << BatchCount << ", channels=" << Channels
        << ", height=" << Height << ", width=" << Width;

    int64_t NhwcOutputShape[] = {int64_t(BatchCount), int64_t(Height), int64_t(Width), int64_t(Channels)};

    std::fill_n(Output, OutputBufferElements, -0.5f);
    std::fill_n(OutputReference, OutputBufferElements, -0.5f);

    MlasReorderOutputNhwc(NhwcOutputShape, Input, Output);
    ReferenceReorderOutput(BatchCount, Channels, Height, Width, Input, OutputReference, true);
    ASSERT_EQ(memcmp(Output, OutputReference, OutputBufferElements * sizeof(float)), 0)
        << " [Nhwc] batch=" << BatchCount << ", channels=" << Channels
        << ", height=" << Height << ", width=" << Width;
  }

  void ReferenceReorderOutput(size_t BatchCount,
                              size_t Channels,
                              size_t Height,
                              size_t Width,
                              const float* Input,
                              float* Output,
                              bool NhwcFormat) {
    size_t NchwcChannels = (Channels + (BlockSize - 1)) & ~(BlockSize - 1);
    size_t SpatialSize = Height * Width;

    size_t ChannelStride = NhwcFormat ? 1 : SpatialSize;
    size_t SpatialStride = NhwcFormat ? Channels : 1;

    for (size_t n = 0; n < BatchCount; n++) {
      for (size_t c = 0; c < Channels; c++) {
        const float* input = Input + ((c & ~(BlockSize - 1)) * SpatialSize) + (c & (BlockSize - 1));
        float* output = Output + (c * ChannelStride);

        for (size_t hw = 0; hw < SpatialSize; hw++) {
          output[hw * SpatialStride] = input[hw * BlockSize];
        }
      }

      Input += NchwcChannels * SpatialSize;
      Output += Channels * SpatialSize;
    }
  }

 public:
  static const char* GetTestSuiteName() {
    static const std::string suite_name("ReorderOutput");
    return suite_name.c_str();
  }

  void ExecuteShort(void) override {
    for (size_t c = 1; c < 48; c++) {
      Test(1, c, 112, 112);
      Test(4, c, 15, 21);
      Test(16, c, 11, 11);
    }
  }
};

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  return (MlasNchwcGetBlockSize() > 1 && is_short_execute)
             ? MlasDirectShortExecuteTests<MlasReorderOutputTest>::RegisterShortExecute()
             : 0;
});

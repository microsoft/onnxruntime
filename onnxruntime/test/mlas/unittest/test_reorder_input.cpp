// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_util.h"

// Exercises MlasReorderInputNchw (NCHW -> NCHWc). On AVX-512 the block size is
// 16 and a full 16-channel block takes the MlasReorderInputNchwBlock16Avx512F
// fast path; this test asserts that path (and the scalar tail path for partial
// blocks) is bit-exact with a plain scalar reference.
class MlasReorderInputTest : public MlasTestBase {
 private:
  const size_t BlockSize = MlasNchwcGetBlockSize();

  MatrixGuardBuffer<float> BufferInput;
  MatrixGuardBuffer<float> BufferOutput;
  MatrixGuardBuffer<float> BufferOutputReference;

  void Test(size_t Channels, size_t Height, size_t Width) {
    const size_t InputSize = Height * Width;
    const size_t NchwcChannels = (Channels + BlockSize - 1) & ~(BlockSize - 1);

    // MlasReorderInputNchw gathers input channels four at a time, so it reads
    // up to the next multiple of four channels; allocate the source rounded up
    // accordingly (the extra channels are zero and land in NCHWc padding lanes).
    const size_t PaddedInputChannels = (Channels + 3) & ~size_t(3);
    const size_t InputBufferElements = PaddedInputChannels * InputSize;
    const size_t OutputBufferElements = NchwcChannels * InputSize;

    float* Input = BufferInput.GetBuffer(InputBufferElements);
    float* Output = BufferOutput.GetBuffer(OutputBufferElements);
    float* OutputReference = BufferOutputReference.GetBuffer(OutputBufferElements);

    // Zero the padding channels [Channels, PaddedInputChannels) so the gather of
    // a partial group reads defined values and the reference agrees.
    for (size_t c = Channels; c < PaddedInputChannels; c++) {
      std::fill_n(Input + c * InputSize, InputSize, 0.0f);
    }

    // Padding lanes of a partial trailing block must be written as zero by the
    // routine; seed both buffers with a sentinel so a missed zero-fill fails.
    std::fill_n(Output, OutputBufferElements, -0.5f);
    std::fill_n(OutputReference, OutputBufferElements, -0.5f);

    MlasReorderInputNchw(Input, Output, Channels, InputSize);
    ReferenceReorderInput(Channels, InputSize, Input, OutputReference);

    ASSERT_EQ(memcmp(Output, OutputReference, OutputBufferElements * sizeof(float)), 0)
        << " channels=" << Channels << ", height=" << Height << ", width=" << Width;
  }

  // NCHW source [Channels][InputSize] -> NCHWc dest, laid out as blocks of
  // BlockSize channels, each block storing [InputSize][BlockSize]. Channels
  // beyond the real count in a trailing block are zero-padded.
  void ReferenceReorderInput(size_t Channels,
                             size_t InputSize,
                             const float* Input,
                             float* Output) {
    const size_t NumBlocks = (Channels + BlockSize - 1) / BlockSize;

    for (size_t b = 0; b < NumBlocks; b++) {
      float* block = Output + b * BlockSize * InputSize;
      for (size_t hw = 0; hw < InputSize; hw++) {
        for (size_t c = 0; c < BlockSize; c++) {
          const size_t channel = b * BlockSize + c;
          block[hw * BlockSize + c] =
              (channel < Channels) ? Input[channel * InputSize + hw] : 0.0f;
        }
      }
    }
  }

 public:
  static const char* GetTestSuiteName() {
    static const std::string suite_name("ReorderInput");
    return suite_name.c_str();
  }

  void ExecuteShort(void) override {
    // Channel counts span exact-block multiples (16, 32, 48 -> AVX-512 fast
    // path only) and partial trailing blocks (scalar tail path), including 1.
    for (size_t c = 1; c < 48; c++) {
      Test(c, 112, 112);  // large spatial, InputSize multiple of 16
      Test(c, 15, 21);    // InputSize = 315, not a multiple of 16 (tail spatial)
      Test(c, 11, 11);    // small odd spatial
    }
  }
};

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  return (MlasNchwcGetBlockSize() > 1 && is_short_execute)
             ? MlasDirectShortExecuteTests<MlasReorderInputTest>::RegisterShortExecute()
             : 0;
});

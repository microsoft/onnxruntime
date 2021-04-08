// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_util.h"

class MlasQLinearGlobalAveragePoolU8Test : public MlasTestBase {
 private:
  MatrixGuardBuffer<uint8_t> BufferInput;
  MatrixGuardBuffer<uint8_t> BufferOutput;
  MatrixGuardBuffer<uint8_t> BufferOutputReference;

  static void CalculateGlobalAvgPoolU8(
      const uint8_t* x, int64_t batch, int64_t channel, int64_t hw, bool channel_last,
      uint8_t* y, int32_t x_zero_point, float x_scale, int32_t y_zero_point, float y_scale) {
    int32_t bias = -x_zero_point * static_cast<int32_t>(hw);
    int64_t stride_image = channel_last ? channel : 1;
    int64_t stride_channel = channel_last ? 1 : hw;

    for (int64_t b = 0; b < batch; ++b) {
      const uint8_t* bx = x + b * hw * channel;
      uint8_t* by = y + b * channel;
      for (int64_t c = 0; c < channel; ++c) {
        const uint8_t* ix = bx + c * stride_channel;
        int32_t sum = 0;
        for (int64_t i = 0; i < hw; ++i) {
          sum += static_cast<int32_t>(*ix);
          ix += stride_image;
        }
        sum += bias;
        int32_t r = static_cast<int32_t>(std::nearbyintf(x_scale * sum / static_cast<float>(hw) / y_scale));
        r += y_zero_point;
        r = std::min(255, r);
        r = std::max(0, r);
        by[c] = static_cast<uint8_t>(r);
      }
    }
  }

  static void CompareResultWithGold(size_t Batch, size_t Channel,
                                    uint8_t* Output, uint8_t* OutputReference, std::string& info) {
    size_t n = 0;
    for (size_t b = 0; b < Batch; ++b) {
      for (size_t c = 0; c < Channel; c++) {
        int diff = abs((int)Output[n] - (int)OutputReference[n]);
        ASSERT_LE(diff, 1) << " got:" << int(Output[n]) << " expecting:" << int(OutputReference[n]) << " @[" << b << "," << c << "], " << info.c_str();
      }
    }
  }

  static std::string GetTestInfo(bool channel_last,
                                 size_t Batch,
                                 size_t Stride,
                                 size_t Channel,
                                 size_t ImageSize,
                                 float InputScale,
                                 uint8_t InputZeroPoint,
                                 float OutputScale,
                                 uint8_t OutputZeroPoint) {
    std::stringstream ss;
    ss << (channel_last ? "Nhwc_" : "Nchw_");
    ss << Batch << "x [C=" << Stride << "-" << Channel << "] x" << ImageSize << "-";
    ss << "(" << (int)InputZeroPoint << "," << InputScale << "," << (int)OutputZeroPoint << "," << OutputScale << ")";
    return ss.str();
  }

  void Test(bool channel_last,
            size_t Batch,
            size_t Stride,
            size_t Channel,
            size_t ImageSize,
            float InputScale,
            uint8_t InputZeroPoint,
            float OutputScale,
            uint8_t OutputZeroPoint,
            int32_t UnalignedOffset = 0) {
    size_t N = Batch * Stride * ImageSize;
    size_t ResultLen = Batch * Stride;
    uint8_t* Input = BufferInput.GetBuffer(N);
    uint8_t* Output = BufferOutput.GetBuffer(ResultLen);
    uint8_t* Gold = BufferOutputReference.GetBuffer(ResultLen);
    std::string test_info = GetTestInfo(
        channel_last, Batch, Stride, Channel, ImageSize,
        InputScale, InputZeroPoint, OutputScale, OutputZeroPoint);

    std::default_random_engine generator(static_cast<unsigned>(N));
    std::uniform_int_distribution<int> distribution(0, 255);
    for (size_t n = 0; n < N; n++) {
      Input[n] = static_cast<uint8_t>(distribution(generator));
    }
    CalculateGlobalAvgPoolU8(
        Input, Batch, Stride, ImageSize, channel_last,
        Gold, InputZeroPoint, InputScale, OutputZeroPoint, OutputScale);

    if (!channel_last) {
      std::vector<int32_t> acc(MlasQLinearSafePaddingElementCount(sizeof(int32_t), ResultLen + UnalignedOffset));
      MlasQLinearGlobalAveragePoolNchw(
          Input, InputScale, InputZeroPoint, Output,
          OutputScale, OutputZeroPoint, ResultLen, ImageSize, acc.data() + UnalignedOffset);
    } else {
      std::vector<int32_t> acc(MlasQLinearSafePaddingElementCount(sizeof(int32_t), Channel + UnalignedOffset));
      std::vector<uint8_t> zero(MlasQLinearSafePaddingElementCount(sizeof(uint8_t), Channel + UnalignedOffset));
      if (Stride == Channel) {
        MlasQLinearGlobalAveragePoolNhwc(
            Input, InputScale, InputZeroPoint, Output,
            OutputScale, OutputZeroPoint, Batch, ImageSize, Stride, Channel,
            acc.data() + UnalignedOffset, zero.data() + UnalignedOffset);
      } else {
        for (size_t tc = 0; tc < Stride; tc += Channel) {
          size_t cg = ((tc + Channel <= Stride) ? Channel : (Stride - tc));
          MlasQLinearGlobalAveragePoolNhwc(
              Input + tc, InputScale, InputZeroPoint, Output + tc,
              OutputScale, OutputZeroPoint, Batch, ImageSize, Stride, cg,
              acc.data() + UnalignedOffset, zero.data() + UnalignedOffset);
        }
      }
    }

    CompareResultWithGold(Batch, Channel, Output, Gold, test_info);
  }

 public:
  static const char* GetTestSuiteName() {
    static const std::string suite_name("QLinearGlobalAvgPool");
    return suite_name.c_str();
  }

  void ExecuteShort(void) override {
    static const uint8_t zero_points[] = {0, 18, 128, 231, 255};
    static const float scales[] = {18.0f, 90.0f};
    static const size_t Batch[] = {1, 3};
    static const size_t Stride[] = {7, 8, 63, 256};
    static const size_t ImageSize[] = {7, 8, 64};
    static int unalign_offset = 0;

    for (int channel_last = 0; channel_last <= 1; ++channel_last) {
      for (size_t b = 0; b < _countof(Batch); b++) {
        for (size_t xzp = 0; xzp < _countof(zero_points); xzp++) {
          for (size_t yzp = 0; yzp < _countof(zero_points); yzp++) {
            for (size_t xs = 0; xs < _countof(scales); ++xs) {
              for (size_t ys = 0; ys < _countof(scales); ++ys) {
                for (size_t i = 0; i < _countof(ImageSize); i++) {
                  for (size_t s = 0; s < _countof(Stride); s++) {
                    Test(channel_last, Batch[b], Stride[s], Stride[s], ImageSize[i],
                         scales[xs], zero_points[xzp], scales[ys], zero_points[yzp], unalign_offset);
                    if (channel_last == 1 && Stride[s] > 32) {
                      Test(channel_last, Batch[b], Stride[s], 32, ImageSize[i],
                           scales[xs], zero_points[xzp], scales[ys], zero_points[yzp], unalign_offset);
                    }
                    unalign_offset = (unalign_offset + 1) & 3;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
};

template <> MlasQLinearGlobalAveragePoolU8Test* MlasTestFixture<MlasQLinearGlobalAveragePoolU8Test>::mlas_tester(nullptr);

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  return is_short_execute ? MlasDirectShortExecuteTests<MlasQLinearGlobalAveragePoolU8Test>::RegisterShortExecute() : 0;
});

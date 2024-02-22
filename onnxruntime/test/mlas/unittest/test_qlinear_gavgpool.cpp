// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_util.h"

#include <vector>

template <typename T8Bits>
class MlasQLinearGlobalAveragePoolTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<T8Bits> BufferInput;
  MatrixGuardBuffer<T8Bits> BufferOutput;
  MatrixGuardBuffer<T8Bits> BufferOutputReference;
  static const std::vector<T8Bits> ZeroPoints;

  static void CalculateGlobalAvgPool(
      const T8Bits* x, int64_t batch, int64_t channel, int64_t hw, bool channel_last,
      T8Bits* y, int32_t x_zero_point, float x_scale, int32_t y_zero_point, float y_scale) {
    int32_t bias = -x_zero_point * static_cast<int32_t>(hw);
    int64_t stride_image = channel_last ? channel : 1;
    int64_t stride_channel = channel_last ? 1 : hw;

    for (int64_t b = 0; b < batch; ++b) {
      const T8Bits* bx = x + b * hw * channel;
      T8Bits* by = y + b * channel;
      for (int64_t c = 0; c < channel; ++c) {
        const T8Bits* ix = bx + c * stride_channel;
        int32_t sum = 0;
        for (int64_t i = 0; i < hw; ++i) {
          sum += static_cast<int32_t>(*ix);
          ix += stride_image;
        }
        sum += bias;
        int32_t r = static_cast<int32_t>(std::nearbyintf(x_scale * sum / static_cast<float>(hw) / y_scale));
        r += y_zero_point;
        r = std::min((int32_t)(std::numeric_limits<T8Bits>::max()), r);
        r = std::max((int32_t)(std::numeric_limits<T8Bits>::lowest()), r);
        by[c] = static_cast<T8Bits>(r);
      }
    }
  }

  static void CompareResultWithGold(size_t Batch, size_t Channel,
                                    T8Bits* Output, T8Bits* OutputReference, std::string& info) {
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
                                 T8Bits InputZeroPoint,
                                 float OutputScale,
                                 T8Bits OutputZeroPoint) {
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
            T8Bits InputZeroPoint,
            float OutputScale,
            T8Bits OutputZeroPoint,
            int32_t UnalignedOffset = 0) {
    size_t N = Batch * Stride * ImageSize;
    size_t ResultLen = Batch * Stride;
    T8Bits* Input = BufferInput.GetBuffer(N);
    T8Bits* Output = BufferOutput.GetBuffer(ResultLen);
    T8Bits* Gold = BufferOutputReference.GetBuffer(ResultLen);
    std::string test_info = GetTestInfo(
        channel_last, Batch, Stride, Channel, ImageSize,
        InputScale, InputZeroPoint, OutputScale, OutputZeroPoint);

    std::default_random_engine generator(static_cast<unsigned>(N));
    std::uniform_int_distribution<int> distribution(std::numeric_limits<T8Bits>::lowest(), std::numeric_limits<T8Bits>::max());
    for (size_t n = 0; n < N; n++) {
      Input[n] = static_cast<T8Bits>(distribution(generator));
    }
    CalculateGlobalAvgPool(
        Input, Batch, Stride, ImageSize, channel_last,
        Gold, InputZeroPoint, InputScale, OutputZeroPoint, OutputScale);

    if (!channel_last) {
      std::vector<int32_t> acc(MlasQLinearSafePaddingElementCount(sizeof(int32_t), ResultLen + UnalignedOffset));
      MlasQLinearGlobalAveragePoolNchw(
          Input, InputScale, InputZeroPoint, Output,
          OutputScale, OutputZeroPoint, ResultLen, ImageSize, acc.data() + UnalignedOffset);
    } else {
      std::vector<int32_t> acc(MlasQLinearSafePaddingElementCount(sizeof(int32_t), Channel + UnalignedOffset));
      std::vector<T8Bits> zero(MlasQLinearSafePaddingElementCount(sizeof(T8Bits), Channel + UnalignedOffset));
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
    constexpr bool is_signed = std::is_signed<T8Bits>::value;
    static const std::string suite_name(is_signed ? "QLinearGlobalAvgPoolS8" : "QLinearGlobalAvgPoolU8");
    return suite_name.c_str();
  }

  void ExecuteShort(void) override {
    static const float scales[] = {18.0f, 90.0f};
    static const size_t Batch[] = {1, 3};
    static const size_t Stride[] = {7, 8, 63, 256};
    static const size_t ImageSize[] = {7, 8, 64};
    static int unalign_offset = 0;

    for (int channel_last = 0; channel_last <= 1; ++channel_last) {
      for (size_t b = 0; b < _countof(Batch); b++) {
        for (size_t xzp = 0; xzp < ZeroPoints.size(); xzp++) {
          for (size_t yzp = 0; yzp < ZeroPoints.size(); yzp++) {
            for (size_t xs = 0; xs < _countof(scales); ++xs) {
              for (size_t ys = 0; ys < _countof(scales); ++ys) {
                for (size_t i = 0; i < _countof(ImageSize); i++) {
                  for (size_t s = 0; s < _countof(Stride); s++) {
                    Test(channel_last != 0, Batch[b], Stride[s], Stride[s], ImageSize[i],
                         scales[xs], ZeroPoints[xzp], scales[ys], ZeroPoints[yzp], unalign_offset);
                    if (channel_last == 1 && Stride[s] > 32) {
                      Test(channel_last != 0, Batch[b], Stride[s], 32, ImageSize[i],
                           scales[xs], ZeroPoints[xzp], scales[ys], ZeroPoints[yzp], unalign_offset);
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

template <>
const std::vector<int8_t> MlasQLinearGlobalAveragePoolTest<int8_t>::ZeroPoints = {-128, -110, 1, 103, 127};

template <>
const std::vector<uint8_t> MlasQLinearGlobalAveragePoolTest<uint8_t>::ZeroPoints = {0, 18, 128, 231, 255};

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  if (is_short_execute) {
    return MlasDirectShortExecuteTests<MlasQLinearGlobalAveragePoolTest<int8_t>>::RegisterShortExecute() +
           MlasDirectShortExecuteTests<MlasQLinearGlobalAveragePoolTest<uint8_t>>::RegisterShortExecute();
  }
  return (size_t)0;
});

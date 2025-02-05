// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_util.h"

class MlasComputeExpTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<float> BufferInput;
  MatrixGuardBuffer<float> BufferOutput;
  MatrixGuardBuffer<float> BufferOutputReference;

  void Test(size_t N, float MinimumValue, float MaximumValue) {
    float* Input = BufferInput.GetBuffer(N);
    float* Output = BufferOutput.GetBuffer(N);
    float* OutputReference = BufferOutputReference.GetBuffer(N);

    std::default_random_engine generator(static_cast<unsigned>(N));
    std::uniform_real_distribution<float> distribution(MinimumValue, MaximumValue);

    for (size_t n = 0; n < N; n++) {
      Input[n] = distribution(generator);
    }

    for (size_t n = 0; n < N; n++) {
      OutputReference[n] = std::exp(Input[n]);
    }

    MlasComputeExp(Input, Output, N);

    constexpr float AbsoluteTolerance = 1e-6f;
    constexpr float RelativeTolerance = 1e-6f;

    for (size_t n = 0; n < N; n++) {
      float diff = std::fabs(Output[n] - OutputReference[n]);
      ASSERT_TRUE(diff <= AbsoluteTolerance || diff <= std::fabs(OutputReference[n]) * RelativeTolerance)
          << " @" << n << " of " << N << ", got: " << Output[n] << ", expecting: " << OutputReference[n];
    }
  }

 public:
  static const char* GetTestSuiteName() {
    static const std::string suite_name("Exp");
    return suite_name.c_str();
  }

  void ExecuteShort(void) override {
    for (size_t n = 1; n < 128; n++) {
      Test(n, -10.f, 10.f);
    }
  }
};

class MyComputeExpTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<float> BufferInput;
  MatrixGuardBuffer<float> BufferOutput;
  MatrixGuardBuffer<float> BufferOutputReference;

const struct {
    float LowerRange;
    float UpperRange;
    float LowerRangeSumExp;
    float UpperRangeSumExp;
    float RoundingBias;
    float Log2Reciprocal;
    float Log2High;
    float Log2Low;
    float poly_0;
    float poly_1;
    float poly_2;
    float poly_3;
    float poly_4;
    float poly_56;
    int32_t MinimumExponent;
    int32_t MaximumExponent;
} MlasExpConstants = {
    -103.9720840454f, // -150 * ln2
    88.7762626647950f, // 128 * ln2
    -88.3762626647949f,
    88.3762626647949f, // 127.5 * ln2
    12582912.f, // 1.5 * 2^23
    1.44269504088896341f,
    -6.93145752e-1f,
    -1.42860677e-6f,
    0x1.694000p-10, // 6! // TODO: these polynomials may be chosen by optimization, even though difference is small. Test small number errors mine vs. hers.
    0x1.125edcp-7, // 5!
    0x1.555b5ap-5, // 4!
    0x1.555450p-3, // 3!
    0x1.fffff6p-2, // 2!
    0x1.000000p+0,
    int32_t(0xC1000000), // -126
    int32_t(0x3F800000), // 1.0f
};

const struct {
    _Float16 LowerRange;
    _Float16 UpperRange;
    _Float16 LowerRangeSumExp;
    _Float16 UpperRangeSumExp;
    _Float16 RoundingBias;
    _Float16 Log2Reciprocal;
    _Float16 Log2High;
    _Float16 Log2Low;
    _Float16 Log2Lowest;
    _Float16 poly_0;
    _Float16 poly_1;
    _Float16 poly_2;
    _Float16 poly_3;
    _Float16 poly_4;
    _Float16 poly_56;
    int16_t MinimumExponent;
    int16_t MaximumExponent;
} MlasExp16Constants = {
    -17.328679513f16, // -25 * ln2 cc55
    11.090354888f16, // 16 * ln2 498c
    -10.743781298f16, // -15.5 * ln2 c95f
    10.743781298f16, // 15.5 * ln2 495f
    1536.f16, // 1.5 * 2^10 6600
    1.4423828125f16, // 1/ln2, 3dc5
    -6.9287109375e-1f16, // 0xb98b
    -2.758502960205078e-4f16, // 0x8c85
    -2.384185791015625e-7f16, // 0x8004
    1.388888888888889e-3f16, // 1/6! 0x15b0
    8.333333333333333e-3f16, // 1/5! 0x2044
    4.1666666666666664e-2f16, // 1/4! 0x2955
    1.6666666e-1f16, // 1/3! 0x3155
    0.5f16, // 1/2! 0x3800
    1.0f16, // 1/1! 0x3c00
    int16_t(0xC800), // -14
    int16_t(0x3C00), // 15
};

  void print_hex(std::string note, _Float16 x) {
    int16_t i = *reinterpret_cast<int16_t*>(&x);
    std::cout << note << std::hex << i << std::dec << std::endl;
  }

  void print_hex(std::string note, float x) {
    int i = *reinterpret_cast<int*>(&x);
    std::cout << note << std::hex << i << std::dec << std::endl;
  }

  void print_hex(std::string note, int x) {
    std::cout << note << std::hex << x << std::dec << std::endl;
  }

  _Float16 my_exp(_Float16 x) {
    bool debug = false;
    x = std::min(std::max(x, MlasExp16Constants.LowerRange), MlasExp16Constants.UpperRange);

    auto biased = x * MlasExp16Constants.Log2Reciprocal + MlasExp16Constants.RoundingBias;
    if (debug) print_hex("biased ", biased);
    auto m = biased - MlasExp16Constants.RoundingBias;
    if (debug) print_hex("m ", m);

    _Float16 r = m * MlasExp16Constants.Log2High + x;
    r = m * MlasExp16Constants.Log2Low + r;
    r = m * MlasExp16Constants.Log2Lowest + r;
    if (debug) print_hex("r ", r);

    int16_t bias_i = *reinterpret_cast<int16_t*>(&biased);
    int16_t overflow = bias_i << 10;
    if (debug) print_hex("overflow ", overflow);
    auto normal = overflow;

    normal = std::min(normal, MlasExp16Constants.MaximumExponent);
    normal = std::max(normal, MlasExp16Constants.MinimumExponent);
    if (debug) print_hex("clampped normal ", normal);

    overflow = overflow - normal;
    if (debug) print_hex("lowered overflow ", overflow);
    overflow = overflow + MlasExp16Constants.MaximumExponent;
    if (debug) print_hex("adjusted overflow ", overflow);
    normal = normal + MlasExp16Constants.MaximumExponent;
    if (debug) print_hex("adjusted normal ", normal);

    auto p = (_Float16)MlasExp16Constants.poly_0;
    p = p * r + (_Float16)MlasExp16Constants.poly_1;
    p = p * r + (_Float16)MlasExp16Constants.poly_2;
    p = p * r + (_Float16)MlasExp16Constants.poly_3;
    p = p * r + (_Float16)MlasExp16Constants.poly_4;
    p = p * r + (_Float16)MlasExp16Constants.poly_56;

    _Float16 overflow_f = *reinterpret_cast<_Float16*>(&overflow);
    _Float16 normal_f = *reinterpret_cast<_Float16*>(&normal);
    r = r * overflow_f;
    p = p * r + overflow_f;
    p = p * normal_f;

    return p;
  }

  float my_exp(float x) {
    x = std::min(std::max(x, MlasExpConstants.LowerRange), MlasExpConstants.UpperRange);

    auto biased = x * MlasExpConstants.Log2Reciprocal + MlasExpConstants.RoundingBias;
    print_hex("biased ", biased);
    auto m = biased - MlasExpConstants.RoundingBias;
    print_hex("m ", m);

    float r = m * MlasExpConstants.Log2High + x;
    r = m * MlasExpConstants.Log2Low + r;
    print_hex("r ", r);

    int32_t bias_i = *reinterpret_cast<int*>(&biased);
    auto overflow = bias_i << 23;
    print_hex("overflow ", overflow);
    auto normal = overflow;

    normal = std::min(normal, MlasExpConstants.MaximumExponent);
    normal = std::max(normal, MlasExpConstants.MinimumExponent);
    print_hex("clampped normal ", normal);

    overflow = overflow - normal;
    print_hex("lowered overflow ", overflow);
    overflow = overflow + MlasExpConstants.MaximumExponent;
    print_hex("adjusted overflow ", overflow);
    normal = normal + MlasExpConstants.MaximumExponent;
    print_hex("adjusted normal ", normal);

    auto p = MlasExpConstants.poly_0;
    p = p * r + MlasExpConstants.poly_1;
    p = p * r + MlasExpConstants.poly_2;
    p = p * r + MlasExpConstants.poly_3;
    p = p * r + MlasExpConstants.poly_4;
    p = p * r + MlasExpConstants.poly_56;

    float overflow_f = *reinterpret_cast<float*>(&overflow);
    float normal_f = *reinterpret_cast<float*>(&normal);
    r = r * overflow_f;
    p = p * r + overflow_f;
    p = p * normal_f;

    return p;
  }

  float my_exp_no_overflow(float x) {
    x = std::min(std::max(x, MlasExpConstants.LowerRange), MlasExpConstants.UpperRange);

    auto biased = x * MlasExpConstants.Log2Reciprocal + MlasExpConstants.RoundingBias;
    print_hex("biased ", biased);
    auto m = biased - MlasExpConstants.RoundingBias;
    print_hex("m ", m);

    float r = m * MlasExpConstants.Log2High + x;
    r = m * MlasExpConstants.Log2Low + r;
    print_hex("r ", r);

    int32_t bias_i = *reinterpret_cast<int*>(&biased);
    auto normal = bias_i << 23;
    print_hex("clampped normal ", normal);
    normal = normal + MlasExpConstants.MaximumExponent;
    print_hex("adjusted normal ", normal);

    auto p = MlasExpConstants.poly_0;
    p = p * r + MlasExpConstants.poly_1;
    p = p * r + MlasExpConstants.poly_2;
    p = p * r + MlasExpConstants.poly_3;
    p = p * r + MlasExpConstants.poly_4;
    p = p * r + MlasExpConstants.poly_56;
    p = p * r + MlasExpConstants.poly_56;


    float normal_f = *reinterpret_cast<float*>(&normal);
    p = p * normal_f;

    return p;
  }

  void Test(float x) {
    float ref = std::exp(x);
    float out = my_exp_no_overflow(x);

    constexpr float AbsoluteTolerance = 1e-6f;
    constexpr float RelativeTolerance = 1e-6f;

    float diff = std::fabs(out - ref);
    ASSERT_TRUE(diff <= AbsoluteTolerance || diff <= std::fabs(ref) * RelativeTolerance)
        << " of " << 1 << ", got: " << out << ", expecting: " << ref;
    std::cout << "result: " << out << ", expecting: " << ref << std::endl;
  }

  void Test(_Float16 x) {
    float ref = std::exp(static_cast<float>(x));
    float out = my_exp(x);

    constexpr float AbsoluteTolerance = 1e-6f;
    constexpr float RelativeTolerance = 1e-6f;

    float diff = std::abs(out - ref);
    // ASSERT_TRUE(diff <= AbsoluteTolerance || diff <= std::fabs(ref) * RelativeTolerance)
    //     << " of " << 1 << ", got: " << out << ", expecting: " << ref << " diff " << diff / ref;
    std::cout << "x " << (float)x << ", result: " << out << ", expecting: " << ref << " diff " << diff / ref << std::endl;
  }

const struct {
    float LowerRange;
    float UpperRange;
    float alpha_13;
    float alpha_11;
    float alpha_9;
    float alpha_7;
    float alpha_5;
    float alpha_3;
    float alpha_1;
    float beta_6;
    float beta_4;
    float beta_2;
    float beta_0;
} MlasTanhConstants = {
    -9.0f,
    9.0f,
    -2.76076847742355e-16f,
    2.00018790482477e-13f,
    -8.60467152213735e-11f,
    5.12229709037114e-08f,
    1.48572235717979e-05f,
    6.37261928875436e-04f,
    4.89352455891786e-03f,
    1.19825839466702e-06f, // TODO: test errors
    1.18534705686654e-04f,
    2.26843463243900e-03f,
    4.89352518554385e-03f,
};

const struct {
    _Float16 LowerRange;
    _Float16 UpperRange;
    _Float16 alpha_9;
    _Float16 alpha_7;
    _Float16 alpha_5;
    _Float16 alpha_3;
    _Float16 alpha_1;
    _Float16 beta_10;
    _Float16 beta_8;
    _Float16 beta_6;
    _Float16 beta_4;
    _Float16 beta_2;
    _Float16 beta_0;
} MlasTanh16Constants = {
    -5.0f16, // c500
    5.0f16, // 4500
    2.755731922398589e-06f16, // 0x002e
    0.00019841269841269839f16, // 0xa80
    0.008333333333333333f16, // 0x2044
    0.16666666666666666f16, // 0x3155
    1.f16, // 0x3c00
    2.7557319223985894e-07f16, // 0x0005
    2.48015873015873e-05f16, // 0x01a0
    0.001388888888888889f16, // 0x15b0
    0.041666666666666664f16, // 0x2955
    0.5f16, // 0x3800
    1.f16, // 0x3c00
};

  float my_tanh(float Value) {
    float v_tmp;
    v_tmp = (Value < MlasTanhConstants.LowerRange) ? MlasTanhConstants.LowerRange : Value;
    Value = (v_tmp > MlasTanhConstants.UpperRange) ? MlasTanhConstants.UpperRange : v_tmp;

    float ValueSquared = Value * Value;

    float p;
    p = ValueSquared * MlasTanhConstants.alpha_13 + MlasTanhConstants.alpha_11;
    p = p * ValueSquared + MlasTanhConstants.alpha_9;
    p = p * ValueSquared + MlasTanhConstants.alpha_7;
    p = p * ValueSquared + MlasTanhConstants.alpha_5;
    p = p * ValueSquared + MlasTanhConstants.alpha_3;
    p = p * ValueSquared + MlasTanhConstants.alpha_1;
    p = p * Value;

    float q;
    q = ValueSquared * MlasTanhConstants.beta_6 + MlasTanhConstants.beta_4;
    q = q * ValueSquared + MlasTanhConstants.beta_2;
    q = q * ValueSquared + MlasTanhConstants.beta_0;

    return (p / q);
  }

  _Float16 my_tanh(_Float16 Value) {
    _Float16 v_tmp;
    v_tmp = (Value < MlasTanh16Constants.LowerRange) ? MlasTanh16Constants.LowerRange : Value;
    Value = (v_tmp > MlasTanh16Constants.UpperRange) ? MlasTanh16Constants.UpperRange : v_tmp;

    _Float16 ValueSquared = Value * Value;

    _Float16 p = MlasTanh16Constants.alpha_9;
    p = p * ValueSquared + MlasTanh16Constants.alpha_7;
    p = p * ValueSquared + MlasTanh16Constants.alpha_5;
    p = p * ValueSquared + MlasTanh16Constants.alpha_3;
    p = p * ValueSquared + MlasTanh16Constants.alpha_1;
    p = p * Value;

    _Float16 q = MlasTanh16Constants.beta_10;
    q = q * ValueSquared + MlasTanh16Constants.beta_8;
    q = q * ValueSquared + MlasTanh16Constants.beta_6;
    q = q * ValueSquared + MlasTanh16Constants.beta_4;
    q = q * ValueSquared + MlasTanh16Constants.beta_2;
    q = q * ValueSquared + MlasTanh16Constants.beta_0;

    return (p / q);
  }

  _Float16 fast_tanh(_Float16 x) {
    _Float16 x2 = x * x;
    _Float16 a = x * (135.1350f16 + x2 * (17.3250f16 + x2 * (.3780f16 + x2)));
    _Float16 b = 135.1350f16 + x2 * (62.3700f16 + x2 * (3.1500f16 + x2 * .0280f16));
    return a / b;
  }

  _Float16 my_tanh_no_overflow(_Float16 Value) {
    if (Value > 0.5f16) {
      _Float16 exp = my_exp(Value);
      return (exp - 1.0f16/exp) / (exp + 1.0f16/exp);
    } else {
      return my_tanh(Value);
    }
  }

  void test_tanh(float x) {
    float ref = std::tanh(x);
    float out = my_tanh(x);
    float diff = std::abs(out - ref);
    std::cout << "result: " << out << ", expecting: " << ref << " diff " << diff / ref << std::endl;
  }

  void test_tanh(_Float16 x) {
    float ref = std::tanh((float)x);
    float out = my_tanh(x);
    float diff = std::abs(out - ref);
    std::cout << "x " << (float)x << ", result: " << out << ", expecting: " << ref << " diff " << diff / ref << std::endl;
  }

  void test_tanh_no_overflow(_Float16 x) {
    float ref = std::tanh((float)x);
    float out = my_tanh_no_overflow(x);
    float diff = std::abs(out - ref);
    std::cout << "x " << (float)x << ", result: " << out << ", expecting: " << ref << " diff " << diff / ref << std::endl;
  }

 public:
  static const char* GetTestSuiteName() {
    static const std::string suite_name("MyExp");
    return suite_name.c_str();
  }

  void ExecuteShort(void) override {
    // print_hex("lower range ", MlasExp16Constants.LowerRange);
    // print_hex("upper range ", MlasExp16Constants.UpperRange);
    // print_hex("lower range sum exp ", MlasExp16Constants.LowerRangeSumExp);
    // print_hex("upper range sum exp ", MlasExp16Constants.UpperRangeSumExp);
    // print_hex("rounding bias ", MlasExp16Constants.RoundingBias);
    // print_hex("log2 reciprocal ", MlasExp16Constants.Log2Reciprocal);
    // print_hex("h ", (_Float16)MlasExp16Constants.Log2High);
    // print_hex("l ", (_Float16)MlasExp16Constants.Log2Low);
    // print_hex("ll ", MlasExp16Constants.Log2Lowest);
    // print_hex("r ", (_Float16)MlasExp16Constants.Log2Reciprocal);
    // print_hex("poly0 ", (_Float16)MlasExp16Constants.poly_0);
    // print_hex("poly1 ", (_Float16)MlasExp16Constants.poly_1);
    // print_hex("poly2 ", (_Float16)MlasExp16Constants.poly_2);
    // print_hex("poly3 ", (_Float16)MlasExp16Constants.poly_3);
    // print_hex("poly4 ", (_Float16)MlasExp16Constants.poly_4);
    // Test(.01f16);
    print_hex("lower range ", MlasTanh16Constants.LowerRange);
    print_hex("upper range ", MlasTanh16Constants.UpperRange);
    print_hex("alpha_9 ", MlasTanh16Constants.alpha_9);
    print_hex("alpha_7 ", MlasTanh16Constants.alpha_7);
    print_hex("alpha_5 ", MlasTanh16Constants.alpha_5);
    print_hex("alpha_3 ", MlasTanh16Constants.alpha_3);
    print_hex("alpha_1 ", MlasTanh16Constants.alpha_1);
    print_hex("beta_10 ", MlasTanh16Constants.beta_10);
    print_hex("beta_8 ", MlasTanh16Constants.beta_8);
    print_hex("beta_6 ", MlasTanh16Constants.beta_6);
    print_hex("beta_4 ", MlasTanh16Constants.beta_4);
    print_hex("beta_2 ", MlasTanh16Constants.beta_2);
    print_hex("beta_0 ", MlasTanh16Constants.beta_0);
    for (_Float16 x = 0.f16; x <= 9.f16; x += 0.005f16) {
      test_tanh_no_overflow(x);
    }
  }
};

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  // no long execute needed
  if (is_short_execute) {
    return MlasDirectShortExecuteTests<MlasComputeExpTest>::RegisterShortExecute() +
           MlasDirectShortExecuteTests<MyComputeExpTest>::RegisterShortExecute();
  }
  return 0ul;
});

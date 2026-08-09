// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_util.h"

class MlasQLinearBinaryOpTest : public MlasTestBase {
 public:
  typedef void(MLASCALL* QLinearBinaryOpS8)(
      const int8_t* InputA, float ScaleA, int32_t ZeroPointA,
      const int8_t* InputB, float ScaleB, int32_t ZeroPointB,
      float ScaleC, int32_t ZeroPointC, int8_t* OutputC,
      size_t N, bool IsScalarB);
  typedef void(MLASCALL* QLinearBinaryOpU8)(
      const uint8_t* InputA, float ScaleA, int32_t ZeroPointA,
      const uint8_t* InputB, float ScaleB, int32_t ZeroPointB,
      float ScaleC, int32_t ZeroPointC, uint8_t* OutputC,
      size_t N, bool IsScalarB);

 private:
  std::function<float(float, float)> ScalarOp;
  std::string ScalarOpName;
  QLinearBinaryOpS8 QLinearS8Op;
  QLinearBinaryOpU8 QLinearU8Op;
  MatrixGuardBuffer<uint8_t> BufferInputA;
  MatrixGuardBuffer<uint8_t> BufferInputB;
  MatrixGuardBuffer<uint8_t> BufferOutput;
  MatrixGuardBuffer<uint8_t> BufferOutputReference;

  template <typename T>
  T QLinearBinaryScalar(T a,
                        float ScaleA,
                        int32_t ZeroPointA,
                        T b,
                        float ScaleB,
                        int32_t ZeroPointB,
                        float ScaleC,
                        int32_t ZeroPointC) {
    constexpr int qmax = std::numeric_limits<T>::max();
    constexpr int qmin = std::numeric_limits<T>::min();

    float ValueA = ScaleA * (static_cast<int>(a) - ZeroPointA);
    float ValueB = ScaleB * (static_cast<int>(b) - ZeroPointB);
    float ValueC = std::nearbyintf(ScalarOp(ValueA, ValueB) / ScaleC) + ZeroPointC;
    int qc = static_cast<int>(ValueC);
    qc = std::min(qc, qmax);
    qc = std::max(qc, qmin);
    return static_cast<T>(qc);
  }

  template <typename T>
  void Test(void(MLASCALL* QLinearBinaryOp)(
                const T* InputA, float ScaleA, int32_t ZeroPointA,
                const T* InputB, float ScaleB, int32_t ZeroPointB,
                float ScaleC, int32_t ZeroPointC, T* OutputC,
                size_t N, bool IsScalarB),
            size_t N,
            bool IsScalarB,
            float ScaleA,
            int32_t ZeroPointA,
            float ScaleB,
            int32_t ZeroPointB,
            float ScaleC,
            int32_t ZeroPointC) {
    T* InputA = (T*)BufferInputA.GetBuffer(N);
    T* InputB = (T*)BufferInputB.GetBuffer(IsScalarB ? 1 : N);
    T* OutputC = (T*)BufferOutput.GetBuffer(N);
    T* OutputReference = (T*)BufferOutputReference.GetBuffer(N);

    constexpr int MinimumValue = (int)std::numeric_limits<T>::min();
    constexpr int MaximumValue = (int)std::numeric_limits<T>::max();
    std::default_random_engine generator(static_cast<unsigned>(N));
    std::uniform_int_distribution<int> distribution(MinimumValue, MaximumValue);

    if (IsScalarB) {
      InputB[0] = static_cast<T>(distribution(generator));
    }
    for (size_t n = 0; n < N; n++) {
      InputA[n] = static_cast<T>(distribution(generator));
      if (!IsScalarB) {
        InputB[n] = static_cast<T>(distribution(generator));
      }
      OutputReference[n] = QLinearBinaryScalar(InputA[n], ScaleA, ZeroPointA, InputB[IsScalarB ? 0 : n], ScaleB, ZeroPointB, ScaleC, ZeroPointC);
    }

    QLinearBinaryOp(InputA, ScaleA, ZeroPointA, InputB, ScaleB, ZeroPointB, ScaleC, ZeroPointC, OutputC, N, IsScalarB);

    for (size_t n = 0; n < N; n++) {
      int diff = (int)OutputC[n] - (int)OutputReference[n];
      ASSERT_TRUE(diff >= -1 && diff <= 1)
          << ", IsScalarB=" << static_cast<int>(IsScalarB) << ", @" << n << " of " << N << ", "
          << static_cast<int>(InputA[n]) << "(" << ScaleA << "," << ZeroPointA << "), "
          << static_cast<int>(InputB[IsScalarB ? 0 : n]) << "(" << ScaleB << "," << ZeroPointB << ") ==> "
          << static_cast<int>(OutputC[n]) << "(" << ScaleC << "," << ZeroPointC << "), "
          << " expecting:" << static_cast<int>(OutputReference[n]);
    }
  }

 public:
  explicit MlasQLinearBinaryOpTest(std::function<float(float, float)> P_ScalarOp,
                                   const std::string& P_ScalarOpName,
                                   QLinearBinaryOpS8 P_QLinearS8Op,
                                   QLinearBinaryOpU8 P_QLinearU8Op)
      : ScalarOp(P_ScalarOp),
        ScalarOpName(P_ScalarOpName),
        QLinearS8Op(P_QLinearS8Op),
        QLinearU8Op(P_QLinearU8Op) {
  }

  void ExecuteShort(void) override {
    static const uint8_t zero_points[] = {0, 18, 75, 128, 157, 231, 255};
    static const float c_scales[] = {18.0f, 90.0f};

    const int8_t* s_zero_points = (const int8_t*)(&zero_points[0]);
    for (size_t a = 0; a < _countof(zero_points); a++) {
      for (size_t b = 0; b < _countof(zero_points); b++) {
        for (size_t c = 0; c < _countof(zero_points); c++) {
          for (size_t s = 0; s < _countof(c_scales); s++) {
            for (size_t n = 1; n < 128; n++) {
              // u8, vector + vector
              Test<uint8_t>(QLinearU8Op, n, false, 10.f, zero_points[a], 10.f, zero_points[b], c_scales[s], zero_points[c]);

              // u8, vector + scalar
              Test<uint8_t>(QLinearU8Op, n, true, 10.f, zero_points[a], 10.f, zero_points[b], c_scales[s], zero_points[c]);

              // s8, vector + vector
              Test<int8_t>(QLinearS8Op, n, false, 10.f, s_zero_points[a], 10.f, s_zero_points[b], c_scales[s], s_zero_points[c]);

              // s8, vector + scalar
              Test<int8_t>(QLinearS8Op, n, true, 10.f, s_zero_points[a], 10.f, s_zero_points[b], c_scales[s], s_zero_points[c]);
            }
          }
        }
      }
    }
  }
};

class MlasQLinearAddTest : public MlasQLinearBinaryOpTest {
 public:
  MlasQLinearAddTest() : MlasQLinearBinaryOpTest(
                             [](float a, float b) { return a + b; },
                             "+",
                             MlasQLinearAdd<int8_t>,
                             MlasQLinearAdd<uint8_t>) {}

  static const char* GetTestSuiteName() {
    static const std::string suite_name("QLinearAdd");
    return suite_name.c_str();
  }
};

class MlasQLinearMulTest : public MlasQLinearBinaryOpTest {
 public:
  MlasQLinearMulTest() : MlasQLinearBinaryOpTest(
                             [](float a, float b) { return a * b; },
                             "*",
                             MlasQLinearMul<int8_t>,
                             MlasQLinearMul<uint8_t>) {}

  static const char* GetTestSuiteName() {
    static const std::string suite_name("QLinearMul");
    return suite_name.c_str();
  }
};

static bool UNUSED_VARIABLE added_to_main = AddTestRegister([](bool is_short_execute) {
  size_t count = 0;
  if (is_short_execute) {
    count += MlasDirectShortExecuteTests<MlasQLinearAddTest>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasQLinearMulTest>::RegisterShortExecute();
  }
  return count;
});

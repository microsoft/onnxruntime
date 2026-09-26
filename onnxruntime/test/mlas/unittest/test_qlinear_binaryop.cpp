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

 protected:
  std::function<double(double, double)> ScalarOp;
  std::string ScalarOpName;
  QLinearBinaryOpS8 QLinearS8Op;
  QLinearBinaryOpU8 QLinearU8Op;
  MatrixGuardBuffer<uint8_t> BufferInputA;
  MatrixGuardBuffer<uint8_t> BufferInputB;
  MatrixGuardBuffer<uint8_t> BufferOutput;
  MatrixGuardBuffer<uint8_t> BufferOutputReference;

  // Computed in double rather than float: at the extreme quantization scales
  // exercised by the extreme-scale tests below, the float32 reference used
  // to compute ValueA/ValueB/ValueC can itself overflow to Inf, and casting
  // an Inf or NaN float to int is undefined behavior. Double has enough
  // range/precision that this reference stays exact for the scales used
  // throughout this file (existing tests use scales of 10/18/90, which are
  // unaffected by this change).
  template <typename T>
  T QLinearBinaryScalar(T a,
                        float ScaleA,
                        int32_t ZeroPointA,
                        T b,
                        float ScaleB,
                        int32_t ZeroPointB,
                        float ScaleC,
                        int32_t ZeroPointC) {
    constexpr double qmax = std::numeric_limits<T>::max();
    constexpr double qmin = std::numeric_limits<T>::min();

    double ValueA = static_cast<double>(ScaleA) * (static_cast<int>(a) - ZeroPointA);
    double ValueB = static_cast<double>(ScaleB) * (static_cast<int>(b) - ZeroPointB);
    double ValueC = std::nearbyint(ScalarOp(ValueA, ValueB) / static_cast<double>(ScaleC)) + ZeroPointC;
    ValueC = std::min(ValueC, qmax);
    ValueC = std::max(ValueC, qmin);
    return static_cast<T>(static_cast<int>(ValueC));
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
  explicit MlasQLinearBinaryOpTest(std::function<double(double, double)> P_ScalarOp,
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
                             [](double a, double b) { return a + b; },
                             "+",
                             MlasQLinearAdd<int8_t>,
                             MlasQLinearAdd<uint8_t>) {}

  static const char* GetTestSuiteName() {
    static const std::string suite_name("QLinearAdd");
    return suite_name.c_str();
  }
};

class MlasQLinearMulTest : public MlasQLinearBinaryOpTest {
 private:
  // QLinearMul's ARM64 NEON kernel precomputes ScaleA * ScaleB / ScaleC as a
  // single float32 scalar before scaling the dequantized integer product.
  // ScaleA * ScaleB alone can overflow float32 (or, in the other direction,
  // round into subnormals or flush to zero) even when the true ratio is a
  // small, well-represented number; this exercises that overflow/underflow
  // boundary with deterministic inputs cycling between a zero delta
  // (a == ZeroPointA, the Inf*0 = NaN case) and large nonzero deltas (the
  // Inf*nonzero = saturate case), rotated across N so both cases land in
  // different lanes of the kernel's 16-wide vector chunks and its narrower
  // final chunk. When ZeroPointA sits at the type's boundary a nonzero
  // delta only has room to saturate in one direction, so at those zero
  // points both nonzero-delta cases collapse onto that one direction; the
  // non-boundary zero points in ExecuteShort's zero-point set still exercise
  // both saturation directions. QLinearAdd has no product term and is not
  // affected by this class of bug.
  template <typename T>
  void TestExtremeScale(void(MLASCALL* QLinearBinaryOp)(
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
    constexpr int MinimumValue = (int)std::numeric_limits<T>::min();
    constexpr int MaximumValue = (int)std::numeric_limits<T>::max();

    T* InputA = (T*)BufferInputA.GetBuffer(N);
    T* InputB = (T*)BufferInputB.GetBuffer(IsScalarB ? 1 : N);
    T* OutputC = (T*)BufferOutput.GetBuffer(N);
    T* OutputReference = (T*)BufferOutputReference.GetBuffer(N);

    // A fixed nonzero delta for B that never wraps past the type's range,
    // regardless of which zero point is used.
    const int DeltaB = (ZeroPointB < MaximumValue) ? 1 : -1;
    const T FixedInputB = static_cast<T>(ZeroPointB + DeltaB);
    for (size_t n = 0; n < (IsScalarB ? size_t(1) : N); n++) {
      InputB[n] = FixedInputB;
    }

    for (size_t n = 0; n < N; n++) {
      int delta;
      switch (n % 3) {
        case 0:
          delta = 0;  // a == ZeroPointA: exercises the Inf * 0 = NaN case
          break;
        case 1:
          delta = 100;  // exercises Inf * nonzero, saturating away from ZeroPointA
          break;
        default:
          delta = -100;  // exercises Inf * nonzero, saturating the other direction
          break;
      }
      // If ZeroPointA already sits at the type's boundary, clamping a delta
      // that pushes further past that boundary collapses back to
      // ZeroPointA itself, silently turning this into another zero-delta
      // case instead of the intended nonzero-delta case. Flip to the other
      // direction instead so a nonzero delta always stays nonzero, even
      // though at that boundary both the delta=100 and delta=-100 cases
      // then land on the same saturated value (see the class comment above).
      int v = std::min(std::max(ZeroPointA + delta, MinimumValue), MaximumValue);
      if (delta != 0 && v == ZeroPointA) {
        v = std::min(std::max(ZeroPointA - delta, MinimumValue), MaximumValue);
      }
      InputA[n] = static_cast<T>(v);
      OutputReference[n] = QLinearBinaryScalar(InputA[n], ScaleA, ZeroPointA, InputB[IsScalarB ? 0 : n], ScaleB, ZeroPointB, ScaleC, ZeroPointC);
    }

    QLinearBinaryOp(InputA, ScaleA, ZeroPointA, InputB, ScaleB, ZeroPointB, ScaleC, ZeroPointC, OutputC, N, IsScalarB);

    for (size_t n = 0; n < N; n++) {
      int diff = (int)OutputC[n] - (int)OutputReference[n];
      ASSERT_TRUE(diff >= -1 && diff <= 1)
          << "TestExtremeScale, IsScalarB=" << static_cast<int>(IsScalarB) << ", @" << n << " of " << N << ", "
          << static_cast<int>(InputA[n]) << "(" << ScaleA << "," << ZeroPointA << "), "
          << static_cast<int>(InputB[IsScalarB ? 0 : n]) << "(" << ScaleB << "," << ZeroPointB << ") ==> "
          << static_cast<int>(OutputC[n]) << "(" << ScaleC << "," << ZeroPointC << "), "
          << " expecting:" << static_cast<int>(OutputReference[n]);
    }
  }

 public:
  MlasQLinearMulTest() : MlasQLinearBinaryOpTest(
                             [](double a, double b) { return a * b; },
                             "*",
                             MlasQLinearMul<int8_t>,
                             MlasQLinearMul<uint8_t>) {}

  static const char* GetTestSuiteName() {
    static const std::string suite_name("QLinearMul");
    return suite_name.c_str();
  }

  void ExecuteShort(void) override {
    MlasQLinearBinaryOpTest::ExecuteShort();

// The scale-ratio overflow/underflow this covers is only fixed in the ARM64
// NEON kernel; the SSE2, LSX, VSX, ZVECTOR and generic C++ kernels compute
// the ratio the same lossy way (or worse, see the PR description), so this
// coverage would fail on those platforms until they are fixed too.
#if defined(MLAS_TARGET_ARM64)
    struct ScaleTriple {
      float ScaleA;
      float ScaleB;
      float ScaleC;
    };
    // Symmetric overflow (both scales huge); asymmetric overflow (one scale
    // huge, one ordinary); underflow (both scales tiny, product subnormal);
    // and two guards that must keep passing: a triple right at the edge of
    // the float32 overflow boundary (should already have been exact), and
    // the counterexample that rules out the rejected `(ScaleA/ScaleC)*ScaleB`
    // reassociation (that expression overflows here; ScaleA*ScaleB does not).
    static const ScaleTriple extreme_scales[] = {
        {1.85e19f, 1.85e19f, 1.71125e38f},
        {3.0e38f, 2.0f, 3.0e38f},
        {3.0e-23f, 3.0e-23f, 1.4013e-45f},
        {1.7e19f, 2.0e19f, 1.7e38f},
        {1.0e20f, 1.0e-38f, 1.0e-20f},
    };
    static const uint8_t zero_points_a[] = {0, 128, 255};
    static const uint8_t zero_points_b[] = {128, 255, 0};
    static const uint8_t zero_points_c[] = {255, 0, 128};
    const int8_t* s_zero_points_a = (const int8_t*)(&zero_points_a[0]);
    const int8_t* s_zero_points_b = (const int8_t*)(&zero_points_b[0]);
    const int8_t* s_zero_points_c = (const int8_t*)(&zero_points_c[0]);

    static const size_t test_sizes[] = {1, 15, 16, 17, 32, 33};

    for (const auto& scale : extreme_scales) {
      for (size_t zp = 0; zp < _countof(zero_points_a); zp++) {
        for (size_t n : test_sizes) {
          TestExtremeScale<uint8_t>(QLinearU8Op, n, false,
                                    scale.ScaleA, zero_points_a[zp], scale.ScaleB, zero_points_b[zp],
                                    scale.ScaleC, zero_points_c[zp]);
          TestExtremeScale<uint8_t>(QLinearU8Op, n, true,
                                    scale.ScaleA, zero_points_a[zp], scale.ScaleB, zero_points_b[zp],
                                    scale.ScaleC, zero_points_c[zp]);
          TestExtremeScale<int8_t>(QLinearS8Op, n, false,
                                   scale.ScaleA, s_zero_points_a[zp], scale.ScaleB, s_zero_points_b[zp],
                                   scale.ScaleC, s_zero_points_c[zp]);
          TestExtremeScale<int8_t>(QLinearS8Op, n, true,
                                   scale.ScaleA, s_zero_points_a[zp], scale.ScaleB, s_zero_points_b[zp],
                                   scale.ScaleC, s_zero_points_c[zp]);
        }
      }
    }
#endif  // defined(MLAS_TARGET_ARM64)
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

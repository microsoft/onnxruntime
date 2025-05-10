#include "test_util.h"
#include "mlas_qnbit.h"

#undef GGML_COMMON_DECL
#define GGML_COMMON_DECL_C
#include "../../ggml/src/ggml-common.h"
#include "core/framework/float16.h"

///
#include "test/common/random_generator.h"
#include "core/common/span_utils.h"
///

class MlasLowBitQgemmTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<float> BufferA;
  MatrixGuardBuffer<int8_t> BufferQuantAData;
  MatrixGuardBuffer<float> BufferB;
  MatrixGuardBuffer<uint8_t> BufferQuantBData;
  MatrixGuardBuffer<float> BufferDequantizedB;
  MatrixGuardBuffer<float> BufferBias;
  MatrixGuardBuffer<float> BufferC;
  MatrixGuardBuffer<float> BufferCReference;

  void CallReferenceGemm(
      size_t M,
      size_t N,
      size_t K,
      const float* A,
      const float* B,
      const float* Bias,
      float* C) {
    float* c = C;
    for (size_t m = 0; m < M; ++m) {
      const float* a = A + m * K;
      for (size_t n = 0; n < N; ++n) {
        float sum = 0.0f;
        for (size_t k = 0; k < K; ++k) {
          sum += a[k] * B[n * K + k];  // column major
        }
        if (Bias != nullptr) {
          sum += Bias[n];
        }
        c[m * N + n] = sum;
      }
    }
  }

 public:
  void QuantizeDequantize(std::vector<float>& raw_vals,
                          std::vector<uint8_t>& quant_vals,
                          int32_t N,
                          int32_t K,
                          const std::string& quant_type_name,
                          MLAS_THREADPOOL* tp) {
    size_t quant_size = MlasLowBitQuantizeSizeInByte(N, K, quant_type_name);
    quant_vals.resize(quant_size);

    MlasLowBitQuantize(&raw_vals[0], N, K, quant_type_name, &quant_vals[0], tp);

    size_t dequant_size = MlasLowBitDequantizeDataCount(N, K, quant_type_name);
    raw_vals.resize(dequant_size);

    MlasLowBitDequantize(&quant_vals[0], N, K, quant_type_name, &raw_vals[0], tp);
  }

  void Test(size_t M, size_t N, size_t K, size_t BatchSize, const std::string& a_type_name, const std::string& b_type_name,
            bool WithThreadpool, bool WithBias) {
    MLAS_THREADPOOL* Threadpool = WithThreadpool ? GetMlasThreadPool() : nullptr;

    bool requantize_a = MlasLowBitCanQuantize(a_type_name) && MlasLowBitCanDequantize(a_type_name);

    onnxruntime::test::RandomValueGenerator random{1234};
    std::vector<float> input0_vals(random.Gaussian<float>(onnxruntime::AsSpan({(int64_t)M, (int64_t)K}), 0.0f, requantize_a ? 25.0f : 0.25f));
    std::vector<float> input1_f_vals(random.Gaussian<float>(onnxruntime::AsSpan({(int64_t)K, (int64_t)N}), 0.0f, requantize_a ? 25.0f : 0.25f));
    //input1_f_vals[0] = 0;
    //for (int i = 4; i < K; i++) {
    //  input0_vals[i] = 0;
    //  input1_f_vals[i] = 0;
    //}

    const float* Bias = nullptr;
    if (WithBias) {
      Bias = BufferBias.GetBuffer(N);
      throw std::runtime_error("bias not supported with lowbit matmul");
    }

    float* C = BufferC.GetBuffer(N * M, true);
    float* CReference = BufferCReference.GetBuffer(N * M, true);

    size_t a_quant_size = MlasLowBitQuantizeSizeInByte(M, K, a_type_name);
    std::vector<uint8_t> a_quant_data(a_quant_size);
    size_t b_quant_size = MlasLowBitQuantizeSizeInByte(N, K, b_type_name);
    std::vector<uint8_t> b_quant_data(b_quant_size);

    QuantizeDequantize(input1_f_vals, b_quant_data, static_cast<int32_t>(N), static_cast<int32_t>(K), b_type_name, Threadpool);

    if (requantize_a) {
      QuantizeDequantize(input0_vals, a_quant_data, static_cast<int32_t>(M), static_cast<int32_t>(K), a_type_name, Threadpool);
    } else {
      MlasLowBitQuantize(&input0_vals[0], M, K, a_type_name, &a_quant_data[0], Threadpool);

    }
    CallReferenceGemm(M, N, K, &input0_vals[0], &input1_f_vals[0], Bias, CReference);

    MlasLowBitQGemmBatch(M, N, K, BatchSize, &a_quant_data[0], a_type_name, &b_quant_data[0], b_type_name, C, Threadpool);

    size_t f = 0;
    for (size_t m = 0; m < M; m++) {
      for (size_t n = 0; n < N; n++, f++) {
        ASSERT_TRUE(CloseEnough(C[f], CReference[f], 0.05f))
            << "Expected: " << CReference[f] << " Actual: " << C[f] << "@[" << m << "x" << n << "], "
            << "M=" << M << ", N=" << N << ", K=" << K;
      }
    }
  }

 public:
  static const char* GetTestSuiteName() {
    static std::string suite_name = std::string("MlasLowBitQGemm");
    return suite_name.c_str();
  }
};

//
// Short Execute() test helper to register each test separately by all parameters.
//
class LowBitQgemmShortExecuteTest : public MlasTestFixture<MlasLowBitQgemmTest> {
 public:
  explicit LowBitQgemmShortExecuteTest(size_t M, size_t N, size_t K, size_t BatchSize,
                                       const std::string& a_type_name, const std::string b_type_name, bool WithThreadpool, bool WithBias)
      : M_(M),
        N_(N),
        K_(K),
        BatchSize_(BatchSize),
        a_type_name_(a_type_name),
        b_type_name_(b_type_name),
        WithThreadpool_(WithThreadpool),
        WithBias_(WithBias) {
  }

  void TestBody() override {
    MlasTestFixture<MlasLowBitQgemmTest>::mlas_tester->Test(
        M_, N_, K_, BatchSize_, a_type_name_, b_type_name_,
        WithThreadpool_, WithBias_);
  }

  static size_t RegisterSingleTest(size_t M, size_t N, size_t K,
                                   size_t BatchSize,
                                   const std::string& a_type_name, const std::string b_type_name,
                                   bool WithThreadpool, bool WithBias) {
    size_t tests_registered = 0;

    std::stringstream ss;
    ss << b_type_name << "_" << a_type_name
       << (WithThreadpool ? "SingleThread" : "Threaded")
       << "/M" << M << "xN" << N << "xK" << K
       << "/hasBias" << WithBias;
    auto test_name = ss.str();

    testing::RegisterTest(
        MlasLowBitQgemmTest::GetTestSuiteName(),
        test_name.c_str(),
        nullptr,
        test_name.c_str(),
        __FILE__,
        __LINE__,
        // Important to use the fixture type as the return type here.
        [=]() -> MlasTestFixture<MlasLowBitQgemmTest>* {
          return new LowBitQgemmShortExecuteTest(M, N, K, BatchSize, a_type_name, b_type_name, WithThreadpool, WithBias);
        });

    tests_registered += 1;
    return tests_registered;
  }

  static size_t RegisterShortExecuteTests() {
    const char* type_name_pairs[][2] = {
        /*GGML_TYPE_Q4_0*/ {"q8_0", "q4_0"},
        /*GGML_TYPE_Q4_1*/ {"q8_1", "q4_1"},
        /*GGML_TYPE_Q5_0*/ {"q8_0", "q5_0"},
        /*GGML_TYPE_Q5_1*/ {"q8_1", "q5_1"},
        /*GGML_TYPE_Q8_0*/ {"q8_0", "q8_0"},
        ///*GGML_TYPE_Q8_1*/ {"q8_1", "q8_1"}, // no vec_dot
        /*GGML_TYPE_Q2_K*/ {"q8_K", "q2_K"},
        /*GGML_TYPE_Q3_K*/ {"q8_K", "q3_K"},
        /*GGML_TYPE_Q4_K*/ {"q8_K", "q4_K"},
        /*GGML_TYPE_Q5_K*/ {"q8_K", "q5_K"},
        /*GGML_TYPE_Q6_K*/ {"q8_K", "q6_K"},
        // /*GGML_TYPE_IQ2_XXS*/ {"q8_K", "iq2_xxs"}, // no quantization function
        // /*GGML_TYPE_IQ2_XS*/ {"q8_K", "iq2_xs"},
        // /*GGML_TYPE_IQ3_XXS*/ {"q8_K", "iq3_xxs"},
        // /*GGML_TYPE_IQ3_S*/ {"q8_K", "iq3_s"},
        // /*GGML_TYPE_IQ2_S*/ {"q8_K", "iq2_s"},
        // /*GGML_TYPE_IQ1_S*/ {"q8_K", "iq1_s"},
        // /*GGML_TYPE_IQ1_M*/ {"q8_K", "iq1_m"},
        ///////*GGML_TYPE_IQ4_NL*/ {"q8_0", "iq4_nl"},
        ///////*GGML_TYPE_IQ4_XS*/ {"q8_K", "iq4_xs"},
        // /*GGML_TYPE_Q8_K*/ {"q8_K", "iq4_xs"},
        /*GGML_TYPE_TQ1_0*/ {"q8_K", "tq1_0"},
        /*GGML_TYPE_TQ2_0*/ {"q8_K", "tq2_0"},
        /*GGML_TYPE_I2_S*/ {"i8_s", "i2_s"},
    };

    // std::string a_type_name("q8_0"), b_type_name("q4_0");
    size_t tests_registered = 0;
    size_t BatchSize = 1;
    for (const auto& [a_type_name, b_type_name] : type_name_pairs) {
      for (size_t M : {1, 2}) {
        for (size_t N : {1, 2}) {
          for (size_t K : {256, 512}) {
            for (bool use_thread_pool : {true, false}) {
              for (bool has_bias : {false}) {
                tests_registered += RegisterSingleTest(M, N, K, BatchSize, a_type_name, b_type_name, use_thread_pool, has_bias);
              }
            }
          }
        }
      }
    }
    return tests_registered;
  }

 private:
  size_t M_, N_, K_;
  size_t BatchSize_{1};
  std::string a_type_name_;  // ("q8_0");
  std::string b_type_name_;  // ("q4_0");
  bool WithThreadpool_, WithBias_;
};

static size_t SLowBitQGemmRegisterAllShortExecuteTests() {
  // MLasInitLlama();
  size_t count = 0;
  // TODO: enable these test for 2bit development.
  // count += SQNBitGemmShortExecuteTest<2, 16>::RegisterShortExecuteTests();
  // count += SQNBitGemmShortExecuteTest<2, 32>::RegisterShortExecuteTests();
  // count += SQNBitGemmShortExecuteTest<2, 64>::RegisterShortExecuteTests();
  // count += SQNBitGemmShortExecuteTest<2, 128>::RegisterShortExecuteTests();
  // count += SQNBitGemmShortExecuteTest<2, 256>::RegisterShortExecuteTests();

  count += LowBitQgemmShortExecuteTest::RegisterShortExecuteTests();
  return count;
}

static UNUSED_VARIABLE bool added_to_main = AddTestRegister(
    [](bool is_short_execute) -> size_t {
      if (is_short_execute) {
        return SLowBitQGemmRegisterAllShortExecuteTests();
      }
      return 0;
    });

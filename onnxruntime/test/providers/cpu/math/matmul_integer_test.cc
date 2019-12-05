// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

#include <random>

#if 1
#include <mkl_cblas.h>
#include <chrono>
typedef std::chrono::high_resolution_clock Clock;

#endif

namespace onnxruntime {
namespace test {

TEST(MatmulIntegerOpTest, MatMulInteger_2D) {
  OpTester test("MatMulInteger", 10);
  test.AddInput<uint8_t>("T1", {4, 3}, {11, 7, 3, 10, 6, 2, 9, 5, 1, 8, 4, 0});
  test.AddInput<uint8_t>("T2", {3, 2}, {1, 4, 2, 5, 3, 6});
  test.AddInput<uint8_t>("a_zero_point", {}, {12});
  test.AddInput<uint8_t>("b_zero_point", {}, {0});
  test.AddOutput<int32_t>("T3", {4, 2}, {-38, -83, -44, -98, -50, -113, -56, -128});
  test.Run();
}

TEST(MatmulIntegerOpTest, MatMulInteger) {
  OpTester test("MatMulInteger", 10);
  test.AddInput<uint8_t>("T1", {1, 1}, {11});
  test.AddInput<uint8_t>("T2", {1, 1}, {13});
  test.AddInput<uint8_t>("a_zero_point", {}, {12});
  test.AddInput<uint8_t>("b_zero_point", {}, {12});
  test.AddOutput<int32_t>("T3", {1, 1}, {-1});
  test.Run();
}
TEST(MatmulIntegerOpTest, MatMulInteger_WithZero_ZeroPoint) {
  OpTester test("MatMulInteger", 10);
  test.AddInput<uint8_t>("T1", {4, 3}, {11, 7, 3, 10, 6, 2, 9, 5, 1, 8, 4, 0});
  test.AddInput<uint8_t>("T2", {3, 2}, {1, 4, 2, 5, 3, 6});
  test.AddInput<uint8_t>("a_zero_point", {}, {0});
  test.AddInput<uint8_t>("b_zero_point", {}, {0});
  test.AddOutput<int32_t>("T3", {4, 2}, {34, 97, 28, 82, 22, 67, 16, 52});
  test.Run();
}

template <typename T>
std::vector<T> ToVector(const int* value, int size) {
  std::vector<T> data(size);
  for (int i = 0; i < size; i++)
    data[i] = static_cast<T>(value[i]);
  return data;
}

// [M x N] = [M x K] x [K x N] = [batch_seq x input_dim] x [input_dim x embed_dim]
void RunMatMulIntegerU8S8Test(const int M, const int N, const int K) {
  OpTester test("MatMulInteger", 10);
  static std::default_random_engine e(123);
  static std::uniform_int_distribution<int> n_unsigned(0, 127);
  static std::uniform_int_distribution<int> n_signed(-128, 127);
  Eigen::MatrixXi T1 = Eigen::MatrixXi::Random(K, M)
                           .unaryExpr([](int) { return n_unsigned(e); });
  Eigen::MatrixXi T2 = Eigen::MatrixXi::Random(N, K)
                           .unaryExpr([](int) { return n_signed(e); });
  Eigen::MatrixXi T3 = (T2 * T1).eval();

  test.AddInput<uint8_t>("T1", {M, K},
                         ToVector<uint8_t>(T1.data(), M * K));
  test.AddInput<int8_t>("T2", {K, N},
                        ToVector<int8_t>(T2.data(), K * N), /*is_initializer*/ true);
  test.AddOutput<int32_t>("T3", {M, N},
                          ToVector<int32_t>(T3.data(), M * N));

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kNGraphExecutionProvider});  // currently nGraph provider does not support gemm_u8s8
}
#if 1

template <typename T>
std::vector<T> ToVector2(const int* value, int size) {
  std::vector<T> data(size);
  for (int i = 0; i < size; i++) {
    data[i] = static_cast<T>(value[i]);
    //std::cout << "data " << data[i] << " vs " << value[i] << std::endl;
  }
  return data;
}

// [M x N] = [M x K] x [K x N] = [batch_seq x input_dim] x [input_dim x embed_dim]
void xxx(const int M, const int N, const int K) {
  static std::default_random_engine e(123);
  static std::uniform_int_distribution<int> n_unsigned(0, 127);
  static std::uniform_int_distribution<int> n_signed(-128, 127);
  static std::uniform_real_distribution<float> n_float(-100, 100);
  Eigen::MatrixXi T1 = Eigen::MatrixXi::Random(K, M)
                           .unaryExpr([](int) { return n_signed(e); });
  Eigen::MatrixXi T2 = Eigen::MatrixXi::Random(N, K)
                           .unaryExpr([](int) { return n_signed(e); });
  Eigen::MatrixXi T3 = (T2 * T1).eval();

  std::vector<float> t1 = ToVector2<float>(T1.data(), M * K);
  std::vector<float> t2 = ToVector2<float>(T2.data(), K * N);
  std::vector<float> t3 = ToVector2<float>(T3.data(), M * N);

  std::vector<float> res(M * N, 0);

  // warm up
  cblas_sgemm(CBLAS_LAYOUT::CblasColMajor, CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasNoTrans,
              N, M, K,
              1.0f, t2.data(), N,
              t1.data(), K, 0, res.data(), N);

  auto begin = Clock::now();
  for (int t = 0; t < 1000; t++) {
    cblas_sgemm(CBLAS_LAYOUT::CblasColMajor, CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasNoTrans,
                N, M, K,
                1.0f, t2.data(), N,
                t1.data(), K, 0, res.data(), N);
  }
  auto end = Clock::now();
  auto mkl1 = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
  std::cout << "mkl base:" << (double)(mkl1) / (double)(1000000.0) << " ms" << std::endl;

#if 1
  for (int i = 0; i < M * N; i++) {
    //std::cout << "eigen:" << t3[i] << " vs mkl " << res[i] << std::endl;
    if (t3[i] != res[i]) {
      std::cout << "i :" << i << ", t3: " << t3[i] << " vs mkl" << res[i] << std::endl;
      ORT_THROW("Failed! results wrong!");
    }
  }
#endif
  // pack routine
  // pack t2
  std::vector<float> res2(M * N, 0);
  std::vector<float> a_pack_buf;
  std::vector<float> b_pack_buf;
#if 0
            bool pack_a = true;
            bool pack_b = true;
#else
  bool pack_a = true;
  bool pack_b = false;
#endif

  //int mark_a = 0;
  if (pack_a) {
    int a_sz = cblas_sgemm_pack_get_size(CBLAS_IDENTIFIER::CblasAMatrix, N, M, K);
    a_pack_buf.resize(a_sz);
    std::cout << "meme resize: " << (double)(a_sz) / 1024.0 / 1024.0 << " MB vs " << (double)(N * K) / 1024.0 / 1024.0 << " MB" << std::endl;
    cblas_sgemm_pack(CBLAS_LAYOUT::CblasColMajor, CBLAS_IDENTIFIER::CblasAMatrix, CBLAS_TRANSPOSE::CblasNoTrans, N, M, K, 1.0f, t2.data(), N, a_pack_buf.data());
#if 0
                std::cout << "$$$$$$$$$$$$$$$ resize: " << a_sz << std::endl;
                for (int i=0;i<a_sz;i++) {
                    std::cout<< static_cast<int32_t>(a_pack_buf[i]) << ".";
                }
                int32_t prev = 0;
                for (int i=0; i<a_sz; i++) {
                    auto now = static_cast<int32_t>(a_pack_buf[i]);
                    if (i>0 && prev == 1 && now == 1) {
                        mark_a = i-1;
                        std::cout << "find index: " << i-1 << " xx "<< ", " << std::endl;
                        break;
                    }	
                    prev = now;
                }
                
                if (mark_a !=0)
                for (int i=0; i< K*N; i++) {
                    std::cout << static_cast<int32_t>(a_pack_buf[i+mark_a-1]) << ", ";
                }
#endif
  }

  //if (pack_b) {
  //  int b_sz = cblas_gemm_s8u8s32_pack_get_size(CBLAS_IDENTIFIER::CblasAMatrix, N, M, K);
  //  b_pack_buf.resize(b_sz);
  //  cblas_gemm_s8u8s32_pack(CBLAS_LAYOUT::CblasColMajor, CBLAS_IDENTIFIER::CblasBMatrix, CBLAS_TRANSPOSE::CblasNoTrans, N, M, K, t1.data(), K, b_pack_buf.data());

  //  for (int i = 0; i < b_sz; i++) {
  //    std::cout << static_cast<int32_t>(b_pack_buf[i]) << ".";
  //  }
  //}

  if (pack_a && pack_b) {
    //std::cout << "two side " << std::endl;

    //auto begin = Clock::now();
    //for (int t = 0; t < 1000; t++) {
    //  cblas_gemm_s8u8s32_compute(CBLAS_LAYOUT::CblasColMajor, CBLAS_STORAGE::CblasPacked, CBLAS_STORAGE::CblasPacked, CBLAS_OFFSET::CblasFixOffset,
    //                             N, M, K,
    //                             1, a_pack_buf.data(), K,
    //                             0, b_pack_buf.data(), K, 0, 0, res2.data(), N, &co);
    //}
    //auto end = Clock::now();
    //auto mkl3 = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    //std::cout << "mkl pack3:" << mkl3 << " ms" << std::endl;

  } else if (pack_a && !pack_b) {
    cblas_sgemm_compute(CBLAS_LAYOUT::CblasColMajor, CBLAS_STORAGE::CblasPacked, CBLAS_TRANSPOSE::CblasNoTrans,
                        N, M, K,
                        a_pack_buf.data(), K,
                        t1.data(), K, 0.0f, res2.data(), N);

    auto begin = Clock::now();
    for (int t = 0; t < 1000; t++) {
      cblas_sgemm_compute(CBLAS_LAYOUT::CblasColMajor, CBLAS_STORAGE::CblasPacked, CBLAS_TRANSPOSE::CblasNoTrans,
                          N, M, K,
                          a_pack_buf.data(), K,
                          t1.data(), K, 0.0f, res2.data(), N);
    }
    auto end = Clock::now();
    auto mkl2 = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout << "mkl pack:" << (double)(mkl2) / (double)(1000000.0) << " ms" << std::endl;
  } else if (!pack_a && pack_b) {
    //std::cout << "another side " << std::endl;

    //auto begin = Clock::now();
    //for (int t = 0; t < 1000; t++) {
    //  cblas_gemm_s8u8s32_compute(CBLAS_LAYOUT::CblasColMajor, CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_STORAGE::CblasPacked, CBLAS_OFFSET::CblasFixOffset,
    //                             N, M, K,
    //                             1, t2.data(), K,
    //                             0, b_pack_buf.data(), K, 0, 0, res2.data(), N, &co);
    //}
    //auto end = Clock::now();
    //auto mkl2 = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    //std::cout << "mkl pack:" << mkl2 << " ms" << std::endl;
  }

  for (int i = 0; i < M * N; i++) {
    //std::cout << "eigen:" << t3[i] << " vs mkl2 " <<res2[i] << std::endl;

    if (t3[i] != res2[i]) {
      std::cout << "i :" << i << ", t3: " << t3[i] << " vs mkl2" << res2[i] << std::endl;
      ORT_THROW("Failed! results wrong!");
    }
  }

  std::cout << "Pass!" << std::endl;
}

TEST(MatmulIntegerOpTest, sgemm1) {
  xxx(3072, 768, 768);
  xxx(768, 3072, 768);
}
TEST(MatmulIntegerOpTest, sgemm2) {
  xxx(128, 768, 768);
  xxx(768, 128, 768);
}
#endif

TEST(MatmulIntegerOpTest, MatMulInteger_Uint8_Int8_Scalar) {
  RunMatMulIntegerU8S8Test(1, 1, 32);
  RunMatMulIntegerU8S8Test(1, 1, 260);
  RunMatMulIntegerU8S8Test(1, 1, 288);
}

TEST(MatmulIntegerOpTest, MatMulInteger_Uint8_Int8_GEMV) {
  RunMatMulIntegerU8S8Test(1, 2, 16);
  RunMatMulIntegerU8S8Test(1, 2, 64);
}

TEST(MatmulIntegerOpTest, MatMulInteger_Uint8_Int8_GEMM) {
  RunMatMulIntegerU8S8Test(2, 2, 40);
  RunMatMulIntegerU8S8Test(2, 48, 33);
  RunMatMulIntegerU8S8Test(2, 51, 40);
}

}  // namespace test
}  // namespace onnxruntime

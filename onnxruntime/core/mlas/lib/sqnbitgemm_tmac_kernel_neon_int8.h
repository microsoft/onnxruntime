#include "stdint.h"
#ifdef __cplusplus
extern "C"
#endif
 int32_t qgemm_lut_t1_int8_m128_k4096_n1_b2(void* A, void* LUT, void* Scales, void* LUT_Scales, void* LUT_Biases, void* C);
#ifdef __cplusplus
extern "C"
#endif
 int32_t preprocessor_t1_int8_m8192_k4096_n1_b2(void* B, void* LUT_Scales, void* LUT_Biases, void* QLUT);
#ifdef __cplusplus
extern "C"
#endif
 int32_t qgemm_lut_t1_int8_m256_k4096_n1_b2(void* A, void* LUT, void* Scales, void* LUT_Scales, void* LUT_Biases, void* C);
#ifdef __cplusplus
extern "C"
#endif
 int32_t preprocessor_t1_int8_m28672_k4096_n1_b2(void* B, void* LUT_Scales, void* LUT_Biases, void* QLUT);
#ifdef __cplusplus
extern "C"
#endif
 int32_t qgemm_lut_t1_int8_m128_k14336_n1_b2(void* A, void* LUT, void* Scales, void* LUT_Scales, void* LUT_Biases, void* C);
#ifdef __cplusplus
extern "C"
#endif
 int32_t preprocessor_t1_int8_m8192_k14336_n1_b2(void* B, void* LUT_Scales, void* LUT_Biases, void* QLUT);
#ifdef __cplusplus
extern "C"
#endif
 int32_t qgemm_lut_t1_int8_m512_k4096_n1_b2(void* A, void* LUT, void* Scales, void* LUT_Scales, void* LUT_Biases, void* C);
#ifdef __cplusplus
extern "C"
#endif
 int32_t preprocessor_t1_int8_m2048_k4096_n1_b2(void* B, void* LUT_Scales, void* LUT_Biases, void* QLUT);

 inline int qgemm_lut_int8(int m, int k, int n, int b, void* A, void* LUT, void* Scales, void* LUT_Scales, void* LUT_Biases, void* C) {

    if (m == 128 && k == 4096 && n == 1 && b == 2) return qgemm_lut_t1_int8_m128_k4096_n1_b2(A, LUT, Scales, LUT_Scales, LUT_Biases, C);

    if (m == 256 && k == 4096 && n == 1 && b == 2) return qgemm_lut_t1_int8_m256_k4096_n1_b2(A, LUT, Scales, LUT_Scales, LUT_Biases, C);

    if (m == 128 && k == 14336 && n == 1 && b == 2) return qgemm_lut_t1_int8_m128_k14336_n1_b2(A, LUT, Scales, LUT_Scales, LUT_Biases, C);

    if (m == 512 && k == 4096 && n == 1 && b == 2) return qgemm_lut_t1_int8_m512_k4096_n1_b2(A, LUT, Scales, LUT_Scales, LUT_Biases, C);

    return -1;
}
inline int preprocessor_int8(int m, int k, int n, int b, void* B, void* LUT_Scales, void* LUT_Biases, void* QLUT) {

    if (m == 8192 && k == 4096 && n == 1 && b == 2) return preprocessor_t1_int8_m8192_k4096_n1_b2(B, LUT_Scales, LUT_Biases, QLUT);

    if (m == 28672 && k == 4096 && n == 1 && b == 2) return preprocessor_t1_int8_m28672_k4096_n1_b2(B, LUT_Scales, LUT_Biases, QLUT);

    if (m == 8192 && k == 14336 && n == 1 && b == 2) return preprocessor_t1_int8_m8192_k14336_n1_b2(B, LUT_Scales, LUT_Biases, QLUT);

    if (m == 2048 && k == 4096 && n == 1 && b == 2) return preprocessor_t1_int8_m2048_k4096_n1_b2(B, LUT_Scales, LUT_Biases, QLUT);

    return -1;
}

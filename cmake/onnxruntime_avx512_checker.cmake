check_cxx_source_compiles("
  int main() {
    asm(\"vpxord %zmm0,%zmm0,%zmm0\");
    return 0;
  }"
  COMPILES_AVX512F
)

check_cxx_source_compiles("
  #include <immintrin.h>
  int main() {
    __m512 zeros = _mm512_set1_ps(0.f);
    (void)zeros;
    return 0;
  }"
  COMPILES_AVX512F_INTRINSICS
)
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

check_cxx_source_compiles("
  int main() {
    asm(\"vpmaddwd %zmm0,%zmm0,%zmm0\"); // AVX512BW feature
    asm(\"vandnps %xmm31,%xmm31,%xmm31\"); // AVX512DQ/AVX512VL feature
    return 0;
  }"
  COMPILES_AVX512CORE
)
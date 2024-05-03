#include <cstdint>
#include <cstddef>

bool
llamafile_sgemm(int64_t m, int64_t n, int64_t k, const std::byte *A, int64_t lda, const std::byte *B, int64_t ldb, float *C, int64_t ldc, const float *QuantBScale, int64_t StrideQuantBScale);
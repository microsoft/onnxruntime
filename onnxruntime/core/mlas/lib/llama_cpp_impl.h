size_t
quantize_i2_s(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights);

void
dequantize_i2_s(const void* src, float* dst, int64_t nrow, int64_t n_per_row);

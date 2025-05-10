// add quantize, dequantize, and dot methods too specific to be in llama.cpp there
#include <cstdint>
#include <math.h>
#include <stdlib.h>
#include <cstring>
#include "ggml.h"

#define QK_I2_S 128
#define QK_I2 128

size_t
quantize_i2_s(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* /*quant_weights*/)
{
    // 2 bits per weight

    size_t row_size = ggml_row_size(GGML_TYPE_I2_S, n_per_row);

    int n = static_cast<int>(nrow * n_per_row);

    // f32 -> q8
    double max = 0;
    for (int i = 0; i < n; ++i) {
        max = fmax(max, (double)fabs((double)src[i]));
    }
    double i2_scale = max;

    uint8_t* q8 = (uint8_t*)malloc(n * sizeof(uint8_t));
    for (int i = 0; i < n; i++) {
        if (fabs((double)(src[i])) < 1e-6) {
            q8[i] = 1;
            continue;
        }
        q8[i] = (double)src[i] * i2_scale > 0 ? 2 : 0;
    }

    memset(dst, 0, n * sizeof(uint8_t) / 4);

    // q8 -> 0, 1, 2
    //       |  |  |
    //      -1, 0, 1

    uint8_t* i2_weight = (uint8_t*)dst;
    for (int i = 0; i < n / QK_I2; i++) {
        for (int j = 0; j < QK_I2; j++) {
            int group_idx = j / 32;
            int group_pos = j % 32;
            uint8_t temp = (q8[i * QK_I2 + j] << (6 - 2 * group_idx));
            i2_weight[i * 32 + group_pos] |= temp;
        }
    }

    float* scale_ptr = (float*)((char*)i2_weight + n / 4);
    scale_ptr[0] = static_cast<float>(i2_scale);

    free(q8);

    // 32B for alignment
    return nrow * row_size / 4 + 32;
}

void
dequantize_i2_s(const void* src, float* dst, int64_t nrow, int64_t n_per_row)
{
    // 2 bits per weight, reverse of quantize_i2_s

    const uint8_t* i2_weight = (const uint8_t*)src;
    const float* scale_ptr = (const float*)((const char*)i2_weight + (nrow * n_per_row / 4));
    float i2_scale = scale_ptr[0];

    int n = static_cast<int>(nrow * n_per_row);

    // Initialize output
    memset(dst, 0, n * sizeof(float));

    // Convert packed 2-bit weights back to float
    for (int i = 0; i < n / QK_I2; i++) {
        for (int j = 0; j < QK_I2; j++) {
            int group_idx = j / 32;
            int group_pos = j % 32;
            // Extract 2-bit value
            uint8_t packed = i2_weight[i * 32 + group_pos];
            uint8_t value = (packed >> (6 - 2 * group_idx)) & 0x3;

            // Map back to {-1, 0, 1}
            float dequantized;
            switch (value) {
                case 0:
                    dequantized = -1.0f;
                    break;
                case 1:
                    dequantized = 0.0f;
                    break;
                case 2:
                    dequantized = 1.0f;
                    break;
                default:
                    dequantized = 0.0f;
                    break;  // Should not occur
            }

            // Apply scale
            dst[i * QK_I2 + j] = dequantized * i2_scale;
        }
    }
}

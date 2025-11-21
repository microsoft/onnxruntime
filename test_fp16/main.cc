#include <arm_neon.h>
#include <cstdio>

int main() {
    int16x4_t a_int = vdup_n_s16(int16_t{1});
    int16x4_t b_int = vdup_n_s16(int16_t{2});

    float16x4_t a = vcvt_f16_s16(a_int);
    float16x4_t b = vcvt_f16_s16(b_int);
    float16x4_t c = vadd_f16(a, b);

    int16_t buffer[4]{};
    vst1_f16(buffer, c);

    printf("%d\n", static_cast<int>(buffer[0]));

    return 0;
}

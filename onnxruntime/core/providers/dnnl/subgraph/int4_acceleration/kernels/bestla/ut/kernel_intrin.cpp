#include "kernel_ut.h"
#include "kernel_avx2.h"
#include "kernel_avx512f.h"
namespace bestla {
using namespace utils;
namespace ut {
class UT_Avx512f_decompress_kblock_s4_fp {
 public:
  UT_Avx512f_decompress_kblock_s4_fp() {
    UT_START();
    CheckISA(AVX512F);
    ut<BTLA_DTYPE::S4_CLIP, 2, float, utils::bf16>(32, 128, 128, 128, 0, 32, 128);
    ut<BTLA_DTYPE::S4_CLIP, 2, float, utils::bf16>(32, 96, 96, 96, 0, 32, 96);
    ut<BTLA_DTYPE::S4_CLIP, 1, float, float>(32, 48, 48, 128, 0, 32, 128);
    ut<BTLA_DTYPE::S4_CLIP, 1, float, utils::bf16>(32, 48, 48, 128, 0, 32, 128);
  }

  template <BTLA_DTYPE S4_T, int PACK_ROW, typename ST_T, typename DST_T>
  void ut(int row, int col, int ld_src, int ld_dst, int k_offset, int kblock, int NPad, bool asym = false) {
    printf("Test Case %s_%d_%d: %d %d %d %d %d %d %d %d\n", __FUNCTION__, int(S4_T), PACK_ROW, row, col, ld_src, ld_dst,
           k_offset, kblock, NPad, asym);
    std::vector<utils::int4x2> s4_wei(row * col / 2);
    std::vector<int8_t> s8_wei(col * row);
    std::vector<DST_T> bf16_wei(ld_dst * row);
    std::vector<DST_T> ref_wei(ld_dst * row);
    std::vector<ST_T> scales(col);
    std::vector<int8_t> zero_points(col);
    fill_buffer_randn(s8_wei.data(), s8_wei.size(), int8_t(-128), int8_t(127));
    fill_buffer_randn(scales.data(), scales.size(), ST_T(0.01f), ST_T(0.02f));
    fill_buffer_randn(zero_points.data(), zero_points.size(), (int8_t)(-5), (int8_t)(5));

    for (int i = 0; i < col * row; i += 2) {
      s4_wei[i / 2].x = utils::int4x2::convert(s8_wei[i]);
      s4_wei[i / 2].y = utils::int4x2::convert(s8_wei[i + 1]);
    }
    kernel::ref::decompress_kblock_s4_fp<S4_T, DST_T, PACK_ROW, ST_T>(
        s4_wei.data(), ref_wei.data(), row, col, ld_src, ld_dst, scales.data(), asym ? zero_points.data() : nullptr,
        k_offset, kblock, NPad, cache, CacheSize);
    kernel::avx512f::decompress_kblock_s4_fp<S4_T, DST_T, PACK_ROW, ST_T>(
        s4_wei.data(), bf16_wei.data(), row, col, ld_src, ld_dst, scales.data(), asym ? zero_points.data() : nullptr,
        k_offset, kblock, NPad, cache, CacheSize);
    ut::buffer_error(ref_wei.data(), bf16_wei.data(), bf16_wei.size(), DST_T(BF16_ERR));
  }
};
#ifdef BTLA_KERNEL_INTRIN
static UT_Avx512f_decompress_kblock_s4_fp sUT_Avx512f_decompress_kblock_s4_fp;
#endif
}  // namespace ut
}  // namespace bestla

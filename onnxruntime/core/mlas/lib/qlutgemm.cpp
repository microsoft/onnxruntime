/*++

// TODO: finish filling this out

module includes kernel functions for generating LUT for T-MAC GEMM optimization strategy.
*/
#include <string>
#include <thread>
#include <unordered_map>

#include "qlutgemm.h"

/** T-MAC GEMM kernel Config */
static std::unordered_map<std::string, struct MlasTMACKernelParams> tmac_kernel_configs;




const MlasTMACKernelParams& GetTMACKernelParams(size_t M, size_t N, size_t nbits, size_t block_size) {
    std::string key = std::to_string(M) + "_" + std::to_string(N) + "_" + std::to_string(nbits);
    if (tmac_kernel_configs.count(key)) {
        return tmac_kernel_configs[key];
    }

    MlasTMACKernelParams params;
    params.g = 4;
    params.ngroups_per_elem = 8 / params.g;
    params.simd_n_in = 16;
    params.simd_n_out = 8;
    params.chunk_n = 8;

    params.bits = nbits;
    params.q_group_size = block_size;

    if (block_size % 64 == 0) {
        params.act_group_size = 64;
    } else if (block_size % 32 == 0) {
        params.act_group_size = 32;
    } else {
        // throw error
        ORT_THROW("Unsupported activation group size: ", block_size);;
    }
    params.actk = params.act_group_size / params.g;

    //search space
    std::vector<size_t> bms;
    if (nbits == 1 || nbits == 2 || nbits == 4) {
        bms = {256, 512, 1024, 2048, 320, 640, 1280};
    } else if (nbits == 3) {
        bms = {192, 384, 576, 758};
    }

    std::vector<size_t> bns = {8, 16, 32, 64};
    std::vector<size_t> kfactors = {8, 16};

    double min_time = 1e9;

    // TODO: add profile based policy
    int threads = std::thread::hardware_concurrency();

    float smallest_penalty = 1e9;
    for (int bm: bms) {
        if (M % (bm/nbits) != 0 || bm % nbits != 0) {
            continue;
        }
        size_t num_tiles = M/ (bm/nbits);
        size_t num_groups = (num_tiles + threads - 1) / threads;
        float penalty = 0.1 * num_groups + (num_groups - 1.0 * num_tiles / threads) / num_groups;
        if (penalty < smallest_penalty) {
            smallest_penalty = penalty;
            params.bm = bm;
        }
    }

    size_t largest_kfactor = 0;
    for (size_t kfactor: kfactors) {
        if ((kfactor < params.actk) || (kfactor * params.g > params.q_group_size)) {
            continue;
        }
        if (kfactor > largest_kfactor) {
            largest_kfactor = kfactor;
            params.kfactor = kfactor;
        }
    }

    tmac_kernel_configs[key] = params;
    return tmac_kernel_configs[key];
}

void MlasTMACPackScalesAndZeroPoints(
    size_t N,
    size_t K,
    size_t BitWidth,
    size_t BlkLen,
    bool HasZeroPoint,
    float* PackedQuantBZPBegin,
    const float* QuantBScale,
    const uint8_t* QuantBZeroPoint
)
{
    const MlasTMACKernelParams& tmac_params = GetTMACKernelParams(N, K, 2, BlkLen);
    const size_t bits = tmac_params.bits;
    const size_t g = tmac_params.g;
    const size_t ngroups_per_elem = tmac_params.ngroups_per_elem;
    const size_t simd_n_in = tmac_params.simd_n_in;
    const size_t simd_n_out = tmac_params.simd_n_out;
    const size_t bm = tmac_params.bm;
    const size_t kfactor = tmac_params.kfactor;
    const size_t num_elem_per_byte = 8 / bits;


    for (size_t im = 0; im < N ; im += 1) {
        for (size_t ik = 0; ik < K; ik += BlkLen) {
            size_t idx = (im * K + ik) / BlkLen;
            float scale = QuantBScale[idx];
            float zp;
            if (HasZeroPoint) {
                // zp are two bit packed
                size_t elem_idx = idx % num_elem_per_byte;
                uint8_t v = QuantBZeroPoint[idx / num_elem_per_byte] >> (elem_idx * bits) & (1 << bits) - 1;
                zp = static_cast<float>(v);

                // Note: TMAC does this during model conversion. Since, we follow ORT format, we need to do it here.
                // This seems gptq quantization specific.
                // We should either use different op than matmul_nbits or add attribute to matmul_nbits to indicate this.
                zp = zp - (1 << (bits - 1)) - 1;  // make it signed
                zp = zp * scale;              // store scale * zp
            }

            size_t nb1 = K / BlkLen;
            size_t nb0 = bm / BitWidth * nb1;
            size_t new_im = idx / nb0;
            size_t new_ibm = (idx % nb0) / nb1;
            size_t new_ik = (idx % nb1);

            if (HasZeroPoint) {
                size_t new_isimd = new_ibm % simd_n_out;
                size_t new_idx_outer = new_im * bm / bits * K / BlkLen / simd_n_out + new_ik * bm / bits / simd_n_out + new_ibm / simd_n_out;
                size_t new_idx_scale = new_idx_outer * (simd_n_out * 2) + new_isimd;
                size_t new_idx_zero = new_idx_outer * (simd_n_out * 2) + simd_n_out + new_isimd;

                PackedQuantBZPBegin[new_idx_scale] = scale;
                PackedQuantBZPBegin[new_idx_zero] = zp;
            } else {
                size_t new_idx = new_im * bm / bits * K / BlkLen + new_ik * bm / bits + new_ibm;
                PackedQuantBZPBegin[new_idx] = scale;
            }
        }
    }
}


bool MLASCALL MlasIsTMACAvailable(
    size_t /*BlkBitWidth*/,
    size_t /*BlkLen*/
)
{
    const auto* Dispatch = GetMlasPlatform().LUTGenKernel;
    return Dispatch != nullptr;
    // return Dispatch != nullptr && BlkLen == 4; // only support group sizes of 4 for now
}

// TODO: also pass in a biases reference
bool MLASCALL MlasTmacInitializeTable(
    size_t BlkLen,
    void* QuantBData,     // B in MLFloat16 (per your layout)
    float* QuantBScale,        // scale(s) in float
    int K,
    void* qlut
) {
    // base on lut_ctor_int8_g4
    const auto* Dispatch = GetMlasPlatform().LUTGenKernel;
    if (!Dispatch || !Dispatch->GenerateLUT) return false;

    // Cast target LUT buffer to int8, and prepare half-precision inputs
    auto* lut_i8 = reinterpret_cast<int8_t*>(qlut);
    auto* b_float = reinterpret_cast<float*>(QuantBData);

    const int num_groups = static_cast<int>(K / BlkLen);

    float* biases = new float[num_groups]();

    // Call the dispatch
    Dispatch->GenerateLUT(static_cast<int32_t>(BlkLen), lut_i8, b_float, QuantBScale, biases, K);

    // If you need the bias value elsewhere, read it from b16
    // float bias_f = static_cast<float>(b16);

    return true;
}

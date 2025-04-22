#include "mlas.h"
#include "mlasi.h"
#include "llama_cpp_api.h"
#include "ggml.h"
#include "ggml-cpu.h"

#undef GGML_COMMON_DECL
#define GGML_COMMON_DECL_C
#include "../../ggml/src/ggml-common.h"

#include "mlas_qnbit.h"
#include <vector>
#include <unordered_map>
#include <assert.h>

void
vec_dot_q2_K_q8_K()
{
    // std::vector<uint8_t> b(sizeof(block_q8_0));
    // ggml_vec_dot_t dot_fn = ggml_get_type_traits_cpu(GGML_TYPE_Q4_0)->vec_dot;
    // dot_fn(0, nullptr, 0, nullptr, 0, nullptr, 0, 0);
    // ggml_vec_dot_q4_0_q8_0(0, nullptr, 0, nullptr, 0, nullptr, 0, 0);
    // ggml_is_numa();
}

ggml_type
quant_name_to_type(const std::string& name)
{
    static std::unordered_map<std::string, ggml_type> name_to_ggml_type;
    if (name_to_ggml_type.empty()) {
        for (int i = 0; i < GGML_TYPE_COUNT; i++) {
            ggml_type g_t = static_cast<ggml_type>(i);
            const ggml_type_traits* t = ggml_get_type_traits(g_t);
            name_to_ggml_type[t->type_name] = g_t;
        }
    }

    if (name_to_ggml_type.find(name) != name_to_ggml_type.end()) {
        return name_to_ggml_type[name];
    }
    return GGML_TYPE_COUNT;
}

size_t
MlasLowBitQuantizeSizeInByte(const size_t N, const size_t K, const ggml_type quant_type)
{
    const struct ggml_type_traits* type_traits = ggml_get_type_traits(quant_type);
    const size_t a_blk_count_k = (K + type_traits->blck_size - 1) / type_traits->blck_size;
    size_t a_stride = type_traits->type_size * a_blk_count_k;
    size_t quant_size = N * a_stride;
    return quant_size;
}

size_t
MlasLowBitQuantizeSizeInByte(const size_t N, const size_t K, const std::string& quant_type_name)
{
    ggml_type quant_type = quant_name_to_type(quant_type_name);
    return MlasLowBitQuantizeSizeInByte(N, K, quant_type);
}

size_t
MlasLowBitDequantizeDataCount(const size_t N, const size_t K, const std::string& quant_type_name)
{
    // take into consideration of padding
    ggml_type quant_type = quant_name_to_type(quant_type_name);
    const struct ggml_type_traits* type_traits = ggml_get_type_traits(quant_type);
    const size_t a_blk_count_k = (K + type_traits->blck_size - 1) / type_traits->blck_size;
    size_t a_stride = type_traits->blck_size * a_blk_count_k;
    size_t dequant_size = N * a_stride;
    return dequant_size;
}

void
MlasLowBitQuantize(const float* data, const size_t N, const size_t K, const ggml_type quant_type, uint8_t* quant_data, MLAS_THREADPOOL* ThreadPool)
{
    const struct ggml_type_traits* type_traits = ggml_get_type_traits(quant_type);
    const size_t a_blk_count_k = (K + type_traits->blck_size - 1) / type_traits->blck_size;
    const size_t K_padded = a_blk_count_k * type_traits->blck_size;
    size_t a_stride = type_traits->type_size * a_blk_count_k;
    const struct ggml_type_traits_cpu* cpu_type_traits = ggml_get_type_traits_cpu(quant_type);

    if (ThreadPool == nullptr) {
        for (size_t n = 0; n < N; n++) {
            // padding
            if (K_padded == K) {
                cpu_type_traits->from_float(data + n * K, &quant_data[n * a_stride], K_padded);
            } else {
                std::vector<float> data_padded(K_padded, 0);
                std::copy(data + n * K, data + n * K + K, &data_padded[0]);
                cpu_type_traits->from_float(&data_padded[0], &quant_data[n * a_stride], K_padded);
            }
        }
    } else {
        MlasTrySimpleParallel(ThreadPool, N, [&](ptrdiff_t n) {
            if (K_padded == K) {
                cpu_type_traits->from_float(data + n * K, &quant_data[n * a_stride], K_padded);
            } else {
                // padding
                std::vector<float> data_padded(K_padded, 0);
                std::copy(data + n * K, data + n * K + K, &data_padded[0]);
                cpu_type_traits->from_float(&data_padded[0], &quant_data[n * a_stride], K_padded);
            }
        });
    }
}

void
MlasLowBitQuantize(const float* data, const size_t N, const size_t K, const std::string& quant_type_name, uint8_t* quant_data, MLAS_THREADPOOL* ThreadPool)
{
    const ggml_type quant_type = quant_name_to_type(quant_type_name);
    MlasLowBitQuantize(data, N, K, quant_type, quant_data, ThreadPool);
}

void
MlasLowBitDequantize(const uint8_t* quant_data, const size_t N, const size_t K, const std::string& quant_type_name, float* dequant_data, MLAS_THREADPOOL* ThreadPool)
{
    ggml_type quant_type = quant_name_to_type(quant_type_name);
    const struct ggml_type_traits* type_traits = ggml_get_type_traits(quant_type);
    const size_t a_blk_count_k = (K + type_traits->blck_size - 1) / type_traits->blck_size;
    const size_t K_padded = a_blk_count_k * type_traits->blck_size;
    // MlasLowBitQuantize already padded so K shall be already padded.
    // but that is fine the caller get the dequantized data regardless of K being passed or not.
    assert(K_padded == K);
    size_t a_stride = type_traits->type_size * a_blk_count_k;

    if (ThreadPool == nullptr) {
        for (size_t n = 0; n < N; n++) {
            type_traits->to_float(&quant_data[n * a_stride], &dequant_data[n * K_padded], K_padded);
        }
    } else {
        MlasTrySimpleParallel(ThreadPool, N, [&](ptrdiff_t n) {
            type_traits->to_float(&quant_data[n * a_stride], &dequant_data[n * K_padded], K_padded);
        });
    }
}

void MLASCALL
MlasLowBitQGemmTile(
    const size_t batch_index,
    const size_t M,
    const size_t N,
    const size_t K,
    const size_t RangeStartM,
    const size_t RangeCountM,
    const size_t RangeStartN,
    const size_t RangeCountN,
    const uint8_t* a_quant_data,
    const ggml_type a_quant_type,
    const uint8_t* b_quant_data,
    const ggml_type b_quant_type,
    float* c_data
)
{
    const struct ggml_type_traits_cpu* cpu_type_traits = ggml_get_type_traits_cpu(b_quant_type);
    if (cpu_type_traits->vec_dot_type != a_quant_type) {
        throw std::runtime_error("Quantized gemm input type mismatch!");
    }
    ggml_vec_dot_t cpu_dot_fn = cpu_type_traits->vec_dot;

    if (GGML_TYPE_COUNT == a_quant_type) {
        // a_quant_data is not quantized. quantize it to vec_dot_type of b_quant_type


    }

    const struct ggml_type_traits* a_type_traits = ggml_get_type_traits(a_quant_type);
    const size_t a_blk_count_k = (K + a_type_traits->blck_size - 1) / a_type_traits->blck_size;
    size_t a_stride = a_type_traits->type_size * a_blk_count_k;

    const struct ggml_type_traits* b_type_traits = ggml_get_type_traits(b_quant_type);
    const size_t b_blk_count_k = (K + b_type_traits->blck_size - 1) / b_type_traits->blck_size;
    size_t b_stride = b_type_traits->type_size * b_blk_count_k;

    size_t a_batch_stride = M * K * a_stride;
    size_t b_batch_stride = N * K * b_stride;
    size_t c_batch_stride = M * N;

    float* c_data_row = c_data + batch_index * c_batch_stride;

    const uint8_t* vy = a_quant_data + a_batch_stride * batch_index + RangeStartM * a_stride;

    size_t bs_unused = 0; size_t bx_unused = 0; size_t by_unused = 0; int nrc_unused = 1;

    for (size_t m = RangeStartM; m < RangeStartM + RangeCountM; m++, c_data_row += N, vy += a_stride) {
        float* s = c_data_row;
        const uint8_t* vx = b_quant_data + batch_index * b_batch_stride + RangeStartN * b_stride;
        for (size_t n = RangeStartN; n < RangeStartN + RangeCountN; n++, vx += b_stride, s++) {
          cpu_dot_fn(static_cast<int>(K), s, bs_unused, vx, bx_unused, vy, by_unused, nrc_unused);
        }
    }
}

void MLASCALL
MlasLowBitQGemmBatch(
    const size_t M,
    const size_t N,
    const size_t K,
    const size_t BatchN,
    const void* a_quant_data,
    const std::string& a_quant_type_name,
    const uint8_t* b_quant_data,
    const std::string& b_quant_type_name,
    float *c_data,
    MLAS_THREADPOOL* ThreadPool
)
{
    ggml_type b_quant_type = quant_name_to_type(b_quant_type_name);

    ggml_type a_quant_type = GGML_TYPE_COUNT;
    std::vector<uint8_t> a_quant_data_local;
    if (!a_quant_type_name.empty()) {
        a_quant_type = quant_name_to_type(a_quant_type_name);
    } else {
        // a_quant_data is not quantized. quantize it to vec_dot_type of b_quant_type
        const struct ggml_type_traits_cpu* cpu_type_traits = ggml_get_type_traits_cpu(b_quant_type);
        a_quant_type = cpu_type_traits->vec_dot_type;

        size_t a_quant_size = MlasLowBitQuantizeSizeInByte(M, K, a_quant_type);
        a_quant_data_local.resize(a_quant_size);
        MlasLowBitQuantize((float*)a_quant_data, M, K, a_quant_type, &a_quant_data_local[0], ThreadPool);
        a_quant_data = &a_quant_data_local[0];
    }

    if (ThreadPool == nullptr) {
        for (size_t batch_index = 0; batch_index < BatchN; batch_index++) {
            MlasLowBitQGemmTile(batch_index, M, N, K, 0, M, 0, N, reinterpret_cast<const uint8_t*>(a_quant_data), a_quant_type, b_quant_data, b_quant_type, c_data);
        }
        return;
    }

    const double Complexity = double(M) * double(N) * double(K) * double(BatchN);

    ptrdiff_t TargetThreadCount = ptrdiff_t(Complexity / double(MLAS_QGEMM_THREAD_COMPLEXITY)) + 1;

    ptrdiff_t MaximumThreadCount = MlasGetMaximumThreadCount(ThreadPool) * 8;

    if (TargetThreadCount >= MaximumThreadCount) {
        TargetThreadCount = MaximumThreadCount;
    }

    ptrdiff_t ThreadsPerGemm = TargetThreadCount / BatchN;
    if (ThreadsPerGemm < 1) {
        ThreadsPerGemm = 1;
    }

    constexpr size_t StrideM = 128;

    size_t nc = N;
    if (ThreadsPerGemm > 1) {
        // more than one thread per GEMM

        const size_t BlockedM = MlasDivRoundup(M, StrideM);
        const size_t max_nc = MlasDivRoundup(N * BlockedM, ThreadsPerGemm);
        if (max_nc < nc) {
            nc = std::min(
                nc, MlasDivRoundup(max_nc, MLAS_QGEMM_STRIDEN_THREAD_ALIGN) *
                        MLAS_QGEMM_STRIDEN_THREAD_ALIGN
            );
        }
    }
    const size_t StrideN = nc;

    const size_t ThreadCountM = (M + StrideM - 1) / StrideM;
    const size_t ThreadCountN = (N + StrideN - 1) / StrideN;
    ThreadsPerGemm = ThreadCountM * ThreadCountN;

    MlasTrySimpleParallel(ThreadPool, ThreadsPerGemm * BatchN, [&](ptrdiff_t tid) {
        const auto batch_index = tid / ThreadsPerGemm;
        const auto blk_i = tid % ThreadsPerGemm;

        const ptrdiff_t ThreadIdN = blk_i / ThreadCountM;
        const ptrdiff_t ThreadIdM = blk_i % ThreadCountM;

        const size_t RangeStartM = ThreadIdM * StrideM;
        const size_t RangeCountM = std::min(M - RangeStartM, (size_t)StrideM);

        const size_t RangeStartN = ThreadIdN * StrideN;
        const size_t RangeCountN = std::min(N - RangeStartN, (size_t)StrideN);

        MlasLowBitQGemmTile(
          batch_index,
          M,
          N,
          K,
          RangeStartM,
          RangeCountM,
          RangeStartN,
          RangeCountN,
          reinterpret_cast<const uint8_t*>(a_quant_data),
          a_quant_type,
          b_quant_data,
          b_quant_type,
          c_data
        );
    });

    return;
}

bool
MlasLowBitCanQuantize(const std::string& quant_type_name)
{
    ggml_type quant_type = quant_name_to_type(quant_type_name);
    const struct ggml_type_traits_cpu* cpu_type_traits = ggml_get_type_traits_cpu(quant_type);
    if (cpu_type_traits->from_float != nullptr) {
        return true;
    }

    //const struct ggml_type_traits* type_traits = ggml_get_type_traits(quant_type);
    //if (type_traits->from_float_ref != nullptr) {
    //    return true;
    //}
    return false;
}

bool
MlasLowBitCanDequantize(const std::string& quant_type_name)
{
    ggml_type quant_type = quant_name_to_type(quant_type_name);
    const struct ggml_type_traits* type_traits = ggml_get_type_traits(quant_type);
    if (type_traits->to_float != nullptr) {
        return true;
    }

    return false;
}

class MLasLlamaInitializer {
    public:
        // Static method to ensure the singleton instance is created
        static void EnsureInitialized() {
            static MLasLlamaInitializer instance; // Guaranteed to be initialized only once
        }

    private:
        // Private constructor to initialize ggml only once
        MLasLlamaInitializer() {
            ggml_init_params params;
            params.mem_size = 1024;
            params.mem_buffer = nullptr;
            params.no_alloc = false;
            ggml_init(params);
        }

        // Delete copy constructor and assignment operator to prevent copying
        MLasLlamaInitializer(const MLasLlamaInitializer&) = delete;
        MLasLlamaInitializer& operator=(const MLasLlamaInitializer&) = delete;
};

// Ensure the initializer is called at program startup
namespace {
    const auto& llama_initializer = []() {
        MLasLlamaInitializer::EnsureInitialized();
        return 0;
    }();
}

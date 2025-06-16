#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "cutlass/numeric_types.h"
#include"contrib_ops/cuda/llm/common/cuda_runtime_utils.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemm/fpA_intB_gemm.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemv/fpA_intB_gemv.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace wo = onnxruntime::llm::kernels::fpA_intB_gemv;

void simple_assert(bool flag)
{
    if (!flag)
    {
        throw std::runtime_error("assert failed");
    }
}

struct CudaBuffer
{
    void* _data;
    int _size;

    CudaBuffer(int size_in_bytes)
        : _size(size_in_bytes)
    {
        cudaMalloc(&_data, _size);
    }

    template <typename T = void>
    T* data()
    {
        return reinterpret_cast<T*>(_data);
    }

    void copy_to(void* dst)
    {
        cudaMemcpy(dst, _data, _size, cudaMemcpyDeviceToHost);
    }

    void copy_from(void* src)
    {
        cudaMemcpy(_data, src, _size, cudaMemcpyHostToDevice);
    }

    ~CudaBuffer()
    {
        cudaFree(_data);
    }
};

template <typename T>
float compare(void* _pa, void* _pb, int size, float scale)
{
    auto pa = reinterpret_cast<T*>(_pa);
    auto pb = reinterpret_cast<T*>(_pb);
    float max_diff = 0.f, tot_diff = 0.f;
    float max_val = 0.f;
    int diff_cnt = 0;
    float threshold = 1e-7;
    for (int n = 0; n < size; ++n)
    {
        float va = static_cast<float>(pa[n]);
        float vb = static_cast<float>(pb[n]);
        max_val = std::max(max_val, vb);
        float diff = std::abs(va - vb);
        if (diff > threshold)
        {
            max_diff = std::max(max_diff, diff);
            tot_diff += diff;
            ++diff_cnt;
        }
    }
    float diff_thres = max_val * scale;
#if defined(ENABLE_BF16)
    if constexpr (std::is_same_v<T, __nv_bfloat16>)
    {
        // bfloat16 has fewer mantissa digits than float16(10 bits for fp16 but only 7 bits for bf16), so the cumulative
        // error will be larger.
        diff_thres *= 3.f;
    }
    else
#endif
    {
        diff_thres *= 1.5f;
    }
    printf("max diff %f (diff threshold %f), avg diff %f, diff cnt %d/%d\n", max_diff, diff_thres, tot_diff / diff_cnt,
        diff_cnt, size);
    return max_diff <= diff_thres;
}

template <typename T1, typename T2>
void random_fill(std::vector<T1>& vec, T2 minv, T2 maxv)
{
    std::mt19937 gen(rand());
    std::uniform_real_distribution<float> dis(static_cast<float>(minv), static_cast<float>(maxv));
    for (auto& v : vec)
    {
        v = static_cast<T1>(dis(gen));
    }
}

template <typename T>
std::vector<onnxruntime::llm::cutlass_extensions::CutlassGemmConfig> get_configs(T& runner, int k)
{
    auto configs = runner.getConfigs();
    std::vector<onnxruntime::llm::cutlass_extensions::CutlassGemmConfig> rets;
    for (auto config : configs)
    {
        if (config.stages >= 5)
        {
            continue;
        }
        if (config.split_k_style != onnxruntime::llm::cutlass_extensions::SplitKStyle::NO_SPLIT_K)
        {
            int k_size = (k + config.split_k_factor - 1) / config.split_k_factor;
            if (k_size % 64)
            {
                continue;
            }
        }
        rets.push_back(config);
    }
    return rets;
}

template <wo::KernelType KT>
struct cutlassTypeMapper
{
};

#define CUTLASS_TYPE_MAPPER_REGISTRY(                                                                                  \
    CudaKernelType, KernelInfoStr, CutlassAType, CutlassWType, WElemBits, CutlassQuantOp)                              \
    template <>                                                                                                        \
    struct cutlassTypeMapper<CudaKernelType>                                                                           \
    {                                                                                                                  \
        using AType = CutlassAType;                                                                                    \
        using WType = CutlassWType;                                                                                    \
        static constexpr cutlass::WeightOnlyQuantOp QuantOp = CutlassQuantOp;                                          \
        static constexpr int WSizeInBits = WElemBits;                                                                  \
        static std::string str(int m, int n, int k, int gs)                                                            \
        {                                                                                                              \
            std::stringstream ss;                                                                                      \
            ss << KernelInfoStr << " mnk(" << m << ", " << n << ", " << k << ")";                                      \
            if (gs != 0)                                                                                               \
                ss << ", gs " << gs;                                                                                   \
            return ss.str();                                                                                           \
        }                                                                                                              \
    };
CUTLASS_TYPE_MAPPER_REGISTRY(wo::KernelType::FP16Int8Groupwise, "FP16Int8Groupwise", half, uint8_t, 8,
    cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS);
CUTLASS_TYPE_MAPPER_REGISTRY(wo::KernelType::BF16Int8Groupwise, "BF16Int8Groupwise", __nv_bfloat16, uint8_t, 8,
    cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS);
CUTLASS_TYPE_MAPPER_REGISTRY(wo::KernelType::FP16Int4Groupwise, "FP16Int4Groupwise", half, cutlass::uint4b_t, 4,
    cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS);
CUTLASS_TYPE_MAPPER_REGISTRY(wo::KernelType::BF16Int4Groupwise, "BF16Int4Groupwise", __nv_bfloat16, cutlass::uint4b_t,
    4, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS);
// CUTLASS_TYPE_MAPPER_REGISTRY(wo::KernelType::FP16Int8PerChannel, "FP16Int8PerChannel", half, uint8_t, 8,
//     cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY);
// CUTLASS_TYPE_MAPPER_REGISTRY(wo::KernelType::BF16Int8PerChannel, "BF16Int8PerChannel", __nv_bfloat16, uint8_t, 8,
//     cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY);
// CUTLASS_TYPE_MAPPER_REGISTRY(wo::KernelType::FP16Int4PerChannel, "FP16Int4PerChannel", half, cutlass::uint4b_t, 4,
//     cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY);
// CUTLASS_TYPE_MAPPER_REGISTRY(wo::KernelType::BF16Int4PerChannel, "BF16Int4PerChannel", __nv_bfloat16, cutlass::uint4b_t,
//     4, cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY);

float run_cuda_kernel(wo::Params& params, int warmup, int iter)
{
    int arch = onnxruntime::llm::common::getSMVersion();
    simple_assert(wo::is_supported(arch, params.type));
    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    for (int i = 0; i < warmup; ++i)
    {
        wo::kernel_launcher(arch, params, s);
    }
    cudaEventRecord(begin, s);
    for (int i = 0; i < iter; ++i)
    {
        wo::kernel_launcher(arch, params, s);
    }
    cudaEventRecord(end, s);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, begin, end);
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaStreamDestroy(s);
    return time / iter;
}

template <wo::KernelType KT, typename Runner, typename Config>
void exec_cutlass_kernel(
    void* scaled_act, Runner& runner, wo::Params& params, Config& config, char* ws, size_t ws_size, cudaStream_t stream)
{
    using AType = typename cutlassTypeMapper<KT>::AType;
    static constexpr cutlass::WeightOnlyQuantOp QuantOp = cutlassTypeMapper<KT>::QuantOp;
    void* act = params.act;
    if (params.act_scale)
    {
        onnxruntime::llm::kernels::apply_per_channel_scale_kernel_launcher<AType, AType>(
            reinterpret_cast<AType*>(scaled_act), reinterpret_cast<AType const*>(params.act),
            reinterpret_cast<AType const*>(params.act_scale), params.m, params.k, nullptr, stream);
        act = scaled_act;
    }
    if constexpr (QuantOp == cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY)
    {
        runner.gemm(
            act, params.weight, params.scales, params.out, params.m, params.n, params.k, config, ws, ws_size, stream);
    }
    else if (QuantOp == cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS)
    {
        runner.gemm(act, params.weight, params.scales, params.zeros, params.bias, params.out, params.m, params.n,
            params.k, params.groupsize, config, ws, ws_size, stream);
    }
}

template <wo::KernelType KT>
float run_cutlass_kernel(wo::Params& params, int warmup, int iter)
{
    int arch = onnxruntime::llm::common::getSMVersion();
    simple_assert(KT == params.type);
    simple_assert(wo::is_supported(arch, params.type));
    using AType = typename cutlassTypeMapper<KT>::AType;
    using WType = typename cutlassTypeMapper<KT>::WType;
    CudaBuffer scaled_act(params.m * params.k * sizeof(AType));
    auto runner = std::make_shared<onnxruntime::llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<AType, WType,
        cutlassTypeMapper<KT>::QuantOp>>();
    auto& gemm = *runner;
    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    auto configs = get_configs(gemm, params.k);
    int ws_bytes = gemm.getWorkspaceSize(params.m, params.n, params.k);
    char* ws_ptr = nullptr;
    if (ws_bytes)
        cudaMalloc(&ws_ptr, ws_bytes);
    float fast_time = 1e8;
    auto best_config = configs[0];
    int cfg_i = 0;
    for (auto& config : configs)
    {
        float time = std::numeric_limits<float>::max();
        try
        {
            for (int i = 0; i < 2; ++i)
            {
                exec_cutlass_kernel<KT>(scaled_act.data(), gemm, params, config, ws_ptr, ws_bytes, s);
            }
            cudaEventRecord(begin, s);
            for (int i = 0; i < 5; ++i)
            {
                exec_cutlass_kernel<KT>(scaled_act.data(), gemm, params, config, ws_ptr, ws_bytes, s);
            }
            cudaEventRecord(end, s);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&time, begin, end);
        }
        catch (std::exception const& e)
        {
            std::ostringstream msg;
            msg << "Cannot profile configuration " << cfg_i;
            if constexpr (std::is_same_v<decltype(config), onnxruntime::llm::cutlass_extensions::CutlassGemmConfig>)
            {
                msg << ": " << config.toString();
            }
            msg << "\n (for"
                << " m=" << params.m << ", n=" << params.n << ", k=" << params.k << ")"
                << ", reason: \"" << e.what() << "\". Skipped\n";
            std::cout << msg.str();
            cudaGetLastError(); // Reset the last cudaError to cudaSuccess.
            continue;
        }
        if (time < fast_time)
        {
            fast_time = time;
            best_config = config;
        }
        cfg_i++;
    }

    for (int i = 0; i < warmup; ++i)
    {
        exec_cutlass_kernel<KT>(scaled_act.data(), gemm, params, best_config, ws_ptr, ws_bytes, s);
    }
    cudaEventRecord(begin, s);
    for (int i = 0; i < iter; ++i)
    {
        exec_cutlass_kernel<KT>(scaled_act.data(), gemm, params, best_config, ws_ptr, ws_bytes, s);
    }
    if (ws_ptr)
        cudaFree(ws_ptr);
    cudaEventRecord(end, s);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, begin, end);
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaStreamDestroy(s);
    return time / iter;
}

template <wo::KernelType KT>
bool benchmark_and_verify(int m, int n, int k, int groupsize, int warmup, int iter)
{
    std::srand(20240123);
    simple_assert(m <= 16);
    if constexpr (cutlassTypeMapper<KT>::QuantOp == cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY)
    {
        simple_assert(groupsize == 0);
    }
    else if (cutlassTypeMapper<KT>::QuantOp == cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS)
    {
        simple_assert(groupsize == 64 || groupsize == 128);
    }
    using AType = typename cutlassTypeMapper<KT>::AType;
    using WType = typename cutlassTypeMapper<KT>::WType;
    static constexpr int ASizeInBits = sizeof(AType) * 8;
    static constexpr int WSizeInBits = cutlassTypeMapper<KT>::WSizeInBits;
    int gs_factor = groupsize == 0 ? 1 : groupsize;
    printf("Kernel %s\n", cutlassTypeMapper<KT>::str(m, n, k, groupsize).c_str());

    CudaBuffer d_act(m * k * ASizeInBits / 8);
    CudaBuffer d_act_scale(k * ASizeInBits / 8);
    CudaBuffer d_weight(k * n * WSizeInBits / 8);
    CudaBuffer d_scales(n * k / gs_factor * ASizeInBits / 8);
    CudaBuffer d_zeros(n * k / gs_factor * ASizeInBits / 8);
    CudaBuffer d_bias(n * ASizeInBits / 8);
    CudaBuffer d_out(m * n * ASizeInBits / 8);
    std::vector<AType> h_act(m * k), h_act_scale(k);
    std::vector<uint8_t> h_weight(k * n);
    std::vector<AType> h_scales(n * k), h_zeros(n * k), h_bias(n);
    std::vector<AType> h_out1(m * n), h_out2(m * n);

    random_fill(h_act, -1.f, 1.f);
    random_fill(h_act_scale, -1.f, 1.f);
    random_fill(h_scales, -1.f, 1.f);
    random_fill(h_zeros, -1.f, 1.f);
    random_fill(h_bias, -1.f, 1.f);

    for (uint8_t& v : h_weight)
    {
        v = rand() % 256;
    }

    d_act.copy_from(h_act.data());
    d_act_scale.copy_from(h_act_scale.data());
    d_weight.copy_from(h_weight.data());
    d_scales.copy_from(h_scales.data());
    d_zeros.copy_from(h_zeros.data());
    d_bias.copy_from(h_bias.data());

    void* p_act_scale = nullptr;
    void* p_zeros = nullptr;
    void* p_bias = nullptr;

    if (groupsize != 0)
    {
        p_zeros = d_zeros.data();
        p_bias = d_bias.data();
        p_act_scale = d_act_scale.data();
    }
    wo::Params params(d_act.data(), p_act_scale, d_weight.data(), d_scales.data(), p_zeros, p_bias, d_out.data(), 1.f,
        m, n, k, groupsize, KT);
    float time1, time2;
    time1 = run_cuda_kernel(params, warmup, iter);
    d_out.copy_to(h_out1.data());
    time2 = run_cutlass_kernel<KT>(params, warmup, iter);
    d_out.copy_to(h_out2.data());
    float quant_scale = 1.f / (1 << (WSizeInBits - 1));
    bool pass = compare<AType>(h_out1.data(), h_out2.data(), m * n, quant_scale);
    printf("cuda kernel cost time %.3f us, cutlass kernel cost time %.3f us, cuda speedup %.2f\n\n", time1 * 1000,
        time2 * 1000, time2 / time1);
    return pass;
}

TEST(Kernel, WeightOnly)
{
    int const arch = onnxruntime::llm::common::getSMVersion();
    bool pass;
    int warmup = 10, iter = 30;
    std::vector<int> ms{2, 4, 6, 8, 10, 12, 14};
    std::vector<int> ns{4096};
    std::vector<int> ks{2048};
    for (auto m : ms)
    {
        for (auto n : ns)
        {
            for (auto k : ks)
            {
                // pass = benchmark_and_verify<wo::KernelType::FP16Int8PerChannel>(m, n, k, 0, warmup, iter);
                // EXPECT_TRUE(pass);
                // pass = benchmark_and_verify<wo::KernelType::FP16Int4PerChannel>(m, n, k, 0, warmup, iter);
                // EXPECT_TRUE(pass);
                if (arch >= 75)
                {
                    pass = benchmark_and_verify<wo::KernelType::FP16Int8Groupwise>(m, n, k, 64, warmup, iter);
                    EXPECT_TRUE(pass);
                    pass = benchmark_and_verify<wo::KernelType::FP16Int8Groupwise>(m, n, k, 128, warmup, iter);
                    EXPECT_TRUE(pass);
                    pass = benchmark_and_verify<wo::KernelType::FP16Int4Groupwise>(m, n, k, 64, warmup, iter);
                    EXPECT_TRUE(pass);
                    pass = benchmark_and_verify<wo::KernelType::FP16Int4Groupwise>(m, n, k, 128, warmup, iter);
                    EXPECT_TRUE(pass);
#if defined(ENABLE_BF16)
                    if (arch >= 80)
                    {
                        pass = benchmark_and_verify<wo::KernelType::BF16Int8Groupwise>(m, n, k, 64, warmup, iter);
                        EXPECT_TRUE(pass);
                        pass = benchmark_and_verify<wo::KernelType::BF16Int8Groupwise>(m, n, k, 128, warmup, iter);
                        EXPECT_TRUE(pass);
                        pass = benchmark_and_verify<wo::KernelType::BF16Int4Groupwise>(m, n, k, 64, warmup, iter);
                        EXPECT_TRUE(pass);
                        pass = benchmark_and_verify<wo::KernelType::BF16Int4Groupwise>(m, n, k, 128, warmup, iter);
                        EXPECT_TRUE(pass);
                        // pass = benchmark_and_verify<wo::KernelType::BF16Int8PerChannel>(m, n, k, 0, warmup, iter);
                        // EXPECT_TRUE(pass);
                        // pass = benchmark_and_verify<wo::KernelType::BF16Int4PerChannel>(m, n, k, 0, warmup, iter);
                        // EXPECT_TRUE(pass);
                    }
#endif
                }
            }
        }
    }
}

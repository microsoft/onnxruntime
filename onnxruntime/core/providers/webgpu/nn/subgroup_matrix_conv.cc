// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(__wasm__)

#include "core/providers/webgpu/nn/subgroup_matrix_conv.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "core/common/narrow.h"
#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/math/subgroup_matrix_config.h"
#include "core/providers/webgpu/math/subgroup_matrix_matmul.h"
#include "core/providers/webgpu/nn/conv.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/vendor/intel/math/subgroup_matrix_tiling_selector.h"

namespace onnxruntime {
namespace webgpu {

namespace {

// Lanes per subgroup assumed by the subgroup-matrix kernel. The workgroup runs
// split_k subgroups, so its size is kSubgroupMatrixSubgroupSize * split_k.
constexpr uint32_t kSubgroupMatrixSubgroupSize = 32;

// Combined workgroup-memory (SLM) budget in f16 elements for the two scratch
// buffers the conv kernel uses: the split-K output tiles
// (split_k * tile_m * tile_n) plus the per-subgroup im2col A staging
// (split_k * tile_m * sg_mat_k). 16384 f16 == 32 KB, the common WebGPU
// maxComputeWorkgroupStorageSize. split_k is clamped to fit this.
constexpr uint32_t kMaxConvScratchElems = 16384;

// Returns the device config index for the fixed 8x16x16 F16 subgroup-matrix
// config this kernel is implemented for, or -1 when the device does not report
// it. Also requires the vendor tiling policy (Intel) to be available.
int32_t GetSupportedConfigIndex(const ComputeContextBase& context) {
  int32_t config_index = 0;
  if (!IsSubgroupMatrixConfigSupported(context, /*is_fp16=*/true, config_index) ||
      !supported_subgroup_matrix_configs[config_index].Is(8, 16, 16)) {
    return -1;
  }
  if (context.AdapterInfo().vendor != std::string_view{"intel"}) {
    return -1;
  }
  return config_index;
}

Status GenerateShaderCode8x16x16(ShaderHelper& shader, const ShaderVariableHelper& input_src,
                                 const ShaderVariableHelper& output,
                                 bool has_bias, uint32_t sg_mat_count_m, uint32_t sg_mat_count_n,
                                 uint32_t split_k, uint32_t vec_size, int32_t activation_kind) {
  return WGSL_TEMPLATE_APPLY(shader, "nn/subgroup_matrix_conv_8x16x16.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(activation_kind, activation_kind),
                             WGSL_TEMPLATE_PARAMETER(has_bias, has_bias),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_count_m, sg_mat_count_m),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_count_n, sg_mat_count_n),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_k, 16),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_m, 8),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_n, 16),
                             WGSL_TEMPLATE_PARAMETER(split_k, split_k),
                             WGSL_TEMPLATE_PARAMETER(vec_size, vec_size),
                             WGSL_TEMPLATE_VARIABLE(input_src, input_src),
                             WGSL_TEMPLATE_VARIABLE(output, output));
}

// Subgroup-matrix Conv implementation. Computes a channels-last 2D convolution as
// an implicit GEMM during Compute: A is the im2col of the activation, gathered on
// the fly into workgroup memory (never materialized in global memory), B is the
// transposed weight, and the product is accumulated with
// subgroupMatrixMultiplyAccumulate. The class is intended to support all
// subgroup-matrix configs; for now only 8x16x16 is implemented. The per-problem
// output tiling is supplied by a vendor-specific selector kept internal to this
// impl, which declines problems its policy does not cover.
// It runs its own Conv shape inference to decide eligibility, mirroring
// Conv::ComputeInternal: ConvOptImpl::Compute is called before ComputeInternal has
// resolved auto_pad or allocated the output.
template <bool is_channels_last, bool is_fused>
class SubgroupMatrixConvImpl final : public Conv<is_channels_last, is_fused>::ConvOptImpl {
  using ConvT = Conv<is_channels_last, is_fused>;
  using Base = typename ConvT::ConvOptImpl;

 public:
  SubgroupMatrixConvImpl(const ConvT& parent, int32_t config_index,
                         SubgroupMatrixTilingSelector tiling_selector)
      : Base(parent), config_index_(config_index), tiling_selector_(std::move(tiling_selector)) {}

  Status Compute(ComputeContext& context, /*out*/ bool& handled) override {
    handled = false;

    const ConvT& parent = this->parent_;
    const ConvAttributes& conv_attrs = parent.ConvAttrs();
    const Activation& activation = parent.ConvActivation();

    // The kernel reads the weight in its original OIHW layout (it applies its own
    // OIHW -> OHWI transpose below), so a weight prepacked to HWIO cannot be used.
    // Conv::PrePackInternal skips prepacking exactly for the problems
    // CanApplySubgroupMatrixConv accepts, so this normally never declines.
    if (parent.PrepackedKernel() != nullptr) {
      return Status::OK();
    }
    const auto* src = context.Input<Tensor>(0);
    const auto* weight = context.Input<Tensor>(1);
    if (src == nullptr || weight == nullptr) {
      return Status::OK();
    }
    const TensorShape& weight_shape = weight->Shape();  // OIHW: [Cout, Cin, kh, kw]
    if (!CanApplySubgroupMatrixConv(context, is_channels_last, weight_shape,
                                    narrow<uint32_t>(conv_attrs.group), weight->DataType())) {
      return Status::OK();
    }

    // Shape inference mirrors Conv::ComputeInternal. channels-last and a 4D weight are
    // guaranteed by CanApplySubgroupMatrixConv; only a rank-4 NHWC activation is served
    // (a reshaped Conv1D never gets here -- its kernel becomes 1xW, which
    // CanApplySubgroupMatrixConv rejects).
    const TensorShape& src_shape = src->Shape();  // NHWC: [batch, src_h, src_w, Cin]
    if (src_shape.NumDimensions() != 4) {
      return Status::OK();
    }
    ConvAttributes::ConvPadVector local_pads(conv_attrs.pads.begin(), conv_attrs.pads.end());
    TensorShapeVector local_dilations(conv_attrs.dilations.begin(), conv_attrs.dilations.end());
    TensorShapeVector local_strides(conv_attrs.strides.begin(), conv_attrs.strides.end());
    TensorShapeVector kernel_spacial_shape_vector;
    ORT_RETURN_IF_ERROR(conv_attrs.ComputeKernelShape(weight_shape, kernel_spacial_shape_vector, false));
    if (local_pads.empty()) {
      local_pads.resize(kernel_spacial_shape_vector.size() * 2, 0);
    }
    if (local_dilations.empty()) {
      local_dilations.resize(kernel_spacial_shape_vector.size(), 1);
    }
    if (local_strides.empty()) {
      local_strides.resize(kernel_spacial_shape_vector.size(), 1);
    }
    TensorShapeVector src_shape_vector = src_shape.AsShapeVector();
    TensorShapeVector output_shape_vector = {src_shape[0]};
    TensorShape src_spacial_shape = TensorShape(
        TensorShapeVector(std::next(src_shape_vector.begin()), std::prev(src_shape_vector.end())));
    ORT_RETURN_IF_ERROR(conv_attrs.InferPadsAndOutputShape(src_spacial_shape, kernel_spacial_shape_vector,
                                                           local_strides, local_dilations, local_pads,
                                                           output_shape_vector));
    output_shape_vector.push_back(weight_shape[0]);  // Cout
    const TensorShape output_shape(output_shape_vector);  // NHWC: [batch, out_h, out_w, Cout]

    // Spatial attributes as the kernel's uniforms want them: pads is [begin..., end...]
    // in ONNX order, strides/dilations one entry per spatial dim. static_cast, not
    // narrow, so an input the normal Conv path accepts is not turned into an abort here
    // -- ComputeInternal casts these the same way.
    std::vector<uint32_t> pads;
    std::vector<uint32_t> strides;
    std::vector<uint32_t> dilations;
    auto transform_dim = [](int64_t dim) { return static_cast<int32_t>(dim); };
    std::transform(local_pads.begin(), local_pads.end(), std::back_inserter(pads), transform_dim);
    std::transform(local_strides.begin(), local_strides.end(), std::back_inserter(strides), transform_dim);
    std::transform(local_dilations.begin(), local_dilations.end(), std::back_inserter(dilations), transform_dim);
    if (dilations.size() < 2 || pads.size() < 2 || strides.size() < 2) {
      return Status::OK();
    }

    const uint32_t batch = narrow<uint32_t>(src_shape[0]);
    const uint32_t src_h = narrow<uint32_t>(src_shape[1]);
    const uint32_t src_w = narrow<uint32_t>(src_shape[2]);
    const uint32_t channel_i = narrow<uint32_t>(weight_shape[1]);
    const uint32_t kernel_h = narrow<uint32_t>(weight_shape[2]);
    const uint32_t kernel_w = narrow<uint32_t>(weight_shape[3]);
    const uint32_t output_h = narrow<uint32_t>(output_shape[1]);
    const uint32_t output_w = narrow<uint32_t>(output_shape[2]);

    const uint32_t M = output_h * output_w;                // output pixels per batch slice
    const uint32_t K = kernel_h * kernel_w * channel_i;    // contraction dim
    const uint32_t N = narrow<uint32_t>(weight_shape[0]);  // Cout

    // An empty batch, M or N yields an empty output (ONNX permits zero-length dims).
    // Let the generic Conv paths handle it rather than dispatch a degenerate
    // (zero-workgroup) kernel.
    if (batch == 0 || M == 0 || N == 0) {
      return Status::OK();
    }

    // Pick the per-problem output tiling from the vendor policy (Intel). The GEMM
    // dims map directly: A [M, K] x B [K, N] with one z-slice per conv batch. The
    // Conv selector uses the Conv scratch budget and Conv-specific pretuned data.
    // Last thing that can decline, so it comes before the weight transpose below: an
    // impl that declines must not have allocated or dispatched anything.
    const std::optional<SubgroupMatrixTiling> tiling = tiling_selector_(context, M, N, K, batch);
    if (!tiling) {
      return Status::OK();
    }

    // Past this point the path is committed, so the output can be allocated: an impl
    // that declines must allocate nothing.
    Tensor* output = context.Output(0, output_shape);

    // Transpose the weight OIHW -> OHWI = [Cout, kh, kw, Cin], which flattens to the
    // GEMM right operand B stored column-major: element B[k, n] lives at n*K + k, so
    // the kernel loads B right tiles with is_col_major = true and stride K. (This
    // matches the im2col-matmul weight layout; the alternative HWIO -> row-major
    // stride-N load is the other option.) This allocates a GPU tensor and dispatches
    // a Transpose program, so the path is already committed at this point.
    //
    // A constant weight is transposed once and reused; a non-constant one can change
    // between Runs, so it has to be redone every time. Declining the non-constant case
    // instead would cost more than the transpose it saves.
    // TODO: prepack the constant case, which would drop the cache and the first-Run
    // dispatch too. PrePackInternal currently skips this path entirely because
    // transposed_kernel_ is defined to hold HWIO, not the OHWI this kernel wants.
    Tensor run_local_ohwi_weight;
    const Tensor* ohwi_weight = nullptr;
    if (parent.IsWeightConstant()) {
      ORT_RETURN_IF_ERROR(EnsureTransposedKernel(context, weight, weight_shape));
      ohwi_weight = cached_ohwi_weight_.get();
    } else {
      ORT_RETURN_IF_ERROR(
          TransposeKernel(context, weight, weight_shape, &run_local_ohwi_weight, {0, 2, 3, 1}));
      ohwi_weight = &run_local_ohwi_weight;
    }

    const auto& config = supported_subgroup_matrix_configs[config_index_];
    const uint32_t tile_m = tiling->tile_m;
    const uint32_t tile_n = tiling->tile_n;
    ORT_ENFORCE(tile_m % config.M == 0 && tile_n % config.N == 0,
                "Tiling must be a multiple of the subgroup-matrix shape: ",
                tile_m, "x", tile_n, " vs ", config.M, "x", config.N);

    // Safety net: the Conv selector already enforces the Conv scratch budget
    // (output tiles + im2col A staging), so this clamp normally does nothing; it
    // guards against a stray oversized split-K (e.g. a hand-edited pretuned entry).
    uint32_t split_k = tiling->split_k;
    while (split_k > 1 &&
           split_k * tile_m * (tile_n + config.K) > kMaxConvScratchElems) {
      split_k /= 2;
    }

    const uint32_t sg_mat_count_m = tile_m / config.M;
    const uint32_t sg_mat_count_n = tile_n / config.N;
    const uint32_t dispatch_x = (N + tile_n - 1) / tile_n;
    const uint32_t dispatch_y = (M + tile_m - 1) / tile_m;

    // Vectorize the im2col gather along the input channel (cin), which is the
    // fastest-varying, contiguous NHWC dimension. channel_i % vec_size == 0 keeps
    // each vec chunk inside a single kernel position, so all lanes of a chunk share
    // one bounds check and map to contiguous input.
    const uint32_t vec_size = channel_i % 4 == 0 ? 4 : (channel_i % 2 == 0 ? 2 : 1);

    // Fused-activation epilogue parameters (baked as f32 uniforms; only read by the
    // parameterized kinds Clip / HardSigmoid / LeakyRelu).
    const float act_alpha = activation.activation_params_.values_[0];
    const float act_beta = activation.activation_params_.values_[1];

    const Tensor* bias = context.InputCount() > 2 ? context.Input<Tensor>(2) : nullptr;
    const bool has_bias = bias != nullptr;
    SubgroupMatrixConvProgram program{has_bias, config_index_, sg_mat_count_m, sg_mat_count_n, split_k, vec_size,
                                      activation};
    program.SetWorkgroupSize(kSubgroupMatrixSubgroupSize * split_k);
    program.SetDispatchGroupSize(dispatch_x, dispatch_y, batch);
    program.CacheHint(has_bias, config_index_, sg_mat_count_m, sg_mat_count_n, split_k, vec_size,
                      activation.ToString())
        .AddInputs({{src, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(vec_size)},
                    {ohwi_weight, ProgramTensorMetadataDependency::TypeAndRank, 1}})
        .AddOutput({output, ProgramTensorMetadataDependency::Rank, output_shape, 1})
        .AddUniformVariables({{M},
                              {N},
                              {K},
                              {dispatch_x},
                              {src_h},
                              {src_w},
                              {channel_i},
                              {kernel_h},
                              {kernel_w},
                              {output_w},
                              {dilations},
                              {pads},
                              {strides},
                              {act_alpha},
                              {act_beta}});
    if (has_bias) {
      program.AddInput({bias, ProgramTensorMetadataDependency::None});
    }
    ORT_RETURN_IF_ERROR(context.RunProgram(program));

    handled = true;
    return Status::OK();
  }

 private:
  // Builds the OIHW -> OHWI transposed weight on first use and caches it. Only valid
  // because the caller checked the weight is a constant initializer. impl_ is shared
  // across concurrent Compute calls on the same Conv kernel, so the build is guarded
  // by call_once, which also synchronizes the read of cached_ohwi_weight_ below.
  // The transpose is attempted exactly once: if it fails, every later Run on this
  // Conv node reports the same error rather than retrying.
  Status EnsureTransposedKernel(ComputeContext& context, const Tensor* weight,
                                const TensorShape& weight_shape) {
    std::call_once(transpose_once_, [&]() {
      auto transposed = std::make_unique<Tensor>();
      if (TransposeKernel(context, weight, weight_shape, transposed.get(), {0, 2, 3, 1}).IsOK()) {
        cached_ohwi_weight_ = std::move(transposed);
      }
    });
    return cached_ohwi_weight_ ? Status::OK()
                               : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                                 "Failed to transpose the subgroup-matrix Conv weight to OHWI.");
  }

  const int32_t config_index_;
  SubgroupMatrixTilingSelector tiling_selector_;

  // OIHW -> OHWI transposed constant weight, built once by EnsureTransposedKernel.
  std::once_flag transpose_once_;
  std::unique_ptr<Tensor> cached_ohwi_weight_;
};

}  // namespace

Status SubgroupMatrixConvProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input_src = shader.AddInput("input_src", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  shader.AddInput("input_b", ShaderUsage::UseUniform);
  if (has_bias_) {
    shader.AddInput("bias", ShaderUsage::UseUniform);
  }
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);

  const auto& config = supported_subgroup_matrix_configs[config_index_];
  if (config.Is(8, 16, 16)) {
    return GenerateShaderCode8x16x16(shader, input_src, output, has_bias_, sg_mat_count_m_, sg_mat_count_n_, split_k_,
                                     vec_size_, static_cast<int32_t>(activation_.activation_kind_));
  }
  return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::NOT_IMPLEMENTED,
                "Unsupported subgroup matrix config dimensions.");
}

bool CanApplySubgroupMatrixConv(ComputeContextBase& context,
                                bool is_channels_last,
                                const TensorShape& kernel_shape,
                                uint32_t group,
                                MLDataType data_type) {
  // The subgroup-matrix kernel is F16-only and computes a channels-last,
  // non-grouped 2D convolution. A fused activation is applied in the write-out
  // epilogue, so fused convs are accepted.
  if (data_type != DataTypeImpl::GetType<MLFloat16>()) {
    LOGS_DEFAULT(VERBOSE) << "SubgroupMatrixConv rejected: data_type is not F16.";
    return false;
  }
  if (!is_channels_last || group != 1) {
    LOGS_DEFAULT(VERBOSE) << "SubgroupMatrixConv rejected: requires channels_last && group==1, but got"
                          << " is_channels_last=" << is_channels_last
                          << " group=" << group << ".";
    return false;
  }
  if (kernel_shape.NumDimensions() != 4) {
    LOGS_DEFAULT(VERBOSE) << "SubgroupMatrixConv rejected: kernel is not 4D (rank="
                          << kernel_shape.NumDimensions() << ", shape=" << kernel_shape.ToString() << ").";
    return false;
  }

  const uint32_t channel_output = narrow<uint32_t>(kernel_shape[0]);  // Cout (GEMM N)
  const uint32_t channel_input = narrow<uint32_t>(kernel_shape[1]);   // Cin
  const uint32_t kernel_height = narrow<uint32_t>(kernel_shape[2]);
  const uint32_t kernel_width = narrow<uint32_t>(kernel_shape[3]);

  // Mirror the im2col-matmul path: skip 1xW / Hx1 (and 1x1) kernels, which the
  // plain MatMul / naive conv paths handle better. This also excludes reshaped
  // Conv1D (kernel becomes 1xW).
  if (kernel_height == 1 || kernel_width == 1) {
    LOGS_DEFAULT(VERBOSE) << "SubgroupMatrixConv rejected: 1xW/Hx1/1x1 kernel not handled (kernel_shape[OIHW]="
                          << kernel_shape.ToString() << ", kh=" << kernel_height << " kw=" << kernel_width << ").";
    return false;
  }

  const uint32_t K = kernel_height * kernel_width * channel_input;
  const uint32_t N = channel_output;
  // The subgroup-matrix K must be a whole number of 16-wide blocks, and the f16
  // right-operand (weight) row stride must be 4-byte aligned (even N).
  if (K == 0 || K % 16 != 0 || N == 0 || N % 2 != 0) {
    LOGS_DEFAULT(VERBOSE) << "SubgroupMatrixConv rejected: K/N constraints not met (kernel_shape[OIHW]="
                          << kernel_shape.ToString() << ", Cout(N)=" << N << " Cin=" << channel_input
                          << " kh=" << kernel_height << " kw=" << kernel_width << " K=kh*kw*Cin=" << K
                          << "). Need K%16==0 (K%16=" << (K % 16) << ") and N even (N%2=" << (N % 2) << ").";
    return false;
  }

  const int32_t config_index = GetSupportedConfigIndex(context);
  if (config_index < 0) {
    LOGS_DEFAULT(VERBOSE) << "SubgroupMatrixConv rejected: device does not report the required 8x16x16 F16"
                          << " subgroup-matrix config (or vendor != intel). vendor=\""
                          << std::string_view{context.AdapterInfo().vendor} << "\" architecture=\""
                          << std::string_view{context.AdapterInfo().architecture} << "\".";
    return false;
  }

  LOGS_DEFAULT(VERBOSE) << "SubgroupMatrixConv ACCEPTED: kernel_shape[OIHW]=" << kernel_shape.ToString()
                        << " Cout(N)=" << N << " Cin=" << channel_input << " kh=" << kernel_height
                        << " kw=" << kernel_width << " K=" << K << ".";
  return true;
}

template <bool is_channels_last, bool is_fused>
std::unique_ptr<typename Conv<is_channels_last, is_fused>::ConvOptImpl> CreateSubgroupMatrixConvImpl(
    const Conv<is_channels_last, is_fused>& parent,
    const ComputeContextBase& context) {
  // Only channels-last Conv is served by this kernel; the caller falls back to the
  // normal Conv path for channels-first.
  if constexpr (!is_channels_last) {
    ORT_UNUSED_PARAMETER(parent);
    ORT_UNUSED_PARAMETER(context);
    return nullptr;
  } else {
    // Only run on devices that report the fixed 8x16x16 F16 subgroup-matrix config
    // this kernel is implemented for, and for which a vendor tiling policy exists
    // (Intel today - GetSupportedConfigIndex enforces both).
    const int32_t config_index = GetSupportedConfigIndex(context);
    if (config_index < 0) {
      return nullptr;
    }
    // The Conv selector shares the autotuner hooks (tile override, device-arch
    // capture) with MatMul, but budgets the extra im2col staging the Conv kernel
    // keeps in workgroup memory and consults the Conv-specific pretuned data.
    SubgroupMatrixTilingSelector tiling_selector = intel::CreateSubgroupMatrixConvTilingSelector(context);
    if (!tiling_selector) {
      return nullptr;
    }
    return std::make_unique<SubgroupMatrixConvImpl<is_channels_last, is_fused>>(parent, config_index,
                                                                               std::move(tiling_selector));
  }
}

// Explicit instantiation for every Conv instantiation (see conv.cc).
#define WEBGPU_INSTANTIATE_CREATE_SUBGROUP_MATRIX_CONV_IMPL(CHANNELS_LAST, FUSED)               \
  template std::unique_ptr<Conv<CHANNELS_LAST, FUSED>::ConvOptImpl>                            \
  CreateSubgroupMatrixConvImpl<CHANNELS_LAST, FUSED>(const Conv<CHANNELS_LAST, FUSED>& parent, \
                                                     const ComputeContextBase& context);

WEBGPU_INSTANTIATE_CREATE_SUBGROUP_MATRIX_CONV_IMPL(false, false)
WEBGPU_INSTANTIATE_CREATE_SUBGROUP_MATRIX_CONV_IMPL(false, true)
WEBGPU_INSTANTIATE_CREATE_SUBGROUP_MATRIX_CONV_IMPL(true, false)
WEBGPU_INSTANTIATE_CREATE_SUBGROUP_MATRIX_CONV_IMPL(true, true)

#undef WEBGPU_INSTANTIATE_CREATE_SUBGROUP_MATRIX_CONV_IMPL

}  // namespace webgpu
}  // namespace onnxruntime

#endif  // !defined(__wasm__)

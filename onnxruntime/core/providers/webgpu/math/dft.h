// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"

namespace onnxruntime {
namespace webgpu {

// Where an FFT pass reads its samples from.
enum class FftSource : uint8_t {
  kSignal,   // the operator input (DFT signal, or STFT frames gathered from the signal)
  kScratch,  // a complex vec2<f32> scratch buffer laid out [rows, row_length]
  kChirp,    // the Bluestein convolution kernel, synthesized in the shader
};

// Real-signal packing: a length-2H real transform runs as one length-H complex transform.
enum class FftPacking : uint8_t {
  kNone,
  kRealForward,  // real input -> spectrum (RFFT)
  kRealInverse,  // Hermitian half spectrum -> real output (IRFFT)
};

// Compile-time description of the operator-level signal: how logical sample i of transform row r is
// read from the operator input, and how logical bin k of row r is written to the operator output.
struct FftSignalLayout {
  bool is_stft{false};
  bool has_window{false};
  bool hermitian{false};     // the input is a half spectrum that is Hermitian-extended on load
  bool strided_rows{false};  // consecutive rows are adjacent in memory (DFT along a non-trailing axis)
  bool inverse{false};       // operator direction (sets the Bluestein chirp sign)
  uint32_t input_components{1};
  uint32_t output_components{2};
  uint32_t dft_length{1};
};

// One shared-memory pass: `transforms` length-`length` FFTs per workgroup, `threads` threads each.
// Transform t covers row t / groups and group g = t % groups; its element j is the logical sample
// g * load_group_stride + j * load_stride, and its bin k the logical bin g * store_group_stride +
// k * store_stride. A single-pass transform has one group; a four-step transform uses two passes
// whose groups are the columns and rows of the length-row_length transform.
struct FftPassDesc {
  uint32_t length{1};
  uint32_t threads{1};
  uint32_t transforms{1};
  uint32_t groups{1};
  uint32_t load_group_stride{0};
  uint32_t load_stride{1};
  uint32_t store_group_stride{0};
  uint32_t store_stride{1};
  uint32_t row_length{1};
  bool inverse{false};
  FftSource source{FftSource::kSignal};
  bool to_output{true};
  bool load_chirp{false};     // Bluestein: multiply sample i by chirp(i)
  bool load_multiply{false};  // Bluestein: multiply sample i by the kernel spectrum
  bool store_twiddle{false};  // four-step: multiply bin k of group g by W_row^(g*k)
  bool store_chirp{false};    // Bluestein: multiply bin k by chirp(k)
  double scale{1.0};
  FftPacking packing{FftPacking::kNone};
};

class FftProgram final : public Program<FftProgram> {
 public:
  FftProgram(const FftSignalLayout& signal, const FftPassDesc& pass)
      : Program{"FFT"}, signal_{signal}, pass_{pass} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"count", ProgramUniformVariableDataType::Uint32},
      {"signal_length", ProgramUniformVariableDataType::Uint32},
      {"inner", ProgramUniformVariableDataType::Uint32},
      {"output_length", ProgramUniformVariableDataType::Uint32},
      {"frame_step", ProgramUniformVariableDataType::Uint32},
      {"frames", ProgramUniformVariableDataType::Uint32});

 private:
  FftSignalLayout signal_;
  FftPassDesc pass_;
};

class DFT final : public WebGpuKernel {
 public:
  DFT(const OpKernelInfo& info) : WebGpuKernel{info} {
    opset_ = info.node().SinceVersion();
    is_onesided_ = info.GetAttrOrDefault<int64_t>("onesided", 0) != 0;
    is_inverse_ = info.GetAttrOrDefault<int64_t>("inverse", 0) != 0;
    // Opset 20 moves axis from an attribute to input 2; -2 is its spec default.
    axis_ = opset_ < 20 ? info.GetAttrOrDefault<int64_t>("axis", 1) : -2;
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int64_t axis_;
  bool is_onesided_;
  bool is_inverse_;
  int opset_;
};

class STFT final : public WebGpuKernel {
 public:
  STFT(const OpKernelInfo& info) : WebGpuKernel{info} {
    is_onesided_ = info.GetAttrOrDefault<int64_t>("onesided", 1) != 0;
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  bool is_onesided_;
};

}  // namespace webgpu
}  // namespace onnxruntime

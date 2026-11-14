// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/math/dft.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <utility>

#include "core/common/inlined_containers.h"
#include "core/providers/common.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

// ONNX DFT (opset 17-20) and STFT (opset 17) on a shared FFT engine. Like the MLX execution provider
// (which lowers both ops onto mlx::fft), every transform runs in float32 and follows the strategy of
// MLX's Metal FFT:
//  * lengths whose prime factors are all <= 13 and that fit in workgroup memory run in a single pass:
//    a register-staged mixed-radix (2/3/4/5/7/8/11/13) Stockham FFT, packing several short transforms
//    into one workgroup;
//  * real forward transforms (RFFT, the onesided STFT) and IRFFT run as one complex transform of half
//    the length, packing even/odd samples into the real/imaginary lanes;
//  * longer lengths use the four-step algorithm: two passes over a global scratch buffer, with the
//    inter-pass twiddles fused into the first pass's store;
//  * all other lengths use Bluestein's chirp-z algorithm on a power-of-two convolution, with the chirp
//    pre/post multiplications and the spectrum product fused into the neighbouring passes.
// STFT framing and windowing are fused into the first pass's loads, so frames are never materialized.
// Forward transforms are unnormalized and inverse transforms are scaled by 1/N, matching the CPU kernel.

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    DFT,
    kOnnxDomain,
    17, 19,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", WebGpuSupportedFloatTypes())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<int32_t>(),
                               DataTypeImpl::GetTensorType<int64_t>()})
        .InputMemoryType(OrtMemTypeCPU, 1),
    DFT);

ONNX_OPERATOR_KERNEL_EX(
    DFT,
    kOnnxDomain,
    20,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", WebGpuSupportedFloatTypes())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<int32_t>(),
                               DataTypeImpl::GetTensorType<int64_t>()})
        .InputMemoryType(OrtMemTypeCPU, 1)
        .InputMemoryType(OrtMemTypeCPU, 2),
    DFT);

ONNX_OPERATOR_KERNEL_EX(
    STFT,
    kOnnxDomain,
    17,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", WebGpuSupportedFloatTypes())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<int32_t>(),
                               DataTypeImpl::GetTensorType<int64_t>()})
        .InputMemoryType(OrtMemTypeCPU, 1)
        .InputMemoryType(OrtMemTypeCPU, 3),
    STFT);

namespace {

constexpr uint32_t kMaxWorkgroupSize = 256;
// Upper bound on the complex samples held in workgroup memory (32 KiB); the device limit may lower it.
constexpr uint32_t kMaxSharedLength = 4096;
// Target number of samples each thread holds per transform.
constexpr uint32_t kSamplesPerThread = 8;
constexpr double kPi = 3.141592653589793238462643383279502884;

constexpr uint32_t kRadices[] = {8, 4, 2, 3, 5, 7, 11, 13};

bool Factorize(uint32_t length, InlinedVector<uint32_t>* radices) {
  for (uint32_t radix : kRadices) {
    while (length % radix == 0) {
      length /= radix;
      if (radices != nullptr) {
        radices->push_back(radix);
      }
    }
  }
  return length == 1;
}

uint32_t FloorPow2(uint32_t v) {
  uint32_t p = 1;
  while (p <= v / 2) {
    p *= 2;
  }
  return p;
}

uint64_t CeilPow2(uint64_t v) {
  uint64_t p = 1;
  while (p < v) {
    p *= 2;
  }
  return p;
}

std::string F(double value) {
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(9) << value;
  return oss.str();
}

std::string U(uint64_t value) { return std::to_string(value) + "u"; }

// Emits `let {y}{k} = ...` for the unscaled length-x.size() DFT of the samples named in `x`. Butterflies
// are scalarized into named values (no arrays) so they stay in registers.
void EmitButterfly(OStringStream& os, const std::string& ind, const InlinedVector<std::string>& x,
                   const std::string& y, int sign) {
  const uint32_t radix = static_cast<uint32_t>(x.size());
  // Multiplication by sign * i.
  auto rot = [&](const std::string& v) {
    return sign < 0 ? "vec2<f32>(" + v + ".y, -" + v + ".x)" : "vec2<f32>(-" + v + ".y, " + v + ".x)";
  };
  auto dft4 = [&](const std::string& a0, const std::string& a1, const std::string& a2, const std::string& a3,
                  const std::string& out) {
    os << ind << "let " << out << "s02 = " << a0 << " + " << a2 << "; let " << out << "d02 = " << a0 << " - " << a2
       << ";\n"
       << ind << "let " << out << "s13 = " << a1 << " + " << a3 << "; let " << out << "d13 = " << a1 << " - " << a3
       << ";\n"
       << ind << "let " << out << "r13 = " << rot(out + "d13") << ";\n"
       << ind << "let " << out << "0 = " << out << "s02 + " << out << "s13; let " << out << "1 = " << out << "d02 + "
       << out << "r13;\n"
       << ind << "let " << out << "2 = " << out << "s02 - " << out << "s13; let " << out << "3 = " << out << "d02 - "
       << out << "r13;\n";
  };
  if (radix == 2) {
    os << ind << "let " << y << "0 = " << x[0] << " + " << x[1] << "; let " << y << "1 = " << x[0] << " - " << x[1]
       << ";\n";
  } else if (radix == 4) {
    dft4(x[0], x[1], x[2], x[3], y);
  } else if (radix == 8) {
    const std::string c = F(std::sqrt(0.5));
    const std::string sc = F(sign * std::sqrt(0.5));
    const std::string e = y + "e", o = y + "o";
    dft4(x[0], x[2], x[4], x[6], e);
    dft4(x[1], x[3], x[5], x[7], o);
    os << ind << "let " << o << "w1 = cmul(" << o << "1, vec2<f32>(" << c << ", " << sc << "));\n"
       << ind << "let " << o << "w2 = " << rot(o + "2") << ";\n"
       << ind << "let " << o << "w3 = cmul(" << o << "3, vec2<f32>(-" << c << ", " << sc << "));\n"
       << ind << "let " << y << "0 = " << e << "0 + " << o << "0; let " << y << "4 = " << e << "0 - " << o << "0;\n";
    for (int k = 1; k < 4; ++k) {
      os << ind << "let " << y << k << " = " << e << k << " + " << o << "w" << k << "; let " << y << k + 4 << " = "
         << e << k << " - " << o << "w" << k << ";\n";
    }
  } else {
    // Odd radix, using the conjugate-pair symmetry of the DFT matrix:
    // y[k] = x0 + sum_j cos(jk) (x[j] + x[r-j]) + i sin(jk) (x[j] - x[r-j]), and y[r-k] its mirror.
    const uint32_t half = radix / 2;
    for (uint32_t j = 1; j <= half; ++j) {
      os << ind << "let " << y << "p" << j << " = " << x[j] << " + " << x[radix - j] << "; let " << y << "m" << j
         << " = " << x[j] << " - " << x[radix - j] << ";\n";
    }
    os << ind << "let " << y << "0 = " << x[0];
    for (uint32_t j = 1; j <= half; ++j) os << " + " << y << "p" << j;
    os << ";\n";
    for (uint32_t k = 1; k <= half; ++k) {
      os << ind << "let " << y << "a" << k << " = " << x[0];
      for (uint32_t j = 1; j <= half; ++j) {
        os << " + " << F(std::cos(sign * 2.0 * kPi * ((j * k) % radix) / radix)) << " * " << y << "p" << j;
      }
      os << ";\n"
         << ind << "let " << y << "b" << k << " = ";
      for (uint32_t j = 1; j <= half; ++j) {
        os << (j > 1 ? " + " : "") << F(std::sin(sign * 2.0 * kPi * ((j * k) % radix) / radix)) << " * " << y << "m"
           << j;
      }
      os << ";\n"
         << ind << "let " << y << k << " = " << y << "a" << k << " + vec2<f32>(-" << y << "b" << k << ".y, " << y << "b"
         << k << ".x);\n"
         << ind << "let " << y << radix - k << " = " << y << "a" << k << " - vec2<f32>(-" << y << "b" << k << ".y, "
         << y << "b" << k << ".x);\n";
    }
  }
}

// How a stage reads its legs and writes its outputs, given an index into the current transform.
struct StageIo {
  std::function<std::string(const std::string&)> read;
  std::function<std::string(const std::string&, const std::string&)> write;
  bool staged{false};        // reads and writes shared memory: stage legs in registers behind a barrier
  bool writes_shared{true};  // a fused last stage writes global memory and needs no trailing barrier
};

// One radix-`radix` Stockham stage over a `length`-sample transform.
void EmitStage(OStringStream& os, uint32_t radix, uint32_t sub_transform, uint32_t length, uint32_t threads,
               int sign, const StageIo& io) {
  const uint32_t butterflies = length / radix;
  const uint32_t per_thread = (butterflies + threads - 1) / threads;
  const bool guarded = butterflies % threads != 0;
  auto leg = [](uint32_t q, uint32_t j) { return "x" + std::to_string(q) + "_" + std::to_string(j); };
  auto open = [&](uint32_t q) {
    os << "    {\n      let t = lt + " << U(q * threads) << ";\n";
    if (guarded) os << "      if (t < " << U(butterflies) << ") {\n";
  };
  auto close = [&]() {
    if (guarded) os << "      }\n";
    os << "    }\n";
  };
  auto load = [&](uint32_t q) {
    if (sub_transform > 1) {
      os << "      let tw = f32(t % " << U(sub_transform) << ");\n";
    }
    for (uint32_t j = 0; j < radix; ++j) {
      const std::string src = io.read("t + " + U(j * butterflies));
      os << "      " << leg(q, j) << " = ";
      if (j == 0 || sub_transform == 1) {
        os << src << ";\n";
      } else {
        os << "cmul(" << src << ", expi(" << F(sign * 2.0 * kPi * j / (static_cast<double>(radix) * sub_transform))
           << " * tw));\n";
      }
    }
  };
  auto compute = [&](uint32_t q) {
    os << "      let p = t % " << U(sub_transform) << ";\n"
       << "      let ob = (t - p) * " << U(radix) << " + p;\n";
    InlinedVector<std::string> legs;
    for (uint32_t j = 0; j < radix; ++j) legs.push_back(leg(q, j));
    EmitButterfly(os, "      ", legs, "y", sign);
    for (uint32_t k = 0; k < radix; ++k) {
      os << "      " << io.write("ob + " + U(k * sub_transform), "y" + std::to_string(k)) << "\n";
    }
  };

  os << "  {\n";
  for (uint32_t q = 0; q < per_thread; ++q) {
    for (uint32_t j = 0; j < radix; ++j) {
      os << "    var " << leg(q, j) << ": vec2<f32>;\n";
    }
  }
  if (io.staged) {
    for (uint32_t q = 0; q < per_thread; ++q) {
      open(q);
      load(q);
      close();
    }
    os << "    workgroupBarrier();\n";
    for (uint32_t q = 0; q < per_thread; ++q) {
      open(q);
      compute(q);
      close();
    }
  } else {
    for (uint32_t q = 0; q < per_thread; ++q) {
      open(q);
      load(q);
      compute(q);
      close();
    }
  }
  if (io.writes_shared) os << "    workgroupBarrier();\n";
  os << "  }\n";
}

std::string CacheKey(const FftSignalLayout& s, const FftPassDesc& p) {
  std::ostringstream key;
  key << s.is_stft << s.has_window << s.hermitian << s.strided_rows << s.inverse << s.input_components
      << s.output_components << "_" << s.dft_length << "_" << p.length << "_" << p.threads << "_" << p.transforms
      << "_" << p.groups << "_" << p.load_group_stride << "_" << p.load_stride << "_" << p.store_group_stride << "_"
      << p.store_stride << "_" << p.row_length << "_" << p.inverse << static_cast<int>(p.source) << p.to_output
      << p.load_chirp << p.load_multiply << p.store_twiddle << p.store_chirp << static_cast<int>(p.packing)
      << "_"
      << F(p.scale);
  return key.str();
}

}  // namespace

Status FftProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const FftSignalLayout& s = signal_;
  const FftPassDesc& p = pass_;
  const uint32_t n = p.length;
  const uint32_t tpw = p.transforms;
  const uint32_t workgroup = p.threads * tpw;
  const uint32_t dl = s.dft_length;
  const int sign = p.inverse ? 1 : -1;

  const ShaderVariableHelper* x = nullptr;
  const ShaderVariableHelper* window = nullptr;
  const ShaderVariableHelper* scratch = nullptr;
  const ShaderVariableHelper* kernel = nullptr;
  if (p.source == FftSource::kSignal) {
    x = &shader.AddInput("x", ShaderUsage::UseUniform);
    if (s.is_stft && s.has_window) {
      window = &shader.AddInput("window", ShaderUsage::UseUniform);
    }
  } else if (p.source == FftSource::kScratch) {
    scratch = &shader.AddInput("scratch", ShaderUsage::UseUniform);
    if (p.load_multiply) {
      kernel = &shader.AddInput("kernel", ShaderUsage::UseUniform);
    }
  }
  const auto& out = p.to_output ? shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias)
                                : shader.AddOutput("scratch_out", ShaderUsage::UseUniform);

  InlinedVector<uint32_t> radices;
  ORT_RETURN_IF_NOT(Factorize(n, &radices), "FFT pass length ", n, " is not 13-smooth.");

  auto& impl = shader.AdditionalImplementation();
  impl << "var<workgroup> smem: array<vec2<f32>, " << tpw * n << ">;\n"
       << "fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {\n"
       << "  return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);\n"
       << "}\n"
       << "fn expi(a: f32) -> vec2<f32> { return vec2<f32>(cos(a), sin(a)); }\n";

  if (p.load_chirp || p.store_chirp || p.source == FftSource::kChirp) {
    // chirp(i) = exp(sign * i*pi * i^2 / N), with i^2 reduced mod 2N exactly (and without u32 overflow).
    const uint64_t modulus = 2ull * dl;
    impl << "fn chirp(i: u32) -> vec2<f32> {\n"
         << "  let a = i % " << U(modulus) << ";\n"
         << "  var r = 0u;\n"
         << "  if (a < 65536u) {\n"
         << "    r = (a * a) % " << U(modulus) << ";\n"
         << "  } else {\n"
         << "    var b = a;\n"
         << "    var y = a;\n"
         << "    loop {\n"
         << "      if (y == 0u) { break; }\n"
         << "      if ((y & 1u) != 0u) { r = (r + b) % " << U(modulus) << "; }\n"
         << "      b = (b * 2u) % " << U(modulus) << ";\n"
         << "      y = y >> 1u;\n"
         << "    }\n"
         << "  }\n"
         << "  return expi(" << F((s.inverse ? kPi : -kPi) / dl) << " * f32(r));\n"
         << "}\n";
  }

  if (p.source == FftSource::kSignal) {
    // Complex samples are bound as vec2 elements, so offsets count samples.
    auto element = [&](const std::string& offset) {
      return s.input_components == 2 ? "vec2<f32>(" + x->GetByOffset(offset) + ")"
                                     : "vec2<f32>(f32(" + x->GetByOffset(offset) + "), 0.0)";
    };
    // `local_row` indexes the rows of this dispatch; operator I/O starts `row_offset` rows in.
    impl << "fn read_signal(local_row: u32, i: u32) -> vec2<f32> {\n"
         << "  let row = local_row + uniforms.row_offset;\n";
    if (s.is_stft) {
      impl << "  let batch = row / uniforms.frames;\n"
           << "  let frame = row - batch * uniforms.frames;\n"
           << "  let off = batch * uniforms.signal_length + frame * uniforms.frame_step + i;\n";
      if (window != nullptr) {
        impl << "  return " << element("off") << " * f32(" << window->GetByOffset("i") << ");\n";
      } else {
        impl << "  return " << element("off") << ";\n";
      }
    } else if (s.strided_rows) {
      impl << "  let outer = row / uniforms.inner;\n"
           << "  let within = row - outer * uniforms.inner;\n"
           << "  return " << element("(outer * uniforms.signal_length + i) * uniforms.inner + within") << ";\n";
    } else {
      impl << "  return " << element("row * uniforms.signal_length + i") << ";\n";
    }
    impl << "}\n";

    const uint32_t half_spectrum = dl / 2 + 1;
    if (p.packing == FftPacking::kRealInverse) {
      // Half-spectrum bin k in [0, N/2]; missing bins are zero. The imaginary parts of the DC and
      // Nyquist bins cannot contribute to a real signal and are dropped.
      impl << "fn half_bin(row: u32, k: u32) -> vec2<f32> {\n"
           << "  if (k >= min(uniforms.signal_length, " << U(half_spectrum) << ")) { return vec2<f32>(0.0); }\n"
           << "  let v = read_signal(row, k);\n"
           << "  if (k == 0u || k == " << U(dl / 2) << ") { return vec2<f32>(v.x, 0.0); }\n"
           << "  return v;\n"
           << "}\n";
    }
    impl << "fn load_sample(row: u32, i: u32) -> vec2<f32> {\n";
    if (p.row_length > dl) {  // Bluestein zero padding
      impl << "  if (i >= " << U(dl) << ") { return vec2<f32>(0.0); }\n";
    }
    if (s.hermitian) {
      // The spectrum is the Hermitian extension of the half spectrum, zero-padded or truncated when
      // dft_length implies a different bin count than the input provides.
      impl << "  let provided = min(uniforms.signal_length, " << U(half_spectrum) << ");\n"
           << "  if (i < " << U(half_spectrum) << ") {\n"
           << "    if (i < provided) { return read_signal(row, i); }\n"
           << "    return vec2<f32>(0.0);\n"
           << "  }\n"
           << "  let m = " << U(dl) << " - i;\n"
           << "  if (m < provided) { let h = read_signal(row, m); return vec2<f32>(h.x, -h.y); }\n"
           << "  return vec2<f32>(0.0);\n";
    } else if (s.is_stft) {
      impl << "  return read_signal(row, i);\n";
    } else {
      impl << "  if (i < uniforms.signal_length) { return read_signal(row, i); }\n"
           << "  return vec2<f32>(0.0);\n";
    }
    impl << "}\n";
  } else if (p.source == FftSource::kScratch) {
    impl << "fn load_sample(row: u32, i: u32) -> vec2<f32> {\n"
         << "  let v = " << scratch->GetByOffset("row * " + U(p.row_length) + " + i") << ";\n";
    if (kernel != nullptr) {
      impl << "  return cmul(v, " << kernel->GetByOffset("i") << ");\n";
    } else {
      impl << "  return v;\n";
    }
    impl << "}\n";
  } else {
    // Bluestein kernel b[m] = conj(chirp(m)) for |m| < N, circularly wrapped to length row_length.
    impl << "fn load_sample(row: u32, i: u32) -> vec2<f32> {\n"
         << "  var m = i;\n"
         << "  if (m >= " << U(dl) << ") { m = " << U(p.row_length) << " - i; }\n"
         << "  if (m >= " << U(dl) << ") { return vec2<f32>(0.0); }\n"
         << "  let c = chirp(m);\n"
         << "  return vec2<f32>(c.x, -c.y);\n"
         << "}\n";
  }

  impl << "fn store_sample(local_row: u32, k: u32, v: vec2<f32>) {\n";
  if (p.to_output) {
    impl << "  let row = local_row + uniforms.row_offset;\n";
    if (s.strided_rows) {
      impl << "  let outer = row / uniforms.inner;\n"
           << "  let within = row - outer * uniforms.inner;\n"
           << "  let off = (outer * uniforms.output_length + k) * uniforms.inner + within;\n";
    } else {
      impl << "  let off = row * uniforms.output_length + k;\n";
    }
    const char* value = s.output_components == 2 ? "vec2<output_element_t>(v)" : "output_element_t(v.x)";
    impl << "  " << out.SetByOffset("off", value) << "\n";
  } else {
    impl << "  " << out.SetByOffset("local_row * " + U(p.row_length) + " + k", "v") << "\n";
  }
  impl << "}\n";

  if (p.packing == FftPacking::kRealForward) {
    // Bin k in [0, N/2] of a real signal x from Z = FFT_{N/2}(x[2m] + i x[2m+1]):
    // X[k] = E[k] + W_N^k O[k], E = (Z[k] + conj Z[-k]) / 2, O = (Z[k] - conj Z[-k]) / 2i.
    impl << "fn real_bin(sb: u32, k: u32) -> vec2<f32> {\n"
         << "  let a = smem[sb + k % " << U(n) << "];\n"
         << "  let b = smem[sb + (" << U(n) << " - k) % " << U(n) << "];\n"
         << "  let bc = vec2<f32>(b.x, -b.y);\n"
         << "  let e = (a + bc) * 0.5;\n"
         << "  let d = (a - bc) * 0.5;\n"
         << "  return e + cmul(vec2<f32>(d.y, -d.x), expi(" << F(-2.0 * kPi / dl) << " * f32(k)));\n"
         << "}\n";
  }

  // Consecutive threads walk consecutive transforms when the transforms' samples are interleaved in
  // memory (strided loads/stores), and consecutive samples otherwise, so global accesses coalesce.
  const bool load_interleaved = p.load_stride != 1 || (p.source == FftSource::kSignal && s.strided_rows);
  const bool store_interleaved = p.store_stride != 1 || (p.to_output && s.strided_rows);
  auto split = [&](bool interleaved, const std::string& per_transform) {
    return interleaved ? "    let tt = e % " + U(tpw) + ";\n    let j = e / " + U(tpw) + ";\n"
                       : "    let tt = e / " + per_transform + ";\n    let j = e % " + per_transform + ";\n";
  };
  auto row_and_group = [&]() {
    if (p.groups == 1) {
      return std::string("      let row = gt;\n      let g = 0u;\n");
    }
    return "      let row = gt / " + U(p.groups) + ";\n      let g = gt - row * " + U(p.groups) + ";\n";
  };

  // fetch(t, j): sample j of transform t, with the load hooks applied (zero past the last transform).
  impl << "fn fetch(gt: u32, j: u32) -> vec2<f32> {\n"
       << "  if (gt >= uniforms.count) { return vec2<f32>(0.0); }\n"
       << row_and_group();
  if (p.packing == FftPacking::kRealForward) {
    impl << "  return vec2<f32>(load_sample(row, 2u * j).x, load_sample(row, 2u * j + 1u).x);\n";
  } else if (p.packing == FftPacking::kRealInverse) {
    // Z[k] = E[k] + i O[k], E = (X[k] + conj X[N/2-k]) / 2, O = (X[k] - conj X[N/2-k]) W_N^-k / 2.
    impl << "  let a = half_bin(row, j);\n"
         << "  let b = half_bin(row, " << U(n) << " - j);\n"
         << "  let bc = vec2<f32>(b.x, -b.y);\n"
         << "  let o = cmul((a - bc) * 0.5, expi(" << F(2.0 * kPi / dl) << " * f32(j)));\n"
         << "  return (a + bc) * 0.5 + vec2<f32>(-o.y, o.x);\n";
  } else {
    impl << "  let i = g * " << U(p.load_group_stride) << " + j * " << U(p.load_stride) << ";\n";
    if (p.load_chirp) {
      impl << "  return cmul(load_sample(row, i), chirp(i));\n";
    } else {
      impl << "  return load_sample(row, i);\n";
    }
  }
  impl << "}\n";

  if (p.packing == FftPacking::kNone) {
    // emit(t, j, v): bin j of transform t, with the store hooks applied.
    impl << "fn emit(gt: u32, j: u32, value: vec2<f32>) {\n"
         << "  if (gt >= uniforms.count) { return; }\n"
         << row_and_group()
         << "  let k = g * " << U(p.store_group_stride) << " + j * " << U(p.store_stride) << ";\n";
    if (p.to_output) {
      impl << "  if (k >= uniforms.output_length) { return; }\n";
    }
    impl << "  var v = value;\n";
    if (p.store_twiddle) {
      impl << "  v = cmul(v, expi(" << F(sign * 2.0 * kPi / p.row_length) << " * f32(g * j)));\n";
    }
    if (p.store_chirp) {
      impl << "  v = cmul(v, chirp(k));\n";
    }
    if (p.scale != 1.0) {
      impl << "  v = v * " << F(p.scale) << ";\n";
    }
    impl << "  store_sample(row, k, v);\n"
         << "}\n";
  }

  // Unless consecutive transforms interleave in memory, the first stage reads its legs straight from
  // global memory and the last stage writes its outputs straight back: both access patterns already
  // coalesce, and skipping the shared-memory staging saves a full pass and barrier on each side.
  const bool direct_load = !load_interleaved && !radices.empty();
  const bool direct_store = !store_interleaved && p.packing == FftPacking::kNone && !radices.empty();

  auto& body = shader.MainFunctionBody();
  body << "  let tl = local_idx / " << U(p.threads) << ";\n"
       << "  let lt = local_idx % " << U(p.threads) << ";\n"
       << "  let sbase = tl * " << U(n) << ";\n"
       << "  let first = workgroup_idx * " << U(tpw) << ";\n"
       << "  let gt_self = first + tl;\n";

  if (!direct_load) {
    body << "  for (var e = local_idx; e < " << U(tpw * n) << "; e += " << U(workgroup) << ") {\n"
         << split(load_interleaved, U(n))
         << "    smem[tt * " << U(n) << " + j] = fetch(first + tt, j);\n"
         << "  }\n"
         << "  workgroupBarrier();\n";
  }

  uint32_t sub_transform = 1;
  for (size_t stage = 0; stage < radices.size(); ++stage) {
    StageIo io;
    if (stage == 0 && direct_load) {
      io.read = [](const std::string& index) { return "fetch(gt_self, " + index + ")"; };
    } else {
      io.read = [](const std::string& index) { return "smem[sbase + " + index + "]"; };
    }
    if (stage + 1 == radices.size() && direct_store) {
      io.write = [](const std::string& index, const std::string& value) {
        return "emit(gt_self, " + index + ", " + value + ");";
      };
    } else {
      io.write = [](const std::string& index, const std::string& value) {
        return "smem[sbase + " + index + "] = " + value + ";";
      };
    }
    // Stages run in place, so a stage that both reads and writes shared memory must stage its legs in
    // registers behind a barrier.
    io.writes_shared = !(stage + 1 == radices.size() && direct_store);
    io.staged = !(stage == 0 && direct_load) && io.writes_shared;
    EmitStage(body, radices[stage], sub_transform, n, p.threads, sign, io);
    sub_transform *= radices[stage];
  }

  if (p.packing == FftPacking::kRealForward) {
    body << "  for (var e = local_idx; e < " << U(tpw) << " * uniforms.output_length; e += " << U(workgroup) << ") {\n"
         << split(store_interleaved, "uniforms.output_length")
         << "    let gt = first + tt;\n"
         << "    if (gt < uniforms.count) {\n"
         << "      var v: vec2<f32>;\n"
         << "      if (j <= " << U(n) << ") {\n"
         << "        v = real_bin(tt * " << U(n) << ", j);\n"
         << "      } else {\n"
         << "        let m = real_bin(tt * " << U(n) << ", " << U(dl) << " - j);\n"
         << "        v = vec2<f32>(m.x, -m.y);\n"
         << "      }\n"
         << "      store_sample(gt, j, v);\n"
         << "    }\n"
         << "  }\n";
  } else if (p.packing == FftPacking::kRealInverse) {
    body << "  for (var e = local_idx; e < " << U(tpw * dl) << "; e += " << U(workgroup) << ") {\n"
         << split(store_interleaved, U(dl))
         << "    let gt = first + tt;\n"
         << "    if (gt < uniforms.count) {\n"
         << "      let z = smem[tt * " << U(n) << " + j / 2u];\n"
         << "      let r = select(z.y, z.x, (j & 1u) == 0u) * " << F(1.0 / n) << ";\n"
         << "      store_sample(gt, j, vec2<f32>(r, 0.0));\n"
         << "    }\n"
         << "  }\n";
  } else if (!direct_store) {
    body << "  for (var e = local_idx; e < " << U(tpw * n) << "; e += " << U(workgroup) << ") {\n"
         << split(store_interleaved, U(n))
         << "    emit(first + tt, j, smem[tt * " << U(n) << " + j]);\n"
         << "  }\n";
  }

  return Status::OK();
}

namespace {

// Operator-level problem: `rows` independent transforms described by `layout`.
struct FftProblem {
  FftSignalLayout layout;
  uint32_t rows{0};
  uint32_t signal_length{0};
  uint32_t inner{1};
  uint32_t output_length{0};
  uint32_t frame_step{0};
  uint32_t frames{1};
  const Tensor* x{nullptr};
  const Tensor* window{nullptr};
  Tensor* y{nullptr};
};

struct FftStep {
  FftPassDesc pass;
  uint32_t rows{0};
  int in_buffer{-1};
  int kernel_buffer{-1};
  int out_buffer{-1};
};

// Endpoints and fused pre/post-processing of one whole transform.
struct FftEnds {
  FftSource source{FftSource::kSignal};
  int in_buffer{-1};
  int kernel_buffer{-1};
  bool load_chirp{false};
  bool load_multiply{false};
  bool to_output{true};
  int out_buffer{-1};
  bool store_chirp{false};
  double scale{1.0};
};

class FftPlan {
 public:
  explicit FftPlan(uint32_t capacity) : capacity_{capacity} {}

  bool FitsOnePass(uint64_t length) const {
    return length <= capacity_ && Factorize(static_cast<uint32_t>(length), nullptr);
  }

  // length = n1 * n2 with both factors single-pass lengths, as balanced as possible.
  bool SplitFourStep(uint64_t length, uint32_t& n1, uint32_t& n2) const {
    if (length > static_cast<uint64_t>(capacity_) * capacity_) {
      return false;
    }
    bool found = false;
    for (uint32_t d = 2; d <= capacity_; ++d) {
      if (length % d != 0 || length / d > capacity_ || !FitsOnePass(d) || !FitsOnePass(length / d)) {
        continue;
      }
      const uint32_t e = static_cast<uint32_t>(length / d);
      if (!found || std::max(d, e) < std::max(n1, n2)) {
        n1 = d;
        n2 = e;
        found = true;
      }
    }
    return found;
  }

  bool CanTransform(uint64_t length) const {
    uint32_t n1 = 0, n2 = 0;
    return FitsOnePass(length) || SplitFourStep(length, n1, n2);
  }

  FftPassDesc MakePass(uint32_t length) const {
    FftPassDesc pass;
    pass.length = length;
    pass.threads = std::min(kMaxWorkgroupSize, FloorPow2(std::max(1u, length / kSamplesPerThread)));
    pass.transforms = std::max(1u, std::min(kMaxWorkgroupSize / pass.threads, capacity_ / length));
    pass.row_length = length;
    return pass;
  }

  int AddBuffer(uint32_t rows, uint32_t length) {
    buffers_.push_back({rows, length});
    return static_cast<int>(buffers_.size() - 1);
  }

  void AddStep(const FftPassDesc& pass, uint32_t rows, int in_buffer, int kernel_buffer, int out_buffer) {
    steps_.push_back({pass, rows, in_buffer, kernel_buffer, out_buffer});
  }

  // Appends the passes of one length-`length` FFT over `rows` rows: one pass when it fits in workgroup
  // memory, otherwise a four-step pair (columns, then rows) with load hooks on the first pass and store
  // hooks on the last.
  Status AppendFft(uint32_t length, uint32_t rows, bool inverse, const FftEnds& ends) {
    auto apply_load = [&](FftPassDesc& pass) {
      pass.inverse = inverse;
      pass.row_length = length;
      pass.source = ends.source;
      pass.load_chirp = ends.load_chirp;
      pass.load_multiply = ends.load_multiply;
    };
    auto apply_store = [&](FftPassDesc& pass) {
      pass.to_output = ends.to_output;
      pass.store_chirp = ends.store_chirp;
      pass.scale = ends.scale;
    };
    if (FitsOnePass(length)) {
      FftPassDesc pass = MakePass(length);
      apply_load(pass);
      apply_store(pass);
      AddStep(pass, rows, ends.in_buffer, ends.kernel_buffer, ends.out_buffer);
      return Status::OK();
    }
    uint32_t n1 = 0, n2 = 0;
    ORT_RETURN_IF_NOT(SplitFourStep(length, n1, n2), "WebGPU FFT: unsupported transform length ", length);
    const int temp = AddBuffer(rows, length);

    // Columns: X1[k1, j2] = W_N^(j2 k1) * FFT_n1(x[j1 * n2 + j2]).
    FftPassDesc columns = MakePass(n1);
    apply_load(columns);
    columns.groups = n2;
    columns.load_group_stride = 1;
    columns.load_stride = n2;
    columns.store_group_stride = 1;
    columns.store_stride = n2;
    columns.store_twiddle = true;
    columns.to_output = false;
    AddStep(columns, rows, ends.in_buffer, ends.kernel_buffer, temp);

    // Rows: X[k1 + n1 * k2] = FFT_n2(X1[k1, j2]).
    FftPassDesc row_pass = MakePass(n2);
    row_pass.inverse = inverse;
    row_pass.row_length = length;
    row_pass.source = FftSource::kScratch;
    row_pass.groups = n1;
    row_pass.load_group_stride = n2;
    row_pass.load_stride = 1;
    row_pass.store_group_stride = 1;
    row_pass.store_stride = n1;
    apply_store(row_pass);
    AddStep(row_pass, rows, temp, -1, ends.out_buffer);
    return Status::OK();
  }

  Status Build(const FftProblem& problem) {
    const FftSignalLayout& layout = problem.layout;
    const uint32_t n = layout.dft_length;
    const bool real_forward = !layout.inverse && layout.input_components == 1;
    if (n % 2 == 0 && FitsOnePass(n / 2) && (real_forward || layout.hermitian)) {
      FftPassDesc pass = MakePass(n / 2);
      pass.packing = layout.hermitian ? FftPacking::kRealInverse : FftPacking::kRealForward;
      pass.inverse = layout.hermitian;
      pass.row_length = n;
      AddStep(pass, problem.rows, -1, -1, -1);
      return Status::OK();
    }

    FftEnds direct;
    direct.scale = layout.inverse ? 1.0 / n : 1.0;
    if (CanTransform(n)) {
      return AppendFft(n, problem.rows, layout.inverse, direct);
    }

    // Bluestein: X[k] = chirp(k) * IFFT_M(FFT_M(x * chirp) * FFT_M(conj chirp))[k], M >= 2N - 1.
    const uint64_t m = CeilPow2(2ull * n - 1);
    ORT_RETURN_IF_NOT(CanTransform(m), "WebGPU FFT: transform length ", n, " is too large.");
    const uint32_t m32 = static_cast<uint32_t>(m);
    const int signal_spectrum = AddBuffer(problem.rows, m32);
    const int kernel_spectrum = AddBuffer(1, m32);

    FftEnds forward;
    forward.load_chirp = true;
    forward.to_output = false;
    forward.out_buffer = signal_spectrum;
    ORT_RETURN_IF_ERROR(AppendFft(m32, problem.rows, false, forward));

    FftEnds kernel;
    kernel.source = FftSource::kChirp;
    kernel.to_output = false;
    kernel.out_buffer = kernel_spectrum;
    ORT_RETURN_IF_ERROR(AppendFft(m32, 1, false, kernel));

    FftEnds convolve;
    convolve.source = FftSource::kScratch;
    convolve.in_buffer = signal_spectrum;
    convolve.kernel_buffer = kernel_spectrum;
    convolve.load_multiply = true;
    convolve.store_chirp = true;
    convolve.scale = direct.scale / static_cast<double>(m);
    return AppendFft(m32, problem.rows, true, convolve);
  }

  // Bytes of the largest scratch buffer that grows with the number of transform rows, per row.
  uint64_t ScratchBytesPerRow(uint32_t rows) const {
    uint64_t bytes = 0;
    for (const auto& [buffer_rows, length] : buffers_) {
      if (buffer_rows == rows) {
        bytes = std::max<uint64_t>(bytes, uint64_t{length} * 2 * sizeof(float));
      }
    }
    return bytes;
  }

  // Runs the plan on operator rows [row_offset, row_offset + problem.rows). `scratch` holds the scratch
  // buffers: when empty they are allocated into it, otherwise they come from a plan with the same buffer
  // layout and at least as many rows, and this plan uses their leading rows.
  Status Run(ComputeContext& context, const FftProblem& problem, uint32_t row_offset,
             InlinedVector<Tensor>& scratch) const {
    const bool allocate = scratch.empty();
    ORT_RETURN_IF_NOT(allocate || scratch.size() == buffers_.size(), "WebGPU FFT: scratch layout mismatch.");
    InlinedVector<Tensor> buffers;
    buffers.reserve(buffers_.size());
    for (size_t i = 0; i < buffers_.size(); ++i) {
      const auto& [rows, length] = buffers_[i];
      const TensorShape shape({static_cast<int64_t>(rows), static_cast<int64_t>(length), 2});
      if (allocate) {
        scratch.push_back(context.CreateGPUTensor(DataTypeImpl::GetType<float>(), shape));
      }
      const Tensor& buffer = scratch[i];
      ORT_RETURN_IF_NOT(buffer.Shape().Size() >= shape.Size(), "WebGPU FFT: scratch buffer too small.");
      buffers.emplace_back(buffer.DataType(), shape, const_cast<void*>(buffer.DataRaw()), buffer.Location());
    }
    for (const FftStep& step : steps_) {
      FftPassDesc pass = step.pass;
      const uint32_t count = step.rows * pass.groups;
      // Small batches do not need wide workgroups.
      pass.transforms = std::max(1u, std::min(pass.transforms, static_cast<uint32_t>(CeilPow2(count))));

      FftProgram program{problem.layout, pass};
      if (pass.source == FftSource::kSignal) {
        program.AddInput({problem.x, ProgramTensorMetadataDependency::Type, ProgramInput::Flatten,
                          static_cast<int>(problem.layout.input_components)});
        if (problem.layout.is_stft && problem.layout.has_window) {
          program.AddInput({problem.window, ProgramTensorMetadataDependency::Type});
        }
      } else if (pass.source == FftSource::kScratch) {
        program.AddInput({&buffers[step.in_buffer], ProgramTensorMetadataDependency::None, ProgramInput::Flatten, 2});
        if (pass.load_multiply) {
          program.AddInput({&buffers[step.kernel_buffer], ProgramTensorMetadataDependency::None, ProgramInput::Flatten, 2});
        }
      }
      if (pass.to_output) {
        program.AddOutput({problem.y, ProgramTensorMetadataDependency::Type, ProgramOutput::Flatten,
                           static_cast<int>(problem.layout.output_components)});
      } else {
        program.AddOutput({&buffers[step.out_buffer], ProgramTensorMetadataDependency::None, ProgramOutput::Flatten, 2});
      }
      program.CacheHint(CacheKey(problem.layout, pass))
          .SetWorkgroupSize(pass.threads * pass.transforms)
          .SetDispatchGroupSize((count + pass.transforms - 1) / pass.transforms)
          .AddUniformVariables({{count},
                                {problem.signal_length},
                                {problem.inner},
                                {problem.output_length},
                                {problem.frame_step},
                                {problem.frames},
                                {row_offset}});
      ORT_RETURN_IF_ERROR(context.RunProgram(program));
    }
    return Status::OK();
  }

 private:
  uint32_t capacity_;
  InlinedVector<std::pair<uint32_t, uint32_t>> buffers_;
  InlinedVector<FftStep> steps_;
};

uint32_t SharedCapacity(const ComputeContext& context) {
  const uint32_t complex_slots = context.DeviceLimits().maxComputeWorkgroupStorageSize / (2 * sizeof(float));
  return FloorPow2(std::max(1u, std::min(complex_slots, kMaxSharedLength)));
}

Status RunFft(ComputeContext& context, const FftProblem& problem) {
  const uint32_t capacity = SharedCapacity(context);
  FftPlan plan{capacity};
  ORT_RETURN_IF_ERROR(plan.Build(problem));

  // Scratch buffers hold every row, so run the rows in chunks whose scratch fits one buffer and one binding.
  const auto& limits = context.DeviceLimits();
  const uint64_t max_bytes = std::min<uint64_t>(limits.maxBufferSize, limits.maxStorageBufferBindingSize);
  const uint64_t row_bytes = plan.ScratchBytesPerRow(problem.rows);
  const uint64_t chunk_rows = row_bytes == 0 ? problem.rows : std::max<uint64_t>(1, max_bytes / row_bytes);
  InlinedVector<Tensor> scratch;
  if (chunk_rows >= problem.rows) {
    return plan.Run(context, problem, 0, scratch);
  }
  // The first chunk allocates the scratch for chunk_rows rows and every later chunk reuses it: the buffer
  // manager keeps released buffers until the queue is submitted, so per-chunk scratch would pile up.
  for (uint32_t begin = 0; begin < problem.rows;) {
    FftProblem part = problem;
    part.rows = static_cast<uint32_t>(std::min<uint64_t>(chunk_rows, problem.rows - begin));
    FftPlan part_plan{capacity};
    ORT_RETURN_IF_ERROR(part_plan.Build(part));
    ORT_RETURN_IF_ERROR(part_plan.Run(context, part, begin, scratch));
    begin += part.rows;
  }
  return Status::OK();
}

Status ReadOptionalScalar(const Tensor* tensor, const char* name, int64_t& value) {
  if (tensor == nullptr) {
    return Status::OK();
  }
  ORT_RETURN_IF_NOT(tensor->Shape().IsScalar(), name, " must be a scalar value.");
  if (tensor->DataType() == DataTypeImpl::GetType<int64_t>()) {
    value = tensor->Data<int64_t>()[0];
  } else {
    value = static_cast<int64_t>(tensor->Data<int32_t>()[0]);
  }
  return Status::OK();
}

bool FitsU32(int64_t value) { return value >= 0 && value <= std::numeric_limits<uint32_t>::max(); }

}  // namespace

Status DFT::ComputeInternal(ComputeContext& context) const {
  const auto* input_tensor = context.Input(0);
  const TensorShape& input_shape = input_tensor->Shape();
  const int64_t rank = static_cast<int64_t>(input_shape.NumDimensions());
  ORT_RETURN_IF(rank < 2, "DFT input must have at least 2 dimensions.");

  const int64_t input_components = input_shape[onnxruntime::narrow<size_t>(rank - 1)];
  ORT_RETURN_IF(input_components != 1 && input_components != 2,
                "DFT input's innermost dimension must be 1 (real) or 2 (complex).");

  int64_t axis = axis_;
  if (opset_ >= 20 && context.InputCount() > 2) {
    ORT_RETURN_IF_ERROR(ReadOptionalScalar(context.Input(2), "axis", axis));
  }
  axis = HandleNegativeAxis(axis, rank);
  ORT_RETURN_IF(axis == rank - 1,
                "DFT axis must refer to a signal dimension, not the innermost (real/imaginary) dimension.");
  ORT_RETURN_IF(is_inverse_ && is_onesided_ && input_components != 2,
                "Inverse one-sided DFT (IRFFT) requires complex-valued input (innermost dimension 2).");

  const int64_t signal_length = input_shape[onnxruntime::narrow<size_t>(axis)];
  int64_t length = is_inverse_ && is_onesided_ ? (signal_length - 1) * 2 : signal_length;
  if (context.InputCount() > 1 && context.Input(1) != nullptr) {
    ORT_RETURN_IF_ERROR(ReadOptionalScalar(context.Input(1), "dft_length", length));
    ORT_RETURN_IF(length <= 0, "dft_length must be greater than zero.");
  }
  ORT_RETURN_IF(length < 0 || length > (int64_t{1} << 30), "Invalid DFT length: ", length);

  const int64_t output_components = is_inverse_ && is_onesided_ ? 1 : 2;
  const int64_t output_length = is_onesided_ && !is_inverse_ ? length / 2 + 1 : length;

  TensorShape output_shape = input_shape;
  output_shape[onnxruntime::narrow<size_t>(axis)] = output_length;
  output_shape[onnxruntime::narrow<size_t>(rank - 1)] = output_components;
  auto* output_tensor = context.Output(0, output_shape);
  if (output_shape.Size() == 0) {
    return Status::OK();
  }

  // Transforms are packed between the axis and the innermost (real/imaginary) dimension.
  int64_t inner = 1;
  for (int64_t d = axis + 1; d < rank - 1; ++d) {
    inner *= input_shape[onnxruntime::narrow<size_t>(d)];
  }
  int64_t rows = 1;
  for (int64_t d = 0; d < rank - 1; ++d) {
    if (d != axis) {
      rows *= input_shape[onnxruntime::narrow<size_t>(d)];
    }
  }
  ORT_RETURN_IF_NOT(FitsU32(rows) && FitsU32(inner) && FitsU32(signal_length) && FitsU32(output_length),
                    "DFT shape is too large for the WebGPU execution provider.");

  FftProblem problem;
  problem.layout.hermitian = is_inverse_ && is_onesided_;
  problem.layout.strided_rows = inner > 1;
  problem.layout.inverse = is_inverse_;
  problem.layout.input_components = static_cast<uint32_t>(input_components);
  problem.layout.output_components = static_cast<uint32_t>(output_components);
  problem.layout.dft_length = static_cast<uint32_t>(length);
  problem.rows = static_cast<uint32_t>(rows);
  problem.signal_length = static_cast<uint32_t>(signal_length);
  problem.inner = static_cast<uint32_t>(inner);
  problem.output_length = static_cast<uint32_t>(output_length);
  problem.x = input_tensor;
  problem.y = output_tensor;
  return RunFft(context, problem);
}

Status STFT::ComputeInternal(ComputeContext& context) const {
  const auto* signal = context.Input(0);
  const TensorShape& signal_shape = signal->Shape();
  const size_t rank = signal_shape.NumDimensions();
  ORT_RETURN_IF(rank != 2 && rank != 3, "STFT signal must have shape [batch, length] or [batch, length, 1 or 2].");
  const int64_t components = rank == 3 ? signal_shape[2] : 1;
  ORT_RETURN_IF(components != 1 && components != 2, "STFT signal's innermost dimension must be 1 (real) or 2 (complex).");
  const int64_t batch = signal_shape[0];
  const int64_t signal_length = signal_shape[1];

  int64_t frame_step = 0;
  ORT_RETURN_IF_ERROR(ReadOptionalScalar(context.Input(1), "frame_step", frame_step));
  ORT_RETURN_IF(frame_step <= 0, "frame_step must be greater than zero.");

  const Tensor* window = context.InputCount() > 2 ? context.Input(2) : nullptr;
  const Tensor* frame_length_tensor = context.InputCount() > 3 ? context.Input(3) : nullptr;
  int64_t frame_length = signal_length;
  if (window != nullptr) {
    ORT_RETURN_IF(window->Shape().NumDimensions() != 1, "STFT window must be a 1-D tensor.");
    frame_length = window->Shape()[0];
  }
  if (frame_length_tensor != nullptr) {
    int64_t explicit_length = 0;
    ORT_RETURN_IF_ERROR(ReadOptionalScalar(frame_length_tensor, "frame_length", explicit_length));
    ORT_RETURN_IF(window != nullptr && explicit_length != frame_length,
                  "If both frame_length and window are set, then the size of the window must be equal to the frame_length.");
    frame_length = explicit_length;
  }
  ORT_RETURN_IF(frame_length <= 0 || frame_length > (int64_t{1} << 30), "Invalid STFT frame length: ", frame_length);
  ORT_RETURN_IF(frame_length > signal_length, "Ensure that the dft size is smaller than the signal.");

  const int64_t frames = (signal_length - frame_length) / frame_step + 1;
  const int64_t output_length = is_onesided_ ? frame_length / 2 + 1 : frame_length;
  auto* output = context.Output(0, TensorShape({batch, frames, output_length, 2}));
  if (output->Shape().Size() == 0) {
    return Status::OK();
  }
  ORT_RETURN_IF_NOT(FitsU32(batch * frames) && FitsU32(signal_length) && FitsU32(frame_step),
                    "STFT shape is too large for the WebGPU execution provider.");

  FftProblem problem;
  problem.layout.is_stft = true;
  problem.layout.has_window = window != nullptr;
  problem.layout.input_components = static_cast<uint32_t>(components);
  problem.layout.output_components = 2;
  problem.layout.dft_length = static_cast<uint32_t>(frame_length);
  problem.rows = static_cast<uint32_t>(batch * frames);
  problem.signal_length = static_cast<uint32_t>(signal_length);
  problem.output_length = static_cast<uint32_t>(output_length);
  problem.frame_step = static_cast<uint32_t>(frame_step);
  problem.frames = static_cast<uint32_t>(frames);
  problem.x = signal;
  problem.window = window;
  problem.y = output;
  return RunFft(context, problem);
}

}  // namespace webgpu
}  // namespace onnxruntime

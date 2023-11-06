/**
 * @file epilogue_helpers.h
 *
 * This file includes types for the epilogues. The empty structs exist so we can signal to template
 * code the type of epilogue we want to run, and let the underlying code specify the details such as
 * element types, accumulator type and elements per vector access.
 *
 */

#pragma once

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/scale_type.h"
#include "cutlass/functional.h"
#include "cutlass/half.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_generic.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/epilogue/thread/linear_combination_silu.h"

namespace cutlass {
namespace epilogue {
namespace thread {

__forceinline__ __device__ float copysignf_pos(float a, float b) {
  float r;
  r = __int_as_float(__float_as_int(a) | (__float_as_int(b) & 0x80000000));
  return r;
}

__forceinline__ __device__ float tanh_opt(float x) {
#if (__CUDACC_VER_MAJOR__ < 11) || (__CUDA_ARCH__ < 750)
  const float exp_val = -1.f * fabs(2 * x);
  return copysignf_pos((1.0f - __expf(exp_val)) / (__expf(exp_val) + 1.0f), x);
#else
  return fast_tanh(x);
#endif
}

template <>
struct GELU_taylor<float> {
  static const bool kIsHeavy = true;
  CUTLASS_DEVICE
  float operator()(float const& z) const {
    float k0 = float(0.7978845608028654);
    float k1 = float(0.044715);

    return float(
        cutlass::constants::half<float>() * z * (cutlass::constants::one<float>() + tanh_opt(k0 * z * (cutlass::constants::one<float>() + k1 * z * z))));
  }

  using Params = LinearCombinationGenericParams<float>;

  CUTLASS_DEVICE
  float operator()(float const& scalar, Params const& params_) const {
    return this->operator()(scalar);
  }
};

}  // namespace thread
}  // namespace epilogue
}  // namespace cutlass

namespace fastertransformer {

struct EpilogueOpBiasSilu {};

struct EpilogueOpBiasReLU {};

struct EpilogueOpBiasFtGelu {};

struct EpilogueOpBias {};

struct EpilogueOpNoBias {};

template <typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator, typename Op>
struct Epilogue {
};

template <typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator>
struct Epilogue<ElementType, ElementsPerVectorAccess, ElementAccumulator, EpilogueOpBiasSilu> {
  using Op = cutlass::epilogue::thread::LinearCombinationSilu<ElementType,
                                                              ElementsPerVectorAccess,
                                                              ElementAccumulator,
                                                              ElementAccumulator,
                                                              cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
};

template <typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator>
struct Epilogue<ElementType, ElementsPerVectorAccess, ElementAccumulator, EpilogueOpBiasReLU> {
  using Op = cutlass::epilogue::thread::LinearCombinationRelu<ElementType,
                                                              ElementsPerVectorAccess,
                                                              ElementAccumulator,
                                                              ElementAccumulator,
                                                              cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
};

template <typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator>
struct Epilogue<ElementType, ElementsPerVectorAccess, ElementAccumulator, EpilogueOpBiasFtGelu> {
  using Op = cutlass::epilogue::thread::LinearCombinationGeneric<cutlass::epilogue::thread::GELU_taylor,
                                                                 ElementType,
                                                                 ElementsPerVectorAccess,
                                                                 ElementAccumulator,
                                                                 ElementAccumulator,
                                                                 cutlass::epilogue::thread::ScaleType::NoBetaScaling,
                                                                 cutlass::FloatRoundStyle::round_to_nearest,
                                                                 true>;
};

template <typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator>
struct Epilogue<ElementType, ElementsPerVectorAccess, ElementAccumulator, EpilogueOpBias> {
  using Op = cutlass::epilogue::thread::LinearCombination<ElementType,
                                                          ElementsPerVectorAccess,
                                                          ElementAccumulator,
                                                          ElementAccumulator,
                                                          cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
};

template <typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator>
struct Epilogue<ElementType, ElementsPerVectorAccess, ElementAccumulator, EpilogueOpNoBias> {
  using Op = cutlass::epilogue::thread::LinearCombination<ElementType,
                                                          ElementsPerVectorAccess,
                                                          ElementAccumulator,
                                                          ElementAccumulator,
                                                          cutlass::epilogue::thread::ScaleType::Default>;
};

}  // namespace fastertransformer

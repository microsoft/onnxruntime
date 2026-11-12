/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    conv_activation_kernel_rvv.cpp

Abstract:

    This module implements RVV kernels for the fused activations on riscv64.

    The kinds that are pure element-wise arithmetic are covered here: Identity,
    which only has to apply the bias, Relu, LeakyRelu, Clip and HardSigmoid. Tanh
    and Logistic are not, because MlasActivation applies the bias and then routes
    those two through MlasComputeTanh and MlasComputeLogistic, which have kernels
    of their own.

    Each kind reproduces the arithmetic of its MLAS_ACTIVATION_FUNCTION
    specialization in activate.cpp, so the results are the same. The comparisons
    are a compare plus a merge rather than vfmin/vfmax for the reason given in
    unary_clamp_rvv.h; the bounds here are runtime values, so the clamps cannot
    come from that header.

    The generic kernel splits each row into a four wide vector body and a scalar
    remainder. Both compute the same thing, so there is no separate tail here:
    vsetvl covers the end of the row.

--*/

#include "mlasi.h"

#if defined(MLAS_USE_RVV)

#include <riscv_vector.h>

namespace {

//
// Mirrors MlasMaximumFloat32x4(MlasBroadcastFloat32x4(Bound), Value):
// (Bound > Value) ? Bound : Value, and Value when the compare is unordered.
//
MLAS_FORCEINLINE
vfloat32m4_t
ClampLower(
    vfloat32m4_t Value,
    float Bound,
    size_t vl
    )
{
    return __riscv_vfmerge_vfm_f32m4(Value, Bound,
                                     __riscv_vmflt_vf_f32m4_b8(Value, Bound, vl), vl);
}

//
// Mirrors MlasMinimumFloat32x4(MlasBroadcastFloat32x4(Bound), Value):
// (Value > Bound) ? Bound : Value, and Value when the compare is unordered.
//
MLAS_FORCEINLINE
vfloat32m4_t
ClampUpper(
    vfloat32m4_t Value,
    float Bound,
    size_t vl
    )
{
    return __riscv_vfmerge_vfm_f32m4(Value, Bound,
                                     __riscv_vmfgt_vf_f32m4_b8(Value, Bound, vl), vl);
}

//
// One vector's worth of each activation. The parameters are read once by the
// caller and passed in as scalars, matching the broadcasts the generic
// MLAS_ACTIVATION_FUNCTION constructors set up.
//

struct ActivationIdentityRvv {
    explicit ActivationIdentityRvv(const MLAS_ACTIVATION*) {}

    MLAS_FORCEINLINE vfloat32m4_t Activate(vfloat32m4_t Value, size_t) const
    {
        return Value;
    }
};

struct ActivationReluRvv {
    explicit ActivationReluRvv(const MLAS_ACTIVATION*) {}

    // MlasMaximumFloat32x4(Zero, Value)
    MLAS_FORCEINLINE vfloat32m4_t Activate(vfloat32m4_t Value, size_t vl) const
    {
        return ClampLower(Value, 0.0f, vl);
    }
};

struct ActivationLeakyReluRvv {
    float Alpha;

    explicit ActivationLeakyReluRvv(const MLAS_ACTIVATION* Activation)
        : Alpha(Activation->Parameters.LeakyRelu.alpha)
    {
    }

    // The form this target compiles, MlasBlendFloat32x4(ValueTimesAlpha, Value,
    // Zero < Value), i.e. (0 < Value) ? Value : Value * Alpha. The comparison is
    // the strict one, and being unordered it selects Value * Alpha for a NaN, the
    // way every other implementation of this activation does.
    MLAS_FORCEINLINE vfloat32m4_t Activate(vfloat32m4_t Value, size_t vl) const
    {
        const vfloat32m4_t ValueTimesAlpha = __riscv_vfmul_vf_f32m4(Value, Alpha, vl);
        return __riscv_vmerge_vvm_f32m4(ValueTimesAlpha, Value,
                                        __riscv_vmfgt_vf_f32m4_b8(Value, 0.0f, vl), vl);
    }
};

struct ActivationClipRvv {
    float Minimum;
    float Maximum;

    explicit ActivationClipRvv(const MLAS_ACTIVATION* Activation)
        : Minimum(Activation->Parameters.Clip.minimum),
          Maximum(Activation->Parameters.Clip.maximum)
    {
    }

    MLAS_FORCEINLINE vfloat32m4_t Activate(vfloat32m4_t Value, size_t vl) const
    {
        return ClampUpper(ClampLower(Value, Minimum, vl), Maximum, vl);
    }
};

struct ActivationHardSigmoidRvv {
    float Alpha;
    float Beta;

    explicit ActivationHardSigmoidRvv(const MLAS_ACTIVATION* Activation)
        : Alpha(Activation->Parameters.HardSigmoid.alpha),
          Beta(Activation->Parameters.HardSigmoid.beta)
    {
    }

    // Value * Alpha + Beta, then Minimum(1.0, .), then Maximum(0.0, .), in that
    // order, as the generic specialization does.
    MLAS_FORCEINLINE vfloat32m4_t Activate(vfloat32m4_t Value, size_t vl) const
    {
        Value = __riscv_vfmadd_vf_f32m4(Value, Alpha, __riscv_vfmv_v_f_f32m4(Beta, vl), vl);
        return ClampLower(ClampUpper(Value, 1.0f, vl), 0.0f, vl);
    }
};

//
// Steps over the output matrix the same way MlasActivationKernel does: one row
// at a time, with the bias broadcast from a single element per row.
//
template <typename ActivationType, bool AddBias>
void
ActivationLoopRvv(
    const MLAS_ACTIVATION* Activation,
    float* Buffer,
    const float* Bias,
    size_t M,
    size_t N,
    size_t ldc
    )
{
    const ActivationType ActivationFunction(Activation);

    while (M-- > 0) {
        float* buffer = Buffer;
        size_t n = N;

        float bias = 0.0f;
        if (AddBias) {
            bias = *Bias++;
        }

        while (n > 0) {
            const size_t vl = __riscv_vsetvl_e32m4(n);

            vfloat32m4_t Vector = __riscv_vle32_v_f32m4(buffer, vl);
            if (AddBias) {
                Vector = __riscv_vfadd_vf_f32m4(Vector, bias, vl);
            }
            __riscv_vse32_v_f32m4(buffer, ActivationFunction.Activate(Vector, vl), vl);

            buffer += vl;
            n -= vl;
        }

        Buffer += ldc;
    }
}

template <typename ActivationType>
void
ActivationDispatchBiasRvv(
    const MLAS_ACTIVATION* Activation,
    float* Buffer,
    const float* Bias,
    size_t M,
    size_t N,
    size_t ldc
    )
{
    if (Bias != nullptr) {
        ActivationLoopRvv<ActivationType, true>(Activation, Buffer, Bias, M, N, ldc);
    } else {
        ActivationLoopRvv<ActivationType, false>(Activation, Buffer, Bias, M, N, ldc);
    }
}

}  // namespace

bool
MLASCALL
MlasActivationRvv(
    const MLAS_ACTIVATION* Activation,
    float* Buffer,
    const float* Bias,
    size_t M,
    size_t N,
    size_t ldc
    )
/*++

Routine Description:

    This routine applies the bias and the activation using RVV, for the
    activation kinds that are pure element-wise arithmetic.

Arguments:

    Activation - Supplies the parameters for the activation.

    Buffer - Supplies the output matrix.

    Bias - Supplies the optional bias vector, one element per row.

    M - Supplies the number of rows in the output matrix, and the number of
        elements of the bias vector.

    N - Supplies the number of columns of the output matrix.

    ldc - Supplies the number of elements per row of the output matrix.

Return Value:

    Returns true when this routine handled the activation. Returns false for
    the kinds it does not cover, leaving them to the caller.

--*/
{
    switch (Activation->ActivationKind) {
        case MlasIdentityActivation:
            // Without a bias there is nothing to apply, which is what the
            // generic MlasActivationKernel<MlasIdentityActivation, false>
            // specialization also does.
            if (Bias != nullptr) {
                ActivationLoopRvv<ActivationIdentityRvv, true>(
                    Activation, Buffer, Bias, M, N, ldc);
            }
            return true;

        case MlasReluActivation:
            ActivationDispatchBiasRvv<ActivationReluRvv>(Activation, Buffer, Bias, M, N, ldc);
            return true;

        case MlasLeakyReluActivation:
            ActivationDispatchBiasRvv<ActivationLeakyReluRvv>(Activation, Buffer, Bias, M, N, ldc);
            return true;

        case MlasClipActivation:
            ActivationDispatchBiasRvv<ActivationClipRvv>(Activation, Buffer, Bias, M, N, ldc);
            return true;

        case MlasHardSigmoidActivation:
            ActivationDispatchBiasRvv<ActivationHardSigmoidRvv>(
                Activation, Buffer, Bias, M, N, ldc);
            return true;

        default:
            // Tanh and Logistic apply the bias and then go through
            // MlasComputeTanh/MlasComputeLogistic, which have their own
            // kernels, so they are left to the generic path.
            return false;
    }
}

#endif  // MLAS_USE_RVV

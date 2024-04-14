/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    activate.cpp

Abstract:

    This module implements the fused activation and bias addition routines.

--*/

#include "mlasi.h"

//
// Templates for bias addition functions.
//

template<bool AddBias>
struct MLAS_BIAS_ADDITION;

template<>
struct MLAS_BIAS_ADDITION<true>
{
    MLAS_FLOAT32X4 BiasBroadcast;

    void LoadNext(const float*& Bias)
    {
        BiasBroadcast = MlasBroadcastFloat32x4(Bias++);
    }

    MLAS_FLOAT32X4 Add(MLAS_FLOAT32X4 Value)
    {
        return MlasAddFloat32x4(Value, BiasBroadcast);
    }

    float Add(float Value)
    {
        return Value + MlasExtractLaneFloat32x4<0>(BiasBroadcast);
    }
};

template<>
struct MLAS_BIAS_ADDITION<false>
{
    void LoadNext(const float*& Bias)
    {
        MLAS_UNREFERENCED_PARAMETER(Bias);
    }

    MLAS_FLOAT32X4 Add(MLAS_FLOAT32X4 Value)
    {
        return Value;
    }

    float Add(float Value)
    {
        return Value;
    }
};

//
// Templates for activation functions.
//

template<MLAS_ACTIVATION_KIND ActivationKind>
struct MLAS_ACTIVATION_FUNCTION;

template<>
struct MLAS_ACTIVATION_FUNCTION<MlasIdentityActivation>
{
    MLAS_ACTIVATION_FUNCTION(const MLAS_ACTIVATION* Activation)
    {
        MLAS_UNREFERENCED_PARAMETER(Activation);
    }

    MLAS_FLOAT32X4 Activate(MLAS_FLOAT32X4 Value)
    {
        return Value;
    }

    float Activate(float Value)
    {
        return Value;
    }
};

template<>
struct MLAS_ACTIVATION_FUNCTION<MlasReluActivation>
{
    const MLAS_FLOAT32X4 ZeroFloat32x4 = MlasZeroFloat32x4();

    MLAS_ACTIVATION_FUNCTION(const MLAS_ACTIVATION* Activation)
    {
        MLAS_UNREFERENCED_PARAMETER(Activation);
    }

    MLAS_FLOAT32X4 Activate(MLAS_FLOAT32X4 Value)
    {
        return MlasMaximumFloat32x4(ZeroFloat32x4, Value);
    }

    float Activate(float Value)
    {
#if defined(MLAS_SSE2_INTRINSICS)
        return _mm_cvtss_f32(Activate(_mm_set_ss(Value)));
#else
        return std::max(Value, 0.0f);
#endif
    }
};

template<>
struct MLAS_ACTIVATION_FUNCTION<MlasLeakyReluActivation>
{
    const MLAS_FLOAT32X4 ZeroFloat32x4 = MlasZeroFloat32x4();

    MLAS_FLOAT32X4 AlphaBroadcast;

    MLAS_ACTIVATION_FUNCTION(const MLAS_ACTIVATION* Activation)
    {
        AlphaBroadcast = MlasBroadcastFloat32x4(&Activation->Parameters.LeakyRelu.alpha);
    }

    MLAS_FLOAT32X4 Activate(MLAS_FLOAT32X4 Value)
    {
        MLAS_FLOAT32X4 ValueTimesAlpha = MlasMultiplyFloat32x4(Value, AlphaBroadcast);

#if defined(MLAS_NEON_INTRINSICS)
#if defined(_WIN32)
        return vbslq_f32(vcleq_z_f32_ex(Value), ValueTimesAlpha, Value);
#else
        // N.B. Standard NEON headers lack an intrinsic for the "vcle #0" form.
        return vbslq_f32(vcleq_f32(Value, ZeroFloat32x4), ValueTimesAlpha, Value);
#endif
#elif defined(MLAS_AVX_INTRINSICS)
        return _mm_blendv_ps(ValueTimesAlpha, Value, _mm_cmple_ps(ZeroFloat32x4, Value));
#elif defined(MLAS_SSE2_INTRINSICS)
        return MlasBlendFloat32x4(ValueTimesAlpha, Value, _mm_cmple_ps(ZeroFloat32x4, Value));
#elif defined(MLAS_VSX_INTRINSICS)
        return vec_sel(ValueTimesAlpha, Value, vec_cmple(ZeroFloat32x4, Value));
#elif defined(MLAS_LSX_INTRINSICS)
        return MlasBlendFloat32x4(ValueTimesAlpha, Value, (__m128)__lsx_vfcmp_cle_s(ZeroFloat32x4, Value));
#else
        return MlasBlendFloat32x4(ValueTimesAlpha, Value, ZeroFloat32x4 < Value);
#endif
    }

    float Activate(float Value)
    {
        float ValueTimesAlpha = Value * MlasExtractLaneFloat32x4<0>(AlphaBroadcast);

#if defined(MLAS_SSE2_INTRINSICS)
        return (Value >= MlasExtractLaneFloat32x4<0>(ZeroFloat32x4)) ? Value : ValueTimesAlpha;
#else
        return (Value >= 0.0f) ? Value : ValueTimesAlpha;
#endif
    }
};

template<>
struct MLAS_ACTIVATION_FUNCTION<MlasClipActivation>
{
    MLAS_FLOAT32X4 MinimumBroadcast;
    MLAS_FLOAT32X4 MaximumBroadcast;

    MLAS_ACTIVATION_FUNCTION(const MLAS_ACTIVATION* Activation)
    {
        MinimumBroadcast = MlasBroadcastFloat32x4(&Activation->Parameters.Clip.minimum);
        MaximumBroadcast = MlasBroadcastFloat32x4(&Activation->Parameters.Clip.maximum);
    }

    MLAS_FLOAT32X4 Activate(MLAS_FLOAT32X4 Value)
    {
        Value = MlasMaximumFloat32x4(MinimumBroadcast, Value);
        Value = MlasMinimumFloat32x4(MaximumBroadcast, Value);

        return Value;
    }

    float Activate(float Value)
    {
#if defined(MLAS_SSE2_INTRINSICS)
        return _mm_cvtss_f32(Activate(_mm_set_ss(Value)));
#else
        Value = std::max(Value, MlasExtractLaneFloat32x4<0>(MinimumBroadcast));
        Value = std::min(Value, MlasExtractLaneFloat32x4<0>(MaximumBroadcast));

        return Value;
#endif
    }
};

template<>
struct MLAS_ACTIVATION_FUNCTION<MlasHardSigmoidActivation>
{
    MLAS_FLOAT32X4 AlphaBroadcast;
    MLAS_FLOAT32X4 BetaBroadcast;
    MLAS_FLOAT32X4 MinimumBroadcast;
    MLAS_FLOAT32X4 MaximumBroadcast;

    MLAS_ACTIVATION_FUNCTION(const MLAS_ACTIVATION* Activation)
    {
        AlphaBroadcast = MlasBroadcastFloat32x4(&Activation->Parameters.HardSigmoid.alpha);
        BetaBroadcast = MlasBroadcastFloat32x4(&Activation->Parameters.HardSigmoid.beta);
        MinimumBroadcast = MlasZeroFloat32x4();
        MaximumBroadcast = MlasBroadcastFloat32x4(1.0f);
    }

    MLAS_FLOAT32X4 Activate(MLAS_FLOAT32X4 Value)
    {
        Value = MlasMultiplyAddFloat32x4(Value, AlphaBroadcast, BetaBroadcast);
        Value = MlasMinimumFloat32x4(MaximumBroadcast, Value);
        Value = MlasMaximumFloat32x4(MinimumBroadcast, Value);

        return Value;
    }

    float Activate(float Value)
    {
#if defined(MLAS_SSE2_INTRINSICS)
        return _mm_cvtss_f32(Activate(_mm_set_ss(Value)));
#else
        Value = MlasExtractLaneFloat32x4<0>(AlphaBroadcast) * Value + MlasExtractLaneFloat32x4<0>(BetaBroadcast);
        Value = std::min(Value, MlasExtractLaneFloat32x4<0>(MaximumBroadcast));
        Value = std::max(Value, MlasExtractLaneFloat32x4<0>(MinimumBroadcast));

        return Value;
#endif
    }
};

template<MLAS_ACTIVATION_KIND ActivationKind, bool AddBias>
void
MlasActivationKernel(
    const MLAS_ACTIVATION* Activation,
    float* Buffer,
    const float* Bias,
    size_t M,
    size_t N,
    size_t ldc
    )
/*++

Routine Description:

    This routine steps over the output matrix and invokes the templated bias
    addition and activation functions.

Arguments:

    Activation - Supplies the parameters for the activation.

    Buffer - Supplies the output matrix.

    Bias - Supplies the optional bias vector.

    M - Supplies the number of elements of the bias vector and the number of
        rows in the output matrix.

    N - Supplies the number of columns of the output matrix.

    ldc - Supplies the number of elements per row of the output matrix.

Return Value:

    None.

--*/
{
    MLAS_ACTIVATION_FUNCTION<ActivationKind> ActivationFunction(Activation);
    MLAS_BIAS_ADDITION<AddBias> BiasAddition;

    //
    // Step through each row of the output matrix.
    //

    while (M-- > 0) {

        float* buffer = Buffer;
        size_t n = N;

        BiasAddition.LoadNext(Bias);

        if (n >= 4) {

            do {

                MLAS_FLOAT32X4 Vector = BiasAddition.Add(MlasLoadFloat32x4(buffer));
                MlasStoreFloat32x4(buffer, ActivationFunction.Activate(Vector));
                buffer += 4;
                n -= 4;

            } while (n >= 4);
        }

        while (n > 0) {

            float Scalar = BiasAddition.Add(*buffer);
            *buffer++ = ActivationFunction.Activate(Scalar);
            n -= 1;
        }

        Buffer += ldc;
    }
}

template<>
inline
void
MlasActivationKernel<MlasIdentityActivation, false>(
    const MLAS_ACTIVATION* Activation,
    float* Buffer,
    const float* Bias,
    size_t M,
    size_t N,
    size_t ldc
    )
/*++

Routine Description:

    This routine is invoked for the special case of an identity operation with
    no bias addition, which translates to a no-op.

Arguments:

    Activation - Supplies the parameters for the activation.

    Buffer - Supplies the output matrix.

    Bias - Supplies the optional bias vector.

    M - Supplies the number of elements of the bias vector and the number of
        rows in the output matrix.

    N - Supplies the number of columns of the output matrix.

    ldc - Supplies the number of elements per row of the output matrix.

Return Value:

    None.

--*/
{
    //
    // No operation.
    //

    MLAS_UNREFERENCED_PARAMETER(Activation);
    MLAS_UNREFERENCED_PARAMETER(Buffer);
    MLAS_UNREFERENCED_PARAMETER(Bias);
    MLAS_UNREFERENCED_PARAMETER(M);
    MLAS_UNREFERENCED_PARAMETER(N);
    MLAS_UNREFERENCED_PARAMETER(ldc);
}

template<MLAS_ACTIVATION_KIND ActivationKind>
inline
void
MlasActivationKernel(
    const MLAS_ACTIVATION* Activation,
    float* Buffer,
    const float* Bias,
    size_t M,
    size_t N,
    size_t ldc
    )
/*++

Routine Description:

    This routine invokes the appropriate activation kernel based on the
    optional bias vector.

Arguments:

    Activation - Supplies the parameters for the activation.

    Buffer - Supplies the output matrix.

    Bias - Supplies the optional bias vector.

    M - Supplies the number of elements of the bias vector and the number of
        rows in the output matrix.

    N - Supplies the number of columns of the output matrix.

    ldc - Supplies the number of elements per row of the output matrix.

Return Value:

    None.

--*/
{
    if (Bias != nullptr) {
        MlasActivationKernel<ActivationKind, true>(Activation, Buffer, Bias, M, N, ldc);
    } else {
        MlasActivationKernel<ActivationKind, false>(Activation, Buffer, Bias, M, N, ldc);
    }
}

void
MLASCALL
MlasActivation(
    const MLAS_ACTIVATION* Activation,
    float* Buffer,
    const float* Bias,
    size_t M,
    size_t N,
    size_t ldc
    )
/*++

Routine Description:

    This routine applies an activation function to the output matrix after
    optionally adding a bias vector.

Arguments:

    Activation - Supplies the parameters for the activation.

    Buffer - Supplies the output matrix.

    Bias - Supplies the optional bias vector.

    M - Supplies the number of elements of the bias vector and the number of
        rows in the output matrix.

    N - Supplies the number of columns of the output matrix.

    ldc - Supplies the number of elements per row of the output matrix.

Return Value:

    None.

--*/
{
    switch (Activation->ActivationKind) {

        case MlasIdentityActivation:
        {
            MlasActivationKernel<MlasIdentityActivation>(Activation, Buffer, Bias, M, N, ldc);
            break;
        }

        case MlasReluActivation:
        {
            MlasActivationKernel<MlasReluActivation>(Activation, Buffer, Bias, M, N, ldc);
            break;
        }

        case MlasLeakyReluActivation:
        {
            MlasActivationKernel<MlasLeakyReluActivation>(Activation, Buffer, Bias, M, N, ldc);
            break;
        }

        case MlasTanhActivation:
        {
            if (Bias != nullptr) {
                MlasActivationKernel<MlasIdentityActivation, true>(Activation, Buffer, Bias, M, N, ldc);
            }

            if (N == ldc) {
                MlasComputeTanh(Buffer, Buffer, M * N);
            } else {
                while (M-- > 0) {
                    MlasComputeTanh(Buffer, Buffer, N);
                    Buffer += ldc;
                }
            }

            break;
        }

        case MlasLogisticActivation:
        {
            if (Bias != nullptr) {
                MlasActivationKernel<MlasIdentityActivation, true>(Activation, Buffer, Bias, M, N, ldc);
            }

            if (N == ldc) {
                MlasComputeLogistic(Buffer, Buffer, M * N);
            } else {
                while (M-- > 0) {
                    MlasComputeLogistic(Buffer, Buffer, N);
                    Buffer += ldc;
                }
            }

            break;
        }

        case MlasClipActivation:
        {
            MlasActivationKernel<MlasClipActivation>(Activation, Buffer, Bias, M, N, ldc);
            break;
        }

        case MlasHardSigmoidActivation:
        {
            MlasActivationKernel<MlasHardSigmoidActivation>(Activation, Buffer, Bias, M, N, ldc);
            break;
        }

        case MlasActivationKindCount:
        {
            MLAS_THROW_EX(std::runtime_error, "bad mlas activation kind");
            break;
        }
    }
}

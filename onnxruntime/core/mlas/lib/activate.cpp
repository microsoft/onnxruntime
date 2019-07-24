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
        return (std::max)(0.0f, Value);
    }
};

template<>
struct MLAS_ACTIVATION_FUNCTION<MlasLeakyReluActivation>
{
    const MLAS_FLOAT32X4 ZeroFloat32x4 = MlasZeroFloat32x4();

    MLAS_FLOAT32X4 AlphaBroadcast;

    MLAS_ACTIVATION_FUNCTION(const MLAS_ACTIVATION* Activation)
    {
        AlphaBroadcast = MlasBroadcastFloat32x4(&Activation->alpha);
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
        __m128 Selection = _mm_cmple_ps(ZeroFloat32x4, Value);
        return _mm_or_ps(_mm_and_ps(Value, Selection), _mm_andnot_ps(Selection, ValueTimesAlpha));
#else
#error Unsupported architecture.
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

template<MLAS_ACTIVATION_KIND ActivationKind, bool AddBias>
void
MlasActivationKernel(
    const MLAS_ACTIVATION* Activation,
    const float* Input,
    const float* Bias,
    size_t M,
    float* Output,
    size_t N,
    size_t ldc
    )
/*++

Routine Description:

    This routine steps over the output matrix and invokes the templated bias
    addition and activation functions.

Arguments:

    Activation - Supplies the parameters for the activation.

    Input - Supplies the input matrix.

    Bias - Supplies the optional bias vector.

    M - Supplies the number of elements of the bias vector and the number of
        rows in the output matrix.

    Output - Supplies the output matrix.

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

        const float* input = Input;
        float* output = Output;
        size_t n = N;

        BiasAddition.LoadNext(Bias);

        if (n >= 4) {

            do {

                MLAS_FLOAT32X4 Vector = BiasAddition.Add(MlasLoadFloat32x4(input));
                MlasStoreFloat32x4(output, ActivationFunction.Activate(Vector));
                input += 4;
                output += 4;
                n -= 4;

            } while (n >= 4);
        }

        while (n > 0) {

            float Scalar = BiasAddition.Add(*input++);
            *output++ = ActivationFunction.Activate(Scalar);
            n -= 1;
        }

        Input += ldc;
        Output += ldc;
    }
}

template<>
inline
void
MlasActivationKernel<MlasIdentityActivation, false>(
    const MLAS_ACTIVATION* Activation,
    const float* Input,
    const float* Bias,
    size_t M,
    float* Output,
    size_t N,
    size_t ldc
    )
/*++

Routine Description:

    This routine is invoked for the special case of an identity operation with
    no bias addition, which translates to a no-op.

Arguments:

    Activation - Supplies the parameters for the activation.

    Input - Supplies the input matrix.

    Bias - Supplies the optional bias vector.

    M - Supplies the number of elements of the bias vector and the number of
        rows in the output matrix.

    Output - Supplies the output matrix.

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
    MLAS_UNREFERENCED_PARAMETER(Input);
    MLAS_UNREFERENCED_PARAMETER(Bias);
    MLAS_UNREFERENCED_PARAMETER(M);
    MLAS_UNREFERENCED_PARAMETER(Output);
    MLAS_UNREFERENCED_PARAMETER(N);
    MLAS_UNREFERENCED_PARAMETER(ldc);
}

template<MLAS_ACTIVATION_KIND ActivationKind>
inline
void
MlasActivationKernel(
    const MLAS_ACTIVATION* Activation,
    const float* Input,
    const float* Bias,
    size_t M,
    float* Output,
    size_t N,
    size_t ldc
    )
/*++

Routine Description:

    This routine invokes the appropriate activation kernel based on the
    optional bias vector.

Arguments:

    Activation - Supplies the parameters for the activation.

    Input - Supplies the input matrix.

    Bias - Supplies the optional bias vector.

    M - Supplies the number of elements of the bias vector and the number of
        rows in the output matrix.

    Output - Supplies the output matrix.

    N - Supplies the number of columns of the output matrix.

    ldc - Supplies the number of elements per row of the output matrix.

Return Value:

    None.

--*/
{
    if (Bias != nullptr) {
        MlasActivationKernel<ActivationKind, true>(Activation, Input, Bias, M, Output, N, ldc);
    } else {
        MlasActivationKernel<ActivationKind, false>(Activation, Input, Bias, M, Output, N, ldc);
    }
}

void
MLASCALL
MlasActivation(
    const MLAS_ACTIVATION* Activation,
    const float* Input,
    const float* Bias,
    size_t M,
    float* Output,
    size_t N,
    size_t ldc
    )
/*++

Routine Description:

    This routine applies an activation function to the output matrix after
    optionally adding a bias vector.

Arguments:

    Activation - Supplies the parameters for the activation.

    Input - Supplies the input matrix.

    Bias - Supplies the optional bias vector.

    M - Supplies the number of elements of the bias vector and the number of
        rows in the output matrix.

    Output - Supplies the output matrix.

    N - Supplies the number of columns of the output matrix.

    ldc - Supplies the number of elements per row of the output matrix.

Return Value:

    None.

--*/
{
    switch (Activation->ActivationKind) {

        case MlasIdentityActivation:
        {
            MlasActivationKernel<MlasIdentityActivation>(Activation, Input, Bias, M, Output, N, ldc);
            break;
        }

        case MlasReluActivation:
        {
            MlasActivationKernel<MlasReluActivation>(Activation, Input, Bias, M, Output, N, ldc);
            break;
        }

        case MlasLeakyReluActivation:
        {
            MlasActivationKernel<MlasLeakyReluActivation>(Activation, Input, Bias, M, Output, N, ldc);
            break;
        }

        case MlasTanhActivation:
        {
            if (Bias != nullptr) {
                MlasActivationKernel<MlasIdentityActivation, true>(Activation, Input, Bias, M, Output, N, ldc);
            }

            if (N == ldc) {
                MlasComputeTanh(Output, Output, M * N);
            } else {
                while (M-- > 0) {
                    MlasComputeTanh(Output, Output, N);
                    Output += ldc;
                }
            }

            break;
        }

        case MlasLogisticActivation:
        {
            if (Bias != nullptr) {
                MlasActivationKernel<MlasIdentityActivation, true>(Activation, Input, Bias, M, Output, N, ldc);
            }

            if (N == ldc) {
                MlasComputeLogistic(Output, Output, M * N);
            } else {
                while (M-- > 0) {
                    MlasComputeLogistic(Output, Output, N);
                    Output += ldc;
                }
            }

            break;
        }
    }
}

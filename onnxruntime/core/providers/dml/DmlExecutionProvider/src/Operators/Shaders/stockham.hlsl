// TBUFFER is the data type to read from src and write to dst.
// Arithmetic is always done in FP32.
#if !defined(TBUFFER)
#define TBUFFER float
#endif

RWStructuredBuffer<TBUFFER> src : register(u0);
RWStructuredBuffer<TBUFFER> dst : register(u1);
RWStructuredBuffer<TBUFFER> window : register(u2);

cbuffer Constants
{
    uint StartIndex;
    uint ElementCount;
    uint DFTIteration;
    uint IsInverse;
    uint4 InputSizes;
    uint4 InputStrides;
    uint4 OutputSizes;
    uint4 OutputStrides;
    uint4 WindowSizes; // [1, 1, DFTLength, 1 or 2]
    uint4 WindowStrides;
    uint HasWindow;
    float ChirpLength;
    float Scale;
    uint DFTLength;
};

// Returns the indices for the real and complex output uav
uint2 ComputeDestIndex(uint index)
{
    uint2 dftOutputIndex = uint2(index * OutputStrides[2], 0);
    dftOutputIndex.y = dftOutputIndex.x + OutputStrides[3];
    return dftOutputIndex;
}

// The returned value is float2, corresponding to the complex number at the index
float2 ReadSourceValue(uint3 index)
{
    float2 window_value = float2(1, 0);
    float2 value = float2(0, 0);

    bool hasWindow = HasWindow == 1;
    [flatten]
    if (hasWindow && index.y < (uint)WindowSizes[2])
    {
        uint windowIndexReal = index.y * WindowStrides[2];
        window_value.x = window[windowIndexReal];

        uint windowIndexImaginary = windowIndexReal + WindowStrides[3];
        [branch]
        if (WindowSizes[3] == 2)
        {
            window_value.y = window[windowIndexImaginary];
        }
    }

    [flatten]
    if (index.y < (uint)InputSizes[1])
    {
        uint indexReal =
            index.x * InputStrides[0] +
            index.y * InputStrides[1] +
            index.z * InputStrides[2];
        value.x = src[indexReal];

        // If real valued, value.y is defaulted to 0
        // If complex valued input, assign the complex part to non-zero...
        [branch]
        if (InputSizes[3] == 2) {
            uint indexImaginary = indexReal + InputStrides[3];
            value.y = src[indexImaginary];
        }
    }

    float2 weighted_value = float2(value.x * window_value.x - value.y * window_value.y,
                                   value.x * window_value.y + value.y * window_value.x);
    return weighted_value;
}

uint3 DecomposeIndex(uint index)
{
    uint temp = index % (OutputSizes[1] * OutputSizes[2]);

    uint3 idx = uint3(0, 0, 0);
    idx.x = index / (OutputSizes[1] * OutputSizes[2]);
    idx.y = temp / OutputSizes[2]; // This corresponds to the s1'th element of the dft
    idx.z = temp % OutputSizes[2];
    return idx;
}

float2 CalculateChirp(uint n, float N)
{
    if (N == 0)
    {
        return float2(1, 0);
    }

    static const float PI = 3.14159265f;
    // chirp[n] = e^(i * direction * pi * n * n / N)
    // the direction is encoded into the N!
    float theta = PI * n * n / N;
    return float2(cos(theta), sin(theta));
}

[numthreads(64, 1, 1)]
void DFT(uint3 dtid : SV_DispatchThreadId)
{
    uint index = StartIndex + dtid.x;
    if (index < ElementCount)
    {
        uint halfTotalDFTLength = DFTLength >> 1;
        uint N = 1U << DFTIteration;
        uint halfN = 1U << (DFTIteration - 1);

        // Get input even and odd indices
        // Decompose the current index into its location in the packed tensor
        uint2 inputEvenOddIndexPair = uint2(0, 0);
        uint3 idx = DecomposeIndex(index);
        inputEvenOddIndexPair.x = (idx.y >> DFTIteration) * halfN + (idx.y % halfN);
        inputEvenOddIndexPair.y = inputEvenOddIndexPair.x + halfTotalDFTLength;

        // Create full index for even and odd values
        uint3 inputEvenIdx = uint3(idx.x, inputEvenOddIndexPair.x, idx.z);
        uint3 inputOddIdx = uint3(idx.x, inputEvenOddIndexPair.y, idx.z);

        // Read input even and odd values
        float2 inputEvenValue = ReadSourceValue(inputEvenIdx);
        float2 inputOddValue = ReadSourceValue(inputOddIdx);

        // Create coefficient
        // w(k, N) = e^(i*2*pi * k / N)
        uint k = idx.y % N;
        static const float PI = 3.14159265f;
        static const float TAU = PI * 2;
        bool isInverse = IsInverse == 1;
        const float inverseMultiplier = isInverse ? 1.f : -1.f;
        float theta = inverseMultiplier * TAU * (float)k / (float)N;
        float2 w = float2(cos(theta), sin(theta));

        uint2 outputIndex = ComputeDestIndex(index);
        float2 unweighted;
        unweighted.x = Scale * (inputEvenValue.x + (w.x * inputOddValue.x - w.y * inputOddValue.y));
        unweighted.y = Scale * (inputEvenValue.y + (w.x * inputOddValue.y + w.y * inputOddValue.x));

        // When ChirpLength is 0, then chirp should evaluate to (1,0), which is a no-op.
        [branch]
        if (ChirpLength == 0)
        {
            dst[outputIndex.x] = (TBUFFER)(unweighted.x);
            dst[outputIndex.y] = (TBUFFER)(unweighted.y);
        }
        else {
            float2 chirp = CalculateChirp(k, ChirpLength);
            dst[outputIndex.x] = (TBUFFER)(unweighted.x * chirp.x - unweighted.y * chirp.y);
            dst[outputIndex.y] = (TBUFFER)(unweighted.x * chirp.y + unweighted.y * chirp.x);
        }
    }
}

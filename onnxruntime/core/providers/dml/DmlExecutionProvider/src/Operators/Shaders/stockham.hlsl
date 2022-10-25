RWStructuredBuffer<float> src : register(u0);
RWStructuredBuffer<float> dst : register(u1);

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
    float2 value = float2(0, 0);

    uint indexReal =
        index.x * InputStrides[0] +
        index.y * InputStrides[1] +
        index.z * InputStrides[2];
    value.x = src[indexReal];

    // If real valued, value.y is defaulted to 0
    // If complex valued input, assign the complex part to non-zero...
    if (InputSizes[3] == 2) {
        uint indexImaginary = indexReal + InputStrides[3];
        value.y = src[indexImaginary];
    }

    return value;
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

[numthreads(64, 1, 1)]
void DFT(uint3 dtid : SV_DispatchThreadId)
{
    uint index = StartIndex + dtid.x;
    if (index < ElementCount)
    {
        uint halfTotalDFTLength = DFTLength / 2;
        uint N = 1 << DFTIteration;
        uint halfN = 1 << (DFTIteration - 1);

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
        dst[outputIndex.x] = Scale * (inputEvenValue.x + (w.x * inputOddValue.x - w.y * inputOddValue.y));
        dst[outputIndex.y] = Scale * (inputEvenValue.y + (w.x * inputOddValue.y + w.y * inputOddValue.x));
    }
}

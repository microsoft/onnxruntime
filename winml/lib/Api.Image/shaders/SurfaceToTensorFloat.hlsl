//
// Copyright (c) Microsoft Corporation. All rights reserved.
//
// This shader converts a DX texture (BGRA/BGRX/RGBA/GRAY) into NCHW FLOAT Tensor with channel order RGB/BGR/GRAY
//

Texture2D<float4> input : register(t0); // SRV

#ifdef FP16
RWBuffer<float> output : register(u0); // UAV
#else
RWStructuredBuffer<float> output : register(u0); // UAV
#endif

cbuffer cbCS : register(b0)
{
    uint height;
    uint width;
};

[numthreads(16, 4, 1)]
void SurfaceToTensorBGR8(uint3 globalThreadId : SV_DispatchThreadId)
{
    if (globalThreadId.x < width && globalThreadId.y < height)
    {
        float4 inputPixel = input.Load(globalThreadId);
        inputPixel = clamp(inputPixel * 255, 0, 255);

        // Calculate the size of a single plan of color. 
        uint planeSize = width * height;
        uint threadOffset = width * globalThreadId.y + globalThreadId.x;

        output[threadOffset] = inputPixel.b;
        output[threadOffset + planeSize] = inputPixel.g;
        output[threadOffset + planeSize * 2] = inputPixel.r;
    }
}

[numthreads(16, 4, 1)]
void SurfaceToTensorGRAY8(uint3 globalThreadId : SV_DispatchThreadId)
{
    if (globalThreadId.x < width && globalThreadId.y < height)
    {
        float4 inputPixel = input.Load(globalThreadId);
        inputPixel = clamp(inputPixel * 255, 0, 255);

        // Calculate the size of a single plan of color. 
        uint planeSize = width * height;
        uint threadOffset = width * globalThreadId.y + globalThreadId.x;

        output[threadOffset] = inputPixel.r;
        output[threadOffset + planeSize] = inputPixel.g;
        output[threadOffset + planeSize * 2] = inputPixel.b;
    }
}

[numthreads(16, 4, 1)]
void SurfaceToTensorRGB8(uint3 globalThreadId : SV_DispatchThreadId)
{
    if (globalThreadId.x < width && globalThreadId.y < height)
    {
        float4 inputPixel = input.Load(globalThreadId);
        inputPixel = clamp(inputPixel * 255, 0, 255);

        float grayValue = 0.2126 * inputPixel.r + 0.7152 * inputPixel.g + 0.0722 * inputPixel.b;
        uint threadOffset = width * globalThreadId.y + globalThreadId.x;

        output[threadOffset] = grayValue;
    }
}

[numthreads(16, 4, 1)]
void SurfaceGRAY8ToTensorBGR8(uint3 globalThreadId : SV_DispatchThreadId)
{
    if (globalThreadId.x < width && globalThreadId.y < height)
    {
        float4 inputPixel = input.Load(globalThreadId);
        float gray = clamp(inputPixel.r * 255, 0, 255);

        // Calculate the size of a single plan of color. 
        uint planeSize = width * height;
        uint threadOffset = width * globalThreadId.y + globalThreadId.x;
        output[threadOffset] = gray;
        output[threadOffset + planeSize] = gray;
        output[threadOffset + planeSize * 2] = gray;
    }
}

[numthreads(16, 4, 1)]
void SurfaceGRAY8ToTensorGRAY8(uint3 globalThreadId : SV_DispatchThreadId)
{
    if (globalThreadId.x < width && globalThreadId.y < height)
    {
        float4 inputPixel = input.Load(globalThreadId);
        float gray = clamp(inputPixel.r * 255, 0, 255);

        uint threadOffset = width * globalThreadId.y + globalThreadId.x;
        output[threadOffset] = gray;
    }
}
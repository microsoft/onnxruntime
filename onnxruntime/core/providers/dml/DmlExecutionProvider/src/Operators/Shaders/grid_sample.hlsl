// TBUFFER is the data type to read from src and write to dst.
// Arithmetic is always done in FP32.
#if !defined(TBUFFER1)
#define TBUFFER1 float
#endif
#if !defined(TBUFFER2)
#define TBUFFER2 float
#endif

RWStructuredBuffer<TBUFFER1> input : register(u0);
RWStructuredBuffer<TBUFFER2> grid : register(u1);
RWStructuredBuffer<TBUFFER1> output : register(u2);

static const uint Zeros = 0;
static const uint Border = 1;
static const uint Reflection = 2;

static const uint Bilinear = 0;
static const uint Nearest = 1;
static const uint Bicubic = 2;

cbuffer Constants
{
    uint StartIndex;
    uint ElementCount;
    uint Mode;
    uint PaddingMode;
    uint4 InputSizes;
    uint4 InputStrides;
    uint4 GridSizes;
    uint4 GridStrides;
    uint4 OutputSizes;
    uint4 OutputStrides;
    uint AlignCorners;
};

uint4 DecomposeIndex(uint index)
{
    uint4 idx = uint4(0, 0, 0, 0);

    uint n = OutputSizes.x;
    uint c = OutputSizes.y;
    uint h = OutputSizes.z;
    uint w = OutputSizes.w;

    uint width_denominator   = 1;
    uint height_denominator  = w;
    uint channel_denominator = w * h;
    uint batch_denominator   = w * h * c;

    idx.x = (index)                                                                                     / batch_denominator; // batch
    idx.y = (index - (idx.x * OutputStrides.x))                                                         / channel_denominator; // channel
    idx.z = (index - (idx.x * OutputStrides.x) - (idx.y * OutputStrides.y))                             / height_denominator; // height
    idx.w = (index - (idx.x * OutputStrides.x) - (idx.y * OutputStrides.y) - (idx.z * OutputStrides.z)) / width_denominator; // width
    return idx;
}
// Returns the indices for the real and complex output uav
float2 FetchGridVector(uint4 index)
{
    // The index is in (n,c,h,w)
    // The shape of gridsizes (and gridstrides) is (n,   h, w, 2)
    uint n = index.x;
    uint c = index.y;
    uint h = index.z;
    uint w = index.w;

    float4 gridIdx = float4(n, h, w, 0);
    float2 flattenedGridIndex = float2(0, 0);
    flattenedGridIndex.x = dot(gridIdx, GridStrides);
    flattenedGridIndex.y = flattenedGridIndex.x + GridStrides.w;
    return float2((float)grid[flattenedGridIndex.x],
                  (float)grid[flattenedGridIndex.y]);
}

// {x_min, y_min, x_max, y_max};
float4 CalculateBorders()
{
  // Force float here to avoid possible issue in integer T case
  float2 mins = float2(-0.5f, -0.5f);
  float2 maxes = float2(InputSizes.w - 0.5f, // W_in
                 InputSizes.z - 0.5f); // H_in

  if (AlignCorners) {
    mins = float2(0.f, 0.f);
    maxes = float2(InputSizes.w - 1.f, // W_in
                    InputSizes.z - 1.f); // H_in
  }
  return float4(mins.xy, maxes.xy);
}


// Reflect by the near border till within the borders
// Use float for borders to avoid potential issues with integer T
float Reflect(float x, float x_min, float x_max) {
  float range = x_max - x_min;
  if (x < x_min) {
    float dx = x_min - x;
    uint n = dx / range;
    float r = dx - n * range;
    if (n % 2 == 0) {
      x = x_min + r;
    } else {
      x = x_max - r;
    }
  } else if (x > x_max) {
    float dx = x - x_max;
    uint n = dx / range;
    float r = dx - n * range;
    if (n % 2 == 0) {
      x = x_max - r;
    } else {
      x = x_min + r;
    }
  }
  // else fallthrough
  return x;
}


// Restore normalized location to actual image location
//   When align_corners is true:
//     Normalized location (-1, -1) points to the top-left pixel.
//     Normalized location (1, 1) points to the bottom-right pixel.
//   When align_corners is false [default]:
//     Normalized location (-1, -1) points to the top-left pixel minus half
//     pixel in both directions, i.e, (-0.5, -0.5) in actual image space.
//     Normalized location (1, 1) points to the bottom-right pixel plus half
//     pixel in both directions, i.e. (H - 0.5, W - 0.5) in actual image space.
float2 DenormalizeInput(float2 n, float4 border)
{
  float2 dims = InputSizes.wz; // w-h
  if (AlignCorners == 1)
  {
    // AlignCorners: true => [-1, 1] to [0, dims - 1]
    n = (n + 1) / 2.f * (dims - 1);
  }
  else
  {
    // AlignCorners: false => [-1, 1] to [-0.5, dims - 0.5]
    n =  ((n + 1) * dims - 1) / 2.f;
  }

  if (Mode == Nearest)
  {
    n = round(n);
  }

  float x_min = border.x;
  float y_min = border.y;
  float x_max = border.z;
  float y_max = border.w;

  if (n.x < x_min || n.x > x_max || n.y < y_min || n.y > y_max) {  // out of bound
    if (PaddingMode == Border) {
        // use original border in both align_corner cases
        n.x = clamp(n.x, 0, InputSizes.w - 1);
        n.y = clamp(n.y, 0, InputSizes.z - 1);
    } else if (PaddingMode == Reflection) {
        n.x = Reflect(n.x, x_min, x_max);
        n.y = Reflect(n.y, y_min, y_max);
    }
  }  // out of bound

  return n;
}

float FetchInputPixel(uint4 index)
{
    // index and InputStrides is in (n,c, h, w)
    return (float)input[dot(index, InputStrides)];
}

float PixelAtGrid(float4 inputIdx, float4 border) {
  float pixel = 0;  // default 0

  if (PaddingMode == Zeros)
  {
    if (inputIdx.w >= 0 && (uint)inputIdx.w < (uint)InputSizes.w &&
        inputIdx.z >= 0 && (uint)inputIdx.z < (uint)InputSizes.z)
    {
      pixel = FetchInputPixel(inputIdx);
    }
  }
  else if (PaddingMode == Border)
  {
    uint w = clamp(inputIdx.w, 0, InputSizes.w - 1);
    uint z = clamp(inputIdx.z, 0, InputSizes.z - 1);
    pixel = FetchInputPixel(float4(inputIdx.xy, z, w));
  }
  else if (PaddingMode == Reflection)
  {
    uint w = Reflect(inputIdx.w, border.x, border.z);
    uint z = Reflect(inputIdx.z, border.y, border.w);
    pixel = FetchInputPixel(float4(inputIdx.xy, z, w));
  }

  return pixel;
}

float BicubicConvolutionPFunction(float t, float fminus1, float f0, float f1, float f2)
{
    static const float a = -.75;
    static const float4x4 bicubicConvolutionMatrix =
    {
         0,    1,     0,   0,
         a,    0,    -a,   0,
      -2*a, -3-a, 3+2*a,   a,
         a,  2+a,  -2-a,  -a
    };

    float4 t_vec = float4(      1,  t, t * t, t * t * t);
    float4 f_vec = float4(fminus1, f0,    f1,        f2);
    return mul(t_vec, mul(bicubicConvolutionMatrix, f_vec));
}

[numthreads(64, 1, 1)]
void GridSample(uint3 dtid : SV_DispatchThreadId)
{
    uint n = StartIndex + dtid.x;
    if (n < ElementCount)
    {
        float4 border = CalculateBorders();
        uint4 index = DecomposeIndex(n);
        float2 flowVector = FetchGridVector(index);
        float2 inputWidthAndHeightIdx = DenormalizeInput(flowVector, border);
        float4 inputIdx = float4(index.x,                   // N
                                 index.y,                   // C
                                 inputWidthAndHeightIdx.y,  // H
                                 inputWidthAndHeightIdx.x); // W

        if (Mode == Nearest)
        {
            output[n] = (TBUFFER1)(PixelAtGrid(inputIdx, border));
        }
        else if (Mode == Bilinear)
        {
            float x1 = floor(inputIdx.w);
            float y1 = floor(inputIdx.z);
            float x2 = x1 + 1;
            float y2 = y1 + 1;

            float p11 = PixelAtGrid(float4(index.x, index.y, y1, x1), border);
            float p12 = PixelAtGrid(float4(index.x, index.y, y1, x2), border);
            float p21 = PixelAtGrid(float4(index.x, index.y, y2, x1), border);
            float p22 = PixelAtGrid(float4(index.x, index.y, y2, x2), border);

            // p11--p12
            //  |    |
            // p21--p22
            float p1 = lerp(p11, p12, frac(inputIdx.w));
            float p2 = lerp(p21, p22, frac(inputIdx.w));
            float p = lerp(p1, p2, frac(inputIdx.z));
            output[n] = (TBUFFER1)(p);
        }
        else if (Mode == Bicubic)
        {
            float x0 = floor(inputIdx.w) - 1;
            float y0 = floor(inputIdx.z) - 1;

            float f00 = PixelAtGrid(float4(index.x, index.y, y0 + 0, x0 + 0), border);
            float f01 = PixelAtGrid(float4(index.x, index.y, y0 + 0, x0 + 1), border);
            float f02 = PixelAtGrid(float4(index.x, index.y, y0 + 0, x0 + 2), border);
            float f03 = PixelAtGrid(float4(index.x, index.y, y0 + 0, x0 + 3), border);
            float f10 = PixelAtGrid(float4(index.x, index.y, y0 + 1, x0 + 0), border);
            float f11 = PixelAtGrid(float4(index.x, index.y, y0 + 1, x0 + 1), border);
            float f12 = PixelAtGrid(float4(index.x, index.y, y0 + 1, x0 + 2), border);
            float f13 = PixelAtGrid(float4(index.x, index.y, y0 + 1, x0 + 3), border);
            float f20 = PixelAtGrid(float4(index.x, index.y, y0 + 2, x0 + 0), border);
            float f21 = PixelAtGrid(float4(index.x, index.y, y0 + 2, x0 + 1), border);
            float f22 = PixelAtGrid(float4(index.x, index.y, y0 + 2, x0 + 2), border);
            float f23 = PixelAtGrid(float4(index.x, index.y, y0 + 2, x0 + 3), border);
            float f30 = PixelAtGrid(float4(index.x, index.y, y0 + 3, x0 + 0), border);
            float f31 = PixelAtGrid(float4(index.x, index.y, y0 + 3, x0 + 1), border);
            float f32 = PixelAtGrid(float4(index.x, index.y, y0 + 3, x0 + 2), border);
            float f33 = PixelAtGrid(float4(index.x, index.y, y0 + 3, x0 + 3), border);

            float tx = frac(inputIdx.w);
            float ty = frac(inputIdx.z);

            float bminus1 = BicubicConvolutionPFunction(ty, f00, f10, f20, f30);
            float b0      = BicubicConvolutionPFunction(ty, f01, f11, f21, f31);
            float b1      = BicubicConvolutionPFunction(ty, f02, f12, f22, f32);
            float b2      = BicubicConvolutionPFunction(ty, f03, f13, f23, f33);
            float p       = BicubicConvolutionPFunction(tx, bminus1, b0, b1, b2);

            output[n] = (TBUFFER1)(p);
        }
    }
}

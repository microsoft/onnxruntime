// FIXME: LICENSE NOTICE:
// adapted from TNN original BSD3.

// launch a grid with total [X = RoundToMultiple(width, k), Y = height] threads,
// height is not rounded up because threads only continuous in x dimension.
__kernel void CopyBufferNCHWToImage2D(
    const int width, const int height,  // image, width = CeilDiv(C,4)*W*4, height = N*H
    __global const float* data,
    /*const int N,*/ const int C, const int H, const int W,
    __write_only image2d_t output) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  if (x < width && y < height) {
    const int n = y / height;
    const int h = y % height;
    const int w = x % width;
    const int c = (x / width) * 4;  // every thread handle 4 elements, gather read, vector write

    // indexing into the NCHW data
    const int HW = H * W;
    const int base_index = (C * n + c) * HW + W * h + w;

    // channel is not consecutive in memory
    const int remain_channel = C - c;
    int i = base_index;
    float4 v = 0;
    if (remain_channel >= 4) {
      v.x = data[i];
      i += HW;
      v.y = data[i];
      i += HW;
      v.z = data[i];
      i += HW;
      v.w = data[i];
    } else if (remain_channel == 3) {
      v.x = data[i];
      i += HW;
      v.y = data[i];
      i += HW;
      v.z = data[i];
    } else if (remain_channel == 2) {
      v.x = data[i];
      i += HW;
      v.y = data[i];
    } else if (remain_channel == 1) {
      v.x = data[i];
    }

    write_imagef(output, (int2)(x, y), v);
  }
}

// launch a grid with total [X = RoundToMultiple(width, k), Y = height] threads
__kernel void CopyImage2DToBufferNCHW(
    int width, int height,  // width = N*H, height = CeilDiv(C,4)*W*4
    __read_only image2d_t data,
    __global float* output,
    int N, int C, int H, int W) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  if (x < width && y < height) {
    const int n = y / H;
    const int h = y % H;
    const int w = x % width;
    const int c = (x / width) * 4; // every thread handle 4 elements, vector read, scatter write

    // indexing into the NCHW data
    const int HW = H * W;
    const int base_index = (C * n + c) * HW + W * h + w;

    const float4 v = read_imagef(data, (int2)(x, y));
    const int remain_channel = C - c;
    int i = base_index;
    if (remain_channel >= 4) {
      output[i] = v.x;
      i += HW;
      output[i] = v.y;
      i += HW;
      output[i] = v.z;
      i += HW;
      output[i] = v.w;
    } else if (remain_channel == 3) {
      output[i] = v.x;
      i += HW;
      output[i] = v.y;
      i += HW;
      output[i] = v.z;
    } else if (remain_channel == 2) {
      output[i] = v.x;
      i += HW;
      output[i] = v.y;
    } else if (remain_channel == 1) {
      output[i] = v.x;
    }
  }
}

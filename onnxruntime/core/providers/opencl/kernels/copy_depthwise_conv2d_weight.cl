// FIXME: LICENSE NOTICE: adapted from TNN original BSD3.

__kernel void CopyDepthwiseConvWeightBufferToImage(
    const int width, const int height,  // image, width = K_h*K_w*C_o, height = CeilDiv(C_i, 4), where C_o == 1
    __global const float* data,
    __private const int4 kernel_shape,
    __private const int HW,  // K_h * K_w
    __write_only image2d_t output) {
#define C_o kernel_shape.s0
#define C_i kernel_shape.s1
#define K_h kernel_shape.s2
#define K_w kernel_shape.s3
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if (x >= width || y >= height) return;

  const int ci = y * 4;
  const int kh = x / K_w;
  const int kw = x % K_w;

  // (K_h*K_w)*ci + K_w*kh + kw
  const int base_offset = mad24(mad24(ci, K_h, kh), K_w, kw);

  // FIXME: factor this into a SAFE_GATHER_LOAD_VEC4
  float4 v = 0;
  const int num_remain = C_i - ci;
  int offset = base_offset;
  if (num_remain >= 4) {
    v.x = data[offset];
    offset += HW;
    v.y = data[offset];
    offset += HW;
    v.z = data[offset];
    offset += HW;
    v.w = data[offset];
  } else if (num_remain == 3) {
    v.x = data[offset];
    offset += HW;
    v.y = data[offset];
    offset += HW;
    v.z = data[offset];
  } else if (num_remain == 2) {
    v.x = data[offset];
    offset += HW;
    v.y = data[offset];
  } else if (num_remain == 1) {
    v.x = data[offset];
  }

  write_imagef(output, (int2)(x, y), v);
#undef K_w
#undef K_h
#undef C_i
#undef C_o
}

__kernel void CopyTensor1DToImage2D(
    __global const float* data4,
    int nfloat4,  // num of float4 elements, full-width and remainder
    __write_only image2d_t output) {
  int x = get_global_id(0);
  int width = get_global_size(0);  // image width
  int y = get_global_id(1);
  int gidx = width * y + x;

  if (gidx < nfloat4) {
    write_imagef(output, (int2)(x, y), data4[x + y * width]);
  }
}

__kernel void CopyImage2DToTensor1D() {
}

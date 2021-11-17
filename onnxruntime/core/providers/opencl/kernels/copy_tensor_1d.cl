__kernel void CopyBuffer1DToImage2D(
    __global const float4* data4,
    int nelem4,  // num of float4 elements, full-width and remainder
    __write_only image2d_t output) {
  int x = get_global_id(0);
  int width = get_global_size(0);  // image width
  int y = get_global_id(1);
  int gidx = width * y + x;

  write_imagef(output, (int2)(x, y), (float4)(x, y, 0, 0));

  if (gidx < nelem4) {
    write_imagef(output, (int2)(x, y), data4[gidx]);
  }
}

__kernel void CopyImage2DToBuffer1D(
    __read_only image2d_t data,
    __global float4* output4,
    int nelem4  // number of float4 elements, full-width and remainder
) {
  int x = get_global_id(0);
  int width = get_global_size(0);  // image width
  int y = get_global_id(1);
  int gidx = width * y + x;

  if(gidx < nelem4) {
    output4[gidx] = read_imagef(data, (int2)(x, y));
  }
}

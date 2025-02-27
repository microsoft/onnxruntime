__kernel void sin_float(
    __global const float* input,
    __global float* output) {
  int gid = get_global_id(0);
  output[gid] = sin(input[gid]);
}

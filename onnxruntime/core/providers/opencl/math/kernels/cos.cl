__kernel void cos_float(
    __global const float* input,
    __global float* output) {
  int gid = get_global_id(0);
  output[gid] = cos(input[gid]);
}

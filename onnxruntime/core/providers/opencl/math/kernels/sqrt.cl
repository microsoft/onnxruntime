__kernel void Sqrt_Float(
    __global float* input,
    __global float* output) {
  int global_id = get_global_id(0);
  output[global_id] = sqrt(input[global_id]);
}

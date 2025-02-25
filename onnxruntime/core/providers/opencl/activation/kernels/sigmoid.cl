__kernel void Sigmoid_Float(
    __global float* input,
    __global float* output) {
  int global_id = get_global_id(0);
  output[global_id] = 1.0f / (1.0f + exp(-input[global_id]));
}

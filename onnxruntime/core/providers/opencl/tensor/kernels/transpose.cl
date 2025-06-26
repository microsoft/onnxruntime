__kernel void Transpose(__global long* p_perm, __global float* input,
                        __global float* output, __global long* input_shape,
                        __global long* output_shape, long ndim) {
  int gid = get_global_id(0);

  int input_index = gid;
  int output_index = 0;
  int output_stride = 1;

  int coords[5];
  for (int i = ndim - 1; i >= 0; --i) {
    coords[i] = input_index % input_shape[i];
    input_index /= input_shape[i];
  }

  for (int i = ndim - 1; i >= 0; --i) {
    output_index += coords[p_perm[i]] * output_stride;
    output_stride *= output_shape[i];
  }

  output[output_index] = input[gid];
}

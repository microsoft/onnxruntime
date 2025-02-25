__kernel void softmax_float_opset13(__global float* input,
                                    __global long* input_shape,
                                    __global float* output,
                                    const int maxdims,
                                    const int axis) {
  int global_id = get_global_id(0);

  int axis_size = input_shape[axis];

  int offset = 1;
  for (int i = axis + 1; i < maxdims; i++) {
    offset *= input_shape[i];
  }

  int start_index = (global_id / offset) * (offset * axis_size) + (global_id % offset);

  float max_val = input[start_index];
  for (int i = 1; i < axis_size; i++) {
    float val = input[start_index + i * offset];
    if (val > max_val) {
      max_val = val;
    }
  }

  float sum = 0.0f;
  for (int i = 0; i < axis_size; i++) {
    sum += exp(input[start_index + i * offset] - max_val);
  }

  for (int i = 0; i < axis_size; i++) {
    output[start_index + i * offset] = exp(input[start_index + i * offset] - max_val) / sum;
  }
}

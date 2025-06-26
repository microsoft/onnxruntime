__kernel void ReduceMean_float(__global float* input,
                               __global float* output,
                               __global long* input_shape,
                               __global long* output_shape,
                               __global long* axis,
                               int num_axes,
                               int num_dims) {
  int gid = get_global_id(0);

  int indices[5];
  int temp = gid;
  for (int i = num_dims - 1; i >= 0; --i) {
    indices[i] = temp % output_shape[i];
    temp /= output_shape[i];
  }

  float sum = 0.0f;
  int count = 1;

  for (int i = 0; i < num_axes; ++i) {
    int ax = axis[i];
    count *= input_shape[ax];
  }

  int input_indices[5];
  for (int i = 0; i < num_dims; ++i) {
    input_indices[i] = indices[i];
  }

  for (int i = 0; i < count; ++i) {
    int offset = i;
    for (int j = num_axes - 1; j >= 0; --j) {
      int ax = axis[j];
      input_indices[ax] = offset % input_shape[ax];
      offset /= input_shape[ax];
    }

    int input_offset = 0;
    int stride_multiplier = 1;
    for (int k = num_dims - 1; k >= 0; --k) {
      input_offset += input_indices[k] * stride_multiplier;
      stride_multiplier *= input_shape[k];
    }
    sum += input[input_offset];
  }

  float mean = sum / count;

  int output_offset = 0;
  int stride_multiplier = 1;
  for (int i = num_dims - 1; i >= 0; --i) {
    output_offset += indices[i] * stride_multiplier;
    stride_multiplier *= output_shape[i];
  }
  output[output_offset] = mean;
}

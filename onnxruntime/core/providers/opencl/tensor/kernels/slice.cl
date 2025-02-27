__kernel void Slice(__global uchar* input,
                    __global uchar* output,
                    long input_dimensions,
                    long element_size,
                    __global long* input_starts,
                    __global long* input_steps,
                    __global long* input_strides,
                    __global long* output_strides,
                    long output_dimensions,
                    long max_input_size) {
  int global_id = get_global_id(0);

  long output_idx = global_id;
  long input_offset = 0;

  for (long i = 0; i < output_dimensions; ++i) {
    long coord = output_idx / output_strides[i];
    output_idx %= output_strides[i];

    long input_coord = input_starts[i] + coord * input_steps[i];
    input_offset += input_coord * input_strides[i];
    if (input_offset * element_size >= max_input_size) {
      return;
    }
  }

  for (long j = 0; j < element_size; ++j) {
    output[global_id * element_size + j] = input[input_offset * element_size + j];
  }
}

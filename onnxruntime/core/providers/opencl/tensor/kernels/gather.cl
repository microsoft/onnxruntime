__kernel void Gather(__global long* indices_tensor,
                     __global uchar* input,
                     __global uchar* output,
                     int is_string_type,
                     long element_bytes,
                     long block_size,
                     long M,
                     long N,
                     long data_batch_bytes,
                     long gathered_batch_bytes,
                     __global long* input_data_shape,
                     long axis) {
  int global_id = get_global_id(0);
  int batch = global_id / N;
  int i = global_id % N;

  long axis_dim_limit = input_data_shape[axis];

  long idx = indices_tensor[i];

  if (idx < -axis_dim_limit || idx >= axis_dim_limit) {
    return;
  }

  idx = idx < 0 ? idx + axis_dim_limit : idx;

  long src_offset_batch = batch * data_batch_bytes;
  long dst_offset_batch = batch * gathered_batch_bytes;
  long src_offset = src_offset_batch + idx * block_size;
  long dst_offset = dst_offset_batch + i * block_size;

  if (is_string_type) {
    __global char* src_str = (__global char*)(input + src_offset);
    __global char* dst_str = (__global char*)(output + dst_offset);
    for (long j = 0; j < element_bytes; ++j) {
      dst_str[j] = src_str[j];
    }
  } else {
    for (long j = 0; j < block_size; ++j) {
      output[dst_offset + j] = input[src_offset + j];
    }
  }
}

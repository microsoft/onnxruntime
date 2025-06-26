__kernel void Concat(__global uchar* Out,
                     __global uchar* In,
                     __global long* Out_Shape,
                     __global long* In_Shape,
                     long InTensor_Num,
                     long axis,
                     long dims,
                     long element_bytes) {
  long global_id = get_global_id(0);

  long out_total_elements = 1;
  for (int i = 0; i < dims; i++) {
    out_total_elements *= Out_Shape[i];
  }

  if (global_id >= out_total_elements) {
    return;
  }

  long out_idx[5];
  long temp = global_id;
  for (int i = dims - 1; i >= 0; i--) {
    out_idx[i] = temp % Out_Shape[i];
    temp /= Out_Shape[i];
  }

  long axis_offset = out_idx[axis];
  long current_offset = 0;

  long in_tensor_idx = 0;
  for (int i = 0; i < InTensor_Num; i++) {
    long axis_dim_size = In_Shape[i * dims + axis];

    if (axis_offset < current_offset + axis_dim_size) {
      in_tensor_idx = i;
      break;
    }

    current_offset += axis_dim_size;
  }

  long in_idx[5];
  for (int i = 0; i < dims; i++) {
    if (i == axis) {
      in_idx[i] = axis_offset - current_offset;
    } else {
      in_idx[i] = out_idx[i];
    }
  }

  long in_offset = 0;
  for (int j = 0; j < in_tensor_idx; j++) {
    long in_tensor_elements = 1;
    for (int k = 0; k < dims; k++) {
      in_tensor_elements *= In_Shape[j * dims + k];
    }
    in_offset += in_tensor_elements;
  }

  long in_index = 0;
  long factor = 1;
  for (int i = dims - 1; i >= 0; i--) {
    in_index += in_idx[i] * factor;
    factor *= In_Shape[in_tensor_idx * dims + i];
  }

  long out_pos = global_id * element_bytes;
  long in_pos = (in_offset + in_index) * element_bytes;

  for (int byte = 0; byte < element_bytes; byte++) {
    Out[out_pos + byte] = In[in_pos + byte];
  }
}

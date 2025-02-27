__kernel void Trilu_float(__global float* input,
                          __global float* output,
                          __global long* shape,
                          long dims,
                          long k_,
                          int upper) {
  long M = shape[dims - 2];
  long N = shape[dims - 1];
  long k_val = k_;

  if (k_val < -(M - 1) || k_val > (N - 1)) {
    k_val = 0;
  }
  long global_idx = get_global_id(0);

  long coord[5];
  long index = global_idx;
  for (long i = dims - 1; i >= 0; --i) {
    coord[i] = index % shape[i];
    index /= shape[i];
  }

  long row = coord[dims - 2];
  long col = coord[dims - 1];

  if (upper) {
    if (col - row >= k_val) {
      output[global_idx] = input[global_idx];
    } else {
      output[global_idx] = 0.0f;
    }
  } else {
    if (col - row <= k_val) {
      output[global_idx] = input[global_idx];
    } else {
      output[global_idx] = 0.0f;
    }
  }
}

__kernel void Trilu_double(__global double* input,
                           __global double* output,
                           __global long* shape,
                           long dims,
                           long k_,
                           int upper) {
  long M = shape[dims - 2];
  long N = shape[dims - 1];
  long k_val = k_;

  if (k_val < -(M - 1) || k_val > (N - 1)) {
    k_val = 0;
  }
  long global_idx = get_global_id(0);

  long coord[5];
  long index = global_idx;
  for (long i = dims - 1; i >= 0; --i) {
    coord[i] = index % shape[i];
    index /= shape[i];
  }

  long row = coord[dims - 2];
  long col = coord[dims - 1];

  if (upper) {
    if (col - row >= k_val) {
      output[global_idx] = input[global_idx];
    } else {
      output[global_idx] = 0.0f;
    }
  } else {
    if (col - row <= k_val) {
      output[global_idx] = input[global_idx];
    } else {
      output[global_idx] = 0.0f;
    }
  }
}

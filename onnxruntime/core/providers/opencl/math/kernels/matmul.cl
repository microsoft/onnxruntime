__kernel void MatMul_Batch_Float(
    const int trans_a,
    const int trans_b,
    __global float* A,
    __global float* B,
    __global float* C,
    __global long* Ashapes,
    __global long* Bshapes,
    __global long* Cshapes,
    const int maxdims) {
  int global_id = get_global_id(0);

  int indices[5] = {0, 0, 0, 0, 0};
  int temp_id = global_id;
  for (int i = maxdims - 1; i >= 0; --i) {
    indices[i] = temp_id % Cshapes[i];
    temp_id /= Cshapes[i];
  }

  long batch_size = 1;
  for (int i = 0; i < maxdims - 2; ++i) {
    batch_size *= Cshapes[i];
  }
  long C_height = Cshapes[maxdims - 2];
  long C_width = Cshapes[maxdims - 1];

  int batch_index = 0;
  for (int i = 0; i < maxdims - 2; ++i) {
    batch_index = batch_index * Cshapes[i] + indices[i];
  }
  int row = indices[maxdims - 2];
  int col = indices[maxdims - 1];

  float value = 0.0f;

  long A_height = Ashapes[maxdims - 2];
  long A_width = Ashapes[maxdims - 1];
  long B_height = Bshapes[maxdims - 2];
  long B_width = Bshapes[maxdims - 1];

  for (int k = 0; k < (trans_a ? A_height : A_width); ++k) {
    int A_row_index = trans_a ? k : row;
    int A_col_index = trans_a ? row : k;
    int B_row_index = trans_b ? col : k;
    int B_col_index = trans_b ? k : col;

    int A_index = 0;
    int B_index = 0;

    for (int i = 0; i < maxdims - 2; ++i) {
      if (Ashapes[i] != 1) {
        A_index = A_index * Ashapes[i] + (indices[i] % Ashapes[i]);
      }
    }
    A_index = A_index * A_height + A_row_index;
    A_index = A_index * A_width + A_col_index;

    for (int i = 0; i < maxdims - 2; ++i) {
      if (Bshapes[i] != 1) {  
        B_index = B_index * Bshapes[i] + (indices[i] % Bshapes[i]);
      }
    }
    B_index = B_index * B_height + B_row_index;
    B_index = B_index * B_width + B_col_index;

    float a_val = A[A_index];
    float b_val = B[B_index];
    value += a_val * b_val;
  }

  C[global_id] = value;
}

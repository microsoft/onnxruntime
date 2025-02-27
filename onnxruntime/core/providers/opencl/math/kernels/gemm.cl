__kernel void Gemm_Float(
    const int trans_a,
    const int trans_b,
    const long M,
    const long N,
    const long K,
    __global float* A,
    __global float* B,
    __global float* C,
    __global long* Cshapes,
    __global float* Output,
    const int C_exist) {
  int row = get_global_id(0);
  int col = get_global_id(1);

  if (row < M && col < N) {
    float sum = 0.0;

    for (int k = 0; k < K; ++k) {
      float a_val = (trans_a == 0) ? A[row * K + k] : A[k * M + row];
      float b_val = (trans_b == 0) ? B[k * N + col] : B[col * K + k];
      sum += a_val * b_val;
    }

    if (C_exist) {
      int C_rows = Cshapes[0];
      int C_cols = Cshapes[1];
      if (C_rows == M && C_cols == N) {
        sum += C[row * N + col];
      } else if (C_rows == 1 && C_cols == N) {
        sum += C[col];
      } else if (C_rows == M && C_cols == 1) {
        sum += C[row];
      } else if (C_rows == 1 && C_cols == 1) {
        sum += C[0];
      }
    }

    Output[row * N + col] = sum;
  }
}

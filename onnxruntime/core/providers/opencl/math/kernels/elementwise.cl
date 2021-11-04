__kernel void NAME(__global T* a, __global T* b, __global T* c, int nelement) {
  int i = get_global_id(0);
  if (i < nelement) {
    c[i] = OP(a[i], b[i]);
  }
}

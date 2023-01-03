import os
import sys

os.environ["KERNEL_EXPLORER_BUILD_DIR"] = "/home/guangyunhan/onnxruntime-ke/build_rocm/Release"
sys.path.insert(0, "/home/guangyunhan/onnxruntime-ke/onnxruntime/python/tools/kernel_explorer/kernels")

import kernel_explorer as ke
import numpy as np

batch = 64
M = 512
K = 768
N = 3072
O = 768

np.random.seed(0)
a = 0.01 * np.random.uniform(size=(batch, M, K))
b = 0.01 * np.random.uniform(size=(1, K, N))
bias = 0.1 * np.zeros((M, N))
# bias = np.random.uniform(size=(N,))
c = 0.1 * np.random.uniform(size=(1, N, O))

ref = (a @ b + bias) @ c

a = a.astype(np.float16)
b = b.astype(np.float16)
bias = bias.astype(np.float16)
c = c.astype(np.float16)
my = np.zeros_like(ref, dtype=np.float16)

dev_a = ke.DeviceArray(a)
dev_b = ke.DeviceArray(b)
dev_bias = ke.DeviceArray(bias)
dev_c = ke.DeviceArray(c)
dev_my = ke.DeviceArray(my)

op = ke.BatchedGemmBiasGeluGemm_half(dev_a, dev_b, dev_bias, dev_c, dev_my, M, K, N, O, batch)
op.Run()
dev_my.UpdateHostNumpyArray()

op.Profile()

diff = (ref - my) / ref
print(diff[[60], :, [0]])
# print(ref[0][0])
# print(my[0][0])

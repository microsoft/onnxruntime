import os
import sys

os.environ["KERNEL_EXPLORER_BUILD_DIR"] = "/home/guangyunhan/onnxruntime/build_rocm/Release"
sys.path.insert(0, "/home/guangyunhan/onnxruntime/onnxruntime/python/tools/kernel_explorer/kernels")

import kernel_explorer as ke
import numpy as np
import torch

batchsize = 64
seq_len = 512
num_heads = 12
head_dim = 64
scale = 0.125

np.random.seed(0)
q = np.random.normal(size=(batchsize, num_heads, seq_len, head_dim))
k = np.random.normal(size=(batchsize, num_heads, seq_len, head_dim))
v = np.random.normal(size=(batchsize, num_heads, seq_len, head_dim))

attn = torch.softmax(torch.matmul(torch.Tensor(q), torch.Tensor(k).transpose(2, 3)) * scale, dim=-1)
ref_out = torch.permute(torch.matmul(attn, torch.Tensor(v)), [0, 2, 1, 3]).numpy()

q = q.astype(np.float16)
k = k.astype(np.float16)
v = v.astype(np.float16)
out = np.zeros((batchsize, seq_len, num_heads, head_dim), dtype=np.float16)

dev_q = ke.DeviceArray(q)
dev_k = ke.DeviceArray(k)
dev_v = ke.DeviceArray(v)
dev_out = ke.DeviceArray(out)

op = ke.BatchedGemmSoftmaxGemmPermute_half(dev_q, dev_k, dev_v, dev_out, batchsize, seq_len, num_heads, head_dim, scale)
op.Run()
dev_out.UpdateHostNumpyArray()

diff = ref_out - out

import numpy as np
import torch

M=512
K=768
N=1024
dtype=np.float16
ranks=4

def test_torch():
    a = torch.randn((M, K), dtype=torch.float16, device=0)
    b = torch.randn((K, N), dtype=torch.float16, device=0)

    y = a @ b

    b_split = torch.chunk(b, ranks, axis=1)

    res = []
    for i in range(ranks):
        res.append(a @ b_split[i])

    y_split = torch.concat(res, axis=1)

    if torch.allclose(y, y_split):
        print(f'SAME, y-shape: {y_split.shape}, type: {y_split.dtype}')
    else:
        diff = abs(y - y_split)
        rel_diff = abs(diff / y)
        print(f'not SAME, diff: {diff.max()}, rel-diff: {rel_diff.max()}')

def test_numpy():
    a = np.random.rand(M, K).astype(dtype)
    b = np.random.rand(K, N).astype(dtype)

    y = np.dot(a, b)

    b_split = np.split(b, ranks, axis=1)

    res = []
    for i in range(ranks):
        res.append(np.dot(a, b_split[i]))

    y_split = np.concatenate(res, axis=1)

    if np.allclose(y, y_split):
        print(f'SAME, y-shape: {y_split.shape}, type: {y_split.dtype}')
    else:
        print(f'not SAME, y-shape: {y_split.shape}, type: {y_split.dtype}')


if __name__ == '__main__':
    test_torch()

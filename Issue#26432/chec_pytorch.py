# import torch
# import numpy as np

# def run_case(desc, scale_arr):
#     print(f"\n=== {desc} | scale.shape={scale_arr.shape} ===")
#     try:
#         X = torch.arange(30, dtype=torch.float32).reshape(2,5,3)
#         eps = 1e-5

#         scale = torch.tensor(scale_arr, dtype=torch.float32)

#         Y = torch.nn.functional.rms_norm(X, normalized_shape=(3,), weight=scale, eps=eps)

#         print("✓ Ran OK. Output shape:", tuple(Y.shape))
#         print("Sample output (first row):\n", Y[0,0])
#     except Exception as e:
#         print("✗ Failed:")
#         print(e)

# run_case("scalar_scale", np.array(1.5, dtype=np.float32))       # shape ()
# run_case("broadcast_1x1x1", np.ones((1,1,1), dtype=np.float32)) # shape (1,1,1)
# run_case("vector_len3", np.array([1.5,1.5,1.5], dtype=np.float32))  # shape (3,)

import torch

X = torch.arange(30, dtype=torch.float32).reshape(2, 5, 3)

layer = torch.nn.RMSNorm(normalized_shape=(3,), eps=1e-5, elementwise_affine=True)

# === מקרה תקין ===
with torch.no_grad():
    layer.weight.copy_(torch.tensor([1.5, 1.5, 1.5]))  # shape (3,)
print("✓ valid weight (3,) copied OK")

# === מקרה שגוי: ברודקאסט (1,1,1) ===
try:
    with torch.no_grad():
        layer.weight.copy_(torch.ones((1,1,1)))  # shape (1,1,1)
except Exception as e:
    print("\n✗ illegal broadcast attempt (1,1,1):", e)

# === מקרה שגוי: סקלר אמיתי (shape=()) ===
try:
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(1.5))  # scalar
    print("✓ valid weight (3,) copied OK")
except Exception as e:
    print("\n✗ illegal scalar attempt (shape=()):", e)

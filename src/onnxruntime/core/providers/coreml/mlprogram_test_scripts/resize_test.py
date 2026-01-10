import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb

target = ct.target.iOS15

x_shape = (1, 1, 3, 6)

use_scale = False  # set this to test upsample vs resize


@mb.program(input_specs=[mb.TensorSpec(shape=x_shape)], opset_version=target)
def prog(x):
    global use_scale  # noqa

    if use_scale:
        align = mb.const(val=False)
        scale_h = mb.const(val=float(1 / 3))
        scale_w = mb.const(val=float(1 / 3))
        z = mb.upsample_bilinear(x=x, scale_factor_height=scale_h, scale_factor_width=scale_w, align_corners=align)
    else:
        size_h = mb.const(val=1)
        size_w = mb.const(val=2)
        sampling_mode = mb.const(val="UNALIGN_CORNERS")
        z = mb.resize_bilinear(x=x, target_size_height=size_h, target_size_width=size_w, sampling_mode=sampling_mode)

    return z


print(prog)

# Convert to ML program
m = ct.convert(prog, minimum_deployment_target=target, compute_precision=ct.precision.FLOAT32)

x = np.array(
    [
        [
            [
                [1, 2, 3, 4, 5, 6],
                [7, 8, 9, 10, 11, 12],
                [13, 14, 15, 16, 17, 18],
            ]
        ]
    ],
    dtype=np.float32,
)

# spec = m.get_spec()
# print(spec)

print(m.predict({"x": x}))

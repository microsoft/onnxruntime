# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------


import os
import subprocess
import tempfile

import numpy as np
import numpy.typing as npt


class Q4dqWrapper:
    """A wrapper to native command line onnxruntime_mlas_q4dq"""

    def __init__(self, exepath: str):
        self.q4dq_cmd = exepath

    def quantize(self, fp32weight: npt.ArrayLike, quant_type: int) -> np.ndarray:
        """4b quantize fp32 weight to a blob"""

        array = fp32weight.astype(np.float32)
        if len(array.shape) != 2:
            raise Exception("Only 2D fp32 array accepted!")
        rows, cols = array.shape

        with tempfile.TemporaryDirectory() as tmpdirname:
            fp32file = os.path.join(tmpdirname, "fp32weight")
            array.tofile(fp32file)

            q4file = os.path.join(tmpdirname, "q4weight")

            cmd = "{cmdpath} q {k} {n} --quant_type {qtype} --input_file {fp32} --output_file {q4} --output_format bin".format(
                cmdpath=self.q4dq_cmd, k=rows, n=cols, qtype=quant_type, fp32=fp32file, q4=q4file
            )
            subprocess.run(cmd, shell=True)

            if not os.path.isfile(q4file):
                raise Exception("Quantization failed, 4b quantization is not yet supported on this platform!")

            packed = np.fromfile(q4file, dtype="uint8")
            return packed

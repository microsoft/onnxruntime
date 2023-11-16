# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import onnx

from ...calibrate import CalibrationDataReader, CalibrationMethod
from ...quantize import StaticQuantConfig

def get_qnn_qdq_config(model_input: Path,
                       calibration_data_reader: CalibrationDataReader,
                       calibrate_method=CalibrationMethod.MinMax):
    model = onnx.load_model(model_input)
    # TODO: Parse model nodes to setup overrides.
    return StaticQuantConfig(calibration_data_reader,
                             calibrate_method=calibrate_method,
                             extra_options={"MinimumRealRange": 0.0001,
                                 "DedicatedQDQPair": True})

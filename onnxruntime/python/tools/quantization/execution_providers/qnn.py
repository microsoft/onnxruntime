# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------


def qnn_preprocess_model(*args, **kwargs):
    """Report that QNN model preprocessing is no longer supported."""
    raise NotImplementedError("QNN model preprocessing is no longer supported.")


def get_qnn_qdq_config(*args, **kwargs):
    """Report that QNN quantization configuration is no longer supported."""
    raise NotImplementedError("QNN quantization configuration is no longer supported.")

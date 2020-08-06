
import numpy as np
import os
import sys
import torch

from numpy.testing import assert_allclose
from onnxruntime.capi.training import orttrainer
from onnxruntime.capi.ort_trainer import ORTTrainer as Legacy_ORTTrainer

def compare_onnx_weights(model_a, model_b, verbose=False, rtol=1e-4):
    r"""Compare whether weights between 'model_a' and 'model_b' ONNX models are within
    a certain tolerance 'rtol'

    Compares the weights of two different ONNX models and throws an error when they diverge
    Args:
        model_a, model_b (ORTTrainer): Two instances of ORTTrainer with the same model structure
        verbose (bool, default is False): Indicates if the max absolute difference for each layer should be
    calculated and printed for debug information.
        rtol (float, default is 1e-4): Tolerance for divergence.
    """
    assert isinstance(model_a, orttrainer.ORTTrainer) and isinstance(model_b, orttrainer.ORTTrainer)
    state_dict_a, state_dict_b = model_a._training_session.get_state(), model_b._training_session.get_state()
    assert len(state_dict_a.items()) == len(state_dict_b.items())
    _compare_state_dict_weights(state_dict_a, state_dict_b, verbose, rtol)


def compare_legacy_onnx_weights(model_a, model_b, verbose=False, rtol=1e-4):
    r"""Compare whether weights between 'model_a' (legacy API ONNX model) and 'model_b' (new API ONNX model)
    are within a certain tolerance 'rtol'

    Compares the weights of two different ONNX models and throws an error when they diverge
    Args:
        model_a, model_b (ORTTrainer): Two instances of ORTTrainer with the same model structure
        verbose (bool, default is False): Indicates if the max absolute difference for each layer should be
    calculated and printed for debug information.
        rtol (float, default is 1e-4): Tolerance for divergence.
    """
    assert isinstance(model_a, orttrainer.ORTTrainer) and isinstance(model_b, Legacy_ORTTrainer)
    state_dict_a, state_dict_b = model_a._training_session.get_state(), model_b.session.get_state()
    assert len(state_dict_a.items()) == len(state_dict_b.items())
    _compare_state_dict_weights(state_dict_a, state_dict_b, verbose, rtol)

def _compare_state_dict_weights(state_dict_a, state_dict_b, verbose=False, rtol=1e-4):
    for (a_name, a_val), (b_name, b_val) in zip(state_dict_a.items(), state_dict_b.items()):
        np_a_vals = np.array(a_val).flatten()
        np_b_vals = np.array(b_val).flatten()
        assert np_a_vals.shape == np_b_vals.shape
        if verbose:
            print(f'Weight name: {a_name}: absolute difference: {np.abs(np_a_vals-np_b_vals).max()}')
        assert_allclose(a_val, b_val, rtol=rtol, err_msg=f"Weight mismatch for {a_name}")

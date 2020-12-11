import numpy as np
import os
import sys
import torch

from numpy.testing import assert_allclose
from onnxruntime.training import orttrainer
from onnxruntime.capi.ort_trainer import ORTTrainer as Legacy_ORTTrainer


def assert_model_outputs(output_a, output_b, verbose=False, rtol=1e-7, atol=0):
    r"""Asserts whether output_a and output_b difference is within specified tolerance

    Args:
        output_a, output_b (list): Two list with of numeric values
        verbose (bool, default is False): if True, prints absolute difference for each weight
        rtol (float, default is 1e-7): Max relative difference
        atol (float, default is 1e-4): Max absolute difference
    """
    assert isinstance(output_a, list) and isinstance(output_b, list),\
        "output_a and output_b must be list of numbers"
    assert len(output_a) == len(output_b), "output_a and output_b must have the same length"

    # for idx in range(len(output_a)):
    assert_allclose(output_a, output_b, rtol=rtol, atol=atol, err_msg=f"Model output value mismatch")

def assert_onnx_weights(model_a, model_b, verbose=False, rtol=1e-7, atol=0):
    r"""Asserts whether weight difference between models a and b differences are within specified tolerance

    Compares the weights of two different ONNX models (model_a and model_b)
    and raises AssertError when they diverge by more than atol or rtol

    Args:
        model_a, model_b (ORTTrainer): Two instances of ORTTrainer with the same model structure
        verbose (bool, default is False): if True, prints absolute difference for each weight
        rtol (float, default is 1e-7): Max relative difference
        atol (float, default is 1e-4): Max absolute difference
    """
    assert isinstance(model_a, orttrainer.ORTTrainer) and isinstance(model_b, orttrainer.ORTTrainer)
    state_dict_a, state_dict_b = model_a._training_session.get_state(), model_b._training_session.get_state()
    assert len(state_dict_a.items()) == len(state_dict_b.items())
    _assert_state_dict_weights(state_dict_a, state_dict_b, verbose, rtol, atol)


def assert_legacy_onnx_weights(model_a, model_b, verbose=False, rtol=1e-7, atol=0):
    r"""Asserts whether weight difference between models a and b differences are within specified tolerance

    Compares the weights of a legacy model model_a and experimental model_b model
    and raises AssertError when they diverge by more than atol or rtol.

    Args:
        model_a (ORTTrainer): Instance of legacy ORTTrainer
        model_b (ORTTrainer): Instance of experimental ORTTrainer
        verbose (bool, default is False): if True, prints absolute difference for each weight.
        rtol (float, default is 1e-7): Max relative difference
        atol (float, default is 1e-4): Max absolute difference
    """
    assert isinstance(model_a, orttrainer.ORTTrainer) and isinstance(model_b, Legacy_ORTTrainer)
    state_dict_a, state_dict_b = model_a._training_session.get_state(), model_b.session.get_state()
    assert len(state_dict_a.items()) == len(state_dict_b.items())
    _assert_state_dict_weights(state_dict_a, state_dict_b, verbose, rtol, atol)


def _assert_state_dict_weights(state_dict_a, state_dict_b, verbose, rtol, atol):
    r"""Asserts whether dicts a and b value differences are within specified tolerance

    Compares the weights of two model's state_dict dicts and raises AssertError
    when they diverge by more than atol or rtol

    Args:
        model_a (ORTTrainer): Instance of legacy ORTTrainer
        model_b (ORTTrainer): Instance of experimental ORTTrainer
        verbose (bool, default is False): if True, prints absolute difference for each weight.
        rtol (float, default is 1e-7): Max relative difference
        atol (float, default is 1e-4): Max absolute difference
    """

    for (a_name, a_val), (b_name, b_val) in zip(state_dict_a.items(), state_dict_b.items()):
        np_a_vals = np.array(a_val).flatten()
        np_b_vals = np.array(b_val).flatten()
        assert np_a_vals.shape == np_b_vals.shape
        if verbose:
            print(f'Weight name: {a_name}: absolute difference: {np.abs(np_a_vals-np_b_vals).max()}')
        assert_allclose(a_val, b_val, rtol=rtol, atol=atol, err_msg=f"Weight mismatch for {a_name}")

# TODO: thiagofc: Checkpoint related for redesign
def _get_name(name):
    if os.path.exists(name):
        return name
    rel = os.path.join("testdata", name)
    if os.path.exists(rel):
        return rel
    this = os.path.dirname(__file__)
    data = os.path.join(this, "..", "testdata")
    res = os.path.join(data, name)
    if os.path.exists(res):
        return res
    raise FileNotFoundError("Unable to find '{0}' or '{1}' or '{2}'".format(name, rel, res))

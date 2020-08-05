
import numpy as np
import os
import sys
import torch

from numpy.testing import assert_allclose

# Inputs are the state_dicts for each of the model (dictionary for each layer and weights)
def compare_weights(model_a, model_b, prnt=False, rtol=1e-4):
    for (a_name, a_val), (b_name, b_val) in zip(model_a.items(), model_b.items()):
        np_a_vals = np.array(a_val).flatten()
        np_b_vals = np.array(b_val).flatten()
        assert np_a_vals.shape == np_b_vals.shape
        if prnt:
            print(a_name, np.abs(np_a_vals-np_b_vals).max())
        assert_allclose(a_val, b_val, rtol=rtol, err_msg="weight mismatch")
         




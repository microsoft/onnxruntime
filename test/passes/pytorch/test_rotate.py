# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from unittest.mock import patch

import pytest
import torch

from olive.data.template import huggingface_data_config_template
from olive.model import HfModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.pytorch.rotate import QuaRot, SpinQuant
from test.utils import get_tiny_phi3, make_local_tiny_llama


def common_test_rotate(rotate_pass, tmp_path, model_path, rotate_mode, atol, **config_kwargs):
    if model_path == "tiny-llama":
        input_model = make_local_tiny_llama(tmp_path / "model")
    else:
        input_model = get_tiny_phi3()

    p = create_pass_from_dict(rotate_pass, {"rotate_mode": rotate_mode, **config_kwargs}, disable_search=True)

    output_path = str(tmp_path / "output")
    output_model = p.run(input_model, output_path)

    assert isinstance(output_model, HfModelHandler)
    assert input_model.model_path != output_model.model_path

    original_model = input_model.load_model()
    rotated_model = output_model.load_model()

    i = torch.randint(0, 10, (1, 2))
    with torch.no_grad():
        original_output = original_model(i)
        rotated_output = rotated_model(i)
        # Cast to same dtype before comparison since rotated model may be saved/loaded in a different dtype
        assert torch.allclose(original_output.logits.float(), rotated_output.logits.float(), atol=atol)


@pytest.mark.parametrize("model_path", ["tiny-phi3", "tiny-llama"])
@pytest.mark.parametrize("rotate_mode", ["hadamard", "random"])
def test_quarot(tmp_path, model_path, rotate_mode):
    common_test_rotate(QuaRot, tmp_path, model_path, rotate_mode, 1e-5)


def get_patched_data_config(model_name_or_path, trust_remote_code, **kwargs):
    return huggingface_data_config_template(
        model_name=model_name_or_path,
        task="text-generation",
        load_dataset_config={
            "data_name": "Salesforce/wikitext",
            "subset": "wikitext-2-raw-v1",
            "split": "train",
            "trust_remote_code": trust_remote_code,
        },
        pre_process_data_config={
            "add_special_tokens": False,
            "max_seq_len": 10,
            "max_samples": 8,
            "trust_remote_code": trust_remote_code,
        },
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires a GPU")
@pytest.mark.parametrize("model_path", ["tiny-phi3", "tiny-llama"])
@pytest.mark.parametrize("rotate_mode", ["hadamard", "random"])
@patch("olive.passes.pytorch.rotate.get_calibration_data_config", side_effect=get_patched_data_config)
def test_spinquant(_, tmp_path, model_path, rotate_mode):
    common_test_rotate(
        SpinQuant,
        tmp_path,
        model_path,
        rotate_mode,
        # training updates the rotation matrix so the outputs are slightly different
        5e-3,
        # not all gpus support bf16
        # gradient checkpointing doesn't work on v100 CI agent
        training_args={"bf16": False, "gradient_checkpointing": False},
    )

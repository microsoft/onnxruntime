# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import pytest
import torch

from olive.model import PyTorchModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.pytorch.capture_layer_annotations import CaptureLayerAnnotations


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = torch.nn.Linear(4, 4)
        self.mlp = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.mlp(self.attn(x))


def _make_input_model():
    return PyTorchModelHandler(model_loader=lambda _: SimpleModel())


class TestCaptureLayerAnnotations:
    def test_layer_annotations_stored_in_model_attributes(self, tmp_path):
        annotations = {"encoder": ["attn", "mlp"], "decoder": ["cross_attn"]}
        p = create_pass_from_dict(CaptureLayerAnnotations, {"layer_annotations": annotations}, disable_search=True)

        out = p.run(_make_input_model(), str(tmp_path))

        assert out.model_attributes is not None
        assert out.model_attributes["layer_annotations"] == annotations

    def test_output_is_deep_copy(self, tmp_path):
        annotations = {"layer0": ["sub1"]}
        p = create_pass_from_dict(CaptureLayerAnnotations, {"layer_annotations": annotations}, disable_search=True)
        input_model = _make_input_model()

        out = p.run(input_model, str(tmp_path))

        assert out is not input_model
        assert input_model.model_attributes is None or "layer_annotations" not in (input_model.model_attributes or {})

    def test_preserves_existing_model_attributes(self, tmp_path):
        annotations = {"enc": ["attn"]}
        p = create_pass_from_dict(CaptureLayerAnnotations, {"layer_annotations": annotations}, disable_search=True)
        input_model = _make_input_model()
        input_model.model_attributes = {"some_key": "some_value"}

        out = p.run(input_model, str(tmp_path))

        assert out.model_attributes["some_key"] == "some_value"
        assert out.model_attributes["layer_annotations"] == annotations

    def test_validate_config_rejects_empty_annotations(self):
        from olive.hardware import DEFAULT_CPU_ACCELERATOR

        config = CaptureLayerAnnotations.generate_config(DEFAULT_CPU_ACCELERATOR, {"layer_annotations": {}})
        assert CaptureLayerAnnotations.validate_config(config, DEFAULT_CPU_ACCELERATOR) is False

    def test_validate_config_accepts_non_empty_annotations(self):
        from olive.hardware import DEFAULT_CPU_ACCELERATOR

        config = CaptureLayerAnnotations.generate_config(
            DEFAULT_CPU_ACCELERATOR, {"layer_annotations": {"enc": ["attn"]}}
        )
        assert CaptureLayerAnnotations.validate_config(config, DEFAULT_CPU_ACCELERATOR) is True

    @pytest.mark.parametrize(
        "annotations",
        [
            {"encoder": ["attn"]},
            {"a": ["x"], "b": ["y", "z"]},
        ],
    )
    def test_various_annotation_mappings(self, annotations, tmp_path):
        p = create_pass_from_dict(CaptureLayerAnnotations, {"layer_annotations": annotations}, disable_search=True)

        out = p.run(_make_input_model(), str(tmp_path))

        assert out.model_attributes["layer_annotations"] == annotations

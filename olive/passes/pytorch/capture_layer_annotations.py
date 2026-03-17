# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from copy import deepcopy
from typing import Union

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import HfModelHandler, PyTorchModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class CaptureLayerAnnotations(Pass):
    """Capture layer annotation metadata for an ONNX model.

    Given a mapping of layer names to node-name substrings, attaches a
    ``layer_annotations`` dictionary to the model attributes.  Downstream
    ONNX conversion passes will read this attribute and annotate each ONNX
    node whose name contains a matching substring with a ``layer_ann``
    metadata property.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "layer_annotations": PassConfigParam(
                type_=dict,
                required=True,
                description=(
                    "Mapping of layer name to a list of node-name substrings. "
                    'For example: {"encoder": ["attn", "mlp"], "decoder": ["cross_attn"]}. '
                    "During ONNX conversion every node whose name contains a listed substring "
                    "will receive a metadata property 'layer_ann' set to the layer name."
                ),
            ),
        }

    @classmethod
    def validate_config(
        cls,
        config: type[BasePassConfig],
        accelerator_spec: AcceleratorSpec,
    ) -> bool:
        if not super().validate_config(config, accelerator_spec):
            return False

        if not config.layer_annotations:
            logger.info("layer_annotations must be a non-empty dictionary.")
            return False

        return True

    def _run_for_config(
        self, model: Union[HfModelHandler, PyTorchModelHandler], config: type[BasePassConfig], output_model_path: str
    ) -> Union[HfModelHandler, PyTorchModelHandler]:
        model.model = None
        output_model = deepcopy(model)
        output_model.model_attributes = model_attributes = output_model.model_attributes or {}
        model_attributes["layer_annotations"] = config.layer_annotations

        return output_model

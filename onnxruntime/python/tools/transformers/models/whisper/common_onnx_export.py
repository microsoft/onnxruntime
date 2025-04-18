# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import inspect

import torch
from models.torch_export_patches import bypass_export_some_errors, string_type, torch_deepcopy
from models.torch_export_patches.patch_inputs import convert_dynamic_axes_into_dynamic_shapes, replace_dynamic_shapes
from packaging import version
from transformers.cache_utils import EncoderDecoderCache


def export_to_onnx(
    model,
    inputs,
    out_path,
    export_params,
    input_names,
    output_names,
    dynamic_axes,
    opset_version=18,
    do_constant_folding=True,
    verbose=False,
    use_dynamo_export=False,
    custom_opsets=None,
    dynamic_shapes_addition=None,
    dynamic_shapes_deletion=None,
):
    if not use_dynamo_export:
        torch.onnx.export(
            model,
            args=inputs,
            f=out_path,
            export_params=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
            verbose=verbose,
            dynamo=use_dynamo_export,
            custom_opsets=custom_opsets,
        )
        return

    print(f"[export_to_onnx] (A) checking the model {type(model)} is working with inputs={string_type(inputs, with_shape=True)}")
    model(*torch_deepcopy(inputs))
    print("[export_to_onnx] done.")
    if model.__class__.__name__ == "WhisperDecoder":
        n_layers = len(inputs[-1])
        print(f"[export_to_onnx] WhisperDecoder: {n_layers} layers")
        dynamic_shapes = (
            {0: 'batch_size', 1: 'sequence_length'},
            {0: 'batch_size'},
            [
                (
                    {0: 'batch_size', 2: 'past_sequence_length'},
                    {0: 'batch_size', 2: 'past_sequence_length'},
                    {0: 'batch_size', 2: 'past_sequence_length'},
                    {0: 'batch_size', 2: 'past_sequence_length'},
                )
            ] * n_layers
        )
    else:
        raise NotImplementedError(f"dynamic axes {dynamic_axes} not yet supported for class {type(model)}")

    print(f"[export_to_onnx] dynamic_shapes={dynamic_shapes!r}")
    if version.Version(torch.__version__) < version.Version("2.7"):
        # This section is only needed for torch==2.6. The workaround implemented here
        # to fix bugs is not necessary with torch>=2.7.
        # - strings are not allowed with torch 2.6, so we replace them by DYNAMIC
        # - TypePromotion was fixed in torch==2.7
        from onnxscript import opset18 as op

        dynamic_shapes = replace_dynamic_shapes(
            dynamic_shapes,
            dict(batch_size=torch.export.Dim("batch_size")),
            default_value=torch.export.Dim.DYNAMIC,
        )

        # TypePromotion cannot fix a type issue after the conversion.
        # We insert an additional CastLike when the exporter
        def custom_aten_ge(self, other):
            if isinstance(other, (int, float)):
                return op.GreaterOrEqual(self, op.CastLike(other, self))
            return op.GreaterOrEqual(self, other)

        with bypass_export_some_errors(patch_transformers=True):
            # ONNX pass TypePromotion crashes for torch 2.6.
            # It can be bypassed by exporting first into an exported program.
            # We then need to apply run_decompositions() before onnx conversion starts.
            ep = torch.export.export(
                model,
                inputs,
                dynamic_shapes=dynamic_shapes,
                strict=False,
            )
            ep = ep.run_decompositions()
            torch.onnx.export(
                ep,
                inputs,
                out_path,
                dynamic_shapes=dynamic_shapes,
                dynamo=True,
                verbose=verbose,
                optimize=True,
                custom_translation_table={torch.ops.aten.ge.Scalar: custom_aten_ge},
                custom_opsets=custom_opsets,
            )
    else:
        with bypass_export_some_errors(patch_transformers=True):
            torch.onnx.export(
                model,
                inputs,
                out_path,
                dynamic_shapes=dynamic_shapes,
                dynamo=True,
                verbose=verbose,
                optimize=True,
            )

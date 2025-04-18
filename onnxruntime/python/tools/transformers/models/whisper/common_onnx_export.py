# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from packaging import version
import torch

from models.torch_export_patches import bypass_export_some_errors
from models.torch_export_patches.patch_inputs import convert_dynamic_axes_into_dynamic_shapes, replace_dynamic_shapes


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

    model_args, model_kwargs, dynamic_shapes = convert_dynamic_axes_into_dynamic_shapes(
        model, args=inputs, dynamic_axes=dynamic_axes, prefix_mapping={"present": "past_key_values"}
    )

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
                (),
                kwargs=model_kwargs,
                dynamic_shapes=dynamic_shapes,
                strict=False,
            )
            ep = ep.run_decompositions()
            torch.onnx.export(
                ep,
                (),
                out_path,
                kwargs=model_kwargs,
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
                (),
                out_path,
                kwargs=model_kwargs,
                dynamic_shapes=dynamic_shapes,
                dynamo=True,
                verbose=verbose,
                optimize=True,
            )

import copy
import importlib
import logging
from pathlib import Path

from packaging import version

from .calibrate import CalibrationDataReader
from .quant_utils import load_model_with_shape_infer


class WeightOnlyQuantConfig:
    def __init__(
        self,
        algorithm,
        group_size=32,
        scheme="sym",
        accuracy_level=0,
        use_external_data_format=False,
    ):
        """This is the Base class for Weight Only Quant Configuration.

        Args:
            algorithm:
                weight only quantize algorithm name.
            group_size:
                how many elements share one scale/zp. -1 indicates the per-channel
                quantization per output channel.
            scheme:
                symmetrize or asymmetric calibration data for weights.
            accuracy_level:
                support 0 (default fp32), 1 (optimized fp32 for intel CPU), 2 (fp16), 3 (bf16), 4 (int8). Set to 0 by default.
            use_external_data_format:
                option used for large size (>2GB) model. Set to False by default.
        """
        """This is the Base class for Weight Only Quant Configuration.

        Args:
            algorithm:
                weight only quantize algorithm name.
        """
        self.algorithm = algorithm
        self.group_size = group_size
        self.scheme = scheme
        self.use_external_data_format = use_external_data_format
        self.accuracy_level = accuracy_level


class RTNWeightOnlyQuantConfig(WeightOnlyQuantConfig):
    def __init__(
        self,
        group_size=32,
        scheme="sym",
        accuracy_level=0,
        ratios=None,
        use_external_data_format=False,
    ):
        """
        This is a class for round-to-nearest (RTN) algorithm Weight Only Quant Configuration.
        RTN is the most straightforward way to quantize weight using scale maps.

        Args:
            group_size:
                how many elements share one scale/zp. -1 indicates the per-channel
                quantization per output channel.
            scheme:
                symmetrize or asymmetric calibration data for weights.
            accuracy_level:
                support 0 (default fp32), 1 (optimized fp32 for intel CPU), 2 (fp16), 3 (bf16), 4 (int8). Set to 0 by default.
            use_external_data_format:
                option used for large size (>2GB) model. Set to False by default.
        """
        if ratios is None:
            ratios = {}
        super().__init__(
            algorithm="RTN",
            group_size=group_size,
            scheme=scheme,
            accuracy_level=accuracy_level,
            use_external_data_format=use_external_data_format
        )
        self.ratios = ratios


class GPTQWeightOnlyQuantConfig(WeightOnlyQuantConfig):
    def __init__(
        self,
        calibration_data_reader: CalibrationDataReader,
        group_size=32,
        scheme="asym",
        percdamp=0.01,
        blocksize=128,
        actorder=False,
        mse=False,
        perchannel=True,
        accuracy_level=0,
        use_external_data_format=False,
    ):
        """
        This is a class for GPTQ algorithm Weight Only Quant Configuration.
        GPTQ algorithm provides more accurate quantization but requires more computational resources.

        Args:
            calibration_data_reader:
                a calibration data reader. It enumerates calibration data and generates inputs for the original model.
            group_size:
                how many elements share one scale/zp. -1 indicates the per-channel
                quantization per output channel.
            scheme:
                symmetrize or asymmetric calibration data for weights.
            percdamp:
                percent of the average Hessian diagonal to use for dampening.
            blocksize (int, optional):
                channel number in one block to execute a GPTQ quantization iteration.
            actorder (bool, optional):
                whether rearrange Hessian matrix considering the diag's value.
            mse (bool, optional):
                whether get scale and zero point with mse error.
            perchannel (bool, optional):
                whether quantize weight per-channel.
            accuracy_level:
                support 0 (default fp32), 1 (optimized fp32 for intel CPU), 2 (fp16), 3 (bf16), 4 (int8). Set to 0 by default.
            use_external_data_format:
                option used for large size (>2GB) model. Set to False by default.
        """
        super().__init__(
            algorithm="GPTQ",
            group_size=group_size,
            scheme=scheme,
            accuracy_level=accuracy_level,
            use_external_data_format=use_external_data_format,
        )
        self.calibration_data_reader = calibration_data_reader
        self.percdamp = percdamp
        self.blocksize = blocksize
        self.actorder = actorder
        self.mse = mse
        self.perchannel = perchannel


def _generate_weight_only_node_config(model, group_size, scheme):
    """Generate weight only quant configuration for nodes.

    Args:
        model:
            onnx.ModelProto.
        group_size:
            how many elements share one scale/zp. -1 indicates the per-channel
            quantization per output channel.
        scheme:
            symmetrize or asymmetric calibration data for weights.

    Returns:
        dict: weight only quant configuration for nodes.
    """
    weight_only_node_config = {}
    template_config = {"bits": 4, "group_size": group_size, "scheme": scheme}
    for node in model.graph.node:
        if node.op_type in ["MatMul"]:
            weight_only_node_config[node.name] = template_config
    return weight_only_node_config


def quantize_weight_only(
    model_input: Path,
    model_output: Path,
    weight_only_config: WeightOnlyQuantConfig,
):
    """Weight Only Quantize a model with WeightOnlyQuantConfig. Please refer to
       https://github.com/intel/neural-compressor/blob/master/docs/source/quantization_weight_only.md
       for more details on weight only quantization.

    Args:
        model_input (Path): Path to the model to weight only quantize.
        model_output (Path): Path to save the quantized model.
        weight_only_config (WeightOnlyQuantConfig): Weight Only Quantization Configuration.

    Raises:
        RuntimeError: Raise RuntimeError if neural-compressor is not correctly installed.
    """
    try:
        importlib.import_module("neural_compressor")
    except Exception as e:
        logging.error(f"{e}.")
        raise RuntimeError("neural-compressor is not correctly installed. Please check your environment.") from e

    import neural_compressor

    assert version.parse(neural_compressor.__version__) >= version.parse(
        "2.3.0"
    ), "Require neural-compressor >= 2.3.0 to support weight only quantization!"

    def inc_dataloader():
        data_reader = copy.deepcopy(weight_only_config.calibration_data_reader)
        for data in data_reader:
            yield data, None

    model = load_model_with_shape_infer(Path(model_input))
    scheme = weight_only_config.scheme
    group_size = weight_only_config.group_size
    accuracy_level = weight_only_config.accuracy_level
    weight_only_node_config = _generate_weight_only_node_config(model, group_size, scheme)

    algorithm = weight_only_config.algorithm
    if algorithm == "RTN":
        from neural_compressor.adaptor.ox_utils.weight_only import rtn_quantize

        ratios = weight_only_config.ratios

        model = rtn_quantize(
            model=model_input,
            weight_config=weight_only_node_config,
            ratios=ratios,
            accuracy_level=accuracy_level,
        )
    elif algorithm == "GPTQ":
        from neural_compressor.adaptor.ox_utils.weight_only import gptq_quantize

        percdamp = weight_only_config.percdamp
        blocksize = weight_only_config.blocksize
        actorder = weight_only_config.actorder
        mse = weight_only_config.mse
        perchannel = weight_only_config.perchannel
        dataloader = inc_dataloader()

        model = gptq_quantize(
            model=model_input,
            weight_config=weight_only_node_config,
            dataloader=dataloader,
            n_samples=-1,
            percdamp=percdamp,
            blocksize=blocksize,
            actorder=actorder,
            mse=mse,
            perchannel=perchannel,
            accuracy_level=accuracy_level,
        )

    model.save_model_to_file(model_output, weight_only_config.use_external_data_format)

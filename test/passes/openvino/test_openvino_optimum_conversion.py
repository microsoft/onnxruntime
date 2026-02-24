# -------------------------------------------------------------------------
# Copyright (c) Intel Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path

import pytest

from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.openvino.optimum_intel import OpenVINOOptimumConversion
from test.utils import get_hf_model, package_version_at_least

pytestmark = pytest.mark.openvino


@pytest.mark.skipif(
    not package_version_at_least("optimum", "2.1.0"),
    reason="Requires optimum >= 2.1.0",
)
def test_openvino_optimum_conversion_pass_convert_with_tokenizers(tmp_path):
    # setup
    input_hf_model = get_hf_model("hf-internal-testing/tiny-random-PhiForCausalLM")
    openvino_optimum_conversion_config = {}

    p = create_pass_from_dict(OpenVINOOptimumConversion, openvino_optimum_conversion_config, disable_search=True)

    # create output folder
    output_folder = str(Path(tmp_path) / "openvino_optimum_convert")

    # execute
    ov_output_model = p.run(input_hf_model, output_folder)

    # define the XML and BIN file paths if openvino models are produced
    xml_file = Path(ov_output_model.model_path) / "openvino_model.xml"
    bin_file = Path(ov_output_model.model_path) / "openvino_model.bin"

    # test if the model xml and bin files are created
    assert xml_file.exists()
    assert xml_file.is_file()
    assert bin_file.exists()
    assert bin_file.is_file()


@pytest.mark.skipif(
    not package_version_at_least("optimum", "2.1.0"),
    reason="Requires optimum >= 2.1.0",
)
def test_openvino_optimum_conversion_pass_convert_without_tokenizers(tmp_path):
    # setup
    input_hf_model = get_hf_model("hf-internal-testing/tiny-random-PhiForCausalLM")
    openvino_optimum_conversion_config = {"extra_args": {"disable_convert_tokenizer": True}}

    p = create_pass_from_dict(OpenVINOOptimumConversion, openvino_optimum_conversion_config, disable_search=True)

    # create output folder
    output_folder = str(Path(tmp_path) / "openvino_optimum_convert")

    # execute
    ov_output_model = p.run(input_hf_model, output_folder)

    # define the XML and BIN file paths if openvino models are produced
    xml_file = Path(ov_output_model.model_path) / "openvino_model.xml"
    bin_file = Path(ov_output_model.model_path) / "openvino_model.bin"

    # test if the model xml and bin files are created
    assert xml_file.exists()
    assert xml_file.is_file()
    assert bin_file.exists()
    assert bin_file.is_file()


@pytest.mark.skipif(
    not package_version_at_least("optimum", "2.1.0"),
    reason="Requires optimum >= 2.1.0",
)
def test_openvino_optimum_conversion_pass_convert_with_weight_compression(tmp_path):
    # setup
    input_hf_model = get_hf_model("hf-internal-testing/tiny-random-PhiForCausalLM")
    openvino_optimum_conversion_config = {
        "ov_quant_config": {
            "weight_format": "int4",
            "dataset": "wikitext2",
            "group_size": 1,
            "ratio": 1.0,
            "awq": True,
            "scale_estimation": True,
        }
    }

    p = create_pass_from_dict(OpenVINOOptimumConversion, openvino_optimum_conversion_config, disable_search=True)

    # create output folder
    output_folder = str(Path(tmp_path) / "openvino_optimum_convert")

    # execute
    ov_output_model = p.run(input_hf_model, output_folder)

    # define the XML and BIN file paths if openvino models are produced
    xml_file = Path(ov_output_model.model_path) / "openvino_model.xml"
    bin_file = Path(ov_output_model.model_path) / "openvino_model.bin"

    # test if the model xml and bin files are created
    assert xml_file.exists()
    assert xml_file.is_file()
    assert bin_file.exists()
    assert bin_file.is_file()


@pytest.mark.skipif(
    not package_version_at_least("optimum", "2.1.0"),
    reason="Requires optimum >= 2.1.0",
)
def test_openvino_optimum_conversion_pass_convert_with_quantization(tmp_path):
    # setup
    input_hf_model = get_hf_model("hf-internal-testing/tiny-random-clip-zero-shot-image-classification")
    openvino_optimum_conversion_config = {
        "extra_args": {"device": "npu"},
        "ov_quant_config": {"weight_format": "int8"},
    }

    p = create_pass_from_dict(OpenVINOOptimumConversion, openvino_optimum_conversion_config, disable_search=True)

    # create output folder
    output_folder = str(Path(tmp_path) / "openvino_optimum_convert")

    # execute
    ov_output_model = p.run(input_hf_model, output_folder)

    # define the XML and BIN file paths if openvino models are produced
    xml_file = Path(ov_output_model.model_path) / "openvino_model.xml"
    bin_file = Path(ov_output_model.model_path) / "openvino_model.bin"

    # test if the model xml and bin files are created
    assert xml_file.exists()
    assert xml_file.is_file()
    assert bin_file.exists()
    assert bin_file.is_file()


@pytest.mark.skipif(
    not package_version_at_least("optimum", "2.1.0"),
    reason="Requires optimum >= 2.1.0",
)
def test_openvino_optimum_conversion_pass_convert_multiple_components_without_main(tmp_path):
    # setup
    input_hf_model = get_hf_model("hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration")
    openvino_optimum_conversion_config = {
        "ov_quant_config": {
            "weight_format": "int4",
            "dataset": "contextual",
            "group_size": 1,
            "ratio": 1.0,
            "awq": True,
            "scale_estimation": True,
            "num_samples": 1,
        },
        "extra_args": {"task": "image-text-to-text"},
    }

    p = create_pass_from_dict(OpenVINOOptimumConversion, openvino_optimum_conversion_config, disable_search=True)

    # create output folder
    output_folder = str(Path(tmp_path) / "openvino_optimum_convert")

    # execute
    ov_output_model = p.run(input_hf_model, output_folder)

    # no openvino_model is expected for this test
    # instead only component models are expected
    for component_name, _ in ov_output_model.get_model_components():
        xml_file = Path(ov_output_model.model_path) / component_name / f"{component_name}.xml"
        bin_file = Path(ov_output_model.model_path) / component_name / f"{component_name}.bin"

        # test if the model xml and bin files are created
        assert xml_file.exists()
        assert xml_file.is_file()
        assert bin_file.exists()
        assert bin_file.is_file()

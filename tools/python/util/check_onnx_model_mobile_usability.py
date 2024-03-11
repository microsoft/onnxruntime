# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import logging
import pathlib

# need this before the mobile helper imports for some reason
logging.basicConfig(format="%(levelname)s:  %(message)s")

from .mobile_helpers import check_model_can_use_ort_mobile_pkg, usability_checker  # noqa: E402


def check_usability():
    parser = argparse.ArgumentParser(
        description="""Analyze an ONNX model to determine how well it will work in mobile scenarios, and whether
        it is likely to be able to use the pre-built ONNX Runtime Mobile Android or iOS package.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--check_mobile_package",
        action="store_true",
        required=False,
        help="Check if the model is likely to work well with the ORT Mobile package. "
        "This package supports a limited set of operators and ONNX opsets, and requires an ORT format model. "
        "These limitations are to reduce binary size. In general the 'full' packages for mobile platforms are "
        "recommended as they support all operators and opsets, and ONNX format models."
        "[Android] Mobile package: onnxruntime-mobile, Full package: onnxruntime-android "
        "[iOS] Mobile package: onnxruntime-mobile-{objc|c}, Full package: onnxruntime-{objc|c}",
    )
    parser.add_argument(
        "--config_path",
        help="Path to required operators and types configuration used to build the pre-built ORT mobile package.",
        required=False,
        type=pathlib.Path,
        default=check_model_can_use_ort_mobile_pkg.get_default_config_path(),
    )
    parser.add_argument(
        "--log_level", choices=["debug", "info", "warning", "error"], default="info", help="Logging level"
    )
    parser.add_argument("model_path", help="Path to ONNX model to check", type=pathlib.Path)

    args = parser.parse_args()
    logger = logging.getLogger("check_usability")

    if args.log_level == "debug":
        logger.setLevel(logging.DEBUG)
    elif args.log_level == "info":
        logger.setLevel(logging.INFO)
    elif args.log_level == "warning":
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.ERROR)

    try_eps = usability_checker.analyze_model(args.model_path, skip_optimize=False, logger=logger)

    if args.check_mobile_package:
        can_use_mobile_package = check_model_can_use_ort_mobile_pkg.run_check(args.model_path, args.config_path, logger)

        if can_use_mobile_package:
            logger.info(
                "The mobile package requires an ORT format model.\n"
                "Run `python -m onnxruntime.tools.convert_onnx_models_to_ort ...` to convert the ONNX model to ORT "
                "format.\n"
                "By default, the conversion tool will create an ORT format model with saved optimizations which can "
                "potentially be applied at runtime (with a .with_runtime_opt.ort file extension) for use with NNAPI "
                "or CoreML, and a fully optimized ORT format model (with a .ort file extension) for use with the CPU "
                "EP."
            )

            if try_eps:
                logger.info(
                    "As NNAPI or CoreML may provide benefits with this model it is recommended to compare the "
                    "performance of the <model>.with_runtime_opt.ort model using the NNAPI EP on Android, and the "
                    "CoreML EP on iOS, against the performance of the <model>.ort model using the CPU EP."
                )
            else:
                logger.info("For optimal performance the model should be used with the CPU EP. ")
    else:
        if try_eps:
            logger.info(
                "As NNAPI or CoreML may provide benefits with this model it is recommended to compare the "
                "performance of the model using the NNAPI EP on Android, and the CoreML EP on iOS, "
                "against the performance using the CPU EP."
            )
        else:
            logger.info("For optimal performance the model should be used with the CPU EP. ")


if __name__ == "__main__":
    check_usability()

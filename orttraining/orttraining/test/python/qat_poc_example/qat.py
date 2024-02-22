import logging
import os

import onnx
import quantize
import utils
from model import create_training_artifacts, get_models
from train import train_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s:%(lineno)s:%(funcName)s::%(message)s",
)

if __name__ == "__main__":
    logging.info("Quantization aware training: POC with MNIST Dataset")

    pt_model, onnx_model = get_models()
    model_dir = "model_dir"
    model_name = "mnist"
    utils.makedir("model_dir")

    logging.info("Saving ONNX model to path: %s", os.path.join(model_dir, f"{model_name}.onnx"))
    onnx.save(onnx_model, os.path.join(model_dir, f"{model_name}.onnx"))

    logging.info(
        "Begining Quantization process for model saved at: %s",
        os.path.join(model_dir, f"{model_name}.onnx"),
    )
    logging.info("Skipping model preprocessing step. As QAT requires a un preprocessed model.")

    # Avoid preprocessing the model as QAT requires a un preprocessed model.
    """
    preprocessed_model_name = "mnist_preprocessed"
    quantize.preprocess(
        os.path.join(model_dir, f"{model_name}.onnx"),
        os.path.join(model_dir, f"{preprocessed_model_name}.onnx"),
    )
    """

    logging.info("Initializing static quantization.")
    quantized_model_name = "mnist_quantized"
    quantize.quantize_static(
        os.path.join(model_dir, f"{model_name}.onnx"),
        os.path.join(model_dir, f"{quantized_model_name}.onnx"),
    )

    logging.info("Preparing the training artifacts for QAT.")
    training_model_name = "mnist_qat_"
    artifacts_dir = os.path.join(model_dir, "training_artifacts")
    utils.makedir(artifacts_dir)
    training_artifacts = create_training_artifacts(
        os.path.join(model_dir, f"{quantized_model_name}.onnx"),
        artifacts_dir=artifacts_dir,
        model_prefix=training_model_name,
    )

    logging.info("Starting training of QAT model.")
    train_model(*training_artifacts)

    # TODO(baijumeswani): Export trained QAT model for inferencing.

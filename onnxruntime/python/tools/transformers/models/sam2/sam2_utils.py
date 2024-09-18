# -------------------------------------------------------------------------
# Copyright (R) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import os

import torch
from sam2.build_sam import build_sam2
from sam2.modeling.sam2_base import SAM2Base

logger = logging.getLogger(__name__)


def get_model_cfg(model_type) -> str:
    assert model_type in ["sam2_hiera_tiny", "sam2_hiera_small", "sam2_hiera_large", "sam2_hiera_base_plus"]
    if model_type == "sam2_hiera_tiny":
        model_cfg = "sam2_hiera_t.yaml"
    elif model_type == "sam2_hiera_small":
        model_cfg = "sam2_hiera_s.yaml"
    elif model_type == "sam2_hiera_base_plus":
        model_cfg = "sam2_hiera_b+.yaml"
    else:
        model_cfg = "sam2_hiera_l.yaml"
    return model_cfg


def build_sam2_model(checkpoint_dir: str, model_type: str, device="cpu") -> SAM2Base:
    sam2_checkpoint = os.path.join(checkpoint_dir, f"{model_type}.pt")
    model_cfg = get_model_cfg(model_type)
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    return sam2_model


def get_decoder_onnx_path(dir: str, model_type, multimask_output) -> str:
    return os.path.join(dir, f"{model_type}_decoder" + ("_multi" if multimask_output else "") + ".onnx")


def get_image_encoder_onnx_path(dir: str, model_type) -> str:
    return os.path.join(dir, f"{model_type}_image_encoder.onnx")


def encoder_shape_dict(batch_size: int, height: int, width: int):
    assert height == 1024 and width == 1024, "Only 1024x1024 images are supported."
    return {
        "image": [batch_size, 3, height, width],
        "image_features_0": [batch_size, 32, height // 4, width // 4],
        "image_features_1": [batch_size, 64, height // 8, width // 8],
        "image_embeddings": [batch_size, 256, height // 16, width // 16],
    }


def decoder_shape_dict(
    original_image_height: int,
    original_image_width: int,
    num_labels: int = 1,
    max_points: int = 16,
    num_masks: int = 1,
) -> dict:
    height: int = 1024
    width: int = 1024
    return {
        "image_features_0": [1, 32, height // 4, width // 4],
        "image_features_1": [1, 64, height // 8, width // 8],
        "image_embeddings": [1, 256, height // 16, width // 16],
        "point_coords": [num_labels, max_points, 2],
        "point_labels": [num_labels, max_points],
        "input_masks": [num_labels, 1, height // 4, width // 4],
        "has_input_masks": [num_labels],
        "original_image_size": [2],
        "masks": [num_labels, num_masks, original_image_height, original_image_width],
        "iou_predictions": [num_labels, num_masks],
        "low_res_masks": [num_labels, num_masks, height // 4, width // 4],
    }


def compare_tensors_with_tolerance(
    name: str,
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    atol=5e-3,
    rtol=1e-4,
    mismatch_percentage_tolerance=0.1,
) -> bool:
    assert tensor1.shape == tensor2.shape
    a = tensor1.clone().float()
    b = tensor2.clone().float()

    differences = torch.abs(a - b)
    mismatch_count = (differences > (rtol * torch.max(torch.abs(a), torch.abs(b)) + atol)).sum().item()

    total_elements = a.numel()
    mismatch_percentage = (mismatch_count / total_elements) * 100

    passed = mismatch_percentage < mismatch_percentage_tolerance

    log_func = logger.error if not passed else logger.info
    log_func(
        "%s: mismatched elements percentage %.2f (%d/%d). Verification %s (threshold=%.2f).",
        name,
        mismatch_percentage,
        mismatch_count,
        total_elements,
        "passed" if passed else "failed",
        mismatch_percentage_tolerance,
    )

    return passed


def random_sam2_input_image(batch_size=1, image_height=1024, image_width=1024) -> torch.Tensor:
    image = torch.randn(batch_size, 3, image_height, image_width).cpu()
    return image


def setup_logger(verbose=True):
    if verbose:
        logging.basicConfig(format="[%(filename)s:%(lineno)s - %(funcName)20s()] %(message)s")
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.basicConfig(format="[%(message)s")
        logging.getLogger().setLevel(logging.WARNING)

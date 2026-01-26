# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import numpy as np
import numpy.typing as npt


def cosine_similarity(actual: npt.NDArray[float], expected: npt.NDArray[float]) -> float:
    # Flatten the values for actual and expected
    actual, expected = actual.flatten(), expected.flatten()
    actual = actual.astype(float)
    expected = expected.astype(float)

    if actual.shape != expected.shape:
        raise ValueError(f"CosineSimilarity: shape mismatch {actual.shape} vs {expected.shape}")

    # Replace NaN or Inf values with 0.
    actual = np.nan_to_num(actual, nan=0.0, posinf=0.0, neginf=0.0)
    expected = np.nan_to_num(expected, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize the vectors to avoid overflow.
    actual_norm = np.linalg.norm(actual)
    expected_norm = np.linalg.norm(expected)

    if actual_norm == 0 or expected_norm == 0:
        # Zero norm encountered: cosine similarity is undefined; use 0.0 by convention.
        return 0.0

    actual_normalized = actual / actual_norm
    expected_normalized = expected / expected_norm

    # Calculate dot product and norms.
    num = np.dot(actual_normalized, expected_normalized.T)
    denom = np.linalg.norm(actual_normalized) * np.linalg.norm(expected_normalized)

    if denom == 0.0:
        # Zero denominator encountered: cosine similarity is undefined; use 0.0 by convention.
        return 0.0

    similarity_score = num / denom

    if np.isnan(similarity_score):
        return 0.0

    return similarity_score


def assert_cosine_similar(actual: npt.NDArray[float], expected: npt.NDArray[float], tolerance: float) -> None:
    assert cosine_similarity(actual, expected) >= tolerance

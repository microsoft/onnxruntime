# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This code is used to generate the test cases for the AffineGrid operator
# in onnxruntime/test/providers/cpu/tensor/affine_grid_test.cc

import argparse

import numpy as np
import torch
from torch.nn.functional import affine_grid

opset_version = 20
parser = argparse.ArgumentParser(description="Generate test cases for the AffineGrid operator.")
parser.add_argument("--dim", type=int, choices=[2, 3], help="Dimension of the test cases (2 or 3)")
args = parser.parse_args()

if args.dim is None or args.dim == 2:
    align_corners_options = [False, True]
    angles = [10, 60]
    translations = [np.array([0.3, -0.5]), np.array([-0.5, -0.5])]
    scales = [np.array([1.5, 0.5]), np.array([3.0, 5.5])]
    sizes = [[1, 1, 3, 2], [2, 10, 2, 3]]
    test_count = 0

    for align_corners in align_corners_options:
        for angle, translation, scale in zip(angles, translations, scales):
            for size in sizes:
                theta = np.array([], dtype=np.float32)
                for _ in range(size[0]):
                    angle_radian = (angle / 180.0) * np.pi
                    theta = np.append(
                        theta,
                        [
                            np.cos(angle_radian) * scale[0],
                            -np.sin(angle_radian),
                            translation[0],
                            np.sin(angle_radian),
                            np.cos(angle_radian) * scale[1],
                            translation[1],
                        ],
                    )
                theta = theta.reshape(size[0], 2, 3)
                theta = torch.Tensor(theta)
                grid = affine_grid(theta, size, align_corners=align_corners)

                # Print the C++ code for the test case
                print(f"TEST(AffineGridTest, test_2d_{test_count}) {{")
                print(f'  OpTester test("AffineGrid", {opset_version});')
                print(f'  test.AddAttribute("align_corners", (int64_t){1 if align_corners else 0});')
                print(
                    f"  test.AddInput<float>(\"theta\", {{{theta.shape[0]}, {theta.shape[1]}, {theta.shape[2]}}}, {{{', '.join([f'{x:.6f}f' for x in theta.flatten()])}}});"
                )
                print(
                    f'  test.AddInput<int64_t>("size", {{{len(size)}}}, {{{size[0]}, {size[1]}, {size[2]}, {size[3]}}});'
                )
                print(
                    f"  test.AddOutput<float>(\"grid\", {{{size[0]}, {size[2]}, {size[3]}, 2}}, {{{', '.join([f'{x:.4f}f' for x in grid.flatten()])}}});"
                )
                print("  test.Run();")
                print("}\n")
                test_count += 1


if args.dim is None or args.dim == 3:
    align_corners_options = [False, True]
    angles = [[10, 20], [60, -30]]
    translations = [np.array([0.3, -0.5, 1.8]), np.array([-0.5, -0.5, 0.3])]
    scales = [np.array([1.5, 2.0, 0.5]), np.array([0.3, 3.0, 5.5])]
    sizes = [[1, 1, 3, 2, 2], [2, 10, 2, 2, 3]]
    test_count = 0

    for align_corners in align_corners_options:
        for angle, translation, scale in zip(angles, translations, scales):
            for size in sizes:
                theta = np.array([], dtype=np.float32)
                for _ in range(size[0]):
                    angle_radian_x = (angle[0] / 180.0) * np.pi
                    angle_radian_y = (angle[1] / 180.0) * np.pi
                    rot_matrix_x = np.array(
                        [
                            [1, 0, 0],
                            [0, np.cos(angle_radian_x), -np.sin(angle_radian_x)],
                            [0, np.sin(angle_radian_x), np.cos(angle_radian_x)],
                        ]
                    )
                    rot_matrix_y = np.array(
                        [
                            [np.cos(angle_radian_y), 0, np.sin(angle_radian_y)],
                            [0, 1, 0],
                            [-np.sin(angle_radian_y), 0, np.cos(angle_radian_y)],
                        ]
                    )
                    rot_matrix = np.matmul(rot_matrix_x, rot_matrix_y)
                    rot_matrix = rot_matrix * scale.reshape(3, 1)
                    rot_matrix = np.append(rot_matrix, np.reshape(translation, (3, 1)), axis=1)
                    theta = np.append(theta, rot_matrix.flatten())
                theta = theta.reshape(size[0], 3, 4)
                theta = torch.Tensor(theta)
                grid = affine_grid(theta, size, align_corners=align_corners)

                # Print the C++ code for the test case
                print(f"TEST(AffineGridTest, test_3d_{test_count}) {{")
                print(f'  OpTester test("AffineGrid", {opset_version});')
                print(f'  test.AddAttribute("align_corners", (int64_t){1 if align_corners else 0});')
                print(
                    f"  test.AddInput<float>(\"theta\", {{{theta.shape[0]}, {theta.shape[1]}, {theta.shape[2]}}}, {{{', '.join([f'{x:.6f}f' for x in theta.flatten()])}}});"
                )
                print(
                    f'  test.AddInput<int64_t>("size", {{{len(size)}}}, {{{size[0]}, {size[1]}, {size[2]}, {size[3]}, {size[4]}}});'
                )
                print(
                    f"  test.AddOutput<float>(\"grid\", {{{size[0]}, {size[2]}, {size[3]}, {size[4]}, 3}}, {{{', '.join([f'{x:.4f}f' for x in grid.flatten()])}}});"
                )
                print("  test.Run();")
                print("}\n")
                test_count += 1

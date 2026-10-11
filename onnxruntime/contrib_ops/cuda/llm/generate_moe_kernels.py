#!/usr/bin/env python3
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License.
#
# Generate MoE GEMM kernels for SM80+:
#   python generate_moe_kernels.py -a "80;90" -o ./moe_gemm/launchers

import argparse
import glob
import os
from itertools import product

# CUDA type names for SM80 fused kernels.
CudaTypeName = {
    "bf16": "cutlass::bfloat16_t",
    "f16": "cutlass::half_t",
    "f32": "float",
}

# CUDA type names for SM90 TMA WS kernels (uses raw CUDA types)
CudaTypeNameSm90 = {
    "bf16": "SafeBF16",  # Alias defined in moe_gemm_tma_ws_launcher.inl
    "f16": "half",
    "f32": "float",
}

# Epilogue tags for SM80 fused kernels
EpilogueTagType = {
    "silu": "onnxruntime::llm::cutlass_extensions::EpilogueOpDefaultSilu",
    "gelu": "onnxruntime::llm::cutlass_extensions::EpilogueOpDefaultFtGelu",
}

# Epilogue tags for SM90 TMA WS kernels (must be EpilogueOpDefault for TMA WS)
EpilogueTagSm90 = {
    "default": "EpilogueOpDefault",
}

# Fusion types for SM90
FusionTypes = {
    "none": "NONE",
    "finalize": "FINALIZE",
}


def get_sm80_moe_template_instantiation(element_type, weight_type, tile_m, tile_n, tile_k, stages, epilogue_tag):
    """Generate a template instantiation for sm80_generic_fused_moe_gemm_kernelLauncher."""
    elem_cuda = CudaTypeName[element_type]
    weight_cuda = CudaTypeName[weight_type]
    epi_tag = EpilogueTagType[epilogue_tag]

    return f"""template void sm80_generic_fused_moe_gemm_kernelLauncher<{elem_cuda}, {weight_cuda}, {tile_m}, {tile_n}, {tile_k}, {stages}, {epi_tag}>(
    {elem_cuda} const*, {weight_cuda} const*, {elem_cuda} const*, bool, {elem_cuda}*,
    int64_t const*, int64_t, int64_t, int64_t, int, int, cudaStream_t, int*);
"""


def get_sm90_tma_ws_instantiation(
    arch_tag,
    dtype,
    weight_type,
    output_type,
    epi_tag,
    fusion,
    cta_m,
    cta_n,
    cta_k,
    cga_m,
    cga_n,
    cga_k,
    is_mxfpx,
    has_bias,
):
    """Generate an INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM macro call for SM90+."""
    dtype_cuda = CudaTypeNameSm90[dtype]
    weight_cuda = CudaTypeNameSm90[weight_type]
    output_cuda = CudaTypeNameSm90[output_type]
    epi = EpilogueTagSm90[epi_tag]
    fuse = FusionTypes[fusion]
    mxfpx = "true" if is_mxfpx else "false"
    bias = "true" if has_bias else "false"

    return f"INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM({arch_tag}, {dtype_cuda}, {weight_cuda}, {output_cuda}, {epi}, {fuse}, {cta_m}, {cta_n}, {cta_k}, {cga_m}, {cga_n}, {cga_k}, {mxfpx}, {bias})"


def generate_sm80_moe_operations():
    """Generate all SM80 MoE template instantiations."""
    operations = []

    # Data types: activation = weight (same type for fp16/bf16)
    data_types = [
        ("f16", "f16"),  # FP16
        ("bf16", "bf16"),  # BF16
    ]

    # Tile shapes: (MaxTileM, TileN, TileK)
    # From TensorRT-LLM generate_sm80_fused_grouped_gemm_operations():
    # cta_shapes_mnk = [(16, 128, 64), (16, 256, 64), (32, 128, 64), (64, 128, 64), (128, 128, 64)]
    # For gated activations TileN is halved internally, so 128->64, 256->128
    tile_shapes = [
        (16, 128, 64),
        (16, 256, 64),
        (32, 128, 64),
        (64, 128, 64),
        (128, 128, 64),
    ]

    # Stages from TRT-LLM: [2, 3, 4]
    stages_list = [2, 3, 4]

    # Epilogue tags for SwiGLU activation
    epilogue_tags = ["silu", "gelu"]

    for (elem_type, weight_type), (tile_m, tile_n, tile_k), stages, epi_tag in product(
        data_types, tile_shapes, stages_list, epilogue_tags
    ):
        operations.append(
            {
                "element_type": elem_type,
                "weight_type": weight_type,
                "tile_m": tile_m,
                "tile_n": tile_n,
                "tile_k": tile_k,
                "stages": stages,
                "epilogue_tag": epi_tag,
            }
        )

    return operations


def generate_sm90_tma_ws_operations():
    """Generate SM90 TMA Warp Specialized Grouped GEMM operations.

    Based on TensorRT-LLM's generate_sm90_grouped_gemm_operations().
    """
    operations = []

    # Data types
    data_types = [
        ("f16", "f16", "f16"),  # FP16
        ("bf16", "bf16", "bf16"),  # BF16
    ]

    # CTA shapes: M must be 128 for grouped GEMM
    # From TRT-LLM: M_TILES = [128], N_TILES = [16, 32, 64, 128, 256]
    m_tiles = [128]
    n_tiles = [16, 32, 64, 128, 256]
    cta_shapes_mn = [*list(product(m_tiles, n_tiles)), (256, 128)]

    # CGA (Cluster) shapes
    cga_shapes = list(product([1, 2], [1, 2], [1]))

    # Fusion types - SM90 supports fused finalize
    fusions = ["none", "finalize"]

    for (dtype, wtype, otype), (cta_m, cta_n), (cga_m, cga_n, cga_k), fusion in product(
        data_types, cta_shapes_mn, cga_shapes, fusions
    ):
        # Calculate K based on data type (128 bits / element size)
        bits_per_element = 16  # fp16 and bf16 are 16 bits
        cta_k = 128 * 8 // bits_per_element  # = 64

        operations.append(
            {
                "arch_tag": "Sm90",
                "dtype": dtype,
                "weight_type": wtype,
                "output_type": otype,
                "epi_tag": "default",
                "fusion": fusion,
                "cta_m": cta_m,
                "cta_n": cta_n,
                "cta_k": cta_k,
                "cga_m": cga_m,
                "cga_n": cga_n,
                "cga_k": cga_k,
                "is_mxfpx": False,
                "has_bias": False,
            }
        )

    return operations


def get_sm80_file_content(operations, arch):
    """Generate the content for a SM80 generated .cu file."""
    assert operations

    instantiations = []
    for op in operations:
        inst = get_sm80_moe_template_instantiation(
            op["element_type"],
            op["weight_type"],
            op["tile_m"],
            op["tile_n"],
            op["tile_k"],
            op["stages"],
            op["epilogue_tag"],
        )
        instantiations.append(inst)

    instantiation_block = "\n".join(instantiations)

    # Determine the exclusion guard based on arch
    exclude_macro = f"EXCLUDE_SM_{arch}"

    file_content = f"""/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 *
 * Auto-generated MoE GEMM kernel instantiations for SM{arch}.
 * DO NOT EDIT MANUALLY.
 */

#ifndef {exclude_macro}
#include "contrib_ops/cuda/llm/moe_gemm/launchers/fused_moe_gemm_launcher_sm80.inl"

namespace onnxruntime::llm::kernels::cutlass_kernels {{

#ifdef ENABLE_BF16
{instantiation_block}
#else
// BF16 not enabled, only instantiate FP16 variants
{get_fp16_only_instantiations(operations)}
#endif

}}  // namespace onnxruntime::llm::kernels::cutlass_kernels
#endif  // {exclude_macro}
"""
    return file_content


def get_sm90_file_content(operations, arch, dtype):
    """Generate the content for a SM90 TMA WS generated .cu file."""
    assert operations

    instantiations = []
    for op in operations:
        inst = get_sm90_tma_ws_instantiation(
            op["arch_tag"],
            op["dtype"],
            op["weight_type"],
            op["output_type"],
            op["epi_tag"],
            op["fusion"],
            op["cta_m"],
            op["cta_n"],
            op["cta_k"],
            op["cga_m"],
            op["cga_n"],
            op["cga_k"],
            op["is_mxfpx"],
            op["has_bias"],
        )
        instantiations.append(inst)

    instantiation_block = "\n".join(instantiations)

    file_content = f"""/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 *
 * Auto-generated SM90 TMA Warp Specialized Grouped GEMM instantiations for {dtype.upper()}.
 * DO NOT EDIT MANUALLY.
 */

#ifndef EXCLUDE_SM_90
#ifdef COMPILE_HOPPER_TMA_GROUPED_GEMMS

#include "contrib_ops/cuda/llm/moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl"

namespace onnxruntime::llm::kernels::cutlass_kernels {{

{instantiation_block}

}}  // namespace onnxruntime::llm::kernels::cutlass_kernels

#endif  // COMPILE_HOPPER_TMA_GROUPED_GEMMS
#endif  // EXCLUDE_SM_90
"""
    return file_content


def get_fp16_only_instantiations(operations):
    """Generate instantiations for FP16 only."""
    fp16_ops = [op for op in operations if op["element_type"] == "f16"]
    instantiations = []
    for op in fp16_ops:
        inst = get_sm80_moe_template_instantiation(
            op["element_type"],
            op["weight_type"],
            op["tile_m"],
            op["tile_n"],
            op["tile_k"],
            op["stages"],
            op["epilogue_tag"],
        )
        instantiations.append(inst)
    return "\n".join(instantiations)


def write_file(content, output_file):
    """Write the generated content to a file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Avoid changing modified time if file content is up to date
    if os.path.exists(output_file):
        with open(output_file) as f:
            if f.read() == content:
                print(f"File {output_file} is up to date")
                return

    with open(output_file, mode="w") as f:
        f.write(content)
    print(f"Generated {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MoE GEMM kernel instantiations")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument(
        "-a", "--architectures", type=str, required=True, help="Architectures to generate kernels for (e.g., '80;90')"
    )

    args = parser.parse_args()

    arches = args.architectures.split(";")
    output_dir = os.path.abspath(args.output_dir)

    def has_arch(sm):
        return f"{sm}" in arches or f"{sm}-real" in arches

    # Generate SM80 MoE kernels (fused gated activations)
    if has_arch(80) or has_arch(90):  # SM90 also uses SM80 kernels for non-TMA path
        operations = generate_sm80_moe_operations()

        # Split by (element type, tile shape) into separate files. Each SM80 template
        # instantiation is a full CUTLASS GEMM kernel that nvcc compiles serially within
        # a translation unit, so packing all of them into one file per dtype creates a
        # long single-file critical path. Emitting one file per tile shape lets the build
        # system (Ninja) compile these kernels in parallel and shrinks the slowest object.
        groups = {}
        for op in operations:
            key = (op["element_type"], op["tile_m"], op["tile_n"], op["tile_k"])
            groups.setdefault(key, []).append(op)

        current_files = set()
        for (dtype, tile_m, tile_n, tile_k), ops in groups.items():
            file_name = f"fused_moe_gemm_sm80_{dtype}_m{tile_m}_n{tile_n}_k{tile_k}.generated.cu"
            current_files.add(file_name)
            output_file = os.path.join(output_dir, file_name)
            content = get_sm80_file_content(ops, 80)
            write_file(content, output_file)

        # Remove stale SM80 generated files (e.g. the old monolithic per-dtype files or
        # split files from a previous tile configuration) so they are not compiled.
        for stale in glob.glob(os.path.join(output_dir, "fused_moe_gemm_sm80_*.generated.cu")):
            if os.path.basename(stale) not in current_files:
                os.remove(stale)
                print(f"Removed stale {stale}")

    # Generate SM90 TMA Warp Specialized Grouped GEMM kernels
    if has_arch(90):
        operations = generate_sm90_tma_ws_operations()

        # Group by dtype for separate files
        groups = {}
        for op in operations:
            key = op["dtype"]
            if key not in groups:
                groups[key] = []
            groups[key].append(op)

        for dtype, ops in groups.items():
            output_file = os.path.join(output_dir, f"moe_gemm_tma_ws_sm90_{dtype}.generated.cu")
            content = get_sm90_file_content(ops, 90, dtype)
            write_file(content, output_file)

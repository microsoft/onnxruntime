#  Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# Generate fpA intB GEMM kernels:
#   pip install nvidia-cutlass
#   python generate_kernels.py -a "90" -o  ./fpA_intB_gemm/launchers

import argparse
import enum
import os
from itertools import product

from cutlass_library import (
    DataType,
    DataTypeNames,
    DataTypeSize,
    DataTypeTag,
    EpilogueScheduleSuffixes,
    EpilogueScheduleTag,
    EpilogueScheduleType,
    GemmKind,
    GemmKindNames,
    KernelScheduleSuffixes,
    KernelScheduleTag,
    KernelScheduleType,
)


################################################################################
# Epilogue Tag enum and string utils
class LlmEpilogueTag(enum.Enum):
    epilogue_op_default = enum.auto()
    epilogue_op_bias = enum.auto()
    epilogue_op_silu = enum.auto()
    epilogue_op_gelu = enum.auto()


class LlmEpilogueFusion(enum.Enum):
    epilogue_fusion_none = enum.auto()
    epilogue_fusion_finalize = enum.auto()


EpiTagNames = {
    LlmEpilogueTag.epilogue_op_default: "lc",  # linear combination
    LlmEpilogueTag.epilogue_op_bias: "lc_bias",  # linear combination with bias addition
    LlmEpilogueTag.epilogue_op_silu: "silu",  # silu or swiglu
    LlmEpilogueTag.epilogue_op_gelu: "gelu",  # gelu or geglu
}

EpiTag = {
    LlmEpilogueTag.epilogue_op_default: "onnxruntime::llm::cutlass_extensions::EpilogueOpDefault",
    LlmEpilogueTag.epilogue_op_bias: "onnxruntime::llm::cutlass_extensions::EpilogueOpBias",
    LlmEpilogueTag.epilogue_op_silu: "onnxruntime::llm::cutlass_extensions::EpilogueOpDefaultSilu",
    LlmEpilogueTag.epilogue_op_gelu: "onnxruntime::llm::cutlass_extensions::EpilogueOpDefaultFtGelu",
}

EpiFusion = {
    LlmEpilogueFusion.epilogue_fusion_none: "onnxruntime::llm::TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE",
    LlmEpilogueFusion.epilogue_fusion_finalize: "onnxruntime::llm::TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE",
}

EpiFusionSuffixes = {
    None: "",
    LlmEpilogueFusion.epilogue_fusion_none: "EpilogueFusion_NONE",
    LlmEpilogueFusion.epilogue_fusion_finalize: "EpilogueFusion_FINALIZE",
}


################################################################################
# Quantization Operation and string utils
class LlmQuantOp(enum.Enum):
    per_column_scale_only = enum.auto()
    finegrained_scale_only = enum.auto()
    finegrained_scale_and_zeros = enum.auto()
    none = enum.auto()


QuantOpNames = {
    LlmQuantOp.per_column_scale_only: "cs",
    LlmQuantOp.finegrained_scale_only: "fgs",
    LlmQuantOp.finegrained_scale_and_zeros: "fgsz",
    LlmQuantOp.none: "noquant",
}

QuantOpTag = {
    LlmQuantOp.per_column_scale_only: "cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY",
    LlmQuantOp.finegrained_scale_only: "cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY",
    LlmQuantOp.finegrained_scale_and_zeros: "cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS",
    LlmQuantOp.none: "void",
}

################################################################################
# The activations, biases, scales and zeros are instantiated using CUDA types,
# not CUTLASS types. This map materializes the name of the CUDA type.


def get_data_type_bits(type):
    return DataTypeSize[type]


def get_data_type_names(type):
    return DataTypeNames[type]


CudaTypeName = {
    DataType.e4m3: "__nv_fp8_e4m3",
    DataType.bf16: "__nv_bfloat16",
    DataType.f16: "half",
    DataType.f32: "float",
}


################################################################################
# A data structure holding all info to instantiate gemm launchers in TRT LLM.
class LlmGemmLauncher:
    def __init__(
        self,
        gemm_kind,
        arch,
        act_type,
        weight_type,
        scalezero_type,
        bias_type,
        output_type,
        quant_op,
        epi_tag,
        cta_shape,
        warp_shape,
        stages,
        cga_shape,
        mainloop_schedule,
        epi_schedule,
        epi_fusion=None,
    ):
        self.gemm_kind = gemm_kind
        self.arch = arch
        self.act_type = act_type
        self.weight_type = weight_type
        self.scalezero_type = scalezero_type
        self.bias_type = bias_type
        self.output_type = output_type
        self.quant_op = quant_op
        self.epi_tag = epi_tag
        self.cta_shape = cta_shape
        self.warp_shape = warp_shape
        self.stages = stages
        self.cga_shape = cga_shape
        self.mainloop_schedule = mainloop_schedule
        self.epi_schedule = epi_schedule
        self.epi_fusion = epi_fusion

    def __repr__(self):
        kernel_prefix = f"{GemmKindNames[self.gemm_kind]}_sm{self.arch}_{get_data_type_names(self.act_type)}_{get_data_type_names(self.weight_type)}_{get_data_type_names(self.scalezero_type)}_{get_data_type_names(self.bias_type)}_{get_data_type_names(self.output_type)}_{QuantOpNames[self.quant_op]}_{EpiTagNames[self.epi_tag]}_{self.cta_shape[0]}x{self.cta_shape[1]}x{self.cta_shape[2]}_{self.warp_shape[0]}x{self.warp_shape[1]}x{self.warp_shape[2]}_{self.stages}"

        hopper_suffix = f"_{self.cga_shape[0]}x{self.cga_shape[1]}x{self.cga_shape[2]}{KernelScheduleSuffixes[self.mainloop_schedule]}{EpilogueScheduleSuffixes[self.epi_schedule]}{EpiFusionSuffixes[self.epi_fusion]}"

        if self.arch >= 90:
            return kernel_prefix + hopper_suffix
        elif self.arch > 100:
            raise ValueError(f"SM{self.arch} not supported yet.")
        return kernel_prefix


################################################################################
def tuple_to_cute_shape(shape):
    return f"cute::Shape<cute::Int<{shape[0]}>, cute::Int<{shape[1]}>, cute::Int<{shape[2]}>>"


def instantiate_operation_tma_warp_specialized(operation):
    act_tag = CudaTypeName[operation.act_type]
    scale_zero_tag = CudaTypeName[operation.scalezero_type]
    bias_tag = CudaTypeName[operation.bias_type]
    out_tag = CudaTypeName[operation.output_type]

    quant_op = QuantOpTag[operation.quant_op]
    epi_tag = EpiTag[operation.epi_tag]

    cute_cta_shape = tuple_to_cute_shape(operation.cta_shape)
    cute_cga_shape = tuple_to_cute_shape(operation.cga_shape)

    kernel_sched = KernelScheduleTag[operation.mainloop_schedule]
    epi_sched = EpilogueScheduleTag[operation.epi_schedule]

    assert operation.gemm_kind == GemmKind.Gemm
    weight_tag = DataTypeTag[operation.weight_type]

    return f"""
template void sm90_generic_mixed_gemm_kernelLauncher<{act_tag}, {weight_tag}, {scale_zero_tag}, {bias_tag}, {out_tag},
{quant_op}, {epi_tag},
{cute_cta_shape}, {cute_cga_shape},
{kernel_sched}, {epi_sched}> (
const {act_tag}*, const {weight_tag}*, const {scale_zero_tag}*, const {scale_zero_tag}*, const {bias_tag}*, const float,
{out_tag}*, int, int, int, const int, onnxruntime::llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);
"""


def instantiate_operation(insts_list, operation):
    if operation.arch >= 90:
        insts_list.append(instantiate_operation_tma_warp_specialized(operation))


def get_file_content(launcher_inl_files, operations):
    assert operations
    include_list = list()
    for file in launcher_inl_files:
        include_list.append(f'#include "{file}"')
    includes = "\n".join(include_list)

    insts_list = list()
    for op in operations:
        instantiate_operation(insts_list, op)
    instantiations = "\n".join(insts_list)

    file_content = f"""{includes}
namespace onnxruntime::llm
{{
namespace kernels
{{
namespace cutlass_kernels
{{

{instantiations}

}} // namespace cutlass_kernels
}} // namespace kernels
}} // namespace onnxruntime::llm
"""
    return file_content


def write_file(launcher_inl_files, operations, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # Avoid changing modified time if file content is up to date
    content = get_file_content(launcher_inl_files, operations)
    if os.path.exists(output_file):
        with open(output_file) as f:
            if f.read() == content:
                return
    with open(output_file, mode="w") as f:
        f.write(content)


def elementwise(x, y, f):
    return tuple(f(a, b) for (a, b) in zip(x, y, strict=False))


def is_gemm_op_valid(op):
    tile_m, tile_n, _ = op.cta_shape
    cga_m, cga_n, _ = op.cga_shape

    if cga_m == 1 and cga_n == 1:
        return True

    if cga_m == 2 and cga_n == 1 and tile_m >= 128:
        return True

    if cga_m == 1 and cga_n == 2 and tile_n >= 128:
        return True

    if cga_m == 2 and cga_n == 2 and tile_m >= 128 and tile_n >= 128:
        return True

    return False


################################################################################
def generate_sm90_mixed_gemm_operations(enable_fp8=False):
    arch = 90

    # For legacy reasons, we use unsigned types for the weights. The instanitated template
    # will remap those back to the signed type.
    # Takes the form (activation_type, weight_type, scalezero_type, bias_type, output_type)
    supported_dtypes = [
        (DataType.f16, DataType.u4, DataType.f16, DataType.f16, DataType.f16),
        (DataType.f16, DataType.u8, DataType.f16, DataType.f16, DataType.f16),
        (DataType.bf16, DataType.u4, DataType.bf16, DataType.bf16, DataType.bf16),
        (DataType.bf16, DataType.u8, DataType.bf16, DataType.bf16, DataType.bf16),
    ]

    if enable_fp8:
        supported_dtypes = [
            *supported_dtypes,
            (DataType.e4m3, DataType.u4, DataType.f16, DataType.f16, DataType.f16),
            (DataType.e4m3, DataType.u4, DataType.f16, DataType.bf16, DataType.bf16),
        ]

    quant_ops = [LlmQuantOp.finegrained_scale_and_zeros, LlmQuantOp.finegrained_scale_only]

    epi_tags = [LlmEpilogueTag.epilogue_op_bias]

    m_tiles = [64, 128]
    n_tiles = [16, 32, 64, 128, 256]
    cta_shapes_mn = product(m_tiles, n_tiles)

    warp_shape = [4, 1, 1]
    stages = 0  # auto

    cga_shapes = product([1, 2], [1, 2], [1])

    partial_args = product(supported_dtypes, quant_ops, epi_tags, cta_shapes_mn, cga_shapes)

    operations = list()
    for dtype_combo, quant_op, epi_tag, cta_shape_mn, cga_shape in partial_args:
        max_k_bits = 128 * 8
        cta_shape_k = max_k_bits // get_data_type_bits(dtype_combo[0])
        cta_shape_mnk = (*cta_shape_mn, cta_shape_k)

        use_coop = cta_shape_mn[0] == 128
        mainloop_schedule = (
            KernelScheduleType.TmaWarpSpecializedCooperative
            if use_coop
            else KernelScheduleType.TmaWarpSpecializedPingpong
        )
        epi_schedule = (
            EpilogueScheduleType.TmaWarpSpecializedCooperative if use_coop else EpilogueScheduleType.TmaWarpSpecialized
        )

        mixed_gemm_operation = LlmGemmLauncher(
            GemmKind.Gemm,
            arch,
            *dtype_combo,
            quant_op,
            epi_tag,
            cta_shape_mnk,
            warp_shape,
            stages,
            cga_shape,
            mainloop_schedule,
            epi_schedule,
        )

        if is_gemm_op_valid(mixed_gemm_operation):
            operations.append(mixed_gemm_operation)

    return operations


def generate_sm90_operations(is_arch_enabled):
    operations = generate_sm90_mixed_gemm_operations()
    return operations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print the output directory")

    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument("-a", "--architectures", type=str, required=True, help="Architectures to generate kernels for")

    args = parser.parse_args()

    arches = args.architectures.split(";")

    output_dir = os.path.abspath(args.output_dir)

    include_map = {
        (GemmKind.Gemm, 90): ["contrib_ops/cuda/llm/fpA_intB_gemm/launchers/fpA_intB_launcher_sm90.inl"],
    }

    def has_arch(sm):
        return f"{sm}" in arches or f"{sm}-real" in arches

    # The goal here is to group kernels with common instantiations together in order to reduce template instantiation overheads.
    # Template instantiation dominates the time in a compilation unit, so it is the most important factor to improve.
    operations = []
    operations += generate_sm90_operations(has_arch(90))

    op_groups = dict()
    for op in operations:
        dict_key = (op.gemm_kind, op.arch, op.cta_shape[0])
        op_group = op_groups.get(dict_key, list())
        op_group.append(op)
        op_groups[dict_key] = op_group

    file_counter = 1
    for key, value in op_groups.items():
        gemm_kind, _, _ = key
        out_file = os.path.join(output_dir, f"fpA_intB_gemm_launcher_{file_counter}.generated.cu")
        write_file(include_map[key[:2]], value, out_file)
        file_counter += 1

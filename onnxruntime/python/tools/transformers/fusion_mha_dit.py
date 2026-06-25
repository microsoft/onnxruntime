# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from logging import getLogger

import numpy as np
from fusion_base import Fusion
from fusion_utils import FusionUtils
from onnx import NodeProto, helper, numpy_helper
from onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionMultiHeadAttentionDiT(Fusion):
    """
    Fuse MultiHeadAttention for Diffusion Transformer (DiT) models like F5-TTS.

    Recognizes attention patterns where Q, K, V are pre-computed (e.g., after RoPE)
    and K is pre-transposed, with optional Cast nodes for mixed-precision (FP16) inference
    and a custom scalar scale factor before Softmax.

    Supported patterns (anchored at Softmax):

        MatMul(Q, K^T) → [Cast(FP16→FP32)] → Mul(scale) → Softmax → [Cast(FP32→FP16)] → MatMul(attn, V)
            → Transpose(perm=0,2,1,3) → Reshape → output

    Where:
        - Q is in BNSH format (post-RoPE or post-projection)
        - K is pre-transposed to BNHS format (via Transpose(perm=0,1,3,2) or natively)
        - V is in BNSH format
        - Scale is an arbitrary scalar constant (e.g., 100.0 for DiT, or 1/sqrt(d_k))
        - Cast nodes are optional (present in FP16 models for FP32 Softmax stability)
    """

    def __init__(self, model: OnnxModel):
        super().__init__(model, fused_op_type="MultiHeadAttention", search_op_types=["Softmax"])

    def get_scale_from_mul(self, mul_node: NodeProto) -> float | None:
        """Extract the scalar scale constant from a Mul node.

        The scale can be in either input[0] or input[1].

        Returns:
            float: the scale value, or None if not found.
        """
        for i in range(2):
            value = self.model.get_constant_value(mul_node.input[i])
            if value is not None:
                if isinstance(value, np.ndarray):
                    if value.size == 1:
                        return float(value.item())
                elif isinstance(value, (int, float)):
                    return float(value)
        return None

    def get_data_input_of_mul(self, mul_node: NodeProto) -> int | None:
        """Determine which input of Mul is the data (non-constant) input.

        Returns:
            int: the input index (0 or 1) of the data input, or None if ambiguous.
        """
        is_scalar_constant = [False, False]
        for i in range(2):
            value = self.model.get_constant_value(mul_node.input[i])
            if value is not None:
                if isinstance(value, np.ndarray):
                    is_scalar_constant[i] = value.size == 1
                elif isinstance(value, (int, float)):
                    is_scalar_constant[i] = True

        if is_scalar_constant[0] and not is_scalar_constant[1]:
            return 1
        if is_scalar_constant[1] and not is_scalar_constant[0]:
            return 0
        return None

    def detect_num_heads(self, tensor_name: str, output_name_to_node: dict) -> int:
        """Detect num_heads by walking upstream from a BNSH tensor looking for a Reshape node.

        Typical upstream patterns:
            Reshape(shape=[B, S, N, H]) → Transpose(perm=0,2,1,3) → ... → tensor_BNSH
            Reshape(shape=Concat(..., N, H)) → Transpose(perm=0,2,1,3) → ... → tensor_BNSH

        Returns:
            int: number of heads, or 0 if not detected.
        """
        if tensor_name not in output_name_to_node:
            return 0

        # Walk upstream looking for Transpose → Reshape pattern (possibly with other ops between)
        current = output_name_to_node[tensor_name]
        depth = 0
        max_depth = 10  # Don't walk too far

        while current is not None and depth < max_depth:
            if current.op_type == "Transpose":
                perm = OnnxModel.get_node_attribute(current, "perm")
                if perm == [0, 2, 1, 3]:
                    # Found the BSNH → BNSH transpose, look for Reshape before it
                    if current.input[0] in output_name_to_node:
                        parent = output_name_to_node[current.input[0]]
                        num_heads = self._get_num_heads_from_reshape(parent)
                        if num_heads > 0:
                            return num_heads
                    # Also try looking one step further (e.g., through Add/Norm)
                    break

            # Move to the first input
            if current.input[0] in output_name_to_node:
                current = output_name_to_node[current.input[0]]
            else:
                break
            depth += 1

        return 0

    def _get_num_heads_from_reshape(self, node: NodeProto) -> int:
        """Extract num_heads from a Reshape node's shape parameter.

        Handles:
            - Static shape constant: [B, S, num_heads, head_dim]
            - Concat-based shape: Concat([B_dim], [S_dim], [num_heads], [head_dim])
        """
        if node.op_type != "Reshape":
            return 0

        # Try static shape constant
        if len(node.input) >= 2:
            shape_value = self.model.get_constant_value(node.input[1])
            if shape_value is not None and isinstance(shape_value, np.ndarray) and shape_value.size == 4:
                return int(shape_value[2])

        # Try Concat-based shape
        if len(node.input) >= 2 and node.input[1] in {n.output[0] for n in self.model.get_nodes_by_op_type("Concat")}:
            concat_nodes = [n for n in self.model.get_nodes_by_op_type("Concat") if n.output[0] == node.input[1]]
            if concat_nodes and len(concat_nodes[0].input) == 4:
                value = self.model.get_constant_value(concat_nodes[0].input[2])
                if value is not None:
                    if isinstance(value, np.ndarray) and value.size == 1:
                        return int(value.item())

        return 0

    def detect_num_heads_from_input_shape(self, tensor_name: str) -> int:
        """Try to detect num_heads from a BNSH tensor's shape in graph inputs or value_info.

        For BNSH tensors, the N dimension (index 1) is num_heads.
        """
        # Check graph inputs (e.g., when Q/K/V are direct graph inputs)
        for inp in self.model.model.graph.input:
            if inp.name == tensor_name:
                shape = inp.type.tensor_type.shape
                if shape and len(shape.dim) == 4:
                    dim_n = shape.dim[1]
                    if dim_n.dim_value > 0:
                        return dim_n.dim_value

        # Check value_info
        for vi in self.model.model.graph.value_info:
            if vi.name == tensor_name:
                shape = vi.type.tensor_type.shape
                if shape and len(shape.dim) == 4:
                    dim_n = shape.dim[1]
                    if dim_n.dim_value > 0:
                        return dim_n.dim_value
        return 0

    def detect_num_heads_from_output(self, reshape_out: NodeProto, transpose_out: NodeProto) -> int:
        """Try to detect num_heads from the output Transpose's input shape.

        The Transpose converts BNSH -> BSNH. The N dimension gives us num_heads.
        """
        return self.detect_num_heads_from_input_shape(transpose_out.input[0])

    def reshape_to_3d(self, input_name: str, output_name: str) -> str:
        """Add a Reshape node to convert 4D BxSxNxH to 3D BxSxD.

        Args:
            input_name: input name for the 4D tensor of shape BxSxNxH.
            output_name: output name for the 3D tensor of shape BxSxD.

        Returns:
            str: the output name.
        """
        new_dims_name = "bsnh_to_bsd_reshape_dims"
        new_dims = self.model.get_initializer(new_dims_name)
        if new_dims is None:
            new_dims = numpy_helper.from_array(np.array([0, 0, -1], dtype="int64"), name=new_dims_name)
            self.model.add_initializer(new_dims, self.this_graph_name)
        reshape_node = helper.make_node(
            "Reshape",
            inputs=[input_name, new_dims_name],
            outputs=[output_name],
            name=self.model.create_node_name("Reshape"),
        )
        self.nodes_to_add.append(reshape_node)
        self.node_name_to_graph_name[reshape_node.name] = self.this_graph_name
        return output_name

    def transpose_bnsh_to_bsnh(self, input_name: str) -> str:
        """Add a Transpose node to convert BNSH to BSNH format."""
        output_name = input_name + "_BSNH"
        transpose_node = helper.make_node(
            "Transpose",
            [input_name],
            [output_name],
            name=self.model.create_node_name("Transpose", name_prefix="Transpose_BNSH_to_BSNH"),
            perm=[0, 2, 1, 3],
        )
        self.nodes_to_add.append(transpose_node)
        self.node_name_to_graph_name[transpose_node.name] = self.this_graph_name
        return output_name

    def transpose_bnhs_to_bnsh(self, input_name: str) -> str:
        """Add a Transpose node to convert BNHS to BNSH format."""
        output_name = input_name + "_BNSH"
        transpose_node = helper.make_node(
            "Transpose",
            [input_name],
            [output_name],
            name=self.model.create_node_name("Transpose", name_prefix="Transpose_BNHS_to_BNSH"),
            perm=[0, 1, 3, 2],
        )
        self.nodes_to_add.append(transpose_node)
        self.node_name_to_graph_name[transpose_node.name] = self.this_graph_name
        return output_name

    def create_multihead_attention_node(
        self,
        q: str,
        k: str,
        v: str,
        output: str,
        num_heads: int,
        scale: float | None = None,
    ) -> NodeProto:
        """Create a MultiHeadAttention node.

        Args:
            q: name of query input (BSD format, 3D).
            k: name of key input (BNSH format, 4D).
            v: name of value input (BNSH format, 4D).
            output: output name of MHA.
            num_heads: number of attention heads.
            scale: optional custom scale factor for attention logits.

        Returns:
            NodeProto: the node created.
        """
        assert num_heads > 0

        mha_inputs = [q, k, v]
        mha_outputs = [output]

        mha_node = helper.make_node(
            "MultiHeadAttention",
            inputs=mha_inputs,
            outputs=mha_outputs,
            name=self.model.create_node_name("MultiHeadAttention"),
        )

        mha_node.domain = "com.microsoft"
        mha_node.attribute.extend([helper.make_attribute("num_heads", num_heads)])

        if scale is not None:
            mha_node.attribute.extend([helper.make_attribute("scale", scale)])

        return mha_node

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        assert node.op_type == "Softmax"
        softmax = node

        # Softmax output shall not be graph output.
        if self.model.find_graph_output(softmax.output[0]):
            return

        # MultiHeadAttention normalizes along the last axis. Only fuse when Softmax
        # uses the last axis (default for opset 13+), otherwise semantics would change.
        # For opset < 13, ONNX Softmax defaults axis to 1 when omitted, so an absent
        # axis attribute is NOT safe to fuse — it would silently change semantics.
        axis = OnnxModel.get_node_attribute(softmax, "axis")
        if axis is not None and axis not in (-1, 3):
            return
        if axis is None and self.model.get_opset_version() < 13:
            return

        # ========================================================================
        # Match output path: Softmax → [Cast] → MatMul → Transpose → Reshape
        # ========================================================================
        cast_after_softmax = None

        # Try with Cast first (FP16 models: Softmax → Cast(FP32→FP16) → MatMul → ...)
        child_nodes = self.model.match_child_path(
            softmax,
            ["Cast", "MatMul", "Transpose", "Reshape"],
            [(0, 0), (0, 0), (0, 0), (0, 0)],
            input_name_to_nodes,
        )
        if child_nodes is not None:
            cast_after_softmax, matmul_sv, transpose_out, reshape_out = child_nodes
        else:
            # Try without Cast (FP32 models: Softmax → MatMul → Transpose → Reshape)
            child_nodes = self.model.match_child_path(
                softmax,
                ["MatMul", "Transpose", "Reshape"],
                [(0, 0), (0, 0), (0, 0)],
                input_name_to_nodes,
            )
            if child_nodes is None:
                return
            matmul_sv, transpose_out, reshape_out = child_nodes

        # Verify the output Transpose is BNSH → BSNH
        if not FusionUtils.check_node_attribute(transpose_out, "perm", [0, 2, 1, 3]):
            return

        # ========================================================================
        # Match input path: MatMul → [Cast] → Mul → Softmax
        # ========================================================================
        cast_before_softmax = None

        # Try: Softmax ← Mul ← Cast ← MatMul (FP16 model)
        parent_nodes = self.model.match_parent_path(
            softmax,
            ["Mul", "Cast", "MatMul"],
            [0, None, 0],
        )
        if parent_nodes is not None:
            mul_scale, cast_before_softmax, matmul_qk = parent_nodes
        else:
            # Try: Softmax ← Mul ← MatMul (FP32 model)
            parent_nodes = self.model.match_parent_path(
                softmax,
                ["Mul", "MatMul"],
                [0, None],
            )
            if parent_nodes is None:
                return
            mul_scale, matmul_qk = parent_nodes

        # ========================================================================
        # Extract scale from Mul
        # ========================================================================
        scale = self.get_scale_from_mul(mul_scale)
        if scale is None:
            logger.debug("fuse_dit_attention: failed to extract scale from Mul node")
            return

        # Determine which Mul input is data vs scale constant
        data_input_idx = self.get_data_input_of_mul(mul_scale)
        if data_input_idx is None:
            return

        # Verify the data input connects to the Cast/MatMul
        expected_data_source = cast_before_softmax.output[0] if cast_before_softmax else matmul_qk.output[0]
        if mul_scale.input[data_input_idx] != expected_data_source:
            # Try matching with the other parent path index for Mul
            if cast_before_softmax:
                parent_nodes_alt = self.model.match_parent_path(
                    softmax,
                    ["Mul", "Cast", "MatMul"],
                    [0, 1 - data_input_idx, 0],
                )
                if parent_nodes_alt is None:
                    return
                mul_scale, cast_before_softmax, matmul_qk = parent_nodes_alt
            else:
                parent_nodes_alt = self.model.match_parent_path(
                    softmax,
                    ["Mul", "MatMul"],
                    [0, 1 - data_input_idx],
                )
                if parent_nodes_alt is None:
                    return
                mul_scale, matmul_qk = parent_nodes_alt

        # ========================================================================
        # Get Q, K^T, V
        # ========================================================================
        q_bnsh = matmul_qk.input[0]
        k_transposed_input = matmul_qk.input[1]
        v_bnsh = matmul_sv.input[1]

        # In the cast-wrapped path, V may have been cast to match the post-Softmax
        # attention weights' type (e.g., V cast from FP32 to FP16). The fused MHA op
        # requires Q, K, V to share the same element type (type parameter T), so we
        # trace V back through any Cast that was inserted for type consistency.
        v_traced_through_cast = False
        if cast_after_softmax is not None and v_bnsh in output_name_to_node:
            v_producer = output_name_to_node[v_bnsh]
            if v_producer.op_type == "Cast":
                v_bnsh = v_producer.input[0]
                v_traced_through_cast = True

        # Check if K^T comes from Transpose(perm=0,1,3,2) — if so, use K_BNSH directly
        k_transpose_node = self.model.match_parent(
            matmul_qk, "Transpose", input_index=1, output_name_to_node=output_name_to_node
        )
        if k_transpose_node is not None and FusionUtils.check_node_attribute(k_transpose_node, "perm", [0, 1, 3, 2]):
            k_bnsh = k_transpose_node.input[0]
        else:
            # K is natively in BNHS format, add a Transpose to convert to BNSH
            k_bnsh = self.transpose_bnhs_to_bnsh(k_transposed_input)

        # Trace K back through any Cast to find the source tensor's element type.
        # When K reaches matmul_qk through a casted BNHS tensor (e.g., a Cast FP32→FP16
        # inserted before the pre-transpose), get_dtype(k_bnsh) returns the Cast output
        # type rather than the source type. Tracing through the Cast lets us compare the
        # true source dtype against Q, matching the same pattern used for V above.
        # Note: we do NOT walk through synthesised Transpose nodes (added by
        # transpose_bnhs_to_bnsh) for guard purposes — those have no type annotation and
        # walking through them would make the dtype unresolvable, incorrectly triggering
        # a bail-out on the normal FP16 test path.
        k_traced_through_cast = False
        k_bnsh_for_dtype = k_bnsh
        if k_bnsh_for_dtype in output_name_to_node:
            k_producer = output_name_to_node[k_bnsh_for_dtype]
            if k_producer.op_type == "Cast":
                k_bnsh_for_dtype = k_producer.input[0]
                k_traced_through_cast = True

        # Verify Q, K, V element types are consistent. MHA's type parameter T binds
        # all three inputs to the same type — mismatches produce invalid graphs.
        q_dtype = self.model.get_dtype(q_bnsh)
        k_dtype = self.model.get_dtype(k_bnsh_for_dtype)
        v_dtype = self.model.get_dtype(v_bnsh)
        if q_dtype is not None and v_dtype is not None and q_dtype != v_dtype:
            logger.debug("fuse_dit_attention: Q/V element type mismatch (%s vs %s), skipping", q_dtype, v_dtype)
            return
        if q_dtype is not None and k_dtype is not None and q_dtype != k_dtype:
            logger.debug("fuse_dit_attention: Q/K element type mismatch (%s vs %s), skipping", q_dtype, k_dtype)
            return
        # When casts are present and V was NOT traced back through one, we can't be
        # sure V matches Q/K's type. Bail out if types are unverifiable — proceeding
        # could create a type-invalid MHA (e.g., Q=float32 but V=float16).
        # When V WAS traced back, the trace-back already recovered the pre-cast tensor,
        # so type consistency is structurally guaranteed.
        if (cast_before_softmax is not None or cast_after_softmax is not None) and not v_traced_through_cast:
            if q_dtype is None or v_dtype is None:
                logger.debug(
                    "fuse_dit_attention: cast nodes present, V not traced through Cast, "
                    "types unverifiable (q=%s, v=%s), skipping",
                    q_dtype,
                    v_dtype,
                )
                return
        # Bail out only when K was traced through a Cast (positive evidence of a type
        # conversion on the K path) AND the source dtype is known AND it differs from Q.
        # When K's dtype is simply unresolvable (None) with no Cast on its path, we
        # preserve prior behaviour and do not bail — the dtype-mismatch check above
        # already handles the case where both sides are known and differ.
        if k_traced_through_cast and q_dtype is not None and k_dtype is not None and q_dtype != k_dtype:
            logger.debug(
                "fuse_dit_attention: K Cast source dtype mismatch with Q (%s vs %s), skipping",
                k_dtype,
                q_dtype,
            )
            return

        # ========================================================================
        # Detect num_heads
        # ========================================================================
        num_heads = self.detect_num_heads(q_bnsh, output_name_to_node)
        if num_heads <= 0:
            # Try detecting from V path
            num_heads = self.detect_num_heads(v_bnsh, output_name_to_node)
        if num_heads <= 0:
            # Try detecting from graph input/value_info shapes (e.g., when Q/V are graph inputs)
            num_heads = self.detect_num_heads_from_input_shape(q_bnsh)
        if num_heads <= 0:
            num_heads = self.detect_num_heads_from_input_shape(v_bnsh)
        if num_heads <= 0:
            # Try detecting from output shape info
            num_heads = self.detect_num_heads_from_output(reshape_out, transpose_out)
        if num_heads <= 0:
            logger.debug("fuse_dit_attention: failed to detect num_heads")
            return

        # ========================================================================
        # Convert Q from BNSH to BSD (required by MHA op)
        # ========================================================================
        q_bsnh = self.transpose_bnsh_to_bsnh(q_bnsh)
        q_bsd = self.reshape_to_3d(q_bsnh, q_bsnh + "_BSD")

        # ========================================================================
        # Single-consumer guard: bail out if any matched intermediate tensor feeds
        # another node. Removing multi-consumer intermediates would break the graph.
        # ========================================================================
        intermediate_outputs = [matmul_qk.output[0], mul_scale.output[0], softmax.output[0]]
        if cast_before_softmax is not None:
            intermediate_outputs.append(cast_before_softmax.output[0])
        if cast_after_softmax is not None:
            intermediate_outputs.append(cast_after_softmax.output[0])
        for tensor_name in intermediate_outputs:
            if tensor_name in input_name_to_nodes and len(input_name_to_nodes[tensor_name]) > 1:
                logger.debug("fuse_dit_attention: intermediate %s has multiple consumers, skipping", tensor_name)
                return

        # ========================================================================
        # Create MultiHeadAttention node
        # ========================================================================
        mha_node = self.create_multihead_attention_node(
            q=q_bsd,
            k=k_bnsh,
            v=v_bnsh,
            output=reshape_out.output[0],
            num_heads=num_heads,
            scale=scale,
        )
        self.nodes_to_add.append(mha_node)
        self.node_name_to_graph_name[mha_node.name] = self.this_graph_name

        # Remove fused nodes
        nodes_to_remove = [matmul_sv, transpose_out, reshape_out]
        if cast_after_softmax is not None:
            nodes_to_remove.append(cast_after_softmax)

        # Guard: bail out if any downstream intermediate has consumers outside the
        # nodes we are about to remove. The MHA output (reshape_out.output[0]) is the
        # value that will be produced by the fused node, so it must be kept.
        if not self.model.is_safe_to_fuse_nodes(
            nodes_to_remove, [reshape_out.output[0]], input_name_to_nodes, output_name_to_node
        ):
            logger.debug("fuse_dit_attention: downstream nodes have external consumers, skipping")
            return

        self.nodes_to_remove.extend(nodes_to_remove)

        # Use prune graph to remove remaining unreferenced nodes
        self.prune_graph = True

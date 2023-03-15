import onnx
import packaging.version as pv
from onnx import ModelProto, TensorProto, helper

from python.tools.quantization.onnx_model_converter_base import ConverterBase


class FP16ConverterFinal(ConverterBase):
    def __init__(self, model=None, allow_list=None):
        super().__init__(model, allow_list)

    def process(self, keep_io_types=True):
        self.convert(self.model, keep_io_types)
        return None

    def convert(
        self,
        model: ModelProto,
        keep_io_types=False,
        disable_shape_infer=False,
        force_fp16_initializers=False,
    ) -> ModelProto:
        """
        Convert tensor float type in the ONNX ModelProto input to tensor

        :param model: ONNX ModelProto object
        :param keep_io_types: If True, keep the original input/output tensor types.
        :param disable_shape_infer: Type/shape information is needed for conversion to work.
                                    Set to True only if the model already has type/shape information for all tensors.
        :return: converted ONNX ModelProto object

        """

        func_infer_shape = None
        if not disable_shape_infer and pæv.Version(onnx.__version__) >= pv.Version("1.2.0"):
            try:
                from onnx.shape_inference import infer_shapes

                func_infer_shape = infer_shapes
            finally:
                pass
        if not isinstance(model, ModelProto):
            raise ValueError("Expected model type is an ONNX ModelProto but got %s" % type(model))

        # create a queue for BFS
        queue = []
        value_info_list = []
        node_list = []
        # type inference on input model
        if func_infer_shape is not None:
            model = func_infer_shape(model)
        queue.append(model)
        name_mapping = {}
        graph_io_to_skip = set()
        io_casts = set()
        graph_inputs = [n.name for n in model.graph.input if n.type.tensor_type.elem_type == TensorProto.FLOAT]
        graph_outputs = [n.name for n in model.graph.output if n.type.tensor_type.elem_type == TensorProto.FLOAT]

        if keep_io_types:
            for i, graph_input in enumerate(model.graph.input):  # checking graph inputs
                if graph_input.name in graph_inputs:
                    output_name = "graph_input_cast_" + str(i)
                    name_mapping[graph_input.name] = output_name
                    graph_io_to_skip.add(graph_input.name)

                    node_name = "graph_input_cast" + str(i)
                    new_value_info = model.graph.value_info.add()
                    new_value_info.CopyFrom(graph_input)
                    new_value_info.name = output_name
                    new_value_info.type.tensor_type.elem_type = TensorProto.FLOAT16
                    # add Cast node (from tensor(float) to tensor(float16) after graph input
                    new_node = [
                        helper.make_node(
                            "Cast",
                            [graph_input.name],
                            [output_name],
                            to=TensorProto.FLOAT16,
                            name=node_name,
                        )
                    ]
                    model.graph.node.extend(new_node)
                    value_info_list.append(new_value_info)
                    io_casts.add(node_name)

            for i, graph_output in enumerate(model.graph.output):
                if graph_output.name in graph_outputs:
                    input_name = "graph_output_cast_" + str(i)
                    name_mapping[graph_output.name] = input_name
                    graph_io_to_skip.add(graph_output.name)

                    node_name = "graph_output_cast" + str(i)
                    # add Cast node (from tensor(float16) to tensor(float) before graph output
                    new_value_info = model.graph.value_info.add()
                    new_value_info.CopyFrom(graph_output)
                    new_value_info.name = input_name
                    new_value_info.type.tensor_type.elem_type = TensorProto.FLOAT16
                    new_node = [
                        helper.make_node(
                            "Cast",
                            [input_name],
                            [graph_output.name],
                            to=TensorProto.FLOAT,
                            name=node_name,
                        )
                    ]
                    model.graph.node.extend(new_node)
                    value_info_list.append(new_value_info)
                    io_casts.add(node_name)
        # fp32_initializers: Dict[str, InitializerTracker] = {}
        while queue:
            next_level = []
            for model_ in queue:
                # if model_ is model, push model_.graph (GraphProto)
                if isinstance(model_, ModelProto):
                    next_level.append(model_.graph)
                # if model_ is model.graph, push model_.node.attribute (AttributeProto)
                if isinstance(model_, GraphProto):
                    for initializer in model_.initializer:  # TensorProto type
                        if initializer.data_type == TensorProto.FLOAT:
                            assert initializer.name not in fp32_initializers
                            fp32_initializers[initializer.name] = InitializerTracker(initializer)
                    for node in model_.node:
                        # if node is in the block list (doesn't support float16), no conversion for the node,
                        # and save the node for further processing
                        if node.name in io_casts:
                            continue
                        for i in range(len(node.input)):
                            if node.input[i] in name_mapping:
                                node.input[i] = name_mapping[node.input[i]]
                        for i in range(len(node.output)):
                            if node.output[i] in name_mapping:
                                node.output[i] = name_mapping[node.output[i]]
                        # don't push the attr into queue for the node in node_keep_data_type_list,
                        # so it will not be converted to float16
                        is_node_blocked = node.op_type not in self.allow_list
                        for node_input in node.input:
                            if node_input in fp32_initializers:
                                fp32_initializers[node_input].add_node(node, is_node_blocked)
                        if is_node_blocked:
                            node_list.append(node)
                        else:
                            if node.op_type == "Cast":
                                for attr in node.attribute:
                                    if attr.name == "to" and attr.i == TensorProto.FLOAT:
                                        attr.i = TensorProto.FLOAT16
                                        break
                            for attr in node.attribute:
                                next_level.append(attr)
                # if model_ is model.graph.node.attribute, push model_.g and model_.graphs (GraphProto)
                # and process node.attribute.t and node.attribute.tensors (TensorProto)
                if isinstance(model_, AttributeProto):
                    next_level.append(model_.g)
                    for graph in model_.graphs:
                        next_level.append(graph)
                    model_.t.CopyFrom(self._convert_tensor_float_to_float16(model_.t))
                    for tensor in model_.tensors:
                        self._convert_tensor_float_to_float16(tensor)
                # if model_ is graph, process graph.initializer(TensorProto), input, output and value_info (
                # ValueInfoProto)
                if isinstance(model_, GraphProto):
                    for initializer in model_.initializer:  # TensorProto type
                        if initializer.data_type == TensorProto.FLOAT:
                            initializer = self._convert_tensor_float_to_float16(initializer)
                            value_info_list.append(self._make_value_info_from_tensor(initializer))
                    # for all ValueInfoProto with tensor(float) type in input, output and value_info, convert them to
                    # tensor(float16) except map and seq(map). And save them in value_info_list for further processing
                    for val_info in itertools.chain(model_.input, model_.output, model_.value_info):
                        if val_info.type.tensor_type.elem_type == TensorProto.FLOAT:
                            if val_info.name not in graph_io_to_skip:
                                val_info.type.tensor_type.elem_type = TensorProto.FLOAT16
                                value_info_list.append(val_info)
            queue = next_level
        for key, value in fp32_initializers.items():
            # By default, to avoid precision loss, do not convert an initializer to fp16 when it is used only by fp32
            # nodes.
            if force_fp16_initializers or value.fp16_nodes:
                value.initializer = self._convert_tensor_float_to_float16(value.initializer)
                value_info_list.append(self._make_value_info_from_tensor(value.initializer))
                if value.fp32_nodes and not force_fp16_initializers:
                    logger.info(
                        f"initializer is used by both fp32 and fp16 nodes. Consider add these nodes to block list:"
                        f"{value.fp16_nodes}"
                    )

        return model

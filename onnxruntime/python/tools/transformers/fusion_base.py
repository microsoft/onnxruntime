# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from collections import defaultdict
from logging import getLogger
from typing import List, Union

from onnx_model import OnnxModel

logger = getLogger(__name__)


class Fusion:
    def __init__(
        self,
        model: OnnxModel,
        fused_op_type: str,
        search_op_types: Union[str, List[str]],
        description: str = None,
    ):
        self.search_op_types: List[str] = [search_op_types] if isinstance(search_op_types, str) else search_op_types
        self.fused_op_type: str = fused_op_type
        self.description: str = f"{fused_op_type}({description})" if description else fused_op_type
        self.model: OnnxModel = model
        self.nodes_to_remove: List = []
        self.nodes_to_add: List = []
        self.prune_graph: bool = False
        self.node_name_to_graph_name: dict = {}
        self.this_graph_name: str = None
        # It is optional that subclass updates fused_count since we will also check nodes_to_add to get counter.
        self.fused_count: defaultdict = defaultdict(int)

    def increase_counter(self, fused_op_name):
        self.fused_count[fused_op_name] += 1

    def apply(self):
        logger.debug(f"start {self.description} fusion...")
        input_name_to_nodes = self.model.input_name_to_nodes()
        output_name_to_node = self.model.output_name_to_node()

        # This assumes that two search ops will not be fused at same time!
        for search_op_type in self.search_op_types:
            for node in self.model.get_nodes_by_op_type(search_op_type):
                graph = self.model.get_graph_by_node(node)
                if graph is None:
                    raise Exception("Can not find node in any graphs")
                self.this_graph_name = graph.name
                self.fuse(node, input_name_to_nodes, output_name_to_node)

        op_list = [node.op_type for node in self.nodes_to_add]
        if self.fused_count:
            for key, value in self.fused_count.items():
                if value:
                    logger.info(f"Fused {key}: {value}")
        else:
            count = op_list.count(self.fused_op_type)
            if count > 0:
                logger.info(f"Fused {self.description}: {count}")

        self.model.remove_nodes(self.nodes_to_remove)
        self.model.add_nodes(self.nodes_to_add, self.node_name_to_graph_name)

        if self.prune_graph:
            self.model.prune_graph()
        elif self.nodes_to_remove or self.nodes_to_add:
            self.model.update_graph()

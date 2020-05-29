#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------
from logging import getLogger
from OnnxModel import OnnxModel
from typing import Union, List

logger = getLogger(__name__)


class Fusion:
    def __init__(self, model: OnnxModel, name: str, search_op_types: Union[str, List[str]]):
        self.search_op_types: List[str] = [search_op_types] if isinstance(search_op_types, str) else search_op_types
        self.name: str = name
        self.model: OnnxModel = model
        self.nodes_to_remove: List = []
        self.nodes_to_add: List = []
        self.prune_graph: bool = False

    def apply(self):
        logger.debug(f"start {self.name} fusion...")

        input_name_to_nodes = self.model.input_name_to_nodes()
        output_name_to_node = self.model.output_name_to_node()

        # This assumes that two search ops will not be fused at same time!
        for search_op_type in self.search_op_types:
            for node in self.model.get_nodes_by_op_type(search_op_type):
                self.fuse(node, input_name_to_nodes, output_name_to_node)

        logger.info(f"Fused {self.name} count: {len(self.nodes_to_add)}")

        self.model.remove_nodes(self.nodes_to_remove)
        self.model.add_nodes(self.nodes_to_add)

        if self.prune_graph:
            self.model.prune_graph()
        elif self.nodes_to_remove or self.nodes_to_add:
            self.model.update_graph()

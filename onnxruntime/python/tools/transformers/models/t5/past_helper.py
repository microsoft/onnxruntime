# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)


class PastKeyValuesHelper:
    """Helper functions to process past key values for encoder-decoder model"""

    @staticmethod
    def get_past_names(num_layers, present: bool = False):
        past_self_names = []
        past_cross_names = []
        for i in range(num_layers):
            past_self_names.extend(
                [f"present_key_self_{i}", f"present_value_self_{i}"]
                if present
                else [f"past_key_self_{i}", f"past_value_self_{i}"]
            )
            past_cross_names.extend(
                [f"present_key_cross_{i}", f"present_value_cross_{i}"]
                if present
                else [f"past_key_cross_{i}", f"past_value_cross_{i}"]
            )
        return past_self_names + past_cross_names

    @staticmethod
    def group_by_self_or_cross(present_key_values):
        """Split present state from grouped by layer to grouped by self/cross attention.
        Before: (past_key_self_0, past_value_self_0, past_key_cross_0, past_value_cross_0), (past_key_self_1, past_value_self_1, past_key_cross_1, past_value_cross_1), ...
        After: (past_key_self_0, past_value_self_0, past_key_self_1, past_value_self_1, ...), (past_key_cross_0, past_value_cross_0, past_key_cross_1, past_value_cross_1, ...)

        """
        present_self = []
        present_cross = []
        for i, present_layer_i in enumerate(present_key_values):
            assert len(present_layer_i) == 4, f"Expected to have four items. Got {len(present_layer_i)}"
            (
                present_key_self,
                present_value_self,
                present_key_cross,
                present_value_cross,
            ) = present_layer_i
            present_self.extend([present_key_self, present_value_self])
            present_cross.extend([present_key_cross, present_value_cross])
        return present_self, present_cross

    @staticmethod
    def group_by_layer(past, num_layers):
        """Reorder past state from grouped by self/cross attention to grouped by layer.
        Before: past_key_self_0, past_value_self_0, past_key_self_1, past_value_self_1, ..., past_key_cross_0, past_value_cross_0, past_key_cross_1, past_value_cross_1, ...
        After: (past_key_self_0, past_value_self_0, past_key_cross_0, past_value_cross_0), (past_key_self_1, past_value_self_1, past_key_cross_1, past_value_cross_1),
        """
        assert len(past) == 4 * num_layers
        return tuple(
            [
                past[2 * i],
                past[2 * i + 1],
                past[2 * num_layers + 2 * i],
                past[2 * num_layers + 2 * i + 1],
            ]
            for i in range(num_layers)
        )

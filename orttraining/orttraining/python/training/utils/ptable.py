# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from typing import List


class Row:
    """A row in a PTable"""

    def __init__(self, columns: List[str]) -> None:
        self._columns: List[str] = columns  # List of strings
        self._annotation_table = None  # Optional PTable used for displaying detailed information about the feature row.

    def append_annotation_table(self, ptable) -> None:
        self._annotation_table = ptable


class PTable:
    """A table that can be printed to the console."""

    def __init__(self, sortable=False) -> None:
        self._rows: List[Row] = []
        self._column_count = None
        self._sortable = sortable  # allow the rows to be sorted by the first column

    def add_row(self, columns: List[str]) -> Row:
        """Add a row to the table. The number of columns must match the number of columns in the table."""
        if self._column_count is None:
            self._column_count = len(columns)
        assert self._column_count == len(columns)
        row = Row(columns)
        self._rows.append(row)
        return row

    def get_string(self, first_column_width=None, second_column_width=None) -> str:
        """Serialize the table to a string."""
        if len(self._rows) == 0:
            return ""

        # Collect the max width of each column
        column_widths = []
        for row in self._rows:
            if column_widths:
                assert len(column_widths) == len(row._columns)
            else:
                column_widths = [0] * len(row._columns)
            for i, column in enumerate(row._columns):
                column_widths[i] = max(column_widths[i], len(str(column)))

        if first_column_width:
            column_widths[0] = max(first_column_width, column_widths[0])

        if second_column_width:
            column_widths[2] = max(second_column_width, column_widths[2])

        serialized_table = ""
        if self._sortable:
            sorted_rows = sorted(self._rows, key=lambda row: row._columns[0])
        else:
            sorted_rows = self._rows

        for row in sorted_rows:
            for i, column in enumerate(row._columns):
                serialized_table += f"{str(column).ljust(column_widths[i] + 2)}"
            serialized_table += "\n"
            if row._annotation_table:
                serialized_table += row._annotation_table.get_string(
                    first_column_width=column_widths[0], second_column_width=column_widths[2]
                )

        return serialized_table

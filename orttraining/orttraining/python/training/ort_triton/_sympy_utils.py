# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import re
from typing import Any, List

import sympy


def extract_shape_from_symbol(symbol: str) -> int:
    match = re.match(r"i(\d+)_dim(\d+)_(\d+)", symbol)
    assert match
    return int(match.group(3))


def sympy_dot(seq1: List[sympy.Expr], seq2: List[sympy.Expr]) -> sympy.Expr:
    assert len(seq1) == len(seq2)
    return sympy.expand(sum(a * b for a, b in zip(seq1, seq2)))


def parse_shape(shape: List[Any]) -> List[sympy.Expr]:
    symbol_shapes = []
    for dim in shape:
        symbol_dim = dim
        if isinstance(dim, str):
            symbol_dim = sympy.Symbol(re.sub(r"[^a-zA-Z0-9_]+", "_", dim))
        elif isinstance(dim, int):
            symbol_dim = sympy.Integer(dim)
        symbol_shapes.append(symbol_dim)
    return symbol_shapes

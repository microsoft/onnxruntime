import inspect
import os
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union
import torch
from torch._subclasses.fake_tensor import FakeTensorMode


def _catch_produce_guards_and_solve_constraints(
    previous_function: Callable,
    fake_mode: "FakeTensorMode",  # noqa: F821
    gm: "torch.fx.GraphModule",  # noqa: F821
    dynamic_shapes: Union[Dict[str, Any], Tuple[Any], List[Any], None],
    equalities_inputs: "EqualityConstraint",  # noqa: F821
    original_signature: inspect.Signature,
    _is_torch_jit_trace: bool = False,
    verbose: int = 0,
):
    try:
        return previous_function(
            fake_mode=fake_mode,
            gm=gm,
            dynamic_shapes=dynamic_shapes,
            equalities_inputs=equalities_inputs,
            original_signature=original_signature,
            _is_torch_jit_trace=_is_torch_jit_trace,
        )
    except Exception as e:
        if not int(os.environ.get("SKIP_SOLVE_CONSTRAINTS", "1")):
            raise
        if verbose:
            print(
                f"[_catch_produce_guards_and_solve_constraints] ERROR"
                f"produce_guards_and_solve_constraints failed, "
                f"use SKIP_SOLVE_CONSTRAINTS=0 to avoid skipping\n"
                f"fake_mode={fake_mode}\n"
                f"dynamic_shapes={dynamic_shapes}\n"
                f"equalities_inputs={equalities_inputs}\n"
                f"original_signature={original_signature}\n"
                f"_is_torch_jit_trace={_is_torch_jit_trace}\n"
                f"exc={e}\ngm={gm}"
            )


def patch__check_input_constraints_for_graph(
    previous_function: Callable,
    input_placeholders: list[torch.fx.Node],
    flat_args_with_path,
    range_constraints,
    verbose: int = 0,
) -> None:
    try:
        return previous_function(input_placeholders, flat_args_with_path, range_constraints)
    except Exception as e:
        if not int(os.environ.get("SKIP_SOLVE_CONSTRAINTS", "1")):
            raise
        if verbose:
            print(
                f"[_check_input_constraints_for_graph] ERROR"
                f"_check_input_constraints_for_graph failed, "
                f"use SKIP_SOLVE_CONSTRAINTS=0 to avoid skipping\n"
                f"input_placeholders={input_placeholders}\n"
                f"range_constraints={range_constraints}\n"
                f"exc={e}"
            )


def patched_infer_size(a, b):
    """Patches ``torch._subclasses.fake_impls.infer_size``."""
    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

    dimsA = len(a)
    dimsB = len(b)
    ndim = max(dimsA, dimsB)
    expandedSizes = [0] * ndim
    for i in range(ndim - 1, -1, -1):
        offset = ndim - 1 - i
        dimA = dimsA - 1 - offset
        dimB = dimsB - 1 - offset
        sizeA = a[dimA] if dimA >= 0 else 1
        sizeB = b[dimB] if dimB >= 0 else 1

        # NB: It is very important to test for broadcasting, before testing
        # sizeA == sizeB.  This is because the broadcasting tests are likely
        # to be statically known (in particular, if sizeA/sizeB is unbacked
        # but size-like, we will unsoundly assume they never equal 1), but
        # the sizeA == sizeB test may not be statically known.  However, once
        # we have established that no broadcasting is happening, the
        # sizeA == sizeB is now expect_true and we can defer it as a runtime
        # assert (this works because Python will return the terminal
        # expression of an or statement as-is, without bool()'ing it; if this
        # were not the case, we'd need to write this using torch.sym_or() or
        # something like that).
        try:
            b1 = guard_size_oblivious(sizeA == 1)
        except torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode:
            b1 = False
        try:
            b2 = guard_size_oblivious(sizeB == 1)
        except torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode:
            b2 = False
        try:
            b3 = guard_size_oblivious(sizeA == sizeB)
        except torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode:
            b3 = False
        if b1 or b2 or b3:
            expandedSizes[i] = sizeB if guard_size_oblivious(sizeA == 1) else sizeA
        else:
            # In this case, the current implementation of torch fails (17/12/2024).
            # Try model SmolLM.
            expandedSizes[i] = torch.sym_max(sizeA, sizeB)
    return tuple(expandedSizes)


def patched__broadcast_shapes(*_shapes):
    """Patches ``torch._refs._broadcast_shapes``."""
    from functools import reduce
    from torch._prims_common import IntLike
    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

    shapes = tuple(
        (x,) if isinstance(x, IntLike) else x for x in filter(lambda x: x is not None, _shapes)
    )

    # Short-circuits on no input
    if len(shapes) == 0:
        return None

    # Type checking
    # TODO: make common validations available as utils
    for shape in shapes:
        assert isinstance(shape, Sequence)

    # Computes common shape
    common_shape = [  # List[Union[int, torch.SymInt]]
        1,
    ] * reduce(max, (len(shape) for shape in shapes))
    for _arg_idx, shape in enumerate(shapes):
        for idx in range(-1, -1 - len(shape), -1):
            if guard_size_oblivious(common_shape[idx] == 1):
                if shape[idx] < 0:
                    raise ValueError(
                        "Attempting to broadcast a dimension with negative length!"
                    )
                common_shape[idx] = shape[idx]
            elif guard_size_oblivious(shape[idx] != 1):
                common_shape[idx] = torch.sym_max(common_shape[idx], shape[idx])

    return common_shape


class patched_ShapeEnv:

    def _set_replacement(
        self, a: "sympy.Symbol", tgt: "sympy.Expr", msg: str  # noqa: F821
    ) -> None:
        """
        Adds or updates a replacement for a symbol.
        Use this instead of `self.replacements[a] = tgt`.
        """
        if tgt == self.replacements.get(a, None):
            return

        if a in tgt.free_symbols:
            return

        import sympy
        from torch._logging import structured
        from torch.utils._traceback import CapturedTraceback
        from torch._logging import trace_structured
        from torch._guards import TracingContext
        from torch.utils._sympy.functions import FloorToInt, CeilToInt
        from torch.utils._sympy.solve import try_solve
        from torch.fx.experimental.symbolic_shapes import (
            _is_supported_equivalence,
            ValueRanges,
        )

        # Precondition: a == tgt
        assert isinstance(a, sympy.Symbol)

        if self.allow_complex_guards_as_runtime_asserts and not _is_supported_equivalence(tgt):
            # continuing leads to placeholder shapes
            # having complex expressions that we can't resolve
            return

        # Handles nested tensor symbolic variables which don't have
        # var_to_range bounds
        tgt_bound = None
        if a in self.var_to_range:
            src_bound = self.var_to_range[a]

            # First, refine the value range of a based on the computed value range
            # of tgt.  This is always OK to do, even if we decide not to do the
            # substitution in the end.  This might be a no-op, if a already has
            # a tighter bound
            tgt_bound = self.bound_sympy(tgt)
            self._update_var_to_range(a, tgt_bound)

            # Next, check if we can update the range of free symbols in tgt
            # based on the range in a. But only do it if:
            #  - the source bound non-trivially improves over what we get out of
            #    the existing bounds.
            #  - the replacement is univariate and we can invert the tgt expression
            if not tgt_bound.issubset(src_bound) and len(tgt.free_symbols) == 1:
                b = next(iter(tgt.free_symbols))
                # Try to invert the equality
                r = try_solve(sympy.Eq(a, tgt), b, floordiv_inequality=False)
                if r is not None:
                    self.log.debug(
                        "set_replacement: solve for %s in %s == %s gives %s",
                        b,
                        a,
                        tgt,
                        r,
                    )
                    # The solution here can be non-integral, for example, if
                    # we have s0 = 2*s1, then s1 = s0/2.  What we would like
                    # to do is calculated the bounds in arbitrary precision,
                    # and then requantize the bound to integers when we are
                    # done.
                    rat_b_bound = self.bound_sympy(r[1])
                    b_bound = ValueRanges(
                        CeilToInt(rat_b_bound.lower), FloorToInt(rat_b_bound.upper)
                    )
                    self._update_var_to_range(b, b_bound, self.var_to_range_sloc[a])
                    tgt_bound = self.bound_sympy(tgt)
                    assert tgt_bound.issubset(
                        src_bound
                    ), f"{tgt_bound=} not a subset of {src_bound=}"

            # TODO: Should we propagate size-like-ness?
            #
            # Pros: if u0 is size-like, intuitively u0 == u1 should cause u1
            # to become size-like.
            #
            # Cons: if u0 is size-like, what about u0 - 1 == u1?  You CAN'T
            # propagate in this case, because what if u0 == 0, then u1 is negative
            # and clearly isn't a size.  So, at minimum, any f(x) whose value
            # range isn't [0, inf] given x in [0, inf] cannot propagate
            # size-like-ness.  But there are many situations where you could
            # imagine u1 is going to be size-like and actually you just didn't
            # have a refined enough value range on u0.  Since even innocuous
            # looking arithmetic operations can destroy size-like-ness, it's
            # best to not propagate it at all and force the user to annotate it
            # as necessary.
            #
            # Compromise: we preserve size-like-ness only for exact equality
            # and nothing else.
            if a in self.size_like and isinstance(tgt, sympy.Symbol):
                self.size_like.add(tgt)
            elif isinstance(tgt, sympy.Symbol) and tgt in self.size_like:
                self.size_like.add(a)

            # Now, decide if we will do the substitution.
            #
            #  - If the source has a non-trivial range, only substitute if
            #    we preserve this range.  Note that we may have propagated
            #    the src_range to free variables in tgt when tgt is univariate
            #    and we could find an inverse, which helps us achieve this.
            #    This ensures we never "forget" about user defined ranges,
            #    even if they end up being defined on composite formulas
            #    like s0 + s1.
            #
            #  - If the variable is unbacked, only substitute if the substitution
            #    would preserve the bounds also under size-like-ness conditions.

            if not tgt_bound.issubset(src_bound):
                self.log.debug(
                    "skipped set_replacement %s = %s (%s) [%s not subset of %s]",
                    a,
                    tgt,
                    msg,
                    tgt_bound,
                    src_bound,
                )
                return
            elif a in self.size_like:
                tgt_bound_so = self.bound_sympy(tgt, size_oblivious=True)
                src_bound_so = self.bound_sympy(a, size_oblivious=True)
                if not tgt_bound_so.issubset(src_bound_so):
                    self.log.debug(
                        "skipped set_replacement %s = %s (%s) "
                        "[%s not subset of %s (size-oblivious conditions)]",
                        a,
                        tgt,
                        msg,
                        tgt_bound_so,
                        src_bound_so,
                    )
                    return

        if isinstance(tgt, (sympy.Integer, sympy.Float)):
            # specializing to a constant, which is likely unexpected (unless
            # you specified dynamic=True)

            user_tb = TracingContext.extract_stack()
            trace_structured(
                "symbolic_shape_specialization",
                metadata_fn=lambda: {
                    "symbol": repr(a),
                    "sources": [s.name() for s in self.var_to_sources.get(a, [])],
                    "value": repr(tgt),
                    "reason": msg,
                    "stack": structured.from_traceback(
                        CapturedTraceback.extract(skip=1).summary()
                    ),
                    "user_stack": (structured.from_traceback(user_tb) if user_tb else None),
                },
            )

            # if config.print_specializations:
            #    self.log.warning(
            #         "Specializing %s to %s", self.var_to_sources[a][0].name(), tgt
            #     )
            #     self.log.debug("SPECIALIZATION", stack_info=True)
        assert msg != "range_refined_to_singleton", (
            f"A dynamic dimension becomes static! "
            f"a={a!r}, tgt={tgt!r}, msg={msg!r}, tgt_bound={tgt_bound}"
        )
        # log.info("set_replacement %s = %s (%s) %s", a, tgt, msg, tgt_bound)
        self.replacements[a] = tgt
        # NB: the replacement may get refined, but the user will find the
        # FIRST one most useful (TODO: Maybe we could consider tracking all of
        # them)
        if a not in self.replacements_slocs:
            self.replacements_slocs[a] = self._get_sloc()
        self._update_version_counter()

        # When specializing 'a == tgt', the equality should be also conveyed to
        # Z3, in case an expression uses 'a'.
        self._add_target_expr(sympy.Eq(a, tgt, evaluate=False))

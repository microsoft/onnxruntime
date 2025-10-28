# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import datetime
import functools
import logging
import re
import sys
import time
from collections.abc import Callable, Sequence
from typing import Any

from .task import Task

ALL_TASKS: list[str] = []
HIDDEN_TASKS: set[str] = set()
PUBLIC_TASKS: list[str] = []
TASK_DEPENDENCIES: dict[str, list[str]] = {}
TASK_DESCRIPTIONS: dict[str, str] = {}
SUMMARIZERS: list[str] = []


def task(func):
    ALL_TASKS.append(func.__name__)
    return func


def implementation_detail(func):
    HIDDEN_TASKS.add(func.__name__)
    task(func)
    return func


def public_task(description: str):
    def add_task(func):
        PUBLIC_TASKS.append(func.__name__)
        TASK_DESCRIPTIONS[func.__name__] = description
        task(func)
        return func

    return add_task


def depends(deps: Sequence[str | tuple[bool, str | Sequence[str]]]):
    def add_dep(func: Any) -> Any:
        task_deps: list[str] = []
        for dep in deps:
            if isinstance(dep, str):
                task_deps.append(dep)
            elif dep[0]:
                if isinstance(dep[1], str):
                    task_deps.append(dep[1])
                else:
                    task_deps.extend(dep[1])
        TASK_DEPENDENCIES[func.__name__] = task_deps
        return func

    return add_dep


def summarizer(func):
    SUMMARIZERS.append(func.__name__)
    return func


class Step:
    """A named Task within a Plan."""

    def __init__(self, step_id: str, task: Task):
        self._step_id = step_id
        self._task = task

    def __repr__(self) -> str:
        return self._step_id

    @property
    def step_id(self) -> str:
        return self._step_id

    @property
    def task(self) -> Task:
        return self._task


class Plan:
    """An ordered list of Tasks to execute."""

    _steps: list[Step]
    _skips: list[re.Pattern]
    _plan_duration = datetime.timedelta | None
    _step_durations: list[tuple[str, datetime.timedelta]]

    def __init__(self) -> None:
        self._steps = []
        self._skips = []
        # mypy is wrong here:
        self._plan_duration = None  # type: ignore[assignment]
        self._step_durations = []
        self.enable_report = True

    def add_step(self, task: Task, step_id: str | None = None) -> str:
        if step_id is None:
            # Default to the name of the calling function
            step_id = sys._getframe(1).f_code.co_name

        if self.count_step(step_id) > 10:
            raise RuntimeError(
                f"Refusing to add step '{step_id}' more than 10 times. Perhaps the planner is in an infinite loop?"
            )
        self._steps.append(Step(step_id, task))
        return step_id

    def count_step(self, step_id: str) -> int:
        step_count = 0
        for s in self._steps:
            if s.step_id == step_id:
                step_count += 1
        return step_count

    def for_each(self, func: Callable[[str, Task], None]) -> None:
        for s in self._steps:
            func(s.step_id, s.task)

    def has_step(self, step_id: str) -> bool:
        return any(s.step_id == step_id for s in self._steps)

    def is_skipped(self, step_id: str) -> bool:
        return any(r.match(step_id) for r in self._skips)

    def print(self) -> None:
        for step in self._steps:
            step_msg = step.step_id
            if not step.task.does_work():
                step_msg += " (no-op)"
            if self.is_skipped(step.step_id):
                step_msg += " (skipped)"
            print(step_msg)

    def print_report(self) -> None:
        """Print a report on how long steps in the plan took."""

        if not self.enable_report or len(self._step_durations) < 1:
            return

        step_id_lens = [len(s) for s, d in self._step_durations]
        max_step_id_len = functools.reduce(lambda a, b: max(b, a), step_id_lens)  # type: ignore
        print(f"{'Step':^{max_step_id_len}} {'Duration':^14}")
        print(f"{'-':-^{max_step_id_len}} {'-':-^14}")
        for step_id, duration in self._step_durations:
            print(f"{step_id:<{max_step_id_len}} {duration!s:<14}")
        if self._plan_duration:
            print(f"{'-':-^{max_step_id_len}} {'-':-^14}")
            print(f"{'Total':<{max_step_id_len}} {self._plan_duration!s:<14}")

    def run(self) -> None:
        start_time = time.monotonic()

        def run_task(step_id: str, task: Task) -> None:
            if self.is_skipped(step_id):
                logging.warning(f"Skipping {step_id}")
            else:
                step_start_time = time.monotonic()

                caught: Exception | None = None
                try:
                    task.run()
                except Exception as ex:
                    logging.error(f"Error while running {step_id}.")
                    caught = ex
                step_end_time = time.monotonic()
                if task.does_work():
                    self._step_durations.append(
                        (
                            step_id,
                            datetime.timedelta(seconds=step_end_time - step_start_time),
                        )
                    )
                if caught is not None:
                    raise caught

        try:
            self.for_each(run_task)
        finally:
            end_time = time.monotonic()
            # mypy is wrong here:
            self._plan_duration = datetime.timedelta(seconds=end_time - start_time)  # type: ignore[assignment]

    def skip(self, pattern: str) -> None:
        self._skips.append(re.compile(pattern))

    @property
    def steps(self) -> list[str]:
        return [s.step_id for s in self._steps]

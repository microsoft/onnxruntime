# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""
Self-tests for build_and_test.py.
"""

from . import plan


def assert_task_dependencies_exist(task_library: type) -> None:
    # Check ALL_TASKS for consistency
    for task_name in plan.ALL_TASKS:
        task_func = getattr(task_library, task_name, None)
        assert task_func is not None, f"{task_name} is not defined in TaskLibrary"
        assert callable(task_func), f"{task_name} is not a function"
    task_set = set(plan.ALL_TASKS)
    assert len(task_set) == len(plan.ALL_TASKS), "Some task is defined twice"

    # Make sure ancillary task collections at least refer to real tasks
    unknown_publics = [t for t in plan.PUBLIC_TASKS if t not in task_set]
    assert len(unknown_publics) == 0, f"These public tasks are not defined as tasks: {', '.join(unknown_publics)}"
    unknown_descriptions = [t for t in plan.TASK_DESCRIPTIONS if t not in task_set]
    assert len(unknown_publics) == 0, (
        f"These descriptions are associated with undefined tasks: {', '.join(unknown_descriptions)}"
    )

    # Ensure that summarizers exist
    unknown_summarizers = [s for s in plan.SUMMARIZERS if getattr(task_library, s, None) is None]
    assert len(unknown_publics) == 0, (
        f"These summarizers have no implementing function: {', '.join(unknown_summarizers)}"
    )

    # Verify that all dependency keys and values are tasks
    for task, deps in plan.TASK_DEPENDENCIES.items():
        assert task in task_set, f"Task {task} has declared dependencies, but it does not exist"
        unknown_deps = [d for d in deps if d not in task_set]
        assert len(unknown_deps) == 0, f"Task {task} depends on these unknown dependencies: {', '.join(unknown_deps)}"
        assert task not in deps, f"Task {task} depends on itself."


def assert_tasks_sorted() -> None:
    """Ensure that tasks are sorted alphabetically with ci_ functions following all others."""
    last_task: str | None = None
    in_ci_section = False
    for task_name in plan.ALL_TASKS:
        if not last_task:
            last_task = task_name
            continue
        if task_name.startswith("ci_") and not in_ci_section:
            in_ci_section = True
            last_task = task_name
            continue
        assert task_name > last_task, (
            f"TaskLibrary functions are sorted incorrectly. {task_name} must come before {last_task}."
        )
        last_task = task_name

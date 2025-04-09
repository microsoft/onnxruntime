#!/usr/bin/env python3

import argparse
import logging
import sys
from typing import Callable, List

from ep_build import self_test
from ep_build.logging import initialize_logging
from ep_build.plan import (
    ALL_TASKS,
    PUBLIC_TASKS,
    SUMMARIZERS,
    TASK_DEPENDENCIES,
    Plan,
    depends,
    public_task,
    task,
)
from ep_build.task import (
    ListTasksTask,
    NoOpTask,
)


if __name__ == "__main__":
    initialize_logging()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Build and test all the things.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "task",
        type=str,
        nargs="*",
        help='Task(s) to run. Specify "list" to show all tasks.',
    )

    parser.add_argument(
        "--skip", metavar="TASK_RE", type=str, nargs="+", help="List of tasks to skip."
    )

    parser.add_argument(
        "--only",
        action="store_true",
        help="Run only the listed task(s), skipping any dependencies.",
    )

    parser.add_argument(
        "--print-task-graph",
        action="store_true",
        help="Print the task library in DOT format and exit. Combine with --task to highlight what would run.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the plan, rather than running it."
    )

    args = parser.parse_args()
    return args


class TaskLibrary:

    @staticmethod
    def to_dot(highlight: List[str] = []) -> str:
        elements: List[str] = []
        for tsk in ALL_TASKS:
            task_attrs: List[str] = []
            if tsk in PUBLIC_TASKS:
                task_attrs.append("style=filled")
            if tsk in highlight:
                task_attrs.append("penwidth=4.0")
            if len(task_attrs) > 0:
                elements.append(f"{tsk} [{' '.join(task_attrs)}]")
            else:
                elements.append(tsk)
        for tsk in TASK_DEPENDENCIES:
            for dep in TASK_DEPENDENCIES[tsk]:
                elements.append(f"{tsk} -> {dep}")
        elements_str = "\n".join([f"  {element};" for element in elements])
        return f"digraph {{\n{elements_str}\n}}"

    @public_task("Print a list of commonly used tasks; see also --task=list_all.")
    @depends(["list_public"])
    def list(self, plan: Plan) -> str:
        return plan.add_step(NoOpTask())

    @task
    def list_all(self, plan: Plan) -> str:
        return plan.add_step(ListTasksTask(ALL_TASKS))

    @task
    def list_public(self, plan: Plan) -> str:
        return plan.add_step(ListTasksTask(PUBLIC_TASKS))


def plan_from_dependencies(
    main_tasks: List[str],
) -> Plan:
    task_library = TaskLibrary()
    plan = Plan()

    # We always run summarizers, which perform conditional work on the output
    # of other steps.
    work_list = SUMMARIZERS

    # The work list is processed as a stack, so LIFO. We reverse the user-specified
    # tasks so that they (and their dependencies) can be expressed in a natural order.
    work_list.extend(reversed(main_tasks))

    for task_name in work_list:
        if not hasattr(task_library, task_name):
            logging.fatal(f"Task '{task_name}' does not exist.")
            sys.exit(1)

    while len(work_list) > 0:
        task_name = work_list.pop()
        if plan.has_step(task_name):
            continue
        unfulfilled_deps: List[str] = []
        for dep in TASK_DEPENDENCIES.get(task_name, []):
            if not plan.has_step(dep):
                unfulfilled_deps.append(dep)
                assert hasattr(
                    task_library, dep
                ), f"Non-existent task '{dep}' was declared as a dependency for '{task_name}'."
        if len(unfulfilled_deps) == 0:
            # add task_name to plan
            task_adder: Callable[[Plan], str] = getattr(task_library, task_name)
            added_step = task_adder(plan)
            assert (
                added_step == task_name
            ), f"Task function '{task_name}' added a task with incorrect id '{added_step}'."
        else:
            # Look at task_name again later when its deps are satisfied
            work_list.append(task_name)
            work_list.extend(reversed(unfulfilled_deps))

    return plan


def plan_from_task_list(
    tasks: List[str],
) -> Plan:
    task_library = TaskLibrary()
    plan = Plan()
    for task_name in tasks:
        # add task_name to plan
        task_adder: Callable[[Plan], str] = getattr(task_library, task_name)
        task_adder(plan)
    return plan


def run_self_test():
    self_test.assert_task_dependencies_exist(TaskLibrary)
    self_test.assert_tasks_sorted()


def build_and_test():
    initialize_logging()

    args = parse_arguments()

    plan = Plan()

    if len(args.task) > 0:
        planner = plan_from_task_list if args.only else plan_from_dependencies
        plan = planner(args.task)

    if args.skip is not None:
        for skip in args.skip:
            plan.skip(skip)

    if args.print_task_graph:
        print(TaskLibrary.to_dot(plan.steps))
        sys.exit(0)
    elif len(args.task) == 0:
        logging.error("At least one task or --print-task-graph is required.")
        sys.exit(1)

    if args.dry_run:
        plan.print()
    else:
        caught = None
        try:
            plan.run()
        except Exception as ex:
            caught = ex
        print()
        plan.print_report()
        print()
        if caught:
            raise caught


if __name__ == "__main__":
    run_self_test()
    build_and_test()

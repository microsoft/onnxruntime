#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import argparse
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path

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
    CompositeTask,
    ExtractArchiveTask,
    ListTasksTask,
    NoOpTask,
    PyTestTask,
)
from ep_build.tasks.build import (
    BuildEpLinuxTask,
    BuildEpWindowsTask,
    QdcTestsTask,
)
from ep_build.tasks.python import CreateVenvTask, RunLinterTask
from ep_build.util import (
    DEFAULT_PYTHON,
    REPO_ROOT,
    is_host_arm64,
    is_host_linux,
    is_host_mac,
    is_host_windows,
)

QAIRT_SDK_ROOT_ENV_VAR = "QAIRT_SDK_ROOT"
QNN_SDK_ROOT_ENV_VAR = "QNN_SDK_ROOT"
SNPE_ROOT_ENV_VAR = "SNPE_ROOT"
VENV_PATH = REPO_ROOT / "venv"


if __name__ == "__main__":
    initialize_logging()


def parse_arguments():
    help_description = """Build and test ONNX Runtime.

Environment variables
---------------------
  ANDROID_HOME/ANDROID_NDK_HOME
    If both are specified, they are used instead of installing known good
    versions into build/tools.

  JAVA_HOME
    If specified, it is used instead of installing a known good version into build/tools.

  ORT_BUILD_TOOLS_PATH
    If specified, use this directory for build-managed tools instead of build/tools.

  ORT_NIGHTLY_BUILD
    If set to 1, instruct the ORT build to use a more verbose version string.

  QAIRT_SDK_ROOT
  QNN_SDK_ROOT
  SNPE_ROOT
    If specified, any of these will be used in place of the LKG QAIRT version.
    They are searched in the above order.

  QDC_API_TOKEN
    API token for use with testing in Qualcomm Device Cloud.
"""

    parser = argparse.ArgumentParser(
        description=help_description,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "task",
        type=str,
        nargs="*",
        help='Task(s) to run. Specify "list" to show all tasks.',
    )

    parser.add_argument("--dry-run", action="store_true", help="Print the plan, rather than running it.")
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
        "--qairt-sdk",
        type=Path,
        help=f"Path to QAIRT SDK. Overrides default version and {QAIRT_SDK_ROOT_ENV_VAR}, {QNN_SDK_ROOT_ENV_VAR}, and {SNPE_ROOT_ENV_VAR} environment variables.",
    )
    parser.add_argument("--skip", metavar="TASK_RE", type=str, nargs="+", help="List of tasks to skip.")

    args = parser.parse_args()
    return args


class TaskLibrary:
    """
    The collection of tasks the build is capable of running and the relationships between them.
    In other words, this is the dependency graph.
    """

    def __init__(
        self,
        python_executable: Path,
        venv_path: Path,
        qairt_sdk_root: Path | None,
    ) -> None:
        self.__python_executable = python_executable
        self.__venv_path = venv_path
        self.__qairt_sdk_root = qairt_sdk_root

    @staticmethod
    def to_dot(highlight: list[str] | None = None) -> str:
        """
        Used by --print-task-graph to create a GraphViz representation of the build graph.
        """
        elements: list[str] = []
        for tsk in ALL_TASKS:
            task_attrs: list[str] = []
            if tsk in PUBLIC_TASKS:
                task_attrs.append("style=filled")
            if highlight and tsk in highlight:
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

    @task
    @depends(["build_ort_android"])
    def archive_ort_android(self, plan: Plan) -> str:
        if is_host_linux() or is_host_mac():
            return plan.add_step(
                BuildEpLinuxTask(
                    "Archiving ONNX Runtime for Android",
                    self.__venv_path,
                    "android",
                    self.__qairt_sdk_root,
                    "archive",
                )
            )
        elif is_host_windows():
            return plan.add_step(
                BuildEpWindowsTask(
                    "Archiving ONNX Runtime for Android",
                    self.__venv_path,
                    "android",
                    "arm64",
                    "Release",
                    self.__qairt_sdk_root,
                    "archive",
                )
            )
        else:
            raise NotImplementedError("Archiving for Android on this host is not supported.")

    @task
    @depends(["build_ort_linux"])
    def archive_ort_linux(self, plan: Plan) -> str:
        return plan.add_step(
            BuildEpLinuxTask(
                "Archiving ONNX Runtime for Linux",
                self.__venv_path,
                "linux",
                self.__qairt_sdk_root,
                "archive",
            )
        )

    @task
    @depends(["build_ort_windows_arm64"])
    def archive_ort_windows_arm64(self, plan: Plan) -> str:
        return plan.add_step(
            BuildEpWindowsTask(
                "Archiving ONNX Runtime for Windows on ARM64",
                self.__venv_path,
                "windows",
                "arm64",
                "Release",
                self.__qairt_sdk_root,
                "archive",
            )
        )

    @task
    @depends(["build_ort_windows_arm64ec"])
    def archive_ort_windows_arm64ec(self, plan: Plan) -> str:
        return plan.add_step(
            BuildEpWindowsTask(
                "Archiving ONNX Runtime for Windows on ARM64EC",
                self.__venv_path,
                "windows",
                "arm64ec",
                "Release",
                self.__qairt_sdk_root,
                "archive",
            )
        )

    @task
    @depends(["build_ort_windows_x86_64"])
    def archive_ort_windows_x86_64(self, plan: Plan) -> str:
        return plan.add_step(
            BuildEpWindowsTask(
                "Archiving ONNX Runtime for Windows on x86_64",
                self.__venv_path,
                "windows",
                "x86_64",
                "Release",
                self.__qairt_sdk_root,
                "archive",
            )
        )

    @public_task("Build ONNX Runtime")
    @depends(
        [
            (is_host_linux(), "build_ort_linux"),
            (is_host_windows(), "build_ort_windows_host"),
        ]
    )
    def build(self, plan: Plan) -> str:
        return plan.add_step(NoOpTask())

    @task
    @depends(["create_venv"])
    def build_ort_android(self, plan: Plan) -> str:
        if is_host_linux() or is_host_mac():
            return plan.add_step(
                BuildEpLinuxTask(
                    "Building ONNX Runtime for Android",
                    self.__venv_path,
                    "android",
                    self.__qairt_sdk_root,
                    "build",
                )
            )
        elif is_host_windows():
            return plan.add_step(
                BuildEpWindowsTask(
                    "Building ONNX Runtime for Android",
                    self.__venv_path,
                    "android",
                    "arm64",
                    "Release",
                    self.__qairt_sdk_root,
                    "build",
                )
            )
        else:
            raise NotImplementedError("Building for Android on this host is not supported.")

    @task
    @depends(["create_venv"])
    def build_ort_linux(self, plan: Plan) -> str:
        return plan.add_step(
            BuildEpLinuxTask(
                "Building ONNX Runtime for Linux",
                self.__venv_path,
                "linux",
                self.__qairt_sdk_root,
                "build",
            )
        )

    @task
    @depends(["create_venv"])
    def build_ort_windows_arm64(self, plan: Plan) -> str:
        return plan.add_step(
            BuildEpWindowsTask(
                "Building ONNX Runtime for Windows on ARM64",
                self.__venv_path,
                "windows",
                "arm64",
                "Release",
                self.__qairt_sdk_root,
                "build",
            )
        )

    @task
    @depends(["create_venv"])
    def build_ort_windows_arm64ec(self, plan: Plan) -> str:
        return plan.add_step(
            BuildEpWindowsTask(
                "Building ONNX Runtime for Windows on ARM64EC",
                self.__venv_path,
                "windows",
                "arm64ec",
                "Release",
                self.__qairt_sdk_root,
                "build",
            )
        )

    @task
    @depends(
        [
            (is_host_arm64(), "build_ort_windows_arm64"),
            (not is_host_arm64(), "build_ort_windows_x86_64"),
        ]
    )
    def build_ort_windows_host(self, plan: Plan) -> str:
        return plan.add_step(NoOpTask())

    @task
    @depends(["create_venv"])
    def build_ort_windows_x86_64(self, plan: Plan) -> str:
        return plan.add_step(
            BuildEpWindowsTask(
                "Building ONNX Runtime for Windows on x86_64",
                self.__venv_path,
                "windows",
                "x86_64",
                "Release",
                self.__qairt_sdk_root,
                "build",
            )
        )

    @task
    def create_venv(self, plan: Plan) -> str:
        return plan.add_step(CreateVenvTask(self.__python_executable, self.__venv_path))

    @task
    def extract_ort_linux(self, plan: Plan) -> str:
        return plan.add_step(
            ExtractArchiveTask(
                "Extracting ONNX Runtime for Linux",
                REPO_ROOT / "build" / "onnxruntime-tests-linux.tar.bz2",
                REPO_ROOT / "build" / "linux" / "Release",
            )
        )

    @task
    def extract_ort_windows_arm64(self, plan: Plan) -> str:
        return plan.add_step(
            ExtractArchiveTask(
                "Extracting ONNX Runtime for Windows on ARM64",
                REPO_ROOT / "build" / "onnxruntime-tests-windows-arm64.zip",
                REPO_ROOT / "build" / "windows-arm64" / "Release",
            )
        )

    @task
    def extract_ort_windows_x86_64(self, plan: Plan) -> str:
        return plan.add_step(
            ExtractArchiveTask(
                "Extracting ONNX Runtime for Windows on x86_64",
                REPO_ROOT / "build" / "onnxruntime-tests-windows-x86_64.zip",
                REPO_ROOT / "build" / "windows-x86_64" / "Release",
            )
        )

    @public_task("Generate build\\vs\\Debug\\onnxruntime.sln")
    @depends(["create_venv"])
    def generate_sln(self, plan: Plan) -> str:
        return plan.add_step(
            BuildEpWindowsTask(
                "Generating Visual Studio .sln",
                self.__venv_path,
                "windows",
                "x86_64",
                "Debug",
                self.__qairt_sdk_root,
                "generate_sln",
            )
        )

    @public_task("Run the source linter")
    @depends(["create_venv"])
    def lint(self, plan: Plan) -> str:
        return plan.add_step(RunLinterTask(self.__venv_path))

    @public_task("Run the source linter and fix any issues automatically")
    @depends(["create_venv"])
    def lint_and_fix(self, plan: Plan) -> str:
        return plan.add_step(RunLinterTask(self.__venv_path, auto_fix=True))

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

    @public_task("Test ONNX Runtime")
    @depends(
        [
            (is_host_linux(), "test_ort_linux"),
            (is_host_windows(), "test_ort_windows"),
        ]
    )
    def test(self, plan: Plan) -> str:
        return plan.add_step(NoOpTask())

    @task
    @depends(["build_ort_linux"])
    def test_ort_linux(self, plan: Plan) -> str:
        return plan.add_step(
            BuildEpLinuxTask(
                "Testing ONNX Runtime for Linux",
                self.__venv_path,
                "linux",
                self.__qairt_sdk_root,
                "test",
            )
        )

    @task
    @depends(["archive_ort_android"])
    def test_ort_local_android(self, plan: Plan) -> str:
        env = dict(os.environ)
        test_root = REPO_ROOT / "build" / "qdc_test_root"
        env["QDC_TEST_ROOT"] = str(test_root)
        env["MODEL_TEST_ROOT"] = str(REPO_ROOT / "build" / "android" / "model_tests")

        # This is a pretty slow way to do this, but it's easy to implement
        # and essentially free to maintain. If you find yourself using this
        # often enough that your life would be better if we didn't roundtrip
        # through a zip file, please open a Jira and we'll invest more here.
        return plan.add_step(
            CompositeTask(
                group_name=None,
                tasks=[
                    ExtractArchiveTask(
                        "Extracting ONNX Runtime for Android",
                        REPO_ROOT / "build" / "onnxruntime-tests-android.zip",
                        test_root,
                    ),
                    PyTestTask(
                        "Testing ONNX Runtime for Android with a local device",
                        self.__venv_path,
                        ["tests"],
                        env=env,
                        cwd=REPO_ROOT / "qcom" / "scripts" / "linux" / "appium",
                    ),
                ],
            )
        )

    @task
    @depends(["archive_ort_android"])
    def test_ort_qdc_android(self, plan: Plan) -> str:
        return plan.add_step(
            QdcTestsTask(
                "Testing ONNX Runtime for Android in QDC",
                self.__venv_path,
                ["android"],
                extra_args=[
                    "--append-android-package=onnx_models:model_tests/onnx_models",
                ],
            )
        )

    @task
    @depends(["archive_ort_windows_arm64"])
    def test_ort_qdc_windows_arm64(self, plan: Plan) -> str:
        return plan.add_step(
            QdcTestsTask(
                "Testing ONNX Runtime for Windows on ARM64 in QDC",
                self.__venv_path,
                ["windows"],
                extra_args=[
                    "--append-windows-package=onnx_models:model_tests/onnx_models",
                ],
            )
        )

    @task
    @depends(
        [
            (is_host_arm64(), "test_ort_windows_arm64"),
            (not is_host_arm64(), "test_ort_windows_x86_64"),
        ]
    )
    def test_ort_windows(self, plan: Plan) -> str:
        return plan.add_step(NoOpTask())

    @task
    @depends(["build_ort_windows_arm64"])
    def test_ort_windows_arm64(self, plan: Plan) -> str:
        return plan.add_step(
            BuildEpWindowsTask(
                "Testing ONNX Runtime for Windows on ARM64",
                self.__venv_path,
                "windows",
                "arm64",
                "Release",
                self.__qairt_sdk_root,
                "test",
            )
        )

    @task
    @depends(["build_ort_windows_arm64ec"])
    def test_ort_windows_arm64ec(self, plan: Plan) -> str:
        return plan.add_step(
            BuildEpWindowsTask(
                "Testing ONNX Runtime for Windows on ARM64EC",
                self.__venv_path,
                "windows",
                "arm64ec",
                "Release",
                self.__qairt_sdk_root,
                "test",
            )
        )

    @task
    @depends(["build_ort_windows_x86_64"])
    def test_ort_windows_x86_64(self, plan: Plan) -> str:
        return plan.add_step(
            BuildEpWindowsTask(
                "Testing ONNX Runtime for Windows on x86_64",
                self.__venv_path,
                "windows",
                "x86_64",
                "Release",
                self.__qairt_sdk_root,
                "test",
            )
        )


def get_qairt_sdk_root(root_from_args: Path | None) -> Path | None:
    qairt_sdk: Path | None = None
    if root_from_args is not None:
        qairt_sdk = root_from_args
    elif QAIRT_SDK_ROOT_ENV_VAR in os.environ:
        qairt_sdk = Path(os.environ[QAIRT_SDK_ROOT_ENV_VAR])
    elif QNN_SDK_ROOT_ENV_VAR in os.environ:
        qairt_sdk = Path(os.environ[QNN_SDK_ROOT_ENV_VAR])
    elif SNPE_ROOT_ENV_VAR in os.environ:
        qairt_sdk = Path(os.environ[SNPE_ROOT_ENV_VAR])
    else:
        # Let the build fetch QAIRT
        return None

    if not qairt_sdk.exists():
        raise FileNotFoundError(f"QAIRT SDK root {qairt_sdk} does not exist.")

    return qairt_sdk


def plan_from_dependencies(
    main_tasks: list[str],
    python_executable: Path,
    venv_path: Path,
    qairt_sdk_root: Path | None,
) -> Plan:
    """
    Uses a work list algorithm to create a Plan to build the given tasks and their
    dependencies in a valid order. This is the default planner.
    """
    task_library = TaskLibrary(python_executable, venv_path, qairt_sdk_root)
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
        unfulfilled_deps: list[str] = []
        for dep in TASK_DEPENDENCIES.get(task_name, []):
            if not plan.has_step(dep):
                unfulfilled_deps.append(dep)
                assert hasattr(task_library, dep), (
                    f"Non-existent task '{dep}' was declared as a dependency for '{task_name}'."
                )
        if len(unfulfilled_deps) == 0:
            # add task_name to plan
            task_adder: Callable[[Plan], str] = getattr(task_library, task_name)
            added_step = task_adder(plan)
            assert added_step == task_name, (
                f"Task function '{task_name}' added a task with incorrect id '{added_step}'."
            )
        else:
            # Look at task_name again later when its deps are satisfied
            work_list.append(task_name)
            work_list.extend(reversed(unfulfilled_deps))

    return plan


def plan_from_task_list(
    tasks: list[str],
    python_executable: Path,
    venv_path: Path,
    qairt_sdk_root: Path | None,
) -> Plan:
    """
    Planner that just instantiates the given tasks with no attempt made to satisfy dependencies.
    Used by --only.
    """
    task_library = TaskLibrary(python_executable, venv_path, qairt_sdk_root)
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
    qairt_sdk_root = get_qairt_sdk_root(args.qairt_sdk)

    plan = Plan()

    if len(args.task) > 0:
        planner = plan_from_task_list if args.only else plan_from_dependencies
        plan = planner(args.task, DEFAULT_PYTHON, VENV_PATH, qairt_sdk_root)

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

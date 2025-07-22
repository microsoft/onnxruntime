# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import logging
import shutil
import tarfile
import zipfile
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from .github import end_group, start_group
from .util import BASH_EXECUTABLE, run_with_venv

REPO_ROOT = (Path(__file__).parent / ".." / "..").resolve()


class Task(ABC):
    def __init__(self, group_name: str | None) -> None:
        """
        Initialize a new instance.

        Args:
          * :group_name: Used for logging and grouping messages. None is valid.
        """
        self.group_name = group_name

    @abstractmethod
    def does_work(self) -> bool:
        """
        Return True if this task actually does something (e.g., runs commands).
        """

    @abstractmethod
    def run_task(self) -> None:
        """
        Entry point for implementations: perform the task's action.
        """

    def run(self) -> None:
        """
        Entry point for callers: perform any startup/teardown tasks and call run_task.
        """
        if self.group_name:
            start_group(self.group_name)
        self.run_task()
        if self.group_name:
            end_group()


class ExtractArchiveTask(Task):
    def __init__(
        self,
        group_name: str | None,
        archive_path: Path,
        dest_dir: Path,
    ) -> None:
        super().__init__(group_name)
        self.__archive_path = archive_path
        self.__dest_dir = dest_dir

    def does_work(self) -> bool:
        return True

    def run_task(self) -> None:
        logging.debug(f"Extracting archive {self.__archive_path} to {self.__dest_dir}.")
        if tarfile.is_tarfile(self.__archive_path):
            with tarfile.TarFile.open(self.__archive_path) as tar:
                tar.extractall(self.__dest_dir)
        elif zipfile.is_zipfile(self.__archive_path):
            with zipfile.ZipFile(self.__archive_path, "r") as zip:
                zip.extractall(self.__dest_dir)
        else:
            raise NotImplementedError(f"Unsupported archive type: {self.__archive_path}.")


class FailTask(Task):
    """A Task that unconditionally fails."""

    def __init__(self, message: str) -> None:
        super().__init__(group_name=None)
        self._message = message

    def does_work(self) -> bool:
        return True

    def run_task(self) -> None:
        raise RuntimeError(self._message)


class ListTasksTask(Task):
    def __init__(self, tasks: list[str]) -> None:
        super().__init__(group_name=None)
        self.tasks = tasks

    def does_work(self) -> bool:
        return False

    def run_task(self) -> None:
        from . import plan

        for task_name in sorted(self.tasks):
            print(task_name)
            description = plan.TASK_DESCRIPTIONS.get(task_name, None)
            if description:
                print(f"    {description}")


class NoOpTask(Task):
    """A Task that does nothing."""

    def __init__(self, group_name: str | None = None) -> None:
        super().__init__(group_name=group_name)

    def does_work(self) -> bool:
        return False

    def run_task(self) -> None:
        pass


class PrintMessageTask(NoOpTask):
    def __init__(self, message: str) -> None:
        super().__init__(group_name=None)
        self.__message = message

    def run_task(self) -> None:
        logging.info(self.__message)


class RemovePathsTask(Task):
    """
    A path to remove files and directories.
    """

    def __init__(
        self,
        group_name: str | None,
        paths: Path | Iterable[Path],
    ) -> None:
        super().__init__(group_name)
        self.__paths: Iterable[Path] = paths if isinstance(paths, Iterable) else [paths]

    def does_work(self) -> bool:
        return True

    def run_task(self) -> None:
        for path in self.__paths:
            if not path.exists():
                continue
            if path.is_file():
                logging.debug(f"Deleting file {path}")
                path.unlink()
            elif path.is_dir():
                logging.debug(f"Removing directory {path}")
                shutil.rmtree(path)
            else:
                raise RuntimeError(f"Unknown file type: {path}")


class RunExecutablesWithVenvTask(Task):
    """
    A task that runs a list of explicit executables with a specific Python
    virtual environment enabled.
    """

    def __init__(
        self,
        group_name: str | None,
        venv: Path | None,
        executables_and_args: list[list[str]],
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
    ) -> None:
        super().__init__(group_name)
        self.__venv = venv
        self.__executables_and_args = executables_and_args
        self.__env = env
        self.__cwd = cwd

    def does_work(self) -> bool:
        return True

    def run_task(self) -> None:
        for executable_and_args in self.__executables_and_args:
            run_with_venv(self.__venv, executable_and_args, env=self.__env, cwd=self.__cwd)

    @property
    def executables_and_args(self) -> list[list[str]]:
        return self.__executables_and_args

    @executables_and_args.setter
    def executables_and_args(self, value: list[list[str]]) -> None:
        self.__executables_and_args = value

    @property
    def env(self) -> Mapping[str, str] | None:
        return self.__env


class RunExecutablesTask(RunExecutablesWithVenvTask):
    """
    A task that runs a list of explicit executables.
    """

    def __init__(
        self,
        group_name: str | None,
        executables_and_args: list[list[str]],
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
    ) -> None:
        super().__init__(group_name, None, executables_and_args, env, cwd)


class BashScriptsWithVenvTask(RunExecutablesWithVenvTask):
    """
    A Task that runs bash scripts with a specific Python virtual environment enabled.
    """

    def __init__(
        self,
        group_name: str | None,
        venv: Path | None,
        scripts_and_args: list[list[str]],
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
    ) -> None:
        executables_and_args = [[BASH_EXECUTABLE] + s_a for s_a in scripts_and_args]  # noqa: RUF005
        super().__init__(group_name, venv, executables_and_args, env, cwd)


class BashScriptsTask(BashScriptsWithVenvTask):
    """
    A Task that runs bash scripts.
    """

    def __init__(
        self,
        group_name: str | None,
        scripts_and_args: list[list[str]],
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
    ) -> None:
        super().__init__(group_name, None, scripts_and_args, env, cwd)


class CompositeTask(Task):
    """
    A Task composed of a list of other Tasks.
    """

    def __init__(self, group_name: str | None, tasks: Iterable[Task]) -> None:
        super().__init__(group_name)
        self.tasks = tasks

    def does_work(self) -> bool:
        return any(t.does_work() for t in self.tasks)

    def run_task(self) -> None:
        for task in self.tasks:
            task.run()


class ConditionalTask(Task):
    """
    A Task that runs one of two alternatives, depending on the result of
    a predicate function call.
    """

    def __init__(
        self,
        group_name: str | None,
        condition: Callable[[], bool],
        true_task: Task,
        false_task: Task,
    ) -> None:
        super().__init__(group_name)
        self.condition = condition
        self.true_task = true_task
        self.false_task = false_task

    def does_work(self) -> bool:
        if self.condition():
            return self.true_task.does_work()
        else:
            return self.false_task.does_work()

    def run_task(self) -> None:
        if self.condition():
            self.true_task.run()
        else:
            self.false_task.run()


class PyTestTask(RunExecutablesWithVenvTask):
    def __init__(
        self,
        group_name: str | None,
        venv: Path | None,
        files_or_dirs: list[str],
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
    ):
        cmd = [["pytest", *files_or_dirs]]
        super().__init__(group_name, venv, cmd, env, cwd)

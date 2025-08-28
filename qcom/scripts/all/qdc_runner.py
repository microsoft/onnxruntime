#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import argparse
import concurrent.futures
import functools
import logging
import os
import random
import shutil
import tempfile
import time
import uuid
import zipfile
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import ClassVar, NamedTuple

import backoff
import requests
from package_manager import PackageManager
from prettytable import PrettyTable
from qdc_public_api_client import Client
from qdc_public_api_client.api.artifacts import (
    post_artifacts_startupload,
    post_artifacts_uuid_continueupload,
    post_artifacts_uuid_endupload,
)
from qdc_public_api_client.api.jobs import (
    get_jobs_downloadlogs,
    get_jobs_id,
    get_jobs_job_id_logs,
    post_jobs,
)
from qdc_public_api_client.models.artifact_type import ArtifactType
from qdc_public_api_client.models.artifact_type_0 import ArtifactType0
from qdc_public_api_client.models.create_job_type_0 import CreateJobType0
from qdc_public_api_client.models.job_logs_type_0 import JobLogsType0
from qdc_public_api_client.models.job_mode import JobMode
from qdc_public_api_client.models.job_result import JobResult
from qdc_public_api_client.models.job_state import JobState
from qdc_public_api_client.models.job_submission_parameter import JobSubmissionParameter
from qdc_public_api_client.models.job_type import JobType
from qdc_public_api_client.models.job_type_0 import JobType0
from qdc_public_api_client.models.log_upload_status import LogUploadStatus
from qdc_public_api_client.models.post_artifacts_uuid_continueupload_body import (
    PostArtifactsUuidContinueuploadBody,
)
from qdc_public_api_client.models.test_framework import TestFramework
from qdc_public_api_client.types import File, Response, Unset

ORT_APPIUM_ARCHIVE = (
    Path(__file__).parent.parent.parent.parent / "build" / "onnxruntime-tests-android-aarch64.zip"
).absolute()
ORT_TEST_WIN_ARM64 = (
    Path(__file__).parent.parent.parent.parent / "build" / "onnxruntime-tests-windows-arm64.zip"
).absolute()


#
# Test Definitions
#


class Platform(Enum):
    ANDROID = 1
    WINDOWS = 2


class TestDevice(NamedTuple):
    device_id: str
    device_name: str


class TestConfig(NamedTuple):
    test_name: str
    platform: Platform
    devices: list[TestDevice]

    def get_device_ids(self) -> list[str]:
        ids = []
        for d in self.devices:
            ids.append(d.device_id)
        return ids


# fmt: off
QDC_TESTS = [
    TestConfig("Mobile Device", Platform.ANDROID, [
        TestDevice("3625030", "Lanai SM8650"),
        # TODO: [AISW-140749] Unit test failures on Pakala in QAIRT 2.35
        # TestDevice("3700521", "Pakala SM8750"),
    ]),
    TestConfig("Windows on Snapdragon", Platform.WINDOWS, [
        TestDevice("3625029", "Hamoa SC8380XP"),
    ]),
]
# fmt: on

#
# Requests & Retries
#


# Will sleep {backoff factor} * (2 ** ({number of previous retries})) seconds
BACKOFF_FACTOR = 0.75
# Try 6 times total, for a max total delay of about 20s
MAX_TRIES = 6


def retry_with_backoff():
    """
    Decorator for retrying calls to web services.
    """

    def wrapper(func):
        @backoff.on_exception(
            wait_gen=backoff.expo,
            exception=(
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ),
            max_tries=MAX_TRIES,
        )
        @backoff.on_predicate(
            wait_gen=backoff.expo,
            max_tries=MAX_TRIES,
            predicate=lambda response: response.status_code
            in [
                429,  # Too many requests
                500,  # Internal Server Error
                502,  # Bad Gateway
                503,  # Service Unavailable
                504,  # Gateway Timeout
            ],
        )
        @functools.wraps(func)
        def inner(*args, **kwargs):
            return func(*args, **kwargs)

        return inner

    return wrapper


class RetryingSessionWithTimeout(requests.Session):
    @retry_with_backoff()
    def request(self, *args, **kwargs):
        # Our webserver is configured to timeout requests after 25 seconds,
        # so we specify a slightly higher max timeout for all requests.
        # By default, requests waits forever.
        # https://requests.readthedocs.io/en/latest/user/quickstart/#timeouts
        default_kwargs = {"timeout": 28}

        return super().request(*args, **{**default_kwargs, **kwargs})


def create_session():
    return RetryingSessionWithTimeout()


#
# Runs
#


class RunStatus(Enum):
    UNKNOWN = 0
    PENDING = 1
    RUNNING = 2
    SUCCESS = 3
    FAILED = 4


class Run(ABC):
    def __init__(self, test_name: str, url: str):
        self.test_name = test_name
        self.status = RunStatus.UNKNOWN
        self.url = url

    def is_done(self):
        return self.status == RunStatus.SUCCESS or self.status == RunStatus.FAILED

    @abstractmethod
    def poll(self) -> None:
        pass

    @abstractmethod
    def get_farm_name(self) -> str:
        pass


class TestRuns:
    _runs: ClassVar[list[Run]] = []

    def add_run(self, run: Run) -> None:
        self._runs.append(run)

    def has_pending_tasks(self) -> bool:
        return any(not run.is_done() for run in self._runs)

    def success(self) -> bool:
        return all(run.status == RunStatus.SUCCESS for run in self._runs)

    def poll_runs(self) -> None:
        for run in self._runs:
            if not run.is_done():
                run.poll()
                timestr = time.strftime("%H:%M:%S")
                logging.info(f"[{timestr}] {run.test_name} - {run.status.name}")

    def get_status_table(self) -> PrettyTable:
        table = PrettyTable()
        table.align = "l"
        table.field_names = [
            "Farm",
            "Test",
            "Status",
        ]
        for run in self._runs:
            table.add_row(
                [
                    run.get_farm_name(),
                    run.test_name,
                    run.status.name,
                ]
            )
        return table

    def get_failure_urls(self) -> PrettyTable:
        table = PrettyTable()
        table.align = "l"
        table.field_names = [
            "Test",
            "URL",
        ]

        for run in self._runs:
            if run.status != RunStatus.SUCCESS:
                table.add_row([run.test_name, run.url])
        return table


#
# QDC runner & friends
#


class QdcClient:
    _QDC_PROD_API_URL = "https://apigwx-aws.qualcomm.com/saga/v1/public/qdc"

    def __init__(self, api_url: str | None, on_behalf_of: str | None):
        api_token = os.environ["QDC_API_TOKEN"]

        if not api_url or len(api_url) == 0:
            self.api_url = self._QDC_PROD_API_URL
        else:
            self.api_url = api_url

        qdc_headers = {
            "X-QCOM-AppName": "ONNX QNN EP unit tests",
            "X-QCOM-ClientType": "Python",
            "X-QCOM-TracingId": str(uuid.uuid4()),
            "X-QCOM-TokenType": "apikey",
            "Authorization": api_token,
        }

        if on_behalf_of is not None:
            qdc_headers["X-QCOM-OnBehalfOf"] = on_behalf_of

        self._client = Client(base_url=self.api_url, headers=qdc_headers)

    def get_job_status(self, job_id: int) -> JobType0:
        response = self.get_jobs_id(job_id)

        if response.status_code != 200:
            raise RuntimeError(
                f"Error starting job on QDC. Response code={response.status_code}: {response.content.decode('utf-8')}"
            )

        if type(response.parsed) is not JobType0:
            raise RuntimeError("QDC jobs_id response missing required fields.")

        return response.parsed

    def get_job_url(self, job_id: int) -> str:
        if self.api_url == self._QDC_PROD_API_URL:
            return f"https://qdc.qualcomm.com/#/reports/job/automated/{job_id}"
        else:
            return "User overridden API endpoint provided; unknown job URL"

    def download_logs(self, job_id: int, log_dir: Path) -> None:
        # Only collect logs whose names contain these:
        partial_names = [
            ".results.txt",
            ".results.xml",
            "_stdout.txt",
            "test_dbg.stdout",
        ]

        all_job_log_names = self._get_full_job_log_names(job_id)
        desired_job_log_names = [PurePosixPath(n) for n in all_job_log_names if any(p in n for p in partial_names)]
        logging.info(f"Job produced {len(all_job_log_names)} logs, {len(desired_job_log_names)} are interesting.")

        log_dir.mkdir(parents=True, exist_ok=True)
        for qdc_log_path in desired_job_log_names:
            local_log_path = log_dir / PurePosixPath(qdc_log_path).name
            self._download_log(qdc_log_path, local_log_path)

    @retry_with_backoff()
    def post_artifacts_startupload(self, *args, **kwargs) -> Response["ArtifactType0 | None"]:
        return post_artifacts_startupload.sync_detailed(
            client=self._client,
            *args,  # noqa: B026
            **kwargs,
        )

    @retry_with_backoff()
    def post_artifacts_uuid_endupload(self, *args, **kwargs) -> Response["ArtifactType0 | None"]:
        return post_artifacts_uuid_endupload.sync_detailed(
            client=self._client,
            *args,  # noqa: B026
            **kwargs,
        )

    @retry_with_backoff()
    def post_artifacts_uuid_continueupload(self, *args, **kwargs) -> Response["ArtifactType0 | None"]:
        return post_artifacts_uuid_continueupload.sync_detailed(
            client=self._client,
            *args,  # noqa: B026
            **kwargs,
        )

    @retry_with_backoff()
    def post_jobs(self, *args, **kwargs) -> Response["JobType0 | None"]:
        return post_jobs.sync_detailed(
            client=self._client,
            *args,  # noqa: B026
            **kwargs,
        )

    @retry_with_backoff()
    def get_jobs_id(self, *args, **kwargs) -> Response["JobType0 | None"]:
        return get_jobs_id.sync_detailed(
            client=self._client,
            *args,  # noqa: B026
            **kwargs,
        )

    def _download_log(self, qdc_log_path: PurePosixPath, local_log_path: Path) -> None:
        logging.info(f"Downloading QDC log {qdc_log_path} to {local_log_path}.")
        download_response = self._download_zipped_log(qdc_log_path)
        if download_response.status_code != 200:
            raise RuntimeError(
                f"Cannot download log file. Response code={download_response.status_code}: {download_response.content.decode('utf-8')}"
            )

        # QDC artifacts are returned in zip files.
        with tempfile.NamedTemporaryFile(suffix="log.zip") as zip_file:
            zip_path = Path(zip_file.name)
            logging.debug(f"Saving zipped log contents to {zip_path}.")
            zip_file.write(download_response.content)

            logging.debug(f"Extracting log archive to {local_log_path}.")
            with zipfile.ZipFile(zip_file, "r") as archive:
                archive.extractall(local_log_path.parent)

    @retry_with_backoff()
    def _download_zipped_log(self, qdc_log_path: PurePosixPath) -> Response[File]:
        return get_jobs_downloadlogs.sync_detailed(client=self._client, path=str(qdc_log_path))

    @retry_with_backoff()
    def _get_job_log_list(self, job_id: int) -> Response[list["JobLogsType0 | None"]]:
        return get_jobs_job_id_logs.sync_detailed(job_id=job_id, client=self._client)

    def _get_full_job_log_names(self, job_id: int) -> list[str]:
        @backoff.on_predicate(
            backoff.constant,
            lambda status: status in (LogUploadStatus.INPROGRESS, LogUploadStatus.NOLOGS),
            max_time=60 * 10,
            interval=5,
        )
        def _get_final_log_status(job_id: int) -> LogUploadStatus:
            log_status = self.get_job_status(job_id).log_upload_status
            assert not isinstance(log_status, Unset)
            return log_status

        log_status = _get_final_log_status(job_id)
        if log_status != LogUploadStatus.COMPLETED:
            return []

        response = self._get_job_log_list(job_id)
        if response.status_code != 200:
            raise RuntimeError(
                f"Error getting job logs. Response code={response.status_code}: {response.content.decode('utf-8')}",
            )

        log_list = response.parsed
        if log_list is None:
            raise RuntimeError("Error getting job logs. Job logs response does not contain the required fields.")
        return [lf.filename for lf in log_list if lf is not None and isinstance(lf.filename, str)]


class QdcRunner:
    _JOB_TIMEOUT_MINUTES = 15

    class QdcJobConfig(NamedTuple):
        upload_uuids: list[str]
        test_framework: TestFramework
        entry_script: str | None

    def __init__(
        self,
        test_name: str,
        on_behalf_of: str | None,
        log_dir: Path | None,
        test_runs: TestRuns,
        api_url: str,
        enable_platforms: list[Platform],
        override_android_id: str,
        override_windows_id: str,
        android_archive: Path,
        windows_archive: Path,
    ):
        logging.info("Initializing QDC test runner for project.")
        self._client = QdcClient(api_url, on_behalf_of)
        self._test_name = test_name
        self._log_dir = log_dir

        enable_android = len(enable_platforms) == 0 or Platform.ANDROID in enable_platforms
        enable_windows = len(enable_platforms) == 0 or Platform.WINDOWS in enable_platforms
        have_override = override_android_id or override_windows_id

        self._job_config = {}

        if (not have_override or override_android_id) and enable_android:
            apk_test_pkg_uuid = self._upload_artifact(android_archive, ArtifactType.TESTSCRIPT)
            self._job_config[Platform.ANDROID] = QdcRunner.QdcJobConfig([apk_test_pkg_uuid], TestFramework.APPIUM, None)

        if (not have_override or override_windows_id) and enable_windows:
            windows_test_uuid = self._upload_artifact(windows_archive, ArtifactType.TESTSCRIPT)
            self._job_config[Platform.WINDOWS] = QdcRunner.QdcJobConfig(
                [windows_test_uuid],
                TestFramework.POWERSHELL,
                "C:\\Temp\\TestContent\\run_tests.ps1",
            )

        if override_android_id and enable_android:
            self._schedule_tests(
                test_runs,
                [
                    TestConfig(
                        "Override Android",
                        Platform.ANDROID,
                        [TestDevice(override_android_id, override_android_id)],
                    )
                ],
            )
        if override_windows_id and enable_windows:
            self._schedule_tests(
                test_runs,
                [
                    TestConfig(
                        "Override Windows",
                        Platform.WINDOWS,
                        [TestDevice(override_windows_id, override_windows_id)],
                    )
                ],
            )
        if not have_override:
            tests = QDC_TESTS
            if enable_platforms is not None and len(enable_platforms) > 0:
                tests = [t for t in tests if t.platform in enable_platforms]
            self._schedule_tests(test_runs, tests)

    def _finish_run(self, job_id: int, platform: Platform, status: RunStatus) -> RunStatus:
        if self._log_dir is not None:
            log_dir = Path(str(self._log_dir).replace("%p", platform.name.lower()))
            logging.info(f"Downloading log files to {log_dir}.")
            self._client.download_logs(job_id, log_dir)
            return status

        return status

    def _schedule_tests(self, test_runs: TestRuns, tests: list[TestConfig]) -> None:
        logging.info("Scheduling QDC tests.")
        for test in tests:
            device = random.choice(test.devices)

            run = self._schedule_test(
                f"{self._test_name}: {test.test_name} - {device.device_name}",
                device.device_id,
                test.platform,
            )
            test_runs.add_run(run)

    def _upload_artifact(self, artifact_filename: Path, artifact_type: ArtifactType) -> str:
        def check_http_response(response: Response["ArtifactType0 | None"]) -> None:
            if response.status_code == 200:
                return

            raise RuntimeError(
                f"Error uploading artifact to QDC. Response code={response.status_code}: {response.content.decode('utf-8')}"
            )

        logging.info(f"Starting QDC artifact upload: {artifact_filename}.")

        # start upload
        response = self._client.post_artifacts_startupload(
            artifact_type=artifact_type,
            filename=f"{uuid.uuid4()}-{artifact_filename.name}",
        )
        check_http_response(response)

        if type(response.parsed) is not ArtifactType0:
            raise RuntimeError("QDC startupload response missing required fields.")
        artifact_upload_uuid = str(response.parsed.uuid)

        logging.info(f"Starting QDC upload for artifact={artifact_upload_uuid}.")

        # "continue" upload
        self._upload_artifact_continue_helper(artifact_filename, artifact_upload_uuid)

        # end upload
        response = self._client.post_artifacts_uuid_endupload(uuid=artifact_upload_uuid)
        check_http_response(response)

        logging.info(f"Finished QDC upload for artifact={artifact_upload_uuid}.")

        return artifact_upload_uuid

    def _upload_artifact_continue_helper(
        self,
        artifact_filename: Path,
        artifact_uuid: str,
    ) -> None:
        # This code is based on modified sample code provided by QDC.
        # QDC is going to beef up the client to simplify this from our perspective, so this
        # will be a short-lived function.

        def _do_upload(
            artifact_upload_uuid: str,
            buffer,
            offset: int,
            part: int,
            size: int,
        ):
            return self._client.post_artifacts_uuid_continueupload(
                uuid=artifact_upload_uuid,
                body=PostArtifactsUuidContinueuploadBody(
                    file=File(
                        payload=buffer,
                        file_name="artifact.zip",
                        mime_type="application/x-zip",
                    )
                ),
                offset=offset,
                part=part,
                size=size,
            )

        # This should be 10*1024*1024, but there's some problem on the mock API server that rejects the max size.
        CHUNK_SIZE_IN_BYTES = 10000000  # noqa: N806
        NUM_PARALLEL_CHUNK_TASKS = 4  # noqa: N806

        last_successfully_uploaded_chunks: list[int] = []

        # Chunk file & upload
        index = 0
        has_more_chunks_to_upload = True
        part = 1
        bad_request = False

        with open(artifact_filename, "rb") as file:
            while has_more_chunks_to_upload and not bad_request:
                chunk_upload_tasks: list[concurrent.futures.Future] = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_PARALLEL_CHUNK_TASKS) as executor:
                    while len(chunk_upload_tasks) < NUM_PARALLEL_CHUNK_TASKS:
                        buffer = file.read(CHUNK_SIZE_IN_BYTES)

                        # Create local variables in the loop to avoid race condition where part and index get updated
                        # before being passed on to newly created Task.
                        _part = part
                        _index = index
                        size = len(buffer)

                        if not buffer:
                            has_more_chunks_to_upload = False
                            break

                        # Skip the part if its already uploaded in prev attempts
                        if len(last_successfully_uploaded_chunks) > 0 and _part in last_successfully_uploaded_chunks:
                            logging.info(f"QDC continueupload: data for chunk #{_part} is already uploaded.")
                        else:
                            logging.info(
                                f"QDC continueupload: creating a task to upload chunk #{_part}, offset={_index}, size={size}."
                            )
                            chunk_upload_tasks.append(
                                executor.submit(
                                    _do_upload,
                                    artifact_uuid,
                                    buffer,
                                    _index,
                                    _part,
                                    size,
                                )
                            )

                            if size < CHUNK_SIZE_IN_BYTES:
                                has_more_chunks_to_upload = False
                                break

                        part += 1
                        index += size

                    if len(chunk_upload_tasks) > 0:
                        # Wait for tasks to finish
                        for chunk_task in chunk_upload_tasks:
                            r = chunk_task.result()
                            if not r:
                                raise RuntimeError("Error uploading to QDC.")
                            elif r.status_code != 200:
                                raise RuntimeError(f"Error uploading to QDC: status code={r.status_code}.")

    def _schedule_test(self, test_name: str, device_id: str, platform: Platform) -> Run:
        qdc_job_payload = CreateJobType0(
            target_id=device_id,
            job_name=test_name[:32],
            external_job_id=test_name[:32],
            job_type=JobType.AUTOMATED,
            job_mode=JobMode.APPLICATION,
            timeout_in_minutes=self._JOB_TIMEOUT_MINUTES,
            test_framework=self._job_config[platform].test_framework,
            job_artifacts=self._job_config[platform].upload_uuids,
            job_parameters=[JobSubmissionParameter.WIFIENABLED],
        )

        entry_script = self._job_config[platform].entry_script
        if entry_script:
            qdc_job_payload.entry_script = entry_script

        response = self._client.post_jobs(body=qdc_job_payload)

        if response.status_code != 200:
            response_content = response.content.decode("utf-8")
            raise RuntimeError(
                f"Error starting job '{test_name}' on QDC. Response code={response.status_code}: {response_content}"
            )

        if type(response.parsed) is not JobType0 or not response.parsed.job_id:
            raise RuntimeError("QDC jobs response missing required fields.")

        job_id = response.parsed.job_id
        job_url = self._client.get_job_url(job_id)
        logging.info(f"Started QDC job id={job_id}: {test_name}.")
        logging.info(f"  Job URL: {job_url}")

        return QdcRun(self, platform, test_name, job_id, job_url)

    def get_run_status(self, job_id: int, platform: Platform) -> RunStatus:
        # Approximate mapping of state and result
        #
        # JobState                                      JobResult
        # -------------------------------------------------------------------------
        # Submitted, Dispatched, Running, Setup     Pending
        # Completed                                 Successful, Unsuccessful, Error
        # Aborted                                   Aborted

        job_status = self._client.get_job_status(job_id)

        result = job_status.result
        state = job_status.state

        if state == JobState.UNDEFINED:
            return RunStatus.UNKNOWN
        elif state in [JobState.DISPATCHED, JobState.SUBMITTED]:
            return RunStatus.PENDING
        elif state in [JobState.RUNNING, JobState.SETUP]:
            return RunStatus.RUNNING
        elif state == JobState.COMPLETED and result == JobResult.SUCCESSFUL:
            return self._finish_run(job_id, platform, RunStatus.SUCCESS)

        return self._finish_run(job_id, platform, RunStatus.FAILED)


class QdcRun(Run):
    def __init__(
        self,
        runner: QdcRunner,
        platform: Platform,
        test_name: str,
        job_id: int,
        url: str,
    ):
        super().__init__(test_name, url)
        self.platform = platform
        self.runner = runner
        self.job_id = job_id

    def poll(self) -> None:
        self.status = self.runner.get_run_status(self.job_id, self.platform)

    def get_farm_name(self) -> str:
        return "QDC"


#
# Main interface
#


class RunnerCli:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(description="Device Farm test runner.")
        parser.add_argument(
            "--enable-platforms",
            action="store",
            help="Only run tests on these platforms.",
            nargs="+",
            type=lambda p: Platform[p.upper()],
            default=[],
        )
        parser.add_argument(
            "--name",
            action="store",
            help="Test name.",
            required=False,
            default="localtest",
        )
        parser.add_argument(
            "--log-dir",
            action="store",
            type=Path,
            help="If specified, download QDC logs to this directory. %p is expanded to platform name.",
            required=False,
        )
        parser.add_argument(
            "--on-behalf-of",
            action="store",
            help="Set X-QCOM-OnBehalfOf header to QDC job submission request.",
            required=False,
        )
        parser.add_argument(
            "--qdc-api-url",
            action="store",
            help="QDC API URL.",
            required=False,
        )
        parser.add_argument(
            "--qdc-android-id",
            action="store",
            help="Run a QDC test using the specified Android device device ID.",
            required=False,
        )
        parser.add_argument(
            "--qdc-windows-id",
            action="store",
            help="Run a QDC test using the specified Windows device device ID.",
            required=False,
        )
        parser.add_argument(
            "--append-android-package",
            metavar="PACKAGE:SUBDIR",
            type=str,
            action="store",
            help="Append the contents of this package to the Android test archive in the given subdirectory.",
            required=False,
        )
        parser.add_argument(
            "--append-windows-package",
            metavar="PACKAGE:SUBDIR",
            type=str,
            action="store",
            help="Append the contents of this package to the Windows test archive in the given subdirectory.",
            required=False,
        )
        args = parser.parse_args()

        provided_name = args.name[:100].replace("/", "_")

        # use a generous substring to avoid max test and upload name of 256 characters.
        unique_name = time.strftime("%Y%m%d%H%M%S")
        self.test_name = f"ORT-{provided_name}-{unique_name}"
        self.log_dir: Path | None = args.log_dir
        self.on_behalf_of: str | None = args.on_behalf_of

        self.enable_platforms: list[Platform] = args.enable_platforms
        self.qdc_api_url: str = args.qdc_api_url
        self.qdc_android_id: str = args.qdc_android_id
        self.qdc_windows_id: str = args.qdc_windows_id

        self.append_android_package: str = args.append_android_package
        self.append_windows_package: str = args.append_windows_package

        logging.info(f"Initialized runner. Test name prefix={self.test_name}")

    def run(self) -> bool:
        test_runs = TestRuns()

        android_archive = ORT_APPIUM_ARCHIVE
        windows_archive = ORT_TEST_WIN_ARM64

        with tempfile.TemporaryDirectory(prefix="QdcRunner") as tmpdir:
            if self.append_android_package is not None:
                android_archive = self.__append_package(android_archive, Path(tmpdir), self.append_android_package)
            if self.append_windows_package is not None:
                windows_archive = self.__append_package(windows_archive, Path(tmpdir), self.append_windows_package)

            QdcRunner(
                self.test_name,
                self.on_behalf_of,
                self.log_dir,
                test_runs,
                self.qdc_api_url,
                self.enable_platforms,
                self.qdc_android_id,
                self.qdc_windows_id,
                android_archive,
                windows_archive,
            )
            logging.info(f"Initialized runner. Test name prefix={self.test_name}")

            while test_runs.has_pending_tasks():
                test_runs.poll_runs()
                time.sleep(10)

        print()
        print(test_runs.get_status_table())

        if not test_runs.success():
            print()
            print("Some jobs failed.")
            print(test_runs.get_failure_urls())
            return False

        return True

    @classmethod
    def __add_directory(cls, archive_path: Path, addendum_root: Path, prefix: Path) -> None:
        """Add a directory to an archive."""
        for filename in addendum_root.glob("**/*"):
            file_to_archive = filename
            if filename.is_symlink():
                file_to_archive = filename.resolve()
                if filename.is_dir():
                    cls.__add_directory(archive_path, filename, prefix / filename.name)
            else:
                with zipfile.ZipFile(archive_path, mode="a") as archive:
                    arcname = prefix / file_to_archive.relative_to(addendum_root)
                    archive.write(file_to_archive, arcname)

    @classmethod
    def __append_package(cls, orig_archive_path: Path, new_archive_dir: Path, package_and_subdir: str) -> Path:
        """Add the contents of a package into the QDC payload archive."""
        package, subdir = package_and_subdir.split(":")
        logging.info(f"Adding contents of package {package} into payload archive's {subdir} directory.")
        pm = PackageManager(package)
        pm.install()
        new_archive_path = new_archive_dir / orig_archive_path.name
        shutil.copyfile(orig_archive_path, new_archive_path)
        cls.__add_directory(new_archive_path, pm.get_content_dir(), Path(subdir))
        return new_archive_path


log_format = "[%(asctime)s] [qdc_runner.py] [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format, force=True)

runner = RunnerCli()
if not runner.run():
    logging.error("Not all tests were successful.")
    exit(1)

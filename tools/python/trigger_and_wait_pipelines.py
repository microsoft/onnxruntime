# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas
import requests
from azure.kusto.data import DataFormat, KustoConnectionStringBuilder
from azure.kusto.ingest import IngestionProperties, QueuedIngestClient, ReportLevel

logger = logging.getLogger(__name__)

ADO_ORGANIZATION = "aiinfra"
DEFAULT_PROJECT = "Lotus"
POLL_INTERVAL_SECONDS = 60

KUSTO_CLUSTER = "maia-cicd.westus3"
KUSTO_DATABASE = "ci-logs"
KUSTO_TABLE = "onnx_pipeline_run_status_test"


class BuildResult(Enum):
    SUCCEEDED = "succeeded"
    PARTIALLY_SUCCEEDED = "partiallySucceeded"
    FAILED = "failed"
    CANCELED = "canceled"
    NONE = "none"


class BuildState(Enum):
    UNKNOWN = "unknown"
    IN_PROGRESS = "inProgress"
    CANCELING = "canceling"
    COMPLETED = "completed"


@dataclass
class PipelineConfig:
    id: int
    name: str
    project: str = DEFAULT_PROJECT
    template_parameters: dict[str, Any] = field(default_factory=dict)
    variables: dict[str, str] = field(default_factory=dict)


@dataclass
class TriggeredRun:
    config: PipelineConfig
    run_id: int
    web_url: str
    state: BuildState = BuildState.UNKNOWN
    result: BuildResult = BuildResult.NONE


@dataclass
class PipelineStatusRecord:
    Timestamp: str
    OrchestratorBuildId: str
    Branch: str
    PipelineName: str
    PipelineId: int
    RunId: int
    State: str
    Result: str
    Url: str


PIPELINE_REGISTRY: list[PipelineConfig] = [
    PipelineConfig(
        id=841,
        name="Python packaging pipeline",
    ),
    PipelineConfig(
        id=940,
        name="Zip-Nuget-Java-Nodejs Packaging Pipeline",
        template_parameters={
            "IsReleaseBuild": True,
            "NugetPackageSuffix": "NONE",
        },
    ),
    PipelineConfig(
        id=2138,
        name="Nuget - Packaging - CUDA13",
        template_parameters={
            "IsReleaseBuild": True,
            "NugetPackageSuffix": "NONE",
        },
    ),
    PipelineConfig(
        id=1299,
        name="Python-CUDA-Packaging-Pipeline",
    ),
    PipelineConfig(
        id=2104,
        name="Python CUDA 13 Packaging Pipeline",
    ),
    PipelineConfig(
        id=1625,
        name="Python DML Packaging Pipeline",
    ),
    PipelineConfig(
        id=1234,
        name="QNN_Nuget_Windows",
        template_parameters={
            "IsReleaseBuild": True,
            "NugetPackageSuffix": "NONE",
        },
    ),
    PipelineConfig(
        id=1994,
        name="DML Nuget Pipeline",
        template_parameters={
            "DoEsrp": True,
            "IsReleaseBuild": True,
            "NugetPackageSuffix": "NONE",
        },
    ),
    PipelineConfig(
        id=1080,
        name="Npm Packaging Pipeline",
        template_parameters={
            "NpmPublish": "nightly (@dev)",
        },
    ),
    PipelineConfig(
        id=995,
        name="onnxruntime-ios-packaging-pipeline",
        template_parameters={
            "buildType": "release",
        },
    ),
    PipelineConfig(
        id=2107,
        name="WebGPU Python Packaging Pipeline",
    ),
]


def get_token() -> str:
    token = os.environ.get("SYSTEM_ACCESSTOKEN", "").strip()
    if not token:
        logger.error("SYSTEM_ACCESSTOKEN environment variable is not set.")
        sys.exit(1)
    return token


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def trigger_pipeline(config: PipelineConfig, branch: str, token: str) -> TriggeredRun | None:
    logger.info("Triggering '%s' (ID %d) on branch '%s' in project '%s'...", config.name, config.id, branch, config.project)

    run_url = (
        f"https://dev.azure.com/{ADO_ORGANIZATION}/{config.project}"
        f"/_apis/pipelines/{config.id}/runs?api-version=7.1-preview.1"
    )

    payload: dict[str, Any] = {
        "resources": {"repositories": {"self": {"refName": branch}}}
    }

    if config.template_parameters:
        payload["templateParameters"] = {k: str(v) for k, v in config.template_parameters.items()}

    if config.variables:
        payload["variables"] = {k: {"value": str(v)} for k, v in config.variables.items()}

    try:
        resp = requests.post(run_url, headers=_headers(token), data=json.dumps(payload))
        resp.raise_for_status()
        data = resp.json()
        run_id = data["id"]
        web_url = data.get("_links", {}).get("web", {}).get("href", "N/A")
        logger.info("Triggered run %d: %s", run_id, web_url)
        return TriggeredRun(config=config, run_id=run_id, web_url=web_url)
    except requests.exceptions.RequestException as e:
        logger.error("Failed to trigger '%s': %s", config.name, e)
        if hasattr(e, "response") and e.response is not None:
            logger.error("Response: %s", e.response.text)
        return None


def get_run_status(run: TriggeredRun, token: str) -> TriggeredRun:
    url = (
        f"https://dev.azure.com/{ADO_ORGANIZATION}/{run.config.project}"
        f"/_apis/build/builds/{run.run_id}?api-version=7.1"
    )
    try:
        resp = requests.get(url, headers=_headers(token))
        resp.raise_for_status()
        data = resp.json()
        status_str = data.get("status", "unknown")
        result_str = data.get("result", "none") or "none"

        try:
            run.state = BuildState(status_str)
        except ValueError:
            run.state = BuildState.UNKNOWN

        try:
            run.result = BuildResult(result_str)
        except ValueError:
            run.result = BuildResult.NONE

    except requests.exceptions.RequestException as e:
        logger.warning("Failed to get status of run %d ('%s'): %s", run.run_id, run.config.name, e)

    return run


def wait_for_run(
    run: TriggeredRun,
    token: str,
    poll_interval: int,
    branch: str,
    kusto_client: QueuedIngestClient | None,
) -> TriggeredRun:
    logger.info("Waiting for '%s' (run %d) to complete (polling every %ds)...", run.config.name, run.run_id, poll_interval)

    while True:
        time.sleep(poll_interval)
        get_run_status(run, token)
        publish_run_status(run, branch, kusto_client)

        if run.state == BuildState.COMPLETED:
            symbol = "OK" if run.result == BuildResult.SUCCEEDED else "FAIL"
            logger.info("[%s] '%s' (run %d): %s", symbol, run.config.name, run.run_id, run.result.value.upper())
            return run

        logger.info("'%s' (run %d): state=%s", run.config.name, run.run_id, run.state.value)


def _create_kusto_client() -> QueuedIngestClient | None:
    """Create a Kusto ingest client. Returns None if KUSTO_TABLE is not set."""
    if not KUSTO_TABLE:
        logger.info("KUSTO_TABLE is empty — Kusto publishing disabled.")
        return None
    kcsb = KustoConnectionStringBuilder.with_az_cli_authentication(f"https://ingest-{KUSTO_CLUSTER}.kusto.windows.net")
    return QueuedIngestClient(kcsb)


def publish_run_status(
    run: TriggeredRun,
    branch: str,
    kusto_client: QueuedIngestClient | None,
) -> None:
    if kusto_client is None:
        return
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    build_id = os.environ.get("BUILD_BUILDID", "")
    record = PipelineStatusRecord(
        Timestamp=now_str,
        OrchestratorBuildId=build_id,
        Branch=branch,
        PipelineName=run.config.name,
        PipelineId=run.config.id,
        RunId=run.run_id,
        State=run.state.value,
        Result=run.result.value,
        Url=run.web_url,
    )
    ingestion_props = IngestionProperties(
        database=KUSTO_DATABASE,
        table=KUSTO_TABLE,
        data_format=DataFormat.CSV,
        report_level=ReportLevel.FailuresAndSuccesses,
    )
    df = pandas.DataFrame([vars(record)])
    try:
        kusto_client.ingest_from_dataframe(df, ingestion_properties=ingestion_props)
        logger.info(
            "Published status to Kusto: '%s' (run %d) state=%s result=%s",
            run.config.name, run.run_id, run.state.value, run.result.value,
        )
    except Exception as e:
        logger.warning("Failed to publish to Kusto for run %d: %s", run.run_id, e)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trigger an Azure DevOps pipeline and wait for it to complete.",
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="Target branch (without refs/heads/ prefix). Default: main.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=POLL_INTERVAL_SECONDS,
        help=f"Seconds between status polls. Default: {POLL_INTERVAL_SECONDS}.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be triggered without actually triggering.",
    )
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = build_parser()
    args = parser.parse_args()
    branch = f"refs/heads/{args.branch}"

    configs = list(PIPELINE_REGISTRY)

    logger.info("Branch    : %s", branch)
    logger.info("Pipelines : %d", len(configs))

    for cfg in configs:
        logger.info("  [%d] %s (project: %s)", cfg.id, cfg.name, cfg.project)
        if cfg.template_parameters:
            logger.info("       Template params: %s", cfg.template_parameters)

    if args.dry_run:
        logger.info("DRY RUN — no pipelines were triggered.")
        logger.info("Verifying Kusto connectivity...")
        kusto_client = _create_kusto_client()
        if kusto_client is not None:
            test_run = TriggeredRun(
                config=PipelineConfig(id=0, name="dry-run-connectivity-test", project=""),
                run_id=0,
                web_url="",
                state=BuildState.COMPLETED,
                result=BuildResult.SUCCEEDED,
            )
            publish_run_status(test_run, branch, kusto_client)
        else:
            logger.warning("Kusto client could not be created — check configuration.")
        return 0

    token = get_token()
    kusto_client = _create_kusto_client()

    triggered: list[TriggeredRun] = []
    for cfg in configs:
        run = trigger_pipeline(cfg, branch, token)
        if run:
            triggered.append(run)
            publish_run_status(run, branch, kusto_client)
        else:
            logger.error("Failed to trigger '%s'", cfg.name)

    if not triggered:
        logger.error("All pipeline triggers failed.")
        return 1

    pending = list(triggered)
    while pending:
        time.sleep(args.poll_interval)
        still_pending: list[TriggeredRun] = []
        for run in pending:
            get_run_status(run, token)
            publish_run_status(run, branch, kusto_client)
            if run.state == BuildState.COMPLETED:
                symbol = "OK" if run.result == BuildResult.SUCCEEDED else "FAIL"
                logger.info("[%s] '%s' (run %d): %s", symbol, run.config.name, run.run_id, run.result.value.upper())
            else:
                logger.info("'%s' (run %d): state=%s", run.config.name, run.run_id, run.state.value)
                still_pending.append(run)
        pending = still_pending

    any_failed = any(run.result != BuildResult.SUCCEEDED for run in triggered)

    logger.info("=" * 60)
    for run in triggered:
        status = "PASS" if run.result == BuildResult.SUCCEEDED else "FAIL"
        logger.info("[%s] %s (run %d): %s — %s", status, run.config.name, run.run_id, run.result.value, run.web_url)
    logger.info("=" * 60)

    if any_failed:
        logger.error("RESULT: FAILED")
        return 1

    logger.info("RESULT: SUCCESS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import requests

logger = logging.getLogger(__name__)

ADO_ORGANIZATION = "aiinfra"
DEFAULT_PROJECT = "Lotus"
POLL_INTERVAL_SECONDS = 60


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


PIPELINE_REGISTRY: list[PipelineConfig] = [
    PipelineConfig(
        id=841,
        name="Python packaging pipeline",
        project="Lotus",
    ),
]


def get_azure_cli_token() -> str:
    ado_resource_id = "499b84ac-1321-427f-aa17-267ca6975798"
    command = [
        "az.cmd" if sys.platform == "win32" else "az",
        "account",
        "get-access-token",
        "--resource",
        ado_resource_id,
        "--query",
        "accessToken",
        "--output",
        "tsv",
    ]
    try:
        logger.info("Acquiring Azure DevOps token via Azure CLI")
        process = subprocess.run(command, capture_output=True, text=True, check=True, encoding="utf-8")
        token = process.stdout.strip()
        if not token:
            raise ValueError("Token from 'az' command is empty.")
        logger.info("Successfully acquired token.")
        return token
    except FileNotFoundError:
        logger.error("Azure CLI ('az') is not installed in the PATH.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logger.error("Failed to acquire token: %s", e.stderr)
        sys.exit(1)


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


def wait_for_run(run: TriggeredRun, token: str, poll_interval: int) -> TriggeredRun:
    logger.info("Waiting for '%s' (run %d) to complete (polling every %ds)...", run.config.name, run.run_id, poll_interval)

    while True:
        time.sleep(poll_interval)
        get_run_status(run, token)

        if run.state == BuildState.COMPLETED:
            symbol = "OK" if run.result == BuildResult.SUCCEEDED else "FAIL"
            logger.info("[%s] '%s' (run %d): %s", symbol, run.config.name, run.run_id, run.result.value.upper())
            return run

        logger.info("'%s' (run %d): state=%s", run.config.name, run.run_id, run.state.value)


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
        return 0

    token = get_azure_cli_token()

    triggered: list[TriggeredRun] = []
    for cfg in configs:
        run = trigger_pipeline(cfg, branch, token)
        if run:
            triggered.append(run)
        else:
            logger.error("Failed to trigger '%s'", cfg.name)

    if not triggered:
        logger.error("All pipeline triggers failed.")
        return 1

    any_failed = False
    for run in triggered:
        wait_for_run(run, token, args.poll_interval)
        if run.result != BuildResult.SUCCEEDED:
            any_failed = True

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

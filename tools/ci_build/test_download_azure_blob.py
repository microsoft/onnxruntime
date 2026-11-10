# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
ACTION = ROOT / ".github" / "actions" / "download-azure-blob" / "action.yml"
STEPS = yaml.safe_load(ACTION.read_text(encoding="utf-8"))["runs"]["steps"]

# These mocks never invoke Azure, IMDS, or a token endpoint.
MOCKS = r"""
function az {
  if (($args -join ' ') -ne 'version --output json') { throw 'Unexpected Azure CLI operation' }
  $global:LASTEXITCODE = [int]$env:MOCK_CLI_EXIT
  @{ 'azure-cli' = $env:MOCK_CLI_VERSION } | ConvertTo-Json
}
function azcopy.exe {
  if ($args[0] -eq 'env') {
    $global:LASTEXITCODE = 0
    return $env:MOCK_AZCOPY_ENV
  }
  if ($args[0] -ne 'cp') { throw 'Unexpected AzCopy operation' }
  if ($env:AZCOPY_AUTO_LOGIN_TYPE -ne 'AZCLI') { throw 'Must use Azure CLI authentication' }
  if (-not (Test-Path -LiteralPath "$env:AZURE_CONFIG_DIR\mock-token")) { throw 'Login lost before copy' }
  if ($env:AZCOPY_TENANT_ID -ne 'test-tenant') { throw 'Missing tenant' }
  ConvertTo-Json -InputObject @($args) -Compress | Set-Content -LiteralPath $env:MOCK_COPY_ARGUMENTS
  Set-Content -LiteralPath "$env:AZCOPY_LOG_LOCATION\mock-log" -Value 'test'
  Set-Content -LiteralPath "$env:AZCOPY_JOB_PLAN_LOCATION\mock-plan" -Value 'test'
  $global:LASTEXITCODE = [int]$env:MOCK_COPY_EXIT
}
"""


class TestDownloadAzureBlob(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="blob action test ")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.environment = {
            **os.environ,
            "RUNNER_TEMP": str(self.directory),
            "GITHUB_OUTPUT": str(self.directory / "outputs"),
            "BLOB_CLIENT_ID": "test-client",
            "BLOB_TENANT_ID": "test-tenant",
            "BLOB_SOURCE": "https://lotusscus.blob.core.windows.net/models/cuda_sdk/v12.8",
            "BLOB_DESTINATION": str(self.directory / "download with spaces"),
            "MOCK_CLI_VERSION": "2.30.0",
            "MOCK_CLI_EXIT": "0",
            "MOCK_AZCOPY_ENV": "AZCOPY_AUTO_LOGIN_TYPE: DEVICE SPN MSI AZCLI",
            "MOCK_COPY_EXIT": "0",
            "MOCK_COPY_ARGUMENTS": str(self.directory / "copy-arguments.json"),
        }

    def run_step(self, index, **environment):
        script = self.directory / f"step-{index}.ps1"
        script.write_text("$ErrorActionPreference = 'Stop'\n" + MOCKS + STEPS[index]["run"], encoding="utf-8")
        return subprocess.run(
            ["pwsh", "-NoProfile", "-NonInteractive", "-File", str(script)],
            env={**self.environment, **environment},
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )

    def prepare(self):
        result = self.run_step(0)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = Path(self.environment["GITHUB_OUTPUT"]).read_text(encoding="utf-8").splitlines()[-1]
        self.assertTrue(output.startswith("directory="))
        return Path(output.removeprefix("directory="))

    def test_isolated_state_and_cleanup_after_success_or_failure(self):
        unrelated = self.directory / "unrelated-azure"
        unrelated.mkdir()
        (unrelated / "sentinel").touch()
        for exit_code in (0, 23):
            with self.subTest(exit_code=exit_code):
                state = self.prepare()
                self.assertEqual(state.parent, self.directory)
                self.assertTrue(state.name.startswith("blob-download-"))
                for child in ("azure", "logs", "plans"):
                    self.assertTrue((state / child).is_dir())
                (state / "azure" / "mock-token").touch()
                result = self.run_step(
                    2,
                    AZURE_CONFIG_DIR=str(state / "azure"),
                    AZCOPY_AUTO_LOGIN_TYPE=STEPS[2]["env"]["AZCOPY_AUTO_LOGIN_TYPE"],
                    AZCOPY_TENANT_ID="test-tenant",
                    AZCOPY_LOG_LOCATION=str(state / "logs"),
                    AZCOPY_JOB_PLAN_LOCATION=str(state / "plans"),
                    MOCK_COPY_EXIT=str(exit_code),
                )
                self.assertEqual(result.returncode, exit_code, result.stderr)
                arguments = json.loads(Path(self.environment["MOCK_COPY_ARGUMENTS"]).read_text(encoding="utf-8-sig"))
                self.assertEqual(
                    arguments,
                    ["cp", "--recursive", self.environment["BLOB_SOURCE"], self.environment["BLOB_DESTINATION"]],
                )
                result = self.run_step(3, BLOB_STATE_DIRECTORY=str(state))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertFalse(state.exists())
                self.assertTrue((unrelated / "sentinel").exists())

    def test_cleanup_after_login_failure_and_unique_invocations(self):
        first, second = self.prepare(), self.prepare()
        self.assertNotEqual(first, second)
        for state in (first, second):
            (state / "azure" / "partial-login").touch()
            result = self.run_step(3, BLOB_STATE_DIRECTORY=str(state))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(state.exists())

    def test_missing_configuration_fails_before_state_or_login(self):
        for name in ("BLOB_CLIENT_ID", "BLOB_TENANT_ID", "BLOB_SOURCE", "BLOB_DESTINATION"):
            with self.subTest(name=name):
                result = self.run_step(0, **{name: " "})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(f"Missing {name}", result.stderr)
                self.assertFalse(Path(self.environment["GITHUB_OUTPUT"]).exists())

    def test_incompatible_tools_fail_and_state_can_be_cleaned(self):
        for overrides, message in (
            ({"MOCK_CLI_VERSION": "2.29.0"}, "Azure CLI 2.30.0"),
            ({"MOCK_CLI_EXIT": "1"}, "Unable to determine Azure CLI version"),
            ({"MOCK_AZCOPY_ENV": "DEVICE SPN MSI"}, "AzCopy must support"),
        ):
            with self.subTest(overrides=overrides):
                result = self.run_step(0, **overrides)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                output = Path(self.environment["GITHUB_OUTPUT"]).read_text(encoding="utf-8").splitlines()[-1]
                state = Path(output.removeprefix("directory="))
                result = self.run_step(3, BLOB_STATE_DIRECTORY=str(state))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertFalse(state.exists())

    def test_action_authentication_and_cleanup_contract(self):
        login, copy, cleanup = STEPS[1:]
        self.assertEqual(login["uses"], "azure/login@7184910d9eb2b1c5e48f7073824a90609bb9b6d6")
        self.assertEqual(login["with"]["auth-type"], "SERVICE_PRINCIPAL")
        self.assertTrue(login["with"]["allow-no-subscriptions"])
        self.assertEqual(login["with"]["client-id"], "${{ inputs.client-id }}")
        self.assertEqual(login["with"]["tenant-id"], "${{ inputs.tenant-id }}")
        self.assertEqual(login["env"]["AZURE_CONFIG_DIR"], copy["env"]["AZURE_CONFIG_DIR"])
        self.assertEqual(login["env"]["AZURE_LOGIN_POST_CLEANUP"], "false")
        self.assertEqual(login["env"]["AZURE_LOGIN_PRE_CLEANUP"], "false")
        self.assertEqual(cleanup["if"], "always() && steps.state.outputs.directory != ''")
        self.assertNotIn("vars.", ACTION.read_text(encoding="utf-8"))
        self.assertNotIn("GITHUB_ENV", ACTION.read_text(encoding="utf-8"))

    def test_all_sdk_sources_destinations_and_job_permissions(self):
        cuda12 = "https://lotusscus.blob.core.windows.net/models/cuda_sdk/v12.8"
        cuda13 = "https://lotusscus.blob.core.windows.net/models/cuda_sdk/v13.0"
        tensorrt = "https://lotusscus.blob.core.windows.net/models/local/TensorRT-10.14.1.48.Windows.win10.cuda-12.9"
        expected = {
            "windows_cuda.yml": {"build": [cuda12], "test": [cuda12]},
            "windows_cuda_plugin.yml": {"build": [cuda12], "test": [cuda12]},
            "windows_cuda_no_cudnn.yml": {
                "build": [
                    cuda13,
                    "https://lotusscus.blob.core.windows.net/models/cudnn_sdk/${{ env.CUDNN_FOLDER }}",
                ],
                "test": [cuda13],
            },
            "windows_tensorrt.yml": {"build": [cuda12, tensorrt], "test": [cuda12, tensorrt]},
            "windows_gpu_doc_gen.yml": {"runtime_build": [cuda12], "kernel_documentation_generate": [cuda12]},
        }
        for filename, jobs in expected.items():
            path = ROOT / ".github" / "workflows" / filename
            text = path.read_text(encoding="utf-8")
            workflow = yaml.safe_load(text)
            self.assertNotIn("AZCOPY_MSI", text)
            self.assertNotIn("azcopy.exe cp", text)
            for name, job in workflow["jobs"].items():
                with self.subTest(workflow=filename, job=name):
                    downloads = [
                        step for step in job["steps"] if step.get("uses") == "./.github/actions/download-azure-blob"
                    ]
                    self.assertEqual([step["with"]["source"] for step in downloads], jobs.get(name, []))
                    if not downloads:
                        self.assertNotIn("id-token", job.get("permissions", {}))
                    else:
                        self.assertEqual(job["permissions"], {"contents": "read", "id-token": "write"})
                        for step in downloads:
                            self.assertEqual(step["with"]["client-id"], "${{ vars.STORAGE_BLOB_READ_CLIENT_ID }}")
                            self.assertEqual(step["with"]["tenant-id"], "${{ vars.AZURE_TENANT_ID }}")
                            self.assertEqual(step["with"]["destination"], "${{ runner.temp }}")


if __name__ == "__main__":
    if not shutil.which("pwsh"):
        raise SystemExit("PowerShell 7 (pwsh) is required for the offline Blob action tests.")
    unittest.main()

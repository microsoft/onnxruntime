# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import io
import subprocess
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

import mirror_as_universal_package as publisher


class MirrorAsUniversalPackageTest(unittest.TestCase):
    def setUp(self):
        self.content = b"release archive"
        self.sha256 = hashlib.sha256(self.content).hexdigest()
        self.url = "https://example.com/releases/tool.zip?download=1"
        self.filename = "tool.zip"
        self.argv = [
            "mirror_as_universal_package.py",
            "--url",
            self.url,
            "--name",
            "example-tool",
            "--version",
            "4.0.0",
            "--sha256",
            self.sha256,
        ]
        self.package_path = None

    def capture_publish(self, command, **kwargs):
        self.assertEqual(kwargs, {"check": True})
        self.assertEqual(command[:4], ["az.cmd", "artifacts", "universal", "publish"])
        self.package_path = Path(command[command.index("--path") + 1])
        self.assertEqual([path.name for path in self.package_path.iterdir()], [self.filename])
        self.assertEqual((self.package_path / self.filename).read_bytes(), self.content)
        return subprocess.CompletedProcess(command, 0)

    def test_publishes_verified_archive_and_cleans_up(self):
        with (
            patch("sys.argv", self.argv),
            patch.object(publisher.shutil, "which", return_value="az.cmd"),
            patch.object(publisher.urllib.request, "urlopen", return_value=io.BytesIO(self.content)) as download,
            patch.object(publisher.subprocess, "run", side_effect=self.capture_publish) as publish,
        ):
            publisher.main()
        download.assert_called_once_with(self.url, timeout=60)
        command = publish.call_args.args[0]
        self.assertEqual(command[command.index("--version") + 1], "4.0.0")
        self.assertEqual(command[command.index("--name") + 1], "example-tool")
        self.assertEqual(command[command.index("--feed") + 1], "Lotus")
        self.assertEqual(command[command.index("--organization") + 1], "https://dev.azure.com/aiinfra")
        self.assertEqual(command[command.index("--scope") + 1], "organization")
        self.assertNotIn("--project", command)
        self.assertFalse(self.package_path.exists())

    def test_arbitrary_url_package_name_filename_and_feed(self):
        self.filename = "custom.tar.gz"
        argv = [
            "mirror_as_universal_package.py",
            "--url",
            "https://downloads.example.org/download?id=123",
            "--name",
            "another-package",
            "--filename",
            self.filename,
            "--version",
            "1.2.3-rc.1",
            "--sha256",
            self.sha256.upper(),
            "--organization",
            "https://dev.azure.com/example",
            "--feed",
            "tools",
        ]
        with (
            patch("sys.argv", argv),
            patch.object(publisher.shutil, "which", return_value="az.cmd"),
            patch.object(publisher.urllib.request, "urlopen", return_value=io.BytesIO(self.content)) as download,
            patch.object(publisher.subprocess, "run", side_effect=self.capture_publish) as publish,
        ):
            publisher.main()
        download.assert_called_once_with("https://downloads.example.org/download?id=123", timeout=60)
        command = publish.call_args.args[0]
        for flag, value in (
            ("--version", "1.2.3-rc.1"),
            ("--name", "another-package"),
            ("--organization", "https://dev.azure.com/example"),
            ("--feed", "tools"),
            ("--scope", "organization"),
        ):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertNotIn("--project", command)

    def test_url_filename_is_decoded(self):
        argv = self.argv.copy()
        argv[argv.index("--url") + 1] = "https://example.com/my%20tool.zip"
        with patch("sys.argv", argv):
            self.assertEqual(publisher.parse_args().filename, "my tool.zip")

    def test_explicit_filename_allows_url_without_path(self):
        argv = self.argv.copy()
        argv[argv.index("--url") + 1] = "https://example.com"
        argv.extend(["--filename", "tool.zip"])
        with patch("sys.argv", argv):
            self.assertEqual(publisher.parse_args().filename, "tool.zip")

    def test_bad_checksum_prevents_publish(self):
        with (
            patch("sys.argv", self.argv),
            patch.object(publisher.shutil, "which", return_value="az.cmd"),
            patch.object(publisher.urllib.request, "urlopen", return_value=io.BytesIO(b"corrupted archive")),
            patch.object(publisher.subprocess, "run") as publish,
            self.assertRaisesRegex(ValueError, "SHA256 mismatch"),
        ):
            publisher.main()
        publish.assert_not_called()

    def test_publish_failure_is_not_swallowed_and_cleans_up(self):
        def fail_publish(command, **kwargs):
            self.capture_publish(command, **kwargs)
            raise subprocess.CalledProcessError(1, command)

        with (
            patch("sys.argv", self.argv),
            patch.object(publisher.shutil, "which", return_value="az.cmd"),
            patch.object(publisher.urllib.request, "urlopen", return_value=io.BytesIO(self.content)),
            patch.object(publisher.subprocess, "run", side_effect=fail_publish),
            self.assertRaises(subprocess.CalledProcessError),
        ):
            publisher.main()
        self.assertFalse(self.package_path.exists())

    def test_download_failure_prevents_publish(self):
        with (
            patch("sys.argv", self.argv),
            patch.object(publisher.shutil, "which", return_value="az.cmd"),
            patch.object(publisher.urllib.request, "urlopen", side_effect=urllib.error.URLError("download failed")),
            patch.object(publisher.subprocess, "run") as publish,
            self.assertRaises(urllib.error.URLError),
        ):
            publisher.main()
        publish.assert_not_called()

    def test_missing_azure_cli_prevents_download(self):
        with (
            patch("sys.argv", self.argv),
            patch.object(publisher.shutil, "which", return_value=None),
            patch.object(publisher.urllib.request, "urlopen") as download,
            self.assertRaisesRegex(FileNotFoundError, "Azure CLI"),
        ):
            publisher.main()
        download.assert_not_called()

    def test_check_download_only_skips_azure_cli_and_cleans_up(self):
        create_directory = publisher.tempfile.TemporaryDirectory

        def capture_directory(**kwargs):
            directory = create_directory(**kwargs)
            self.package_path = Path(directory.name)
            return directory

        with (
            patch("sys.argv", [*self.argv, "--check-download-only"]),
            patch.object(publisher.shutil, "which") as find_az,
            patch.object(publisher.urllib.request, "urlopen", return_value=io.BytesIO(self.content)) as download,
            patch.object(publisher.subprocess, "run") as publish,
            patch.object(publisher.tempfile, "TemporaryDirectory", side_effect=capture_directory),
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            publisher.main()
        download.assert_called_once_with(self.url, timeout=60)
        find_az.assert_not_called()
        publish.assert_not_called()
        self.assertIn(f"Verified SHA256: {self.sha256}", stdout.getvalue())
        self.assertIn("skipping publication", stdout.getvalue())
        self.assertFalse(self.package_path.exists())

    def test_check_download_only_rejects_bad_checksum(self):
        with (
            patch("sys.argv", [*self.argv, "--check-download-only"]),
            patch.object(publisher.shutil, "which") as find_az,
            patch.object(publisher.urllib.request, "urlopen", return_value=io.BytesIO(b"corrupted archive")),
            patch.object(publisher.subprocess, "run") as publish,
            self.assertRaisesRegex(ValueError, "SHA256 mismatch"),
        ):
            publisher.main()
        find_az.assert_not_called()
        publish.assert_not_called()

    def test_invalid_inputs(self):
        for flag, value in (
            ("--version", "v4.0.0"),
            ("--version", "../4"),
            ("--sha256", "invalid"),
            ("--url", "file:///local.zip"),
            ("--url", "https://example.com/"),
            ("--url", "https://example.com/%2e%2e"),
            ("--url", "https://example.com/..%2fother.zip"),
        ):
            with self.subTest(flag=flag, value=value):
                argv = self.argv.copy()
                argv[argv.index(flag) + 1] = value
                with (
                    patch("sys.argv", argv),
                    patch("sys.stderr", new_callable=io.StringIO),
                    self.assertRaises(SystemExit),
                ):
                    publisher.parse_args()

    def test_invalid_filenames(self):
        for filename in ("", ".", "..", "../file", "..\\file", "C:\\file", "file:stream", "tool.zip.", "tool.zip "):
            with (
                self.subTest(filename=filename),
                patch("sys.argv", [*self.argv, "--filename", filename]),
                patch("sys.stderr", new_callable=io.StringIO),
                self.assertRaises(SystemExit),
            ):
                publisher.parse_args()


if __name__ == "__main__":
    unittest.main()

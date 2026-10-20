# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Host-only compact-review tests and retained raw-byte negative tests."""

import csv
import hashlib
import importlib.util
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("review_results.py")
SPEC = importlib.util.spec_from_file_location("review_results", SCRIPT)
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)
PACKAGE = SCRIPT.parents[1]


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.package = self.root / "package"
        self.package.mkdir()
        shutil.copytree(PACKAGE / "evidence", self.package / "evidence")
        shutil.copy2(PACKAGE / "README.md", self.package / "README.md")

    def test_all_published_medians(self):
        result = review.check_compact(self.package)
        self.assertEqual((result["allocation_runs"], result["earlier_runs"]), (12, 45))
        self.assertEqual(
            [r["process_inference_peak_mib"] for r in result["allocation_medians"]], [8002, 5954, 59848, 27082]
        )
        self.assertEqual(
            [r["post_generator_cleanup_device_capacity_mib"] for r in result["allocation_medians"]],
            [7555, 5507, 59401, 26635],
        )
        self.assertEqual(len(result["earlier_medians"]), 15)
        self.assertEqual(result["phase_b_token_records"], 12)

    def test_byte_comparison_rejects_changes_truncation_and_wrong_digest(self):
        left = struct.pack("<2f", 1.0, 2.0)
        digest = hashlib.sha256(left).hexdigest()
        review.equal_bytes(left, left, 8, digest)
        for right in (struct.pack("<2f", 1.0, 2.0001), left[:-1]):
            with self.assertRaises(ValueError):
                review.equal_bytes(left, right, 8, digest)
        with self.assertRaisesRegex(ValueError, "hash"):
            review.equal_bytes(left, left, 8, "0" * 64)

    def test_duplicate_repetition_rejected(self):
        path = self.package / "evidence/memory-before-after.csv"
        lines = path.read_text().splitlines()
        lines[3] = lines[1]
        path.write_text("\n".join(lines) + "\n")
        with self.assertRaisesRegex(ValueError, "repetitions"):
            review.check_compact(self.package)

    def test_nonfinite_and_negative_measurements_rejected(self):
        path = self.package / "evidence/memory-before-after.csv"
        original = path.read_text()
        for value in ("NaN", "inf", "-1"):
            with self.subTest(value=value):
                path.write_text(original.replace("8384.0", value, 1))
                with self.assertRaisesRegex(ValueError, "Invalid"):
                    review.check_compact(self.package)

    def test_arena_arithmetic_rejected(self):
        path = self.package / "evidence/memory-before-after.csv"
        path.write_text(path.read_text().replace("1325.765625", "1326.765625", 1))
        with self.assertRaisesRegex(ValueError, "Arena"):
            review.check_compact(self.package)

    def test_stale_readme_rejected(self):
        path = self.package / "README.md"
        path.write_text(path.read_text().replace("| 59848 |", "| 59849 |"))
        with self.assertRaisesRegex(ValueError, "Published allocation"):
            review.check_compact(self.package)

    def test_parity_false_or_missing_reference_rejected(self):
        path = self.package / "evidence/parity-checks.json"
        original = json.loads(path.read_text())
        for field, value in (("matches_phase_a", False), ("sha256", "0" * 64)):
            modified = json.loads(json.dumps(original))
            modified["generated_tokens"]["phase_b_runs"][0][field] = value
            path.write_text(json.dumps(modified))
            with self.assertRaisesRegex(ValueError, "reference"):
                review.check_compact(self.package)

    def test_single_earlier_input_change_rejected_even_if_median_unchanged(self):
        path = self.package / "evidence/earlier-runtime-comparison.csv"
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            columns = reader.fieldnames
            rows = list(reader)
        rows[0]["prompt_tokens"] = str(int(rows[0]["prompt_tokens"]) + 1)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "Earlier input"):
            review.check_compact(self.package)

    def test_duplicate_token_run_rejected(self):
        path = self.package / "evidence/parity-checks.json"
        value = json.loads(path.read_text())
        runs = value["generated_tokens"]["phase_b_runs"]
        runs[2] = runs[0]
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "repetitions"):
            review.check_compact(self.package)

    def test_csv_quoted_scalars_supported_and_duplicate_header_rejected(self):
        path = self.root / "quoted.csv"
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream, quoting=csv.QUOTE_ALL)
            writer.writerow(["variant", "number"])
            writer.writerow(["baseline", "1"])
        self.assertEqual(review.read_csv(path, {"variant"}), [{"variant": "baseline", "number": 1.0}])
        path.write_text("variant,number,number\nbaseline,1,2\n")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            review.read_csv(path, {"variant"})

    def test_integrity_checks_content_coverage_and_traversal(self):
        files = sorted(p for p in self.package.rglob("*") if p.is_file())
        manifest = self.package / "SHA256SUMS"
        manifest.write_text(
            "".join(
                f"{review.digest(file.read_bytes())}  {file.relative_to(self.package).as_posix()}\n" for file in files
            )
        )
        self.assertEqual(review.verify_integrity(self.package), len(files))
        extra = self.package / "unlisted.txt"
        extra.write_text("not hashed\n")
        with self.assertRaisesRegex(ValueError, "coverage"):
            review.verify_integrity(self.package)
        extra.unlink()
        first = files[0]
        first.write_bytes(first.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            review.verify_integrity(self.package)
        manifest.write_text(f"{'0' * 64}  ../external\n")
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            review.verify_integrity(self.package)

    def test_git_roundtrip_preserves_historical_patch_and_package_integrity(self):
        patch_hashes = {
            "repro/allocation/historical-tests-normalized.patch": "1485beda53090cc53f4c1cfc0a19c3938d2f4f720a881632e7398d25eb7f038b",
            "repro/allocation/production.diff": "34723df6327bb3430ff00a05d8329d3c69660ceaf67a7efd3d2b268314124e38",
        }
        patches = {name: (PACKAGE / name).read_bytes() for name in patch_hashes}
        self.assertIn(b"\r\n", patches["repro/allocation/historical-tests-normalized.patch"])
        for name, expected_hash in patch_hashes.items():
            self.assertEqual(hashlib.sha256(patches[name]).hexdigest(), expected_hash)
        env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_OPTIONAL_LOCKS="0")

        def git(directory, *args):
            result = subprocess.run(
                ["git", "-c", f"core.attributesfile={os.devnull}", "-C", str(directory), *args],
                env=env,
                capture_output=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            return result.stdout

        for conversion in ("false", "input", "true"):
            with self.subTest(autocrlf=conversion):
                root = self.root / conversion
                repository = root / "repository"
                repository.mkdir(parents=True)
                (repository / ".gitattributes").write_text("* text=auto\n", encoding="utf-8")
                shutil.copytree(PACKAGE, repository / "package", ignore=shutil.ignore_patterns("__pycache__"))
                git(repository, "init", "--quiet")
                git(repository, "-c", f"core.autocrlf={conversion}", "add", "--all")
                for name, expected in patches.items():
                    self.assertEqual(git(repository, "show", f":package/{name}"), expected)

                native_checkout = root / "native-checkout"
                git(
                    repository,
                    "-c",
                    f"core.autocrlf={conversion}",
                    "checkout-index",
                    f"--prefix={native_checkout}/",
                    "--",
                    *(f"package/{name}" for name in patches),
                )
                for name, expected in patches.items():
                    self.assertEqual((native_checkout / "package" / name).read_bytes(), expected)

                # Full package checksums describe the documented Linux/LF checkout.
                checkout = root / "checkout"
                git(
                    repository,
                    "-c",
                    "core.autocrlf=false",
                    "-c",
                    "core.eol=lf",
                    "checkout-index",
                    "--all",
                    f"--prefix={checkout}/",
                )
                self.assertEqual(review.verify_integrity(checkout / "package"), review.verify_integrity(PACKAGE))
                exported = root / "staged.patch"
                exported.write_bytes(
                    git(repository, "diff", "--cached", "--binary", "--full-index", "--no-ext-diff", "--no-textconv")
                )
                applied = root / "applied"
                applied.mkdir()
                git(applied, "apply", "--check", str(exported))
                git(applied, "apply", "--whitespace=nowarn", str(exported))
                for name, expected in patches.items():
                    self.assertEqual((applied / "package" / name).read_bytes(), expected)
                self.assertEqual(review.verify_integrity(applied / "package"), review.verify_integrity(PACKAGE))

    def cli(self, *args):
        return subprocess.run(
            [sys.executable, "-B", str(SCRIPT), "--package", str(self.package), *map(str, args)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    def test_cli_distinguishes_unperformed_checks(self):
        result = self.cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["package_integrity"]["status"], "not_run")
        self.assertTrue(report["raw_vector_comparison"].startswith("not_run"))
        self.assertEqual(report["fresh_gpu_reproduction"], "not_run")

    def test_missing_raw_input_fails_not_skips(self):
        result = self.cli("--raw-captures", self.root / "missing")
        self.assertEqual(result.returncode, 2)
        self.assertIn("missing", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_missing_evidence_fails(self):
        (self.package / "evidence/parity-checks.json").unlink()
        result = self.cli()
        self.assertEqual(result.returncode, 2)
        self.assertIn("Evidence review failed", result.stderr)

    def test_output_is_exclusive_and_outside_package(self):
        output = self.root / "review.json"
        self.assertEqual(self.cli("--output", output).returncode, 0)
        original = output.read_bytes()
        self.assertEqual(self.cli("--output", output).returncode, 2)
        self.assertEqual(output.read_bytes(), original)
        self.assertEqual(self.cli("--output", self.package / "new.json").returncode, 2)


if __name__ == "__main__":
    unittest.main()

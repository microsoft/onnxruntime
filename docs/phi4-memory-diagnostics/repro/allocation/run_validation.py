# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Regression and four Phase A captures; invoked only by allocation.py run."""

from pathlib import Path

import config as settings
import monitor
import support as s

RELATED_TESTS = ":".join(
    (
        "ModelTests.GreedySearchGptFp32",
        "ModelTests.BeamSearchGptFp32",
        "ModelTests.GreedySearchGptCuda",
        "ModelTests.BeamSearchGptCuda",
        "CAPITests.GetOutputCAPI",
        "CAPITests.GetLogitsCAPI",
        "CAPITests.SetLogitsCAPI",
    )
)
TEST_JOBS = (
    ("baseline-allocation", "baseline", "allocation_tests", "*NoPromptSizedFp32*", 9, True),
    ("baseline-semantics", "baseline", "allocation_tests", "*-*NoPromptSizedFp32*", 20, False),
    ("patched-regression", "patched", "allocation_tests", "*", 29, False),
    ("baseline-existing", "baseline", "existing_tests", RELATED_TESTS, 7, False),
    ("patched-existing", "patched", "existing_tests", RELATED_TESTS, 7, False),
)


def tests(config, root, nv, handle, identities):
    root.mkdir()
    records = {}
    for name, variant, key, selector, count, expected_failure in TEST_JOBS:
        directory = root / name
        directory.mkdir()
        executable = config["runtimes"][variant][key]
        settings.verify_native(executable, "tests")
        s.require(settings.runtime_hashes(config, variant) == identities[variant], "Runtime changed before tests")
        command = [executable["path"], f"--gtest_filter={selector}", f"--gtest_output=xml:{directory / 'gtest.xml'}"]
        state, _ = monitor.launch(config, directory, command, variant, nv, handle, cwd=Path(executable["path"]).parent)
        records[name] = s.test_result(directory / "gtest.xml", state["exit_code"], count, expected_failure)
        s.save(directory / "classified-result.json", records[name])
        s.require(settings.runtime_hashes(config, variant) == identities[variant], "Runtime changed during tests")
    s.save(root / "gate.json", {"passed": True, "results": records, "runtime_sha256": identities})


def prepare(config, directory, context, variant, phase):
    raw = Path(config["inputs"][str(context)]).read_bytes()
    s.validate_input(raw, context)
    (directory / "input-ids.i32le").write_bytes(raw)
    workload = s.load(s.MANIFEST)["workloads"][str(context)]
    effective = workload["effective_config"]
    s.verify_config(effective, context)
    source = Path(config["model_dir"]) / "genai_config.json"
    s.require(s.sha256(source) == workload["source_config_sha256"], "Model source configuration changed")
    s.save(directory / "source-config.json", s.load(source))
    s.save(directory / "effective-config.json", effective)
    executable = config["phase_b"] if phase == "b" else config["runtimes"][variant]["phase_a"]
    settings.verify_native(executable, phase)
    identity = {
        "input_sha256": s.INPUT_HASHES[context],
        "input_tokens": s.COUNTS[context],
        "source_config_sha256": s.sha256(source),
        "effective_config_sha256": s.sha256(directory / "effective-config.json"),
        "runtime_sha256": settings.runtime_hashes(config, variant),
        "executable_sha256": executable["sha256"],
        "gpu_uuid": config["gpu_uuid"],
        "physical_gpu_index": config["gpu_index"],
        "phase": phase,
    }
    s.save(directory / "identity.json", identity)

    def ready():
        s.require((directory / "loaded-input-ids.i32le").read_bytes() == raw, "App input mismatch")
        s.require(s.load(directory / "applied-config.json") == effective, "Applied config mismatch")
        s.require(s.load(directory / "search-readback.json") == effective["search"], "Search readback mismatch")
        s.require(s.sha256(executable["path"]) == executable["sha256"], "Executable changed before GO")
        s.save(
            directory / "loaded-hashes-before.json",
            s.verify_maps(directory / "loaded-maps-before.txt", identity["runtime_sha256"]),
        )

    command = [
        executable["path"],
        config["model_dir"],
        config["ort_home"],
        config["runtimes"][variant]["genai_library_dir"],
        str(directory / "input-ids.i32le"),
        str(context),
        str(directory),
        str(directory / "effective-config.json"),
    ]
    return identity, command, ready


def verify_capture_state(directory, context, phase_b):
    expected = {
        "nominal_context": context,
        "input_tokens": s.COUNTS[context],
        "sequence_tokens": s.COUNTS[context] + 64,
        "generated_tokens": 64,
        "max_length": s.COUNTS[context] + 64,
        "chunk_size": 0,
        "shrink_calls": 0,
    }
    if phase_b:
        expected.update(diagnostic_logits_calls=0, model_alive_at_cleanup_snapshot=True)
    s.require(s.load(directory / "capture-state.json") == expected, "Capture workload mismatch")


def phase_a(config, root, nv, handle, identities):
    gate = s.load(root.parent / "test-results/gate.json")
    s.require(gate["passed"] is True and gate["runtime_sha256"] == identities, "Regression gate not valid")
    root.mkdir()
    comparisons = {}
    for context in s.COUNTS:
        directories = []
        for variant in s.VARIANTS:
            directory = root / f"{variant}-{context}"
            directory.mkdir()
            identity, command, ready = prepare(config, directory, context, variant, "a")
            s.require(identity["runtime_sha256"] == identities[variant], "Runtime changed after tests")
            state, _ = monitor.launch(config, directory, command, variant, nv, handle, "a", ready)
            s.require(state["exit_code"] == 0, f"Phase A failed: {directory}")
            s.save(
                directory / "loaded-hashes-after.json",
                s.verify_maps(directory / "loaded-maps-after.txt", identities[variant]),
            )
            s.verify_logits(directory)
            s.verify_tokens(directory)
            verify_capture_state(directory, context, False)
            directories.append(directory)
        comparisons[str(context)] = s.compare_phase_a(*directories)
        s.save(root / "comparisons.json", comparisons)
    s.save(
        root / "complete.json",
        {"captures": 4, "comparisons": comparisons, "phase_b_runs": 0, "runtime_sha256": identities},
    )


def verify_phase_a(root, identities):
    """Recheck real capture files, not just a boolean completion marker."""
    complete = s.load(root / "complete.json")
    s.require(
        complete["captures"] == 4 and complete["phase_b_runs"] == 0 and complete["runtime_sha256"] == identities,
        "Phase A identity/completion mismatch",
    )
    gate = s.load(root.parent / "test-results/gate.json")
    s.require(gate["passed"] is True and gate["runtime_sha256"] == identities, "Regression gate invalid")
    for name, _, _, _, count, expected_failure in TEST_JOBS:
        directory = root.parent / "test-results" / name
        result = s.test_result(
            directory / "gtest.xml", s.load(directory / "status.json")["exit_code"], count, expected_failure
        )
        s.require(result == gate["results"][name], "Regression classification changed")
    for context in s.COUNTS:
        directories = [root / f"{variant}-{context}" for variant in s.VARIANTS]
        s.require(
            s.compare_phase_a(*directories) == complete["comparisons"][str(context)],
            "Phase A correctness files changed",
        )
        for variant, directory in zip(s.VARIANTS, directories, strict=True):
            identity = s.load(directory / "identity.json")
            s.require(identity["runtime_sha256"] == identities[variant], "Phase A runtime mismatch")
            s.require(s.load(directory / "status.json")["exit_code"] == 0, "Phase A child did not succeed")
            s.validate_input((directory / "input-ids.i32le").read_bytes(), context)
            s.require(
                (directory / "loaded-input-ids.i32le").read_bytes() == (directory / "input-ids.i32le").read_bytes(),
                "Phase A loaded input changed",
            )
            verify_capture_state(directory, context, False)
            for when in ("before", "after"):
                loaded = s.load(directory / f"loaded-hashes-{when}.json")
                s.require(
                    {path: digest for path, digest in loaded.items() if Path(path).name.startswith(s.RUNTIME_PREFIXES)}
                    == identities[variant],
                    "Phase A loaded runtime identity changed",
                )
    return complete

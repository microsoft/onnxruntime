# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import subprocess
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

from prune_macos_sccache import BUDGET_BYTES, MAIN_REF, PILOT_REF, gh_api, list_caches, run, select_evictions


def cache(cache_id, size, variant="a", **overrides):
    return {
        "id": cache_id,
        "key": f"sccache-v1-macOS-ARM64-{variant * 64}-{cache_id:040x}",
        "ref": MAIN_REF,
        "size_in_bytes": size,
        "created_at": (datetime(2026, 11, 1, tzinfo=timezone.utc) + timedelta(minutes=cache_id)).isoformat(),
        **overrides,
    }


def pilot(cache_id, size):
    return cache(cache_id, size, ref=PILOT_REF, key=f"macos-sccache-v2-arm64-arm64-Debug-{'a' * 64}-{'b' * 40}")


def ids(plan):
    return [entry["id"] for entry in plan.deletions]


def pages(entries):
    return json.dumps([{"actions_caches": entries}])


class PruneMacosSccacheTest(unittest.TestCase):
    def test_scope_excludes_other_namespaces_and_refs(self):
        entries = [
            cache(1, 10, key="vcpkg-macos-cache"),
            cache(2, 10, key=cache(2, 10)["key"].replace("macOS", "Linux")),
            cache(3, 10, key=cache(3, 10)["key"].replace("macOS", "Windows")),
            cache(4, 10, ref="refs/pull/42/merge"),
            cache(5, 10, ref="refs/heads/feature"),
            cache(6, 10, key="sccache-v1-macOS-unrecognized"),
            {**pilot(7, 10), "ref": MAIN_REF},
            {**pilot(8, 10), "ref": "refs/heads/feature"},
            cache(9, 10),
            pilot(10, 10),
        ]
        self.assertEqual(ids(select_evictions(entries, 0)), [10, 9])

    def test_empty_exact_and_under_budget_need_no_eviction(self):
        self.assertEqual(ids(select_evictions([], 0)), [])
        self.assertEqual(ids(select_evictions([cache(1, BUDGET_BYTES)])), [])
        self.assertEqual(ids(select_evictions([cache(1, 1)])), [])

    def test_intel_and_both_pilot_versions_are_included(self):
        intel = cache(1, 10, key=cache(1, 10)["key"].replace("ARM64", "X64"))
        old_pilot = {**pilot(2, 10), "key": pilot(2, 10)["key"].replace("-v2-", "-v1-")}
        self.assertEqual(ids(select_evictions([intel, old_pilot, pilot(3, 10)], 0)), [2, 3, 1])

    def test_aggregate_budget_preserves_newest_per_prefix(self):
        entries = [cache(1, 8), cache(2, 8, "b"), cache(3, 8), cache(4, 8, "b"), pilot(5, 8)]
        plan = select_evictions(entries[::-1], 24)
        self.assertEqual(ids(plan), [5, 1])
        self.assertEqual(plan.total_bytes, 40)
        self.assertEqual(plan.remaining_bytes, 24)
        self.assertEqual(plan.evicted_baselines, [])

    def test_multiple_generations_and_prefixes_are_independent(self):
        entries = [cache(i + 1, 1_000_000_000, f"{i % 10:x}") for i in range(170)]
        plan = select_evictions(entries)
        self.assertEqual(len(plan.deletions), 140)
        self.assertEqual(plan.remaining_bytes, BUDGET_BYTES)
        self.assertEqual(ids(plan), list(range(1, 141)))
        self.assertEqual(plan.evicted_baselines, [])

    def test_oversized_protected_set_evicts_oldest_baselines(self):
        plan = select_evictions([cache(1, 20), cache(2, 20, "b")], 30)
        self.assertEqual(ids(plan), [1])
        self.assertEqual([entry["id"] for entry in plan.evicted_baselines], [1])
        self.assertEqual(ids(select_evictions([cache(1, 31)], 30)), [1])

    def test_equal_creation_times_break_ties_by_id(self):
        first = cache(1, 10)
        second = cache(2, 10, created_at=first["created_at"], last_accessed_at="2000-01-01T00:00:00Z")
        self.assertEqual(ids(select_evictions([second, first], 10)), [1])

    def test_invalid_budget_and_scoped_metadata_fail_closed(self):
        for budget in [-1, float("nan"), 1.5, True]:
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                select_evictions([], budget)
        for overrides in [
            {"id": -1},
            {"id": True},
            {"size_in_bytes": -1},
            {"size_in_bytes": False},
            {"created_at": "invalid"},
            {"created_at": "2026-11-01T00:00:00"},
        ]:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                select_evictions([cache(1, 1, **overrides)])

    def test_pagination_and_duplicate_ids(self):
        api = Mock(return_value=json.dumps([{"actions_caches": [cache(1, 1)]}, {"actions_caches": [cache(2, 1)]}]))
        self.assertEqual(len(list_caches("owner/repo", api)), 2)
        self.assertEqual(api.call_count, 2)
        for call in api.call_args_list:
            self.assertIn("--paginate", call.args[0])
            self.assertIn("--slurp", call.args[0])
        self.assertIn("per_page=100&key=sccache-v1-macOS-", api.call_args_list[0].args[0][-1])
        self.assertIn("per_page=100&key=macos-sccache-v", api.call_args_list[1].args[0][-1])

    def test_dry_run_never_deletes_and_reports_baseline_loss(self):
        api = Mock(return_value=pages([cache(1, BUDGET_BYTES + 1)]))
        log = Mock()
        run("owner/repo", api=api, log=log)
        self.assertEqual(api.call_count, 2)
        self.assertTrue(all("DELETE" not in call.args[0] for call in api.call_args_list))
        self.assertTrue(any("::warning::" in call.args[0] for call in log.call_args_list))
        self.assertTrue(any("Would delete cache 1" in call.args[0] for call in log.call_args_list))

    def test_apply_deletes_exact_ids_and_rechecks_aggregate(self):
        before = pages([cache(1, BUDGET_BYTES), cache(2, 1)])
        after = pages([cache(2, 1)])
        api = Mock(side_effect=[before, before, "", after, after])
        run("owner/repo", apply=True, api=api, log=Mock())
        self.assertEqual(api.call_count, 5)
        self.assertEqual(api.call_args_list[2].args[0], ["--method", "DELETE", "repos/owner/repo/actions/caches/1"])
        self.assertTrue(all("DELETE" not in call.args[0] for call in api.call_args_list[:2]))
        self.assertTrue(all("DELETE" not in call.args[0] for call in api.call_args_list[3:]))

    def test_listing_malformed_response_and_delete_errors_propagate(self):
        with self.assertRaisesRegex(RuntimeError, "403"):
            run("owner/repo", api=Mock(side_effect=RuntimeError("403")), log=Mock())
        for response in ["bad JSON", "[]", "[{}]", "[null]"]:
            with self.subTest(response=response), self.assertRaises(ValueError):
                list_caches("owner/repo", Mock(return_value=response))
        before = pages([cache(1, BUDGET_BYTES + 1)])
        with self.assertRaisesRegex(RuntimeError, "503"):
            run("owner/repo", apply=True, api=Mock(side_effect=[before, before, RuntimeError("503")]), log=Mock())

    def test_second_namespace_failure_prevents_deletion(self):
        api = Mock(side_effect=[pages([cache(1, BUDGET_BYTES + 1)]), RuntimeError("Second namespace failed")])
        with self.assertRaisesRegex(RuntimeError, "Second namespace failed"):
            run("owner/repo", apply=True, api=api, log=Mock())
        self.assertEqual(api.call_count, 2)
        self.assertTrue(all("DELETE" not in call.args[0] for call in api.call_args_list))

    def test_concurrent_uploads_fail_post_cleanup_check(self):
        before = pages([cache(1, BUDGET_BYTES + 1)])
        with self.assertRaisesRegex(RuntimeError, "remains over budget"):
            run("owner/repo", apply=True, api=Mock(side_effect=[before, before, "", before, before]), log=Mock())

    def test_gh_api_invocation_and_process_errors(self):
        with patch("prune_macos_sccache.subprocess.check_output", return_value="[]") as check_output:
            self.assertEqual(gh_api(["--paginate"]), "[]")
            check_output.assert_called_once_with(["gh", "api", "--paginate"], text=True, encoding="utf-8")
        with (
            patch(
                "prune_macos_sccache.subprocess.check_output",
                side_effect=subprocess.CalledProcessError(1, ["gh", "api"]),
            ),
            self.assertRaises(subprocess.CalledProcessError),
        ):
            gh_api([])


if __name__ == "__main__":
    unittest.main()

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import subprocess
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

from prune_sccache import BUDGET_BYTES, MAIN_REF, gh_api, list_caches, run, select_evictions


def cache(cache_id, size, variant="a", os_name="macOS", arch="ARM64", **overrides):
    return {
        "id": cache_id,
        "key": f"sccache-v1-{os_name}-{arch}-{variant * 64}-{cache_id:040x}",
        "ref": MAIN_REF,
        "size_in_bytes": size,
        "created_at": (datetime(2026, 11, 1, tzinfo=timezone.utc) + timedelta(minutes=cache_id)).isoformat(),
        **overrides,
    }


def ids(plan):
    return [entry["id"] for entry in plan.deletions]


def pages(entries):
    return json.dumps([{"actions_caches": entries}])


class PruneSccacheTest(unittest.TestCase):
    def test_scope_excludes_other_namespaces_and_refs(self):
        entries = [
            cache(1, 10, key="vcpkg-macos-cache"),
            cache(2, 10, key="codeql-cache"),
            cache(3, 10, key=cache(3, 10)["key"].replace("sccache-v1-", "sccache-v2-")),
            cache(4, 10, ref="refs/pull/42/merge"),
            cache(5, 10, ref="refs/heads/feature"),
            cache(6, 10, key="sccache-v1-macOS-unrecognized"),
            cache(7, 10, key=f"macos-sccache-v1-arm64-Debug-{'a' * 64}-{'b' * 40}"),
            cache(8, 10, key=f"macos-sccache-v2-arm64-Debug-{'a' * 64}-{'b' * 40}"),
            cache(9, 10),
            cache(10, 10, os_name="Linux", ref="refs/heads/feature"),
        ]
        self.assertEqual(ids(select_evictions(entries, 0)), [9])

    def test_empty_exact_and_under_budget_need_no_eviction(self):
        self.assertEqual(BUDGET_BYTES, 150_000_000_000)
        self.assertEqual(ids(select_evictions([], 0)), [])
        self.assertEqual(ids(select_evictions([cache(1, BUDGET_BYTES)])), [])
        self.assertEqual(ids(select_evictions([cache(1, 1)])), [])

    def test_each_os_has_an_independent_budget(self):
        entries = [
            cache(1, 10, os_name="macOS", arch="X64"),
            cache(2, 10, os_name="Linux", arch="ARM64"),
            cache(3, 10, os_name="Windows", arch="X64"),
            cache(4, 10, variant="b", os_name="Linux", arch="ARM64"),
            cache(5, 10, os_name="FutureOS", arch="FutureArch"),
        ]
        plan = select_evictions(entries, 30)
        self.assertEqual(plan.total_bytes, 50)
        self.assertEqual(plan.remaining_bytes, 50)
        self.assertEqual(ids(plan), [])
        self.assertEqual(plan.total_bytes_by_os, {"macOS": 10, "Linux": 20, "Windows": 10, "FutureOS": 10})

    def test_os_budget_combines_architectures_and_build_variants(self):
        entries = [
            cache(1, 20, os_name="Linux", arch="X64"),
            cache(2, 20, variant="b", os_name="Linux", arch="ARM64"),
            cache(3, 10, os_name="Windows"),
            cache(4, 10, os_name="macOS"),
        ]
        plan = select_evictions(entries, 30)
        self.assertEqual(ids(plan), [1])
        self.assertEqual(plan.remaining_bytes_by_os, {"Linux": 20, "Windows": 10, "macOS": 10})
        self.assertEqual([entry["id"] for entry in plan.evicted_baselines], [1])

    def test_each_over_budget_os_is_evicted_independently(self):
        entries = [
            cache(1, 20, os_name="Linux"),
            cache(2, 20, os_name="Windows"),
            cache(3, 20, os_name="Linux"),
            cache(4, 20, os_name="Windows"),
            cache(5, 1, os_name="macOS"),
        ]
        plan = select_evictions(entries, 30)
        self.assertEqual(ids(plan), [1, 2])
        self.assertEqual(plan.remaining_bytes_by_os, {"Linux": 20, "Windows": 20, "macOS": 1})
        self.assertEqual(plan.evicted_baselines, [])

    def test_aggregate_budget_preserves_newest_per_prefix(self):
        entries = [cache(1, 8), cache(2, 8, os_name="Linux"), cache(3, 8), cache(4, 8, os_name="Linux")]
        plan = select_evictions(entries[::-1], 8)
        self.assertEqual(ids(plan), [1, 2])
        self.assertEqual(plan.total_bytes, 32)
        self.assertEqual(plan.remaining_bytes, 16)
        self.assertEqual(plan.evicted_baselines, [])

    def test_multiple_generations_and_prefixes_are_independent(self):
        entries = [cache(i + 1, BUDGET_BYTES // 30, f"{i % 10:x}") for i in range(170)]
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
        api = Mock(
            return_value=json.dumps([{"actions_caches": [cache(1, 1)]}, {"actions_caches": [cache(1, 1), cache(2, 1)]}])
        )
        self.assertEqual(len(list_caches("owner/repo", api)), 2)
        self.assertEqual(api.call_count, 1)
        for call in api.call_args_list:
            self.assertIn("--paginate", call.args[0])
            self.assertIn("--slurp", call.args[0])
        self.assertIn("per_page=100&key=sccache-v1-", api.call_args_list[0].args[0][-1])

    def test_dry_run_never_deletes_and_reports_baseline_loss(self):
        api = Mock(return_value=pages([cache(1, BUDGET_BYTES + 1)]))
        log = Mock()
        run("owner/repo", api=api, log=log)
        self.assertEqual(api.call_count, 1)
        self.assertTrue(all("DELETE" not in call.args[0] for call in api.call_args_list))
        self.assertTrue(any("::warning::" in call.args[0] for call in log.call_args_list))
        self.assertTrue(any("Would delete cache 1" in call.args[0] for call in log.call_args_list))

    def test_apply_deletes_exact_ids_and_rechecks_aggregate(self):
        before = pages([cache(1, BUDGET_BYTES), cache(2, 1)])
        after = pages([cache(2, 1)])
        api = Mock(side_effect=[before, "", after])
        run("owner/repo", apply=True, api=api, log=Mock())
        self.assertEqual(api.call_count, 3)
        self.assertEqual(api.call_args_list[1].args[0], ["--method", "DELETE", "repos/owner/repo/actions/caches/1"])
        self.assertNotIn("DELETE", api.call_args_list[0].args[0])
        self.assertNotIn("DELETE", api.call_args_list[2].args[0])

    def test_listing_malformed_response_and_delete_errors_propagate(self):
        with self.assertRaisesRegex(RuntimeError, "403"):
            run("owner/repo", api=Mock(side_effect=RuntimeError("403")), log=Mock())
        for response in ["bad JSON", "[]", "[{}]", "[null]"]:
            with self.subTest(response=response), self.assertRaises(ValueError):
                list_caches("owner/repo", Mock(return_value=response))
        before = pages([cache(1, BUDGET_BYTES + 1)])
        with self.assertRaisesRegex(RuntimeError, "503"):
            run("owner/repo", apply=True, api=Mock(side_effect=[before, RuntimeError("503")]), log=Mock())

    def test_later_page_failure_prevents_deletion(self):
        # gh --paginate fails the entire command if any page request fails.
        error = subprocess.CalledProcessError(1, ["gh", "api"], output=pages([cache(1, BUDGET_BYTES + 1)]))
        api = Mock(side_effect=error)
        with self.assertRaises(subprocess.CalledProcessError):
            run("owner/repo", apply=True, api=api, log=Mock())
        self.assertEqual(api.call_count, 1)
        self.assertTrue(all("DELETE" not in call.args[0] for call in api.call_args_list))

    def test_concurrent_uploads_fail_post_cleanup_check(self):
        before = pages([cache(1, BUDGET_BYTES + 1)])
        with self.assertRaisesRegex(RuntimeError, "remains over budget"):
            run("owner/repo", apply=True, api=Mock(side_effect=[before, "", before]), log=Mock())

    def test_post_cleanup_allows_combined_storage_above_one_os_budget(self):
        inventory = pages([cache(1, BUDGET_BYTES), cache(2, BUDGET_BYTES, os_name="Linux")])
        api = Mock(return_value=inventory)
        plan = run("owner/repo", apply=True, api=api, log=Mock())
        self.assertEqual(plan.total_bytes, 2 * BUDGET_BYTES)
        self.assertEqual(ids(plan), [])
        self.assertEqual(api.call_count, 2)
        self.assertTrue(all("DELETE" not in call.args[0] for call in api.call_args_list))

    def test_gh_api_invocation_and_process_errors(self):
        with patch("prune_sccache.subprocess.check_output", return_value="[]") as check_output:
            self.assertEqual(gh_api(["--paginate"]), "[]")
            check_output.assert_called_once_with(["gh", "api", "--paginate"], text=True, encoding="utf-8")
        with (
            patch(
                "prune_sccache.subprocess.check_output",
                side_effect=subprocess.CalledProcessError(1, ["gh", "api"]),
            ),
            self.assertRaises(subprocess.CalledProcessError),
        ):
            gh_api([])


if __name__ == "__main__":
    unittest.main()

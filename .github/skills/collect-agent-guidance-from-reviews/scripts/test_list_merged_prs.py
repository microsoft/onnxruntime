import unittest
from contextlib import redirect_stderr
from datetime import datetime, timezone
from io import StringIO
from unittest import mock

import list_merged_prs
from collection_marker import MARKER_PATTERN, generate_marker

UTC = timezone.utc


def timestamp(day: int, hour: int = 0) -> datetime:
    return datetime(2026, 8, day, hour, tzinfo=UTC)


def pull_request(number: int, body: str, merged_at: datetime | None = None) -> dict:
    return {
        "number": number,
        "url": f"https://github.com/microsoft/onnxruntime/pull/{number}",
        "title": f"PR {number}",
        "body": body,
        "mergedAt": merged_at.isoformat().replace("+00:00", "Z") if merged_at else None,
        "baseRefName": "main",
        "author": {"login": "author"},
        "mergeCommit": {"oid": f"{number:040x}"},
    }


class ListMergedPullRequestsTest(unittest.TestCase):
    @mock.patch.object(list_merged_prs, "run_gh")
    def test_resolve_repository_uses_script_directory(self, run_gh):
        run_gh.return_value = "microsoft/onnxruntime\n"

        self.assertEqual(list_merged_prs.resolve_repository(), "microsoft/onnxruntime")
        self.assertEqual(
            run_gh.call_args,
            mock.call(
                ["repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"],
                cwd=list_merged_prs.Path(list_merged_prs.__file__).resolve().parent,
            ),
        )

    def test_timestamp_arguments_are_normalized_to_utc(self):
        self.assertEqual(list_merged_prs.parse_timestamp_argument("2026-08-01"), timestamp(1))
        self.assertEqual(list_merged_prs.parse_timestamp_argument("2026-08-01T02:00:00+02:00"), timestamp(1))

    def test_timestamp_arguments_require_timezone_for_explicit_times(self):
        with self.assertRaisesRegex(ValueError, "must include a UTC offset"):
            list_merged_prs.parse_timestamp_argument("2026-08-01T00:00:00")

    def test_marker_round_trip_accepts_plain_lines_with_optional_indentation(self):
        marker = generate_marker(timestamp(1), timestamp(2))

        for rendered in (marker, f"```text\n{marker}\n```", f"  {marker}\r\n"):
            with self.subTest(rendered=rendered):
                match = MARKER_PATTERN.search(rendered)
                self.assertIsNotNone(match)
                self.assertEqual(match.group("since"), "2026-08-01T00:00:00Z")
                self.assertEqual(match.group("through"), "2026-08-02T00:00:00Z")

    def test_marker_pattern_rejects_markdown_decorations(self):
        marker = generate_marker(timestamp(1), timestamp(2))

        for rendered in (f"> {marker}", f"- {marker}", f"`{marker}`"):
            with self.subTest(rendered=rendered):
                self.assertIsNone(MARKER_PATTERN.search(rendered))

    @mock.patch.object(list_merged_prs, "search_pull_requests")
    def test_discover_since_skips_malformed_and_future_markers(self, search):
        search.return_value = [
            pull_request(
                1,
                "Agent-Guidance-Collection: version=1; base=main; "
                "harvested-since=oops; harvested-through=2026-08-02T00:00:00Z",
            ),
            pull_request(2, generate_marker(timestamp(1), timestamp(3))),
            pull_request(3, generate_marker(timestamp(1), timestamp(9))),
            pull_request(4, generate_marker(timestamp(1), timestamp(4))),
        ]

        stderr = StringIO()
        with redirect_stderr(stderr):
            cursor = list_merged_prs.discover_since("microsoft/onnxruntime", timestamp(8))

        self.assertEqual(cursor, (timestamp(4), 4))
        self.assertIn("PR #1: timestamp must use UTC", stderr.getvalue())
        self.assertNotIn("PR #3", stderr.getvalue())

    @mock.patch.object(list_merged_prs, "search_pull_requests")
    def test_discover_since_warns_about_later_marker_when_requested(self, search):
        search.return_value = [
            pull_request(1, generate_marker(timestamp(1), timestamp(3))),
            pull_request(2, generate_marker(timestamp(1), timestamp(9))),
        ]

        stderr = StringIO()
        with redirect_stderr(stderr):
            cursor = list_merged_prs.discover_since(
                "microsoft/onnxruntime",
                timestamp(8),
                warn_about_later_markers=True,
            )

        self.assertEqual(cursor, (timestamp(3), 1))
        self.assertIn(
            "PR #2: harvested-through 2026-08-09T00:00:00Z is later than the collection cutoff 2026-08-08T00:00:00Z",
            stderr.getvalue(),
        )

    @mock.patch.object(list_merged_prs, "search_pull_requests")
    def test_discover_since_rejects_pr_with_multiple_markers(self, search):
        search.return_value = [
            pull_request(
                1,
                f"{generate_marker(timestamp(1), timestamp(2))}\n{generate_marker(timestamp(2), timestamp(3))}",
            )
        ]

        stderr = StringIO()
        with redirect_stderr(stderr), self.assertRaisesRegex(SystemExit, "no valid merged collection marker"):
            list_merged_prs.discover_since("microsoft/onnxruntime", timestamp(4))

        self.assertIn("PR #1: expected exactly one marker line, found 2", stderr.getvalue())

    @mock.patch.object(list_merged_prs, "search_pull_requests")
    def test_discover_since_warns_about_malformed_marker(self, search):
        search.return_value = [
            pull_request(
                1,
                "Agent-Guidance-Collection: version=1; base=main; "
                "harvested-since=oops; harvested-through=2026-08-02T00:00:00Z",
            )
        ]

        stderr = StringIO()
        with redirect_stderr(stderr), self.assertRaisesRegex(SystemExit, "no valid merged collection marker"):
            list_merged_prs.discover_since("microsoft/onnxruntime", timestamp(3))

        self.assertIn("PR #1: timestamp must use UTC", stderr.getvalue())

    @mock.patch.object(list_merged_prs, "search_pull_requests")
    def test_discover_since_warns_with_invalid_interval_values(self, search):
        search.return_value = [pull_request(1, generate_marker(timestamp(2), timestamp(3)).replace("08-02", "08-04"))]

        stderr = StringIO()
        with redirect_stderr(stderr), self.assertRaisesRegex(SystemExit, "no valid merged collection marker"):
            list_merged_prs.discover_since("microsoft/onnxruntime", timestamp(5))

        self.assertIn(
            "harvested-since (2026-08-04T00:00:00Z) must be earlier than harvested-through (2026-08-03T00:00:00Z)",
            stderr.getvalue(),
        )

    @mock.patch.object(list_merged_prs, "search_pull_requests")
    def test_list_candidates_uses_precise_half_open_window(self, search):
        search.return_value = [
            pull_request(1, "", timestamp(1)),
            pull_request(2, "", timestamp(1, 1)),
            pull_request(3, "", timestamp(2)),
            pull_request(4, "", timestamp(2, 1)),
        ]

        candidates = list_merged_prs.list_candidates(
            "microsoft/onnxruntime",
            timestamp(1),
            timestamp(2),
        )

        self.assertEqual([candidate["number"] for candidate in candidates], [2, 3])
        self.assertIn(
            "merged:2026-08-01T00:00:00Z..2026-08-02T00:00:00Z",
            search.call_args.args[1],
        )


if __name__ == "__main__":
    unittest.main()

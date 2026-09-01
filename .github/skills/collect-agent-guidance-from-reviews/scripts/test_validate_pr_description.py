import unittest
from datetime import datetime, timezone

from collection_marker import generate_marker
from validate_pr_description import validate_pr_description

SINCE = datetime(2026, 8, 1, tzinfo=timezone.utc)
THROUGH = datetime(2026, 8, 2, tzinfo=timezone.utc)
MARKER = generate_marker(SINCE, THROUGH)
COLLECTION_OUTPUT = {
    "since": "2026-08-01T00:00:00Z",
    "through": "2026-08-02T00:00:00Z",
    "marker": MARKER,
}


class ValidatePrDescriptionTest(unittest.TestCase):
    def test_accepts_exact_frozen_marker(self):
        validate_pr_description(f"## Collection metadata\n\n```text\n{MARKER}\n```\n", COLLECTION_OUTPUT)

    def test_rejects_multiple_markers(self):
        with self.assertRaisesRegex(ValueError, "expected exactly one marker line, found 2"):
            validate_pr_description(f"{MARKER}\n{MARKER}\n", COLLECTION_OUTPUT)

    def test_rejects_different_collection_window(self):
        other_marker = generate_marker(SINCE, datetime(2026, 8, 3, tzinfo=timezone.utc))

        with self.assertRaisesRegex(ValueError, "does not match the frozen collection window"):
            validate_pr_description(other_marker, COLLECTION_OUTPUT)

    def test_rejects_internally_inconsistent_collection_output(self):
        collection_output = {**COLLECTION_OUTPUT, "through": "2026-08-03T00:00:00Z"}

        with self.assertRaisesRegex(ValueError, "marker does not match its since and through fields"):
            validate_pr_description(MARKER, collection_output)

    def test_rejects_non_string_collection_output_fields(self):
        collection_output = {**COLLECTION_OUTPUT, "through": 3}

        with self.assertRaisesRegex(ValueError, "collection output fields must be strings"):
            validate_pr_description(MARKER, collection_output)


if __name__ == "__main__":
    unittest.main()

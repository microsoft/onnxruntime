# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest

from compile_contributors import is_bot, is_invalid, sort_contributors


class ContributorFilteringTest(unittest.TestCase):
    def test_is_bot_excludes_claude(self):
        for name in ("claude", "@claude", " @claude "):
            with self.subTest(name=name):
                self.assertTrue(is_bot(name))

    def test_is_bot_preserves_similar_human_login(self):
        self.assertFalse(is_bot("claudette"))

    def test_is_invalid_preserves_claude_for_csv(self):
        self.assertFalse(is_invalid("claude"))


class SortContributorsTest(unittest.TestCase):
    def test_sort_by_contributions_orders_by_count_then_contributor(self):
        contributors = {"charlie": 1, "bravo": 3, "alpha": 3}

        self.assertEqual(
            sort_contributors(contributors, "contributions"),
            [("alpha", 3), ("bravo", 3), ("charlie", 1)],
        )

    def test_sort_by_contributor_orders_alphabetically(self):
        contributors = {"charlie": 3, "bravo": 2, "alpha": 1}

        self.assertEqual(
            sort_contributors(contributors, "contributor"),
            [("alpha", 1), ("bravo", 2), ("charlie", 3)],
        )


if __name__ == "__main__":
    unittest.main()

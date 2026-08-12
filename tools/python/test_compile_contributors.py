# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest

from compile_contributors import sort_contributors


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

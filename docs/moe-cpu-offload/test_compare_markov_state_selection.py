import itertools
import unittest

import numpy as np
from compare_markov_state_selection import compare_sequence, iter_selections, update_selection, validate_matrix


class TestMarkovStateSelection(unittest.TestCase):
    def test_replacements_maximize_one_step_coverage_within_budget(self):
        probabilities = np.array([0.05, 0.3, 0.15, 0.1, 0.4])
        initial = frozenset({0, 2, 3})
        for budget in (0, 1, 2):
            with self.subTest(budget=budget):
                selected, changes = update_selection(initial, probabilities, budget)
                feasible = [
                    set(candidate)
                    for candidate in itertools.combinations(range(5), len(initial))
                    if len(set(candidate) - initial) <= budget
                ]
                best = max(sum(probabilities[state] for state in candidate) for candidate in feasible)
                self.assertAlmostEqual(sum(probabilities[state] for state in selected), best)
                self.assertEqual(len(selected), len(initial))
                self.assertEqual(changes, len(selected - initial))
                self.assertLessEqual(changes, budget)

    def test_equal_probabilities_do_not_trigger_replacements(self):
        selected, changes = update_selection(frozenset({1, 2}), np.full(4, 0.25), 1)
        self.assertEqual(selected, frozenset({1, 2}))
        self.assertEqual(changes, 0)

    def test_deterministic_cycle_is_predicted_with_one_replacement(self):
        matrix = np.roll(np.eye(4), 1, axis=1)
        validate_matrix(matrix)
        sequence = np.array([0, 1, 2, 3] * 5 + [0])
        result = compare_sequence(matrix, sequence, 2, 1, 0.5, 4, delay=1, transition_model="known")
        self.assertEqual(result["initial_set"], [0, 3])
        self.assertEqual(result["frozen_hit_rate"], 0.5)
        self.assertEqual(result["greedy_hit_rate"], 1.0)
        self.assertEqual(result["greedy_expected_hit_rate"], 1.0)
        self.assertLessEqual(result["changes_per_step"], 1)

    def test_zero_budget_matches_the_frozen_baseline(self):
        matrix = np.roll(np.eye(4), 1, axis=1)
        result = compare_sequence(matrix, np.array([0, 1, 2, 3, 0]), 2, 0, 0.9, 0)
        self.assertEqual(result["greedy_hits"], result["frozen_hits"])
        self.assertEqual(result["final_greedy_set"], result["initial_set"])
        self.assertEqual(result["changes"], 0)

    def test_selection_does_not_use_the_next_observation(self):
        matrix = np.tile([0.5, 0.3, 0.15, 0.05], (4, 1))
        result = compare_sequence(matrix, np.array([0, 3]), 2, 1, 0.9, 0, delay=1, transition_model="known")
        self.assertEqual(result["final_greedy_set"], [0, 1])
        self.assertEqual(result["greedy_hits"], 0)
        self.assertAlmostEqual(result["greedy_expected_hit_rate"], 0.8)

    def test_ema_training_stops_before_measuring_the_frozen_set(self):
        matrix = np.full((4, 4), 0.25)
        result = compare_sequence(matrix, np.array([3, 3, 3, 3, 0, 1, 0, 1]), 2, 1, 0, 3)
        self.assertEqual(result["initial_set"], [0, 3])
        self.assertEqual(result["frozen_hits"], 2)
        self.assertEqual(result["steps"], 4)

    def test_invalid_transition_matrices_are_rejected(self):
        for matrix in (
            np.ones((2, 3)),
            np.array([[0.5, 0.5], [0.2, 0.2]]),
            np.array([[0.5, 0.5], [-0.1, 1.1]]),
            np.array([[0.5, 0.5], [np.nan, 0.5]]),
            np.eye(2),
        ):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                validate_matrix(matrix)

    def test_learned_two_step_cycle_uses_pending_set_and_delay(self):
        matrix = np.roll(np.eye(4), 1, axis=1)
        sequence = np.array([0, 1, 2, 3] * 4 + [0])
        selections = list(iter_selections(sequence, 4, 2, 1, 0.5, 4, 2, beta=0))
        self.assertEqual(selections[0], (frozenset({0, 3}), frozenset({0, 3}), 0))
        self.assertEqual(selections[1][1:], (frozenset({2, 3}), 1))
        self.assertEqual(selections[2][1:], (frozenset({2, 3}), 0))
        self.assertEqual(selections[3][1:], (frozenset({0, 3}), 1))
        previous = selections[0][0]
        for _, selected, swaps in selections:
            self.assertEqual(len(selected), 2)
            self.assertEqual(swaps, len(selected - previous))
            self.assertLessEqual(swaps, 1)
            previous = selected
        result = compare_sequence(matrix, sequence, 2, 1, 0.5, 4, beta=0)
        self.assertEqual(result["greedy_hits"], result["steps"] - 1)

    def test_estimated_selection_does_not_receive_true_transitions(self):
        sequence = np.array([0, 1, 2, 3] * 4 + [0])
        matrices = (np.roll(np.eye(4), 1, axis=1), np.full((4, 4), 0.25))
        results = [compare_sequence(matrix, sequence, 2, 1, 0.5, 4) for matrix in matrices]
        for key in ("initial_set", "final_greedy_set", "greedy_hits", "changes"):
            self.assertEqual(results[0][key], results[1][key])

    def test_future_observations_cannot_change_committed_selections(self):
        prefix = [0, 1, 2, 3, 0]
        first = list(iter_selections(np.array([*prefix, 1, 2, 3, 0]), 4, 2, 1, 0.5, 0, 2))
        second = list(iter_selections(np.array([*prefix, 3, 3, 3, 3]), 4, 2, 1, 0.5, 0, 2))
        self.assertEqual(first[:6], second[:6])

    def test_unobserved_rows_use_current_ema(self):
        selections = list(iter_selections(np.array([3, 2, 1, 0]), 4, 2, 1, 0, 0, 2))
        self.assertEqual(selections[0][0], frozenset({0, 1}))
        self.assertEqual(selections[2][1:], (frozenset({1, 2}), 1))

    def test_transition_ema_is_updated_during_measurement(self):
        sequence = np.array([0, 1, 2, 3] * 3 + [0])
        selections = list(iter_selections(sequence, 4, 2, 1, 1, 0, 2, beta=0))
        self.assertIn(3, selections[6][1])
        self.assertEqual(selections[6][2], 1)

    def test_beta_controls_transition_recency_per_departure(self):
        sequence = np.array([0, 2, 0, 2, 0, 2, 0, 4, 0, 1])
        for beta, expected in ((0, {1, 4}), (0.5, {1, 4}), (0.9, {1, 2}), (1, {0, 1})):
            with self.subTest(beta=beta):
                selections = list(iter_selections(sequence, 5, 2, 1, 1, 8, 1, beta=beta))
                self.assertEqual(selections[0][1], frozenset(expected))

    def test_new_transition_row_starts_from_previous_state_ema(self):
        sequence = np.array([0, 3, 2, 1, 1, 1, 2, 0])
        selections = list(iter_selections(sequence, 4, 2, 1, 0.5, 6, 1, beta=1))
        self.assertEqual(selections[0][0], frozenset({1, 2}))
        self.assertEqual(selections[0][1], frozenset({2, 3}))


if __name__ == "__main__":
    unittest.main()

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import importlib.util
import os
import sys
import unittest

# Load module directly
file_path = os.path.join(
    os.path.dirname(__file__),
    "../../python/production_debt.py",
)
spec = importlib.util.spec_from_file_location("ort_production_debt", file_path)
production_debt_mod = importlib.util.module_from_spec(spec)
sys.modules["ort_production_debt"] = production_debt_mod
spec.loader.exec_module(production_debt_mod)

ProductionDebtSessionGate = production_debt_mod.ProductionDebtSessionGate
TechnicalDueDiligenceLedger = production_debt_mod.TechnicalDueDiligenceLedger
GENESIS_HASH = production_debt_mod.GENESIS_HASH


class TestProductionDebtSessionGate(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = ProductionDebtSessionGate(
            never_equate_intent_to_approval=True,
            max_acceptable_odi=12.0,
        )

    def test_clean_session_run_passes_readiness(self) -> None:
        report = self.gate.evaluate_session_run(
            session_id="ort_directml_npu_session",
            allocated_arena_bytes=4000000000,
            utilized_tensor_bytes=4150000000,
            run_latency_ms=22.5,
            cpu_fallback_nodes=0,
            un_gated_mutations=0,
        )
        self.assertTrue(report.is_production_ready)
        self.assertLessEqual(report.odi_score, 12.0)
        self.assertEqual(len(report.critical_smells), 0)
        self.assertTrue(bool(report.receipt_hash))

    def test_degraded_session_run_fails_debt(self) -> None:
        report = self.gate.evaluate_session_run(
            session_id="uncalibrated_ort_session",
            allocated_arena_bytes=4000000000,
            utilized_tensor_bytes=11000000000,  # 2.75x arena sprawl
            run_latency_ms=120.0,  # High latency
            cpu_fallback_nodes=2,  # 2 CPU fallback nodes
            un_gated_mutations=1,  # 1 un-gated mutation
        )
        self.assertFalse(report.is_production_ready)
        self.assertGreater(report.odi_score, 50.0)
        self.assertIn("HIGH_ARENA_MEMORY_SPRAWL_2.75X", report.critical_smells)
        self.assertIn("HIGH_SESSION_RUN_LATENCY_120.0MS", report.critical_smells)
        self.assertIn("DETECTED_2_SILENT_CPU_FALLBACK_NODES", report.critical_smells)
        self.assertIn("DETECTED_1_UNGATED_GRAPH_MUTATIONS", report.critical_smells)

    def test_cryptographic_ledger_integrity(self) -> None:
        self.gate.evaluate_session_run("session-1")
        self.gate.evaluate_session_run("session-2")
        self.gate.evaluate_session_run("session-3")

        entries = self.gate.ledger.get_ledger_entries()
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["prev_hash"], GENESIS_HASH)
        self.assertEqual(entries[1]["prev_hash"], entries[0]["curr_hash"])
        self.assertEqual(entries[2]["prev_hash"], entries[1]["curr_hash"])
        self.assertTrue(self.gate.ledger.verify_ledger_integrity())


if __name__ == "__main__":
    unittest.main()

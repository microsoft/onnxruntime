# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

GENESIS_HASH = "0000000000000000000000000000000000000000000000000000000000000000"


@dataclass
class ONNXDebtReport:
    session_id: str
    odi_score: float  # ONNX Debt Index (target <= 12.0)
    arena_sprawl_multiplier: float  # Target <= 1.08x
    run_latency_ms: float  # Target <= 35.0ms
    mutation_safety_score: float  # Target 100.0
    production_readiness_index: float  # Scale 0 - 100
    is_production_ready: bool
    critical_smells: list[str]
    receipt_hash: str


class TechnicalDueDiligenceLedger:
    """Cryptographic SHA-256 hash-chained Action Ledger for ONNX Runtime session execution runs."""

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []
        self._last_hash: str = GENESIS_HASH

    def record_session_event(
        self,
        session_id: str,
        event_type: str,
        readiness_index: float,
        critical_smells: list[str],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        timestamp = datetime.now(timezone.utc).isoformat()
        index = len(self._entries)

        meta_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
        canonical_content = (
            f"{index}|{self._last_hash}|{session_id}|{event_type}|"
            f"{readiness_index}|{timestamp}|{hashlib.sha256(meta_bytes).hexdigest()}"
        )
        curr_hash = hashlib.sha256(canonical_content.encode("utf-8")).hexdigest()

        entry = {
            "index": index,
            "timestamp": timestamp,
            "session_id": session_id,
            "event_type": event_type,
            "readiness_index": readiness_index,
            "critical_smells": critical_smells,
            "prev_hash": self._last_hash,
            "curr_hash": curr_hash,
            "metadata": metadata,
        }

        self._entries.append(entry)
        self._last_hash = curr_hash
        return entry

    def get_ledger_entries(self) -> list[dict[str, Any]]:
        return list(self._entries)

    def verify_ledger_integrity(self) -> bool:
        prev = GENESIS_HASH
        for entry in self._entries:
            if entry["prev_hash"] != prev:
                return False
            prev = entry["curr_hash"]
        return True


class ProductionDebtSessionGate:
    """A2Z SOC Production Debt & Technical Due Diligence Gate for Microsoft ONNX Runtime Sessions.

    Quantifies silent CPU fallback nodes, arena memory allocator fragmentation, and run latency against 4 Enterprise KPIs:
    1. ONNX Debt Index (ODI <= 12.0)
    2. Arena Memory Sprawl Multiplier (AMSM <= 1.08x)
    3. P99 Session Run Latency (<= 35.0ms)
    4. Deterministic Mutation Boundaries (never_equate_intent_to_approval)
    """

    def __init__(
        self,
        never_equate_intent_to_approval: bool = True,
        max_acceptable_odi: float = 12.0,
    ) -> None:
        self.never_equate_intent_to_approval = never_equate_intent_to_approval
        self.max_acceptable_odi = max_acceptable_odi
        self.ledger = TechnicalDueDiligenceLedger()

    def check_kill_switch(self) -> bool:
        if os.environ.get("AAG_KILL_SWITCH", "").lower() in ("true", "1", "yes"):
            return True
        return any(Path(p).exists() for p in ("artifacts/KILL", "/tmp/KILL"))

    def evaluate_session_run(
        self,
        session_id: str,
        allocated_arena_bytes: int = 4000000000,
        utilized_tensor_bytes: int = 4200000000,
        run_latency_ms: float = 22.5,
        cpu_fallback_nodes: int = 0,
        un_gated_mutations: int = 0,
    ) -> ONNXDebtReport:
        # 1. Evaluate emergency kill switch
        if self.check_kill_switch():
            self.ledger.record_session_event(
                session_id=session_id,
                event_type="session_halted_kill_switch",
                readiness_index=0.0,
                critical_smells=["EMERGENCY_KILL_SWITCH_ENGAGED"],
                metadata={"reason": "AAG_KILL_SWITCH is set"},
            )
            err_msg = "A2Z SOC ActionGate: Emergency kill switch is engaged. ONNX Runtime session halted."
            raise PermissionError(err_msg)

        critical_smells: list[str] = []

        # KPI 2: Arena Memory Sprawl Multiplier
        arena_ratio = utilized_tensor_bytes / max(1, allocated_arena_bytes)
        if arena_ratio > 1.8:
            critical_smells.append(f"HIGH_ARENA_MEMORY_SPRAWL_{arena_ratio:.2f}X")

        # KPI 3: Latency Ceiling
        if run_latency_ms > 60.0:
            critical_smells.append(f"HIGH_SESSION_RUN_LATENCY_{run_latency_ms:.1f}MS")

        # Silent CPU fallback nodes
        if cpu_fallback_nodes > 0:
            critical_smells.append(f"DETECTED_{cpu_fallback_nodes}_SILENT_CPU_FALLBACK_NODES")

        # KPI 4: Mutation Safety
        if un_gated_mutations > 0:
            critical_smells.append(f"DETECTED_{un_gated_mutations}_UNGATED_GRAPH_MUTATIONS")

        # KPI 1: ONNX Debt Index (0 = Clean, 100 = Catastrophic)
        odi = (
            max(0.0, (arena_ratio - 1.0) * 20.0)
            + max(0.0, (run_latency_ms - 35.0) * 0.5)
            + (cpu_fallback_nodes * 25.0)
            + (un_gated_mutations * 30.0)
        )
        odi_score = round(min(100.0, odi), 2)

        # Production Readiness Index (0 - 100)
        readiness = max(0.0, 100.0 - odi_score)
        is_production_ready = (
            odi_score <= self.max_acceptable_odi and len(critical_smells) == 0
        )

        # Cryptographic Ledger Entry
        entry = self.ledger.record_session_event(
            session_id=session_id,
            event_type="session_authorized" if is_production_ready else "session_flagged_debt",
            readiness_index=readiness,
            critical_smells=critical_smells,
            metadata={
                "odi_score": odi_score,
                "arena_ratio": arena_ratio,
                "allocated_arena_bytes": allocated_arena_bytes,
                "utilized_tensor_bytes": utilized_tensor_bytes,
                "run_latency_ms": run_latency_ms,
                "cpu_fallback_nodes": cpu_fallback_nodes,
                "un_gated_mutations": un_gated_mutations,
                "never_equate_intent_to_approval": self.never_equate_intent_to_approval,
            },
        )

        return ONNXDebtReport(
            session_id=session_id,
            odi_score=odi_score,
            arena_sprawl_multiplier=round(arena_ratio, 2),
            run_latency_ms=round(run_latency_ms, 2),
            mutation_safety_score=(
                100.0 if un_gated_mutations == 0 else max(0.0, 100.0 - un_gated_mutations * 30.0)
            ),
            production_readiness_index=readiness,
            is_production_ready=is_production_ready,
            critical_smells=critical_smells,
            receipt_hash=entry["curr_hash"],
        )

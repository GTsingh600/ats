"""Ten lightweight non-ATC *scheduling worlds* for ADAPT-only training.

Each task uses distinct ``airport`` / entity-type labels so ADAPT cannot keyword-match;
structure (windows, risks, ops mix) varies by world index.  All map through the same
AMAN/DMAN machinery after ``apply_adapt_mapping``.
"""

from __future__ import annotations

from typing import Dict, List

try:
    from ..models import (
        Difficulty,
        FlightRecord,
        OperationType,
        PriorityClass,
        RunwaySpec,
        TaskDefinition,
        WakeClass,
    )
except ImportError:
    from models import (
        Difficulty,
        FlightRecord,
        OperationType,
        PriorityClass,
        RunwaySpec,
        TaskDefinition,
        WakeClass,
    )


_SYNTH_DOMAIN_DESCRIPTION = """\
Synthetic multi-resource scheduling domain (non-aviation).
Resources behave like parallel servers; entities are job classes with time windows.
ADAPT must infer wake_class and priority from structural signals only.\
"""


def _resources(prefix: str) -> List[RunwaySpec]:
    ids = [f"{prefix}_R{i}" for i in range(4)]
    return [
        RunwaySpec(
            runway_id=r,
            hourly_capacity=6,
            weather_penalty=1.0,
            notes="shared resource slot",
        )
        for r in ids
    ]


def _flight(
    fid: str,
    airline: str,
    op: OperationType,
    wake: WakeClass,
    sched: int,
    lo: int,
    hi: int,
    runways: List[str],
    *,
    risk: float = 0.0,
    pax: int = 1,
    burn: float = 1.2,
    pri: PriorityClass = PriorityClass.NORMAL,
    notes: str = "",
) -> FlightRecord:
    return FlightRecord(
        flight_id=fid,
        airline=airline,
        operation=op,
        wake_class=wake,
        scheduled_minute=sched,
        earliest_minute=lo,
        latest_minute=hi,
        allowed_runways=runways,
        passengers=pax,
        fuel_burn_per_minute=burn,
        priority=pri,
        connection_risk=risk,
        notes=notes,
    )


def _one_world(idx: int) -> TaskDefinition:
    prefix = f"DOM{idx:02d}"
    rwys = [f"{prefix}_R{i}" for i in range(4)]
    # Distinct entity-type namespaces per world (10 "domains")
    ents = [
        ("CRATE", "PALLET", "BULK", "RUSH"),
        ("BATCH", "STREAM", "MICRO", "MEGA"),
        ("ALPHA", "BETA", "GAMMA", "DELTA"),
        ("NORTH", "SOUTH", "EAST", "WEST"),
        ("TIER1", "TIER2", "TIER3", "TIER4"),
        ("NODE_A", "NODE_B", "NODE_C", "NODE_D"),
        ("CLASS_X", "CLASS_Y", "CLASS_Z", "CLASS_W"),
        ("POOL_1", "POOL_2", "POOL_3", "POOL_4"),
        ("RING_I", "RING_II", "RING_III", "RING_IV"),
        ("UNIT_A", "UNIT_B", "UNIT_C", "UNIT_D"),
    ][idx % 10]
    e0, e1, e2, e3 = ents
    # Tighten windows as idx grows (harder structural pressure)
    slack = max(25, 80 - idx * 5)
    return TaskDefinition(
        task_id=f"synth_world_{idx:02d}_shift",
        title=f"Synthetic domain world {idx} — {prefix}",
        difficulty=Difficulty.MEDIUM if idx < 6 else Difficulty.HARD,
        airport=f"SYN_{prefix}",
        description=_SYNTH_DOMAIN_DESCRIPTION,
        objective="Schedule entities on shared resources without window violations.",
        grading_focus=["window_feasible", "resource_conflicts", "delay"],
        planning_horizon_minutes=240,
        max_steps=4,
        delay_budget=70,
        fuel_budget=400.0,
        fairness_tolerance=2.0,
        runways=_resources(prefix),
        flights=[
            _flight(
                f"{prefix}_J1", e0, OperationType.ARRIVAL, WakeClass.MEDIUM,
                20, 0, slack, rwys, risk=0.15, notes="narrow inbound window",
            ),
            _flight(
                f"{prefix}_J2", e1, OperationType.ARRIVAL, WakeClass.MEDIUM,
                45, 15, slack + 40, rwys, risk=0.55, notes="cascade-sensitive sequence",
            ),
            _flight(
                f"{prefix}_J3", e2, OperationType.DEPARTURE, WakeClass.LIGHT,
                60, 20, slack + 60, rwys, risk=0.25,
            ),
            _flight(
                f"{prefix}_J4", e3, OperationType.DEPARTURE, WakeClass.MEDIUM,
                90, 40, slack + 100, rwys, risk=0.40, notes="authorization-bound release",
            ),
        ],
    )


def synthetic_task_catalog() -> Dict[str, TaskDefinition]:
    return {f"synth_world_{i:02d}_shift": _one_world(i) for i in range(10)}

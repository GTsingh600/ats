"""Grounded curriculum tasks (Level 0–5).

Each level adds exactly one semantic dimension of difficulty. Tasks are
deterministic, built from explicit flight/runway structure (no random
perturbation). Separation rules follow constants.SEPARATION_BY_WAKE (FAA-style
wake categories simplified to H/M/L).

Level 0 — toddler: minimal entities, disjoint runway roles, no ambiguity.
Level 1 — more aircraft, still no shared-runway conflict.
Level 2 — first intentional same-runway sequencing conflict.
Level 3 — overlapping mixed operations on one runway (arrival + departure).
Level 4 — timing uncertainty (asymmetric wide windows; semantic only).
Level 5 — deterministic capacity stress (fixed weather penalty), no RNG.

OpenSky-style traffic: optional future CSV import; canonical IDs here mimic
discrete event patterns only (no synthetic noise generation).
"""

from __future__ import annotations

from typing import List

try:
    from .models import (
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


def _f(
    flight_id: str,
    airline: str,
    operation: OperationType,
    wake_class: WakeClass,
    scheduled: int,
    earliest: int,
    latest: int,
    runways: List[str],
    passengers: int,
    burn: float,
    priority: PriorityClass = PriorityClass.NORMAL,
    connection_risk: float = 0.0,
    notes: str = "",
) -> FlightRecord:
    return FlightRecord(
        flight_id=flight_id,
        airline=airline,
        operation=operation,
        wake_class=wake_class,
        scheduled_minute=scheduled,
        earliest_minute=earliest,
        latest_minute=latest,
        allowed_runways=runways,
        passengers=passengers,
        fuel_burn_per_minute=burn,
        priority=priority,
        connection_risk=connection_risk,
        notes=notes,
    )


# --- Level 0: 1 arrival (RWY A only) + 1 departure (RWY B only), wide windows ---
GC_L0_ISOLATED = TaskDefinition(
    task_id="gc_l0_isolated_rwys",
    title="Grounded L0 — Isolated runway roles (toddler)",
    difficulty=Difficulty.EASY,
    airport="TOY0",
    description=(
        "Canonical toy sector: arrivals use RWY09L only, departures use RWY09R only. "
        "No shared runway between arrival and departure lanes — zero cross-lane ambiguity. "
        "Pattern inspired by segregated-mode ops (see FAA AIM parallel runway concepts, simplified)."
    ),
    objective="Assign each flight at its scheduled minute on its only allowed runway.",
    grading_focus=["Schedule completeness", "Zero runway conflicts", "Stay inside windows"],
    planning_horizon_minutes=45,
    max_steps=4,
    delay_budget=120,
    fuel_budget=400.0,
    fairness_tolerance=20.0,
    runways=[
        RunwaySpec(
            runway_id="09L",
            allowed_operations=[OperationType.ARRIVAL],
            hourly_capacity=30,
            weather_penalty=1.0,
            notes="Arrival-only (grounded curriculum L0).",
        ),
        RunwaySpec(
            runway_id="09R",
            allowed_operations=[OperationType.DEPARTURE],
            hourly_capacity=30,
            weather_penalty=1.0,
            notes="Departure-only (grounded curriculum L0).",
        ),
    ],
    flights=[
        _f("GC_A1", "GC", OperationType.ARRIVAL, WakeClass.MEDIUM, 12, 10, 28, ["09L"], 80, 2.0, notes="L0 arrival; single runway choice."),
        _f("GC_D1", "GC", OperationType.DEPARTURE, WakeClass.MEDIUM, 26, 22, 38, ["09R"], 90, 2.0, notes="L0 departure; single runway choice."),
    ],
)

# --- Level 1: +1 arrival on same arrival runway (sequencing, no departure overlap) ---
GC_L1_TWO_ARRIVALS = TaskDefinition(
    task_id="gc_l1_two_arrivals_same_rw",
    title="Grounded L1 — Two arrivals (entity count)",
    difficulty=Difficulty.EASY,
    airport="TOY0",
    description=(
        "Same arrival-only runway as L0, but two arrivals. "
        "Adds exactly one dimension: aircraft count / sequencing on one runway. "
        "Windows chosen so scheduled_minute order respects M→M separation (3 min)."
    ),
    objective="Sequence two arrivals on 09L with wake spacing; departures unchanged on 09R.",
    grading_focus=["Wake spacing on single arrival runway", "Completeness"],
    planning_horizon_minutes=50,
    max_steps=4,
    delay_budget=120,
    fuel_budget=450.0,
    fairness_tolerance=20.0,
    runways=[
        RunwaySpec(
            runway_id="09L",
            allowed_operations=[OperationType.ARRIVAL],
            hourly_capacity=30,
            weather_penalty=1.0,
            notes="Arrival-only.",
        ),
        RunwaySpec(
            runway_id="09R",
            allowed_operations=[OperationType.DEPARTURE],
            hourly_capacity=30,
            weather_penalty=1.0,
            notes="Departure-only.",
        ),
    ],
    flights=[
        _f("GC_A1", "GC", OperationType.ARRIVAL, WakeClass.MEDIUM, 10, 8, 22, ["09L"], 80, 2.0),
        _f("GC_A2", "GC", OperationType.ARRIVAL, WakeClass.MEDIUM, 16, 14, 30, ["09L"], 75, 2.0),
        _f("GC_D1", "GC", OperationType.DEPARTURE, WakeClass.MEDIUM, 28, 24, 40, ["09R"], 90, 2.0),
    ],
)

# --- Level 2: mixed runway — arrival + departure both allowed on 09C; tight timing creates ordering need ---
GC_L2_MIXED_ORDER = TaskDefinition(
    task_id="gc_l2_mixed_runway_order",
    title="Grounded L2 — Mixed runway ordering (first real conflict class)",
    difficulty=Difficulty.EASY,
    airport="TOY1",
    description=(
        "Single mixed-use runway 09C. One arrival and one departure with windows that overlap "
        "in time if naively placed at scheduled minutes — introduces same-runway sequencing. "
        "Exactly one new dimension vs L1: intentional same-runway conflict potential."
    ),
    objective="Choose order on 09C respecting wake/capacity; both flights must land/depart in-window.",
    grading_focus=["Same-runway sequencing", "No conflicts"],
    planning_horizon_minutes=55,
    max_steps=4,
    delay_budget=100,
    fuel_budget=420.0,
    fairness_tolerance=18.0,
    runways=[
        RunwaySpec(
            runway_id="09C",
            allowed_operations=[OperationType.ARRIVAL, OperationType.DEPARTURE],
            hourly_capacity=25,
            weather_penalty=1.0,
            notes="Mixed-use (grounded L2).",
        ),
    ],
    flights=[
        _f("GC_A10", "GC", OperationType.ARRIVAL, WakeClass.MEDIUM, 14, 10, 24, ["09C"], 70, 2.0),
        _f("GC_D10", "GC", OperationType.DEPARTURE, WakeClass.MEDIUM, 15, 12, 28, ["09C"], 85, 2.0),
    ],
)

# --- Level 3: overlapping bank — two deps + one arr on mixed runway, tighter ---
GC_L3_OVERLAP_BANK = TaskDefinition(
    task_id="gc_l3_overlap_bank",
    title="Grounded L3 — Overlapping mixed bank",
    difficulty=Difficulty.MEDIUM,
    airport="TOY1",
    description=(
        "Mixed runway with two departures and one arrival; windows overlap — "
        "adds overlapping multi-flight coordination (one dimension beyond L2)."
    ),
    objective="Conflict-free schedule on 09C for three flights with overlapping windows.",
    grading_focus=["Multi-flight overlap", "Wake and capacity"],
    planning_horizon_minutes=60,
    max_steps=4,
    delay_budget=110,
    fuel_budget=500.0,
    fairness_tolerance=16.0,
    runways=[
        RunwaySpec(
            runway_id="09C",
            allowed_operations=[OperationType.ARRIVAL, OperationType.DEPARTURE],
            hourly_capacity=22,
            weather_penalty=1.0,
            notes="Mixed-use.",
        ),
    ],
    flights=[
        _f("GC_A20", "GC", OperationType.ARRIVAL, WakeClass.MEDIUM, 12, 8, 22, ["09C"], 72, 2.0),
        _f("GC_D20", "GC", OperationType.DEPARTURE, WakeClass.MEDIUM, 14, 10, 26, ["09C"], 88, 2.0),
        _f("GC_D21", "GC", OperationType.DEPARTURE, WakeClass.MEDIUM, 18, 14, 32, ["09C"], 86, 2.0),
    ],
)

# --- Level 4: timing uncertainty — same structure as L3 but asymmetric wide windows (semantic) ---
GC_L4_TIMING_UNCERTAINTY = TaskDefinition(
    task_id="gc_l4_timing_uncertainty",
    title="Grounded L4 — Timing uncertainty (wide asymmetric windows)",
    difficulty=Difficulty.MEDIUM,
    airport="TOY1",
    description=(
        "Same flight IDs and runway topology as L3-style bank, but earliest/latest widened "
        "asymmetrically to model uncertain ETA/CTOT (no RNG). New dimension: timing uncertainty only."
    ),
    objective="Robust sequencing under wide windows; still conflict-free.",
    grading_focus=["Robust placement under uncertainty", "No conflicts"],
    planning_horizon_minutes=90,
    max_steps=4,
    delay_budget=140,
    fuel_budget=550.0,
    fairness_tolerance=18.0,
    runways=[
        RunwaySpec(
            runway_id="09C",
            allowed_operations=[OperationType.ARRIVAL, OperationType.DEPARTURE],
            hourly_capacity=22,
            weather_penalty=1.0,
            notes="Mixed-use.",
        ),
    ],
    flights=[
        _f("GC_A20", "GC", OperationType.ARRIVAL, WakeClass.MEDIUM, 20, 5, 45, ["09C"], 72, 2.0, notes="Wide window: uncertain inbound time."),
        _f("GC_D20", "GC", OperationType.DEPARTURE, WakeClass.MEDIUM, 22, 8, 50, ["09C"], 88, 2.0, notes="Wide window: ATFM-like slack (fixed)."),
        _f("GC_D21", "GC", OperationType.DEPARTURE, WakeClass.MEDIUM, 28, 12, 55, ["09C"], 86, 2.0, notes="Wide window."),
    ],
)

# --- Level 5: deterministic disturbance — fixed weather penalty (capacity stress) ---
GC_L5_CAPACITY_STRESS = TaskDefinition(
    task_id="gc_l5_capacity_stress",
    title="Grounded L5 — Deterministic capacity stress",
    difficulty=Difficulty.MEDIUM,
    airport="TOY1",
    description=(
        "Same flight set as L3 topology with fixed weather_penalty=1.45 on 09C — "
        "deterministic disturbance reducing effective capacity (no stochastic weather)."
    ),
    objective="Schedule under reduced effective capacity; remain conflict-free.",
    grading_focus=["Capacity stress", "Conflict-free"],
    planning_horizon_minutes=60,
    max_steps=4,
    delay_budget=120,
    fuel_budget=520.0,
    fairness_tolerance=16.0,
    runways=[
        RunwaySpec(
            runway_id="09C",
            allowed_operations=[OperationType.ARRIVAL, OperationType.DEPARTURE],
            hourly_capacity=22,
            weather_penalty=1.45,
            notes="Fixed degradation (grounded L5 deterministic disturbance).",
        ),
    ],
    flights=[
        _f("GC_A20", "GC", OperationType.ARRIVAL, WakeClass.MEDIUM, 12, 8, 22, ["09C"], 72, 2.0),
        _f("GC_D20", "GC", OperationType.DEPARTURE, WakeClass.MEDIUM, 14, 10, 26, ["09C"], 88, 2.0),
        _f("GC_D21", "GC", OperationType.DEPARTURE, WakeClass.MEDIUM, 18, 14, 32, ["09C"], 86, 2.0),
    ],
)


GROUNDED_CURRICULUM_TASKS: List[TaskDefinition] = [
    GC_L0_ISOLATED,
    GC_L1_TWO_ARRIVALS,
    GC_L2_MIXED_ORDER,
    GC_L3_OVERLAP_BANK,
    GC_L4_TIMING_UNCERTAINTY,
    GC_L5_CAPACITY_STRESS,
]

GROUNDED_LEVEL_BY_TASK_ID = {t.task_id: idx for idx, t in enumerate(GROUNDED_CURRICULUM_TASKS)}

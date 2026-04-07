"""Scenario catalog for the ATC optimization benchmark."""

from __future__ import annotations

from typing import Dict, Iterable, List

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


TASKS: List[TaskDefinition] = [
    # ─────────────────────────────────────────────────────────────────────────
    # EASY — Delhi Monsoon Departure Recovery
    # Changes from v1: added 2 LIGHT aircraft (SPJ01, CRJ90) to exercise
    # untested H/M↔L wake separations.  Tightened 3 padded normal-flight
    # windows so every flight presents a real scheduling constraint.
    # ─────────────────────────────────────────────────────────────────────────
    TaskDefinition(
        task_id="delhi_monsoon_recovery_easy",
        title="Delhi Monsoon Departure Recovery",
        difficulty=Difficulty.EASY,
        airport="VIDP",
        description=(
            "A short convective weather burst reduced departure acceptance rates for "
            "the morning bank. The controller must resequence mixed arrivals and "
            "departures — including a charter light piston and a regional feeder — "
            "while protecting a medical flight."
        ),
        objective=(
            "Restore a safe runway sequence, keep the medical departure close to "
            "schedule, correctly space light aircraft around heavier traffic, and "
            "minimize total passenger delay."
        ),
        grading_focus=[
            "Zero runway conflicts including light-aircraft wake spacing",
            "Medical flight protected",
            "Low total delay and fuel burn",
        ],
        planning_horizon_minutes=100,
        max_steps=4,
        delay_budget=105,
        fuel_budget=730.0,
        fairness_tolerance=14.0,
        runways=[
            RunwaySpec(
                runway_id="27L",
                allowed_operations=[OperationType.ARRIVAL, OperationType.DEPARTURE],
                hourly_capacity=18,
                weather_penalty=1.15,
                notes="Usable for mixed operations with reduced wet-runway spacing.",
            ),
            RunwaySpec(
                runway_id="27R",
                allowed_operations=[OperationType.ARRIVAL, OperationType.DEPARTURE],
                hourly_capacity=18,
                weather_penalty=1.15,
                notes="Parallel runway, but departures should be metered carefully.",
            ),
        ],
        flights=[
            # ── Original flights (3 windows tightened) ───────────────────────
            _f("IGO601", "IGO", OperationType.DEPARTURE, WakeClass.MEDIUM,  8,  8, 55, ["27L", "27R"], 180, 2.6),
            _f("AIC221", "AIC", OperationType.DEPARTURE, WakeClass.MEDIUM, 12, 12, 60, ["27L", "27R"], 168, 2.9, PriorityClass.CONNECTION, 0.6, "Tight onward domestic bank."),
            _f("VTI404", "VTI", OperationType.ARRIVAL,   WakeClass.MEDIUM, 15, 15, 48, ["27L"],         156, 4.8, PriorityClass.CONNECTION, 0.7, "Inbound hub connections at risk."),
            _f("AKJ118", "AKJ", OperationType.DEPARTURE, WakeClass.MEDIUM, 18, 18, 38, ["27R"],         150, 2.4),    # was [18,70] — tightened
            _f("IGO882", "IGO", OperationType.ARRIVAL,   WakeClass.MEDIUM, 20, 20, 56, ["27L", "27R"], 174, 5.2, PriorityClass.CONNECTION, 0.5, "Arrival bank should not absorb long holding."),
            _f("AIC911", "AIC", OperationType.DEPARTURE, WakeClass.MEDIUM, 24, 24, 68, ["27L", "27R"], 132, 2.8, PriorityClass.MEDICAL,     0.2, "Medical cargo onboard."),
            _f("VTI721", "VTI", OperationType.DEPARTURE, WakeClass.MEDIUM, 27, 27, 52, ["27L", "27R"], 188, 2.7),    # was [27,84] — tightened
            _f("GOW330", "GOW", OperationType.DEPARTURE, WakeClass.MEDIUM, 30, 30, 55, ["27R"],         186, 2.5),   # was [30,88] — tightened
            # ── NEW: LIGHT aircraft ──────────────────────────────────────────
            # SPJ01 departs 27L only.  Any MEDIUM departure immediately behind a
            # LIGHT needs a 4-minute gap (M←L); any HEAVY needs 6 minutes.
            _f("SPJ01",  "SPJ", OperationType.DEPARTURE, WakeClass.LIGHT,  10, 10, 30, ["27L"],          22, 1.1, PriorityClass.NORMAL, 0.0, "Charter light piston — must be slotted before heavier departures back up 27L."),
            # CRJ90 arrives 27R only.  A HEAVY arriving before it requires 6 min
            # separation; arriving before a HEAVY only 3 min — asymmetry matters.
            _f("CRJ90",  "CRJ", OperationType.ARRIVAL,   WakeClass.LIGHT,  22, 22, 42, ["27R"],          48, 1.8, PriorityClass.NORMAL, 0.0, "Regional feeder — light aircraft wake rules apply on 27R."),
        ],
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # MEDIUM — Mumbai Hub Bank Balance
    # Changes from v1: added DKT22 (LIGHT cargo feeder, rwy 14 only) and
    # JET05 (LIGHT corporate arrival, rwy 27 only).  Tightened IGO774 and
    # AKJ901 windows so the back-of-bank flights are real constraints.
    # ─────────────────────────────────────────────────────────────────────────
    TaskDefinition(
        task_id="mumbai_bank_balance_medium",
        title="Mumbai Hub Bank Balance",
        difficulty=Difficulty.MEDIUM,
        airport="VABB",
        description=(
            "A runway inspection removed one preferred departure flow during the "
            "mid-morning hub bank.  A light cargo feeder and a corporate jet join "
            "the queue alongside international heavies.  Controllers need to protect "
            "fuel-sensitive arrivals and keep delay distribution fair across airlines."
        ),
        objective=(
            "Produce a conflict-free schedule that limits missed bank connections, "
            "correctly spaces light aircraft around heavier traffic, keeps a "
            "fuel-sensitive arrival near the front, and avoids over-penalizing "
            "a single carrier."
        ),
        grading_focus=[
            "Arrival fuel protection",
            "Light-aircraft wake separation correctness",
            "Fairness across airlines",
            "Capacity-aware sequencing",
        ],
        planning_horizon_minutes=120,
        max_steps=4,
        delay_budget=175,
        fuel_budget=1280.0,
        fairness_tolerance=12.0,
        runways=[
            RunwaySpec(
                runway_id="27",
                allowed_operations=[OperationType.ARRIVAL, OperationType.DEPARTURE],
                hourly_capacity=20,
                weather_penalty=1.2,
                notes="Primary runway with wet braking action advisory.",
            ),
            RunwaySpec(
                runway_id="14",
                allowed_operations=[OperationType.DEPARTURE],
                hourly_capacity=14,
                weather_penalty=1.1,
                notes="Departure-only flow after inspection release.",
            ),
        ],
        flights=[
            # ── NEW: LIGHT cargo feeder on dep-only rwy 14 ───────────────────
            # A MEDIUM departure behind DKT22 on 14 needs 4 min (M←L).
            _f("DKT22",  "DKT", OperationType.DEPARTURE, WakeClass.LIGHT,   8,  8, 28, ["14"],           18, 1.0, PriorityClass.NORMAL,     0.0, "Cargo feeder — light aircraft on dep-only rwy 14; sequence before MEDIUM wave."),
            # ── Original flights (2 windows tightened) ───────────────────────
            _f("IGO202", "IGO", OperationType.DEPARTURE, WakeClass.MEDIUM,  5,  5, 70, ["14", "27"],     186, 2.4),
            _f("AIC540", "AIC", OperationType.ARRIVAL,   WakeClass.HEAVY,  10, 10, 44, ["27"],           264, 6.1, PriorityClass.CONNECTION, 0.8, "Long-haul inbound with large connection bank."),
            _f("VTI313", "VTI", OperationType.DEPARTURE, WakeClass.MEDIUM, 13, 13, 78, ["14"],           176, 2.5, PriorityClass.CONNECTION, 0.7),
            _f("AKJ440", "AKJ", OperationType.DEPARTURE, WakeClass.MEDIUM, 17, 17, 82, ["14", "27"],    172, 2.4),
            _f("IGO558", "IGO", OperationType.ARRIVAL,   WakeClass.MEDIUM, 19, 19, 51, ["27"],           181, 5.0, PriorityClass.NORMAL,     0.4),
            _f("AIX117", "AIX", OperationType.DEPARTURE, WakeClass.MEDIUM, 22, 22, 88, ["14"],           164, 2.6),
            _f("AIC612", "AIC", OperationType.DEPARTURE, WakeClass.MEDIUM, 25, 25, 92, ["14", "27"],    158, 2.9, PriorityClass.CONNECTION, 0.6),
            _f("VTI880", "VTI", OperationType.ARRIVAL,   WakeClass.MEDIUM, 28, 28, 55, ["27"],           170, 5.6, PriorityClass.MEDICAL,    0.4, "Special handling passenger onboard."),
            _f("IGO774", "IGO", OperationType.DEPARTURE, WakeClass.MEDIUM, 31, 31, 65, ["14", "27"],    190, 2.5),  # was [31,100] — tightened
            _f("GOW410", "GOW", OperationType.DEPARTURE, WakeClass.MEDIUM, 35, 35, 102, ["14"],          186, 2.3),
            _f("AIC845", "AIC", OperationType.ARRIVAL,   WakeClass.HEAVY,  38, 38, 66, ["27"],           248, 6.3, PriorityClass.CONNECTION, 0.7, "Fuel-sensitive after holding south of field."),
            _f("AKJ901", "AKJ", OperationType.DEPARTURE, WakeClass.MEDIUM, 42, 42, 72, ["14", "27"],    168, 2.4),  # was [42,108] — tightened
            # ── NEW: LIGHT corporate arrival on rwy 27 ────────────────────────
            # If arriving AFTER AIC845 (HEAVY), spacing must be 6 min (H→L).
            # If arriving BEFORE AIC845, only 3 min needed (L→H) — placing JET05
            # early is significantly more efficient, rewarding smart sequencing.
            _f("JET05",  "JET", OperationType.ARRIVAL,   WakeClass.LIGHT,  33, 33, 55, ["27"],            12, 1.2, PriorityClass.NORMAL,     0.0, "Corporate jet arrival — LIGHT, rwy 27 only; heavy-wake separation critical."),
        ],
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # HARD — Bengaluru IRROPS Recovery Command
    # Changes from v1: MED208 window fixed [31,85]→[31,55] (organ transport
    # must be operationally credible).  GOW099 (padding flight) removed.
    # IGO702 window tightened [30,108]→[30,65].  VFR01 (LIGHT arrival)
    # added to force early-slot ordering before HEAVY waves.
    # ─────────────────────────────────────────────────────────────────────────
    TaskDefinition(
        task_id="bengaluru_irrops_hard",
        title="Bengaluru IRROPS Recovery Command",
        difficulty=Difficulty.HARD,
        airport="VOBL",
        description=(
            "A rapidly evolving IRROPS event combines wet-runway spacing, "
            "an emergency inbound, a time-critical organ-transport departure, and a "
            "heavily peaking departure bank.  A light VFR aircraft on a special permit "
            "must be resolved before the heavy wave arrives.  The controller must "
            "sequence mixed traffic with both safety and network impact in mind."
        ),
        objective=(
            "Protect the emergency arrival and medical departure within their strict "
            "windows, slot the light VFR arrival before it is blocked by heavy traffic, "
            "keep all assignments conflict-free under reduced capacity, minimize system "
            "delay, and distribute unavoidable delay fairly across participating airlines."
        ),
        grading_focus=[
            "Emergency and medical traffic prioritized within strict windows",
            "VFR01 slotted before heavy-wave blockage",
            "Zero separation and capacity conflicts",
            "Balanced delay distribution under IRROPS",
        ],
        planning_horizon_minutes=140,
        max_steps=6,
        delay_budget=220,
        fuel_budget=1680.0,
        fairness_tolerance=8.5,
        runways=[
            RunwaySpec(
                runway_id="09L",
                allowed_operations=[OperationType.ARRIVAL, OperationType.DEPARTURE],
                hourly_capacity=18,
                weather_penalty=1.3,
                notes="Mixed-mode runway with longer heavy-to-medium spacing.",
            ),
            RunwaySpec(
                runway_id="09R",
                allowed_operations=[OperationType.ARRIVAL, OperationType.DEPARTURE],
                hourly_capacity=16,
                weather_penalty=1.25,
                notes="Secondary runway handling overflow and tactical recovery.",
            ),
        ],
        flights=[
            _f("IGO110", "IGO", OperationType.DEPARTURE, WakeClass.MEDIUM,  4,  4, 78,  ["09L", "09R"], 184, 2.4),
            _f("AIC781", "AIC", OperationType.ARRIVAL,   WakeClass.HEAVY,   7,  7, 34,  ["09L"],         252, 6.5, PriorityClass.CONNECTION, 0.8),
            # ── NEW: LIGHT VFR on special permit ─────────────────────────────
            # VFR01 must land before the MEDIUM/HEAVY wave builds after t=14.
            # Placed after AIC781 (HEAVY): requires 6-min gap (H→L).
            # Placed before AIC781:        only 3-min gap (L→H) and no blockage.
            # Smart agents will slot it early; naive agents will conflict it.
            _f("VFR01",  "VFR", OperationType.ARRIVAL,   WakeClass.LIGHT,   9,  9, 20,  ["09L", "09R"],  6, 1.0, PriorityClass.NORMAL, 0.0, "Light VFR on special permit — must clear before heavy wave; strict window."),
            _f("VTI220", "VTI", OperationType.DEPARTURE, WakeClass.MEDIUM,  9,  9, 86,  ["09R"],         170, 2.6),
            _f("AKJ662", "AKJ", OperationType.DEPARTURE, WakeClass.MEDIUM, 12, 12, 90,  ["09L", "09R"], 168, 2.3),
            _f("GOW510", "GOW", OperationType.ARRIVAL,   WakeClass.MEDIUM, 14, 14, 46,  ["09L", "09R"], 174, 5.3, PriorityClass.NORMAL, 0.4),
            _f("AIC110", "AIC", OperationType.DEPARTURE, WakeClass.MEDIUM, 16, 16, 92,  ["09L", "09R"], 162, 2.7, PriorityClass.CONNECTION, 0.7),
            _f("IGO455", "IGO", OperationType.ARRIVAL,   WakeClass.MEDIUM, 18, 18, 44,  ["09L"],         176, 5.5, PriorityClass.CONNECTION, 0.6),
            _f("VTI918", "VTI", OperationType.DEPARTURE, WakeClass.MEDIUM, 21, 21, 94,  ["09R"],         188, 2.5),
            _f("MED001", "GOV", OperationType.ARRIVAL,   WakeClass.MEDIUM, 23, 23, 28,  ["09L", "09R"],   6, 7.0, PriorityClass.EMERGENCY, 0.1, "Emergency medical diversion with priority handling."),
            _f("AKJ303", "AKJ", OperationType.DEPARTURE, WakeClass.MEDIUM, 25, 25, 98,  ["09L", "09R"], 178, 2.4),
            _f("AIX770", "AIX", OperationType.DEPARTURE, WakeClass.MEDIUM, 27, 27, 102, ["09R"],         164, 2.6),
            _f("IGO702", "IGO", OperationType.DEPARTURE, WakeClass.MEDIUM, 30, 30, 65,  ["09L", "09R"], 192, 2.4),  # was [30,108] — tightened
            _f("MED208", "AIC", OperationType.DEPARTURE, WakeClass.MEDIUM, 31, 31, 55,  ["09R"],         154, 2.8, PriorityClass.MEDICAL, 0.4, "Time-critical organ transport departure — 24-minute window is firm."),  # was [31,85]
            _f("AIC605", "AIC", OperationType.ARRIVAL,   WakeClass.HEAVY,  33, 33, 60,  ["09L"],         244, 6.4, PriorityClass.CONNECTION, 0.7),
            _f("VTI640", "VTI", OperationType.DEPARTURE, WakeClass.MEDIUM, 36, 36, 112, ["09R"],         182, 2.5),
            _f("AKJ818", "AKJ", OperationType.ARRIVAL,   WakeClass.MEDIUM, 42, 42, 68,  ["09L", "09R"], 171, 5.4, PriorityClass.CONNECTION, 0.5),
            _f("AIC990", "AIC", OperationType.ARRIVAL,   WakeClass.HEAVY,  45, 45, 72,  ["09L"],         246, 6.6, PriorityClass.CONNECTION, 0.8, "Second fuel-sensitive long-haul inbound wave."),
        ],
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # NEW — Hyderabad Cargo Crunch (Medium-Hard)
    # Single runway (09R bird-strike inspection closes 09L).
    # All three wake classes present: HEAVY freighters, MEDIUM passenger jets,
    # LIGHT cargo feeders.  No emergency flight — difficulty comes entirely
    # from runway scarcity and wake turbulence asymmetry.
    # ─────────────────────────────────────────────────────────────────────────
    TaskDefinition(
        task_id="hyderabad_cargo_crunch_medium_hard",
        title="Hyderabad Single-Runway Cargo Crunch",
        difficulty=Difficulty.HARD,
        airport="VOHS",
        description=(
            "A bird-strike inspection grounds runway 09R at Rajiv Gandhi International "
            "during the peak mango-export cargo wave.  All traffic — international "
            "heavy freighters, domestic passenger jets, and light cargo feeders — must "
            "be sequenced through runway 09L alone.  Wake turbulence between all three "
            "aircraft classes is the central constraint."
        ),
        objective=(
            "Sequence all seven aircraft through the single available runway without "
            "separation conflicts, exploiting wake-class asymmetry to maximize "
            "throughput while protecting connection passengers on AIC300."
        ),
        grading_focus=[
            "Zero conflicts on a single runway",
            "Wake-class asymmetry exploited (L→H cheaper than H→L)",
            "Connection passengers on AIC300 protected",
            "Minimal total delay given severe capacity constraint",
        ],
        planning_horizon_minutes=80,
        max_steps=4,
        delay_budget=90,
        fuel_budget=720.0,
        fairness_tolerance=15.0,
        runways=[
            RunwaySpec(
                runway_id="09L",
                allowed_operations=[OperationType.ARRIVAL, OperationType.DEPARTURE],
                hourly_capacity=16,
                weather_penalty=1.1,
                notes="Only active runway — 09R closed for bird-strike inspection.",
            ),
        ],
        flights=[
            # ── Slot ordering puzzle ─────────────────────────────────────────
            # FTR001 (HEAVY arr) → SPJ100 (LIGHT dep): 6-min gap required (H→L)
            # SPJ100 (LIGHT dep) → AIC300 (MEDIUM arr): 3-min gap (L→M) — asymmetry!
            # FTR002 (HEAVY dep) → CRG55 (LIGHT arr): 6-min gap (H→L)
            # CRG55  (LIGHT arr) → VTI200 (MEDIUM dep): only 3-min (L→M) — efficient
            # Optimal ordering saves ~6 minutes of total delay vs naive FCFS.
            _f("FTR001", "FTR", OperationType.ARRIVAL,   WakeClass.HEAVY,  10, 10, 28, ["09L"],  312, 6.8, PriorityClass.NORMAL,      0.0, "Heavy freighter — sets 6-min gap for any following LIGHT traffic."),
            _f("SPJ100", "SPJ", OperationType.DEPARTURE, WakeClass.LIGHT,  14, 14, 32, ["09L"],   20, 1.1, PriorityClass.NORMAL,      0.0, "Light cargo feeder — if after HEAVY arr, needs 6 min; if before, only 3."),
            _f("AIC300", "AIC", OperationType.ARRIVAL,   WakeClass.MEDIUM, 17, 17, 35, ["09L"],  198, 5.4, PriorityClass.CONNECTION,  0.8, "Connection bank closes at t=40; keep delays minimal."),
            _f("FTR002", "FTR", OperationType.DEPARTURE, WakeClass.HEAVY,  21, 21, 42, ["09L"],  285, 5.9, PriorityClass.NORMAL,      0.0, "Outbound freighter — after departure, following LIGHT needs 6-min gap."),
            _f("VTI200", "VTI", OperationType.DEPARTURE, WakeClass.MEDIUM, 23, 23, 44, ["09L"],  182, 2.6, PriorityClass.CONNECTION,  0.5, "Domestic passenger departure — tight window in single-runway crunch."),
            _f("CRG55",  "CRG", OperationType.ARRIVAL,   WakeClass.LIGHT,  27, 27, 46, ["09L"],   16, 1.0, PriorityClass.NORMAL,      0.0, "Light cargo arrival — exploit L→M asymmetry to save slots for VTI200."),
            _f("AIC480", "AIC", OperationType.DEPARTURE, WakeClass.MEDIUM, 30, 30, 52, ["09L"],  175, 2.5, PriorityClass.NORMAL,      0.3, "Second passenger departure — last slot in the crunch window."),
        ],
    ),
]


def task_catalog() -> Dict[str, TaskDefinition]:
    """Return all tasks keyed by task_id."""
    return {t.task_id: t for t in TASKS}


def ordered_tasks() -> Iterable[TaskDefinition]:
    """Return tasks ordered easy → hard for sequential play."""
    order = {Difficulty.EASY: 0, Difficulty.MEDIUM: 1, Difficulty.HARD: 2}
    return sorted(TASKS, key=lambda t: (order.get(t.difficulty, 99), t.task_id))


def render_task_briefing(task: TaskDefinition) -> str:
    """Generate the ATC situation briefing shown to agents.

    Produces a concise, structured text that describes the scenario, runway
    configuration, flight list (with priorities and wake classes), and the
    scoring focus areas.  This is the primary context LLMs receive.
    """
    lines: List[str] = []

    lines.append(f"# ATC BRIEFING — {task.title}")
    lines.append(f"Airport: {task.airport}   Difficulty: {task.difficulty.value.upper()}")
    lines.append(f"Planning horizon: {task.planning_horizon_minutes} min   "
                 f"Delay budget: {task.delay_budget} min   "
                 f"Fuel budget: {task.fuel_budget:.0f} kg")
    lines.append("")
    lines.append(f"## Situation\n{task.description}")
    lines.append("")
    lines.append(f"## Objective\n{task.objective}")
    lines.append("")

    lines.append("## Runways")
    for rwy in task.runways:
        ops = "/".join(op.value for op in rwy.allowed_operations)
        lines.append(
            f"  {rwy.runway_id}: {ops}, capacity {rwy.hourly_capacity}/hr, "
            f"wx_penalty×{rwy.weather_penalty}. {rwy.notes}"
        )
    lines.append("")

    lines.append("## Wake Turbulence Separation (minutes between consecutive movements)")
    lines.append("  Leader→Follower : H→H=4  H→M=3  H→L=6  M→H=3  M→M=3  M→L=4  L→H=3  L→M=3  L→L=3")
    lines.append("")

    lines.append("## Flights (flight_id | airline | op | wake | sched | window | runways | priority | risk | notes)")
    for f in task.flights:
        r_str = ",".join(f.allowed_runways)
        risk = f"-risk={f.connection_risk:.1f}" if f.connection_risk > 0 else ""
        notes = f"  [{f.notes}]" if f.notes else ""
        lines.append(
            f"  {f.flight_id:<8} {f.airline:<4} {f.operation.value:<3} "
            f"{f.wake_class.value:<3} sched={f.scheduled_minute:>3} "
            f"[{f.earliest_minute},{f.latest_minute}] "
            f"rwy={r_str}  {f.priority.value}{risk}{notes}"
        )
    lines.append("")

    lines.append("## Grading Focus")
    for item in task.grading_focus:
        lines.append(f"  • {item}")

    return "\n".join(lines)


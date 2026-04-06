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
    TaskDefinition(
        task_id="delhi_monsoon_recovery_easy",
        title="Delhi Monsoon Departure Recovery",
        difficulty=Difficulty.EASY,
        airport="VIDP",
        description=(
            "A short convective weather burst reduced departure acceptance rates for "
            "the morning bank. The controller must resequence mixed arrivals and "
            "departures while protecting a medical flight."
        ),
        objective=(
            "Restore a safe runway sequence, keep the medical departure close to "
            "schedule, and minimize total passenger delay."
        ),
        grading_focus=[
            "Zero runway conflicts",
            "Medical flight protected",
            "Low total delay and fuel burn",
        ],
        planning_horizon_minutes=100,
        max_steps=4,
        delay_budget=95,
        fuel_budget=680.0,
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
            _f("IGO601", "IGO", OperationType.DEPARTURE, WakeClass.MEDIUM, 8, 8, 55, ["27L", "27R"], 180, 2.6),
            _f("AIC221", "AIC", OperationType.DEPARTURE, WakeClass.MEDIUM, 12, 12, 60, ["27L", "27R"], 168, 2.9, PriorityClass.CONNECTION, 0.6, "Tight onward domestic bank."),
            _f("VTI404", "VTI", OperationType.ARRIVAL, WakeClass.MEDIUM, 15, 15, 48, ["27L"], 156, 4.8, PriorityClass.CONNECTION, 0.7, "Inbound hub connections at risk."),
            _f("AKJ118", "AKJ", OperationType.DEPARTURE, WakeClass.MEDIUM, 18, 18, 70, ["27R"], 150, 2.4),
            _f("IGO882", "IGO", OperationType.ARRIVAL, WakeClass.MEDIUM, 20, 20, 56, ["27L", "27R"], 174, 5.2, PriorityClass.CONNECTION, 0.5, "Arrival bank should not absorb long holding."),
            _f("AIC911", "AIC", OperationType.DEPARTURE, WakeClass.MEDIUM, 24, 24, 68, ["27L", "27R"], 132, 2.8, PriorityClass.MEDICAL, 0.2, "Medical cargo onboard."),
            _f("VTI721", "VTI", OperationType.DEPARTURE, WakeClass.MEDIUM, 27, 27, 84, ["27L", "27R"], 188, 2.7),
            _f("GOW330", "GOW", OperationType.DEPARTURE, WakeClass.MEDIUM, 30, 30, 88, ["27R"], 186, 2.5),
        ],
    ),
    TaskDefinition(
        task_id="mumbai_bank_balance_medium",
        title="Mumbai Hub Bank Balance",
        difficulty=Difficulty.MEDIUM,
        airport="VABB",
        description=(
            "A runway inspection removed one preferred departure flow during the "
            "mid-morning hub bank. Controllers need to protect fuel-sensitive arrivals "
            "and keep delay distribution fair across airlines."
        ),
        objective=(
            "Produce a conflict-free schedule that limits missed bank connections, "
            "keeps a fuel-sensitive arrival near the front, and avoids over-penalizing "
            "a single carrier."
        ),
        grading_focus=[
            "Arrival fuel protection",
            "Fairness across airlines",
            "Capacity-aware sequencing",
        ],
        planning_horizon_minutes=120,
        max_steps=4,
        delay_budget=165,
        fuel_budget=1225.0,
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
            _f("IGO202", "IGO", OperationType.DEPARTURE, WakeClass.MEDIUM, 5, 5, 70, ["14", "27"], 186, 2.4),
            _f("AIC540", "AIC", OperationType.ARRIVAL, WakeClass.HEAVY, 10, 10, 44, ["27"], 264, 6.1, PriorityClass.CONNECTION, 0.8, "Long-haul inbound with large connection bank."),
            _f("VTI313", "VTI", OperationType.DEPARTURE, WakeClass.MEDIUM, 13, 13, 78, ["14"], 176, 2.5, PriorityClass.CONNECTION, 0.7),
            _f("AKJ440", "AKJ", OperationType.DEPARTURE, WakeClass.MEDIUM, 17, 17, 82, ["14", "27"], 172, 2.4),
            _f("IGO558", "IGO", OperationType.ARRIVAL, WakeClass.MEDIUM, 19, 19, 51, ["27"], 181, 5.0, PriorityClass.NORMAL, 0.4),
            _f("AIX117", "AIX", OperationType.DEPARTURE, WakeClass.MEDIUM, 22, 22, 88, ["14"], 164, 2.6),
            _f("AIC612", "AIC", OperationType.DEPARTURE, WakeClass.MEDIUM, 25, 25, 92, ["14", "27"], 158, 2.9, PriorityClass.CONNECTION, 0.6),
            _f("VTI880", "VTI", OperationType.ARRIVAL, WakeClass.MEDIUM, 28, 28, 55, ["27"], 170, 5.6, PriorityClass.MEDICAL, 0.4, "Special handling passenger onboard."),
            _f("IGO774", "IGO", OperationType.DEPARTURE, WakeClass.MEDIUM, 31, 31, 100, ["14", "27"], 190, 2.5),
            _f("GOW410", "GOW", OperationType.DEPARTURE, WakeClass.MEDIUM, 35, 35, 102, ["14"], 186, 2.3),
            _f("AIC845", "AIC", OperationType.ARRIVAL, WakeClass.HEAVY, 38, 38, 66, ["27"], 248, 6.3, PriorityClass.CONNECTION, 0.7, "Fuel-sensitive after holding south of field."),
            _f("AKJ901", "AKJ", OperationType.DEPARTURE, WakeClass.MEDIUM, 42, 42, 108, ["14", "27"], 168, 2.4),
        ],
    ),
    TaskDefinition(
        task_id="bengaluru_irrops_hard",
        title="Bengaluru IRROPS Recovery Command",
        difficulty=Difficulty.HARD,
        airport="VOBL",
        description=(
            "A rapidly evolving irregular-operations event combines wet-runway spacing, "
            "an emergency inbound, a time-critical medical departure, and a heavily "
            "peaking departure bank. The controller must sequence mixed traffic with "
            "both safety and network impact in mind."
        ),
        objective=(
            "Protect the emergency arrival and medical departure, keep all assignments "
            "conflict-free under reduced capacity, minimize system delay, and distribute "
            "unavoidable delay fairly across the participating airlines."
        ),
        grading_focus=[
            "Emergency and medical traffic prioritized",
            "Zero separation and capacity conflicts",
            "Balanced delay distribution under IRROPS",
        ],
        planning_horizon_minutes=140,
        max_steps=6,
        delay_budget=235,
        fuel_budget=1760.0,
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
            _f("IGO110", "IGO", OperationType.DEPARTURE, WakeClass.MEDIUM, 4, 4, 78, ["09L", "09R"], 184, 2.4),
            _f("AIC781", "AIC", OperationType.ARRIVAL, WakeClass.HEAVY, 7, 7, 34, ["09L"], 252, 6.5, PriorityClass.CONNECTION, 0.8),
            _f("VTI220", "VTI", OperationType.DEPARTURE, WakeClass.MEDIUM, 9, 9, 86, ["09R"], 170, 2.6),
            _f("AKJ662", "AKJ", OperationType.DEPARTURE, WakeClass.MEDIUM, 12, 12, 90, ["09L", "09R"], 168, 2.3),
            _f("GOW510", "GOW", OperationType.ARRIVAL, WakeClass.MEDIUM, 14, 14, 46, ["09L", "09R"], 174, 5.3, PriorityClass.NORMAL, 0.4),
            _f("AIC110", "AIC", OperationType.DEPARTURE, WakeClass.MEDIUM, 16, 16, 92, ["09L", "09R"], 162, 2.7, PriorityClass.CONNECTION, 0.7),
            _f("IGO455", "IGO", OperationType.ARRIVAL, WakeClass.MEDIUM, 18, 18, 44, ["09L"], 176, 5.5, PriorityClass.CONNECTION, 0.6),
            _f("VTI918", "VTI", OperationType.DEPARTURE, WakeClass.MEDIUM, 21, 21, 94, ["09R"], 188, 2.5),
            _f("MED001", "GOV", OperationType.ARRIVAL, WakeClass.MEDIUM, 23, 23, 28, ["09L", "09R"], 6, 7.0, PriorityClass.EMERGENCY, 0.1, "Emergency medical diversion with priority handling."),
            _f("AKJ303", "AKJ", OperationType.DEPARTURE, WakeClass.MEDIUM, 25, 25, 98, ["09L", "09R"], 178, 2.4),
            _f("AIX770", "AIX", OperationType.DEPARTURE, WakeClass.MEDIUM, 27, 27, 102, ["09R"], 164, 2.6),
            _f("IGO702", "IGO", OperationType.DEPARTURE, WakeClass.MEDIUM, 30, 30, 108, ["09L", "09R"], 192, 2.4),
            _f("MED208", "AIC", OperationType.DEPARTURE, WakeClass.MEDIUM, 31, 31, 85, ["09R"], 154, 2.8, PriorityClass.MEDICAL, 0.4, "Time-critical organ transport departure."),
            _f("AIC605", "AIC", OperationType.ARRIVAL, WakeClass.HEAVY, 33, 33, 60, ["09L"], 244, 6.4, PriorityClass.CONNECTION, 0.7),
            _f("VTI640", "VTI", OperationType.DEPARTURE, WakeClass.MEDIUM, 36, 36, 112, ["09R"], 182, 2.5),
            _f("GOW099", "GOW", OperationType.DEPARTURE, WakeClass.MEDIUM, 39, 39, 118, ["09L", "09R"], 180, 2.2),
            _f("AKJ818", "AKJ", OperationType.ARRIVAL, WakeClass.MEDIUM, 42, 42, 68, ["09L", "09R"], 171, 5.4, PriorityClass.CONNECTION, 0.5),
            _f("AIC990", "AIC", OperationType.ARRIVAL, WakeClass.HEAVY, 45, 45, 72, ["09L"], 246, 6.6, PriorityClass.CONNECTION, 0.8, "Second fuel-sensitive long-haul inbound wave."),
        ],
    ),
]


def task_catalog() -> Dict[str, TaskDefinition]:
    """Return a dictionary keyed by task id."""

    return {task.task_id: task for task in TASKS}


def ordered_tasks() -> Iterable[TaskDefinition]:
    """Return tasks in easy-to-hard order."""

    return TASKS


def render_task_briefing(task: TaskDefinition) -> str:
    """Build a concise textual briefing for models and humans."""

    runway_lines = [
        f"- Runway {runway.runway_id}: {runway.hourly_capacity}/hr, weather x{runway.weather_penalty:.2f}, "
        f"ops={','.join(op.value for op in runway.allowed_operations)}"
        for runway in task.runways
    ]
    flight_lines = [
        f"- {flight.flight_id} {flight.operation.value} sched={flight.scheduled_minute} "
        f"window=[{flight.earliest_minute},{flight.latest_minute}] "
        f"priority={flight.priority.value} runways={','.join(flight.allowed_runways)} "
        f"pax={flight.passengers}"
        for flight in task.flights
    ]
    return "\n".join(
        [
            f"Airport: {task.airport}",
            f"Scenario: {task.title}",
            f"Difficulty: {task.difficulty.value}",
            f"Objective: {task.objective}",
            "Grading focus:",
            *[f"- {item}" for item in task.grading_focus],
            "Runways:",
            *runway_lines,
            "Flights:",
            *flight_lines,
        ]
    )

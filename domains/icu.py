"""ICU Surge Management domain — three TaskDefinition scenarios for ADAPT.

Entity types (stored in FlightRecord.airline — ADAPT maps these to ATC params):
  TRAUMA   → should map to wake_class=H, priority=emergency
  CARDIAC  → should map to wake_class=M, priority=medical
  POST_OP  → should map to wake_class=M, priority=connection
  ROUTINE  → should map to wake_class=L, priority=normal

Operations:
  ARRIVAL   = patient being admitted to an ICU bed
  DEPARTURE = patient being discharged / transferred out

Resources (runway_id):
  BED_A, BED_B, BED_C, BED_D — shared ICU beds managed by both coordinators

Timing: minute 0 = 08:00, minute 480 = 16:00 (8-hour shift window)
"""

from __future__ import annotations

from typing import Dict

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


ICU_DOMAIN_DESCRIPTION = """\
Domain: Hospital ICU Surge Management
Resources: Four intensive-care beds (BED_A, BED_B, BED_C, BED_D) shared between
  an Admission Coordinator (manages incoming patients) and a
  Discharge Coordinator (manages outgoing patients / transfers).

Entity types and their clinical meaning:
  TRAUMA   — Patient arriving from mass-casualty event or ER. Life-threatening.
             Must be admitted within the narrowest time window. Cannot wait.
  CARDIAC  — Post-cardiac-surgery patient. Time-sensitive but stable for ~30 min.
             High monitoring needs; bed must be fully prepped before arrival.
  POST_OP  — Post-operative patient ready to move to a step-down unit.
             Has a transfer-authorization window (hard deadline from insurance).
  ROUTINE  — Scheduled transfer from step-down or ward. Flexible timing.

Hard deadlines: Some POST_OP discharges have a transfer-authorization expiry —
  the patient MUST vacate the bed before that minute or re-authorization is needed,
  causing a cascade of delays (equivalent to ATFM network slot violations).

Coordination challenge: The Admission Coordinator does not know which beds are
  about to be freed by the Discharge Coordinator. The Discharge Coordinator does
  not know which new patients are incoming. They share the same 4 beds and must
  coordinate through messages to avoid both admitting to an occupied bed and
  discharging into a scheduling gap that blocks the next admission.\
"""

_BEDS = ["BED_A", "BED_B", "BED_C", "BED_D"]

_BED_SPECS = [
    RunwaySpec(
        runway_id=bid,
        hourly_capacity=4,   # max 4 patient changeovers per bed per hour
        weather_penalty=1.0,
        notes="ICU bed — admission and discharge both allowed",
    )
    for bid in _BEDS
]


def icu_task_catalog() -> Dict[str, TaskDefinition]:
    return {
        t.task_id: t
        for t in [_normal_day(), _flu_surge(), _mass_casualty()]
    }


# ── Task 1: Normal ICU Day (Easy) ─────────────────────────────────────────────

def _normal_day() -> TaskDefinition:
    flights = [
        # Admissions (ARRIVAL)
        FlightRecord(
            flight_id="P_ARR_01", airline="ROUTINE", operation=OperationType.ARRIVAL,
            wake_class=WakeClass.MEDIUM,  # placeholder — ADAPT overrides
            scheduled_minute=30, earliest_minute=0, latest_minute=120,
            allowed_runways=_BEDS, passengers=1, fuel_burn_per_minute=1.5,
            priority=PriorityClass.NORMAL,
            notes="Scheduled step-down transfer, routine admission",
        ),
        FlightRecord(
            flight_id="P_ARR_02", airline="ROUTINE", operation=OperationType.ARRIVAL,
            wake_class=WakeClass.MEDIUM,
            scheduled_minute=60, earliest_minute=30, latest_minute=150,
            allowed_runways=_BEDS, passengers=1, fuel_burn_per_minute=1.5,
            priority=PriorityClass.NORMAL,
            notes="Scheduled step-down transfer, routine admission",
        ),
        FlightRecord(
            flight_id="P_ARR_03", airline="CARDIAC", operation=OperationType.ARRIVAL,
            wake_class=WakeClass.MEDIUM,
            scheduled_minute=90, earliest_minute=60, latest_minute=180,
            allowed_runways=_BEDS, passengers=3, fuel_burn_per_minute=5.0,
            priority=PriorityClass.NORMAL,  # placeholder — ADAPT overrides to medical
            connection_risk=0.70,
            notes="Post-cardiac surgery — bed must be fully prepped before arrival",
        ),
        FlightRecord(
            flight_id="P_ARR_04", airline="ROUTINE", operation=OperationType.ARRIVAL,
            wake_class=WakeClass.MEDIUM,
            scheduled_minute=150, earliest_minute=90, latest_minute=240,
            allowed_runways=_BEDS, passengers=1, fuel_burn_per_minute=1.5,
            priority=PriorityClass.NORMAL,
            notes="Scheduled step-down transfer, routine admission",
        ),
        # Discharges (DEPARTURE)
        FlightRecord(
            flight_id="P_DIS_01", airline="POST_OP", operation=OperationType.DEPARTURE,
            wake_class=WakeClass.MEDIUM,
            scheduled_minute=45, earliest_minute=0, latest_minute=180,
            allowed_runways=_BEDS, passengers=1, fuel_burn_per_minute=2.0,
            priority=PriorityClass.NORMAL,  # placeholder — ADAPT overrides to connection
            notes="Transfer-authorization window open; step-down bed confirmed",
        ),
        FlightRecord(
            flight_id="P_DIS_02", airline="POST_OP", operation=OperationType.DEPARTURE,
            wake_class=WakeClass.MEDIUM,
            scheduled_minute=75, earliest_minute=30, latest_minute=180,
            allowed_runways=_BEDS, passengers=1, fuel_burn_per_minute=2.0,
            priority=PriorityClass.NORMAL,
            notes="Transfer-authorization window open; step-down bed confirmed",
        ),
        FlightRecord(
            flight_id="P_DIS_03", airline="POST_OP", operation=OperationType.DEPARTURE,
            wake_class=WakeClass.MEDIUM,
            scheduled_minute=120, earliest_minute=60, latest_minute=240,
            allowed_runways=_BEDS, passengers=1, fuel_burn_per_minute=2.0,
            priority=PriorityClass.NORMAL,
            notes="Transfer-authorization window open",
        ),
        FlightRecord(
            flight_id="P_DIS_04", airline="ROUTINE", operation=OperationType.DEPARTURE,
            wake_class=WakeClass.LIGHT,
            scheduled_minute=180, earliest_minute=90, latest_minute=300,
            allowed_runways=_BEDS, passengers=1, fuel_burn_per_minute=1.5,
            priority=PriorityClass.NORMAL,
            notes="Routine discharge to ward",
        ),
    ]
    return TaskDefinition(
        task_id="icu_normal_day",
        title="ICU Normal Day — Routine Bed Turnover",
        difficulty=Difficulty.EASY,
        airport="GENERAL_HOSPITAL_ICU",
        description=ICU_DOMAIN_DESCRIPTION,
        objective="Coordinate bed admissions and discharges to minimize patient wait time and avoid bed conflicts",
        grading_focus=["bed_conflict_free", "patient_wait_time", "transfer_authorization"],
        planning_horizon_minutes=480,
        max_steps=4,
        delay_budget=60,
        fuel_budget=300.0,
        fairness_tolerance=2.0,
        runways=_BED_SPECS,
        flights=flights,
    )


# ── Task 2: Flu Surge (Medium) ────────────────────────────────────────────────

def _flu_surge() -> TaskDefinition:
    flights = [
        # Admissions
        FlightRecord(
            flight_id="S_ARR_01", airline="CARDIAC", operation=OperationType.ARRIVAL,
            wake_class=WakeClass.MEDIUM,
            scheduled_minute=30, earliest_minute=10, latest_minute=90,
            allowed_runways=_BEDS, passengers=3, fuel_burn_per_minute=5.0,
            priority=PriorityClass.NORMAL, connection_risk=0.70,
            notes="Post-cardiac surgery — time-sensitive admission",
        ),
        FlightRecord(
            flight_id="S_ARR_02", airline="ROUTINE", operation=OperationType.ARRIVAL,
            wake_class=WakeClass.MEDIUM,
            scheduled_minute=45, earliest_minute=20, latest_minute=120,
            allowed_runways=_BEDS, passengers=1, fuel_burn_per_minute=1.5,
            priority=PriorityClass.NORMAL,
            notes="Step-down transfer",
        ),
        FlightRecord(
            flight_id="S_ARR_03", airline="CARDIAC", operation=OperationType.ARRIVAL,
            wake_class=WakeClass.MEDIUM,
            scheduled_minute=60, earliest_minute=40, latest_minute=120,
            allowed_runways=_BEDS, passengers=3, fuel_burn_per_minute=5.0,
            priority=PriorityClass.NORMAL, connection_risk=0.75,
            notes="Post-cardiac surgery — overlapping window with S_ARR_01",
        ),
        FlightRecord(
            flight_id="S_ARR_04", airline="ROUTINE", operation=OperationType.ARRIVAL,
            wake_class=WakeClass.MEDIUM,
            scheduled_minute=90, earliest_minute=60, latest_minute=200,
            allowed_runways=_BEDS, passengers=1, fuel_burn_per_minute=1.5,
            priority=PriorityClass.NORMAL,
            notes="Step-down transfer",
        ),
        FlightRecord(
            flight_id="S_ARR_05", airline="CARDIAC", operation=OperationType.ARRIVAL,
            wake_class=WakeClass.MEDIUM,
            scheduled_minute=100, earliest_minute=80, latest_minute=160,
            allowed_runways=_BEDS, passengers=3, fuel_burn_per_minute=5.0,
            priority=PriorityClass.NORMAL, connection_risk=0.65,
            notes="Post-cardiac surgery — third concurrent CARDIAC creating surge",
        ),
        FlightRecord(
            flight_id="S_ARR_06", airline="ROUTINE", operation=OperationType.ARRIVAL,
            wake_class=WakeClass.MEDIUM,
            scheduled_minute=150, earliest_minute=120, latest_minute=270,
            allowed_runways=_BEDS, passengers=1, fuel_burn_per_minute=1.5,
            priority=PriorityClass.NORMAL,
            notes="Step-down transfer",
        ),
        FlightRecord(
            flight_id="S_ARR_07", airline="ROUTINE", operation=OperationType.ARRIVAL,
            wake_class=WakeClass.MEDIUM,
            scheduled_minute=200, earliest_minute=160, latest_minute=320,
            allowed_runways=_BEDS, passengers=1, fuel_burn_per_minute=1.5,
            priority=PriorityClass.NORMAL,
            notes="Step-down transfer",
        ),
        # Discharges
        FlightRecord(
            flight_id="S_DIS_01", airline="POST_OP", operation=OperationType.DEPARTURE,
            wake_class=WakeClass.MEDIUM,
            scheduled_minute=20, earliest_minute=0, latest_minute=60,
            allowed_runways=_BEDS, passengers=1, fuel_burn_per_minute=2.0,
            priority=PriorityClass.NORMAL,
            notes="TRANSFER AUTH DEADLINE: minute 60 — must vacate before CARDIAC arrivals",
        ),
        FlightRecord(
            flight_id="S_DIS_02", airline="POST_OP", operation=OperationType.DEPARTURE,
            wake_class=WakeClass.MEDIUM,
            scheduled_minute=40, earliest_minute=10, latest_minute=80,
            allowed_runways=_BEDS, passengers=1, fuel_burn_per_minute=2.0,
            priority=PriorityClass.NORMAL,
            notes="TRANSFER AUTH DEADLINE: minute 80",
        ),
        FlightRecord(
            flight_id="S_DIS_03", airline="POST_OP", operation=OperationType.DEPARTURE,
            wake_class=WakeClass.MEDIUM,
            scheduled_minute=70, earliest_minute=40, latest_minute=120,
            allowed_runways=_BEDS, passengers=1, fuel_burn_per_minute=2.0,
            priority=PriorityClass.NORMAL,
            notes="TRANSFER AUTH DEADLINE: minute 120",
        ),
        FlightRecord(
            flight_id="S_DIS_04", airline="POST_OP", operation=OperationType.DEPARTURE,
            wake_class=WakeClass.MEDIUM,
            scheduled_minute=110, earliest_minute=80, latest_minute=180,
            allowed_runways=_BEDS, passengers=1, fuel_burn_per_minute=2.0,
            priority=PriorityClass.NORMAL,
            notes="TRANSFER AUTH DEADLINE: minute 180",
        ),
        FlightRecord(
            flight_id="S_DIS_05", airline="ROUTINE", operation=OperationType.DEPARTURE,
            wake_class=WakeClass.LIGHT,
            scheduled_minute=160, earliest_minute=120, latest_minute=280,
            allowed_runways=_BEDS, passengers=1, fuel_burn_per_minute=1.5,
            priority=PriorityClass.NORMAL,
            notes="Routine discharge to ward, flexible timing",
        ),
    ]
    return TaskDefinition(
        task_id="icu_flu_surge",
        title="ICU Flu Surge — Three Concurrent CARDIAC Admissions",
        difficulty=Difficulty.MEDIUM,
        airport="GENERAL_HOSPITAL_ICU",
        description=ICU_DOMAIN_DESCRIPTION,
        objective="Clear beds before CARDIAC surge arrives while respecting transfer-authorization deadlines",
        grading_focus=["transfer_auth_compliance", "cardiac_admission_delay", "bed_conflict_free"],
        planning_horizon_minutes=480,
        max_steps=4,
        delay_budget=45,
        fuel_budget=400.0,
        fairness_tolerance=2.0,
        runways=_BED_SPECS,
        flights=flights,
    )


# ── Task 3: Mass Casualty Surge (Hard) ────────────────────────────────────────

def _mass_casualty() -> TaskDefinition:
    flights = [
        # TRAUMA admissions — narrow windows, cannot wait
        FlightRecord(
            flight_id="T_ARR_01", airline="TRAUMA", operation=OperationType.ARRIVAL,
            wake_class=WakeClass.MEDIUM,  # placeholder — ADAPT must set to H/emergency
            scheduled_minute=5, earliest_minute=0, latest_minute=10,
            allowed_runways=_BEDS, passengers=5, fuel_burn_per_minute=10.0,
            priority=PriorityClass.NORMAL, connection_risk=0.95,
            notes="EMERGENCY: Mass-casualty arrival — immediate ICU bed required",
        ),
        FlightRecord(
            flight_id="T_ARR_02", airline="TRAUMA", operation=OperationType.ARRIVAL,
            wake_class=WakeClass.MEDIUM,
            scheduled_minute=8, earliest_minute=3, latest_minute=15,
            allowed_runways=_BEDS, passengers=5, fuel_burn_per_minute=10.0,
            priority=PriorityClass.NORMAL, connection_risk=0.95,
            notes="EMERGENCY: Mass-casualty arrival — second trauma patient",
        ),
        FlightRecord(
            flight_id="T_ARR_03", airline="TRAUMA", operation=OperationType.ARRIVAL,
            wake_class=WakeClass.MEDIUM,
            scheduled_minute=12, earliest_minute=8, latest_minute=20,
            allowed_runways=_BEDS, passengers=5, fuel_burn_per_minute=10.0,
            priority=PriorityClass.NORMAL, connection_risk=0.90,
            notes="EMERGENCY: Mass-casualty arrival — third trauma patient",
        ),
        # CARDIAC admissions
        FlightRecord(
            flight_id="T_ARR_04", airline="CARDIAC", operation=OperationType.ARRIVAL,
            wake_class=WakeClass.MEDIUM,
            scheduled_minute=30, earliest_minute=15, latest_minute=75,
            allowed_runways=_BEDS, passengers=3, fuel_burn_per_minute=5.0,
            priority=PriorityClass.NORMAL, connection_risk=0.72,
            notes="Post-cardiac surgery — must admit after trauma queue clears",
        ),
        FlightRecord(
            flight_id="T_ARR_05", airline="CARDIAC", operation=OperationType.ARRIVAL,
            wake_class=WakeClass.MEDIUM,
            scheduled_minute=50, earliest_minute=30, latest_minute=100,
            allowed_runways=_BEDS, passengers=3, fuel_burn_per_minute=5.0,
            priority=PriorityClass.NORMAL, connection_risk=0.68,
            notes="Post-cardiac surgery",
        ),
        FlightRecord(
            flight_id="T_ARR_06", airline="CARDIAC", operation=OperationType.ARRIVAL,
            wake_class=WakeClass.MEDIUM,
            scheduled_minute=80, earliest_minute=60, latest_minute=140,
            allowed_runways=_BEDS, passengers=3, fuel_burn_per_minute=5.0,
            priority=PriorityClass.NORMAL, connection_risk=0.60,
            notes="Post-cardiac surgery",
        ),
        FlightRecord(
            flight_id="T_ARR_07", airline="CARDIAC", operation=OperationType.ARRIVAL,
            wake_class=WakeClass.MEDIUM,
            scheduled_minute=110, earliest_minute=90, latest_minute=170,
            allowed_runways=_BEDS, passengers=3, fuel_burn_per_minute=5.0,
            priority=PriorityClass.NORMAL, connection_risk=0.55,
            notes="Post-cardiac surgery",
        ),
        # ROUTINE admissions
        FlightRecord(
            flight_id="T_ARR_08", airline="ROUTINE", operation=OperationType.ARRIVAL,
            wake_class=WakeClass.LIGHT,
            scheduled_minute=120, earliest_minute=60, latest_minute=240,
            allowed_runways=_BEDS, passengers=1, fuel_burn_per_minute=1.5,
            priority=PriorityClass.NORMAL,
            notes="Step-down transfer — flexible, defer if beds occupied",
        ),
        FlightRecord(
            flight_id="T_ARR_09", airline="ROUTINE", operation=OperationType.ARRIVAL,
            wake_class=WakeClass.LIGHT,
            scheduled_minute=160, earliest_minute=100, latest_minute=280,
            allowed_runways=_BEDS, passengers=1, fuel_burn_per_minute=1.5,
            priority=PriorityClass.NORMAL,
            notes="Step-down transfer — flexible",
        ),
        FlightRecord(
            flight_id="T_ARR_10", airline="ROUTINE", operation=OperationType.ARRIVAL,
            wake_class=WakeClass.LIGHT,
            scheduled_minute=200, earliest_minute=150, latest_minute=330,
            allowed_runways=_BEDS, passengers=1, fuel_burn_per_minute=1.5,
            priority=PriorityClass.NORMAL,
            notes="Step-down transfer — flexible",
        ),
        # Discharges — must clear beds for trauma wave
        FlightRecord(
            flight_id="T_DIS_01", airline="POST_OP", operation=OperationType.DEPARTURE,
            wake_class=WakeClass.MEDIUM,
            scheduled_minute=0, earliest_minute=0, latest_minute=8,
            allowed_runways=_BEDS, passengers=1, fuel_burn_per_minute=2.0,
            priority=PriorityClass.NORMAL,
            notes="URGENT: Must vacate immediately — trauma bay needs bed now",
        ),
        FlightRecord(
            flight_id="T_DIS_02", airline="POST_OP", operation=OperationType.DEPARTURE,
            wake_class=WakeClass.MEDIUM,
            scheduled_minute=2, earliest_minute=0, latest_minute=10,
            allowed_runways=_BEDS, passengers=1, fuel_burn_per_minute=2.0,
            priority=PriorityClass.NORMAL,
            notes="URGENT: Must vacate for second trauma patient — TRANSFER AUTH DEADLINE: minute 10",
        ),
        FlightRecord(
            flight_id="T_DIS_03", airline="POST_OP", operation=OperationType.DEPARTURE,
            wake_class=WakeClass.MEDIUM,
            scheduled_minute=5, earliest_minute=0, latest_minute=15,
            allowed_runways=_BEDS, passengers=1, fuel_burn_per_minute=2.0,
            priority=PriorityClass.NORMAL,
            notes="Must vacate for third trauma — TRANSFER AUTH DEADLINE: minute 15",
        ),
        FlightRecord(
            flight_id="T_DIS_04", airline="POST_OP", operation=OperationType.DEPARTURE,
            wake_class=WakeClass.MEDIUM,
            scheduled_minute=20, earliest_minute=10, latest_minute=60,
            allowed_runways=_BEDS, passengers=1, fuel_burn_per_minute=2.0,
            priority=PriorityClass.NORMAL,
            notes="TRANSFER AUTH DEADLINE: minute 60",
        ),
        FlightRecord(
            flight_id="T_DIS_05", airline="ROUTINE", operation=OperationType.DEPARTURE,
            wake_class=WakeClass.LIGHT,
            scheduled_minute=40, earliest_minute=20, latest_minute=120,
            allowed_runways=_BEDS, passengers=1, fuel_burn_per_minute=1.5,
            priority=PriorityClass.NORMAL,
            notes="Routine discharge — flex to accommodate trauma clearance",
        ),
        FlightRecord(
            flight_id="T_DIS_06", airline="ROUTINE", operation=OperationType.DEPARTURE,
            wake_class=WakeClass.LIGHT,
            scheduled_minute=90, earliest_minute=60, latest_minute=200,
            allowed_runways=_BEDS, passengers=1, fuel_burn_per_minute=1.5,
            priority=PriorityClass.NORMAL,
            notes="Routine discharge",
        ),
    ]
    return TaskDefinition(
        task_id="icu_mass_casualty",
        title="ICU Mass Casualty — Three Trauma Emergencies Simultaneous",
        difficulty=Difficulty.HARD,
        airport="GENERAL_HOSPITAL_ICU",
        description=ICU_DOMAIN_DESCRIPTION,
        objective="Immediately clear beds for 3 simultaneous TRAUMA arrivals while managing CARDIAC queue",
        grading_focus=["trauma_admission_time", "transfer_auth_compliance", "cardiac_sequencing"],
        planning_horizon_minutes=480,
        max_steps=4,
        delay_budget=30,
        fuel_budget=600.0,
        fairness_tolerance=3.0,
        runways=_BED_SPECS,
        flights=flights,
    )

"""Shared constants for ATC optimization environment.

This module centralizes all magic numbers and configuration constants
to support the DRY (Don't Repeat Yourself) principle and ease tuning.
"""

from typing import Dict, Tuple

# ============================================================================
# Wake Turbulence Separation Rules (minutes)
# ============================================================================
# Minimum spacing required between aircraft based on wake classes
# H=Heavy, M=Medium, L=Light
SEPARATION_BY_WAKE: Dict[Tuple[str, str], int] = {
    ("H", "H"): 4,  # Heavy to Heavy: 4 min
    ("H", "M"): 5,  # Heavy to Medium: 5 min
    ("H", "L"): 6,  # Heavy to Light: 6 min
    ("M", "H"): 3,  # Medium to Heavy: 3 min
    ("M", "M"): 3,  # Medium to Medium: 3 min
    ("M", "L"): 4,  # Medium to Light: 4 min
    ("L", "H"): 3,  # Light to Heavy: 3 min
    ("L", "M"): 3,  # Light to Medium: 3 min
    ("L", "L"): 3,  # Light to Light: 3 min
}



# ============================================================================
# Normalized Score Calculation Weights
# ============================================================================
# Weights for combining multiple metrics into final score
# Sum must equal 1.0 for balanced weighting
SCORE_WEIGHTS: Dict[str, float] = {
    "completeness": 0.24,        # Schedule completeness (24%)
    "conflict_free": 0.24,       # Conflict-free ratio (24%)
    "priority": 0.18,            # Priority handling (18%)
    "delay": 0.16,               # Delay efficiency (16%)
    "fairness": 0.10,            # Fairness across airlines (10%)
    "fuel": 0.08,                # Fuel efficiency (8%)
}

# Validate weights sum to 1.0
_WEIGHT_SUM = sum(SCORE_WEIGHTS.values())
assert abs(_WEIGHT_SUM - 1.0) < 0.001, (
    f"SCORE_WEIGHTS must sum to 1.0, got {_WEIGHT_SUM}"
)

# ============================================================================
# Score Penalty Factors
# ============================================================================
# Multipliers applied when schedule quality is suboptimal
COMPLETENESS_PENALTY_FACTOR = 1.0   # Apply when incomplete schedules
CONFLICT_PENALTY_FACTOR = 1.0       # Apply when conflicts detected

# ============================================================================
# Rounding Precision Configuration
# ============================================================================
# Number of decimal places for different metric types
METRIC_PRECISION = 4           # Normalized metrics (0.0-1.0): 4 decimals
FUEL_PRECISION = 2             # Fuel burn in tonnes: 2 decimals
AIRLINE_DELAY_PRECISION = 1    # Airline average delays in minutes: 1 decimal
TIME_PRECISION = 0             # Time values in minutes: integers

# ============================================================================
# Recommendation Thresholds
# ============================================================================
# Thresholds for triggering operational recommendations
FAIRNESS_WARNING_THRESHOLD = 0.7    # Warn if fairness < 70%
FUEL_EFFICIENCY_WARNING_THRESHOLD = 0.7  # Warn if fuel efficiency < 70%

# ============================================================================
# Capacity Calculation
# ============================================================================
# Minimum capacity spacing floor
MINIMUM_RUNWAY_SPACING = 2      # Minimum 2-minute spacing between any operations

# ============================================================================
# Diagnostic/Recommendation Output
# ============================================================================
# Limits on diagnostic and recommendation messages
MAX_DIAGNOSTICS = 12            # Maximum diagnostic messages to include
MAX_RECOMMENDATIONS = 6         # Maximum recommendations to include

# ============================================================================
# Inference Configuration
# ============================================================================
# Defaults for baseline inference script
DEFAULT_MAX_STEPS = 4           # Default iterations per task
DEFAULT_MAX_TOKENS = 1400       # Default LLM max tokens per response
DEFAULT_TEMPERATURE = 0         # Temperature for deterministic responses
DEFAULT_SUCCESS_THRESHOLD = 0.65  # Minimum score to consider "success"

# ============================================================================
# Validation Messages
# ============================================================================
# Common validation and error messages
INVALID_RUNWAY_MSG_TEMPLATE = "{flight_id} cannot use runway {runway}; allowed runways are {allowed}."
OUTSIDE_WINDOW_MSG_TEMPLATE = "{flight_id} is assigned outside its feasible window [{earliest}, {latest}]."
PRIORITY_TOLERANCE_MSG_TEMPLATE = "{flight_id} exceeds the {priority} delay tolerance of {tolerance} minutes."
SPACING_VIOLATION_MSG_TEMPLATE = "Runway {runway} has {prev_flight}->{curr_flight} spaced {actual} minutes apart; needs {required}."

# ============================================================================
# Export for use in other modules
# ============================================================================
__all__ = [
    "SEPARATION_BY_WAKE",
    "SCORE_WEIGHTS",
    "COMPLETENESS_PENALTY_FACTOR",
    "CONFLICT_PENALTY_FACTOR",
    "METRIC_PRECISION",
    "FUEL_PRECISION",
    "AIRLINE_DELAY_PRECISION",
    "TIME_PRECISION",
    "FAIRNESS_WARNING_THRESHOLD",
    "FUEL_EFFICIENCY_WARNING_THRESHOLD",
    "MINIMUM_RUNWAY_SPACING",
    "MAX_DIAGNOSTICS",
    "MAX_RECOMMENDATIONS",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_SUCCESS_THRESHOLD",
    "INVALID_RUNWAY_MSG_TEMPLATE",
    "OUTSIDE_WINDOW_MSG_TEMPLATE",
    "PRIORITY_TOLERANCE_MSG_TEMPLATE",
    "SPACING_VIOLATION_MSG_TEMPLATE",
    # Multi-agent constants
    "AMAN_EMERGENCY_DELAY_TOLERANCE",
    "DMAN_EMERGENCY_DELAY_TOLERANCE",
    "ATFM_DEADLINE_BUFFER_MINUTES",
    "CROSS_LANE_CONFLICT_PENALTY",
    "THEORY_OF_MIND_BONUS",
    "GENERATOR_ESCALATION_THRESHOLD",
    "GENERATOR_FLOOR_THRESHOLD",
    "COORDINATION_SCORE_WEIGHTS",
]


# ============================================================================
# Multi-Agent AMAN/DMAN Constants
# ============================================================================

# Emergency delay tolerances per role (minutes)
AMAN_EMERGENCY_DELAY_TOLERANCE: int = 5   # EMERGENCY arrival must land within 5 min of window
DMAN_EMERGENCY_DELAY_TOLERANCE: int = 5   # EMERGENCY departure must push within 5 min

# ATFM network slot buffer: realistic GDP margin (minutes)
ATFM_DEADLINE_BUFFER_MINUTES: int = 12

# Reward shaping constants
CROSS_LANE_CONFLICT_PENALTY: float = 0.15  # per conflict between AMAN/DMAN slots
THEORY_OF_MIND_BONUS: float = 0.25         # pre-emptive coordination without prompt

# Generator adaptive curriculum thresholds
GENERATOR_ESCALATION_THRESHOLD: float = 0.65  # escalate when agents score above this
GENERATOR_FLOOR_THRESHOLD: float = 0.30        # ease when agents score below this

# Coordination score component weights
COORDINATION_SCORE_WEIGHTS = {
    "zero_cross_conflicts": 0.25,
    "negotiation_efficiency": 0.20,  # 0 rounds=0.20, 1 round=0.10, 2+=0.0
    "emergency_handling": 0.30,
    "preemptive_gap": 0.25,
}

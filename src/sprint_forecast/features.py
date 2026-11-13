"""Feature engineering: turn raw sprint facts into the ratios a scrum master would reason about."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "commitment_ratio",      # committed / recent velocity: the single biggest driver
    "availability",          # share of person-days actually available
    "velocity_volatility",   # how unpredictable the team has been (coefficient of variation)
    "carryover_share",       # share of this sprint that is leftover work
    "large_story_share",
    "external_dependencies",
    "unplanned_share",       # mid-sprint work as a share of the commitment
    "new_member_share",
    "sprint_days",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    velocity = df["velocity_last_3"].clip(lower=1)
    person_days = (df["team_size"] * df["sprint_days"]).clip(lower=1)
    committed = df["committed_points"].clip(lower=1)
    return pd.DataFrame({
        "commitment_ratio": df["committed_points"] / velocity,
        "availability": 1 - (df["leave_days"] / person_days).clip(0, 1),
        "velocity_volatility": df["velocity_std_last_3"] / velocity,
        "carryover_share": (df["carried_over_points"] / committed).clip(0, 1),
        "large_story_share": df["large_story_share"],
        "external_dependencies": df["external_dependencies"],
        "unplanned_share": df["unplanned_points"] / committed,
        "new_member_share": df["new_members"] / df["team_size"].clip(lower=1),
        "sprint_days": df["sprint_days"],
    }, index=df.index)[FEATURES].replace([np.inf, -np.inf], 0).fillna(0)

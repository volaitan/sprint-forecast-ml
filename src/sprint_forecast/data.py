"""Sprint history data: a realistic synthetic generator and a loader for real exports.

Real sprint data is private to each team, so the project ships with a generator whose
hidden rules mirror what agile teams commonly see: over-commitment, absences, unplanned
work, big stories and cross-team dependencies all lower the chance of delivering.
The model is never told these rules; it has to learn them from the data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_COLUMNS = [
    "team_id",
    "sprint_number",
    "sprint_days",           # working days in the sprint
    "team_size",
    "leave_days",            # person-days lost to holiday or sickness
    "committed_points",
    "velocity_last_3",       # mean completed points over the previous three sprints
    "velocity_std_last_3",   # standard deviation of those three
    "carried_over_points",   # unfinished work brought in from the last sprint
    "large_story_share",     # share of committed points in stories of 8+ points
    "external_dependencies", # stories waiting on another team
    "unplanned_points",      # work added mid-sprint (bugs, support)
    "new_members",           # people in their first or second sprint
]
TARGET = "delivered"         # 1 if at least 90% of committed points were completed
DELIVERY_THRESHOLD = 0.9


def generate(n_teams: int = 40, sprints_per_team: int = 30, seed: int = 7) -> pd.DataFrame:
    """Simulate sprint histories for several teams. Same seed -> same data."""
    rng = np.random.default_rng(seed)
    rows = []
    for team in range(n_teams):
        size = int(rng.integers(3, 10))
        base_rate = rng.normal(1.6, 0.35)            # points per person-day for this team
        sprint_days = int(rng.choice([5, 10, 10, 10, 15]))
        history: list[float] = [max(5.0, base_rate * size * sprint_days * rng.normal(0.85, 0.1)) for _ in range(3)]
        carried = 0.0

        for sprint in range(1, sprints_per_team + 1):
            v_mean, v_std = float(np.mean(history[-3:])), float(np.std(history[-3:]))
            # Some sprints fall in holiday season, when far more people are away
            leave_rate = rng.choice([0.02, 0.05, 0.05, 0.18])
            leave = float(rng.poisson(size * sprint_days * leave_rate))
            new_members = int(rng.binomial(1, 0.12)) * int(rng.integers(1, 3))
            # Teams vary in how ambitious they are when planning
            ambition = rng.normal(1.08, 0.2)
            committed = max(3.0, round(v_mean * ambition + carried * 0.6))
            large_share = float(np.clip(rng.beta(2, 5), 0, 1))
            deps = int(rng.poisson(0.8))
            unplanned = float(max(0.0, rng.normal(0.06, 0.06)) * committed)

            # ---- Hidden "true" capacity for this sprint ----
            capacity_factor = 1 - leave / (size * sprint_days)
            capacity = base_rate * size * sprint_days * capacity_factor
            capacity *= 1 - 0.12 * new_members / size * 3
            capacity *= rng.normal(1.0, 0.12)                       # everyday noise
            effective_work = committed + unplanned * 0.9
            risk = 1 + 0.35 * large_share + 0.06 * deps             # big stories and dependencies slip
            completed = min(committed, capacity / risk * committed / max(effective_work, 1))
            completed = max(0.0, round(completed))

            rows.append({
                "team_id": f"T{team:02d}",
                "sprint_number": sprint,
                "sprint_days": sprint_days,
                "team_size": size,
                "leave_days": leave,
                "committed_points": committed,
                "velocity_last_3": round(v_mean, 1),
                "velocity_std_last_3": round(v_std, 1),
                "carried_over_points": round(carried),
                "large_story_share": round(large_share, 2),
                "external_dependencies": deps,
                "unplanned_points": round(unplanned, 1),
                "new_members": new_members,
                "completed_points": completed,
            })
            history.append(completed)
            carried = max(0.0, committed - completed)

    df = pd.DataFrame(rows)
    df[TARGET] = (df["completed_points"] >= DELIVERY_THRESHOLD * df["committed_points"]).astype(int)
    return df


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load real sprint history. Needs every RAW_COLUMNS field plus completed_points."""
    df = pd.read_csv(path)
    missing = [c for c in RAW_COLUMNS + ["completed_points"] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {', '.join(missing)}")
    if (df["committed_points"] <= 0).any():
        raise ValueError("committed_points must be positive")
    df[TARGET] = (df["completed_points"] >= DELIVERY_THRESHOLD * df["committed_points"]).astype(int)
    return df

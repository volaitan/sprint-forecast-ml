"""Train, compare, evaluate and save models.

Splits are made by team (GroupKFold / GroupShuffleSplit), so the test set only contains
teams the model has never seen. Splitting sprints randomly would leak each team's habits
into the test set and make the scores look better than they really are.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import TARGET
from .features import FEATURES, build_features


def candidate_models(seed: int = 0) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]),
        "random_forest": Pipeline([("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=seed, n_jobs=-1))]),
        "gradient_boosting": Pipeline([("model", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed))]),
    }


@dataclass
class TrainingReport:
    best_model: str
    cv_auc: dict[str, float]
    test_metrics: dict[str, float]
    importances: dict[str, float]
    base_rate: float
    n_train: int
    n_test: int
    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)


def train(df: pd.DataFrame, seed: int = 0) -> tuple[Pipeline, TrainingReport, pd.DataFrame]:
    X, y, groups = build_features(df), df[TARGET].to_numpy(), df["team_id"].to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test, y_train, y_test = X.iloc[train_idx], X.iloc[test_idx], y[train_idx], y[test_idx]

    cv = GroupKFold(n_splits=5)
    cv_auc = {}
    for name, pipe in candidate_models(seed).items():
        scores = cross_val_score(pipe, X_train, y_train, groups=groups[train_idx], cv=cv, scoring="roc_auc")
        cv_auc[name] = round(float(scores.mean()), 4)
    best = max(cv_auc, key=cv_auc.get)

    # Calibrate so "70%" really means right about 70% of the time.
    model = CalibratedClassifierCV(candidate_models(seed)[best], method="sigmoid", cv=5)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    test_metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
        "brier": round(float(brier_score_loss(y_test, proba)), 4),
        "baseline_accuracy": round(float(max(y_test.mean(), 1 - y_test.mean())), 4),
    }

    imp = permutation_importance(model, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=seed)
    importances = dict(sorted(
        ((f, round(float(v), 4)) for f, v in zip(FEATURES, imp.importances_mean)), key=lambda kv: kv[1], reverse=True
    ))

    report = TrainingReport(best, cv_auc, test_metrics, importances, round(float(y.mean()), 4), len(train_idx), len(test_idx))
    test_frame = X_test.assign(actual=y_test, predicted=proba)
    return model, report, test_frame


def save(model, report: TrainingReport, folder: str | Path) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, folder / "model.joblib")
    (folder / "metrics.json").write_text(report.to_json())


def load(folder: str | Path):
    return joblib.load(Path(folder) / "model.joblib")


def predict_proba(model, sprint: dict) -> float:
    """Probability that one planned sprint delivers at least 90% of its commitment."""
    X = build_features(pd.DataFrame([sprint]))
    return float(model.predict_proba(X)[:, 1][0])


def recommend_commitment(model, sprint: dict, target: float = 0.8) -> int | None:
    """Largest commitment (in points) that still gives at least `target` chance of delivering."""
    best = None
    velocity = max(1, int(sprint["velocity_last_3"]))
    for points in range(1, int(velocity * 2) + 1):
        if predict_proba(model, {**sprint, "committed_points": points}) >= target:
            best = points
    return best

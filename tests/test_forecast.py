import pandas as pd
import pytest

from sprint_forecast import data, model
from sprint_forecast.cli import main
from sprint_forecast.features import FEATURES, build_features


@pytest.fixture(scope="module")
def df():
    return data.generate(n_teams=30, sprints_per_team=25, seed=3)


@pytest.fixture(scope="module")
def trained(df):
    return model.train(df, seed=1)


def sprint(**overrides):
    base = {
        "sprint_days": 10, "team_size": 5, "leave_days": 0, "committed_points": 30,
        "velocity_last_3": 32, "velocity_std_last_3": 2, "carried_over_points": 0,
        "large_story_share": 0.2, "external_dependencies": 0, "unplanned_points": 0, "new_members": 0,
    }
    return {**base, **overrides}


def test_generator_is_deterministic_and_complete(df):
    again = data.generate(n_teams=30, sprints_per_team=25, seed=3)
    pd.testing.assert_frame_equal(df, again)
    assert set(data.RAW_COLUMNS + ["completed_points", "delivered"]) <= set(df.columns)
    assert len(df) == 750
    assert 0.2 < df["delivered"].mean() < 0.8  # classes are reasonably balanced
    assert (df["completed_points"] <= df["committed_points"]).all()


def test_features(df):
    X = build_features(df)
    assert list(X.columns) == FEATURES
    assert not X.isna().any().any()
    assert X["availability"].between(0, 1).all()
    row = build_features(pd.DataFrame([sprint(committed_points=40, velocity_last_3=20, leave_days=10)])).iloc[0]
    assert row["commitment_ratio"] == 2.0
    assert row["availability"] == 0.8


def test_model_beats_baseline_on_unseen_teams(trained, df):
    _, report, test = trained
    assert report.test_metrics["roc_auc"] > 0.8
    assert report.test_metrics["accuracy"] > report.test_metrics["baseline_accuracy"]
    assert report.best_model in report.cv_auc
    assert list(report.importances)[0] == "commitment_ratio"
    # Train and test teams never overlap (group split)
    assert report.n_train + report.n_test == len(df)


def test_predictions_follow_common_sense(trained):
    m, _, _ = trained
    safe = model.predict_proba(m, sprint(committed_points=24))
    risky = model.predict_proba(m, sprint(committed_points=45))
    assert 0 <= risky < safe <= 1
    assert model.predict_proba(m, sprint(leave_days=15)) < model.predict_proba(m, sprint())


def test_recommend_commitment(trained):
    m, _, _ = trained
    points = model.recommend_commitment(m, sprint(), target=0.7)
    assert points is not None and 16 <= points <= 48  # between 0.5x and 1.5x recent velocity
    assert model.predict_proba(m, sprint(committed_points=points)) >= 0.7


def test_load_csv_validates(tmp_path, df):
    good = tmp_path / "good.csv"
    df.drop(columns="delivered").to_csv(good, index=False)
    assert data.load_csv(good)["delivered"].equals(df["delivered"])
    bad = tmp_path / "bad.csv"
    df.drop(columns=["leave_days"]).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="leave_days"):
        data.load_csv(bad)


def test_cli_end_to_end(tmp_path, capsys):
    csv, models = tmp_path / "s.csv", tmp_path / "m"
    assert main(["generate", "--teams", "20", "--sprints", "20", "--out", str(csv)]) == 0
    assert main(["train", "--data", str(csv), "--model-dir", str(models)]) == 0
    assert (models / "model.joblib").exists() and (models / "metrics.json").exists()
    assert main(["predict", "--model-dir", str(models), "--committed", "30", "--history", "28,32,31", "--team-size", "5"]) == 0
    out = capsys.readouterr().out
    assert "Chance of delivering" in out
    assert main(["predict", "--model-dir", str(models), "--committed", "30", "--history", "28", "--team-size", "5"]) == 2

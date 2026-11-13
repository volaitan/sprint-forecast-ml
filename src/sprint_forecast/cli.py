"""Command line: generate data, train, and forecast a planned sprint."""
from __future__ import annotations

import argparse
import sys

from . import data, model, report

DEFAULT_MODEL_DIR = "models"


def cmd_generate(args) -> int:
    df = data.generate(args.teams, args.sprints, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} sprints from {args.teams} teams to {args.out} ({df['delivered'].mean():.0%} delivered)")
    return 0


def cmd_train(args) -> int:
    df = data.load_csv(args.data) if args.data else data.generate(seed=args.seed)
    fitted, rep, test = model.train(df, seed=args.seed)
    model.save(fitted, rep, args.model_dir)
    print(report.markdown(rep))
    print("\nTop drivers:", ", ".join(list(rep.importances)[:4]))
    if args.charts:
        for p in report.make_charts(df, rep, test, args.charts):
            print("Chart:", p)
    print(f"\nModel saved to {args.model_dir}/model.joblib")
    return 0


def cmd_predict(args) -> int:
    history = [float(x) for x in args.history.split(",") if x.strip()]
    if len(history) < 2:
        print("--history needs at least two past velocities, e.g. 32,35,30", file=sys.stderr)
        return 2
    import numpy as np

    sprint = {
        "sprint_days": args.sprint_days, "team_size": args.team_size, "leave_days": args.leave_days,
        "committed_points": args.committed, "velocity_last_3": float(np.mean(history[-3:])),
        "velocity_std_last_3": float(np.std(history[-3:])), "carried_over_points": args.carried_over,
        "large_story_share": args.large_story_share, "external_dependencies": args.dependencies,
        "unplanned_points": args.unplanned, "new_members": args.new_members,
    }
    m = model.load(args.model_dir)
    p = model.predict_proba(m, sprint)
    verdict = "on track" if p >= 0.7 else "at risk" if p >= 0.4 else "unlikely"
    print(f"Chance of delivering at least 90% of {args.committed:g} points: {p:.0%} ({verdict})")
    safe = model.recommend_commitment(m, sprint, args.confidence)
    if safe:
        print(f"For a {args.confidence:.0%} chance, commit to about {safe} points or fewer.")
    else:
        print(f"No commitment reaches {args.confidence:.0%} confidence with this team setup.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sprint-forecast", description="Predict whether a sprint will deliver.")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="Create a synthetic sprint-history CSV")
    g.add_argument("--teams", type=int, default=40)
    g.add_argument("--sprints", type=int, default=30)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--out", default="data/sprints.csv")
    g.set_defaults(func=cmd_generate)

    t = sub.add_parser("train", help="Train and evaluate models")
    t.add_argument("--data", help="Your own sprint CSV (defaults to synthetic data)")
    t.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    t.add_argument("--charts", metavar="DIR", help="Also save evaluation charts to DIR")
    t.add_argument("--seed", type=int, default=7)
    t.set_defaults(func=cmd_train)

    f = sub.add_parser("predict", help="Forecast a planned sprint")
    f.add_argument("--committed", type=float, required=True, help="Story points you plan to commit to")
    f.add_argument("--history", required=True, help="Recent completed points, oldest first, e.g. 32,35,30")
    f.add_argument("--team-size", type=int, required=True)
    f.add_argument("--sprint-days", type=int, default=10)
    f.add_argument("--leave-days", type=float, default=0)
    f.add_argument("--carried-over", type=float, default=0)
    f.add_argument("--large-story-share", type=float, default=0.2)
    f.add_argument("--dependencies", type=int, default=0)
    f.add_argument("--unplanned", type=float, default=0)
    f.add_argument("--new-members", type=int, default=0)
    f.add_argument("--confidence", type=float, default=0.8)
    f.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    f.set_defaults(func=cmd_predict)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

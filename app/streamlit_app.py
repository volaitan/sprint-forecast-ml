"""Interactive demo: plan a sprint and see the chance of delivering it.

Run with:  streamlit run app/streamlit_app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from sprint_forecast import data, model

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

st.set_page_config(page_title="Sprint Forecast", page_icon="📈", layout="centered")


@st.cache_resource
def get_model():
    if (MODEL_DIR / "model.joblib").exists():
        return model.load(MODEL_DIR)
    fitted, _, _ = model.train(data.generate())
    model.save(fitted, _, MODEL_DIR)
    return fitted


st.title("📈 Sprint Forecast")
st.write("Plan your next sprint and see how likely the team is to deliver at least 90% of what it commits to.")

m = get_model()

with st.form("plan"):
    c1, c2 = st.columns(2)
    history = c1.text_input("Points completed in recent sprints (oldest first)", "30, 34, 31")
    committed = c2.number_input("Points you plan to commit", 1, 500, 36)
    team_size = c1.number_input("Team size", 1, 30, 5)
    sprint_days = c2.selectbox("Sprint length (working days)", [5, 10, 15], index=1)
    leave = c1.number_input("Person-days of leave this sprint", 0.0, 200.0, 2.0)
    carried = c2.number_input("Points carried over from last sprint", 0.0, 200.0, 0.0)
    large = c1.slider("Share of points in large stories (8+)", 0.0, 1.0, 0.25)
    deps = c2.number_input("Stories waiting on other teams", 0, 20, 1)
    unplanned = c1.number_input("Expected unplanned work (points)", 0.0, 100.0, 2.0)
    new_members = c2.number_input("New team members", 0, 10, 0)
    submitted = st.form_submit_button("Forecast")

try:
    past = [float(x) for x in history.split(",") if x.strip()]
except ValueError:
    past = []

if len(past) < 2:
    st.error("Enter at least two past velocities, separated by commas.")
    st.stop()

plan = {
    "sprint_days": sprint_days, "team_size": team_size, "leave_days": leave, "committed_points": committed,
    "velocity_last_3": float(np.mean(past[-3:])), "velocity_std_last_3": float(np.std(past[-3:])),
    "carried_over_points": carried, "large_story_share": large, "external_dependencies": deps,
    "unplanned_points": unplanned, "new_members": new_members,
}
p = model.predict_proba(m, plan)
label = "On track ✅" if p >= 0.7 else "At risk ⚠️" if p >= 0.4 else "Unlikely ❌"

c1, c2, c3 = st.columns(3)
c1.metric("Chance of delivering", f"{p:.0%}")
c2.metric("Verdict", label)
safe = model.recommend_commitment(m, plan, 0.8)
c3.metric("Commit for 80% confidence", f"≤ {safe} pts" if safe else "n/a")

st.subheader("How commitment changes the odds")
points = list(range(max(1, int(plan["velocity_last_3"] * 0.5)), int(plan["velocity_last_3"] * 1.6) + 1))
curve = pd.DataFrame({"Committed points": points, "Chance of delivering": [model.predict_proba(m, {**plan, "committed_points": x}) for x in points]})
st.line_chart(curve, x="Committed points", y="Chance of delivering")
st.caption("Trained on synthetic sprint histories from 40 teams. Train on your own data with `sprint-forecast train --data your.csv`.")

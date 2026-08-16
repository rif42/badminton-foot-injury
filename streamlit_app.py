"""Streamlit front-end for the badminton lower-body injury-risk analyzer.

This app wraps the existing Python analysis pipeline in `src/badminton_risk` so
that users can upload a short video clip, run pose analysis, and inspect the
annotated video plus per-frame CSV report in the browser.
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# Make the `src` package tree importable when `streamlit run` is launched from
# the repository root.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "src"))

from badminton_risk.video_risk_analyzer import analyze_video  # noqa: E402


st.set_page_config(
    page_title="Badminton Lower-Body Injury-Risk Analyzer",
    layout="wide",
)

st.title("Badminton Lower-Body Injury-Risk Analyzer")
st.caption(
    "Educational/demo tool only. This is not a medical diagnostic device. "
    "Consult a qualified clinician for any injury concerns."
)


def _compute_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Return a summary dict for the analyzed dataframe."""
    total = len(df)
    risky = df[df["status"] == "risky"] if "status" in df.columns else pd.DataFrame()
    risky_count = len(risky)
    if risky_count and "core_risk" in risky.columns and "time_sec" in risky.columns:
        peak_risk = float(risky["core_risk"].max())
        start_time = float(risky["time_sec"].min())
        end_time = float(risky["time_sec"].max())
    else:
        peak_risk = None
        start_time = None
        end_time = None
    return {
        "total": total,
        "risky_count": risky_count,
        "peak_risk": peak_risk,
        "start_time": start_time,
        "end_time": end_time,
    }


uploaded_file = st.file_uploader(
    "Upload a short video clip",
    type=["mp4", "mov", "avi", "webm", "mkv"],
)

analyze_clicked = st.button(
    "Analyze",
    disabled=uploaded_file is None,
)

if analyze_clicked and uploaded_file is not None:
    with st.spinner("Analyzing video. Short clips (< 30 s) are recommended..."):
        try:
            ext = os.path.splitext(uploaded_file.name)[1].lower() or ".mp4"
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                input_path = tmp_path / f"input{ext}"
                input_path.write_bytes(uploaded_file.getvalue())

                csv_path = tmp_path / "risk_report.csv"
                video_path = tmp_path / "annotated.mp4"

                analyze_video(
                    input_path=str(input_path),
                    output_csv=str(csv_path),
                    output_video=str(video_path),
                    show_preview=False,
                )

                if csv_path.exists() and csv_path.stat().st_size > 0:
                    df = pd.read_csv(csv_path)
                else:
                    df = pd.DataFrame()

                video_bytes = (
                    video_path.read_bytes() if video_path.exists() else b""
                )
                csv_bytes = (
                    csv_path.read_bytes() if csv_path.exists() else b""
                )

                st.session_state["analysis"] = {
                    "df": df,
                    "video_bytes": video_bytes,
                    "csv_bytes": csv_bytes,
                }
        except Exception as exc:  # noqa: BLE001
            st.session_state["analysis"] = None
            st.error(f"Analysis failed: {exc}")
            with st.expander("Stack trace"):
                st.code(traceback.format_exc())


analysis = st.session_state.get("analysis")
if analysis is None:
    st.info("Upload a video and click **Analyze** to begin.")
else:
    df: pd.DataFrame = analysis["df"]
    video_bytes: bytes = analysis["video_bytes"]

    if video_bytes:
        st.video(video_bytes, format="video/mp4")
    else:
        st.warning("No annotated video was produced.")

    summary = _compute_summary(df)
    total = summary["total"]
    risky_count = summary["risky_count"]
    peak_risk = summary["peak_risk"]
    start_time = summary["start_time"]
    end_time = summary["end_time"]

    if risky_count and peak_risk is not None:
        st.metric(
            label="Risky frames / total frames",
            value=f"{risky_count} / {total}",
        )
        st.metric(label="Peak risk", value=f"{peak_risk:.3f}")
        st.metric(
            label="Risky time range",
            value=f"{start_time:.3f}s – {end_time:.3f}s",
        )
    else:
        st.write(f"Total frames analyzed: **{total}** — no risky frames detected.")

    st.divider()

    show_all = st.checkbox("Show all frames", value=False)
    if df.empty or "status" not in df.columns:
        display_df = df
    else:
        display_df = df if show_all else df[df["status"] == "risky"]

    st.write(
        f"Showing **{len(display_df)}** "
        f"{'frames' if show_all else 'risky frames'} "
        f"out of **{total}** total frames."
    )

    if not display_df.empty:
        numeric_cols = [
            "core_risk",
            "knee_stiffness_risk",
            "ankle_foot_alignment_risk",
            "ankle_roll_risk",
            "ankle_roll_angle_deg",
            "hip_displacement_proxy",
            "landing_asymmetry_score",
        ]
        injury_cols = [
            "injury_names",
            "injury_descriptions",
            "injury_preventions",
        ]
        display_cols = ["frame", "time_sec", "status"] + [
            c for c in numeric_cols if c in display_df.columns
        ] + [c for c in injury_cols if c in display_df.columns]
        display = display_df[display_cols].copy()
        for col in numeric_cols:
            if col in display.columns:
                display[col] = display[col].round(3)

        def _style_status(value: str) -> str:
            if value == "risky":
                return "background-color: #ffcccc"
            if value == "caution":
                return "background-color: #ffffcc"
            return "background-color: #ccffcc"

        styled = display.style.map(_style_status, subset=["status"])
        st.dataframe(styled, use_container_width=True)
    else:
        st.info("No rows to display.")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if video_bytes:
            st.download_button(
                label="Download annotated video (.mp4)",
                data=video_bytes,
                file_name="annotated.mp4",
                mime="video/mp4",
            )
        else:
            st.button(
                label="Download annotated video (.mp4)",
                disabled=True,
                help="No annotated video was produced.",
            )
    with col2:
        if analysis["csv_bytes"]:
            st.download_button(
                label="Download CSV report",
                data=analysis["csv_bytes"],
                file_name="risk_report.csv",
                mime="text/csv",
            )
        else:
            st.button(
                label="Download CSV report",
                disabled=True,
                help="No CSV report was produced.",
            )

"""Streamlit review dashboard for candidate violation events.

Reads the same SQLite store as the API. Run:
    streamlit run src/tvd/ui/dashboard.py -- --db results/events.db

All events are candidates for human review; nothing here acts on them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from tvd.evidence.db import EventStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="results/events.db")
    args = parser.parse_args()

    st.set_page_config(page_title="TVD Review", layout="wide")
    st.title("Traffic Violation Detection — human review")
    st.caption("Candidate events only. No automated penalties; reviewers decide.")

    store = EventStore(args.db)
    events = store.list_events()
    if not events:
        st.info("No events yet. Run scripts/run_pipeline.py on a video.")
        return

    violation_types = sorted({e["violation_type"] for e in events})
    selected = st.sidebar.multiselect("violation type", violation_types, default=violation_types)
    cameras = sorted({e["camera_id"] for e in events})
    selected_cameras = st.sidebar.multiselect("camera", cameras, default=cameras)
    filtered = [
        e for e in events if e["violation_type"] in selected and e["camera_id"] in selected_cameras
    ]

    st.metric("candidate events", len(filtered))
    df = pd.DataFrame(filtered)[
        ["event_id", "violation_type", "track_id", "camera_id", "start_frame", "end_frame"]
    ]
    chosen = st.selectbox("event", df["event_id"].tolist(), index=0)
    st.dataframe(df, width="stretch")

    if chosen:
        event = store.get_event(chosen)
        col_clip, col_meta = st.columns(2)
        with col_clip:
            clip = Path(event["clip_path"])
            if clip.exists():
                st.video(str(clip), alt=f"Evidence clip for event {chosen}")
            frame = Path(event["frame_path"])
            if frame.exists():
                st.image(
                    str(frame),
                    caption="keyframe (blurred)",
                    alt=f"Blurred keyframe for event {chosen}",
                )
        with col_meta:
            meta = Path(event["metadata_path"])
            if meta.exists():
                st.json(json.loads(meta.read_text()))


if __name__ == "__main__":
    main()

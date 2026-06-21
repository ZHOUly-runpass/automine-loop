from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import streamlit as st

from automine_loop.io import read_table
from automine_loop.retrieval.index import FlatIPIndex


st.set_page_config(page_title="AutoMine-Loop", layout="wide")
st.title("AutoMine-Loop")


@st.cache_data
def load_tables():
    manifest = read_table(ROOT / "data" / "manifests" / "data_manifest.parquet")
    quality = read_table(ROOT / "data" / "manifests" / "quality_features.parquet")
    clusters = read_table(ROOT / "data" / "manifests" / "cluster_result.parquet")
    selected = read_table(ROOT / "data" / "manifests" / "selected_samples.parquet")
    return manifest, quality, clusters, selected


manifest, quality, clusters, selected = load_tables()
page = st.sidebar.radio("View", ["Overview", "Image Search", "Clusters", "Active Sampling"])

if page == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Frames", len(manifest))
    c2.metric("Scenes", manifest["scene_token"].nunique())
    c3.metric("Clusters", clusters["cluster_id"].nunique())
    c4.metric("Selected", len(selected))
    st.bar_chart(manifest["split"].value_counts())
    merged = manifest.merge(quality, on="record_id", how="left")
    st.dataframe(merged.head(100), use_container_width=True)

elif page == "Image Search":
    index = FlatIPIndex.load(ROOT / "data" / "indexes" / "frame_index.npz")
    record_id = st.number_input("Query record_id", min_value=int(manifest["record_id"].min()), max_value=int(manifest["record_id"].max()), value=int(manifest["record_id"].iloc[0]))
    top_k = st.slider("Top K", 5, 50, 12)
    result = index.search_by_record_id(int(record_id), top_k=top_k).merge(manifest, on="record_id", how="left")
    cols = st.columns(4)
    for i, row in result.iterrows():
        with cols[i % 4]:
            st.image(str(ROOT / row["image_path"]), caption=f"#{row.record_id} score={row.score:.3f}", use_container_width=True)
    st.dataframe(result, use_container_width=True)

elif page == "Clusters":
    cluster_id = st.selectbox("Cluster", sorted(clusters["cluster_id"].unique()))
    rows = clusters[clusters["cluster_id"] == cluster_id].merge(manifest, on="record_id", how="left").sort_values("distance_to_center")
    st.write(f"{len(rows)} frames")
    cols = st.columns(5)
    for i, row in rows.head(30).iterrows():
        with cols[i % 5]:
            st.image(str(ROOT / row["image_path"]), caption=f"#{row.record_id}", use_container_width=True)
    st.dataframe(rows, use_container_width=True)

else:
    rows = selected.merge(manifest, on="record_id", how="left").sort_values(["round", "score"], ascending=[True, False])
    round_id = st.selectbox("Round", sorted(rows["round"].unique()))
    view = rows[rows["round"] == round_id]
    cols = st.columns(4)
    for i, row in view.iterrows():
        with cols[i % 4]:
            st.image(str(ROOT / row["image_path"]), caption=f"#{row.record_id} {row.score:.3f}", use_container_width=True)
    st.dataframe(view, use_container_width=True)

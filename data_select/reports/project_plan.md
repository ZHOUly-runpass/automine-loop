# AutoMine-Loop Implementation Plan

## MVP

1. Generate or parse a frame manifest.
2. Build hidden oracle labels.
3. Compute quality features and rule scores.
4. Extract frame embeddings.
5. Build flat inner-product retrieval.
6. Run PCA + K-Means or HDBSCAN clustering.
7. Train a multi-label linear probe.
8. Run active sampling and release oracle labels.
9. Display search, clusters, and sampling candidates in Streamlit.

## Next Extensions

- Replace the fallback encoder with DINOv2 and CLIP.
- Add ROI-level crops from a detector or existing perception outputs.
- Add FAISS when the dataset exceeds the simple NumPy index scale.
- Add five-round, three-seed experiments on nuScenes mini and then train/val.
- Add PySpark jobs for metadata aggregation after single-machine behavior is stable.

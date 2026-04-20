# Manufacturing Feasibility Analytics Suite

This project is designed for a manufacturing-feasibility assignment with 3D point clouds.

## Architecture

### Offline
Use `offline_feature_extractor.py` on your local machine against the raw dataset.

This is the right approach when the raw point-cloud collection is very large, such as 9 GB or more.

Example:

```bash
python offline_feature_extractor.py --root_dir /path/to/dataset --output features.parquet --workers 4
```

Expected folder layout:

```text
dataset/
├── feasible/
│   ├── part1.ply
│   ├── part2.ply
│   └── ...
└── infeasible/
    ├── partA.ply
    ├── partB.ply
    └── ...
```

The extractor computes:
- global geometry and bounding-box features
- PCA eigenvalue shape descriptors
- nearest-neighbor spacing features
- convex-hull and compactness features
- multiscale occupancy features
- radial and pairwise-distance histograms
- projection histograms
- local geometric summaries from neighborhood PCA

Output can be:
- CSV
- XLSX
- Parquet

### Online / Streamlit
Upload the extracted feature table to the Streamlit app.

Run locally:

```bash
pip install -r requirements.txt
streamlit run Home.py
```

## Streamlit pages

- **Q1 Visualization**
  Dataset summary, PCA / t-SNE, clustering, missingness, ranked features.

- **Q2 Smart Data Selection**
  Compare random, stratified, balanced, diversity, uncertainty, and hybrid subset strategies.

- **Q3 Feature Engineering**
  Review feature families, add unsupervised augmentations, export augmented tables.

- **Q4 Pipelines and Diagnostics**
  Generate many pipelines, benchmark them, inspect confusion matrices and misclassified samples.

## Deployment

Push the folder to GitHub and deploy `Home.py` as the entrypoint.

For Streamlit Community Cloud, keep:
- `Home.py`
- `pages/`
- `requirements.txt`

in the repository.

## Notes

- This app is intentionally designed around **precomputed features** so the online service stays light.
- Raw point-cloud feature extraction remains fully local.
- If needed, you can also keep a small demo feature table in the repo for testing.

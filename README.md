# Expected data layout

This repo does **not** ship data. Place your own CSVs following the structure
below (or pass custom paths via CLI flags — nothing is hardcoded).

```
data/
├── train/
│   ├── clinical_features.csv
│   ├── tumor_level_features.csv
│   ├── breast_ratio_enhancement_features.csv   # combined modality 3 (see below)
│   └── pcr_target.csv
└── external/                      # optional, for held-out validation
    ├── clinical_features.csv
    ├── tumor_level_features.csv
    ├── breast_ratio_enhancement_features.csv
    └── pcr_target.csv
```

All of the above (except `pcr_target.csv`) are produced automatically by
`src/radiomics_extraction.py` if you're starting from raw MRI + segmentation
masks — see the main `README.md` Step 0 section.

## Row alignment
All CSVs for a given cohort **must have the same number of rows, in the same
subject order**. There is no subject-ID join performed by the code — row `i`
in every file is assumed to be the same patient.

## Column requirements

| File | Content | Notes |
|---|---|---|
| `clinical_features.csv` | Patient-level clinical variables (age, molecular subtype, receptor status, etc.) | Categorical columns are auto-detected (non-numeric dtype) and label-encoded. Keep column names stable between train and external cohorts. |
| `tumor_level_features.csv` | Tumor-region radiomic / imaging features extracted from the segmented tumor mask | Typically the largest modality (dozens–hundreds of columns); MI filtering + PCA is applied automatically. |
| `breast_ratio_enhancement_features.csv` | Breast-level enhancement features (post/sub/rel/ratio stats for the tumor-side breast and background breast) merged with tumor-to-breast ratio features | Produced by `combine_enhancement_and_ratio_features()` in `radiomics_extraction.py` — do not substitute the separate `breast_level_enhancement_features.csv` or `tumor_breast_ratio_features.csv` QC files here, they're missing information the other one has. Medium-sized modality; MI filtering is applied automatically. |
| `pcr_target.csv` | One column with the classification label (default column name: `pcr`) | Any binary or multi-class label works; column name is configurable via `--label_column`. |

## New (external) datasets
When running `extract_features.py` or `predict.py` on a new dataset, its CSVs
only need to contain **compatible column names** — extra columns are dropped,
missing ones are filled with 0.0 (a warning is printed so you can check
whether the fill actually happened for meaningful features). The exact column
set, scaler statistics, and feature selection are all taken from the training
checkpoint, not recomputed — this is what makes cross-cohort predictions
reproducible.

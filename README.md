# Multi-Modal GatedFusion_MultiEncoder for pCR Classification from Breast MRI

A multi-modal deep learning pipeline that predicts pathological complete
response (pCR) from breast MRI. It fuses three modalities — clinical
variables, tumor-level radiomics, and breast-level enhancement/ratio
features — through a TabNet-style sparse-attention encoder per modality and
a learned gated fusion layer (`GatedFusion_MultiEncoder`).

The full pipeline has three stages:

1. **Segmentation** — tumor and whole-breast masks (external tools, linked below)
2. **Feature extraction** — radiomics + clinical features computed from the
   MRI and the segmentation masks (`src/radiomics_extraction.py`, included)
3. **Classification** — the trained `GatedFusion_MultiEncoder` model predicts
   pCR from the extracted features (`src/predict.py`, included)

---

## Pipeline overview

```
  Breast MRI (pre-contrast + post-contrast)
        │
        ▼
  ┌───────────────────────────────────────────────────┐
  │ STEP 1 — Tumor + breast segmentation                │
  │ (external tools — not part of this repo)            │
  │   • BreastDivider — github.com/MIC-DKFZ/BreastDivider│
  │   • MAMA-MIA — github.com/LidiaGarrucho/MAMA-MIA     │
  └───────────────────────────────────────────────────┘
        │  produces: tumor mask + whole-breast mask (NIfTI)
        ▼
  ┌───────────────────────────────────────────────────┐
  │ STEP 2 — Feature extraction                          │   src/radiomics_extraction.py
  │ MRI + masks -> tumor-level radiomics, breast-level    │
  │ enhancement/ratio features, merged with clinical data │
  └───────────────────────────────────────────────────┘
        │  produces 3 CSVs: clinical / tumor-level / breast-level (enhancement+ratio combined)
        ▼
  ┌───────────────────────────────────────────────────┐
  │ STEP 3 — Classification                               │   src/predict.py
  │ Trained GatedFusion_MultiEncoder model                │
  │ (single checkpoint or 5-fold ensemble)                │
  └───────────────────────────────────────────────────┘
        │
        ▼
  predictions.csv  (predicted pCR class + per-class probability)
```

If you want to retrain or reproduce the reported cross-validation / external
validation results instead of just running inference, use `src/train.py`
(see [Usage](#usage) below).

---

## Step 1 — Tumor + breast segmentation (external, not included here)

This repo starts from **already-segmented** tumor and whole-breast masks.
Segmentation itself is not implemented here — use one of these existing,
validated tools to produce the masks first:

- **[BreastDivider](https://github.com/MIC-DKFZ/BreastDivider)** — segments
  the whole breast (and chest wall / background) on breast MRI.
- **[MAMA-MIA](https://github.com/LidiaGarrucho/MAMA-MIA)** — a large public
  breast MRI dataset with expert and automated tumor segmentations, plus
  tooling/baselines for tumor segmentation on DCE-MRI.

Run either (or both — e.g. BreastDivider for the whole-breast mask and a
MAMA-MIA-trained model for the tumor mask) on your MRI cohort to produce, per
patient:
- a **tumor mask**: `TUMOR_MASK_DIR/<patient_id>.nii.gz`
- a **whole-breast mask**: `<patient_dir>/new_breast_mask.nii.gz` (or
  `breast_mask.nii.gz` as a fallback name)

Follow each tool's own installation/usage instructions — refer to their
README files linked above. Once you have both masks per patient, move to
Step 2.

---

## Step 2 — Feature extraction (included: `src/radiomics_extraction.py`)

Given the pre-contrast MRI, post-contrast MRI, tumor mask, and whole-breast
mask for each patient, this script computes all three input modalities used
by the model:

| Modality | Output file | Contents |
|---|---|---|
| 0 — Clinical | `clinical_features.csv` | Patient IDs merged with your own clinical CSV (age, molecular subtype, receptor status, etc.) if you provide one via `CLINICAL_CSV`; otherwise just patient IDs. |
| 1 — Tumor-level | `tumor_level_features.csv` | PyRadiomics features (post-contrast intensity + relative-enhancement texture/shape) computed on the tumor mask, plus tumor enhancement-curve statistics. |
| 2 — Breast-level | `breast_level_enhancement_features.csv` | Breast-side and background-breast enhancement statistics **merged with** tumor-to-breast and tumor-to-background ratio features — computed together per patient inside `build_combined_breast_features()`, so this single file already contains both. |

### Expected input layout

```
PATIENT_ROOT/
    DUKE_001/
        <case>_0000.nii.gz          # pre-contrast MRI
        <case>_0001.nii.gz          # first post-contrast MRI
        new_breast_mask.nii.gz      # preferred whole-breast mask (from Step 1)
        breast_mask.nii.gz          # fallback whole-breast mask name

TUMOR_MASK_DIR/
    duke_001.nii.gz                 # tumor mask (from Step 1), matched to
    duke_002.nii.gz                 # PATIENT_ROOT subfolders by normalized ID
```

### Running it

Install the extra imaging dependencies (not in `requirements.txt`, since
this step is optional if you already have feature CSVs):

```bash
pip install SimpleITK pyradiomics pandas numpy tqdm
```

Edit the `CONFIG` block at the top of `src/radiomics_extraction.py`:

```python
PATIENT_ROOT = Path(r"/path/to/your/mri_dataset/images")
TUMOR_MASK_DIR = Path(r"/path/to/your/mri_dataset/segmentations")
OUTPUT_DIR = Path(r"features_output")
CLINICAL_CSV = None  # or Path(r"/path/to/clinical.csv")
PATIENT_FILTER = "duke"  # substring filter on patient folder names
```

Then run:

```bash
python src/radiomics_extraction.py
```

This writes to `OUTPUT_DIR`:
- `clinical_features.csv`, `tumor_level_features.csv`, `breast_level_enhancement_features.csv` — the 3 modality inputs for Step 3
- `qc_selected_features.csv` — per-patient QC info (voxel counts, mask volumes, failure reasons for any skipped patients)
- `dataset1_manifest_found.csv` — which patient folders were discovered and whether all required files were found

Run this once for your training cohort and, separately, for any new cohort
you want to predict on (pointing `PATIENT_ROOT`/`TUMOR_MASK_DIR`/`OUTPUT_DIR`
at the new cohort's data).

---

## Step 3 — Classification (included: `src/predict.py`)

Feed the 3 CSVs produced in Step 2 into the trained `GatedFusion_MultiEncoder`
model to get pCR predictions. See [Usage](#usage) below for exact commands.

---

## Repository structure

```
.
├── README.md
├── requirements.txt
├── data/
│   └── README.md                   # expected CSV schema
└── src/
    ├── radiomics_extraction.py      # Step 2: MRI + masks -> clinical/tumor/breast-level CSVs
    ├── model.py                     # Sparsemax, TabNetEncoder, GatedFusion, GatedFusion_MultiEncoder
    ├── preprocessing.py              # load_data, per-modality feature pipelines
    ├── metrics.py                    # Balanced Acc / F1 / Sensitivity / Specificity / AUC
    ├── utils.py                      # seeding, Dataset, collate_fn
    ├── train.py                      # 5-fold CV + Optuna HP search + external val
    ├── extract_features.py           # reuse a checkpoint's fitted pipelines on new data (QC)
    └── predict.py                     # Step 3: inference — single checkpoint or 5-fold ensemble
```

---

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

Tested with Python 3.10+, PyTorch 2.x (CPU or CUDA). For Step 2 (feature
extraction), also install `SimpleITK`, `pyradiomics`, and `tqdm` (see above).

---

## Getting the trained weights

Model checkpoints (`.pt` files) are **not committed to this repo** (see
`.gitignore`) since they're large binary files. Attach them to a
[GitHub Release](https://docs.github.com/en/repositories/releasing-projects-on-github)
or host them via Git LFS / Zenodo, and link them here, e.g.:

> Download `best_model.pt` (and/or the 5 per-fold checkpoints) from the
> [Releases page](../../releases) and place them under `results/models/`.

---

## Usage

### Train (reproduce CV + external validation, or retrain on your own cohort)

```bash
python -m src.train \
  --clinical_csv data/train/clinical_features.csv \
  --tumor_csv data/train/tumor_level_features.csv \
  --ratio_csv data/train/breast_level_enhancement_features.csv \
  --target_csv data/train/pcr_target.csv \
  --ext_clinical_csv data/external/clinical_features.csv \
  --ext_tumor_csv data/external/tumor_level_features.csv \
  --ext_ratio_csv data/external/breast_level_enhancement_features.csv \
  --ext_target_csv data/external/pcr_target.csv \
  --out_dir results \
  --n_trials 30
```

(`--ratio_csv` takes the combined breast-level file from Step 2 — its name is
kept as `--ratio_csv` for backward compatibility, but it already contains
both the enhancement and ratio features.)

Outputs (under `results/`):
- `models/fold{1..5}_model.pt` — one checkpoint per fold, each containing
  model weights **and** that fold's fitted preprocessing pipelines (required
  for correct reuse on new data — see note below).
- `models/best_model.pt` — the fold with the best CV balanced accuracy.
- `cm_fold*_cv.png`, `cm_fold*_external.png`, `cm_external_ensemble.png`
- `summary_metrics.csv` — mean ± std of all metrics, CV and external.

Drop the `--ext_*` flags entirely if you have no external validation cohort.

### Predict on new data (Step 3)

Run Step 1 (segmentation) and Step 2 (`radiomics_extraction.py`) on your new
cohort first, then:

Single model:
```bash
python -m src.predict \
  --checkpoint results/models/best_model.pt \
  --clinical_csv new_cohort/clinical_features.csv \
  --tumor_csv new_cohort/tumor_level_features.csv \
  --ratio_csv new_cohort/breast_level_enhancement_features.csv \
  --out_csv predictions.csv
```

5-fold ensemble (majority vote + averaged probability, matching how external
validation was reported during training):
```bash
python -m src.predict \
  --checkpoint_dir results/models \
  --clinical_csv new_cohort/clinical_features.csv \
  --tumor_csv new_cohort/tumor_level_features.csv \
  --ratio_csv new_cohort/breast_level_enhancement_features.csv \
  --out_csv predictions_ensemble.csv
```

### Extract model-ready features for QC (optional)

```bash
python -m src.extract_features \
  --checkpoint results/models/best_model.pt \
  --clinical_csv new_cohort/clinical_features.csv \
  --tumor_csv new_cohort/tumor_level_features.csv \
  --ratio_csv new_cohort/breast_level_enhancement_features.csv \
  --out_dir extracted_features/
```

This applies the exact scaler/feature-selection/PCA fitted during training —
not a freshly-refit version — so the transformed features are directly
comparable to what the model saw in training. Useful for sanity-checking a
new cohort before running full inference.

---

## Important implementation notes

- **Checkpoints are self-contained.** Each `.pt` file saves the fitted
  `RobustScaler` / `SelectKBest` / `PCA` pipeline per modality, the training
  column names (for aligning new-data columns), and the categorical
  `LabelEncoder`s — not just model weights. This is required for correct
  reuse; a model checkpoint without these would silently produce wrong
  predictions on new data (features would be in a different, unscaled /
  unselected space than what the model was trained on).
- **Breast-level features are combined once, at extraction time.** Breast
  enhancement statistics and tumor-breast ratio features come from the same
  masks and the same patients, so `radiomics_extraction.py` merges them into
  a single feature dict per patient (`build_combined_breast_features()`)
  rather than writing two separate files and joining them later. A couple of
  volume columns are computed identically by both feature sets; the
  duplicate is dropped rather than silently overwritten.
- **Binary-classification metrics.** `Sensitivity`/`Specificity` are computed
  relative to the positive class for binary problems (not macro-averaged
  across both classes). A macro-averaged specificity is mathematically
  identical to Balanced Accuracy for binary problems, which would make three
  of five reported metrics redundant — this is fixed in `metrics.py`.
- **No data leakage.** Feature scaling/selection, SMOTE oversampling, and
  hyperparameter tuning are all fit on the training fold only and applied
  (not refit) to the held-out test fold / external cohort.
- **External validation gap.** If external performance is notably lower than
  CV performance, this is typically domain shift (scanner/protocol
  differences, segmentation quality, missing columns being zero-filled,
  etc.) rather than a code issue — check the `WARNING Modality ... cols
  missing` output when loading a new cohort, and confirm segmentation
  quality on a few cases visually before trusting the extracted features.

---

## Citation

If you use this code, please cite: *(add your paper/preprint reference here)*

Also cite the segmentation tools you use:
- BreastDivider: https://github.com/MIC-DKFZ/BreastDivider
- MAMA-MIA: https://github.com/LidiaGarrucho/MAMA-MIA

## License

*(add a LICENSE file — e.g. MIT or Apache-2.0 — before making the repo public)*

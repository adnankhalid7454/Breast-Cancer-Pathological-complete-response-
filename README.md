# Multi-encoder Gated Fusion pCR Classification from Breast MRI

A multi-modal deep learning pipeline that predicts pathological complete
response (pCR) from breast MRI. It fuses three modalities — clinical
variables, tumor-level radiomics, and breast-level enhancement/ratio
features — through a TabNet-style sparse-attention encoder per modality and
a learned gated fusion layer.

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
        │  produces 3 CSVs: clinical / tumor-level / breast-level
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
---

## Step 1 — Tumor + breast segmentation

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
| 2 — Breast-level | `breast_level_enhancement_features.csv` | Breast-side and background-breast enhancement statistics & tumor-to-breast and tumor-to-background ratio features — computed together per patient inside `build_combined_breast_features()` |

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

---

## Step 3 — Classification (included: `src/predict.py`)

Feed the 3 CSVs produced in Step 2 into the trained `GatedFusion_MultiEncoder`
model to get pCR predictions. 

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

Model checkpoints (see
> Download `best_model.pt` from the   https://doi.org/10.5281/zenodo.21238902


---



"""
Reuse a trained model checkpoint's FITTED preprocessing pipelines
(RobustScaler / SelectKBest / PCA, per modality) to transform a NEW
dataset's raw feature CSVs into the exact feature space the model
expects — without running classification.

This is useful for:
  - QC / inspecting what the model will actually see for a new cohort
  - Feeding the transformed features into other tools
  - Debugging column-alignment / missing-feature warnings before inference

Input CSVs must contain the same columns (or a superset/subset — extras are
dropped, missing ones are filled with 0.0 and a warning is printed) as the
training cohort's clinical / tumor-level / tumor-breast-ratio CSVs.

Usage:
    python -m src.extract_features \
        --checkpoint results/models/best_model.pt \
        --clinical_csv new_data/clinical_data.csv \
        --tumor_csv new_data/tumor_level_features.csv \
        --ratio_csv new_data/tumor_breast_ratio_features.csv \
        --out_dir extracted_features/
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch

from src.preprocessing import load_data, transform_pipeline


def main(args):
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    train_col_names = ckpt['train_col_names']
    cat_encoders = ckpt['cat_encoders']
    fold_pipelines = ckpt['fold_pipelines']

    raw_paths = [args.clinical_csv, args.tumor_csv, args.ratio_csv]
    raw_features, _, _, _, _, _ = load_data(
        raw_paths, target_file=None,
        common_columns=train_col_names, cat_encoders=cat_encoders
    )

    os.makedirs(args.out_dir, exist_ok=True)
    names = ['clinical', 'tumor_level', 'tumor_breast_ratio']
    for name, raw, pipe in zip(names, raw_features, fold_pipelines):
        transformed = transform_pipeline(pipe, raw)
        out_path = os.path.join(args.out_dir, f"{name}_transformed.csv")
        col_names = [f"{name}_feat_{i}" for i in range(transformed.shape[1])]
        pd.DataFrame(transformed, columns=col_names).to_csv(out_path, index=False)
        print(f"  Saved: {out_path}  shape={transformed.shape}")

    print(f"\nDone. Checkpoint used: {args.checkpoint} (fold {ckpt.get('fold')})")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Extract model-ready features for a new dataset using a saved checkpoint's fitted pipelines.")
    p.add_argument('--checkpoint', required=True, help='Path to a .pt file saved by train.py (must contain fold_pipelines)')
    p.add_argument('--clinical_csv', required=True)
    p.add_argument('--tumor_csv', required=True)
    p.add_argument('--ratio_csv', required=True)
    p.add_argument('--out_dir', default='extracted_features')
    args = p.parse_args()
    main(args)

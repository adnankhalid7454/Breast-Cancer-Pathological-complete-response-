"""
Run inference on a new dataset with a trained checkpoint (or an ensemble
of per-fold checkpoints, majority-vote + averaged probability, matching
the ensembling strategy used during training/external validation).

Input CSVs must be the SAME modalities used in training (clinical /
tumor-level / tumor-breast-ratio), already produced by your own upstream
segmentation + feature-extraction pipeline (see README — not part of this repo).

Usage (single best model):
    python -m src.predict \
        --checkpoint results/models/best_model.pt \
        --clinical_csv new_data/clinical_data.csv \
        --tumor_csv new_data/tumor_level_features.csv \
        --ratio_csv new_data/tumor_breast_ratio_features.csv \
        --out_csv predictions.csv

Usage (5-fold ensemble):
    python -m src.predict \
        --checkpoint_dir results/models \
        --clinical_csv new_data/clinical_data.csv \
        --tumor_csv new_data/tumor_level_features.csv \
        --ratio_csv new_data/tumor_breast_ratio_features.csv \
        --out_csv predictions_ensemble.csv
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats as scipy_stats

from src.model import TabNet
from src.preprocessing import load_data, transform_pipeline


def load_checkpoint(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model = TabNet(ckpt['feature_dims'], ckpt['num_classes'],
                    n_steps=ckpt['best_hp']['n_steps'],
                    hidden_dim=ckpt['best_hp']['hidden_dim'],
                    gamma=ckpt['best_hp']['gamma'],
                    dropout=ckpt['best_hp']['dropout'])
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, ckpt


def predict_with_checkpoint(ckpt_path, raw_paths):
    model, ckpt = load_checkpoint(ckpt_path)
    train_col_names = ckpt['train_col_names']
    cat_encoders = ckpt['cat_encoders']
    fold_pipelines = ckpt['fold_pipelines']

    raw_features, _, _, _, _, _ = load_data(
        raw_paths, target_file=None,
        common_columns=train_col_names, cat_encoders=cat_encoders
    )
    processed = [transform_pipeline(p, f) for p, f in zip(fold_pipelines, raw_features)]

    tensors = [torch.tensor(f, dtype=torch.float32) for f in processed]
    with torch.no_grad():
        outputs = model(tensors)
        probs = F.softmax(outputs, dim=1).numpy()
        preds = probs.argmax(axis=1)

    return preds, probs, ckpt.get('label_encoder_classes')


def main(args):
    raw_paths = [args.clinical_csv, args.tumor_csv, args.ratio_csv]

    if args.checkpoint_dir:
        ckpt_paths = sorted(glob.glob(os.path.join(args.checkpoint_dir, 'fold*_model.pt')))
        if not ckpt_paths:
            raise FileNotFoundError(f"No fold*_model.pt checkpoints found in {args.checkpoint_dir}")
        print(f"Ensembling {len(ckpt_paths)} fold checkpoints: {ckpt_paths}")

        all_preds, all_probs, class_names = [], [], None
        for ckpt_path in ckpt_paths:
            preds, probs, class_names = predict_with_checkpoint(ckpt_path, raw_paths)
            all_preds.append(preds)
            all_probs.append(probs)

        preds_stack = np.stack(all_preds, axis=0)
        probs_stack = np.stack(all_probs, axis=0)
        final_preds, _ = scipy_stats.mode(preds_stack, axis=0, keepdims=False)
        final_preds = final_preds.flatten()
        final_probs = probs_stack.mean(axis=0)
    else:
        if not args.checkpoint:
            raise ValueError("Provide either --checkpoint or --checkpoint_dir")
        final_preds, final_probs, class_names = predict_with_checkpoint(args.checkpoint, raw_paths)

    n_classes = final_probs.shape[1]
    class_names = class_names or [str(i) for i in range(n_classes)]

    out = pd.DataFrame({
        'predicted_class_idx': final_preds,
        'predicted_class_label': [class_names[i] for i in final_preds],
    })
    for i, cname in enumerate(class_names):
        out[f'prob_{cname}'] = final_probs[:, i]

    out.to_csv(args.out_csv, index=False)
    print(f"\nSaved predictions: {args.out_csv}")
    print(out.head())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Run inference with a trained checkpoint (single model or fold ensemble).")
    p.add_argument('--checkpoint', default=None, help='Path to a single .pt checkpoint')
    p.add_argument('--checkpoint_dir', default=None, help='Directory containing fold1_model.pt ... foldN_model.pt for ensembling')
    p.add_argument('--clinical_csv', required=True)
    p.add_argument('--tumor_csv', required=True)
    p.add_argument('--ratio_csv', required=True)
    p.add_argument('--out_csv', default='predictions.csv')
    args = p.parse_args()
    main(args)

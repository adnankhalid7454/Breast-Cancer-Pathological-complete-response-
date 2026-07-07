"""
Train the multi-modal GatedFusion_MultiEncoder model with 5-fold stratified CV,
per-fold Optuna hyperparameter tuning, and (optional) external validation.

CRITICAL: each fold's checkpoint saves not just the model weights but also
the FITTED preprocessing pipelines (RobustScaler / SelectKBest / PCA) and
column alignment info. Without these, the exact feature space the model was
trained on cannot be reproduced on new data — inference would silently be
wrong. See extract_features.py / predict.py, which load these checkpoints.

Usage:
    python -m src.train --config path/to/config.yaml
    (or edit the CONFIG block below directly)
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from imblearn.over_sampling import SMOTE
from scipy import stats as scipy_stats
import optuna

from src.utils import set_seed, seed_worker, collate_fn, TabularDataset
from src.model import GatedFusion_MultiEncoder
from src.preprocessing import build_feature_pipeline, fit_transform_pipeline, transform_pipeline, load_data
from src.metrics import compute_metrics, print_metrics, plot_cm

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

SEED = 42


def get_weighted_criterion(y_train, num_classes):
    counts = np.bincount(y_train, minlength=num_classes).astype(float)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32),
        label_smoothing=0.05
    )


def run_training(resampled_features, y_resampled, val_features, y_val,
                  feature_dims, num_classes, hp, fold_seed,
                  n_epochs=300, patience=50):
    set_seed(fold_seed)

    train_dataset = TabularDataset(resampled_features, y_resampled)
    val_dataset = TabularDataset(val_features, y_val)

    g = torch.Generator(); g.manual_seed(fold_seed)
    train_loader = DataLoader(train_dataset, batch_size=hp['batch_size'], shuffle=True,
                               collate_fn=collate_fn,
                               worker_init_fn=lambda wid: seed_worker(wid, fold_seed),
                               generator=g)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    model = GatedFusion_MultiEncoder(feature_dims, num_classes, n_steps=hp['n_steps'],
                    hidden_dim=hp['hidden_dim'], gamma=hp['gamma'], dropout=hp['dropout'])
    criterion = get_weighted_criterion(y_resampled, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=hp['T_0'], T_mult=2, eta_min=1e-6)

    best_bal_acc, no_improve, best_state, best_epoch = 0.0, 0, None, 0

    for epoch in range(n_epochs):
        model.train()
        for batch_inputs, batch_labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_inputs), batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch_inputs, batch_labels in val_loader:
                _, preds = torch.max(model(batch_inputs), 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(batch_labels.cpu().numpy())

        bal_acc = balanced_accuracy_score(val_labels, val_preds)
        if bal_acc > best_bal_acc:
            best_bal_acc, best_epoch = bal_acc, epoch + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_bal_acc, best_epoch


def tune_hyperparams(resampled_features, y_resampled, val_features, y_val,
                      feature_dims, num_classes, fold_seed, n_trials=30):
    def objective(trial):
        hp = {
            'lr': trial.suggest_float('lr', 5e-4, 5e-3, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-4, 5e-3, log=True),
            'dropout': trial.suggest_float('dropout', 0.2, 0.5),
            'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128]),
            'n_steps': trial.suggest_int('n_steps', 2, 4),
            'gamma': trial.suggest_float('gamma', 1.0, 2.0),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'T_0': trial.suggest_categorical('T_0', [30, 50, 75]),
        }
        _, bal_acc, _ = run_training(resampled_features, y_resampled, val_features, y_val,
                                      feature_dims, num_classes, hp, fold_seed,
                                      n_epochs=100, patience=20)
        return bal_acc

    study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=fold_seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"    Best trial: bal_acc={study.best_value:.4f}  HP={study.best_params}")
    return study.best_params


def predict_probs(model, features, y):
    model.eval()
    dataset = TabularDataset(features, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for batch_inputs, batch_labels in loader:
            outputs = model(batch_inputs)
            p = F.softmax(outputs, dim=1)
            _, pr = torch.max(outputs, 1)
            preds.extend(pr.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
            probs.append(p.cpu().numpy())
    return np.array(preds), np.array(labels), np.concatenate(probs)


def main(args):
    set_seed(SEED)
    os.makedirs(args.out_dir, exist_ok=True)
    models_dir = os.path.join(args.out_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 60)
    print("  Loading TRAINING dataset...")
    print("=" * 60)
    train_paths = [args.clinical_csv, args.tumor_csv, args.ratio_csv]
    raw_features, y_all, num_classes, le, train_col_names, train_cat_encoders = load_data(
        train_paths, args.target_csv, label_column=args.label_column
    )

    ext_raw_features, y_ext = None, None
    if args.ext_clinical_csv:
        print("\n" + "=" * 60)
        print("  Loading EXTERNAL validation dataset (columns aligned to training)...")
        print("=" * 60)
        ext_paths = [args.ext_clinical_csv, args.ext_tumor_csv, args.ext_ratio_csv]
        ext_raw_features, y_ext, _, _, _, _ = load_data(
            ext_paths, args.ext_target_csv, label_column=args.label_column,
            common_columns=train_col_names, cat_encoders=train_cat_encoders
        )

    N_FOLDS = args.n_folds
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_metrics, fold_ext_metrics = [], []
    all_fold_preds, all_fold_labels, all_fold_probs = [], [], []
    all_ext_preds, all_ext_probs = [], []
    best_overall_bal_acc, best_overall_fold = 0.0, -1

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y_all)), y_all)):
        print(f"\n{'-'*60}\n  FOLD {fold+1}/{N_FOLDS}\n{'-'*60}")
        fold_seed = SEED + fold

        y_train, y_test = y_all[train_idx], y_all[test_idx]
        proc_train, proc_test, fold_pipelines = [], [], []
        for mod_idx, f in enumerate(raw_features):
            pipe = build_feature_pipeline(f.shape[1], mod_idx, seed=fold_seed)
            f_tr = fit_transform_pipeline(pipe, f[train_idx], y_train)
            f_te = transform_pipeline(pipe, f[test_idx])
            proc_train.append(f_tr)
            proc_test.append(f_te)
            fold_pipelines.append(pipe)
        print(f"  Selected features per modality: {[p.shape[1] for p in proc_train]}")

        X_combined = np.concatenate(proc_train, axis=1)
        k_neighbors = min(5, int(np.min(np.bincount(y_train))) - 1)
        smote = SMOTE(random_state=SEED, k_neighbors=max(1, k_neighbors))
        X_res, y_res = smote.fit_resample(X_combined, y_train)
        split_pts = np.cumsum([f.shape[1] for f in proc_train[:-1]])
        resampled_feats = np.split(X_res, split_pts, axis=1)
        feature_dims = [f.shape[1] for f in resampled_feats]
        print(f"  After SMOTE: {dict(pd.Series(y_res).value_counts())}")

        print(f"  Running Optuna ({args.n_trials} trials)...")
        best_hp = tune_hyperparams(resampled_feats, y_res, proc_test, y_test,
                                    feature_dims, num_classes, fold_seed, n_trials=args.n_trials)

        model, _, best_epoch = run_training(resampled_feats, y_res, proc_test, y_test,
                                             feature_dims, num_classes, best_hp, fold_seed,
                                             n_epochs=args.n_epochs, patience=args.patience)

        preds, labels, probs = predict_probs(model, proc_test, y_test)
        metrics, cm = compute_metrics(labels, preds, probs, num_classes)
        fold_metrics.append(metrics)
        all_fold_preds.append(preds); all_fold_labels.append(labels); all_fold_probs.append(probs)
        print_metrics(metrics, f"Fold {fold+1} CV Results (best epoch={best_epoch})")
        plot_cm(cm, f"Fold {fold+1} CV — BalAcc={metrics['Balanced Accuracy']:.3f}",
                os.path.join(args.out_dir, f"cm_fold{fold+1}_cv.png"), num_classes)

        # Save this fold's full checkpoint (weights + fitted pipelines + column info)
        fold_ckpt_path = os.path.join(models_dir, f"fold{fold+1}_model.pt")
        torch.save({
            'fold': fold + 1,
            'best_epoch': best_epoch,
            'model_state': model.state_dict(),
            'feature_dims': feature_dims,
            'num_classes': num_classes,
            'best_hp': best_hp,
            'bal_acc': metrics['Balanced Accuracy'],
            'fold_pipelines': fold_pipelines,        # <-- required for correct reuse
            'train_col_names': train_col_names,      # <-- required for column alignment
            'cat_encoders': train_cat_encoders,       # <-- required for categorical encoding
            'label_encoder_classes': list(le.classes_) if le is not None else None,
        }, fold_ckpt_path)
        print(f"  Saved fold checkpoint: {fold_ckpt_path}")

        if ext_raw_features is not None:
            ext_processed = [transform_pipeline(p, f) for p, f in zip(fold_pipelines, ext_raw_features)]
            ext_preds, y_ext_labels, ext_probs = predict_probs(model, ext_processed, y_ext)
            ext_metrics, ext_cm = compute_metrics(y_ext_labels, ext_preds, ext_probs, num_classes)
            fold_ext_metrics.append(ext_metrics)
            all_ext_preds.append(ext_preds); all_ext_probs.append(ext_probs)
            print_metrics(ext_metrics, f"Fold {fold+1} External Results")
            plot_cm(ext_cm, f"Fold {fold+1} External — BalAcc={ext_metrics['Balanced Accuracy']:.3f}",
                    os.path.join(args.out_dir, f"cm_fold{fold+1}_external.png"), num_classes)

        if metrics['Balanced Accuracy'] > best_overall_bal_acc:
            best_overall_bal_acc = metrics['Balanced Accuracy']
            best_overall_fold = fold + 1
            best_path = os.path.join(models_dir, 'best_model.pt')
            torch.save(torch.load(fold_ckpt_path), best_path)
            print(f"  * New best model -> {best_path} (Fold {fold+1}, BalAcc={best_overall_bal_acc:.4f})")

    # ── Summaries ─────────────────────────────────────────────
    print(f"\n{'='*60}\n  CV SUMMARY (mean +/- std)\n{'='*60}")
    metric_names = list(fold_metrics[0].keys())
    summary_rows = []
    for name in metric_names:
        vals = [m[name] for m in fold_metrics if isinstance(m[name], float)]
        if vals:
            mean_v, std_v = np.mean(vals), np.std(vals)
            print(f"  {name:<25}: {mean_v:.4f} +/- {std_v:.4f}")
            summary_rows.append({'Metric': name, 'Mean': round(mean_v, 4),
                                  'Std': round(std_v, 4), 'Dataset': 'CV'})

    if fold_ext_metrics:
        print(f"\n{'='*60}\n  EXTERNAL SUMMARY (mean +/- std)\n{'='*60}")
        for name in metric_names:
            vals = [m[name] for m in fold_ext_metrics if isinstance(m[name], float)]
            if vals:
                mean_v, std_v = np.mean(vals), np.std(vals)
                print(f"  {name:<25}: {mean_v:.4f} +/- {std_v:.4f}")
                summary_rows.append({'Metric': name, 'Mean': round(mean_v, 4),
                                      'Std': round(std_v, 4), 'Dataset': 'External'})

        ext_preds_stack = np.stack(all_ext_preds, axis=0)
        ext_probs_stack = np.stack(all_ext_probs, axis=0)
        ext_vote, _ = scipy_stats.mode(ext_preds_stack, axis=0, keepdims=False)
        ext_probs_avg = ext_probs_stack.mean(axis=0)
        agg_ext_metrics, agg_ext_cm = compute_metrics(y_ext, ext_vote.flatten(), ext_probs_avg, num_classes)
        print_metrics(agg_ext_metrics, f"External — Ensemble ({N_FOLDS}-fold majority vote)")
        plot_cm(agg_ext_cm, "External — Ensemble CM",
                os.path.join(args.out_dir, "cm_external_ensemble.png"), num_classes)

    pd.DataFrame(summary_rows).to_csv(os.path.join(args.out_dir, 'summary_metrics.csv'), index=False)
    print(f"\nBest model: Fold {best_overall_fold} (BalAcc={best_overall_bal_acc:.4f})")
    print(f"Saved to: {os.path.join(models_dir, 'best_model.pt')}")
    print(f"All fold checkpoints in: {models_dir}/  (use these for ensembled inference)")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Train the multi-modal GatedFusion_MultiEncoder classifier.")
    p.add_argument('--clinical_csv', required=True)
    p.add_argument('--tumor_csv', required=True)
    p.add_argument('--ratio_csv', required=True,
                    help='3rd modality CSV: use breast_level_enhancement_features.csv '
                         '(the combined enhancement+ratio table produced by '
                         'radiomics_extraction.py), not the two separate QC files.')
    p.add_argument('--target_csv', required=True)
    p.add_argument('--label_column', default='pcr')
    p.add_argument('--ext_clinical_csv', default=None)
    p.add_argument('--ext_tumor_csv', default=None)
    p.add_argument('--ext_ratio_csv', default=None,
                    help='External cohort equivalent of --ratio_csv (combined enhancement+ratio file).')
    p.add_argument('--ext_target_csv', default=None)
    p.add_argument('--out_dir', default='results')
    p.add_argument('--n_folds', type=int, default=5)
    p.add_argument('--n_trials', type=int, default=30, help='Optuna trials per fold')
    p.add_argument('--n_epochs', type=int, default=300)
    p.add_argument('--patience', type=int, default=50)
    args = p.parse_args()
    main(args)

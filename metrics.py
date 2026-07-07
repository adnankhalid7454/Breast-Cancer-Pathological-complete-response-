"""
Metrics computation.

NOTE on the binary-classification fix:
For binary problems, sklearn's recall_score(average='macro') is mathematically
identical to balanced_accuracy_score (both equal the mean of per-class recall).
A naive macro-averaged specificity loop over both classes collapses to the same
quantity too. 
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, recall_score,
    roc_auc_score, confusion_matrix
)


def compute_metrics(y_true, y_pred, y_prob, num_classes, pos_label=1):
    cm = confusion_matrix(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)

    if num_classes == 2:
        sensitivity = recall_score(y_true, y_pred, pos_label=pos_label,
                                    average='binary', zero_division=1)
        tn, fp, fn, tp = cm.ravel()
        if pos_label == 0:
            tn, fp, fn, tp = tp, fn, fp, tn
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        sensitivity = recall_score(y_true, y_pred, average='macro', zero_division=1)
        spec_list = []
        for i in range(num_classes):
            tn_i = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
            fp_i = cm[:, i].sum() - cm[i, i]
            spec_list.append(tn_i / (tn_i + fp_i) if (tn_i + fp_i) > 0 else 0.0)
        specificity = float(np.mean(spec_list))

    try:
        roc = (roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
               if num_classes > 2
               else roc_auc_score(y_true, y_prob[:, 1])
               if len(np.unique(y_true)) > 1 else None)
    except Exception as e:
        print(f"  ROC AUC error: {e}")
        roc = None

    return {"Balanced Accuracy": bal_acc,
            "F1-Score (weighted)": f1,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "ROC AUC": roc}, cm


def print_metrics(metrics, header=""):
    if header:
        print(f"\n{'='*55}\n  {header}\n{'='*55}")
    for name, val in metrics.items():
        print(f"  {name:<25}: {f'{val:.4f}' if isinstance(val, float) else str(val)}")
    print(f"{'='*55}")


def plot_cm(cm, title, filepath, num_classes):
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = [f'class={i}' for i in range(num_classes)]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  Saved: {filepath}")

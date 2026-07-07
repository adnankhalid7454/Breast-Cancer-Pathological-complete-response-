"""
Data loading and per-modality feature preprocessing.

Expects three tabular CSVs per cohort (clinical / tumor-level / tumor-breast-ratio),
produced upstream by your segmentation + feature-extraction pipeline (see README —
that upstream step is NOT part of this repo). Row order across the three CSVs and
the target CSV must correspond to the same subjects.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


def build_feature_pipeline(n_features, modality_idx, seed=42):
    """
    Returns an (unfitted) sklearn Pipeline that scales + selects features
    for a given modality.

    modality_idx: 0 = clinical (keep all, just scale)
                  1 = tumor-level (large; MI filter -> PCA)
                  2 = tumor-breast ratio (medium; MI filter only)
    """
    if modality_idx == 0:
        pipe = Pipeline([('scaler', RobustScaler())])
    elif modality_idx == 1:
        k_mi = min(30, n_features)
        pipe = Pipeline([
            ('scaler', RobustScaler()),
            ('mi_sel', SelectKBest(score_func=mutual_info_classif, k=k_mi)),
            ('pca', PCA(n_components=0.95, random_state=seed)),
        ])
    else:
        k_mi = min(15, n_features)
        pipe = Pipeline([
            ('scaler', RobustScaler()),
            ('mi_sel', SelectKBest(score_func=mutual_info_classif, k=k_mi)),
        ])
    return pipe


def fit_transform_pipeline(pipe, X_train, y_train):
    """Fit pipeline on training data, return transformed train features."""
    try:
        return pipe.fit_transform(X_train, y_train)
    except TypeError:
        return pipe.fit_transform(X_train)


def transform_pipeline(pipe, X):
    """Apply an already-fitted pipeline to new data."""
    return pipe.transform(X)


def load_data(file_paths, target_file=None, label_column='pcr',
              drop_columns=None, common_columns=None, cat_encoders=None):
    """
    Load one or more modality CSVs (+ optional target CSV).

    - drop_columns: columns to drop if present (e.g. subject ID columns).
    - common_columns: list of column-name-lists (one per modality), used to
      align a NEW dataset's columns to the training set's columns. Missing
      columns are filled with 0.0 and a warning is printed; extra columns
      are dropped.
    - cat_encoders: dict of {column_name: fitted LabelEncoder}, used to encode
      categorical columns consistently with training. Unseen categories fall
      back to the first known class.

    Returns: datasets (list of np arrays), y (np array or None if no target_file),
             num_classes, label_encoder (or None), col_names_per_modality, cat_encoders
    """
    datasets = []
    drop_columns = drop_columns or []
    col_names_per_modality = []
    cat_encoders = cat_encoders or {}

    for i, path in enumerate(file_paths):
        df = pd.read_csv(path)

        cols = [c for c in drop_columns if c in df.columns]
        if cols:
            df = df.drop(columns=cols)

        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        for col in cat_cols:
            if col in cat_encoders:
                le_cat = cat_encoders[col]
                known = set(le_cat.classes_)
                fallback = le_cat.classes_[0]
                df[col] = df[col].astype(str).apply(lambda v: v if v in known else fallback)
                df[col] = le_cat.transform(df[col].astype(str))
            else:
                le_cat = LabelEncoder()
                df[col] = le_cat.fit_transform(df[col].astype(str))
                cat_encoders[col] = le_cat
            print(f"  Encoded '{col}' -> int  ({path})")

        if common_columns is not None:
            train_cols = common_columns[i]
            missing = [c for c in train_cols if c not in df.columns]
            extra = [c for c in df.columns if c not in train_cols]
            if missing:
                print(f"  WARNING Modality {i+1}: {len(missing)} cols missing -> filled 0: {missing}")
                for c in missing:
                    df[c] = 0.0
            if extra:
                print(f"  WARNING Modality {i+1}: {len(extra)} extra cols dropped: {extra}")
            df = df[train_cols]

        col_names_per_modality.append(list(df.columns))
        datasets.append(df.values)
        print(f"  Modality {i+1}: {df.shape[1]} raw features  ({path})")

    if target_file is None:
        return datasets, None, None, None, col_names_per_modality, cat_encoders

    target_df = pd.read_csv(target_file)
    le = LabelEncoder()
    y = le.fit_transform(target_df[label_column])
    print(f"  Classes       : {list(le.classes_)}")
    print(f"  Total samples : {len(y)}")
    print(f"  Class dist    : {dict(pd.Series(y).value_counts())}")
    return datasets, y, len(le.classes_), le, col_names_per_modality, cat_encoders

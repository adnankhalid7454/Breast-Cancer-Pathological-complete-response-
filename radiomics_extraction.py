#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

You only need to edit the paths in the CONFIG section below, then run this file.

Input structure:
- PATIENT_ROOT/
    DUKE_001/
        *_0000.nii.gz              # pre-contrast MRI
        *_0001.nii.gz              # first post-contrast MRI
        new_breast_mask.nii.gz     # preferred breast mask
        breast_mask.nii.gz         # fallback breast mask

- TUMOR_MASK_DIR/
    duke_001.nii.gz
    duke_002.nii.gz

Outputs:
- tumor_level_features.csv
- breast_level_enhancement_features.csv          (kept for QC/debugging)
- tumor_breast_ratio_features.csv                (kept for QC/debugging)
- breast_ratio_enhancement_features.csv          <-- COMBINED modality 3 (use this one downstream)
- clinical_features.csv
- qc_selected_features.csv
- dataset1_manifest_found.csv

Note on the combined file: the breast-level enhancement features and the
tumor-breast ratio features are computed from the same masks/derived images
for the same patients, and are merged into a single "enhancement + ratio"
feature table (`breast_ratio_enhancement_features.csv`). This combined file
is what should be passed as the 3rd modality (alongside clinical and
tumor-level features) to the downstream training script — rather than
picking one of the two separate files, which discards information.

Install required packages:
    pip install SimpleITK pyradiomics pandas numpy tqdm
"""

# =========================================================
# CONFIG: EDIT ONLY THIS SECTION
# =========================================================

from pathlib import Path

PATIENT_ROOT = Path(r"E:\2_Experiments_MRI\MRI_Data\mama-mia_dataset\images")
TUMOR_MASK_DIR = Path(r"E:\2_Experiments_MRI\MRI_Data\mama-mia_dataset\segmentations\expert")
OUTPUT_DIR = Path(r"features_2")

# Optional clinical file.
# If you do not have clinical CSV now, keep it as None.
CLINICAL_CSV = None
# Example:
# CLINICAL_CSV = Path(r"D:\Dataset1\clinical.csv")

PATIENT_FILTER = "duke"

CLIP_LOWER = 1.0
CLIP_UPPER = 99.0

MIN_TUMOR_VOXELS = 30
MIN_BREAST_VOXELS = 1000
MIN_BACKGROUND_VOXELS = 500

BACKGROUND_EXCLUSION_MM = 10.0

# If the breast mask is one connected component:
# "split" = split left/right by image x-axis and keep tumor side
# "whole" = use entire breast mask
SINGLE_COMPONENT_MODE = "split"

# =========================================================
# DO NOT EDIT BELOW UNLESS NEEDED
# =========================================================

import re
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from radiomics import featureextractor

warnings.filterwarnings("ignore")

EPS = 1e-6


def strip_nii_gz(filename):
    name = Path(filename).name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(filename).stem


def normalize_id(name):
    name = strip_nii_gz(name)
    name = name.lower()
    name = re.sub(r"[^a-z0-9]", "", name)
    return name


def find_single_file(folder, pattern):
    files = sorted(Path(folder).glob(pattern))
    return files[0] if len(files) > 0 else None


def build_tumor_mask_index(tumor_mask_dir):
    tumor_mask_dir = Path(tumor_mask_dir)
    mask_files = list(tumor_mask_dir.glob("*.nii")) + list(tumor_mask_dir.glob("*.nii.gz"))

    index = {}
    for f in mask_files:
        index[normalize_id(f.name)] = f

    return index


def find_breast_mask(patient_dir):
    patient_dir = Path(patient_dir)

    new_mask = patient_dir / "new_breast_mask.nii.gz"
    old_mask = patient_dir / "breast_mask.nii.gz"

    if new_mask.exists():
        return new_mask

    if old_mask.exists():
        return old_mask

    candidates = list(patient_dir.glob("*breast*mask*.nii.gz")) + list(patient_dir.glob("*breast*mask*.nii"))

    if len(candidates) == 0:
        return None

    candidates = sorted(candidates)

    for c in candidates:
        if "new" in c.name.lower():
            return c

    return candidates[0]


def discover_duke_cases(patient_root, tumor_mask_dir, patient_filter="duke"):
    patient_root = Path(patient_root)
    tumor_mask_index = build_tumor_mask_index(tumor_mask_dir)

    rows = []

    for patient_dir in sorted([p for p in patient_root.iterdir() if p.is_dir()]):
        patient_id = patient_dir.name

        if patient_filter.lower() not in patient_id.lower():
            continue

        pre_path = find_single_file(patient_dir, "*_0000.nii.gz")
        post_path = find_single_file(patient_dir, "*_0001.nii.gz")
        breast_mask_path = find_breast_mask(patient_dir)

        patient_key = normalize_id(patient_id)
        tumor_mask_path = tumor_mask_index.get(patient_key, None)

        rows.append({
            "patient_id": patient_id,
            "patient_key": patient_key,
            "patient_dir": str(patient_dir),
            "pre_path": str(pre_path) if pre_path is not None else "",
            "post_path": str(post_path) if post_path is not None else "",
            "tumor_mask_path": str(tumor_mask_path) if tumor_mask_path is not None else "",
            "breast_mask_path": str(breast_mask_path) if breast_mask_path is not None else "",
            "has_pre": pre_path is not None,
            "has_post": post_path is not None,
            "has_tumor_mask": tumor_mask_path is not None,
            "has_breast_mask": breast_mask_path is not None,
        })

    return pd.DataFrame(rows)


def read_image(path, pixel_type=sitk.sitkFloat32):
    img = sitk.ReadImage(str(path))
    return sitk.Cast(img, pixel_type)


def read_mask(path):
    mask = sitk.ReadImage(str(path))
    return sitk.Cast(mask > 0, sitk.sitkUInt8)


def sitk_to_np(img):
    return sitk.GetArrayFromImage(img)


def np_to_sitk(arr, ref, pixel_type=sitk.sitkFloat32):
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(ref)
    return sitk.Cast(out, pixel_type)


def has_same_geometry(img, ref):
    return (
        img.GetSize() == ref.GetSize()
        and np.allclose(img.GetSpacing(), ref.GetSpacing())
        and np.allclose(img.GetOrigin(), ref.GetOrigin())
        and np.allclose(img.GetDirection(), ref.GetDirection())
    )


def resample_to_reference(img, ref, is_mask=False):
    if has_same_geometry(img, ref):
        return img

    interpolator = sitk.sitkNearestNeighbor if is_mask else sitk.sitkBSpline

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)

    out = resampler.Execute(img)

    if is_mask:
        out = sitk.Cast(out > 0, sitk.sitkUInt8)
    else:
        out = sitk.Cast(out, sitk.sitkFloat32)

    return out


def binary_mask_from_array(arr, ref):
    return np_to_sitk(arr.astype(np.uint8), ref, sitk.sitkUInt8)


def mask_voxel_count(mask):
    return int(np.sum(sitk_to_np(mask) > 0))


def mask_volume_ml(mask):
    spacing = mask.GetSpacing()
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    return mask_voxel_count(mask) * voxel_volume_mm3 / 1000.0


def mask_intersection(mask_a, mask_b):
    arr = (sitk_to_np(mask_a) > 0) & (sitk_to_np(mask_b) > 0)
    return binary_mask_from_array(arr, mask_a)


def mask_subtract(mask_a, mask_b):
    arr = (sitk_to_np(mask_a) > 0) & ~(sitk_to_np(mask_b) > 0)
    return binary_mask_from_array(arr, mask_a)


def dilate_mask_mm(mask, radius_mm):
    spacing = mask.GetSpacing()

    radius_vox = [
        max(1, int(round(radius_mm / spacing[0]))),
        max(1, int(round(radius_mm / spacing[1]))),
        max(1, int(round(radius_mm / spacing[2]))),
    ]

    dilated = sitk.BinaryDilate(
        sitk.Cast(mask > 0, sitk.sitkUInt8),
        radius_vox,
        sitk.sitkBall,
        1
    )

    return sitk.Cast(dilated > 0, sitk.sitkUInt8)


def split_single_component_breast_by_tumor_side(breast_mask, tumor_mask):
    breast_arr = sitk_to_np(breast_mask) > 0
    tumor_arr = sitk_to_np(tumor_mask) > 0

    if np.sum(breast_arr) == 0 or np.sum(tumor_arr) == 0:
        return breast_mask

    breast_coords = np.argwhere(breast_arr)
    tumor_coords = np.argwhere(tumor_arr)

    min_x = breast_coords[:, 2].min()
    max_x = breast_coords[:, 2].max()
    mid_x = (min_x + max_x) / 2.0

    tumor_x = tumor_coords[:, 2].mean()

    x_grid = np.arange(breast_arr.shape[2])[None, None, :]

    if tumor_x <= mid_x:
        selected = breast_arr & (x_grid <= mid_x)
    else:
        selected = breast_arr & (x_grid > mid_x)

    if np.sum(selected) == 0:
        return breast_mask

    return binary_mask_from_array(selected, breast_mask)


def select_breast_side_containing_tumor(breast_mask, tumor_mask, single_component_mode="split"):
    breast_arr = sitk_to_np(breast_mask) > 0
    tumor_arr = sitk_to_np(tumor_mask) > 0

    if np.sum(breast_arr) == 0:
        return breast_mask

    cc = sitk.ConnectedComponent(sitk.Cast(breast_mask > 0, sitk.sitkUInt8))
    cc_arr = sitk_to_np(cc)

    labels = sorted([int(x) for x in np.unique(cc_arr) if x != 0])

    if len(labels) == 0:
        return breast_mask

    if len(labels) == 1:
        one_component = binary_mask_from_array(cc_arr == labels[0], breast_mask)
        if single_component_mode == "split":
            return split_single_component_breast_by_tumor_side(one_component, tumor_mask)
        return one_component

    best_label = None
    best_overlap = -1

    for lab in labels:
        comp = cc_arr == lab
        overlap = int(np.sum(comp & tumor_arr))

        if overlap > best_overlap:
            best_overlap = overlap
            best_label = lab

    if best_overlap > 0:
        return binary_mask_from_array(cc_arr == best_label, breast_mask)

    tumor_coords = np.argwhere(tumor_arr)

    if tumor_coords.shape[0] == 0:
        sizes = [(lab, np.sum(cc_arr == lab)) for lab in labels]
        best_label = max(sizes, key=lambda x: x[1])[0]
        return binary_mask_from_array(cc_arr == best_label, breast_mask)

    tumor_centroid = tumor_coords.mean(axis=0)

    best_label = None
    best_dist = np.inf

    for lab in labels:
        comp_coords = np.argwhere(cc_arr == lab)

        if comp_coords.shape[0] == 0:
            continue

        comp_centroid = comp_coords.mean(axis=0)
        dist = np.linalg.norm(comp_centroid - tumor_centroid)

        if dist < best_dist:
            best_dist = dist
            best_label = lab

    return binary_mask_from_array(cc_arr == best_label, breast_mask)


def normalize_pre_post_using_breast_side(pre_img, post_img, breast_side_mask):
    pre = sitk_to_np(pre_img).astype(np.float32)
    post = sitk_to_np(post_img).astype(np.float32)
    mask = sitk_to_np(breast_side_mask) > 0

    vals = np.concatenate([pre[mask], post[mask]])

    if vals.size < 100:
        raise ValueError("Breast-side mask too small for normalization.")

    lo = np.percentile(vals, CLIP_LOWER)
    hi = np.percentile(vals, CLIP_UPPER)

    if hi <= lo:
        raise ValueError("Invalid normalization range.")

    pre_norm = np.clip(pre, lo, hi)
    post_norm = np.clip(post, lo, hi)

    pre_norm = (pre_norm - lo) / (hi - lo + EPS)
    post_norm = (post_norm - lo) / (hi - lo + EPS)

    pre_norm = np.clip(pre_norm, 0, 1)
    post_norm = np.clip(post_norm, 0, 1)

    return np_to_sitk(pre_norm, pre_img), np_to_sitk(post_norm, post_img)


def create_derived_images(pre_norm, post_norm):
    pre = sitk_to_np(pre_norm).astype(np.float32)
    post = sitk_to_np(post_norm).astype(np.float32)

    sub = post - pre

    rel = (post - pre) / (pre + EPS)
    rel = np.clip(rel, -1.0, 5.0)

    ratio = post / (pre + EPS)
    ratio = np.clip(ratio, 0.0, 6.0)

    return {
        "pre": pre_norm,
        "post": post_norm,
        "sub": np_to_sitk(sub, pre_norm),
        "rel": np_to_sitk(rel, pre_norm),
        "ratio": np_to_sitk(ratio, pre_norm),
    }


SELECTED_SHAPE = [
    "MeshVolume",
    "VoxelVolume",
    "SurfaceArea",
    "SurfaceVolumeRatio",
    "Sphericity",
    "Maximum3DDiameter",
    "MajorAxisLength",
    "MinorAxisLength",
    "LeastAxisLength",
    "Elongation",
    "Flatness",
]

SELECTED_FIRSTORDER = [
    "Mean",
    "Median",
    "Variance",
    "InterquartileRange",
    "Entropy",
    "Uniformity",
    "Skewness",
    "Kurtosis",
    "10Percentile",
    "90Percentile",
]

SELECTED_GLCM = [
    "Contrast",
    "Correlation",
    "JointEntropy",
    "Idmn",
]

SELECTED_GLRLM = [
    "RunEntropy",
    "RunLengthNonUniformityNormalized",
]

SELECTED_GLSZM = [
    "ZoneEntropy",
    "ZonePercentage",
    "SizeZoneNonUniformityNormalized",
]


def make_selected_extractor(include_shape=True):
    settings = {
        "binWidth": 0.05,
        "resampledPixelSpacing": [1, 1, 1],
        "interpolator": sitk.sitkBSpline,
        "normalize": False,
        "correctMask": True,
        "geometryTolerance": 1e-5,
        "force2D": False,
        "label": 1,
    }

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

    extractor.disableAllImageTypes()
    extractor.disableAllFeatures()
    extractor.enableImageTypeByName("Original")

    if include_shape:
        extractor.enableFeaturesByName(shape=SELECTED_SHAPE)

    extractor.enableFeaturesByName(firstorder=SELECTED_FIRSTORDER)
    extractor.enableFeaturesByName(glcm=SELECTED_GLCM)
    extractor.enableFeaturesByName(glrlm=SELECTED_GLRLM)
    extractor.enableFeaturesByName(glszm=SELECTED_GLSZM)

    return extractor


def clean_radiomics_output(result, prefix):
    out = OrderedDict()

    for k, v in result.items():
        k = str(k)

        if k.startswith("diagnostics"):
            continue

        try:
            if hasattr(v, "item"):
                v = v.item()
            out[f"{prefix}_{k}"] = float(v)
        except Exception:
            continue

    return out


def extract_selected_radiomics(img, mask, extractor, prefix, min_voxels):
    if mask_voxel_count(mask) < min_voxels:
        return OrderedDict()

    try:
        result = extractor.execute(img, mask)
        return clean_radiomics_output(result, prefix)
    except Exception as e:
        print(f"[WARNING] Radiomics failed for {prefix}: {e}")
        return OrderedDict()


def get_masked_values(img, mask):
    arr = sitk_to_np(img).astype(np.float32)
    m = sitk_to_np(mask) > 0

    vals = arr[m]
    vals = vals[np.isfinite(vals)]

    return vals


def safe_entropy(values, bins=32):
    if values.size < 2:
        return np.nan

    hist, _ = np.histogram(values, bins=bins)
    p = hist.astype(np.float64)
    p = p / (np.sum(p) + EPS)
    p = p[p > 0]

    return float(-np.sum(p * np.log2(p + EPS)))


def compact_stats(values, prefix):
    out = OrderedDict()

    keys = ["mean", "std", "median", "p10", "p90", "iqr", "entropy", "cv"]

    if values.size == 0:
        for k in keys:
            out[f"{prefix}_{k}"] = np.nan
        return out

    mean_val = float(np.mean(values))
    std_val = float(np.std(values))

    out[f"{prefix}_mean"] = mean_val
    out[f"{prefix}_std"] = std_val
    out[f"{prefix}_median"] = float(np.median(values))
    out[f"{prefix}_p10"] = float(np.percentile(values, 10))
    out[f"{prefix}_p90"] = float(np.percentile(values, 90))
    out[f"{prefix}_iqr"] = float(np.percentile(values, 75) - np.percentile(values, 25))
    out[f"{prefix}_entropy"] = safe_entropy(values)
    out[f"{prefix}_cv"] = float(std_val / (abs(mean_val) + EPS))

    return out


def tumor_enhancement_features(derived, tumor_mask):
    out = OrderedDict()

    out["tumor_volume_ml"] = mask_volume_ml(tumor_mask)

    for img_name in ["pre", "post", "sub", "rel", "ratio"]:
        vals = get_masked_values(derived[img_name], tumor_mask)
        out.update(compact_stats(vals, f"tumor_{img_name}"))

    rel_vals = get_masked_values(derived["rel"], tumor_mask)
    sub_vals = get_masked_values(derived["sub"], tumor_mask)

    if rel_vals.size > 0:
        out["tumor_rel_positive_fraction"] = float(np.mean(rel_vals > 0))
        out["tumor_rel_gt_005_fraction"] = float(np.mean(rel_vals > 0.05))
        out["tumor_rel_gt_010_fraction"] = float(np.mean(rel_vals > 0.10))
        out["tumor_rel_gt_025_fraction"] = float(np.mean(rel_vals > 0.25))
        out["tumor_rel_nonenhancing_fraction"] = float(np.mean(rel_vals <= 0.05))
        out["tumor_rel_high_to_low_ratio"] = float(
            np.mean(rel_vals > 0.25) / (np.mean(rel_vals <= 0.05) + EPS)
        )
    else:
        out["tumor_rel_positive_fraction"] = np.nan
        out["tumor_rel_gt_005_fraction"] = np.nan
        out["tumor_rel_gt_010_fraction"] = np.nan
        out["tumor_rel_gt_025_fraction"] = np.nan
        out["tumor_rel_nonenhancing_fraction"] = np.nan
        out["tumor_rel_high_to_low_ratio"] = np.nan

    if sub_vals.size > 0:
        out["tumor_sub_positive_fraction"] = float(np.mean(sub_vals > 0))
    else:
        out["tumor_sub_positive_fraction"] = np.nan

    return out


def breast_level_enhancement_features(derived, breast_side_mask, background_breast_mask):
    out = OrderedDict()

    out["breast_side_volume_ml"] = mask_volume_ml(breast_side_mask)
    out["background_breast_volume_ml"] = mask_volume_ml(background_breast_mask)

    for region_name, mask in [
        ("breast_side", breast_side_mask),
        ("background_breast", background_breast_mask),
    ]:
        for img_name in ["post", "sub", "rel", "ratio"]:
            vals = get_masked_values(derived[img_name], mask)
            out.update(compact_stats(vals, f"{region_name}_{img_name}"))

        rel_vals = get_masked_values(derived["rel"], mask)

        if rel_vals.size > 0:
            out[f"{region_name}_rel_gt_005_fraction"] = float(np.mean(rel_vals > 0.05))
            out[f"{region_name}_rel_gt_010_fraction"] = float(np.mean(rel_vals > 0.10))
            out[f"{region_name}_rel_gt_025_fraction"] = float(np.mean(rel_vals > 0.25))
        else:
            out[f"{region_name}_rel_gt_005_fraction"] = np.nan
            out[f"{region_name}_rel_gt_010_fraction"] = np.nan
            out[f"{region_name}_rel_gt_025_fraction"] = np.nan

    return out


def tumor_breast_ratio_features(derived, tumor_mask, breast_side_mask, background_breast_mask):
    out = OrderedDict()

    tumor_vol = mask_volume_ml(tumor_mask)
    breast_vol = mask_volume_ml(breast_side_mask)
    bg_vol = mask_volume_ml(background_breast_mask)

    out["tumor_volume_ml"] = tumor_vol
    out["breast_side_volume_ml"] = breast_vol
    out["background_breast_volume_ml"] = bg_vol
    out["tumor_to_breast_side_volume_ratio"] = tumor_vol / (breast_vol + EPS)

    for img_name in ["post", "sub", "rel", "ratio"]:
        tumor_vals = get_masked_values(derived[img_name], tumor_mask)
        breast_vals = get_masked_values(derived[img_name], breast_side_mask)
        bg_vals = get_masked_values(derived[img_name], background_breast_mask)

        tumor_mean = np.mean(tumor_vals) if tumor_vals.size else np.nan
        breast_mean = np.mean(breast_vals) if breast_vals.size else np.nan
        bg_mean = np.mean(bg_vals) if bg_vals.size else np.nan

        out[f"tumor_to_breast_side_{img_name}_mean_ratio"] = float(
            tumor_mean / (breast_mean + EPS)
        )
        out[f"tumor_to_background_breast_{img_name}_mean_ratio"] = float(
            tumor_mean / (bg_mean + EPS)
        )

    tumor_rel = get_masked_values(derived["rel"], tumor_mask)
    bg_rel = get_masked_values(derived["rel"], background_breast_mask)

    if tumor_rel.size > 0 and bg_rel.size > 0:
        bg_p75 = np.percentile(bg_rel, 75)
        bg_p90 = np.percentile(bg_rel, 90)
        bg_p95 = np.percentile(bg_rel, 95)

        out["tumor_rel_above_background_p75_fraction"] = float(np.mean(tumor_rel > bg_p75))
        out["tumor_rel_above_background_p90_fraction"] = float(np.mean(tumor_rel > bg_p90))
        out["tumor_rel_above_background_p95_fraction"] = float(np.mean(tumor_rel > bg_p95))

        tumor_enh_frac = float(np.mean(tumor_rel > 0.05))
        bg_enh_frac = float(np.mean(bg_rel > 0.05))

        out["tumor_rel_gt_005_fraction"] = tumor_enh_frac
        out["background_breast_rel_gt_005_fraction"] = bg_enh_frac
        out["tumor_to_background_enhancing_fraction_ratio"] = (
            tumor_enh_frac / (bg_enh_frac + EPS)
        )
    else:
        out["tumor_rel_above_background_p75_fraction"] = np.nan
        out["tumor_rel_above_background_p90_fraction"] = np.nan
        out["tumor_rel_above_background_p95_fraction"] = np.nan
        out["tumor_rel_gt_005_fraction"] = np.nan
        out["background_breast_rel_gt_005_fraction"] = np.nan
        out["tumor_to_background_enhancing_fraction_ratio"] = np.nan

    return out


def combine_enhancement_and_ratio_features(breast_df, ratio_df):
    """
    Merge breast-level enhancement features and tumor-breast ratio features
    into a single combined feature table (used as the 3rd input modality
    downstream, alongside clinical and tumor-level features).

    Both frames are built from the same masks/derived images for the same
    patients in process_patient_selected_features(), but we merge on
    patient_id/patient_key (rather than assuming positional alignment) so
    this function is also safe to call on the two CSVs loaded independently
    later (e.g. for combining an external cohort's features the same way).

    A few volume columns (breast_side_volume_ml, background_breast_volume_ml)
    are computed identically by both extractors — we drop the duplicate from
    `breast_df` and keep the copy from `ratio_df` to avoid _x/_y suffixes.
    """
    dup_cols = [c for c in ["breast_side_volume_ml", "background_breast_volume_ml"]
                if c in breast_df.columns and c in ratio_df.columns]
    breast_df_dedup = breast_df.drop(columns=dup_cols)

    combined = breast_df_dedup.merge(
        ratio_df,
        on=["patient_id", "patient_key"],
        how="inner",
        validate="one_to_one",
    )

    if len(combined) != len(breast_df) or len(combined) != len(ratio_df):
        warnings.warn(
            f"combine_enhancement_and_ratio_features: row count changed after merge "
            f"(breast_df={len(breast_df)}, ratio_df={len(ratio_df)}, combined={len(combined)}). "
            f"Check for patient_id/patient_key mismatches between the two inputs."
        )

    return combined


def process_patient_selected_features(row, extractor_with_shape, extractor_no_shape):
    meta = OrderedDict()
    meta["patient_id"] = row["patient_id"]
    meta["patient_key"] = row["patient_key"]

    qc = OrderedDict(meta)

    try:
        pre = read_image(row["pre_path"])
        post = read_image(row["post_path"])
        tumor = read_mask(row["tumor_mask_path"])
        breast = read_mask(row["breast_mask_path"])

        pre = resample_to_reference(pre, post, is_mask=False)
        tumor = resample_to_reference(tumor, post, is_mask=True)
        breast = resample_to_reference(breast, post, is_mask=True)

        breast_side = select_breast_side_containing_tumor(
            breast,
            tumor,
            single_component_mode=SINGLE_COMPONENT_MODE
        )

        tumor = mask_intersection(tumor, breast_side)

        if mask_voxel_count(tumor) < MIN_TUMOR_VOXELS:
            raise ValueError("Tumor mask too small or not overlapping tumor-side breast.")

        if mask_voxel_count(breast_side) < MIN_BREAST_VOXELS:
            raise ValueError("Tumor-side breast mask too small.")

        pre_norm, post_norm = normalize_pre_post_using_breast_side(pre, post, breast_side)
        derived = create_derived_images(pre_norm, post_norm)

        tumor_dilated = dilate_mask_mm(tumor, BACKGROUND_EXCLUSION_MM)
        background_breast = mask_subtract(breast_side, tumor_dilated)

        if mask_voxel_count(background_breast) < MIN_BACKGROUND_VOXELS:
            raise ValueError("Background breast mask too small after tumor exclusion.")

        tumor_row = OrderedDict(meta)

        tumor_row.update(
            extract_selected_radiomics(
                derived["post"],
                tumor,
                extractor_with_shape,
                prefix="tumor_post",
                min_voxels=MIN_TUMOR_VOXELS,
            )
        )

        tumor_row.update(
            extract_selected_radiomics(
                derived["rel"],
                tumor,
                extractor_no_shape,
                prefix="tumor_rel",
                min_voxels=MIN_TUMOR_VOXELS,
            )
        )

        tumor_row.update(tumor_enhancement_features(derived, tumor))

        breast_row = OrderedDict(meta)
        breast_row.update(
            breast_level_enhancement_features(
                derived,
                breast_side,
                background_breast,
            )
        )

        ratio_row = OrderedDict(meta)
        ratio_row.update(
            tumor_breast_ratio_features(
                derived,
                tumor,
                breast_side,
                background_breast,
            )
        )

        qc["status"] = "success"
        qc["error"] = ""
        qc["tumor_voxels"] = mask_voxel_count(tumor)
        qc["breast_side_voxels"] = mask_voxel_count(breast_side)
        qc["background_breast_voxels"] = mask_voxel_count(background_breast)
        qc["tumor_volume_ml"] = mask_volume_ml(tumor)
        qc["breast_side_volume_ml"] = mask_volume_ml(breast_side)
        qc["background_breast_volume_ml"] = mask_volume_ml(background_breast)
        qc["pre_path"] = row["pre_path"]
        qc["post_path"] = row["post_path"]
        qc["tumor_mask_path"] = row["tumor_mask_path"]
        qc["breast_mask_path"] = row["breast_mask_path"]

        return tumor_row, breast_row, ratio_row, qc

    except Exception as e:
        qc["status"] = "failed"
        qc["error"] = str(e)
        qc["pre_path"] = row.get("pre_path", "")
        qc["post_path"] = row.get("post_path", "")
        qc["tumor_mask_path"] = row.get("tumor_mask_path", "")
        qc["breast_mask_path"] = row.get("breast_mask_path", "")

        empty = OrderedDict(meta)
        return empty, empty, empty, qc


def create_clinical_features(valid_manifest):
    if CLINICAL_CSV is not None and Path(CLINICAL_CSV).exists():
        clinical_df = pd.read_csv(CLINICAL_CSV)

        possible_id_cols = [
            "patient_id",
            "Patient_ID",
            "PatientID",
            "ID",
            "id",
            "case_id",
            "Case_ID",
            "subject_id",
        ]

        id_col = None
        for c in possible_id_cols:
            if c in clinical_df.columns:
                id_col = c
                break

        if id_col is None:
            raise ValueError(
                "Could not find patient ID column in clinical CSV. "
                "Expected one of: " + ", ".join(possible_id_cols)
            )

        clinical_df["patient_key"] = clinical_df[id_col].apply(normalize_id)

        clinical_features_df = valid_manifest[["patient_id", "patient_key"]].merge(
            clinical_df,
            on="patient_key",
            how="left"
        )
    else:
        clinical_features_df = valid_manifest[["patient_id", "patient_key"]].copy()

    clinical_features_df.to_csv(OUTPUT_DIR / "clinical_features.csv", index=False)
    return clinical_features_df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Discovering patient folders...")

    manifest = discover_duke_cases(
        patient_root=PATIENT_ROOT,
        tumor_mask_dir=TUMOR_MASK_DIR,
        patient_filter=PATIENT_FILTER,
    )

    manifest_path = OUTPUT_DIR / "dataset1_manifest_found.csv"
    manifest.to_csv(manifest_path, index=False)

    valid_manifest = manifest[
        manifest["has_pre"]
        & manifest["has_post"]
        & manifest["has_tumor_mask"]
        & manifest["has_breast_mask"]
    ].copy()

    print(f"Total matched patient folders: {len(manifest)}")
    print(f"Complete valid cases: {len(valid_manifest)}")
    print(f"Manifest saved: {manifest_path}")

    if len(valid_manifest) == 0:
        raise RuntimeError("No complete valid cases found. Check paths and naming.")

    print("Creating selected radiomics extractors...")
    extractor_with_shape = make_selected_extractor(include_shape=True)
    extractor_no_shape = make_selected_extractor(include_shape=False)

    tumor_rows = []
    breast_rows = []
    ratio_rows = []
    qc_rows = []

    print("Extracting selected features...")

    for _, row in tqdm(valid_manifest.iterrows(), total=len(valid_manifest)):
        tumor_row, breast_row, ratio_row, qc = process_patient_selected_features(
            row,
            extractor_with_shape,
            extractor_no_shape,
        )

        tumor_rows.append(tumor_row)
        breast_rows.append(breast_row)
        ratio_rows.append(ratio_row)
        qc_rows.append(qc)

    tumor_df = pd.DataFrame(tumor_rows)
    breast_df = pd.DataFrame(breast_rows)
    ratio_df = pd.DataFrame(ratio_rows)
    qc_df = pd.DataFrame(qc_rows)

    # Combine breast-level enhancement + tumor-breast ratio into a single
    # modality-3 feature table for downstream training/inference.
    combined_df = combine_enhancement_and_ratio_features(breast_df, ratio_df)

    tumor_df.to_csv(OUTPUT_DIR / "tumor_level_features.csv", index=False)
    breast_df.to_csv(OUTPUT_DIR / "breast_level_enhancement_features.csv", index=False)
    ratio_df.to_csv(OUTPUT_DIR / "tumor_breast_ratio_features.csv", index=False)
    combined_df.to_csv(OUTPUT_DIR / "breast_ratio_enhancement_features.csv", index=False)
    qc_df.to_csv(OUTPUT_DIR / "qc_selected_features.csv", index=False)

    clinical_df = create_clinical_features(valid_manifest)

    print("\nDone.")
    print("Saved files:")
    print(f"  {OUTPUT_DIR / 'tumor_level_features.csv'}")
    print(f"  {OUTPUT_DIR / 'breast_level_enhancement_features.csv'}  (QC only)")
    print(f"  {OUTPUT_DIR / 'tumor_breast_ratio_features.csv'}  (QC only)")
    print(f"  {OUTPUT_DIR / 'breast_ratio_enhancement_features.csv'}  <-- use this as modality 3")
    print(f"  {OUTPUT_DIR / 'clinical_features.csv'}")
    print(f"  {OUTPUT_DIR / 'qc_selected_features.csv'}")
    print(f"  {OUTPUT_DIR / 'dataset1_manifest_found.csv'}")

    print("\nShapes:")
    print(f"  tumor_level_features: {tumor_df.shape}")
    print(f"  breast_level_enhancement_features: {breast_df.shape}")
    print(f"  tumor_breast_ratio_features: {ratio_df.shape}")
    print(f"  breast_ratio_enhancement_features (combined): {combined_df.shape}")
    print(f"  clinical_features: {clinical_df.shape}")
    print(f"  qc: {qc_df.shape}")

    print("\nQC status:")
    if "status" in qc_df.columns:
        print(qc_df["status"].value_counts(dropna=False))


if __name__ == "__main__":
    main()

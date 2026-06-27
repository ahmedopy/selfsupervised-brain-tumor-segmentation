"""
Uncertainty Calibration + Per-Modality Evaluation
=================================================

Outputs:
1. evaluation_results.csv
   - Per-modality combination Dice scores
   - Label-wise Dice: NCR, ED, ET
   - Clinical Dice: ET, TC, WT
   - Variance values

2. uncertainty_per_case.csv
   - Per-case Dice and HeMIS variance values

3. uncertainty_summary.csv
   - ROI variance quartile summary

4. uncertainty_calibration.png
   - ROI variance quartiles vs Dice
   - ROI variance vs Dice scatter plot
   - Number of modalities vs Dice

5. correlation_heatmap.png
   - Correlation heatmap between Dice, modality count, and variance metrics

6. correlation_matrix.csv
   - Numeric correlation matrix

7. classification_metrics.csv
   - Voxel-wise precision, recall, and F1-score for NCR, ED, ET

8. confusion_matrix_tumor_only_all4.csv
   - Tumor-only NCR/ED/ET confusion matrix for All 4 modalities

9. confusion_matrix_tumor_only_all4.png
   - Tumor-only NCR/ED/ET confusion matrix figure for All 4 modalities

Excluded intentionally:
- Full 4-class background confusion matrix
"""

import os
import csv
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hemis_ssl_model import HeMIS_SSL
from train_eval import BraTS2021Dataset, collate_fn, dice_score


# ---------------------------------------------------------
# Modality combinations
# FLAIR=0, T1ce=1, T1=2, T2=3
# ---------------------------------------------------------
MODALITY_COMBOS = [
    ([0],       "F"),
    ([1],       "T1c"),
    ([2],       "T1"),
    ([3],       "T2"),
    ([0, 1],     "F+T1c"),
    ([0, 2],     "F+T1"),
    ([0, 3],     "F+T2"),
    ([1, 2],     "T1c+T1"),
    ([1, 3],     "T1c+T2"),
    ([2, 3],     "T1+T2"),
    ([0, 1, 2],   "F+T1c+T1"),
    ([0, 1, 3],   "F+T1c+T2"),
    ([0, 2, 3],   "F+T1+T2"),
    ([1, 2, 3],   "T1c+T1+T2"),
    ([0, 1, 2, 3], "All 4"),
]

CLASS_NAMES = {
    1: "NCR",
    2: "ED",
    3: "ET",
}

TUMOR_LABELS = [1, 2, 3]


def safe_mean(values):
    arr = np.array(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    return float(arr.mean())


def safe_div(num, den):
    if den == 0:
        return 0.0
    return float(num / den)


def binary_dice(pred_mask, true_mask, eps=1e-6):
    pred_mask = pred_mask.bool()
    true_mask = true_mask.bool()

    inter = (pred_mask & true_mask).sum().float()
    denom = pred_mask.sum().float() + true_mask.sum().float()

    if denom.item() == 0:
        return 1.0

    dice = (2.0 * inter + eps) / (denom + eps)
    return float(dice.detach().cpu().item())


def clinical_dice_from_prediction(pred, seg):
    """
    BraTS clinical regions:
    ET = label 3
    TC = NCR + ET = labels 1 and 3
    WT = NCR + ED + ET = labels 1, 2, 3
    """
    pred_et = pred == 3
    seg_et = seg == 3

    pred_tc = (pred == 1) | (pred == 3)
    seg_tc = (seg == 1) | (seg == 3)

    pred_wt = pred > 0
    seg_wt = seg > 0

    et = binary_dice(pred_et, seg_et)
    tc = binary_dice(pred_tc, seg_tc)
    wt = binary_dice(pred_wt, seg_wt)

    return et, tc, wt


def extract_dice_values(logits, seg):
    """
    Compatible with current train_eval.py dice_score().

    Current train_eval.py returns:
        scores, et, tc, wt

    Older versions may return:
        scores
    """
    result = dice_score(logits, seg)

    if isinstance(result, tuple) and len(result) == 4:
        scores, clinical_et, clinical_tc, clinical_wt = result

        ncr = float(scores[0])
        ed = float(scores[1])
        et_label = float(scores[2])
        label_mean = float(np.mean(scores))

        clinical_et = float(clinical_et)
        clinical_tc = float(clinical_tc)
        clinical_wt = float(clinical_wt)
        clinical_mean = float(np.mean([clinical_et, clinical_tc, clinical_wt]))

        return ncr, ed, et_label, label_mean, clinical_et, clinical_tc, clinical_wt, clinical_mean

    scores = result
    pred = torch.argmax(logits, dim=1)

    ncr = float(scores[0])
    ed = float(scores[1])
    et_label = float(scores[2])
    label_mean = float(np.mean(scores))

    clinical_et, clinical_tc, clinical_wt = clinical_dice_from_prediction(pred, seg)
    clinical_mean = float(np.mean([clinical_et, clinical_tc, clinical_wt]))

    return ncr, ed, et_label, label_mean, clinical_et, clinical_tc, clinical_wt, clinical_mean


def get_hemis_variance_values(model, modalities, pred, seg, device):
    """
    Returns:
    1. global variance
    2. predicted-tumor-only variance
    3. ROI variance where ROI = predicted tumor OR ground-truth tumor

    ROI variance is more useful than whole-brain variance because
    background voxels dominate the MRI volume.
    """
    model.eval()

    with torch.no_grad():
        _, _, feats3 = model.encode_modalities(modalities)

        present3 = [f for f in feats3 if f is not None]

        if len(present3) <= 1:
            return 0.0, float("nan"), float("nan")

        stacked = torch.stack(present3, dim=0)

        var = stacked.var(dim=0, unbiased=False)
        var_scalar = var.mean(dim=1, keepdim=True)

        var_up = F.interpolate(
            var_scalar,
            size=pred.shape[-3:],
            mode="trilinear",
            align_corners=False
        )

        global_var = float(var_up.mean().detach().cpu().item())

        pred_tumor_mask = (pred > 0).unsqueeze(1)
        roi_mask = ((pred > 0) | (seg > 0)).unsqueeze(1)

        if pred_tumor_mask.sum().item() > 0:
            pred_tumor_var = float(var_up[pred_tumor_mask].mean().detach().cpu().item())
        else:
            pred_tumor_var = float("nan")

        if roi_mask.sum().item() > 0:
            roi_var = float(var_up[roi_mask].mean().detach().cpu().item())
        else:
            roi_var = float("nan")

        return global_var, pred_tumor_var, roi_var


def init_classification_counter():
    return {
        1: {"TP": 0, "FP": 0, "FN": 0},
        2: {"TP": 0, "FP": 0, "FN": 0},
        3: {"TP": 0, "FP": 0, "FN": 0},
    }


def update_classification_counter(counter, pred, seg):
    """
    Voxel-wise one-vs-rest precision/recall/F1 for tumor classes only:
    NCR, ED, ET.
    """
    for label in TUMOR_LABELS:
        tp = ((pred == label) & (seg == label)).sum().item()
        fp = ((pred == label) & (seg != label)).sum().item()
        fn = ((pred != label) & (seg == label)).sum().item()

        counter[label]["TP"] += int(tp)
        counter[label]["FP"] += int(fp)
        counter[label]["FN"] += int(fn)


def compute_classification_metrics(counter):
    row = {}

    precision_values = []
    recall_values = []
    f1_values = []

    for label in TUMOR_LABELS:
        name = CLASS_NAMES[label]

        tp = counter[label]["TP"]
        fp = counter[label]["FP"]
        fn = counter[label]["FN"]

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)

        row[f"Precision_{name}"] = precision
        row[f"Recall_{name}"] = recall
        row[f"F1_{name}"] = f1

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)

    row["Macro_Precision"] = float(np.mean(precision_values))
    row["Macro_Recall"] = float(np.mean(recall_values))
    row["Macro_F1"] = float(np.mean(f1_values))

    return row


def update_tumor_only_confusion_matrix(conf_matrix, pred, seg):
    """
    Tumor-only 3x3 confusion matrix:
    rows = true labels NCR/ED/ET
    cols = predicted labels NCR/ED/ET

    Background is excluded intentionally.
    """
    true_flat = seg.reshape(-1)
    pred_flat = pred.reshape(-1)

    mask = (
        (true_flat >= 1) & (true_flat <= 3) &
        (pred_flat >= 1) & (pred_flat <= 3)
    )

    if mask.sum().item() == 0:
        return conf_matrix

    true_tumor = true_flat[mask] - 1
    pred_tumor = pred_flat[mask] - 1

    idx = true_tumor * 3 + pred_tumor
    counts = torch.bincount(idx.long(), minlength=9).reshape(3, 3)

    conf_matrix += counts.detach().cpu().numpy().astype(np.int64)

    return conf_matrix


def save_tumor_confusion_matrix(args, conf_matrix):
    csv_path = os.path.join(args.output_dir, "confusion_matrix_tumor_only_all4.csv")
    png_path = os.path.join(args.output_dir, "confusion_matrix_tumor_only_all4.png")

    labels = ["NCR", "ED", "ET"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["True\\Pred"] + labels)
        for label, row in zip(labels, conf_matrix):
            writer.writerow([label] + [int(v) for v in row])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(conf_matrix, cmap="Blues")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted tumor class")
    ax.set_ylabel("True tumor class")
    ax.set_title("Tumor-Only Confusion Matrix: All 4 Modalities")

    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(
                j,
                i,
                str(int(conf_matrix[i, j])),
                ha="center",
                va="center",
                fontsize=10,
                color="black"
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved tumor-only confusion matrix CSV -> {csv_path}")
    print(f"Saved tumor-only confusion matrix PNG -> {png_path}")

    return csv_path, png_path


def load_model_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    model.load_state_dict(state, strict=True)

    epoch = "?"
    best_value = "?"

    if isinstance(ckpt, dict):
        epoch = ckpt.get("epoch", "?")
        best_value = ckpt.get("best_dice", ckpt.get("best_loss", "?"))

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Checkpoint epoch: {epoch}")
    print(f"Checkpoint best value: {best_value}")

    return model


def save_correlation_heatmap(args, per_case_rows):
    valid_rows = [
        r for r in per_case_rows
        if r["Num_Modalities"] > 1
        and np.isfinite(r["Variance_Global"])
        and np.isfinite(r["Variance_PredTumor"])
        and np.isfinite(r["Variance_ROI"])
        and np.isfinite(r["Label_Mean"])
        and np.isfinite(r["Clinical_Mean"])
    ]

    if len(valid_rows) < 5:
        print("Not enough valid rows to generate correlation heatmap.")
        return None, None

    heatmap_metrics = {
        "Num_Modalities": np.array([r["Num_Modalities"] for r in valid_rows], dtype=np.float64),
        "Label_Mean_Dice": np.array([r["Label_Mean"] for r in valid_rows], dtype=np.float64),
        "Clinical_Mean_Dice": np.array([r["Clinical_Mean"] for r in valid_rows], dtype=np.float64),
        "Variance_Global": np.array([r["Variance_Global"] for r in valid_rows], dtype=np.float64),
        "Variance_PredTumor": np.array([r["Variance_PredTumor"] for r in valid_rows], dtype=np.float64),
        "Variance_ROI": np.array([r["Variance_ROI"] for r in valid_rows], dtype=np.float64),
    }

    heatmap_names = list(heatmap_metrics.keys())
    heatmap_data = np.stack([heatmap_metrics[name] for name in heatmap_names], axis=1)

    valid_mask = np.isfinite(heatmap_data).all(axis=1)
    heatmap_data = heatmap_data[valid_mask]

    if heatmap_data.shape[0] < 5:
        print("Not enough finite data to generate correlation heatmap.")
        return None, None

    corr_matrix = np.corrcoef(heatmap_data, rowvar=False)

    heatmap_path = os.path.join(args.output_dir, "correlation_heatmap.png")
    matrix_csv_path = os.path.join(args.output_dir, "correlation_matrix.csv")

    with open(matrix_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric"] + heatmap_names)
        for name, row in zip(heatmap_names, corr_matrix):
            writer.writerow([name] + [f"{v:.6f}" for v in row])

    fig_h, ax_h = plt.subplots(figsize=(10, 8))
    im = ax_h.imshow(corr_matrix, vmin=-1, vmax=1, cmap="coolwarm")

    ax_h.set_xticks(np.arange(len(heatmap_names)))
    ax_h.set_yticks(np.arange(len(heatmap_names)))
    ax_h.set_xticklabels(heatmap_names, rotation=45, ha="right")
    ax_h.set_yticklabels(heatmap_names)

    for i in range(len(heatmap_names)):
        for j in range(len(heatmap_names)):
            value = corr_matrix[i, j]
            ax_h.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=9
            )

    ax_h.set_title("Correlation Heatmap: Dice, Modalities, and HeMIS Variance")
    fig_h.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved correlation heatmap -> {heatmap_path}")
    print(f"Saved correlation matrix -> {matrix_csv_path}")

    return heatmap_path, matrix_csv_path


def run_evaluation(args, model, device):
    os.makedirs(args.output_dir, exist_ok=True)

    print("\nLoading validation cases...")

    test_ds = BraTS2021Dataset(
        args.data_dir,
        args.json,
        val_fold=args.val_fold,
        patch_size=args.patch_size,
        missing_prob=0.0,
        augment=False,
        mode="val",
        cache_dir=args.cache_dir
    )

    loader_kwargs = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": args.workers,
        "collate_fn": collate_fn,
        "pin_memory": (device.type == "cuda"),
    }

    if args.workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["persistent_workers"] = True

    loader = DataLoader(test_ds, **loader_kwargs)

    print(f"Validation cases: {len(test_ds)}")
    print("Evaluating all modality combinations...\n")

    model.eval()

    per_case_rows = []

    class_counters = {
        combo_name: init_classification_counter()
        for _, combo_name in MODALITY_COMBOS
    }

    tumor_confusion_all4 = np.zeros((3, 3), dtype=np.int64)

    with torch.no_grad():
        for case_idx, (full_mods_cpu, seg_cpu) in enumerate(loader):
            seg_cpu = seg_cpu.long()

            for present_ids, combo_name in MODALITY_COMBOS:
                n_mods = len(present_ids)

                mods = [
                    full_mods_cpu[i].to(device, non_blocking=True) if i in present_ids else None
                    for i in range(4)
                ]

                seg = seg_cpu.to(device, non_blocking=True)

                with autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(mods)

                pred = torch.argmax(logits, dim=1)

                (
                    ncr,
                    ed,
                    et_label,
                    label_mean,
                    clinical_et,
                    clinical_tc,
                    clinical_wt,
                    clinical_mean
                ) = extract_dice_values(logits, seg)

                var_global, var_pred_tumor, var_roi = get_hemis_variance_values(
                    model,
                    mods,
                    pred,
                    seg,
                    device
                )

                update_classification_counter(
                    class_counters[combo_name],
                    pred,
                    seg
                )

                if combo_name == "All 4":
                    tumor_confusion_all4 = update_tumor_only_confusion_matrix(
                        tumor_confusion_all4,
                        pred,
                        seg
                    )

                per_case_rows.append({
                    "Case": case_idx,
                    "Combination": combo_name,
                    "Num_Modalities": n_mods,

                    "NCR": ncr,
                    "ED": ed,
                    "ET_Label": et_label,
                    "Label_Mean": label_mean,

                    "Clinical_ET": clinical_et,
                    "Clinical_TC": clinical_tc,
                    "Clinical_WT": clinical_wt,
                    "Clinical_Mean": clinical_mean,

                    "Variance_Global": var_global,
                    "Variance_PredTumor": var_pred_tumor,
                    "Variance_ROI": var_roi,
                })

            if (case_idx + 1) % 25 == 0:
                print(f"  Processed {case_idx + 1}/{len(test_ds)} cases")

    print("=" * 110)
    print("Per-Modality Combination Results")
    print("=" * 110)
    print(
        f"{'Combination':<14} "
        f"{'NCR':>7} {'ED':>7} {'ET':>7} {'Mean':>7} "
        f"{'ClinET':>8} {'TC':>7} {'WT':>7} {'ClinM':>7} "
        f"{'VarG':>8} {'VarROI':>8} {'#Mods':>6}"
    )
    print("-" * 110)

    combo_results = []

    for present_ids, combo_name in MODALITY_COMBOS:
        rows = [r for r in per_case_rows if r["Combination"] == combo_name]

        result = {
            "Combination": combo_name,
            "NCR": safe_mean([r["NCR"] for r in rows]),
            "ED": safe_mean([r["ED"] for r in rows]),
            "ET_Label": safe_mean([r["ET_Label"] for r in rows]),
            "Label_Mean": safe_mean([r["Label_Mean"] for r in rows]),

            "Clinical_ET": safe_mean([r["Clinical_ET"] for r in rows]),
            "Clinical_TC": safe_mean([r["Clinical_TC"] for r in rows]),
            "Clinical_WT": safe_mean([r["Clinical_WT"] for r in rows]),
            "Clinical_Mean": safe_mean([r["Clinical_Mean"] for r in rows]),

            "Variance_Global": safe_mean([r["Variance_Global"] for r in rows]),
            "Variance_PredTumor": safe_mean([r["Variance_PredTumor"] for r in rows]),
            "Variance_ROI": safe_mean([r["Variance_ROI"] for r in rows]),

            "Num_Modalities": len(present_ids),
        }

        combo_results.append(result)

        print(
            f"{result['Combination']:<14} "
            f"{result['NCR']:>7.3f} {result['ED']:>7.3f} {result['ET_Label']:>7.3f} {result['Label_Mean']:>7.3f} "
            f"{result['Clinical_ET']:>8.3f} {result['Clinical_TC']:>7.3f} {result['Clinical_WT']:>7.3f} {result['Clinical_Mean']:>7.3f} "
            f"{result['Variance_Global']:>8.4f} {result['Variance_ROI']:>8.4f} {result['Num_Modalities']:>6}"
        )

    avg_result = {
        "Combination": "Average",
        "NCR": safe_mean([r["NCR"] for r in combo_results]),
        "ED": safe_mean([r["ED"] for r in combo_results]),
        "ET_Label": safe_mean([r["ET_Label"] for r in combo_results]),
        "Label_Mean": safe_mean([r["Label_Mean"] for r in combo_results]),

        "Clinical_ET": safe_mean([r["Clinical_ET"] for r in combo_results]),
        "Clinical_TC": safe_mean([r["Clinical_TC"] for r in combo_results]),
        "Clinical_WT": safe_mean([r["Clinical_WT"] for r in combo_results]),
        "Clinical_Mean": safe_mean([r["Clinical_Mean"] for r in combo_results]),

        "Variance_Global": safe_mean([r["Variance_Global"] for r in combo_results]),
        "Variance_PredTumor": safe_mean([r["Variance_PredTumor"] for r in combo_results]),
        "Variance_ROI": safe_mean([r["Variance_ROI"] for r in combo_results]),

        "Num_Modalities": "-",
    }

    print("-" * 110)
    print(
        f"{avg_result['Combination']:<14} "
        f"{avg_result['NCR']:>7.3f} {avg_result['ED']:>7.3f} {avg_result['ET_Label']:>7.3f} {avg_result['Label_Mean']:>7.3f} "
        f"{avg_result['Clinical_ET']:>8.3f} {avg_result['Clinical_TC']:>7.3f} {avg_result['Clinical_WT']:>7.3f} {avg_result['Clinical_Mean']:>7.3f} "
        f"{avg_result['Variance_Global']:>8.4f} {avg_result['Variance_ROI']:>8.4f}"
    )
    print("=" * 110)

    eval_csv = os.path.join(args.output_dir, "evaluation_results.csv")

    eval_fields = [
        "Combination",
        "NCR",
        "ED",
        "ET_Label",
        "Label_Mean",
        "Clinical_ET",
        "Clinical_TC",
        "Clinical_WT",
        "Clinical_Mean",
        "Variance_Global",
        "Variance_PredTumor",
        "Variance_ROI",
        "Num_Modalities",
    ]

    with open(eval_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=eval_fields)
        writer.writeheader()

        for r in combo_results:
            writer.writerow(r)

        writer.writerow(avg_result)

    print(f"\nSaved aggregate table -> {eval_csv}")

    per_case_csv = os.path.join(args.output_dir, "uncertainty_per_case.csv")

    per_case_fields = [
        "Case",
        "Combination",
        "Num_Modalities",
        "NCR",
        "ED",
        "ET_Label",
        "Label_Mean",
        "Clinical_ET",
        "Clinical_TC",
        "Clinical_WT",
        "Clinical_Mean",
        "Variance_Global",
        "Variance_PredTumor",
        "Variance_ROI",
    ]

    with open(per_case_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_case_fields)
        writer.writeheader()
        for r in per_case_rows:
            writer.writerow(r)

    print(f"Saved per-case uncertainty table -> {per_case_csv}")

    class_csv = os.path.join(args.output_dir, "classification_metrics.csv")

    class_fields = [
        "Combination",
        "Num_Modalities",
        "Precision_NCR",
        "Recall_NCR",
        "F1_NCR",
        "Precision_ED",
        "Recall_ED",
        "F1_ED",
        "Precision_ET",
        "Recall_ET",
        "F1_ET",
        "Macro_Precision",
        "Macro_Recall",
        "Macro_F1",
    ]

    class_rows = []

    for present_ids, combo_name in MODALITY_COMBOS:
        metric_row = compute_classification_metrics(class_counters[combo_name])
        metric_row["Combination"] = combo_name
        metric_row["Num_Modalities"] = len(present_ids)
        class_rows.append(metric_row)

    avg_class_row = {
        "Combination": "Average",
        "Num_Modalities": "-",
    }

    for field in class_fields:
        if field not in ["Combination", "Num_Modalities"]:
            avg_class_row[field] = safe_mean([r[field] for r in class_rows])

    class_rows.append(avg_class_row)

    with open(class_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=class_fields)
        writer.writeheader()
        for r in class_rows:
            writer.writerow(r)

    print(f"Saved classification metrics -> {class_csv}")

    save_tumor_confusion_matrix(args, tumor_confusion_all4)

    valid_rows = [
        r for r in per_case_rows
        if r["Num_Modalities"] > 1 and np.isfinite(r["Variance_ROI"])
    ]

    variances = np.array([r["Variance_ROI"] for r in valid_rows], dtype=np.float64)
    dices = np.array([r["Label_Mean"] for r in valid_rows], dtype=np.float64)

    print("\n" + "=" * 70)
    print("Uncertainty Calibration Analysis")
    print("=" * 70)

    if len(valid_rows) < 5 or np.std(variances) == 0 or np.std(dices) == 0:
        correlation = float("nan")
        print("Not enough valid variance diversity to compute reliable correlation.")
    else:
        correlation = float(np.corrcoef(variances, dices)[0, 1])
        print(f"ROI Variance-Dice correlation: {correlation:.4f}")

        if correlation < -0.5:
            print("Strong evidence: higher tumor-region variance is associated with lower Dice.")
        elif correlation < -0.3:
            print("Moderate evidence: higher tumor-region variance tends to reduce Dice.")
        elif correlation < -0.1:
            print("Weak evidence: variance has a small negative relationship with Dice.")
        else:
            print("Weak/unclear evidence: do not strongly claim variance predicts error.")

    order = np.argsort(variances)
    groups = np.array_split(order, 4)

    quartile_rows = []

    for q_idx, group in enumerate(groups, start=1):
        if len(group) == 0:
            row = {
                "Quartile": f"Q{q_idx}",
                "Meaning": "",
                "Count": 0,
                "Mean_Variance_ROI": float("nan"),
                "Mean_Dice": float("nan"),
            }
        else:
            if q_idx == 1:
                meaning = "Lowest uncertainty"
            elif q_idx == 4:
                meaning = "Highest uncertainty"
            else:
                meaning = "Middle uncertainty"

            row = {
                "Quartile": f"Q{q_idx}",
                "Meaning": meaning,
                "Count": int(len(group)),
                "Mean_Variance_ROI": float(variances[group].mean()),
                "Mean_Dice": float(dices[group].mean()),
            }

        quartile_rows.append(row)

    summary_csv = os.path.join(args.output_dir, "uncertainty_summary.csv")

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Quartile",
                "Meaning",
                "Count",
                "Mean_Variance_ROI",
                "Mean_Dice",
            ]
        )
        writer.writeheader()
        for r in quartile_rows:
            writer.writerow(r)

    print(f"Saved uncertainty summary -> {summary_csv}")

    print("\nUncertainty quartiles:")
    for r in quartile_rows:
        print(
            f"  {r['Quartile']} ({r['Meaning']}): "
            f"Variance={r['Mean_Variance_ROI']:.4f}, "
            f"Dice={r['Mean_Dice']:.4f}, "
            f"n={r['Count']}"
        )

    print("\nDice by number of modalities:")
    mod_means = []
    mod_labels = []

    for n in range(1, 5):
        rows_n = [r for r in per_case_rows if r["Num_Modalities"] == n]
        mean_n = safe_mean([r["Label_Mean"] for r in rows_n])
        mod_means.append(mean_n)
        mod_labels.append(f"{n} modality" if n == 1 else f"{n} modalities")
        print(f"  {n} modality/ies: Mean Dice = {mean_n:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Uncertainty Calibration: HeMIS ROI Variance vs Segmentation Accuracy",
        fontsize=14,
        fontweight="bold"
    )

    q_labels = [f"{r['Quartile']}\n{r['Meaning']}" for r in quartile_rows]
    q_dices = [r["Mean_Dice"] for r in quartile_rows]

    axes[0].bar(q_labels, q_dices, edgecolor="black", alpha=0.85)
    axes[0].set_title("Tumor-Region Variance Quartiles")
    axes[0].set_xlabel("Uncertainty Level")
    axes[0].set_ylabel("Mean Dice")
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(axis="y", alpha=0.3)

    for i, val in enumerate(q_dices):
        if np.isfinite(val):
            axes[0].text(i, val + 0.01, f"{val:.3f}", ha="center", fontsize=9)

    axes[1].scatter(variances, dices, alpha=0.5)
    axes[1].set_title(f"Per-Case ROI Variance vs Dice\nr = {correlation:.3f}")
    axes[1].set_xlabel("ROI HeMIS Variance")
    axes[1].set_ylabel("Mean Dice")
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(alpha=0.3)

    if len(valid_rows) >= 5 and np.std(variances) > 0:
        coeff = np.polyfit(variances, dices, 1)
        x_line = np.linspace(variances.min(), variances.max(), 100)
        y_line = coeff[0] * x_line + coeff[1]
        axes[1].plot(x_line, y_line, linewidth=2)

    axes[2].bar(mod_labels, mod_means, edgecolor="black", alpha=0.85)
    axes[2].set_title("More Modalities → Better Accuracy")
    axes[2].set_xlabel("Number of Available Modalities")
    axes[2].set_ylabel("Mean Dice")
    axes[2].set_ylim(0, 1.0)
    axes[2].grid(axis="y", alpha=0.3)

    for i, val in enumerate(mod_means):
        if np.isfinite(val):
            axes[2].text(i, val + 0.01, f"{val:.3f}", ha="center", fontsize=9)

    plt.tight_layout()

    fig_path = os.path.join(args.output_dir, "uncertainty_calibration.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved figure -> {fig_path}")

    heatmap_path, matrix_csv_path = save_correlation_heatmap(args, per_case_rows)

    print("\nDone. Files saved:")
    print(f"  {eval_csv}")
    print(f"  {per_case_csv}")
    print(f"  {summary_csv}")
    print(f"  {fig_path}")
    print(f"  {class_csv}")
    print(f"  {os.path.join(args.output_dir, 'confusion_matrix_tumor_only_all4.csv')}")
    print(f"  {os.path.join(args.output_dir, 'confusion_matrix_tumor_only_all4.png')}")

    if heatmap_path is not None:
        print(f"  {heatmap_path}")

    if matrix_csv_path is not None:
        print(f"  {matrix_csv_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--json", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)

    parser.add_argument("--val_fold", type=int, default=0)
    parser.add_argument("--patch_size", type=int, default=128)

    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--ssl_embed_dim", type=int, default=384)
    parser.add_argument("--ssl_depth", type=int, default=4)
    parser.add_argument("--ssl_patch_size", type=int, default=16)

    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./uncertainty_outputs")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = HeMIS_SSL(
        num_modalities=4,
        num_classes=4,
        base_ch=args.base_ch,
        ssl_embed_dim=args.ssl_embed_dim,
        ssl_depth=args.ssl_depth,
        ssl_patch_size=args.ssl_patch_size
    ).to(device)

    model = load_model_checkpoint(model, args.ckpt, device)
    model.eval()

    run_evaluation(args, model, device)


if __name__ == "__main__":
    main()

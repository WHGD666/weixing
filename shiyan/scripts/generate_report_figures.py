"""Generate report-ready figures from frozen experiment artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
GROUP_COLORS = {"ship": "#2f6f9f", "aircraft": "#d47f27", "vehicle": "#4c956c"}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def setup():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
        }
    )


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_training_comparison(out: Path):
    specs = [
        ("EXP001 baseline", ROOT / "runs/detect/runs/train/exp001_original_yolo11s_baseline/results.csv", "#555555"),
        ("EXP002 data2 revision", ROOT / "runs/detect/runs/train/exp002_data2_manual_revision/results.csv", "#2f6f9f"),
        ("EXP004 continue 40", ROOT / "runs/detect/runs/train/exp004_original_continue40/results.csv", "#d47f27"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, path, color in specs:
        rows = read_csv(path)
        epochs = np.array([int(float(r["epoch"])) for r in rows])
        recall = np.array([float(r["metrics/recall(B)"]) for r in rows])
        map50 = np.array([float(r["metrics/mAP50(B)"]) for r in rows])
        axes[0].plot(epochs, recall, label=label, color=color, linewidth=2)
        axes[1].plot(epochs, map50, label=label, color=color, linewidth=2)
    axes[0].axhline(0.85, color="#b23a48", linestyle="--", linewidth=1.2, label="Recall gate 0.85")
    axes[0].set_title("Validation Recall by training epoch")
    axes[1].set_title("Validation mAP50 by training epoch")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylim(0, 1.02)
        ax.legend(frameon=False, loc="lower right")
    axes[0].set_ylabel("Score")
    fig.suptitle("Training trajectory comparison", fontsize=15, fontweight="bold")
    fig.subplots_adjust(top=0.82, bottom=0.22, wspace=0.22)
    fig.text(0.5, 0.01, "Source: Ultralytics results.csv; validation metrics, not official hidden-test results.", ha="center", fontsize=9, color="#555555")
    save(fig, out / "training_metrics_comparison.png")


def plot_local_gates(out: Path):
    specs = [
        ("EXP003 control", ROOT / "runs/test/EXP003_v2_modality_control/metrics/official_metrics.json"),
        ("EXP003 gray027", ROOT / "runs/test/EXP003_v2_modality_gray027/metrics/official_metrics.json"),
        ("EXP004 raw", ROOT / "runs/test/EXP004_raw_tiled/metrics/official_metrics.json"),
        ("EXP004 control", ROOT / "runs/test/EXP004_modality_control/metrics/official_metrics.json"),
        ("EXP004 gray027", ROOT / "runs/test/EXP004_modality_gray027/metrics/official_metrics.json"),
    ]
    labels, recalls, fdrs, times = [], [], [], []
    for label, path in specs:
        data = read_json(path)
        labels.append(label)
        recalls.append(data["group_mean"]["recall"])
        fdrs.append(data["group_mean"]["fdr"])
        times.append(data["timing"]["max_image_seconds"])
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    bars = axes[0].bar(x, recalls, color="#2f6f9f")
    axes[0].axhline(0.85, color="#b23a48", linestyle="--", label="gate 0.85")
    axes[0].set_title("Group-mean Recall")
    axes[0].set_ylim(0.65, 1.0)
    axes[1].bar(x, fdrs, color="#d47f27")
    axes[1].axhline(0.20, color="#b23a48", linestyle="--", label="gate 0.20")
    axes[1].set_title("Group-mean FDR")
    axes[1].set_ylim(0, 0.25)
    axes[2].bar(x, times, color="#4c956c")
    axes[2].axhline(20, color="#b23a48", linestyle="--", label="gate 20 s")
    axes[2].set_title("Maximum internal image time")
    axes[2].set_ylim(0, 20)
    for ax in axes:
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.legend(frameon=False, loc="upper right")
    axes[0].set_ylabel("Metric value")
    for bar, value in zip(bars, recalls):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.008, f"{value:.3f}", ha="center", fontsize=8)
    fig.suptitle("Internal validation hard-metric comparison", fontsize=15, fontweight="bold")
    fig.subplots_adjust(top=0.82, bottom=0.29, wspace=0.25)
    fig.text(0.5, 0.01, "Recall/FDR use group-mean values; time panel shows maximum per-image time as a conservative diagnostic. All are internal proxies.", ha="center", fontsize=9, color="#555555")
    save(fig, out / "local_validation_gate_comparison.png")


def plot_official_submissions(out: Path):
    rows = [r for r in read_csv(ROOT / "shiyan/experiments/registry/submission_registry.csv") if r["tag"] in {"v1.0", "v2.0"}]
    labels = [f"{r['tag']}\nsubmission {r['submission_id']}" for r in rows]
    metrics = {
        "ship Recall": [float(r["ship_recall"]) for r in rows],
        "aircraft Recall": [float(r["aircraft_recall"]) for r in rows],
        "vehicle Recall": [float(r["vehicle_recall"]) for r in rows],
        "ship FDR": [float(r["ship_false_detection_rate"]) for r in rows],
        "aircraft FDR": [float(r["aircraft_false_detection_rate"]) for r in rows],
        "vehicle FDR": [float(r["vehicle_false_detection_rate"]) for r in rows],
    }
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    width = 0.23
    for i, (name, values) in enumerate(list(metrics.items())[:3]):
        bars = axes[0].bar(x + (i - 1) * width, values, width, label=name, color=list(GROUP_COLORS.values())[i])
        for bar, value in zip(bars, values):
            axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center", fontsize=8)
    for i, (name, values) in enumerate(list(metrics.items())[3:]):
        bars = axes[1].bar(x + (i - 1) * width, values, width, label=name, color=list(GROUP_COLORS.values())[i])
        for bar, value in zip(bars, values):
            axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}", ha="center", fontsize=8)
    axes[0].axhline(0.85, color="#b23a48", linestyle="--", linewidth=1.2, label="Recall gate 0.85")
    axes[1].axhline(0.20, color="#b23a48", linestyle="--", linewidth=1.2, label="FDR gate 0.20")
    axes[0].set_title("Formal hidden-test Recall")
    axes[1].set_title("Formal hidden-test FDR")
    axes[0].set_ylim(0, 1.12)
    axes[1].set_ylim(0, 0.62)
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.legend(frameon=False, loc="upper right")
        ax.set_ylabel("Reported value")
    fig.suptitle("Official submission comparison", fontsize=15, fontweight="bold")
    fig.subplots_adjust(top=0.82, bottom=0.18, wspace=0.20)
    fig.text(0.5, 0.01, "Source: submission_registry.csv; values reported by the competition platform, not local estimates.", ha="center", fontsize=9, color="#555555")
    save(fig, out / "official_hidden_test_comparison.png")


def plot_class_metrics(out: Path):
    data = read_json(ROOT / "runs/test/EXP004_modality_control/metrics/official_metrics.json")
    rows = data["per_class"]
    labels = [r["category_name"] for r in rows]
    recall = [np.nan if r["recall"] is None else r["recall"] for r in rows]
    fdr = [np.nan if r["fdr"] is None else r["fdr"] for r in rows]
    colors = [GROUP_COLORS[r["group"]] for r in rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    axes[0].bar(x, recall, color=colors)
    axes[0].axhline(0.85, color="#b23a48", linestyle="--", linewidth=1.2)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Recall")
    axes[0].set_title("EXP004 control per-class metrics")
    axes[1].bar(x, fdr, color=colors)
    axes[1].axhline(0.20, color="#b23a48", linestyle="--", linewidth=1.2)
    axes[1].set_ylim(0, 0.42)
    axes[1].set_ylabel("FDR")
    axes[1].set_xticks(x, labels, rotation=65, ha="right")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLORS.values()]
    axes[0].legend(handles, GROUP_COLORS.keys(), frameon=False, ncol=3, loc="lower right")
    fig.subplots_adjust(top=0.90, bottom=0.34, hspace=0.30)
    fig.text(0.5, 0.01, "Class-level metrics from the EXP004 control internal validation run; blank classes are omitted by the evaluator.", ha="center", fontsize=9, color="#555555")
    save(fig, out / "EXP004_control_class_metrics.png")


def plot_error_counts(out: Path):
    rows = read_csv(ROOT / "runs/test/EXP004_modality_gray027/error_analysis/error_per_class.csv")
    rows = [r for r in rows if int(r["false_positive"]) + int(r["false_negative"]) > 0]
    rows.sort(key=lambda r: int(r["false_positive"]) + int(r["false_negative"]), reverse=True)
    rows = rows[:12]
    labels = [r["category_name"] for r in rows]
    fp = np.array([int(r["false_positive"]) for r in rows])
    fn = np.array([int(r["false_negative"]) for r in rows])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x, fp, label="False positives", color="#d47f27")
    ax.bar(x, fn, bottom=fp, label="False negatives", color="#2f6f9f")
    ax.set_title("EXP004 gray027 error distribution by class")
    ax.set_ylabel("Count")
    ax.set_xticks(x, labels, rotation=50, ha="right")
    ax.legend(frameon=False)
    fig.subplots_adjust(top=0.88, bottom=0.30)
    fig.text(0.5, 0.01, "Top 12 classes ranked by FP + FN; source: error_per_class.csv.", ha="center", fontsize=9, color="#555555")
    save(fig, out / "EXP004_error_analysis_FP_FN.png")


def plot_data_distribution(out: Path):
    rows = read_csv(ROOT / "shiyan/data_registry/split_assignments/v1_scene_80_20/class_distribution_by_split.csv")
    labels = [r["class_name"] for r in rows]
    train = np.array([int(r["train_objects"]) for r in rows])
    val = np.array([int(r["val_objects"]) for r in rows])
    colors = [GROUP_COLORS[r["group"]] for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(15, 5.5))
    ax.bar(x, train, label="Train objects", color="#9bbbd4")
    ax.bar(x, val, bottom=train, label="Validation objects", color="#d47f27")
    ax.set_title("Dataset object distribution by class")
    ax.set_ylabel("Object count")
    ax.set_xticks(x, labels, rotation=65, ha="right")
    ax.legend(frameon=False)
    fig.subplots_adjust(top=0.88, bottom=0.30)
    fig.text(0.5, 0.01, "Source: frozen v1_scene_80_20 class distribution; colors on labels indicate ship / aircraft / vehicle groups.", ha="center", fontsize=9, color="#555555")
    save(fig, out / "dataset_object_distribution.png")


def plot_modality(out: Path):
    data = read_json(ROOT / "runs/test/EXP003_v2_modality_gray027/modality_summary.json")
    counts = data["modality_counts"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    labels = ["Grayscale", "Color"]
    values = [counts["grayscale"], counts["color"]]
    bars = ax.bar(labels, values, color=["#555555", "#4c956c"])
    ax.set_title("Validation image modality composition")
    ax.set_ylabel("Image count")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 8, str(value), ha="center", fontweight="bold")
    ax.set_ylim(0, max(values) * 1.18)
    fig.text(0.5, 0.01, "Source: EXP003 modality_summary.json; 897-image internal validation split.", ha="center", fontsize=9, color="#555555")
    save(fig, out / "validation_modality_split.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="report_figures")
    args = parser.parse_args()
    out = (ROOT / args.output_dir).resolve()
    setup()
    plot_training_comparison(out / "03_metric_figures")
    plot_local_gates(out / "03_metric_figures")
    plot_official_submissions(out / "03_metric_figures")
    plot_class_metrics(out / "04_error_and_data_figures")
    plot_error_counts(out / "04_error_and_data_figures")
    plot_data_distribution(out / "04_error_and_data_figures")
    plot_modality(out / "04_error_and_data_figures")
    print(f"generated={out}")


if __name__ == "__main__":
    main()

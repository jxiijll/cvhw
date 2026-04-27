#!/usr/bin/env python3
"""
report.py - Generate HW3 report material and validation plots
=============================================================

This script collects training logs and ensemble validation metrics, then creates
figures and a CSV summary for the required additional experiment.

Example:
    python report.py \
        --resnet_log outputs_resnet101/logs \
        --convnext_log outputs_convnextv2/logs \
        --ensemble_eval outputs_ensemble/ensemble_eval.json \
        --out_dir report_outputs
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


METRIC_COLUMNS = ["AP", "AP50", "AP75"]


def find_latest_training_log(path):
    """Return the newest training_log_*.csv under a file or directory path."""
    path = Path(path)
    if path.is_file():
        return path
    candidates = sorted(path.glob("training_log_*.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No training_log_*.csv found in {path}")
    return candidates[-1]


def read_training_log(path):
    """Read one training CSV and coerce numeric values."""
    csv_path = find_latest_training_log(path)
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean = {}
            for key, value in row.items():
                if key == "Epoch":
                    clean[key] = int(float(value))
                else:
                    clean[key] = float(value) if value not in ("", None) else 0.0
            rows.append(clean)
    if not rows:
        raise ValueError(f"Empty training log: {csv_path}")
    return csv_path, rows


def best_metrics(rows):
    """Return the row with the best AP among validation rows."""
    val_rows = [row for row in rows if row.get("AP", 0.0) > 0.0]
    if not val_rows:
        val_rows = rows
    return max(val_rows, key=lambda row: row.get("AP", 0.0))


def plot_training_curves(logs, out_dir):
    """Create training-loss and validation-metric comparison plots."""
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    for label, rows in logs.items():
        plt.plot(
            [row["Epoch"] for row in rows],
            [row["Train_Loss"] for row in rows],
            marker="o",
            linewidth=2,
            label=label,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    loss_path = out_dir / "training_loss_comparison.png"
    plt.tight_layout()
    plt.savefig(loss_path, dpi=200)
    plt.close()

    for metric in METRIC_COLUMNS:
        plt.figure(figsize=(9, 5))
        has_values = False
        for label, rows in logs.items():
            val_rows = [row for row in rows if row.get(metric, 0.0) > 0.0]
            if not val_rows:
                continue
            has_values = True
            plt.plot(
                [row["Epoch"] for row in val_rows],
                [row[metric] for row in val_rows],
                marker="o",
                linewidth=2,
                label=label,
            )
        if has_values:
            plt.xlabel("Epoch")
            plt.ylabel(metric)
            plt.title(f"Validation {metric} Comparison")
            plt.grid(True, alpha=0.3)
            plt.legend()
            metric_path = out_dir / f"validation_{metric.lower()}_comparison.png"
            plt.tight_layout()
            plt.savefig(metric_path, dpi=200)
        plt.close()


def plot_final_bar(results, out_dir):
    """Create a grouped bar chart for final AP/AP50/AP75."""
    labels = list(results.keys())
    x = list(range(len(labels)))
    width = 0.24
    colors = ["#2f6f9f", "#c85a3a", "#5f8d4e"]

    plt.figure(figsize=(9, 5))
    for offset, metric in enumerate(METRIC_COLUMNS):
        values = [results[label].get(metric, 0.0) for label in labels]
        xs = [pos + (offset - 1) * width for pos in x]
        plt.bar(xs, values, width=width, label=metric, color=colors[offset])

    plt.xticks(x, labels)
    plt.ylabel("Score")
    metric_values = [
        metrics.get(metric, 0.0)
        for metrics in results.values()
        for metric in METRIC_COLUMNS
    ]
    ymax = max(0.05, min(1.0, max(metric_values) + 0.1))
    plt.ylim(0, ymax)
    plt.title("Backbone and Ensemble Validation Results")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    out_path = out_dir / "final_validation_bar.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def load_ensemble_metrics(path):
    """Load ensemble validation metrics saved by ensemble.py."""
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_results_csv(results, out_dir):
    """Save the summarized comparison table."""
    out_path = out_dir / "experiment_summary.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", *METRIC_COLUMNS])
        for label, metrics in results.items():
            writer.writerow([label, *[f"{metrics.get(metric, 0.0):.4f}" for metric in METRIC_COLUMNS]])
    return out_path


def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate HW3 validation plots and summary CSV")
    parser.add_argument("--resnet_log", type=str, required=True, help="ResNet101 logs dir or CSV")
    parser.add_argument("--convnext_log", type=str, required=True, help="ConvNeXtV2 logs dir or CSV")
    parser.add_argument("--ensemble_eval", type=str, default=None, help="ensemble_eval.json from ensemble.py")
    parser.add_argument("--out_dir", type=str, default="report_outputs")
    return parser.parse_args()


def main():
    """Entry point."""
    args = get_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    resnet_csv, resnet_rows = read_training_log(args.resnet_log)
    convnext_csv, convnext_rows = read_training_log(args.convnext_log)
    logs = {"ResNet101": resnet_rows, "ConvNeXtV2": convnext_rows}

    results = {
        "ResNet101": best_metrics(resnet_rows),
        "ConvNeXtV2": best_metrics(convnext_rows),
    }
    ensemble_metrics = load_ensemble_metrics(args.ensemble_eval)
    if ensemble_metrics:
        results["Ensemble"] = ensemble_metrics

    plot_training_curves(logs, out_dir)
    bar_path = plot_final_bar(results, out_dir)
    csv_path = write_results_csv(results, out_dir)

    print(f"Read ResNet101 log: {resnet_csv}")
    print(f"Read ConvNeXtV2 log: {convnext_csv}")
    print(f"Saved final comparison figure: {bar_path}")
    print(f"Saved summary table: {csv_path}")


if __name__ == "__main__":
    main()

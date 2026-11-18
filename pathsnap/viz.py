"""Figures for the model comparison. Every figure is drawn from saved result tables only."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def _finish(fig, ax, out: Path) -> None:
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_cv_macro_f1(folds: pd.DataFrame, out: Path) -> None:
    g = folds.groupby("model")["macro_f1"].agg(["mean", "std"]).sort_values("mean", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(g.index, g["mean"], yerr=g["std"].fillna(0), capsize=4, color="#3b6ea5")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro F1 (mean ± std over folds)")
    ax.set_title("Cross-validation Macro F1 by model (training trips, group CV)")
    _finish(fig, ax, out)


def plot_student_comparison(summary: pd.DataFrame, out: Path) -> None:
    top1_col = "student_top1_acc" if "student_top1_acc" in summary else "student_top1"
    s = summary.set_index("model").sort_values(top1_col, ascending=False)
    x = range(len(s))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar([i - w / 2 for i in x], s[top1_col], w, label="Top-1 accuracy", color="#3b6ea5")
    ax.bar([i + w / 2 for i in x], s["student_macro_f1"], w, label="Macro F1", color="#e08e45")
    ax.set_xticks(list(x), s.index)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Student-test comparison (10 trips, scored once)")
    ax.legend()
    _finish(fig, ax, out)


def plot_log_loss(summary: pd.DataFrame, out: Path) -> None:
    s = summary.set_index("model")[["cv_log_loss_mean", "student_log_loss"]]
    s = s.sort_values("student_log_loss")
    x = range(len(s))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar([i - w / 2 for i in x], s["cv_log_loss_mean"], w, label="CV (mean)", color="#3b6ea5")
    ax.bar([i + w / 2 for i in x], s["student_log_loss"], w, label="Student test", color="#e08e45")
    ax.set_xticks(list(x), s.index)
    ax.set_ylabel("Log loss (lower is better)")
    ax.set_title("Probability quality (log loss). Read with the calibration caveat in the report.")
    ax.legend()
    _finish(fig, ax, out)

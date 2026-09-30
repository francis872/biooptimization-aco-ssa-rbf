"""Visualization helpers for GA experiments and result reporting."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix


def save_convergence_plot(
    history: pd.DataFrame,
    output_path: str | Path,
    title: str,
) -> None:
    """Plot best and average fitness over generations."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(history["generation"], history["best_fitness"], label="Best fitness", marker="o")
    plt.plot(history["generation"], history["average_fitness"], label="Average fitness", marker="s")
    if "population_diversity" in history.columns:
        ax2 = plt.gca().twinx()
        ax2.plot(history["generation"], history["population_diversity"], label="Diversity", color="gray", linestyle="--")
        ax2.set_ylabel("Diversity")
        lines1, labels1 = plt.gca().get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines1 + lines2, labels1 + labels2)
    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_confusion_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray, output_path: str | Path) -> None:
    """Save a confusion matrix for classification results."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_cluster_results(
    X: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: str | Path,
    colors: list[str] | None = None,
) -> None:
    """Use PCA to project 5D data to 2D for cluster plots."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    plt.figure(figsize=(7, 6))
    if colors is None:
        colors = ["tab:blue", "tab:orange", "tab:green"]
    for cluster_id in np.unique(labels):
        idx = labels == cluster_id
        plt.scatter(X_2d[idx, 0], X_2d[idx, 1], s=30, c=colors[int(cluster_id) % len(colors)], label=f"Cluster {cluster_id}")
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_real_classes(
    X: np.ndarray,
    y_true: np.ndarray,
    output_path: str | Path,
) -> None:
    """Plot the real class assignments using PCA projection."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    unique_labels = np.unique(y_true)
    for label_val in unique_labels:
        idx = y_true == label_val
        plt.scatter(X_2d[idx, 0], X_2d[idx, 1], s=30, label=str(label_val))
    plt.title("Real class labels")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

#!/usr/bin/env python
"""Plot predictive versus dominant dimensionality for between/within-area mappings."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

COLORS = {"predictive_rrr": "#b2182b", "dominant_pca": "#2166ac"}
LABELS = {"predictive_rrr": "Predictive RRR", "dominant_pca": "Dominant PCA"}


def curve_panel(ax, curves, mapping_type, source, target, title):
    query = curves[
        (curves["mapping_type"] == mapping_type)
        & (curves["source_area"] == source)
        & (curves["target_area"] == target)
    ]
    stats = (
        query.groupby(["model", "dimension"])["ridge_r2"]
        .agg(["mean", "sem"])
        .reset_index()
    )
    for model in ("predictive_rrr", "dominant_pca"):
        values = stats[stats["model"] == model]
        ax.plot(
            values["dimension"], values["mean"],
            color=COLORS[model], label=LABELS[model],
        )
        ax.fill_between(
            values["dimension"],
            values["mean"] - values["sem"],
            values["mean"] + values["sem"],
            color=COLORS[model], alpha=0.2,
        )
    ax.axhline(0, color="0.65", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Number of dimensions")
    ax.set_ylabel("Held-out R2")


def rank_panel(ax, selections, areas):
    area_a, area_b = areas
    comparisons = [(area_a, area_b), (area_b, area_a)]
    for x, (source, target) in enumerate(comparisons):
        between = (
            selections[
                (selections["mapping_type"] == "between")
                & (selections["source_area"] == source)
                & (selections["target_area"] == target)
            ]
            .groupby("sample")["predictive_dimension"].mean()
        )
        within = (
            selections[
                (selections["mapping_type"] == "within")
                & (selections["source_area"] == source)
                & (selections["target_area"] == source)
            ]
            .groupby("sample")["predictive_dimension"].mean()
        )
        for cross_rank, within_rank in zip(between, within):
            ax.plot(
                [x - 0.12, x + 0.12], [cross_rank, within_rank],
                color="0.75", linewidth=1,
            )
        ax.scatter(
            [x - 0.12] * len(between), between,
            color=COLORS["predictive_rrr"],
            label="Between areas" if x == 0 else None,
        )
        ax.scatter(
            [x + 0.12] * len(within), within,
            color=COLORS["dominant_pca"],
            label="Within source area" if x == 0 else None,
        )
    ax.set_xticks([0, 1], [f"{area_a} source", f"{area_b} source"])
    ax.set_ylabel("Inner-CV selected rank")
    ax.set_title("Between- versus within-area rank")
    ax.legend(frameon=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    curves = pd.read_csv(args.result_dir / "dimension_curves.csv")
    selections = pd.read_csv(args.result_dir / "selected_dimensions.csv")
    metadata = json.loads((args.result_dir / "run_metadata.json").read_text())
    areas = metadata.get("areas")
    if areas is None:
        first_between = selections[selections["mapping_type"] == "between"].iloc[0]
        areas = [first_between["source_area"], first_between["target_area"]]
    area_a, area_b = areas

    figure, axes = plt.subplots(2, 3, figsize=(13, 8))
    curve_panel(
        axes[0, 0], curves, "between", area_a, area_b,
        f"{area_a} to {area_b}",
    )
    curve_panel(
        axes[0, 1], curves, "between", area_b, area_a,
        f"{area_b} to {area_a}",
    )
    curve_panel(
        axes[1, 0], curves, "within", area_a, area_a,
        f"{area_a}-A to {area_a}-B",
    )
    curve_panel(
        axes[1, 1], curves, "within", area_b, area_b,
        f"{area_b}-A to {area_b}-B",
    )
    rank_panel(axes[0, 2], selections, areas)
    axes[1, 2].axis("off")
    axes[0, 0].legend(frameon=False)
    figure.suptitle(
        f"Session {metadata['session_id']}, full quiescent interval; "
        f"{metadata['n_neurons_per_population']} neurons per population"
    )
    figure.tight_layout()
    figure.savefig(
        args.result_dir / "dimension_comparison.png",
        dpi=180, bbox_inches="tight",
    )


if __name__ == "__main__":
    main()

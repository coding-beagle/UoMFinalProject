"""
Visualize sim_time vs ik_status from a robot simulation CSV log.

Usage:
    python visualize_ik_status.py <path_to_csv>
    python visualize_ik_status.py data.csv --per-trial
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


import matplotlib.pyplot as plt
import pandas as pd

# Bump up default font sizes across all plots.
plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "figure.titlesize": 19,
    }
)


def load_data(
    csv_path: Path,
    start: float | None = None,
    end: float | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Strip leading/trailing underscores and whitespace that sometimes appear
    # in column names and string cells of this log format.
    df.columns = [c.strip().strip("_") for c in df.columns]
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.strip().str.strip("_")

    required = {"sim_time", "ik_status"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"CSV is missing required columns: {missing}")

    df = df.sort_values("sim_time").reset_index(drop=True)

    # Apply sim_time window filter if requested.
    if start is not None:
        df = df[df["sim_time"] >= start]
    if end is not None:
        df = df[df["sim_time"] <= end]
    df = df.reset_index(drop=True)

    if df.empty:
        sys.exit(
            f"No rows remain after filtering sim_time to "
            f"[{start if start is not None else '-inf'}, "
            f"{end if end is not None else '+inf'}]."
        )

    return df


def encode_status(series: pd.Series) -> tuple[pd.Series, list[str]]:
    """Map ik_status strings to integer codes for plotting.

    Returns the encoded series and the ordered list of category labels,
    so the y-axis can show the original strings.
    """
    # Preferred ordering: success at top, failed at bottom, others in between.
    preferred = ["failed", "timeout", "in_progress", "success"]
    categories = [s for s in preferred if s in series.unique()]
    # Append any unexpected statuses we didn't anticipate.
    categories += [s for s in series.unique() if s not in categories]

    codes = series.map({name: i for i, name in enumerate(categories)})
    return codes, categories


def plot_overall(df: pd.DataFrame, out_path: Path) -> None:
    codes, categories = encode_status(df["ik_status"])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.step(df["sim_time"], codes, where="post", linewidth=1.5, color="#1f77b4")

    # Highlight failure regions for readability.
    if "failed" in categories:
        failed_code = categories.index("failed")
        ax.fill_between(
            df["sim_time"],
            -0.5,
            len(categories) - 0.5,
            where=(codes == failed_code),
            color="red",
            alpha=0.08,
            step="post",
        )

    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    ax.set_ylim(-0.5, len(categories) - 0.5)
    ax.set_xlabel("sim_time (s)")
    ax.set_ylabel("ik_status")
    ax.set_title("IK status over simulation time")
    ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def plot_per_trial(df: pd.DataFrame, out_path: Path) -> None:
    """One subplot per trial — useful when trials overlap in sim_time."""
    if "trial" not in df.columns:
        print("No 'trial' column found; skipping per-trial plot.")
        return

    trials = sorted(df["trial"].unique())
    fig, axes = plt.subplots(
        len(trials),
        1,
        figsize=(12, 2.0 * len(trials) + 0.5),
        sharex=True,
        squeeze=False,
    )

    _, categories = encode_status(df["ik_status"])
    cat_to_code = {name: i for i, name in enumerate(categories)}

    for ax, trial in zip(axes[:, 0], trials):
        sub = df[df["trial"] == trial]
        codes = sub["ik_status"].map(cat_to_code)
        ax.step(sub["sim_time"], codes, where="post", linewidth=1.5)
        ax.set_yticks(range(len(categories)))
        ax.set_yticklabels(categories)
        ax.set_ylim(-0.5, len(categories) - 0.5)
        ax.set_ylabel(f"trial {trial}")
        ax.grid(True, axis="x", alpha=0.3)

    axes[-1, 0].set_xlabel("sim_time (s)")
    fig.suptitle("IK status over sim_time, per trial")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def print_summary(df: pd.DataFrame) -> None:
    total = len(df)
    print(f"\nTotal samples: {total}")
    print(f"sim_time range: {df['sim_time'].min():.3f} to {df['sim_time'].max():.3f} s")
    print("\nik_status counts:")
    counts = df["ik_status"].value_counts()
    for status, n in counts.items():
        print(f"  {status:<12} {n:>6}  ({100 * n / total:5.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Path to the simulation CSV")
    parser.add_argument(
        "--start",
        type=float,
        default=None,
        help="Minimum sim_time to include (inclusive, seconds)",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help="Maximum sim_time to include (inclusive, seconds)",
    )
    parser.add_argument(
        "--per-trial",
        action="store_true",
        help="Also emit a per-trial subplot figure",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("ik_status.png"),
        help="Output PNG path for the overall plot",
    )
    args = parser.parse_args()

    df = load_data(args.csv, start=args.start, end=args.end)
    print_summary(df)
    plot_overall(df, args.out)

    if args.per_trial:
        per_trial_path = args.out.with_name(args.out.stem + "_per_trial.png")
        plot_per_trial(df, per_trial_path)

    plt.show()


if __name__ == "__main__":
    main()

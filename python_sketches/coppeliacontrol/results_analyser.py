"""
Robot Arm Experiment Analyser
──────────────────────────────
Run this script and use the file picker to select one or more CSV files (Group A).
Handles Reach, Transport, and Obstacle Transport experiment CSVs automatically.

Reach CSV columns (required):  result, duration_s, target_x, target_y, target_z
Transport CSV columns:          result, duration_s,
                                cube_x/y/z, drop_x/y/z,
                                start_x/y/z (optional),
                                dist_start_to_cube, dist_start_to_drop (optional),
                                phase_approach_s, phase_grip_s,
                                phase_carry_s, phase_place_s (optional)
Obstacle CSV columns:           all transport columns, plus:
                                total_hits, n_obstacles,
                                penalty_accumulated_s, adjusted_duration_s

Figure 1 — Overview:
  • Left:  Overlaid normal distribution curves of successful run durations
  • Right: Distribution of within-run speeds
  Group A = cool colours (blues/greens), Group B = warm colours (oranges/reds).
  Toggle button switches between per-experiment overlays and combined view.
  "Add Group B…" button loads a second set of CSVs for comparison.

Figure 2 — Comparison (shown when Group B is loaded):
  • Group A pooled vs Group B pooled for durations and speeds
  • Duration KDE overlay, box-plots, speed box-plot — all annotated with ANOVA
  • Group mean speed per move with ±1 SD band

Figure 3 — Transport Phase Analysis (press P, transport/obstacle CSVs only):
  • Stacked bar charts of mean time per phase per experiment
  • Box-plots comparing phase time distributions across groups
  • Scatter: dist_start_to_cube vs duration (task difficulty proxy)

Figure 4 — Collision Analysis (press C, obstacle CSVs only):
  • Bar chart: mean hits per trial per experiment, with ±1 SD error bars
  • Box-plot: per-trial hit count distributions, both groups side-by-side
  • Scatter: n_obstacles vs mean hits per trial (difficulty scaling)
  • Summary table: mean ± SD hits, hit rate (hits/obstacle), % zero-hit trials
"""

import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# ── Disable conflicting matplotlib default keybindings ───────────────────────
import matplotlib

matplotlib.rcParams["keymap.home"] = []
matplotlib.rcParams["keymap.back"] = []
matplotlib.rcParams["keymap.forward"] = []

# ── Colour palettes ───────────────────────────────────────────────────────────
PALETTE_A = ["#4C72B0", "#55A868", "#64B5CD", "#3A9E7A", "#2E6FA3", "#76B7B2"]
PALETTE_B = ["#DD8452", "#C44E52", "#E6A817", "#D45F86", "#C7622E", "#E8734A"]

PHASE_COLOURS = {
    "approach": "#4C72B0",
    "grip": "#55A868",
    "carry": "#E6A817",
    "place": "#C44E52",
}
PHASE_ORDER = ["approach", "grip", "carry", "place"]

BG = "#F8F9FA"

# ── Shared state ──────────────────────────────────────────────────────────────
groups: dict = {"A": [], "B": []}
group_titles: dict = {"A": "Group A", "B": "Group B"}
_comparison_fig = None


# ── CSV type detection ────────────────────────────────────────────────────────


def detect_csv_type(df: pd.DataFrame) -> str:
    """Return 'obstacle', 'transport', or 'reach' based on column presence."""
    if "total_hits" in df.columns:
        return "obstacle"
    if "cube_x" in df.columns:
        return "transport"
    return "reach"


def _has_phase_cols(df: pd.DataFrame) -> bool:
    return "phase_approach_s" in df.columns


def _has_start_cols(df: pd.DataFrame) -> bool:
    return "dist_start_to_cube" in df.columns


# ── Colour / title helpers ────────────────────────────────────────────────────


def _color_for(group: str, index: int) -> str:
    palette = PALETTE_A if group == "A" else PALETTE_B
    return palette[index % len(palette)]


def _tk_root() -> tk.Tk:
    if not hasattr(_tk_root, "_inst") or not _tk_root._inst.winfo_exists():
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        _tk_root._inst = root
    return _tk_root._inst


def _ask_title(group_key: str, default: str) -> str:
    result = [default]
    root = _tk_root()
    win = tk.Toplevel(root)
    win.title("Name this group")
    win.resizable(False, False)
    win.attributes("-topmost", True)
    win.grab_set()
    tk.Label(
        win, text="Enter a display name for this group:", font=("Helvetica", 11), pady=8
    ).pack(padx=16)
    var = tk.StringVar(value=default)
    entry = tk.Entry(win, textvariable=var, font=("Helvetica", 11), width=28)
    entry.pack(padx=16, pady=4)
    entry.select_range(0, tk.END)
    entry.focus_set()

    def _ok(event=None):
        val = var.get().strip()
        result[0] = val if val else default
        win.grab_release()
        win.destroy()

    def _cancel(event=None):
        win.grab_release()
        win.destroy()

    btn_frame = tk.Frame(win)
    btn_frame.pack(pady=10)
    tk.Button(btn_frame, text="OK", width=10, command=_ok).pack(side=tk.LEFT, padx=6)
    tk.Button(btn_frame, text="Cancel", width=10, command=_cancel).pack(
        side=tk.LEFT, padx=6
    )
    win.bind("<Return>", _ok)
    win.bind("<Escape>", _cancel)
    root.wait_window(win)
    return result[0]


# ── CSV loading ───────────────────────────────────────────────────────────────


def load_csv(path: Path):
    """Return (success_df, total_row_count, csv_type). Raises if no successful rows."""
    raw = pd.read_csv(path)
    total = len(raw)
    csv_type = detect_csv_type(raw)
    df = raw[raw["result"] == "success"].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No successful runs found in {path.name}")
    return df, total, csv_type


# ── Speed computation ─────────────────────────────────────────────────────────


def compute_speeds(df: pd.DataFrame, csv_type: str) -> np.ndarray:
    if csv_type == "reach":
        pos = df[["target_x", "target_y", "target_z"]].values
        displacements = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        durations = df["duration_s"].values[1:]
        return displacements / durations
    else:
        # obstacle and transport both have cube/drop positions
        cube = df[["cube_x", "cube_y", "cube_z"]].values
        drop = df[["drop_x", "drop_y", "drop_z"]].values
        distances = np.linalg.norm(drop - cube, axis=1)
        durations = df["duration_s"].values
        return distances / np.where(durations > 0, durations, np.nan)


def anova_summary(groups_data: list, metric: str) -> str:
    f, p = stats.f_oneway(*groups_data)
    sig = "✓ significant" if p < 0.05 else "✗ not significant"
    return f"One-way ANOVA — {metric}\nF = {f:.3f},  p = {p:.4f}  ({sig} at α=0.05)"


def _build_entry(
    name: str, df: pd.DataFrame, color: str, total_rows: int, csv_type: str
) -> dict:
    if len(df) < 2 and csv_type == "reach":
        print(
            f"  ⚠  {name}: only {len(df)} successful trial(s) — skipping speed computation"
        )
        speeds = np.array([])
    else:
        speeds = compute_speeds(df, csv_type)

    phase_splits = {}
    if csv_type in ("transport", "obstacle") and _has_phase_cols(df):
        for ph in PHASE_ORDER:
            col = f"phase_{ph}_s"
            if col in df.columns:
                phase_splits[ph] = df[col].values

    dist_start_to_cube = (
        df["dist_start_to_cube"].values if _has_start_cols(df) else np.array([])
    )
    dist_start_to_drop = (
        df["dist_start_to_drop"].values
        if "dist_start_to_drop" in df.columns
        else np.array([])
    )

    # ── Obstacle-specific columns ─────────────────────────────────────────────
    hits_per_trial = (
        df["total_hits"].values if "total_hits" in df.columns else np.array([])
    )
    n_obstacles_arr = (
        df["n_obstacles"].values if "n_obstacles" in df.columns else np.array([])
    )

    return dict(
        name=name,
        df=df,
        color=color,
        durations=df["duration_s"].values,
        speeds=speeds,
        n_success=len(df),
        n_total=total_rows,
        csv_type=csv_type,
        phase_splits=phase_splits,
        dist_start_to_cube=dist_start_to_cube,
        dist_start_to_drop=dist_start_to_drop,
        hits_per_trial=hits_per_trial,
        n_obstacles_arr=n_obstacles_arr,
    )


def _pool_group(group_key: str):
    entries = groups[group_key]
    durs = (
        np.concatenate([e["durations"] for e in entries]) if entries else np.array([])
    )
    spds_list = [e["speeds"] for e in entries if len(e["speeds"]) > 0]
    spds = np.concatenate(spds_list) if spds_list else np.array([])
    return durs, spds


def _pool_phase_splits(group_key: str) -> dict:
    result = {ph: [] for ph in PHASE_ORDER}
    for e in groups[group_key]:
        if e["csv_type"] not in ("transport", "obstacle"):
            continue
        for ph in PHASE_ORDER:
            arr = e["phase_splits"].get(ph, np.array([]))
            if len(arr):
                result[ph].extend(arr.tolist())
    return {ph: np.array(v) for ph, v in result.items()}


def _load_files_dialog(title: str) -> list:
    root = _tk_root()
    paths = filedialog.askopenfilenames(
        parent=root,
        title=title,
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    return [Path(p) for p in paths]


def _ingest_paths(paths: list, group_key: str):
    existing = {e["name"] for g in groups.values() for e in g}
    for path in paths:
        name = path.stem
        unique = name
        n = 2
        while unique in existing:
            unique = f"{name}_{n}"
            n += 1
        try:
            df, total_rows, csv_type = load_csv(path)
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
            continue
        color = _color_for(group_key, len(groups[group_key]))
        entry = _build_entry(unique, df, color, total_rows, csv_type)
        groups[group_key].append(entry)
        existing.add(unique)
        type_tag = f"[{csv_type}]"
        extra = ""
        if csv_type == "obstacle" and len(entry["hits_per_trial"]) > 0:
            extra = f"  avg hits={entry['hits_per_trial'].mean():.2f}"
        print(
            f"  ✓ Group {group_key}: {unique}  {type_tag}  "
            f"(n={len(entry['durations'])}  μ={entry['durations'].mean():.3f}s{extra})"
        )


# ── Overview figure ───────────────────────────────────────────────────────────


def build_overview_figure():
    from matplotlib.widgets import Button

    state = {"combined": False, "standardised": False, "skip_first_move": False}

    fig = plt.figure(figsize=(14, 7), num="Overview")
    fig.subplots_adjust(left=0.07, right=0.97, top=0.91, bottom=0.22, wspace=0.35)

    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax_btn_mode = fig.add_axes([0.05, 0.05, 0.14, 0.07])
    ax_btn_std = fig.add_axes([0.21, 0.05, 0.13, 0.07])
    ax_btn_skip = fig.add_axes([0.36, 0.05, 0.14, 0.07])
    ax_btn_add_b = fig.add_axes([0.52, 0.05, 0.13, 0.07])
    ax_btn_add_a = fig.add_axes([0.67, 0.05, 0.10, 0.07])
    ax_btn_ren_a = fig.add_axes([0.79, 0.05, 0.09, 0.07])
    ax_btn_ren_b = fig.add_axes([0.90, 0.05, 0.09, 0.07])

    btn_mode = Button(
        ax_btn_mode, "Switch to: Combined", color="#E8EDF7", hovercolor="#C8D4F0"
    )
    btn_std = Button(
        ax_btn_std, "Standardise: Off", color="#F0EDE8", hovercolor="#DDD5C8"
    )
    btn_skip = Button(
        ax_btn_skip, "Skip 1st Move: Off", color="#F0EDE8", hovercolor="#DDD5C8"
    )
    btn_add_b = Button(
        ax_btn_add_b, "Add Group B…", color="#FFF3E0", hovercolor="#FFE0B2"
    )
    btn_add_a = Button(ax_btn_add_a, "Add to A…", color="#E3F2FD", hovercolor="#BBDEFB")
    btn_ren_a = Button(ax_btn_ren_a, "Rename A", color="#F3E5F5", hovercolor="#E1BEE7")
    btn_ren_b = Button(ax_btn_ren_b, "Rename B", color="#FCE4EC", hovercolor="#F8BBD0")

    _btn_axes = [
        ax_btn_mode,
        ax_btn_std,
        ax_btn_skip,
        ax_btn_add_b,
        ax_btn_add_a,
        ax_btn_ren_a,
        ax_btn_ren_b,
    ]
    fig._buttons = [
        btn_mode,
        btn_std,
        btn_skip,
        btn_add_b,
        btn_add_a,
        btn_ren_a,
        btn_ren_b,
    ]
    for b in fig._buttons:
        b.label.set_fontsize(9)

    def _maybe_std(data):
        mu, sigma = data.mean(), data.std()
        if state["standardised"] and sigma > 0:
            return (data - mu) / sigma, mu, sigma
        return data, mu, sigma

    def _draw_normal(ax, data, color, label, linestyle="-", add_hist=False):
        if data is None or len(data) == 0:
            return
        plot_data, mu, sigma = _maybe_std(data)
        p_mu, p_sigma = plot_data.mean(), plot_data.std()
        if p_sigma == 0 or not np.isfinite(p_sigma):
            ax.axvline(
                p_mu,
                color=color,
                linewidth=2.2,
                linestyle="--",
                label=f"{label}  μ={mu:.3f}  (no spread)",
            )
            return
        pad = max(p_sigma * 2, abs(p_mu) * 0.1, 1e-6)
        x = np.linspace(p_mu - pad * 3, p_mu + pad * 3, 300)
        if add_hist:
            bins = max(6, len(plot_data) // 3)
            ax.hist(
                plot_data,
                bins=bins,
                density=True,
                color=color,
                alpha=0.18,
                edgecolor="white",
                linewidth=0.8,
            )
        ax.plot(
            x,
            stats.norm.pdf(x, p_mu, p_sigma),
            color=color,
            linewidth=2.2,
            linestyle=linestyle,
            label=f"{label}  μ={mu:.3f}  σ={sigma:.3f}",
        )
        ax.axvline(p_mu, color=color, linestyle=":", linewidth=1.2, alpha=0.6)

    def _style(a, xlabel, title):
        if state["standardised"]:
            xlabel = f"{xlabel}  [z-score]"
        a.set_xlabel(xlabel, fontsize=20)
        a.set_ylabel("Probability Density", fontsize=20)
        a.set_title(title, fontsize=20, fontweight="bold")
        n_total = sum(len(groups[k]) for k in ("A", "B"))
        if state["combined"]:
            a.legend(fontsize=15, loc="upper right", frameon=True, framealpha=0.9)
        else:
            a.legend(
                fontsize=15,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.18),
                ncol=max(1, min(3, n_total // 2 + 1)),
                frameon=True,
                framealpha=0.9,
            )
        a.grid(axis="y", linestyle="--", alpha=0.4)
        a.spines[["top", "right"]].set_visible(False)
        a.set_facecolor(BG)

    def _active(arr, csv_type="reach"):
        if state["skip_first_move"] and csv_type == "reach" and len(arr) > 1:
            return arr[1:]
        return arr

    def redraw():
        ax1.cla()
        ax2.cla()

        if state["combined"]:
            for gk in ("A", "B"):
                if not groups[gk]:
                    continue
                col = groups[gk][0]["color"]
                lbl = group_titles[gk]
                durs, spds = _pool_group(gk)
                _draw_normal(ax1, durs, col, lbl, add_hist=True)
                _draw_normal(ax2, spds, col, lbl, add_hist=True)
            subtitle = "Combined (per group)"
        else:
            for gk in ("A", "B"):
                ls = "-" if gk == "A" else "--"
                for e in groups[gk]:
                    _draw_normal(
                        ax1,
                        _active(e["durations"], e["csv_type"]),
                        e["color"],
                        e["name"],
                        linestyle=ls,
                    )
                    _draw_normal(
                        ax2,
                        _active(e["speeds"], e["csv_type"]),
                        e["color"],
                        e["name"],
                        linestyle=ls,
                    )
            subtitle = "Per Experiment"

        all_types = {e["csv_type"] for g in groups.values() for e in g}
        if all_types <= {"transport", "obstacle"} and "reach" not in all_types:
            spd_label = "Task Speed (cube→drop dist / duration, m/s)"
        elif "reach" in all_types and len(all_types) == 1:
            spd_label = "Speed (m / s)"
        else:
            spd_label = "Speed (units / s  or  m/s)"

        dur_xl = "Duration (s)" + (
            "  [1st excluded]" if state["skip_first_move"] else ""
        )
        _style(ax1, dur_xl, "Distribution of Run Durations")
        _style(ax2, spd_label, "Distribution of Within-Run Speeds")

        tags = (["standardised"] if state["standardised"] else []) + (
            ["1st excluded"] if state["skip_first_move"] else []
        )
        tag_str = f"  [{', '.join(tags)}]" if tags else ""
        n_a, n_b = len(groups["A"]), len(groups["B"])
        ta, tb = group_titles["A"], group_titles["B"]
        group_str = f"{ta}: {n_a} exp." + (
            f"  |  {tb}: {n_b} exp." if n_b else "  (add Group B to compare)"
        )

        # Append collision summary to suptitle if any obstacle data present
        obs_entries = [
            e for g in groups.values() for e in g if e["csv_type"] == "obstacle"
        ]
        if obs_entries:
            all_hits = np.concatenate(
                [
                    e["hits_per_trial"]
                    for e in obs_entries
                    if len(e["hits_per_trial"]) > 0
                ]
            )
            if len(all_hits):
                group_str += (
                    f"  |  collisions: μ={all_hits.mean():.2f} / trial  (press C)"
                )

        # fig.suptitle(
        #     f"Robot Arm Experiments — {subtitle}{tag_str}\n{group_str}",
        #     fontsize=13,
        #     fontweight="bold",
        # )
        fig.canvas.draw_idle()

    fig._redraw = redraw

    def on_toggle_mode(event):
        state["combined"] = not state["combined"]
        btn_mode.label.set_text(
            "Switch to: Per Exp." if state["combined"] else "Switch to: Combined"
        )
        redraw()

    def on_toggle_std(event):
        state["standardised"] = not state["standardised"]
        btn_std.label.set_text(
            "Standardise: On" if state["standardised"] else "Standardise: Off"
        )
        btn_std.color = "#D8F0D8" if state["standardised"] else "#F0EDE8"
        btn_std.hovercolor = "#B8E0B8" if state["standardised"] else "#DDD5C8"
        redraw()

    def on_toggle_skip(event):
        state["skip_first_move"] = not state["skip_first_move"]
        btn_skip.label.set_text(
            "Skip 1st Move: On" if state["skip_first_move"] else "Skip 1st Move: Off"
        )
        btn_skip.color = "#D8F0D8" if state["skip_first_move"] else "#F0EDE8"
        btn_skip.hovercolor = "#B8E0B8" if state["skip_first_move"] else "#DDD5C8"
        redraw()

    def _add_and_refresh(group_key: str, title: str):
        global _comparison_fig
        paths = _load_files_dialog(title)
        if not paths:
            return
        if not groups[group_key]:
            default = "Group A" if group_key == "A" else "Group B"
            group_titles[group_key] = _ask_title(group_key, default)
        _ingest_paths(paths, group_key)
        redraw()
        if groups["A"] and groups["B"]:
            if _comparison_fig is not None:
                try:
                    plt.close(_comparison_fig)
                except Exception:
                    pass
            _comparison_fig = build_comparison_figure()
            _comparison_fig.canvas.draw_idle()
            _print_anova()

    def _rename_group(group_key: str):
        global _comparison_fig
        current = group_titles[group_key]
        new_title = _ask_title(group_key, current)
        if new_title == current:
            return
        group_titles[group_key] = new_title
        redraw()
        if groups["A"] and groups["B"] and _comparison_fig is not None:
            try:
                plt.close(_comparison_fig)
            except Exception:
                pass
            _comparison_fig = build_comparison_figure()
            _comparison_fig.canvas.draw_idle()

    btn_mode.on_clicked(on_toggle_mode)
    btn_std.on_clicked(on_toggle_std)
    btn_skip.on_clicked(on_toggle_skip)
    btn_add_b.on_clicked(lambda e: _add_and_refresh("B", "Select Group B CSV file(s)"))
    btn_add_a.on_clicked(
        lambda e: _add_and_refresh("A", "Add more Group A CSV file(s)")
    )
    btn_ren_a.on_clicked(lambda e: _rename_group("A"))
    btn_ren_b.on_clicked(lambda e: _rename_group("B"))

    def _refocus(event=None):
        fig.canvas.get_tk_widget().focus_set()

    for _b in fig._buttons:
        _b.on_clicked(_refocus)

    state["btns_visible"] = True

    def _toggle_buttons(event=None):
        visible = not state["btns_visible"]
        state["btns_visible"] = visible
        for ax in _btn_axes:
            ax.set_visible(visible)
        fig.subplots_adjust(bottom=0.22 if visible else 0.05)
        fig.canvas.draw_idle()

    _metrics_state = {"fig": None}
    _phase_state = {"fig": None}
    _collision_state = {"fig": None}

    def _toggle_metrics(event=None):
        existing = _metrics_state["fig"]
        if existing is not None and plt.fignum_exists(existing.number):
            plt.close(existing)
            _metrics_state["fig"] = None
        else:
            mfig = build_metrics_figure()
            _metrics_state["fig"] = mfig
            mfig.canvas.manager.show()
            mfig.canvas.draw_idle()

    def _toggle_phase(event=None):
        existing = _phase_state["fig"]
        if existing is not None and plt.fignum_exists(existing.number):
            plt.close(existing)
            _phase_state["fig"] = None
        else:
            transport_entries = [
                e
                for g in groups.values()
                for e in g
                if e["csv_type"] in ("transport", "obstacle")
            ]
            if not transport_entries:
                messagebox.showinfo(
                    "No transport data",
                    "Load at least one transport or obstacle CSV first.",
                )
                return
            pfig = build_phase_figure()
            _phase_state["fig"] = pfig
            pfig.canvas.manager.show()
            pfig.canvas.draw_idle()

    def _toggle_collision(event=None):
        existing = _collision_state["fig"]
        if existing is not None and plt.fignum_exists(existing.number):
            plt.close(existing)
            _collision_state["fig"] = None
        else:
            obs_entries = [
                e for g in groups.values() for e in g if e["csv_type"] == "obstacle"
            ]
            if not obs_entries:
                messagebox.showinfo(
                    "No obstacle data",
                    "Load at least one obstacle transport CSV first.",
                )
                return
            cfig = build_collision_figure()
            _collision_state["fig"] = cfig
            cfig.canvas.manager.show()
            cfig.canvas.draw_idle()

    def _on_key(event):
        if event.key == "h":
            _toggle_buttons()
        elif event.key == "m":
            _toggle_metrics()
        elif event.key == "p":
            _toggle_phase()
        elif event.key == "c":
            _toggle_collision()

    fig.canvas.mpl_connect("key_press_event", _on_key)
    redraw()
    return fig


# ── Comparison figure ─────────────────────────────────────────────────────────


def build_comparison_figure():
    dur_a, spd_a = _pool_group("A")
    dur_b, spd_b = _pool_group("B")

    n_a_exp, n_b_exp = len(groups["A"]), len(groups["B"])
    ta, tb = group_titles["A"], group_titles["B"]
    label_a = f"{ta}  ({n_a_exp} exp, n={len(dur_a)})"
    label_b = f"{tb}  ({n_b_exp} exp, n={len(dur_b)})"
    col_a, col_b = PALETTE_A[0], PALETTE_B[0]

    fig = plt.figure(figsize=(16, 14), num="Comparison")
    fig.suptitle(f"{ta} vs {tb} — Pooled Comparison", fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.65, wspace=0.35)

    ax_kde = fig.add_subplot(gs[0, :])
    ax_kde.set_facecolor(BG)
    for durs, col, lbl in [(dur_a, col_a, label_a), (dur_b, col_b, label_b)]:
        if len(durs) == 0:
            continue
        mu, sigma = durs.mean(), durs.std()
        x = np.linspace(max(0, durs.min() - 2), durs.max() + 2, 300)
        ax_kde.plot(
            x,
            stats.norm.pdf(x, mu, sigma),
            color=col,
            linewidth=2.5,
            label=f"{lbl}  μ={mu:.2f}s  σ={sigma:.2f}s",
        )
        ax_kde.fill_between(x, stats.norm.pdf(x, mu, sigma), alpha=0.12, color=col)
        ax_kde.axvline(mu, color=col, linestyle=":", linewidth=1.4, alpha=0.7)
    ax_kde.set_xlabel("Duration (s)", fontsize=20)
    ax_kde.set_ylabel("Probability Density", fontsize=20)
    ax_kde.set_title(
        f"Duration Distributions — {ta} vs {tb} (pooled)",
        fontsize=20,
        fontweight="bold",
    )
    ax_kde.legend(fontsize=15)
    ax_kde.grid(axis="y", linestyle="--", alpha=0.4)
    ax_kde.spines[["top", "right"]].set_visible(False)

    ax_db = fig.add_subplot(gs[1, 0])
    ax_db.set_facecolor(BG)
    if len(dur_a) and len(dur_b):
        bp1 = ax_db.boxplot(
            [dur_a, dur_b],
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
        )
        for patch, col in zip(bp1["boxes"], [col_a, col_b]):
            patch.set_facecolor(col)
            patch.set_alpha(0.65)
        ax_db.set_xticks([1, 2])
        ax_db.set_xticklabels([label_a, label_b], fontsize=20)
        ax_db.text(
            0.5,
            -0.28,
            anova_summary([dur_a, dur_b], "Duration"),
            transform=ax_db.transAxes,
            ha="center",
            fontsize=20,
            bbox=dict(boxstyle="round,pad=0.4", fc="#EEF2FF", ec=col_a, alpha=0.9),
        )
    ax_db.set_ylabel("Duration (s)", fontsize=20)
    ax_db.set_title("Duration Box-Plot", fontsize=20, fontweight="bold")
    ax_db.grid(axis="y", linestyle="--", alpha=0.4)
    ax_db.spines[["top", "right"]].set_visible(False)

    ax_sb = fig.add_subplot(gs[1, 1])
    ax_sb.set_facecolor(BG)
    if len(spd_a) > 0 and len(spd_b) > 0:
        bp2 = ax_sb.boxplot(
            [spd_a, spd_b],
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
        )
        for patch, col in zip(bp2["boxes"], [col_a, col_b]):
            patch.set_facecolor(col)
            patch.set_alpha(0.65)
        ax_sb.set_xticks([1, 2])
        ax_sb.set_xticklabels([label_a, label_b], fontsize=20)
        ax_sb.text(
            0.5,
            -0.28,
            anova_summary([spd_a, spd_b], "Speed"),
            transform=ax_sb.transAxes,
            ha="center",
            fontsize=20,
            bbox=dict(boxstyle="round,pad=0.4", fc="#FFF5EE", ec=col_b, alpha=0.9),
        )
    ax_sb.set_ylabel("Speed (m/s or units/s)", fontsize=20)
    ax_sb.set_title("Speed Box-Plot", fontsize=20, fontweight="bold")
    ax_sb.grid(axis="y", linestyle="--", alpha=0.4)
    ax_sb.spines[["top", "right"]].set_visible(False)

    ax_sl = fig.add_subplot(gs[2, :])
    ax_sl.set_facecolor(BG)
    for gk, col, lbl in [("A", col_a, ta), ("B", col_b, tb)]:
        entries = [e for e in groups[gk] if len(e["speeds"]) > 0]
        if not entries:
            continue
        max_len = max(len(e["speeds"]) for e in entries)
        matrix = np.full((len(entries), max_len), np.nan)
        for i, e in enumerate(entries):
            matrix[i, : len(e["speeds"])] = e["speeds"]
        mean_spd = np.nanmean(matrix, axis=0)
        std_spd = np.nanstd(matrix, axis=0)
        xs = np.arange(1, max_len + 1)
        ax_sl.plot(
            xs,
            mean_spd,
            color=col,
            linewidth=2.5,
            marker="o",
            markersize=6,
            markerfacecolor="white",
            markeredgecolor=col,
            markeredgewidth=2,
            label=lbl,
            zorder=3,
        )
        ax_sl.fill_between(
            xs,
            mean_spd - std_spd,
            mean_spd + std_spd,
            color=col,
            alpha=0.15,
            label=f"{lbl} ±1 SD",
        )
    ax_sl.set_xlabel("Move / Trial Index", fontsize=20)
    ax_sl.set_ylabel("Mean Speed (m/s or units/s)", fontsize=20)
    ax_sl.set_title(
        "Group Mean Speed per Move  (±1 SD band)", fontsize=20, fontweight="bold"
    )
    ax_sl.legend(fontsize=15)
    ax_sl.grid(axis="y", linestyle="--", alpha=0.4)
    ax_sl.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return fig


# ── Metrics figure ────────────────────────────────────────────────────────────


def build_metrics_figure():
    has_b = bool(groups["B"])
    n_cols = 2 if has_b else 1
    ta, tb = group_titles["A"], group_titles["B"]

    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(7 * n_cols, 9),
        num="Success Rate",
        gridspec_kw={"height_ratios": [1, 1.6]},
    )
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle("Success Rate Analysis", fontsize=14, fontweight="bold")
    fig.patch.set_facecolor(BG)

    col_info = [("A", ta, PALETTE_A[0])]
    if has_b:
        col_info.append(("B", tb, PALETTE_B[0]))

    for col_idx, (gk, title, col) in enumerate(col_info):
        ax = axes[0, col_idx]
        ax.set_facecolor(BG)
        entries = groups[gk]
        total = sum(e["n_total"] for e in entries)
        success = sum(e["n_success"] for e in entries)
        failure = total - success
        rate = success / total * 100 if total else 0

        _, _, autotexts = ax.pie(
            [success, failure],
            labels=["Success", "Failure"],
            colors=[col, "#D9D9D9"],
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=2),
            textprops=dict(fontsize=10),
        )
        for at in autotexts:
            at.set_fontsize(11)
            at.set_fontweight("bold")
        ax.set_title(
            f"{title}\n{success}/{total} successful  ({rate:.1f}%)",
            fontsize=20,
            fontweight="bold",
            pad=12,
        )

    return fig


# ── Phase analysis figure ─────────────────────────────────────────────────────


def build_phase_figure():
    ta, tb = group_titles["A"], group_titles["B"]

    pie_specs = []
    for gk, label in [("A", ta), ("B", tb)]:
        t_entries = [
            e for e in groups[gk] if e["csv_type"] in ("transport", "obstacle")
        ]
        if not t_entries:
            continue
        splits = _pool_phase_splits(gk)
        means = {
            ph: splits[ph].mean() if len(splits[ph]) > 0 else 0.0 for ph in PHASE_ORDER
        }
        total = sum(means.values())
        if total > 0:
            pie_specs.append((gk, label, means, total))

    n_pies = len(pie_specs)

    all_transport = []
    for gk in ("A", "B"):
        for e in groups[gk]:
            if e["csv_type"] in ("transport", "obstacle"):
                all_transport.append((gk, e))

    fig = plt.figure(figsize=(14, 12), num="Phase Analysis")
    fig.suptitle("Transport Phase Analysis", fontsize=14, fontweight="bold")
    fig.patch.set_facecolor(BG)

    BOTTOM_H = 0.30
    BOTTOM_PAD = 0.06

    ax_box = fig.add_axes([BOTTOM_PAD, 0.07, 0.40, BOTTOM_H])
    ax_scatter = fig.add_axes([BOTTOM_PAD + 0.50, 0.07, 0.40, BOTTOM_H])

    for ax in (ax_box, ax_scatter):
        ax.set_facecolor(BG)
        ax.spines[["top", "right"]].set_visible(False)

    if not pie_specs:
        fig.text(
            0.5,
            0.72,
            "No transport data loaded.",
            ha="center",
            va="center",
            fontsize=13,
        )
        return fig

    max_total = max(spec[3] for spec in pie_specs)
    MAX_S, MIN_S = 0.38, 0.20
    PIE_ROW_CY = 0.68

    centres_x = [0.50] if n_pies == 1 else [0.25, 0.75]
    pie_colours = [PHASE_COLOURS[ph] for ph in PHASE_ORDER]
    pie_explode = [0.03] * len(PHASE_ORDER)

    COLLISION_COLOUR = "#E05C5C"

    for pie_idx, (gk, pie_title, means, total) in enumerate(pie_specs):
        side = MIN_S + (MAX_S - MIN_S) * np.sqrt(total / max_total)
        cx = centres_x[pie_idx]
        rect = [cx - side / 2, PIE_ROW_CY - side / 2, side, side]
        ax_pie = fig.add_axes(rect, aspect="equal")
        ax_pie.set_facecolor(BG)

        values = [means[ph] for ph in PHASE_ORDER]
        slice_labels = [
            f"{ph.capitalize()}\n{means[ph]:.2f}s\n({means[ph]/total*100:.1f}%)"
            for ph in PHASE_ORDER
        ]

        # Check if this group has obstacle (collision) data
        obs_entries_gk = [
            e
            for e in groups[gk]
            if e["csv_type"] == "obstacle" and len(e["hits_per_trial"]) > 0
        ]
        has_collisions = bool(obs_entries_gk)

        if has_collisions:
            # Plain phase pie (unchanged from non-obstacle version)
            wedges, _ = ax_pie.pie(
                values,
                labels=slice_labels,
                colors=pie_colours,
                explode=pie_explode,
                startangle=90,
                wedgeprops=dict(edgecolor="white", linewidth=1.6),
                textprops=dict(fontsize=8.5),
                labeldistance=1.22,
            )
            for w in wedges:
                w.set_alpha(0.88)
            ax_pie.set_title(
                f"{pie_title}\navg trial  {total:.2f}s",
                fontsize=10,
                fontweight="bold",
                pad=24,
            )

            # Avg collisions label below the pie
            all_hits = np.concatenate([e["hits_per_trial"] for e in obs_entries_gk])
            mean_hits = all_hits.mean()
            ax_pie.text(
                0,
                -1.45,
                f"Avg collisions / trial:  {mean_hits:.2f}",
                ha="center",
                va="top",
                fontsize=20,
                color=COLLISION_COLOUR,
                fontweight="bold",
                transform=ax_pie.transData,
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    fc="white",
                    ec=COLLISION_COLOUR,
                    alpha=0.85,
                ),
            )
        else:
            # No collision data — plain phase pie as before
            wedges, _ = ax_pie.pie(
                values,
                labels=slice_labels,
                colors=pie_colours,
                explode=pie_explode,
                startangle=90,
                wedgeprops=dict(edgecolor="white", linewidth=1.6),
                textprops=dict(fontsize=8.5),
                labeldistance=1.22,
            )
            for w in wedges:
                w.set_alpha(0.88)
            ax_pie.set_title(
                f"{pie_title}\navg trial  {total:.2f}s",
                fontsize=10,
                fontweight="bold",
                pad=24,
            )

    if n_pies == 2:
        fig.text(
            0.5,
            0.43,
            "Pie area ∝ average total trial duration",
            ha="center",
            fontsize=20,
            color="#888888",
            style="italic",
        )

    has_b_transport = any(gk == "B" for gk, _ in all_transport)
    group_keys_present = ["A"] + (["B"] if has_b_transport else [])
    group_cols = {"A": PALETTE_A[0], "B": PALETTE_B[0]}
    n_phase_groups = len(group_keys_present)

    positions, box_data, tick_pos, tick_lbl = [], [], [], []
    spacing = n_phase_groups + 1
    for ph_idx, ph in enumerate(PHASE_ORDER):
        base = ph_idx * spacing
        tick_pos.append(base + (n_phase_groups - 1) / 2)
        tick_lbl.append(ph.capitalize())
        for gi, gk in enumerate(group_keys_present):
            arr = _pool_phase_splits(gk).get(ph, np.array([]))
            if len(arr) == 0:
                arr = np.array([0.0])
            positions.append(base + gi)
            box_data.append(arr)

    bp = ax_box.boxplot(
        box_data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(group_cols[group_keys_present[i % n_phase_groups]])
        patch.set_alpha(0.65)

    ax_box.set_xticks(tick_pos)
    ax_box.set_xticklabels(tick_lbl, fontsize=10)
    ax_box.set_ylabel("Phase Duration (s)", fontsize=20)
    ax_box.set_title(
        "Phase Time Distributions by Group", fontsize=20, fontweight="bold"
    )
    ax_box.grid(axis="y", linestyle="--", alpha=0.4)

    from matplotlib.patches import Patch

    ax_box.legend(
        handles=[
            Patch(facecolor=group_cols[gk], alpha=0.65, label=group_titles[gk])
            for gk in group_keys_present
        ],
        fontsize=15,
    )

    has_dist_data = any(len(e["dist_start_to_cube"]) > 0 for _, e in all_transport)
    if has_dist_data:
        for gk, col, lbl in [("A", PALETTE_A[0], ta), ("B", PALETTE_B[0], tb)]:
            entries_t = [e for g, e in all_transport if g == gk]
            if not entries_t:
                continue
            d_arr = np.concatenate(
                [
                    e["dist_start_to_cube"]
                    for e in entries_t
                    if len(e["dist_start_to_cube"]) > 0
                ]
            )
            dur_arr = np.concatenate(
                [e["durations"] for e in entries_t if len(e["dist_start_to_cube"]) > 0]
            )
            if len(d_arr) < 2:
                continue
            ax_scatter.scatter(
                d_arr * 100,
                dur_arr,
                color=col,
                alpha=0.55,
                s=35,
                label=lbl,
                edgecolors="white",
                linewidth=0.4,
            )
            slope, intercept, r, p_val, _ = stats.linregress(d_arr, dur_arr)
            x_fit = np.linspace(d_arr.min(), d_arr.max(), 100)
            ax_scatter.plot(
                x_fit * 100,
                slope * x_fit + intercept,
                color=col,
                linewidth=1.8,
                linestyle="--",
                label=f"{lbl} fit  r={r:.2f} {'✓' if p_val < 0.05 else '✗'}",
            )
        ax_scatter.set_xlabel("Distance: Start → Cube (cm)", fontsize=20)
        ax_scatter.set_ylabel("Trial Duration (s)", fontsize=20)
        ax_scatter.set_title(
            "Task Difficulty: Start Distance vs Duration",
            fontsize=20,
            fontweight="bold",
        )
        ax_scatter.legend(fontsize=15)
        ax_scatter.grid(linestyle="--", alpha=0.4)
    else:
        ax_scatter.text(
            0.5,
            0.5,
            "No start-position data available.\n(requires start_x/y/z columns in CSV)",
            ha="center",
            va="center",
            transform=ax_scatter.transAxes,
            fontsize=10,
            color="#888888",
        )
        ax_scatter.set_title(
            "Task Difficulty (no start-pos data)", fontsize=20, fontweight="bold"
        )

    return fig


# ── Collision analysis figure (obstacle CSVs, press C) ────────────────────────


def build_collision_figure():
    """
    Four-panel collision analysis for obstacle transport experiments.

    Top-left:  Bar chart — mean hits per trial per experiment, ±1 SD error bars.
               Both groups plotted side-by-side with group colour coding.
    Top-right: Box-plot — per-trial hit count distribution per experiment,
               grouped by A / B.
    Bottom-left:  Scatter — n_obstacles vs mean hits per trial, one point per
                  experiment.  Useful for comparing difficulty across configs.
    Bottom-right: Summary stats table — mean ± SD, hit rate (hits/obstacle),
                  % zero-hit trials, total trials.
    """
    ta, tb = group_titles["A"], group_titles["B"]

    # Collect obstacle entries per group
    obs = {
        "A": [e for e in groups["A"] if e["csv_type"] == "obstacle"],
        "B": [e for e in groups["B"] if e["csv_type"] == "obstacle"],
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), num="Collision Analysis")
    fig.suptitle(
        "Collision Analysis — Obstacle Transport", fontsize=14, fontweight="bold"
    )
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(
        hspace=0.45, wspace=0.35, left=0.07, right=0.97, top=0.91, bottom=0.08
    )

    ax_bar, ax_box, ax_scatter, ax_table = (
        axes[0, 0],
        axes[0, 1],
        axes[1, 0],
        axes[1, 1],
    )
    for ax in (ax_bar, ax_box, ax_scatter):
        ax.set_facecolor(BG)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax_table.set_facecolor(BG)
    ax_table.axis("off")

    group_cols = {"A": PALETTE_A[0], "B": PALETTE_B[0]}

    # ── collect all entries in display order ──────────────────────────────────
    all_entries = []  # list of (group_key, entry)
    for gk in ("A", "B"):
        for e in obs[gk]:
            all_entries.append((gk, e))

    if not all_entries:
        for ax in (ax_bar, ax_box, ax_scatter):
            ax.text(
                0.5,
                0.5,
                "No obstacle data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=20,
                color="#888888",
            )
        return fig

    # ── Top-left: bar chart of mean hits per trial ────────────────────────────
    bar_x = np.arange(len(all_entries))
    bar_means = [
        e["hits_per_trial"].mean() if len(e["hits_per_trial"]) else 0
        for _, e in all_entries
    ]
    bar_sds = [
        e["hits_per_trial"].std() if len(e["hits_per_trial"]) > 1 else 0
        for _, e in all_entries
    ]
    bar_cols = [group_cols[gk] for gk, _ in all_entries]
    bar_labels = [e["name"] for _, e in all_entries]

    bars = ax_bar.bar(
        bar_x,
        bar_means,
        color=bar_cols,
        alpha=0.72,
        edgecolor="white",
        linewidth=0.8,
        zorder=2,
    )
    ax_bar.errorbar(
        bar_x,
        bar_means,
        yerr=bar_sds,
        fmt="none",
        color="#333333",
        capsize=5,
        capthick=1.4,
        linewidth=1.4,
        zorder=3,
    )

    # Annotate each bar with its mean value
    for xi, (mean, sd) in enumerate(zip(bar_means, bar_sds)):
        ax_bar.text(
            xi,
            mean + sd + 0.05,
            f"{mean:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#333333",
        )

    ax_bar.set_xticks(bar_x)
    ax_bar.set_xticklabels(bar_labels, rotation=30, ha="right", fontsize=20)
    ax_bar.set_ylabel("Mean Hits per Trial", fontsize=20)
    ax_bar.set_title(
        "Average Collisions per Trial  (±1 SD)", fontsize=20, fontweight="bold"
    )

    # Group legend
    from matplotlib.patches import Patch

    legend_handles = []
    for gk in ("A", "B"):
        if obs[gk]:
            legend_handles.append(
                Patch(facecolor=group_cols[gk], alpha=0.72, label=group_titles[gk])
            )
    if legend_handles:
        ax_bar.legend(handles=legend_handles, fontsize=15)

    # ── Top-right: box-plot of per-trial hit distributions ────────────────────
    box_data = [
        e["hits_per_trial"] if len(e["hits_per_trial"]) else np.array([0])
        for _, e in all_entries
    ]
    box_positions = np.arange(len(all_entries))
    box_cols_list = [group_cols[gk] for gk, _ in all_entries]

    bp = ax_box.boxplot(
        box_data,
        positions=box_positions,
        widths=0.55,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
    )
    for patch, col in zip(bp["boxes"], box_cols_list):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)

    ax_box.set_xticks(box_positions)
    ax_box.set_xticklabels(bar_labels, rotation=30, ha="right", fontsize=20)
    ax_box.set_ylabel("Hits per Trial", fontsize=20)
    ax_box.set_title(
        "Hit Count Distribution per Experiment", fontsize=20, fontweight="bold"
    )

    # ── Bottom-left: scatter n_obstacles vs mean hits ─────────────────────────
    scatter_plotted = False
    for gk in ("A", "B"):
        entries_g = obs[gk]
        if not entries_g:
            continue
        xs, ys, labels_s = [], [], []
        for e in entries_g:
            if len(e["hits_per_trial"]) == 0 or len(e["n_obstacles_arr"]) == 0:
                continue
            n_obs = float(
                e["n_obstacles_arr"].mean()
            )  # typically constant per experiment
            xs.append(n_obs)
            ys.append(e["hits_per_trial"].mean())
            labels_s.append(e["name"])

        if not xs:
            continue

        ax_scatter.scatter(
            xs,
            ys,
            color=group_cols[gk],
            s=80,
            alpha=0.82,
            edgecolors="white",
            linewidth=0.8,
            label=group_titles[gk],
            zorder=3,
        )
        for xi, yi, lbl in zip(xs, ys, labels_s):
            ax_scatter.annotate(
                lbl,
                (xi, yi),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
                color="#444444",
            )
        scatter_plotted = True

    if scatter_plotted:
        # Add a y=x reference line (1 hit per obstacle = 100% hit rate)
        xlim = ax_scatter.get_xlim()
        ref_x = np.linspace(0, max(xlim[1], 1), 100)
        ax_scatter.plot(
            ref_x,
            ref_x,
            color="#aaaaaa",
            linewidth=1.2,
            linestyle=":",
            label="1 hit / obstacle",
            zorder=1,
        )
        ax_scatter.set_xlim(left=0)
        ax_scatter.set_ylim(bottom=0)
        ax_scatter.legend(fontsize=15)
    else:
        ax_scatter.text(
            0.5,
            0.5,
            "No n_obstacles data available",
            ha="center",
            va="center",
            transform=ax_scatter.transAxes,
            fontsize=10,
            color="#888888",
        )

    ax_scatter.set_xlabel("Number of Obstacles", fontsize=20)
    ax_scatter.set_ylabel("Mean Hits per Trial", fontsize=20)
    ax_scatter.set_title("Obstacle Count vs Mean Hits", fontsize=20, fontweight="bold")

    # ── Bottom-right: summary stats table ─────────────────────────────────────
    col_headers = [
        "Experiment",
        "Group",
        "Trials",
        "Mean hits",
        "SD hits",
        "Hits / obs",
        "% zero-hit",
    ]
    rows = []
    for gk, e in all_entries:
        hits = e["hits_per_trial"]
        n_obs_arr = e["n_obstacles_arr"]
        n = len(hits)
        mean = hits.mean() if n > 0 else 0.0
        sd = hits.std() if n > 1 else 0.0
        n_obs_val = n_obs_arr.mean() if len(n_obs_arr) > 0 else np.nan
        hit_rate = (
            mean / n_obs_val if np.isfinite(n_obs_val) and n_obs_val > 0 else np.nan
        )
        pct_zero = (hits == 0).mean() * 100 if n > 0 else 0.0
        rows.append(
            [
                e["name"],
                group_titles[gk],
                str(n),
                f"{mean:.2f}",
                f"{sd:.2f}",
                f"{hit_rate:.3f}" if np.isfinite(hit_rate) else "—",
                f"{pct_zero:.1f}%",
            ]
        )

    # Pooled rows per group
    for gk in ("A", "B"):
        if not obs[gk]:
            continue
        all_hits = np.concatenate(
            [e["hits_per_trial"] for e in obs[gk] if len(e["hits_per_trial"]) > 0]
        )
        all_nobs = np.concatenate(
            [e["n_obstacles_arr"] for e in obs[gk] if len(e["n_obstacles_arr"]) > 0]
        )
        n = len(all_hits)
        mean = all_hits.mean() if n > 0 else 0.0
        sd = all_hits.std() if n > 1 else 0.0
        n_obs_val = all_nobs.mean() if len(all_nobs) > 0 else np.nan
        hit_rate = (
            mean / n_obs_val if np.isfinite(n_obs_val) and n_obs_val > 0 else np.nan
        )
        pct_zero = (all_hits == 0).mean() * 100 if n > 0 else 0.0
        rows.append(
            [
                f"── {group_titles[gk]} pooled ──",
                group_titles[gk],
                str(n),
                f"{mean:.2f}",
                f"{sd:.2f}",
                f"{hit_rate:.3f}" if np.isfinite(hit_rate) else "—",
                f"{pct_zero:.1f}%",
            ]
        )

    table = ax_table.table(
        cellText=rows,
        colLabels=col_headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.55)

    # Style header row
    for j in range(len(col_headers)):
        table[0, j].set_facecolor("#DDEEFF")
        table[0, j].set_text_props(fontweight="bold")

    # Stripe body rows and colour pooled rows
    for i, (gk, _) in enumerate(all_entries):
        row_idx = i + 1
        bg = "#F0F4FF" if gk == "A" else "#FFF4F0"
        for j in range(len(col_headers)):
            table[row_idx, j].set_facecolor(bg)

    # Pooled rows get a slightly darker tint
    n_data_rows = len(all_entries)
    for pi, gk in enumerate(gk for gk in ("A", "B") if obs[gk]):
        row_idx = n_data_rows + pi + 1
        bg = "#D8E8FF" if gk == "A" else "#FFE8D8"
        for j in range(len(col_headers)):
            table[row_idx, j].set_facecolor(bg)
            table[row_idx, j].set_text_props(fontweight="bold")

    ax_table.set_title("Summary Statistics", fontsize=20, fontweight="bold", pad=12)

    return fig


def _print_anova():
    dur_a, spd_a = _pool_group("A")
    dur_b, spd_b = _pool_group("B")
    ta, tb = group_titles["A"], group_titles["B"]
    if len(dur_a) and len(dur_b):
        print("\n" + anova_summary([dur_a, dur_b], f"Duration ({ta} vs {tb})"))
    if len(spd_a) and len(spd_b):
        print(anova_summary([spd_a, spd_b], f"Speed ({ta} vs {tb})"))

    # Collision ANOVA if both groups have obstacle data
    hits_a = np.concatenate(
        [
            e["hits_per_trial"]
            for e in groups["A"]
            if e["csv_type"] == "obstacle" and len(e["hits_per_trial"])
        ]
    )
    hits_b = np.concatenate(
        [
            e["hits_per_trial"]
            for e in groups["B"]
            if e["csv_type"] == "obstacle" and len(e["hits_per_trial"])
        ]
    )
    if len(hits_a) and len(hits_b):
        print(anova_summary([hits_a, hits_b], f"Hits per Trial ({ta} vs {tb})"))


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    root = _tk_root()

    paths = filedialog.askopenfilenames(
        parent=root,
        title="Select Group A CSV file(s)  [hold Ctrl/Cmd for multiple]",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )

    if not paths:
        messagebox.showinfo("No files selected", "No files were selected. Exiting.")
        sys.exit(0)

    group_titles["A"] = _ask_title("A", "Group A")

    print(f"\n{group_titles['A']} — loading {len(paths)} file(s):")
    _ingest_paths([Path(p) for p in paths], "A")

    if not groups["A"]:
        sys.exit(1)

    print("\nKeybinds (click plot first):")
    print("  H — toggle button bar")
    print("  M — toggle success-rate figure")
    print("  P — toggle phase analysis figure  (transport/obstacle CSVs only)")
    print("  C — toggle collision analysis figure  (obstacle CSVs only)")

    build_overview_figure()
    plt.show()


if __name__ == "__main__":
    main()

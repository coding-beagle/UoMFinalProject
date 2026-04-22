import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, TextBox

CSV_PATH = "data.csv"  # <-- update to your file path

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
frames = df["frame"].values.astype(float)
roll = df["roll_deg"].values.astype(float)
data_dict = dict(zip(df["frame"].values.astype(int), roll))

x_start = [int(frames.min())]
x_end = [int(frames.max())]


def get_all_frames():
    return np.arange(x_start[0], x_end[0] + 1, dtype=float)


# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 11))
gs = gridspec.GridSpec(3, 1, hspace=0.6, top=0.93, bottom=0.12)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

y_pad = (roll.max() - roll.min()) * 0.3
ylim = (roll.min() - y_pad, roll.max() + y_pad)

FS_AXIS = 15  # x/y label font size
FS_TITLE = 20  # subplot title font size
FS_MAIN = 20  # suptitle font size
FS_LEG = 15  # legend font size


def style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=FS_TITLE)
    ax.set_xlabel(xlabel, fontsize=FS_AXIS)
    ax.set_ylabel(ylabel, fontsize=FS_AXIS)
    ax.grid(True)


# ── Plot 1 ────────────────────────────────────────────────────────────────────
ax1.scatter(frames, roll, s=8, alpha=0.8, zorder=3)
ax1.set_xlim(x_start[0], x_end[0])
ax1.set_ylim(*ylim)
style_ax(ax1, "Recorded Roll Angle", "frame", "roll_deg")

# ── Plot 2 ────────────────────────────────────────────────────────────────────
ax2.set_xlim(x_start[0], x_end[0])
ax2.set_ylim(*ylim)
style_ax(ax2, "Expected Roll Angle", "frame", "roll_deg")

(wave_line,) = ax2.plot([], [], lw=1.5, color="tab:orange", zorder=4)

# ── Plot 3 ────────────────────────────────────────────────────────────────────
ax3.set_xlim(x_start[0], x_end[0])
style_ax(ax3, "RMSE Roll Angle", "frame", "RMSE (degrees)")

(rmse_line,) = ax3.plot([], [], lw=1.5, color="tab:blue", zorder=4, label="RMSE")
inf_scatter = ax3.scatter(
    [], [], color="red", s=60, zorder=5, label="no data (∞)", marker="|", linewidths=1.5
)
ax3.legend(loc="upper right", fontsize=FS_LEG)

# fig.suptitle("ArUco Marker — Roll Analysis", fontsize=FS_MAIN, fontweight="bold")

# ── Drawing state ─────────────────────────────────────────────────────────────
pts_x = []
pts_y = []
is_drawing = [False]
active_ax = [None]


def clear_drawing():
    pts_x.clear()
    pts_y.clear()
    is_drawing[0] = False
    active_ax[0] = None
    wave_line.set_data([], [])
    rmse_line.set_data([], [])
    inf_scatter.set_offsets(np.empty((0, 2)))
    ax3.set_ylim(0, 1)


def redraw_rmse():
    all_frames = get_all_frames()
    if len(pts_x) < 2:
        return

    order = np.argsort(pts_x)
    wx = np.array(pts_x)[order]
    wy = np.array(pts_y)[order]

    wave_interp = np.interp(all_frames, wx, wy, left=np.nan, right=np.nan)
    wave_line.set_data(all_frames, wave_interp)

    rmse_vals = []
    inf_frames = []

    for i, f in enumerate(all_frames):
        fi = int(f)
        no_wave = np.isnan(wave_interp[i])
        no_data = fi not in data_dict

        if no_wave or no_data:
            rmse_vals.append(np.nan)
            if not no_wave:  # wave drawn here but no measurement
                inf_frames.append(f)
        else:
            rmse_vals.append(abs(data_dict[fi] - wave_interp[i]))

    rmse_arr = np.array(rmse_vals)
    finite = rmse_arr[~np.isnan(rmse_arr)]

    rmse_line.set_data(all_frames, rmse_arr)

    ymax = (finite.max() * 1.3 + 1) if finite.size else 10.0
    ax3.set_ylim(0, ymax)

    if inf_frames:
        inf_x = np.array(inf_frames)
        inf_y = np.full_like(inf_x, ymax * 0.9)
        inf_scatter.set_offsets(np.column_stack([inf_x, inf_y]))
        inf_scatter.set_sizes([60] * len(inf_x))
    else:
        inf_scatter.set_offsets(np.empty((0, 2)))

    fig.canvas.draw_idle()


def on_press(event):
    if event.button != 1 or event.inaxes not in (ax1, ax2):
        return
    is_drawing[0] = True
    active_ax[0] = event.inaxes
    pts_x.clear()
    pts_y.clear()
    pts_x.append(event.xdata)
    pts_y.append(event.ydata)
    wave_line.set_data([], [])
    rmse_line.set_data([], [])
    inf_scatter.set_offsets(np.empty((0, 2)))
    fig.canvas.draw_idle()


def on_motion(event):
    if not is_drawing[0] or event.inaxes is not active_ax[0]:
        return
    if event.xdata is None or event.ydata is None:
        return
    pts_x.append(event.xdata)
    pts_y.append(event.ydata)
    order = np.argsort(pts_x)
    wave_line.set_data(np.array(pts_x)[order], np.array(pts_y)[order])
    fig.canvas.draw_idle()


def on_release(event):
    if not is_drawing[0]:
        return
    is_drawing[0] = False
    active_ax[0] = None
    redraw_rmse()


fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("motion_notify_event", on_motion)
fig.canvas.mpl_connect("button_release_event", on_release)

# ── Widgets ───────────────────────────────────────────────────────────────────
# Clear
ax_btn = fig.add_axes([0.05, 0.03, 0.08, 0.04])
btn = Button(ax_btn, "Clear")


def clear_cb(_):
    clear_drawing()
    fig.canvas.draw_idle()


btn.on_clicked(clear_cb)

# X start
ax_lbl_s = fig.add_axes([0.30, 0.03, 0.10, 0.04])
ax_lbl_s.axis("off")
ax_lbl_s.text(
    1.0,
    0.5,
    "X start:",
    ha="right",
    va="center",
    transform=ax_lbl_s.transAxes,
    fontsize=10,
)

ax_txt_s = fig.add_axes([0.41, 0.03, 0.08, 0.04])
tb_start = TextBox(ax_txt_s, "", initial=str(x_start[0]))


def update_xstart(text):
    try:
        val = int(float(text))
    except ValueError:
        return
    x_start[0] = val
    for ax in (ax1, ax2, ax3):
        ax.set_xlim(val, x_end[0])
    clear_drawing()
    fig.canvas.draw_idle()


tb_start.on_submit(update_xstart)

# X end
ax_lbl_e = fig.add_axes([0.55, 0.03, 0.10, 0.04])
ax_lbl_e.axis("off")
ax_lbl_e.text(
    1.0,
    0.5,
    "X end:",
    ha="right",
    va="center",
    transform=ax_lbl_e.transAxes,
    fontsize=10,
)

ax_txt_e = fig.add_axes([0.66, 0.03, 0.08, 0.04])
tb_end = TextBox(ax_txt_e, "", initial=str(x_end[0]))


def update_xend(text):
    try:
        val = int(float(text))
    except ValueError:
        return
    x_end[0] = val
    for ax in (ax1, ax2, ax3):
        ax.set_xlim(x_start[0], val)
    clear_drawing()
    fig.canvas.draw_idle()


tb_end.on_submit(update_xend)

plt.show()

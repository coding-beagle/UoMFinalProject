"""
reach_experiment.py
───────────────────
Reusable experiment module for the YuMi joystick controller.

Defines:
  • ReachTarget               – a single spatial goal (sphere in sim + HUD feedback)
  • Experiment                – orchestrates a sequence of ReachTargets, logs results
  • TransportTrial            – a single pick-and-place task (grip cube → carry to zone)
  • TransportExperiment       – orchestrates a sequence of TransportTrials, logs results
  • ObstacleTransportTrial    – TransportTrial with a cloud of spherical obstacles
  • ObstacleTransportExperiment – orchestrates ObstacleTransportTrials, logs results

Usage – reach experiment (unchanged):
    from reach_experiment import Experiment

    exp = Experiment(sim, trials=[...])
    exp.update(wrist_pos, dt)
    exp.draw(screen, wrist_pos, fonts, dt)

Usage – transport experiment:
    from reach_experiment import TransportExperiment

    exp = TransportExperiment.from_random(
        sim,
        shoulder_pos=robot_shoulder_world,
        arm_length=ROBOT_ARM_LENGTH,
        n_trials=5,
    )

    # inside sim loop – pass gripper_open (True = open, False = closed)
    # and the current start/resting position of the arm
    exp.update(wrist_pos, gripper_open, dt, start_pos=arm_start_pos)
    exp.draw(screen, wrist_pos, fonts, dt)

Usage – obstacle transport experiment:
    from reach_experiment import ObstacleTransportExperiment

    obs_cfg = ObstacleConfig(
        n_obstacles=12,          # number of spheres
        radius_min=0.03,         # smallest sphere radius  (m)
        radius_max=0.09,         # largest sphere radius   (m)
        margin=0.12,             # keep-clear radius around cube & drop zone
        seed=42,                 # set for reproducibility, None for random
    )

    exp = ObstacleTransportExperiment.from_random(
        sim,
        shoulder_pos=robot_shoulder_world,
        arm_length=ROBOT_ARM_LENGTH,
        n_trials=5,
        obstacle_cfg=obs_cfg,
    )

    exp.update(wrist_pos, gripper_open, dt, start_pos=arm_start_pos)
    exp.draw(screen, wrist_pos, fonts, dt)

    # Obstacle config can be changed between trials:
    exp.obstacle_cfg.n_obstacles = 20
    exp.obstacle_cfg.radius_max  = 0.12
"""

from __future__ import annotations
import time
import math
import random
import pygame
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette (matches the controller UI feel)
# ─────────────────────────────────────────────────────────────────────────────
C_IDLE = (100, 160, 220)  # calm blue  – far from target
C_WARM = (220, 190, 60)  # amber      – getting close
C_HOT = (100, 220, 130)  # green      – inside target zone
C_SUCCESS = (100, 220, 130)
C_FAIL = (220, 80, 80)
C_ORANGE = (230, 130, 40)  # cube / pick phase
C_TEXT_DIM = (130, 130, 130)
C_TEXT_BRT = (210, 210, 210)
C_PANEL_BG = (22, 22, 28, 200)
C_OBSTACLE = (180, 60, 60)  # resting obstacle colour
C_OBSTACLE_HIT = (255, 40, 40)  # flash colour on collision


# ─────────────────────────────────────────────────────────────────────────────
# Hemisphere sampler  (shared by both experiment types)
# ─────────────────────────────────────────────────────────────────────────────
def sample_hemisphere_positions(
    shoulder,
    arm_length,
    n,
    min_reach=0.4,
    max_reach=0.85,
    min_elevation=-10.0,
    max_elevation=80.0,
    az_min=-45.0,
    az_max=45.0,
    az_centre=-90.0,
    seed=None,
):
    rng = random.Random(seed)
    positions = []
    r_min = arm_length * min_reach
    r_max = arm_length * max_reach
    el_lo = math.radians(min_elevation)
    el_hi = math.radians(max_elevation)
    centre_rad = math.radians(az_centre)
    az_lo = centre_rad + math.radians(az_min)
    az_hi = centre_rad + math.radians(az_max)
    while len(positions) < n:
        el = rng.uniform(el_lo, el_hi)
        az = rng.uniform(az_lo, az_hi)
        r = rng.uniform(r_min, r_max)
        x = shoulder[0] + r * math.cos(el) * math.cos(az)
        y = shoulder[1] + r * math.cos(el) * math.sin(az)
        z = shoulder[2] + r * math.sin(el)
        positions.append([x, y, z])
    return positions


# ─────────────────────────────────────────────────────────────────────────────
# ReachTarget
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ReachTarget:
    sim: object
    position: List[float]
    radius: float = 0.04
    dwell_time: float = 0.5
    timeout: float = 15.0
    label: str = "Target"

    _dummy: Optional[int] = field(default=None, init=False, repr=False)
    _elapsed: float = field(default=0.0, init=False, repr=False)
    _dwell_acc: float = field(default=0.0, init=False, repr=False)
    _inside: bool = field(default=False, init=False, repr=False)
    _result: Optional[str] = field(default=None, init=False, repr=False)
    _flash_t: float = field(default=0.0, init=False, repr=False)
    _started: bool = field(default=False, init=False, repr=False)
    _start_wrist: Optional[List[float]] = field(default=None, init=False, repr=False)
    _settle_frames: int = field(default=0, init=False, repr=False)
    MOVE_THRESHOLD: float = field(default=0.02, init=False, repr=False)
    SETTLE_FRAMES: int = field(default=10, init=False, repr=False)

    def __post_init__(self):
        self._spawn_dummy()

    def _spawn_dummy(self):
        sim = self.sim
        self._dummy = sim.createDummy(self.radius * 2)
        if self._dummy is None or self._dummy < 0:
            raise RuntimeError(
                f"createDummy returned invalid handle ({self._dummy}). "
                "Ensure the simulation is running before creating an Experiment."
            )
        sim.setObjectPosition(self._dummy, self.position)
        sim.setObjectAlias(self._dummy, self.label)

    def remove(self):
        if self._dummy is not None:
            try:
                self.sim.removeObject(self._dummy)
            except Exception:
                pass
            self._dummy = None

    def update(self, wrist_pos: List[float], dt: float) -> Optional[str]:
        if self._result is not None:
            self._flash_t = max(0.0, self._flash_t - dt)
            return self._result

        if not self._started:
            self._settle_frames += 1
            if self._settle_frames <= self.SETTLE_FRAMES:
                return None
            if self._start_wrist is None:
                self._start_wrist = list(wrist_pos)
            moved = math.sqrt(
                sum((wrist_pos[i] - self._start_wrist[i]) ** 2 for i in range(3))
            )
            if moved >= self.MOVE_THRESHOLD:
                self._started = True
            dist = math.sqrt(
                sum((wrist_pos[i] - self.position[i]) ** 2 for i in range(3))
            )
            self._inside = dist <= self.radius
            return None

        self._elapsed += dt
        dist = math.sqrt(sum((wrist_pos[i] - self.position[i]) ** 2 for i in range(3)))
        self._inside = dist <= self.radius

        if self._inside:
            self._dwell_acc += dt
            if self._dwell_acc >= self.dwell_time:
                self._result = "success"
                self._flash_t = 0.6
                self.remove()
        else:
            self._dwell_acc = max(0.0, self._dwell_acc - dt * 2)

        if self.timeout > 0 and self._elapsed >= self.timeout:
            if self._result is None:
                self._result = "timeout"
                self._flash_t = 0.6
                self.remove()

        return self._result

    @property
    def finished(self) -> bool:
        return self._result is not None

    @property
    def success(self) -> bool:
        return self._result == "success"

    @property
    def time_remaining(self) -> float:
        if self.timeout <= 0:
            return float("inf")
        if not self._started:
            return self.timeout
        return max(0.0, self.timeout - self._elapsed)

    @property
    def dwell_fraction(self) -> float:
        return min(1.0, self._dwell_acc / self.dwell_time)

    def distance_to(self, wrist_pos: List[float]) -> float:
        return math.sqrt(sum((wrist_pos[i] - self.position[i]) ** 2 for i in range(3)))

    def draw(
        self, surf: pygame.Surface, wrist_pos: List[float], fonts: dict, dt: float = 0.0
    ):
        if self.finished and self._flash_t <= 0:
            return

        dist = self.distance_to(wrist_pos)
        W, H = surf.get_size()

        if self._flash_t > 0 and self._result:
            alpha = int(160 * (self._flash_t / 0.6))
            col = C_SUCCESS if self._result == "success" else C_FAIL
            flash = pygame.Surface((W, H), pygame.SRCALPHA)
            flash.fill((*col, alpha))
            surf.blit(flash, (0, 0))
            msg = "✓  TARGET REACHED" if self._result == "success" else "✗  TIMED OUT"
            lbl = fonts["lg"].render(msg, True, (255, 255, 255))
            surf.blit(
                lbl, (W // 2 - lbl.get_width() // 2, H // 2 - lbl.get_height() // 2)
            )
            return

        ratio = min(1.0, dist / (self.radius * 6))
        if ratio < 0.4:
            hud_col = _lerp_col(C_HOT, C_WARM, ratio / 0.4)
        else:
            hud_col = _lerp_col(C_WARM, C_IDLE, (ratio - 0.4) / 0.6)

        panel_w, panel_h = 260, 110
        px, py = W - panel_w - 10, 250
        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((22, 22, 28, 190))
        surf.blit(panel, (px, py))
        pygame.draw.rect(surf, hud_col, (px, py, panel_w, panel_h), 1)

        surf.blit(fonts["md"].render(self.label, True, C_TEXT_BRT), (px + 10, py + 8))
        dist_txt = f"dist  {dist * 100:.1f} cm"
        surf.blit(fonts["sm"].render(dist_txt, True, hud_col), (px + 10, py + 28))

        bar_x, bar_y = px + 10, py + 50
        bar_w, bar_h = panel_w - 20, 10
        fill = int(bar_w * (1.0 - ratio))
        pygame.draw.rect(surf, (55, 55, 55), (bar_x, bar_y, bar_w, bar_h))
        if fill > 0:
            pygame.draw.rect(surf, hud_col, (bar_x, bar_y, fill, bar_h))

        if self._inside:
            ring_cx, ring_cy = px + panel_w - 28, py + 72
            ring_r = 18
            pygame.draw.circle(surf, (55, 55, 55), (ring_cx, ring_cy), ring_r, 3)
            arc_rect = pygame.Rect(
                ring_cx - ring_r, ring_cy - ring_r, ring_r * 2, ring_r * 2
            )
            end_angle = -math.pi / 2 + self.dwell_fraction * 2 * math.pi
            _draw_arc(surf, C_HOT, arc_rect, -math.pi / 2, end_angle, 3)
            pct = fonts["sm"].render(f"{int(self.dwell_fraction*100)}%", True, C_HOT)
            surf.blit(
                pct, (ring_cx - pct.get_width() // 2, ring_cy - pct.get_height() // 2)
            )

        if self.timeout > 0:
            if not self._started:
                t_txt = fonts["sm"].render("move to start timer", True, (160, 140, 60))
            else:
                tr = self.time_remaining
                t_col = C_FAIL if tr < 3.0 else C_TEXT_DIM
                t_txt = fonts["sm"].render(f"time  {tr:.1f}s", True, t_col)
            surf.blit(t_txt, (px + 10, py + 68 + 20))


# ─────────────────────────────────────────────────────────────────────────────
# Experiment  – sequence of ReachTargets
# ─────────────────────────────────────────────────────────────────────────────
class Experiment:
    def __init__(self, sim, trials: List[dict]):
        self.sim = sim
        self._trial_defs = trials
        self._index = 0
        self._results: List[dict] = []
        self._active: Optional[ReachTarget] = None
        self._start_time = time.time()
        self._trial_start = time.time()
        self._result_logged = False
        self._spawn_next()

    @classmethod
    def from_hemisphere(
        cls,
        sim,
        shoulder_pos: List[float],
        arm_length: float,
        n_trials: int = 6,
        radius: float = 0.04,
        dwell_time: float = 0.5,
        timeout: float = 20.0,
        seed: int = None,
        min_reach: float = 0.4,
        max_reach: float = 0.85,
        min_elevation: float = -20.0,
        max_elevation: float = 60.0,
        az_min: float = -45.0,
        az_max: float = 45.0,
        az_centre: float = -90.0,
    ) -> "Experiment":
        positions = sample_hemisphere_positions(
            shoulder=shoulder_pos,
            arm_length=arm_length,
            n=n_trials,
            min_reach=min_reach,
            max_reach=max_reach,
            min_elevation=min_elevation,
            max_elevation=max_elevation,
            az_min=az_min,
            az_max=az_max,
            az_centre=az_centre,
            seed=seed,
        )
        trials = [
            {
                "pos": pos,
                "radius": radius,
                "dwell_time": dwell_time,
                "timeout": timeout,
                "label": f"Trial {i+1}",
            }
            for i, pos in enumerate(positions)
        ]
        return cls(sim, trials)

    def _spawn_next(self):
        if self._index >= len(self._trial_defs):
            self._active = None
            return
        cfg = self._trial_defs[self._index]
        self._active = ReachTarget(
            sim=self.sim,
            position=cfg["pos"],
            radius=cfg.get("radius", 0.04),
            dwell_time=cfg.get("dwell_time", 0.5),
            timeout=cfg.get("timeout", 15.0),
            label=cfg.get("label", f"Trial {self._index + 1}"),
        )
        self._trial_start = time.time()

    def update(self, wrist_pos: List[float], dt: float):
        if self._active is None:
            return
        result = self._active.update(wrist_pos, dt)
        if result is not None and not self._result_logged:
            self._result_logged = True
            self._results.append(
                {
                    "trial": self._index + 1,
                    "label": self._active.label,
                    "result": result,
                    "duration": time.time() - self._trial_start,
                }
            )
            self._index += 1
        self._maybe_advance()

    def _maybe_advance(self):
        if (
            self._active is not None
            and self._active.finished
            and self._active._flash_t <= 0
        ):
            self._result_logged = False
            self._spawn_next()

    def draw(
        self, surf: pygame.Surface, wrist_pos: List[float], fonts: dict, dt: float = 0.0
    ):
        self._maybe_advance()
        W, H = surf.get_size()
        total = len(self._trial_defs)
        done = len(self._results)
        bar_w = W - 20
        pygame.draw.rect(surf, (45, 45, 55), (10, H - 18, bar_w, 8))
        if total:
            filled = int(bar_w * done / total)
            pygame.draw.rect(surf, C_SUCCESS, (10, H - 18, filled, 8))
        prog_txt = fonts["sm"].render(
            f"trial {min(done+1, total)} / {total}  ·  "
            + (f"{done} done" if done else ""),
            True,
            C_TEXT_DIM,
        )
        surf.blit(prog_txt, (10, H - 34))
        if self._active is not None:
            self._active.draw(surf, wrist_pos, fonts, dt)
        if self.finished:
            _draw_summary(surf, self._results, fonts)

    @property
    def finished(self) -> bool:
        return self._active is None and self._index >= len(self._trial_defs)

    @property
    def results(self) -> List[dict]:
        return list(self._results)

    def summary(self) -> str:
        lines = ["─── Reach Experiment Results ───"]
        for r in self._results:
            lines.append(
                f"  {r['trial']:2d}. {r['label']:20s}  "
                f"{r['result']:8s}  {r['duration']:.2f}s"
            )
        n_ok = sum(1 for r in self._results if r["result"] == "success")
        lines.append(
            f"\n  {n_ok}/{len(self._results)} succeeded  "
            f"| total {time.time()-self._start_time:.1f}s"
        )
        return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# TransportTrial  –  pick up a cube, carry it to a drop zone
# ═════════════════════════════════════════════════════════════════════════════
# Phase flow:
#   APPROACH → GRIP → CARRY → PLACE → done
#
#   APPROACH : move wrist within pick_radius of cube_pos (gripper must be open)
#   GRIP     : close gripper (RT) while inside pick zone → cube "attaches"
#   CARRY    : wrist with cube inside must reach drop_pos
#   PLACE    : open gripper (LT) while inside drop zone → success
#
# The cube is simulated as a CoppeliaSim primitive shape. On GRIP the cube's
# position is locked to follow the wrist each sim step. On PLACE it is
# released.  A timeout covers the whole trial.
#
# Per-phase time splits are tracked cumulatively: if the trial regresses
# (e.g. phase CARRY → APPROACH because gripper opened early) the time
# already accumulated in each phase is preserved and keeps growing.
#
# start_pos is recorded at the beginning of the trial and used to compute
# distances from the arm's resting position to the cube and drop zone,
# giving a measure of task difficulty independent of arm length.
# ═════════════════════════════════════════════════════════════════════════════

_PHASE_APPROACH = "approach"
_PHASE_GRIP = "grip"
_PHASE_CARRY = "carry"
_PHASE_PLACE = "place"
_PHASE_DONE = "done"

_PHASE_LABELS = {
    _PHASE_APPROACH: "1. Move wrist to cube  (open gripper)",
    _PHASE_GRIP: "2. Hold LT to close gripper & grip",
    _PHASE_CARRY: "3. Carry cube to drop zone",
    _PHASE_PLACE: "4. Hold RT to open gripper & place",
    _PHASE_DONE: "Done!",
}

# Ordered list used for display / iteration
_ALL_PHASES = [_PHASE_APPROACH, _PHASE_GRIP, _PHASE_CARRY, _PHASE_PLACE]


class TransportTrial:
    """
    Single pick-and-place trial.

    Parameters
    ----------
    start_pos : list[float] | None
        World-space position of the arm at the moment the trial begins
        (e.g. the neutral/rest pose wrist position). When provided it is
        recorded and reported alongside the cube/drop distances so task
        difficulty can be compared across conditions.  If None the first
        wrist_pos passed to update() is used as the start position.
    """

    def __init__(
        self,
        sim,
        cube_pos: List[float],
        drop_pos: List[float],
        pick_radius: float = 0.06,
        drop_radius: float = 0.06,
        timeout: float = 30.0,
        label: str = "Transport",
        start_pos: Optional[List[float]] = None,
    ):
        self.sim = sim
        self.cube_pos = list(cube_pos)
        self.drop_pos = list(drop_pos)
        self.pick_radius = pick_radius
        self.drop_radius = drop_radius
        self.timeout = timeout
        self.label = label

        # Starting arm position – may be set lazily on first update() call
        self._start_pos: Optional[List[float]] = list(start_pos) if start_pos else None
        self._start_pos_locked = start_pos is not None  # True once finalised

        self._phase = _PHASE_APPROACH
        self._result: Optional[str] = None
        self._elapsed = 0.0
        self._flash_t = 0.0
        self._gripped = False  # cube following wrist?
        self._current_cube_pos = list(cube_pos)

        # ── per-phase cumulative time splits ──────────────────────────────────
        # Accumulates wall-clock seconds spent in each phase across the whole
        # trial, including any time after a regression back to an earlier phase.
        self._phase_times: dict = {p: 0.0 for p in _ALL_PHASES}

        # Spawn sim objects
        self._cube_handle = self._spawn_cube(cube_pos)
        self._drop_dummy = self._spawn_drop_zone(drop_pos, drop_radius)

    # ── sim object helpers ────────────────────────────────────────────────────
    def _spawn_cube(self, pos):
        """Create a small box primitive at pos."""
        sim = self.sim
        size = 0.04  # 4 cm cube side
        handle = sim.createPrimitiveShape(
            sim.primitiveshape_cuboid,
            [size, size, size],
            0,  # options (0 = default, respondable + dynamic)
        )
        if handle is None or handle < 0:
            raise RuntimeError(f"Failed to create cube shape (handle={handle})")
        sim.setObjectPosition(handle, pos)
        sim.setObjectAlias(handle, f"{self.label}_cube")
        # Make it static so it doesn't fall during APPROACH
        sim.setObjectInt32Param(handle, sim.shapeintparam_static, 1)
        return handle

    def _spawn_drop_zone(self, pos, radius):
        """Create a ghost dummy sphere marking the drop zone."""
        sim = self.sim
        handle = sim.createDummy(radius * 2)
        if handle is None or handle < 0:
            raise RuntimeError(f"Failed to create drop-zone dummy (handle={handle})")
        sim.setObjectPosition(handle, pos)
        sim.setObjectAlias(handle, f"{self.label}_dropzone")
        return handle

    def _remove_all(self):
        for attr in ("_cube_handle", "_drop_dummy"):
            h = getattr(self, attr, None)
            if h is not None:
                try:
                    self.sim.removeObject(h)
                except Exception:
                    pass
                setattr(self, attr, None)

    # ── distance helpers ──────────────────────────────────────────────────────
    def _dist(self, a, b):
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))

    # ── start-position distances (computed on demand once start_pos is known) ─
    @property
    def dist_start_to_cube(self) -> Optional[float]:
        """Straight-line distance from the arm start position to the cube."""
        if self._start_pos is None:
            return None
        return self._dist(self._start_pos, self.cube_pos)

    @property
    def dist_start_to_drop(self) -> Optional[float]:
        """Straight-line distance from the arm start position to the drop zone."""
        if self._start_pos is None:
            return None
        return self._dist(self._start_pos, self.drop_pos)

    # ── main update ───────────────────────────────────────────────────────────
    def update(
        self,
        wrist_pos: List[float],
        gripper_open: bool,
        dt: float,
    ) -> Optional[str]:
        """
        Call every sim step.

        Parameters
        ----------
        wrist_pos    : current world-space wrist position [x, y, z]
        gripper_open : True = open (LT), False = closed (RT)
        dt           : elapsed seconds since last call

        Returns
        -------
        Result string once finished ("success" | "timeout"), else None.
        """
        if self._result is not None:
            self._flash_t = max(0.0, self._flash_t - dt)
            return self._result

        # Latch start position on very first call if not provided at init
        if not self._start_pos_locked:
            self._start_pos = list(wrist_pos)
            self._start_pos_locked = True

        self._elapsed += dt

        # ── accumulate time in the current phase ──────────────────────────────
        if self._phase in self._phase_times:
            self._phase_times[self._phase] += dt

        # ── timeout ───────────────────────────────────────────────────────────
        if self.timeout > 0 and self._elapsed >= self.timeout:
            self._set_result("timeout")
            return self._result

        # ── if cube is gripped, move it with the wrist ────────────────────────
        if self._gripped and self._cube_handle is not None:
            self._current_cube_pos = list(wrist_pos)
            self.sim.setObjectPosition(self._cube_handle, -1, wrist_pos)

        dist_to_cube = self._dist(wrist_pos, self._current_cube_pos)
        dist_to_drop = self._dist(wrist_pos, self.drop_pos)

        # ── phase state machine ───────────────────────────────────────────────
        if self._phase == _PHASE_APPROACH:
            # Require gripper open before allowing grip — prevents accidental
            # instant-grip if the user is already holding LT
            if dist_to_cube <= self.pick_radius and gripper_open:
                self._phase = _PHASE_GRIP

        elif self._phase == _PHASE_GRIP:
            if dist_to_cube > self.pick_radius:
                # moved away before gripping → back to approach
                self._phase = _PHASE_APPROACH
            elif not gripper_open:
                # LT held → gripper closed → attach cube
                self._gripped = True
                self._phase = _PHASE_CARRY

        elif self._phase == _PHASE_CARRY:
            if gripper_open:
                # RT held → gripper opened → dropped prematurely
                self._gripped = False
                self._phase = _PHASE_APPROACH
            elif dist_to_drop <= self.drop_radius:
                self._phase = _PHASE_PLACE

        elif self._phase == _PHASE_PLACE:
            if dist_to_drop > self.drop_radius:
                # drifted out while still gripping
                self._phase = _PHASE_CARRY
            elif gripper_open:
                # RT held → gripper opened inside drop zone → success
                self._gripped = False
                self._set_result("success")

        return self._result

    def _set_result(self, result: str):
        self._result = result
        self._flash_t = 0.8
        self._gripped = False
        self._remove_all()

    # ── properties ────────────────────────────────────────────────────────────
    @property
    def finished(self) -> bool:
        return self._result is not None

    @property
    def time_remaining(self) -> float:
        return max(0.0, self.timeout - self._elapsed)

    @property
    def phase_splits(self) -> dict:
        """
        Cumulative seconds spent in each phase, including any time accumulated
        after regressions.  Keys are the phase name strings; values are floats.

        Example
        -------
        {
            "approach": 4.31,
            "grip":     0.82,
            "carry":    6.10,
            "place":    1.05,
        }
        """
        return dict(self._phase_times)

    # ── draw ──────────────────────────────────────────────────────────────────
    def draw(
        self,
        surf: pygame.Surface,
        wrist_pos: List[float],
        fonts: dict,
        dt: float = 0.0,
    ):
        if self.finished and self._flash_t <= 0:
            return

        W, H = surf.get_size()

        # Flash overlay
        if self._flash_t > 0 and self._result:
            alpha = int(180 * (self._flash_t / 0.8))
            col = C_SUCCESS if self._result == "success" else C_FAIL
            flash = pygame.Surface((W, H), pygame.SRCALPHA)
            flash.fill((*col, alpha))
            surf.blit(flash, (0, 0))
            msg = "✓  DELIVERED!" if self._result == "success" else "✗  TIMED OUT"
            lbl = fonts["lg"].render(msg, True, (255, 255, 255))
            surf.blit(
                lbl, (W // 2 - lbl.get_width() // 2, H // 2 - lbl.get_height() // 2)
            )
            return

        # ── HUD panel ─────────────────────────────────────────────────────────
        panel_w, panel_h = 310, 175
        px, py = W - panel_w - 10, 210
        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((22, 22, 28, 190))
        surf.blit(panel, (px, py))

        phase_col = C_ORANGE if self._phase in (_PHASE_APPROACH, _PHASE_GRIP) else C_HOT
        pygame.draw.rect(surf, phase_col, (px, py, panel_w, panel_h), 1)

        surf.blit(fonts["md"].render(self.label, True, C_TEXT_BRT), (px + 10, py + 8))

        # Phase instruction
        instr = _PHASE_LABELS.get(self._phase, "")
        surf.blit(fonts["sm"].render(instr, True, phase_col), (px + 10, py + 28))

        # Distances
        dist_cube = self._dist(wrist_pos, self._current_cube_pos)
        dist_drop = self._dist(wrist_pos, self.drop_pos)

        if self._phase in (_PHASE_APPROACH, _PHASE_GRIP):
            d_txt = f"cube  {dist_cube * 100:.1f} cm away"
            d_col = C_HOT if dist_cube <= self.pick_radius else C_ORANGE
        else:
            d_txt = f"drop zone  {dist_drop * 100:.1f} cm away"
            d_col = C_HOT if dist_drop <= self.drop_radius else C_WARM
        surf.blit(fonts["sm"].render(d_txt, True, d_col), (px + 10, py + 46))

        # Gripper state indicator
        g_txt = (
            "gripper: OPEN  (hold LT to close & grip)"
            if not self._gripped
            else "gripper: CLOSED  (hold RT to open & place)"
        )
        g_col = C_IDLE if not self._gripped else C_HOT
        surf.blit(fonts["sm"].render(g_txt, True, g_col), (px + 10, py + 62))

        # ── per-phase time split mini-bars ────────────────────────────────────
        # Show each phase as a labelled bar whose width reflects the fraction of
        # total elapsed time spent there.  The currently active phase pulses.
        total_t = max(self._elapsed, 0.001)
        bar_section_y = py + 82
        bar_area_w = panel_w - 20
        phase_short = {
            "approach": "appr",
            "grip": "grip",
            "carry": "carr",
            "place": "plac",
        }
        for i, ph in enumerate(_ALL_PHASES):
            t = self._phase_times[ph]
            frac = t / total_t
            bw = int(bar_area_w * frac)
            by = bar_section_y + i * 16
            active = self._phase == ph
            col = phase_col if active else C_TEXT_DIM
            # background track
            pygame.draw.rect(surf, (45, 45, 55), (px + 10, by, bar_area_w, 10))
            if bw > 0:
                pygame.draw.rect(surf, col, (px + 10, by, bw, 10))
            lbl_txt = f"{phase_short.get(ph, ph)}  {t:.1f}s"
            lbl_surf = fonts["sm"].render(lbl_txt, True, col)
            surf.blit(lbl_surf, (px + 10 + bar_area_w + 4, by - 1))

        # Progress dots for phases
        dot_row_y = py + panel_h - 20
        phases = _ALL_PHASES
        dot_x = px + 10
        for i, ph in enumerate(phases):
            done_ph = phases.index(self._phase) > i
            active = self._phase == ph
            col = C_HOT if done_ph else (phase_col if active else (55, 55, 55))
            pygame.draw.circle(surf, col, (dot_x + i * 60 + 15, dot_row_y), 6)
            step_lbl = fonts["sm"].render(
                str(i + 1),
                True,
                (20, 20, 20) if (done_ph or active) else (100, 100, 100),
            )
            surf.blit(
                step_lbl,
                (
                    dot_x + i * 60 + 15 - step_lbl.get_width() // 2,
                    dot_row_y - step_lbl.get_height() // 2,
                ),
            )
            if i < len(phases) - 1:
                line_col = C_HOT if done_ph else (55, 55, 55)
                pygame.draw.line(
                    surf,
                    line_col,
                    (dot_x + i * 60 + 21, dot_row_y),
                    (dot_x + (i + 1) * 60 + 9, dot_row_y),
                    2,
                )

        # Timeout
        tr = self.time_remaining
        t_col = C_FAIL if tr < 5.0 else C_TEXT_DIM
        surf.blit(
            fonts["sm"].render(f"time  {tr:.1f}s", True, t_col),
            (px + 10, py + panel_h - 36),
        )

        # Start-position distances (bottom-left corner, small)
        if self._start_pos is not None:
            d2c = self.dist_start_to_cube
            d2d = self.dist_start_to_drop
            info = f"start→cube {d2c*100:.0f}cm  start→drop {d2d*100:.0f}cm"
            info_surf = fonts["sm"].render(info, True, C_TEXT_DIM)
            surf.blit(info_surf, (10, H - 50))


# ═════════════════════════════════════════════════════════════════════════════
# TransportExperiment  –  orchestrates a sequence of TransportTrials
# ═════════════════════════════════════════════════════════════════════════════
class TransportExperiment:
    """
    Manages a sequence of pick-and-place trials.

    Call update(wrist_pos, gripper_open, dt, start_pos=...) every sim step.
    Call draw(screen, wrist_pos, fonts, dt) every pygame frame.

    Parameters
    ----------
    start_pos : list[float] | None
        If provided at construction time, every trial will be initialised with
        this as the arm's starting position.  You can alternatively pass
        start_pos per-call to update() to capture the live arm position at the
        moment each trial begins.
    """

    def __init__(
        self,
        sim,
        trials: List[dict],
        start_pos: Optional[List[float]] = None,
    ):
        self.sim = sim
        self._trial_defs = trials
        self._default_start_pos = list(start_pos) if start_pos else None
        self._index = 0
        self._results: List[dict] = []
        self._active: Optional[TransportTrial] = None
        self._start_time = time.time()
        self._trial_start = time.time()
        self._result_logged = False
        self._spawn_next()

    @classmethod
    def from_random(
        cls,
        sim,
        shoulder_pos: List[float],
        arm_length: float,
        n_trials: int = 5,
        pick_radius: float = 0.06,
        drop_radius: float = 0.06,
        timeout: float = 30.0,
        min_reach: float = 0.5,
        max_reach: float = 0.80,
        min_elevation: float = -15.0,
        max_elevation: float = 45.0,
        az_min: float = -45.0,
        az_max: float = 45.0,
        az_centre: float = -90.0,
        seed: int = None,
        start_pos: Optional[List[float]] = None,
        min_separation: float = 0.20,
    ) -> "TransportExperiment":
        """
        Generate n_trials pick-and-place tasks with cube and drop positions
        sampled from the reachable hemisphere. Each pair is guaranteed to be
        spatially distinct (>= 15 cm apart).

        Parameters
        ----------
        start_pos : list[float] | None
            Arm starting / resting position for difficulty metrics.
            Can also be supplied live via update().
        """
        rng = random.Random(seed)

        # Sample a generous pool so we have enough candidates left after
        # rejecting pairs that fall closer than min_separation.
        pool_size = max(n_trials * 6, 20)
        all_pos = sample_hemisphere_positions(
            shoulder=shoulder_pos,
            arm_length=arm_length,
            n=pool_size,
            min_reach=min_reach,
            max_reach=max_reach,
            min_elevation=min_elevation,
            max_elevation=max_elevation,
            az_min=az_min,
            az_max=az_max,
            az_centre=az_centre,
            seed=seed,
        )
        rng.shuffle(all_pos)

        def _dist(a, b):
            return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))

        trials = []
        used = [False] * len(all_pos)
        # Iterate until we have n_trials, picking a cube and then the first
        # remaining candidate that is far enough from it.
        i = 0
        while len(trials) < n_trials and i < len(all_pos):
            if used[i]:
                i += 1
                continue
            cube_pos = all_pos[i]
            used[i] = True
            # Find a drop position at least min_separation away
            drop_idx = None
            for j in range(i + 1, len(all_pos)):
                if used[j]:
                    continue
                if _dist(cube_pos, all_pos[j]) >= min_separation:
                    drop_idx = j
                    break
            if drop_idx is None:
                # Couldn't pair this cube — skip it and try the next
                i += 1
                continue
            used[drop_idx] = True
            drop_pos = all_pos[drop_idx]
            trials.append(
                {
                    "cube_pos": cube_pos,
                    "drop_pos": drop_pos,
                    "pick_radius": pick_radius,
                    "drop_radius": drop_radius,
                    "timeout": timeout,
                    "label": f"Transport {len(trials)+1}",
                }
            )
            i += 1

        if len(trials) < n_trials:
            raise RuntimeError(
                f"Could only generate {len(trials)}/{n_trials} transport "
                f"trials with min_separation={min_separation:.2f} m. "
                f"Try lowering min_separation, widening the hemisphere "
                f"sampling bounds, or reducing n_trials."
            )

        return cls(sim, trials, start_pos=start_pos)

    # ── spawn ─────────────────────────────────────────────────────────────────
    def _spawn_next(self, start_pos: Optional[List[float]] = None):
        if self._index >= len(self._trial_defs):
            self._active = None
            return
        cfg = self._trial_defs[self._index]
        # Priority: caller-supplied > constructor default > lazy latch in update()
        sp = start_pos or self._default_start_pos
        self._active = TransportTrial(
            sim=self.sim,
            cube_pos=cfg["cube_pos"],
            drop_pos=cfg["drop_pos"],
            pick_radius=cfg.get("pick_radius", 0.06),
            drop_radius=cfg.get("drop_radius", 0.06),
            timeout=cfg.get("timeout", 30.0),
            label=cfg.get("label", f"Transport {self._index + 1}"),
            start_pos=sp,
        )
        self._trial_start = time.time()

    # ── update ────────────────────────────────────────────────────────────────
    def update(
        self,
        wrist_pos: List[float],
        gripper_open: bool,
        dt: float,
        start_pos: Optional[List[float]] = None,
    ):
        """
        Call every sim step.

        Parameters
        ----------
        wrist_pos    : current world-space wrist position [x, y, z]
        gripper_open : True = open (LT), False = closed (RT)
        dt           : seconds since last call
        start_pos    : optional arm rest position – passed to the *next* trial
                       when the current one ends, so you can supply the live
                       neutral-pose wrist position at trial boundary.
        """
        if self._active is None:
            return

        result = self._active.update(wrist_pos, gripper_open, dt)

        if result is not None and not self._result_logged:
            self._result_logged = True
            self._results.append(
                {
                    "trial": self._index + 1,
                    "label": self._active.label,
                    "result": result,
                    "duration": time.time() - self._trial_start,
                    "phase_splits": self._active.phase_splits,
                    "cube_pos": self._trial_defs[self._index]["cube_pos"],
                    "drop_pos": self._trial_defs[self._index]["drop_pos"],
                    "start_pos": self._active._start_pos,
                    "dist_start_to_cube": self._active.dist_start_to_cube,
                    "dist_start_to_drop": self._active.dist_start_to_drop,
                }
            )
            self._index += 1
            # Stash start_pos so _maybe_advance can forward it to next trial
            self._pending_start_pos = start_pos or wrist_pos

        self._maybe_advance(start_pos=start_pos)

    def _maybe_advance(self, start_pos: Optional[List[float]] = None):
        if (
            self._active is not None
            and self._active.finished
            and self._active._flash_t <= 0
        ):
            self._result_logged = False
            sp = start_pos or getattr(self, "_pending_start_pos", None)
            self._spawn_next(start_pos=sp)

    # ── draw ──────────────────────────────────────────────────────────────────
    def draw(
        self,
        surf: pygame.Surface,
        wrist_pos: List[float],
        fonts: dict,
        dt: float = 0.0,
    ):
        self._maybe_advance()
        W, H = surf.get_size()
        total = len(self._trial_defs)
        done = len(self._results)
        bar_w = W - 20

        pygame.draw.rect(surf, (45, 45, 55), (10, H - 18, bar_w, 8))
        if total:
            filled = int(bar_w * done / total)
            pygame.draw.rect(surf, C_SUCCESS, (10, H - 18, filled, 8))

        prog_txt = fonts["sm"].render(
            f"transport {min(done+1, total)} / {total}  ·  "
            + (f"{done} done" if done else ""),
            True,
            C_TEXT_DIM,
        )
        surf.blit(prog_txt, (10, H - 34))

        if self._active is not None:
            self._active.draw(surf, wrist_pos, fonts, dt)

        if self.finished:
            _draw_summary(
                surf, self._results, fonts, title="TRANSPORT EXPERIMENT COMPLETE"
            )

    # ── properties ────────────────────────────────────────────────────────────
    @property
    def finished(self) -> bool:
        return self._active is None and self._index >= len(self._trial_defs)

    @property
    def results(self) -> List[dict]:
        return list(self._results)

    def summary(self) -> str:
        lines = ["─── Transport Experiment Results ───"]
        for r in self._results:
            sp = r.get("phase_splits", {})
            d2c = r.get("dist_start_to_cube")
            d2d = r.get("dist_start_to_drop")
            dist_info = (
                f"  start→cube {d2c*100:.0f}cm  start→drop {d2d*100:.0f}cm"
                if d2c is not None
                else ""
            )
            splits_str = "  splits: " + "  ".join(
                f"{ph[:4]}={sp.get(ph, 0.0):.2f}s" for ph in _ALL_PHASES
            )
            lines.append(
                f"  {r['trial']:2d}. {r['label']:22s}  "
                f"{r['result']:8s}  {r['duration']:.2f}s"
            )
            if dist_info:
                lines.append(dist_info)
            lines.append(splits_str)
        n_ok = sum(1 for r in self._results if r["result"] == "success")
        lines.append(
            f"\n  {n_ok}/{len(self._results)} delivered  "
            f"| total {time.time()-self._start_time:.1f}s"
        )
        return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# ObstacleConfig  –  parameters for the spherical obstacle cloud
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class ObstacleConfig:
    """
    Controls the number, size, and placement of spherical obstacles scattered
    through the reachable workspace.

    All fields are mutable – you can change them between trials on a live
    ObstacleTransportExperiment and the next spawned trial will pick up the
    new values automatically.

    Parameters
    ----------
    n_obstacles : int
        How many spheres to scatter.  Default 10.
    radius_min : float
        Smallest allowable sphere radius in metres.  Default 0.03 m (3 cm).
    radius_max : float
        Largest allowable sphere radius in metres.  Default 0.08 m (8 cm).
    margin : float
        Keep-clear radius around the cube start position and the drop zone.
        No obstacle centre will be placed within this distance of either
        landmark.  Default 0.12 m.
    seed : int | None
        Random seed for reproducible obstacle layouts.  None = fully random.
    penalty_on_hit : bool
        If True, touching any obstacle adds ``penalty_seconds`` to the
        trial's *reported* duration (does NOT affect the timeout clock).
        Default False.
    penalty_seconds : float
        Seconds added per obstacle contact event.  Default 2.0.
    collision_radius_scale : float
        The collision sphere used for wrist–obstacle detection is the
        obstacle's geometric radius multiplied by this factor.  Values < 1
        give a smaller "hit zone" than the visual sphere; > 1 a larger one.
        Default 1.0 (exact match).
    min_reach : float
        Nearest obstacle distance as a fraction of arm_length.  Default 0.3.
    max_reach : float
        Furthest obstacle distance as a fraction of arm_length.  Default 0.90.
    min_elevation : float
        Lower elevation bound in degrees (negative = below horizontal).
        Default -20.0.
    max_elevation : float
        Upper elevation bound in degrees.  Default 70.0.
    az_min : float
        Minimum azimuth offset from az_centre in degrees.  Default -45.0.
    az_max : float
        Maximum azimuth offset from az_centre in degrees.  Default 45.0.
    az_centre : float
        Centre azimuth for the obstacle cloud in degrees.  Should match the
        az_centre used for cube/drop positions.  Default -90.0.
    shoulder_margin : float
        Keep-clear radius around the shoulder origin.  No obstacle centre will
        be placed within this distance of shoulder_pos, preventing immediate
        collisions at the start of a trial.  Default 0.15 m.
    """

    n_obstacles: int = 10
    radius_min: float = 0.03
    radius_max: float = 0.08
    margin: float = 0.12
    shoulder_margin: float = 0.15
    seed: Optional[int] = None
    penalty_on_hit: bool = False
    penalty_seconds: float = 2.0
    collision_radius_scale: float = 1.0
    # Hemisphere sampling bounds – mirror the same options as from_random()
    min_reach: float = 0.3
    max_reach: float = 0.90
    min_elevation: float = -20.0
    max_elevation: float = 70.0
    az_min: float = -45.0
    az_max: float = 45.0
    az_centre: float = -90.0


# ═════════════════════════════════════════════════════════════════════════════
# _ObstacleSphere  –  single sphere managed inside a trial
# ═════════════════════════════════════════════════════════════════════════════


class _ObstacleSphere:
    """Internal: one spherical obstacle in CoppeliaSim."""

    HIT_FLASH_DURATION = 0.35

    def __init__(self, sim, pos: List[float], radius: float, label: str):
        self.sim = sim
        self.pos = list(pos)
        self.radius = radius
        self._handle: Optional[int] = None
        self._hit_flash: float = 0.0
        self._hit_count: int = 0
        self._contact_active: bool = False  # True while arm is overlapping sphere

        self._handle = sim.createPrimitiveShape(
            sim.primitiveshape_spheroid,
            [radius * 2, radius * 2, radius * 2],
            0,
        )
        if self._handle is None or self._handle < 0:
            raise RuntimeError(
                f"Failed to create obstacle sphere (handle={self._handle})"
            )
        sim.setObjectPosition(self._handle, pos)
        sim.setObjectAlias(self._handle, label)
        # Static so physics doesn't move it; respondable flag does not affect
        # checkCollision — geometry is always tested regardless.
        sim.setObjectInt32Param(self._handle, sim.shapeintparam_static, 1)

    def remove(self):
        if self._handle is not None:
            try:
                self.sim.removeObject(self._handle)
            except Exception:
                pass
            self._handle = None

    def check_collision(self, arm_collection: int, dt: float = 0.0) -> bool:
        """
        Uses sim.checkCollision to test whether any part of the arm (represented
        by arm_collection — a CoppeliaSim collection covering all arm links)
        intersects this obstacle sphere.

        Returns True only on the *leading edge* of a contact (i.e. the first
        frame the arm enters the sphere).  Stays False during sustained contact
        and resets to False once the arm has fully cleared the sphere, so each
        pass-through counts as exactly one hit.

        Also ticks down the HUD flash timer.
        """
        self._hit_flash = max(0.0, self._hit_flash - dt)

        if self._handle is None:
            return False

        # checkCollision returns (result, collidingPairs)
        # result is 1 if colliding, 0 if not.
        result, _ = self.sim.checkCollision(self._handle, arm_collection)
        in_contact = result == 1

        new_hit = False
        if in_contact and not self._contact_active:
            self._contact_active = True
            self._hit_count += 1
            self._hit_flash = self.HIT_FLASH_DURATION
            new_hit = True
        elif not in_contact:
            self._contact_active = False

        return new_hit

    @property
    def is_flashing(self) -> bool:
        return self._hit_flash > 0.0

    @property
    def hit_count(self) -> int:
        return self._hit_count


# ═════════════════════════════════════════════════════════════════════════════
# ObstacleTransportTrial
# ═════════════════════════════════════════════════════════════════════════════


class ObstacleTransportTrial(TransportTrial):
    """
    A TransportTrial augmented with a cloud of spherical obstacles.

    The obstacles are static, non-respondable spheres scattered through the
    workspace.  The trial logic (phases, gripper, timing) is identical to
    TransportTrial; this subclass layers on:

      • Collision detection  – wrist enters an obstacle sphere → hit event
      • Penalty accounting   – optional time penalty added to reported duration
      • Hit statistics        – per-obstacle and aggregate counts logged in results
      • HUD additions         – obstacle count badge + per-hit flash overlay

    Parameters
    ----------
    obstacle_cfg : ObstacleConfig
        Configuration dataclass controlling cloud density, size range, etc.
    arm_collection : int
        CoppeliaSim collection handle covering all arm links, used by
        sim.checkCollision to test the full arm geometry against each obstacle.
        Create once at startup with sim.createCollection and add the arm root.
    shoulder_pos : list[float]
        World-space shoulder position used to sample obstacle positions.
    arm_length : float
        Used with shoulder_pos to bound the sampling volume.
    """

    def __init__(
        self,
        sim,
        cube_pos: List[float],
        drop_pos: List[float],
        obstacle_cfg: ObstacleConfig,
        arm_collection: int,
        shoulder_pos: List[float],
        arm_length: float,
        pick_radius: float = 0.06,
        drop_radius: float = 0.06,
        timeout: float = 30.0,
        label: str = "Obstacle Transport",
        start_pos: Optional[List[float]] = None,
    ):
        # Initialise the base trial (spawns cube + drop zone dummy)
        super().__init__(
            sim=sim,
            cube_pos=cube_pos,
            drop_pos=drop_pos,
            pick_radius=pick_radius,
            drop_radius=drop_radius,
            timeout=timeout,
            label=label,
            start_pos=start_pos,
        )

        self._obstacle_cfg = obstacle_cfg
        self._arm_collection = arm_collection
        self._shoulder_pos = list(shoulder_pos)
        self._arm_length = arm_length

        # Collision counters
        self._total_hits: int = 0
        self._penalty_accumulated: float = 0.0

        # Spawn obstacle spheres
        self._obstacles: List[_ObstacleSphere] = []
        self._spawn_obstacles()

    # ── obstacle placement ────────────────────────────────────────────────────

    def _spawn_obstacles(self):
        """
        Sample n_obstacles positions in the reachable hemisphere, rejecting
        any that land within cfg.margin of the cube or drop zone.
        """
        cfg = self._obstacle_cfg
        rng = random.Random(cfg.seed)

        # Generate a generous pool and filter
        pool_size = cfg.n_obstacles * 8
        candidates = sample_hemisphere_positions(
            shoulder=self._shoulder_pos,
            arm_length=self._arm_length,
            n=pool_size,
            min_reach=cfg.min_reach,
            max_reach=cfg.max_reach,
            min_elevation=cfg.min_elevation,
            max_elevation=cfg.max_elevation,
            az_min=cfg.az_min,
            az_max=cfg.az_max,
            az_centre=cfg.az_centre,
            seed=cfg.seed,
        )
        rng.shuffle(candidates)

        placed = 0
        for pos in candidates:
            if placed >= cfg.n_obstacles:
                break
            # Reject if too close to cube, drop zone, or shoulder origin
            d_cube = math.sqrt(sum((pos[i] - self.cube_pos[i]) ** 2 for i in range(3)))
            d_drop = math.sqrt(sum((pos[i] - self.drop_pos[i]) ** 2 for i in range(3)))
            d_shoulder = math.sqrt(
                sum((pos[i] - self._shoulder_pos[i]) ** 2 for i in range(3))
            )
            if (
                d_cube < cfg.margin
                or d_drop < cfg.margin
                or d_shoulder < cfg.shoulder_margin
            ):
                continue

            radius = rng.uniform(cfg.radius_min, cfg.radius_max)
            sphere = _ObstacleSphere(
                sim=self.sim,
                pos=pos,
                radius=radius,
                label=f"{self.label}_obs{placed}",
            )
            self._obstacles.append(sphere)
            placed += 1

    # ── overridden _remove_all ────────────────────────────────────────────────

    def _remove_all(self):
        """Remove cube, drop zone dummy, AND all obstacle spheres."""
        super()._remove_all()
        for obs in self._obstacles:
            obs.remove()
        self._obstacles.clear()

    # ── overridden update ─────────────────────────────────────────────────────

    def update(
        self,
        wrist_pos: List[float],
        gripper_open: bool,
        dt: float,
    ) -> Optional[str]:
        """
        Extends TransportTrial.update() with per-frame obstacle collision checks.
        """
        # Run base phase logic first
        result = super().update(wrist_pos, gripper_open, dt)

        # Only check obstacles while the trial is still running
        if self._result is None or (self._result is not None and self._flash_t > 0):
            cfg = self._obstacle_cfg
            for obs in self._obstacles:
                new_hit = obs.check_collision(
                    self._arm_collection,
                    dt=dt,
                )
                if new_hit:
                    self._total_hits += 1
                    if cfg.penalty_on_hit:
                        self._penalty_accumulated += cfg.penalty_seconds

        return result

    # ── extra properties ──────────────────────────────────────────────────────

    @property
    def total_hits(self) -> int:
        """Total number of distinct obstacle contact events this trial."""
        return self._total_hits

    @property
    def penalty_accumulated(self) -> float:
        """Total penalty seconds accumulated (0 if penalty_on_hit is False)."""
        return self._penalty_accumulated

    @property
    def adjusted_duration(self) -> float:
        """
        Elapsed time + any accumulated penalty.  Used as the headline metric
        in the results dict when penalty_on_hit is True.
        """
        return self._elapsed + self._penalty_accumulated

    # ── overridden draw ───────────────────────────────────────────────────────

    def draw(
        self,
        surf: pygame.Surface,
        wrist_pos: List[float],
        fonts: dict,
        dt: float = 0.0,
    ):
        W, H = surf.get_size()

        # ── obstacle hit flash (brief red edge pulse) ─────────────────────────
        any_flashing = any(obs.is_flashing for obs in self._obstacles)
        if any_flashing and self._result is None:
            flash_frac = max(
                obs._hit_flash / _ObstacleSphere.HIT_FLASH_DURATION
                for obs in self._obstacles
                if obs.is_flashing
            )
            alpha = int(90 * flash_frac)
            flash_surf = pygame.Surface((W, H), pygame.SRCALPHA)
            flash_surf.fill((*C_OBSTACLE_HIT, alpha))
            surf.blit(flash_surf, (0, 0))

        # ── base HUD (phase panel, phase bars, dot row, distances) ───────────
        super().draw(surf, wrist_pos, fonts, dt)

        if self.finished and self._flash_t <= 0:
            return

        # ── obstacle info badge (top-right corner, above main panel) ─────────
        badge_w, badge_h = 310, 52
        # Align with the main transport panel (same px)
        panel_w = 310
        bx = W - panel_w - 10
        by = 210 - badge_h - 6  # sit just above the main panel

        badge = pygame.Surface((badge_w, badge_h), pygame.SRCALPHA)
        badge.fill((22, 22, 28, 190))
        surf.blit(badge, (bx, by))

        hit_col = C_FAIL if self._total_hits > 0 else C_TEXT_DIM
        pygame.draw.rect(surf, hit_col, (bx, by, badge_w, badge_h), 1)

        n_obs = len(self._obstacles)
        obs_txt = fonts["sm"].render(
            f"obstacles: {n_obs}  |  hits: {self._total_hits}", True, hit_col
        )
        surf.blit(obs_txt, (bx + 10, by + 8))

        if self._obstacle_cfg.penalty_on_hit and self._penalty_accumulated > 0:
            pen_txt = fonts["sm"].render(
                f"+{self._penalty_accumulated:.1f}s penalty", True, C_FAIL
            )
            surf.blit(pen_txt, (bx + 10, by + 26))
        elif self._total_hits == 0:
            clean_txt = fonts["sm"].render("clean path so far ✓", True, C_HOT)
            surf.blit(clean_txt, (bx + 10, by + 26))
        else:
            hit_detail = fonts["sm"].render(
                f"contact events this trial", True, C_TEXT_DIM
            )
            surf.blit(hit_detail, (bx + 10, by + 26))


# ═════════════════════════════════════════════════════════════════════════════
# ObstacleTransportExperiment
# ═════════════════════════════════════════════════════════════════════════════


class ObstacleTransportExperiment:
    """
    Orchestrates a sequence of ObstacleTransportTrials.

    Usage
    -----
    ::

        obs_cfg = ObstacleConfig(n_obstacles=12, radius_min=0.03, radius_max=0.09)

        exp = ObstacleTransportExperiment.from_random(
            sim,
            shoulder_pos=robot_shoulder_world,
            arm_length=ROBOT_ARM_LENGTH,
            n_trials=5,
            obstacle_cfg=obs_cfg,
        )

        # game loop
        exp.update(wrist_pos, gripper_open, dt, start_pos=arm_start)
        exp.draw(screen, wrist_pos, fonts, dt)

        # tweak difficulty mid-experiment (takes effect on next trial):
        exp.obstacle_cfg.n_obstacles = 20
        exp.obstacle_cfg.radius_max  = 0.12

    Parameters
    ----------
    sim
        CoppeliaSim sim handle.
    trials : list[dict]
        Each dict must contain:
          ``cube_pos``, ``drop_pos``, ``shoulder_pos``, ``arm_length``
        and may optionally contain:
          ``pick_radius``, ``drop_radius``, ``timeout``, ``label``
    obstacle_cfg : ObstacleConfig
        Shared configuration; mutate between trials to change difficulty.
    arm_collection : int
        CoppeliaSim collection handle covering all arm links.  Create once
        at startup and pass here — reused across all trials.
    start_pos : list[float] | None
        Default arm rest position for difficulty metrics.
    """

    def __init__(
        self,
        sim,
        trials: List[dict],
        obstacle_cfg: ObstacleConfig,
        arm_collection: int,
        start_pos: Optional[List[float]] = None,
    ):
        self.sim = sim
        self._trial_defs = trials
        self.obstacle_cfg = obstacle_cfg  # public – callers may mutate
        self._arm_collection = arm_collection
        self._default_start_pos = list(start_pos) if start_pos else None
        self._index = 0
        self._results: List[dict] = []
        self._active: Optional[ObstacleTransportTrial] = None
        self._start_time = time.time()
        self._trial_start = time.time()
        self._result_logged = False
        self._spawn_next()

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_random(
        cls,
        sim,
        shoulder_pos: List[float],
        arm_length: float,
        arm_collection: int,
        n_trials: int = 5,
        obstacle_cfg: Optional[ObstacleConfig] = None,
        pick_radius: float = 0.06,
        drop_radius: float = 0.06,
        timeout: float = 45.0,
        min_reach: float = 0.5,
        max_reach: float = 0.80,
        min_elevation: float = -15.0,
        max_elevation: float = 45.0,
        az_min: float = -45.0,
        az_max: float = 45.0,
        az_centre: float = -90.0,
        seed: int = None,
        start_pos: Optional[List[float]] = None,
    ) -> "ObstacleTransportExperiment":
        """
        Generate n_trials obstacle transport tasks with cube and drop positions
        sampled from the reachable hemisphere.

        Parameters
        ----------
        arm_collection : int
            CoppeliaSim collection handle covering all arm links.
        obstacle_cfg : ObstacleConfig | None
            Obstacle cloud configuration.  If None, a default ObstacleConfig()
            is used (10 obstacles, 3–8 cm radius).
        seed : int | None
            Controls both the cube/drop position sampling AND (if
            obstacle_cfg.seed is None) the obstacle placement.
        """
        if obstacle_cfg is None:
            obstacle_cfg = ObstacleConfig()

        rng = random.Random(seed)
        all_pos = sample_hemisphere_positions(
            shoulder=shoulder_pos,
            arm_length=arm_length,
            n=n_trials * 2,
            min_reach=min_reach,
            max_reach=max_reach,
            min_elevation=min_elevation,
            max_elevation=max_elevation,
            az_min=az_min,
            az_max=az_max,
            az_centre=az_centre,
            seed=seed,
        )
        rng.shuffle(all_pos)

        trials = []
        for i in range(n_trials):
            cube_pos = all_pos[i * 2]
            drop_pos = all_pos[i * 2 + 1]
            trials.append(
                {
                    "cube_pos": cube_pos,
                    "drop_pos": drop_pos,
                    "shoulder_pos": shoulder_pos,
                    "arm_length": arm_length,
                    "pick_radius": pick_radius,
                    "drop_radius": drop_radius,
                    "timeout": timeout,
                    "label": f"Obs-Transport {i+1}",
                }
            )
        return cls(
            sim,
            trials,
            obstacle_cfg=obstacle_cfg,
            arm_collection=arm_collection,
            start_pos=start_pos,
        )

    # ── spawn ─────────────────────────────────────────────────────────────────

    def _spawn_next(self, start_pos: Optional[List[float]] = None):
        if self._index >= len(self._trial_defs):
            self._active = None
            return
        cfg = self._trial_defs[self._index]
        sp = start_pos or self._default_start_pos

        # Each new trial takes a snapshot of obstacle_cfg so mid-trial changes
        # do not affect the currently running trial.
        import copy

        cfg_snapshot = copy.copy(self.obstacle_cfg)
        # Give each trial a distinct seed derived from the experiment seed so
        # obstacle layouts differ per trial (but are still reproducible).
        if cfg_snapshot.seed is not None:
            cfg_snapshot.seed = cfg_snapshot.seed + self._index * 1000

        self._active = ObstacleTransportTrial(
            sim=self.sim,
            cube_pos=cfg["cube_pos"],
            drop_pos=cfg["drop_pos"],
            obstacle_cfg=cfg_snapshot,
            arm_collection=self._arm_collection,
            shoulder_pos=cfg["shoulder_pos"],
            arm_length=cfg["arm_length"],
            pick_radius=cfg.get("pick_radius", 0.06),
            drop_radius=cfg.get("drop_radius", 0.06),
            timeout=cfg.get("timeout", 45.0),
            label=cfg.get("label", f"Obs-Transport {self._index + 1}"),
            start_pos=sp,
        )
        self._trial_start = time.time()

    # ── update ────────────────────────────────────────────────────────────────

    def update(
        self,
        wrist_pos: List[float],
        gripper_open: bool,
        dt: float,
        start_pos: Optional[List[float]] = None,
    ):
        """
        Call every sim step.

        Parameters
        ----------
        wrist_pos    : current world-space wrist position
        gripper_open : True = open / False = closed
        dt           : seconds since last call
        start_pos    : optional arm rest position for the *next* trial
        """
        if self._active is None:
            return

        result = self._active.update(wrist_pos, gripper_open, dt)

        if result is not None and not self._result_logged:
            self._result_logged = True
            trial = self._active
            self._results.append(
                {
                    "trial": self._index + 1,
                    "label": trial.label,
                    "result": result,
                    "duration": time.time() - self._trial_start,
                    "adjusted_duration": trial.adjusted_duration,
                    "phase_splits": trial.phase_splits,
                    "total_hits": trial.total_hits,
                    "penalty_accumulated": trial.penalty_accumulated,
                    "n_obstacles": len(trial._obstacles),
                    "cube_pos": self._trial_defs[self._index]["cube_pos"],
                    "drop_pos": self._trial_defs[self._index]["drop_pos"],
                    "start_pos": trial._start_pos,
                    "dist_start_to_cube": trial.dist_start_to_cube,
                    "dist_start_to_drop": trial.dist_start_to_drop,
                }
            )
            self._index += 1
            self._pending_start_pos = start_pos or wrist_pos

        self._maybe_advance(start_pos=start_pos)

    def _maybe_advance(self, start_pos: Optional[List[float]] = None):
        if (
            self._active is not None
            and self._active.finished
            and self._active._flash_t <= 0
        ):
            self._result_logged = False
            sp = start_pos or getattr(self, "_pending_start_pos", None)
            self._spawn_next(start_pos=sp)

    # ── draw ──────────────────────────────────────────────────────────────────

    def draw(
        self,
        surf: pygame.Surface,
        wrist_pos: List[float],
        fonts: dict,
        dt: float = 0.0,
    ):
        self._maybe_advance()
        W, H = surf.get_size()
        total = len(self._trial_defs)
        done = len(self._results)
        bar_w = W - 20

        pygame.draw.rect(surf, (45, 45, 55), (10, H - 18, bar_w, 8))
        if total:
            filled = int(bar_w * done / total)
            pygame.draw.rect(surf, C_SUCCESS, (10, H - 18, filled, 8))

        prog_txt = fonts["sm"].render(
            f"obs-transport {min(done+1, total)} / {total}  ·  "
            + (f"{done} done" if done else ""),
            True,
            C_TEXT_DIM,
        )
        surf.blit(prog_txt, (10, H - 34))

        if self._active is not None:
            self._active.draw(surf, wrist_pos, fonts, dt)

        if self.finished:
            _draw_summary(
                surf,
                self._results,
                fonts,
                title="OBSTACLE TRANSPORT COMPLETE",
                extra_col="total_hits",
                extra_label="hits",
            )

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def finished(self) -> bool:
        return self._active is None and self._index >= len(self._trial_defs)

    @property
    def results(self) -> List[dict]:
        return list(self._results)

    def summary(self) -> str:
        lines = ["─── Obstacle Transport Experiment Results ───"]
        for r in self._results:
            sp = r.get("phase_splits", {})
            d2c = r.get("dist_start_to_cube")
            d2d = r.get("dist_start_to_drop")
            hits = r.get("total_hits", 0)
            penalty = r.get("penalty_accumulated", 0.0)
            adj = r.get("adjusted_duration", r["duration"])
            dist_info = (
                f"  start→cube {d2c*100:.0f}cm  start→drop {d2d*100:.0f}cm"
                if d2c is not None
                else ""
            )
            splits_str = "  splits: " + "  ".join(
                f"{ph[:4]}={sp.get(ph, 0.0):.2f}s" for ph in _ALL_PHASES
            )
            penalty_str = f"  +{penalty:.1f}s penalty" if penalty > 0 else "  clean"
            lines.append(
                f"  {r['trial']:2d}. {r['label']:26s}  "
                f"{r['result']:8s}  {r['duration']:.2f}s  "
                f"(adj {adj:.2f}s)  hits={hits}"
            )
            if dist_info:
                lines.append(dist_info)
            lines.append(splits_str + penalty_str)
        n_ok = sum(1 for r in self._results if r["result"] == "success")
        total_hits = sum(r.get("total_hits", 0) for r in self._results)
        lines.append(
            f"\n  {n_ok}/{len(self._results)} delivered  "
            f"| {total_hits} total obstacle hits  "
            f"| total {time.time()-self._start_time:.1f}s"
        )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Drawing utilities (private)
# ─────────────────────────────────────────────────────────────────────────────
def _lerp_col(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _draw_arc(surf, colour, rect, start_angle, end_angle, width=2):
    if abs(end_angle - start_angle) < 0.01:
        return
    try:
        pygame.draw.arc(surf, colour, rect, start_angle, end_angle, width)
    except Exception:
        pass


def _draw_summary(
    surf: pygame.Surface,
    results: List[dict],
    fonts: dict,
    title: str = "EXPERIMENT COMPLETE",
    extra_col: Optional[str] = None,
    extra_label: Optional[str] = None,
):
    W, H = surf.get_size()
    overlay = pygame.Surface((W, H), pygame.SRCALPHA)
    overlay.fill((10, 10, 18, 200))
    surf.blit(overlay, (0, 0))

    title_surf = fonts["lg"].render(title, True, C_SUCCESS)
    surf.blit(title_surf, (W // 2 - title_surf.get_width() // 2, 60))

    n_ok = sum(1 for r in results if r["result"] == "success")

    # Build the headline score line; append aggregate hits if relevant.
    score_str = f"{n_ok} / {len(results)}  trials succeeded"
    if extra_col == "total_hits":
        total_hits = sum(r.get("total_hits", 0) for r in results)
        hit_col = C_FAIL if total_hits > 0 else C_HOT
        score_str += f"   |   {total_hits} total obstacle hits"
    else:
        hit_col = C_TEXT_BRT

    score = fonts["md"].render(score_str, True, C_TEXT_BRT)
    surf.blit(score, (W // 2 - score.get_width() // 2, 100))

    # If hits are non-zero, render the hit count again in a contrasting colour
    # so it stands out from the white success count.
    if extra_col == "total_hits" and total_hits > 0:
        hits_surf = fonts["md"].render(
            f"{total_hits} total obstacle hits", True, hit_col
        )
        # Overlay just the hits portion — find its x position within the line.
        # Simpler: draw a second, shorter line just below the score.
        surf.blit(hits_surf, (W // 2 - hits_surf.get_width() // 2, 122))

    for i, r in enumerate(results):
        col = C_SUCCESS if r["result"] == "success" else C_FAIL
        sp = r.get("phase_splits", {})
        splits_str = " | ".join(
            f"{ph[:4]} {sp.get(ph, 0.0):.1f}s" for ph in _ALL_PHASES
        )
        extra = ""
        if extra_col and extra_label:
            val = r.get(extra_col, 0)
            extra = f"  {extra_label}={val}"
        txt = (
            f"  {r['trial']:2d}.  {r['label']:24s}  "
            f"{r['result']:8s}  {r['duration']:.2f}s  [{splits_str}]{extra}"
        )
        lbl = fonts["sm"].render(txt, True, col)
        surf.blit(lbl, (W // 2 - lbl.get_width() // 2, 148 + i * 22))

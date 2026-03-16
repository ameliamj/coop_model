#!/usr/bin/env python3
"""
What? – Renderer for model evaluation logs.

What this script does
---------------------
- Loads evaluation logs from an `eval/` directory, or from a model directory
  that contains an `eval/` subdirectory.
- Displays an interactive figure with:
  - an x-position arena view showing each agent, the lever, and the reward port
  - cue indicators for lever cue and reward cue
  - cumulative return over time
  - normalized movement-action probabilities
  - normalized gaze-action probabilities
  - a right-side text panel with the current observation-derived state,
    sampled action, and predicted action at the current timestep
- Lets you step through timesteps and episodes with the keyboard.
- Can export every timestep of one episode or all episodes as PNG frames.
- Can also assemble those frames into `.mp4` videos when `ffmpeg` is installed.

Files this script expects
-------------------------
Required:
- `all_observations.pkl`

Optional but recommended:
- `all_positions.pkl`
- `all_move_actions.pkl`
- `all_gaze_actions.pkl`
- `all_action_probs.pkl`
- `all_rewards.pkl`
- `all_pulls.pkl`
- `all_coops.pkl`
- `args.npy`

How log discovery works
-----------------------
- If `--log-dir` directly contains `all_observations.pkl`, that directory is
  used.
- Otherwise, if `--log-dir/eval/all_observations.pkl` exists, the script uses
  that `eval/` directory automatically.

Interactive controls
--------------------
- `Right arrow`: next timestep
- `Left arrow`: previous timestep
- `Up arrow`: next episode
- `Down arrow`: previous episode
- `Space`: play/pause
- `Home`: first timestep
- `End`: last timestep

How actions are shown
---------------------
- `action(move)` and `action(gaze)` come from the sampled discrete actions in
  `all_move_actions.pkl` and `all_gaze_actions.pkl`.
- `predicted(move)` and `predicted(gaze)` are computed from
  `all_action_probs.pkl` by normalizing the raw action vector and taking the
  most probable action for movement and gaze separately.

How to run it
-------------
Interactive viewer:
    python3 maddpg/classic_log_sim_viewer.py --log-dir /path/to/model_or_eval_dir

Export frames and videos for all episodes:
    python3 maddpg/classic_log_sim_viewer.py \
        --log-dir /path/to/model_or_eval_dir \
        --export-all \
        --export-dir viewer_export \
        --export-fps 8

Export frames and video for one episode only:
    python3 maddpg/classic_log_sim_viewer.py \
        --log-dir /path/to/model_or_eval_dir \
        --export-episode 7 \
        --export-dir viewer_export_ep7 \
        --export-fps 8

Export PNG frames only, with no video creation:
    python3 maddpg/classic_log_sim_viewer.py \
        --log-dir /path/to/model_or_eval_dir \
        --export-all \
        --frames-only

Command-line options
--------------------
- `--log-dir`: model directory or eval directory containing the log files
- `--interval-ms`: playback interval for interactive mode
- `--reward-value`: reward amount used when reconstructing return over time
- `--step-penalty`: per-timestep baseline for return reconstruction
- `--gaze-penalty`: extra penalty added when gaze action is active
- `--rat-image`: optional custom rat image
- `--lever-image`: optional custom lever image
- `--reward-image`: optional custom reward image
- `--export-all`: export every timestep for every episode
- `--export-episode N`: export only one episode, using 1-based episode number
- `--export-dir`: output folder for exported frames/videos
- `--export-fps`: fps used for mp4 creation
- `--export-dpi`: resolution used for exported frames/video
- `--frames-only`: save PNG frames but skip mp4 creation
- `--stream-video`: write mp4 directly without saving PNG frames first
- `--cleanup-frames`: delete saved PNG frames after mp4 creation

Requirements and notes
----------------------
- Video creation requires `ffmpeg` to be available on your system `PATH`.
- If `ffmpeg` is missing, use `--frames-only` to still save every frame.
- Exported frames are written to:
  - `<export-dir>/frames/ep_0001/`
  - `<export-dir>/frames/ep_0002/`
  - etc.
- Exported videos are written to:
  - `<export-dir>/videos/ep_0001.mp4`
  - etc.
- When exporting all episodes, the script also creates:
  - `<export-dir>/videos/all_episodes.mp4`
"""


import argparse
import os
import pickle
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.animation import FFMpegWriter
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, DrawingArea, OffsetImage

try:
    import tkinter as tk
    from tkinter import filedialog
except Exception:
    tk = None
    filedialog = None


ARENA_X_MIN = -1.2
ARENA_X_MAX = 1.2
LEVER_X = -1
PORT_X = 1
Y_TOP = 0.10
Y_BOTTOM = -0.10
EPS = 1e-9
DEFAULT_RAT_IMAGE = None

AGENT_COLORS = {0: "tab:red", 1: "tab:blue"}


@dataclass
class AgentState:
    x_pos: float
    x_vel: float
    x_lever_rel: float
    x_port_rel: float
    x_other_rel: float
    reward_cue: Any
    lever_cue: Any
    lever_action_state: Any
    time_since_pull: Any


class LogSimulationViewer:
    def __init__(
        self,
        log_dir: str,
        interval_ms: int = 150,
        reward_value: float = 100.0,
        step_penalty: float = -1.0,
        gaze_penalty: float = 0.0,
        rat_image: Optional[str] = None,
        lever_image: Optional[str] = None,
        reward_image: Optional[str] = None,
    ):
        self.log_dir = self._resolve_log_dir(log_dir)
        self.interval_ms = interval_ms
        self.reward_value = reward_value
        self.step_penalty = step_penalty
        self.gaze_penalty = gaze_penalty
        self.rat_image = rat_image
        self.lever_image = lever_image
        self.reward_image = reward_image

        self.args_obj, self.args_dict = self._load_args()

        self.observations = self._load_pickle("all_observations.pkl")
        self.positions = self._load_pickle("all_positions.pkl", required=False)
        self.move_actions = self._load_pickle("all_move_actions.pkl", required=False)
        self.gaze_actions = self._load_pickle("all_gaze_actions.pkl", required=False)
        self.action_scores = self._load_pickle("all_action_probs.pkl", required=False)
        self.rewards = self._load_pickle("all_rewards.pkl", required=False)
        self.pulls = self._load_pickle("all_pulls.pkl", required=False) or {}
        self.coops = self._load_pickle("all_coops.pkl", required=False) or {}

        self.observations = self._to_ndarray(self.observations, "all_observations.pkl")
        self.positions = self._to_optional_ndarray(self.positions)
        self.move_actions = self._to_optional_ndarray(self.move_actions)
        self.gaze_actions = self._to_optional_ndarray(self.gaze_actions)
        self.action_scores = self._to_optional_ndarray(self.action_scores)
        self.rewards = self._to_optional_ndarray(self.rewards)

        if self.observations.ndim != 4:
            raise ValueError(
                f"Expected all_observations.pkl to have 4 dimensions, got shape {self.observations.shape}"
            )

        self.episodes = list(range(self.observations.shape[0]))
        if not self.episodes:
            raise ValueError("No episodes found in all_observations.pkl")

        self.episode_idx = 0
        self.t = 0
        self.playing = False

        self.obs_dim = self.observations.shape[-1]
        self.active_agents = self._infer_active_agents()
        self.agent_names = self._infer_agent_names()
        self.cue_layout = self._infer_cue_layout()
        self.num_move_actions = self._infer_num_move_actions()
        self.move_labels = self._move_labels(self.num_move_actions)
        self.gaze_labels = ["no_gaze", "gaze"]
        self.rat_rgba = self._load_rat_image(
            self.rat_image or (DEFAULT_RAT_IMAGE if DEFAULT_RAT_IMAGE and os.path.exists(DEFAULT_RAT_IMAGE) else None)
        )

        self.fig = None
        self.ax_env = None
        self.ax_state = None
        self.ax_reward = None
        self.ax_move_probs = None
        self.ax_gaze_probs = None
        self.timer = None

        self.agent_paths: Dict[int, Any] = {}
        self.rat_artists: Dict[int, Any] = {}
        self.state_text = None
        self.title_text = None
        self.reward_lines: Dict[int, Any] = {}
        self.total_reward_line = None
        self.reward_cursor = None
        self.lever_cue_markers: Dict[int, Any] = {}
        self.reward_cue_markers: Dict[int, Any] = {}
        self.move_prob_bars: Dict[int, Any] = {}
        self.gaze_prob_bars: Dict[int, Any] = {}

    @staticmethod
    def _episode_label(ep: int) -> str:
        return f"ep_{ep + 1:04d}"

    def _model_label(self) -> str:
        base = os.path.basename(self.log_dir.rstrip(os.sep))
        if base == "eval":
            base = os.path.basename(os.path.dirname(self.log_dir.rstrip(os.sep)))
        return base or "model"

    @staticmethod
    def _resolve_log_dir(log_dir: str) -> str:
        path = os.path.abspath(os.path.expanduser(log_dir))
        if os.path.exists(os.path.join(path, "all_observations.pkl")):
            return path
        eval_path = os.path.join(path, "eval")
        if os.path.exists(os.path.join(eval_path, "all_observations.pkl")):
            return eval_path
        raise FileNotFoundError(
            f"Could not find all_observations.pkl in {path} or {eval_path}"
        )

    def _load_pickle(self, filename: str, required: bool = True):
        path = os.path.join(self.log_dir, filename)
        if not os.path.exists(path):
            if required:
                raise FileNotFoundError(f"Missing required log file: {path}")
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def _to_ndarray(value: Any, label: str) -> np.ndarray:
        arr = np.asarray(value)
        if arr.dtype == object:
            raise ValueError(f"{label} is not array-based in this repository format.")
        return arr

    @staticmethod
    def _to_optional_ndarray(value: Any) -> Optional[np.ndarray]:
        if value is None:
            return None
        arr = np.asarray(value)
        if arr.dtype == object:
            return None
        return arr

    def _load_args(self) -> Tuple[Optional[Any], Dict[str, Any]]:
        candidates = [
            os.path.join(self.log_dir, "args.npy"),
            os.path.join(os.path.dirname(self.log_dir), "args.npy"),
        ]
        for candidate in candidates:
            if not os.path.exists(candidate):
                continue
            raw = np.load(candidate, allow_pickle=True)
            try:
                obj = raw.item()
            except Exception:
                obj = raw
            if hasattr(obj, "__dict__"):
                return obj, vars(obj).copy()
            if isinstance(obj, dict):
                return obj, obj.copy()
        return None, {}

    def _infer_active_agents(self) -> List[int]:
        if "n_agents" in self.args_dict:
            n_agents = int(self.args_dict["n_agents"])
            return list(range(max(1, min(n_agents, self.observations.shape[1]))))

        active = []
        for agent_id in range(self.observations.shape[1]):
            if np.any(np.abs(self.observations[:, agent_id]) > EPS):
                active.append(agent_id)
        return active or [0]

    def _infer_agent_names(self) -> Dict[int, str]:
        names = {}
        for agent_id in range(self.observations.shape[1]):
            names[agent_id] = f"agent_{agent_id}"
        if len(self.active_agents) == 1:
            names[self.active_agents[0]] = "agent_0"
        return names

    def _infer_cue_layout(self) -> Tuple[int, Optional[int], Optional[int]]:
        reward_idx = -1
        lever_idx = None
        lever_action_idx = None

        lever_cue_enabled = str(self.args_dict.get("lever_cue", "none")) != "none"
        lever_action_enabled = bool(self.args_dict.get("lever_action", False))

        if lever_cue_enabled and lever_action_enabled:
            reward_idx = -3
            lever_idx = -2
            lever_action_idx = -1
        elif lever_cue_enabled:
            reward_idx = -2
            lever_idx = -1
        elif lever_action_enabled:
            reward_idx = -2
            lever_action_idx = -1
        return reward_idx, lever_idx, lever_action_idx

    def _infer_num_move_actions(self) -> int:
        if self.action_scores is not None and self.action_scores.shape[-1] >= 3:
            return max(1, self.action_scores.shape[-1] - 2)
        if "num_actions" in self.args_dict:
            return int(self.args_dict["num_actions"])
        return 3

    @staticmethod
    def _move_labels(num_actions: int) -> List[str]:
        common = {
            3: ["stay", "left", "right"],
            4: ["stay", "left", "right", "press"],
            5: ["stay", "left", "right", "down", "up"],
        }
        return common.get(num_actions, [f"move_{idx}" for idx in range(num_actions)])

    def _move_action_name(self, action: Any) -> str:
        if action == "n/a":
            return "n/a"
        try:
            idx = int(action)
        except Exception:
            return str(action)
        if 0 <= idx < len(self.move_labels):
            return self.move_labels[idx]
        return f"move_{idx}"

    def _gaze_action_name(self, action: Any) -> str:
        if action == "n/a":
            return "n/a"
        try:
            idx = int(action)
        except Exception:
            return str(action)
        if 0 <= idx < len(self.gaze_labels):
            return self.gaze_labels[idx]
        return f"gaze_{idx}"

    def _episode(self) -> int:
        return self.episodes[self.episode_idx]

    def _episode_len(self, ep: int) -> int:
        return int(self.observations.shape[2])

    @staticmethod
    def _safe_int(value: Any, default: Any = "n/a") -> Any:
        try:
            return int(round(float(value)))
        except Exception:
            return default

    def _obs_at(self, ep: int, agent_id: int, t: int) -> np.ndarray:
        return np.asarray(self.observations[ep, agent_id, t], dtype=float)

    def _position_at(self, ep: int, agent_id: int, t: int) -> np.ndarray:
        if self.positions is not None:
            pos = np.asarray(self.positions[ep, agent_id, t], dtype=float).reshape(-1)
            if pos.size >= 2:
                return pos[:2]
        obs = self._obs_at(ep, agent_id, t)
        if obs.size >= 2:
            return obs[:2]
        return np.zeros(2, dtype=float)

    def _other_agent_id(self, agent_id: int) -> Optional[int]:
        others = [aid for aid in self.active_agents if aid != agent_id]
        return others[0] if others else None

    def _time_since_pull(self, ep: int, agent_id: int, t: int) -> Any:
        pulls = self.pulls.get(ep, {})
        if not isinstance(pulls, dict):
            return "n/a"
        steps = [step for step in pulls.get(agent_id, []) if step <= t]
        if not steps:
            return "n/a"
        return t - max(steps)

    def _state_from_logs(self, ep: int, t: int, agent_id: int) -> AgentState:
        obs = self._obs_at(ep, agent_id, t)
        pos = self._position_at(ep, agent_id, t)

        prev_t = max(0, t - 1)
        prev_pos = self._position_at(ep, agent_id, prev_t)

        x_pos = float(pos[0]) if pos.size >= 1 else 0.0
        if obs.size >= 4:
            x_vel = float(obs[2])
        else:
            x_vel = float(x_pos - prev_pos[0])

        x_lever_rel = float(obs[4]) if obs.size >= 6 else float(LEVER_X - x_pos)
        x_port_rel = float(obs[6]) if obs.size >= 8 else float(PORT_X - x_pos)

        if obs.size >= 10:
            x_other_rel = float(obs[8])
        else:
            other_id = self._other_agent_id(agent_id)
            if other_id is None:
                x_other_rel = 0.0
            else:
                other_x = float(self._position_at(ep, other_id, t)[0])
                x_other_rel = other_x - x_pos

        reward_idx, lever_idx, lever_action_idx = self.cue_layout
        reward_cue = self._safe_int(obs[reward_idx]) if obs.size >= abs(reward_idx) else "n/a"
        lever_cue = "n/a"
        if lever_idx is not None and obs.size >= abs(lever_idx):
            lever_cue = self._safe_int(obs[lever_idx])
        lever_action_state = "n/a"
        if lever_action_idx is not None and obs.size >= abs(lever_action_idx):
            lever_action_state = self._safe_int(obs[lever_action_idx])

        return AgentState(
            x_pos=x_pos,
            x_vel=x_vel,
            x_lever_rel=x_lever_rel,
            x_port_rel=x_port_rel,
            x_other_rel=x_other_rel,
            reward_cue=reward_cue,
            lever_cue=lever_cue,
            lever_action_state=lever_action_state,
            time_since_pull=self._time_since_pull(ep, agent_id, t),
        )

    def _actions_at_t(self, ep: int, t: int, agent_id: int) -> Tuple[Any, Any]:
        move_action = "n/a"
        gaze_action = "n/a"
        if self.move_actions is not None:
            move_action = self._safe_int(self.move_actions[ep, agent_id, t])
        if self.gaze_actions is not None:
            gaze_action = self._safe_int(self.gaze_actions[ep, agent_id, t])
        return move_action, gaze_action

    @staticmethod
    def _scores_to_probs(scores: np.ndarray) -> np.ndarray:
        arr = np.asarray(scores, dtype=float).reshape(-1)
        if arr.size == 0:
            return arr
        arr = np.where(np.isfinite(arr), arr, -1e9)
        if np.all(arr >= 0):
            total = np.sum(arr)
            if total > EPS and abs(total - 1.0) < 1e-4:
                return arr
        arr = arr - np.max(arr)
        exp = np.exp(arr)
        total = np.sum(exp)
        if not np.isfinite(total) or total <= EPS:
            return np.full(arr.shape, 1.0 / arr.size)
        return exp / total

    def _policy_probs_at_t(self, ep: int, t: int, agent_id: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.action_scores is None:
            return np.zeros(self.num_move_actions), np.zeros(2)
        raw = np.asarray(self.action_scores[ep, agent_id, t], dtype=float).reshape(-1)
        if raw.size < self.num_move_actions:
            return np.zeros(self.num_move_actions), np.zeros(2)
        move_scores = raw[: self.num_move_actions]
        gaze_scores = raw[-2:] if raw.size >= self.num_move_actions + 2 else np.zeros(2)
        return self._scores_to_probs(move_scores), self._scores_to_probs(gaze_scores)

    def _predicted_actions_at_t(self, ep: int, t: int, agent_id: int) -> Tuple[str, str]:
        move_probs, gaze_probs = self._policy_probs_at_t(ep, t, agent_id)
        move_pred = "n/a"
        gaze_pred = "n/a"
        if move_probs.size:
            move_pred = self.move_labels[int(np.argmax(move_probs))]
        if gaze_probs.size:
            gaze_pred = self.gaze_labels[int(np.argmax(gaze_probs))]
        return move_pred, gaze_pred

    def _reward_event(self, ep: int, t: int, agent_id: int) -> bool:
        if self.rewards is None:
            return False
        reward = float(self.rewards[ep, agent_id, t])
        threshold = max(5.0, 0.5 * self.reward_value)
        return reward >= threshold

    def _event_flags(self, ep: int, t: int) -> Dict[str, Any]:
        pulls = self.pulls.get(ep, {})
        return {
            "pull_0": t in pulls.get(0, []) if isinstance(pulls, dict) else False,
            "pull_1": t in pulls.get(1, []) if isinstance(pulls, dict) else False,
            "reward_0": self._reward_event(ep, t, 0),
            "reward_1": self._reward_event(ep, t, 1),
            "coop": self._coop_event(ep, t),
        }

    def _coop_event(self, ep: int, t: int) -> bool:
        coops = self.coops.get(ep, [])
        if not isinstance(coops, (list, tuple)):
            return False
        for item in coops:
            if isinstance(item, (list, tuple, np.ndarray)) and t in item:
                return True
        return False

    def _arena_y(self, agent_id: int) -> float:
        return Y_TOP if agent_id == 0 else Y_BOTTOM

    def _return_cumsum(self, ep: int, ep_len: int, agent_id: int) -> List[float]:
        running = 0.0
        out = []
        for t in range(ep_len):
            running += self.step_penalty
            if self._reward_event(ep, t, agent_id):
                running += self.reward_value
            _, gaze_action = self._actions_at_t(ep, t, agent_id)
            if gaze_action == 1:
                running += self.gaze_penalty
            out.append(running)
        return out

    @staticmethod
    def _transparent_border_background(image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image).copy()
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            return arr

        if arr.shape[2] == 3:
            if np.issubdtype(arr.dtype, np.integer):
                alpha = np.full(arr.shape[:2], 255, dtype=arr.dtype)
            else:
                alpha = np.ones(arr.shape[:2], dtype=arr.dtype)
            arr = np.dstack([arr, alpha])

        rgb = arr[..., :3]
        if np.issubdtype(arr.dtype, np.integer):
            threshold = 4
            alpha_zero = 0
        else:
            threshold = 0.02
            alpha_zero = 0.0

        dark = np.all(rgb <= threshold, axis=2)
        visited = np.zeros(dark.shape, dtype=bool)
        stack = []
        height, width = dark.shape

        def enqueue(y: int, x: int) -> None:
            if 0 <= y < height and 0 <= x < width and dark[y, x] and not visited[y, x]:
                visited[y, x] = True
                stack.append((y, x))

        for x in range(width):
            enqueue(0, x)
            enqueue(height - 1, x)
        for y in range(height):
            enqueue(y, 0)
            enqueue(y, width - 1)

        while stack:
            y, x = stack.pop()
            enqueue(y - 1, x)
            enqueue(y + 1, x)
            enqueue(y, x - 1)
            enqueue(y, x + 1)

        arr[..., 3][visited] = alpha_zero
        return arr

    def _load_rat_image(self, image_path: Optional[str]) -> Optional[np.ndarray]:
        if not image_path or not os.path.exists(image_path):
            return None
        return self._transparent_border_background(plt.imread(image_path))

    def _create_icon_abox(
        self,
        image_path: Optional[str],
        drawing_area: DrawingArea,
        x: float,
        y: float,
        zoom: float,
    ) -> AnnotationBbox:
        if image_path and os.path.exists(image_path):
            arr = plt.imread(image_path)
            box = OffsetImage(arr, zoom=zoom)
        else:
            box = drawing_area
        abox = AnnotationBbox(box, (x, y), frameon=False, box_alignment=(0.5, 0.5), zorder=6)
        self.ax_env.add_artist(abox)
        return abox

    def _tinted_rat_image(self, color: str) -> np.ndarray:
        arr = np.asarray(self.rat_rgba).copy()
        rgb = arr[..., :3].astype(float)

        color_rgb = np.array(plt.matplotlib.colors.to_rgb(color), dtype=float)

        if np.issubdtype(arr.dtype, np.integer):
            maxv = np.iinfo(arr.dtype).max
            rgb = rgb / maxv
        else:
            maxv = 1.0

        brightness = rgb.mean(axis=2, keepdims=True)
        tinted = 0.35 * rgb + 0.65 * brightness * color_rgb.reshape(1, 1, 3)
        tinted = np.clip(tinted, 0.0, 1.0)

        if np.issubdtype(arr.dtype, np.integer):
            arr[..., :3] = (tinted * maxv).astype(arr.dtype)
        else:
            arr[..., :3] = tinted.astype(arr.dtype)

        return arr

    def _create_rat_artist(self, x: float, y: float, color: str):
        if self.rat_rgba is not None:
            rat_img = self._tinted_rat_image(color)
            w = 0.16
            h = 0.10
            artist = self.ax_env.imshow(
                rat_img,
                extent=(x - w / 2, x + w / 2, y - h / 2, y + h / 2),
                zorder=6,
                interpolation="nearest",
            )
            return artist

        # Fallback simple dot if no rat image exists
        artist = self.ax_env.scatter([x], [y], s=180, c=[color], edgecolors="black", zorder=6)
        return artist

    def _move_rat_artist(self, artist, x: float, y: float):
        if self.rat_rgba is not None:
            w = 0.16
            h = 0.10
            artist.set_extent((x - w / 2, x + w / 2, y - h / 2, y + h / 2))
        else:
            artist.set_offsets(np.array([[x, y]]))

    def _lever_drawing(self) -> DrawingArea:
        da = DrawingArea(26, 26, 0, 0)
        da.add_artist(patches.Rectangle((8, 2), 10, 5, facecolor="#7a5d2f", edgecolor="black", lw=1))
        da.add_artist(Line2D([13, 13], [7, 18], color="#8f8f8f", lw=2.2))
        da.add_artist(patches.Circle((13, 20), 3.2, facecolor="#d62f2f", edgecolor="black", lw=1))
        return da

    def _droplet_drawing(self) -> DrawingArea:
        da = DrawingArea(24, 26, 0, 0)
        da.add_artist(patches.Circle((12, 9), 7, facecolor="#4aa3df", edgecolor="black", lw=1))
        da.add_artist(
            patches.Polygon(
                [[12, 23], [6, 13], [18, 13]],
                closed=True,
                facecolor="#4aa3df",
                edgecolor="black",
                lw=1,
            )
        )
        return da

    def _build_figure(self):
        self.fig = plt.figure(figsize=(14.4, 8.8))
        gs = self.fig.add_gridspec(
            3, 2, width_ratios=[2.2, 1.18], height_ratios=[1.95, 0.98, 0.92], hspace=0.28, wspace=0.18
        )
        self.ax_env = self.fig.add_subplot(gs[0, 0])
        self.ax_reward = self.fig.add_subplot(gs[1, 0])
        probs_gs = gs[2, 0].subgridspec(1, 2, wspace=0.28)
        self.ax_move_probs = self.fig.add_subplot(probs_gs[0, 0])
        self.ax_gaze_probs = self.fig.add_subplot(probs_gs[0, 1])
        self.ax_state = self.fig.add_subplot(gs[:, 1])
        self.fig.subplots_adjust(top=0.90, bottom=0.08, left=0.07, right=0.98)

        self._draw_static_arena()

        for aid in self.active_agents:
            y = self._arena_y(aid)
            path, = self.ax_env.plot([], [], "-", color=AGENT_COLORS.get(aid, "tab:blue"), lw=1.8, alpha=0.6)
            self.agent_paths[aid] = path
            self.rat_artists[aid] = self._create_rat_artist(0.0, y, AGENT_COLORS.get(aid, "tab:blue"))

        env_legend = []
        for aid in self.active_agents:
            env_legend.append(
                Line2D([0], [0], color=AGENT_COLORS.get(aid, "tab:blue"), lw=2, label=self.agent_names[aid])
            )
        env_legend.extend(
            [
                Line2D([0], [0], marker="s", color="w", markerfacecolor="#d62f2f", markeredgecolor="black", ms=7, lw=0, label="lever"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#4aa3df", markeredgecolor="black", ms=7, lw=0, label="reward"),
            ]
        )
        self.ax_env.legend(
            handles=env_legend,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=min(2, len(env_legend)),
            fontsize=8,
            frameon=False,
            borderaxespad=0.0,
        )
        self.title_text = self.fig.suptitle("", y=0.965, fontsize=12)

        self.ax_state.set_axis_off()
        self.state_text = self.ax_state.text(
            0.02,
            0.98,
            "",
            va="top",
            ha="left",
            family="monospace",
            fontsize=8.5,
        )

        self.ax_reward.set_title("Return Over Time", fontsize=10, pad=12)
        self.ax_reward.set_xlabel("timestep")
        self.ax_reward.set_ylabel("cumulative return")
        self.ax_reward.grid(alpha=0.25)

        for aid in self.active_agents:
            line, = self.ax_reward.plot([], [], color=AGENT_COLORS.get(aid, "tab:blue"), lw=2, label=self.agent_names[aid])
            self.reward_lines[aid] = line
        self.total_reward_line, = self.ax_reward.plot(
            [], [], color="black", lw=2, ls="--", label="total"
        )
        self.reward_cursor = self.ax_reward.axvline(0, color="black", ls="--", lw=1)
        self.ax_reward.legend(loc="upper left", fontsize=8)

        self._build_prob_axes()

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self.timer = self.fig.canvas.new_timer(interval=self.interval_ms)
        self.timer.add_callback(self._tick)
        self.timer.start()

    def _build_prob_axes(self):
        width = 0.36 if len(self.active_agents) <= 1 else 0.32

        move_x = np.arange(len(self.move_labels))
        gaze_x = np.arange(len(self.gaze_labels))

        self.ax_move_probs.set_title("Movement Probabilities", fontsize=10, pad=8)
        self.ax_move_probs.set_ylabel("probability", fontsize=9)
        self.ax_move_probs.set_ylim(0.0, 1.0)
        self.ax_move_probs.set_xticks(move_x)
        self.ax_move_probs.set_xticklabels(self.move_labels, fontsize=8)
        self.ax_move_probs.grid(axis="y", alpha=0.22)
        self.ax_move_probs.tick_params(axis="y", labelsize=8)

        self.ax_gaze_probs.set_title("Gaze Probabilities", fontsize=10, pad=8)
        self.ax_gaze_probs.set_ylim(0.0, 1.0)
        self.ax_gaze_probs.set_xticks(gaze_x)
        self.ax_gaze_probs.set_xticklabels(self.gaze_labels, fontsize=8)
        self.ax_gaze_probs.grid(axis="y", alpha=0.22)
        self.ax_gaze_probs.tick_params(axis="y", labelsize=8)

        offsets = (
            [0.0]
            if len(self.active_agents) == 1
            else np.linspace(-width / 2, width / 2, len(self.active_agents))
        )
        for idx, aid in enumerate(self.active_agents):
            color = AGENT_COLORS.get(aid, "tab:blue")
            move_positions = move_x + offsets[idx]
            gaze_positions = gaze_x + offsets[idx]
            self.move_prob_bars[aid] = self.ax_move_probs.bar(
                move_positions,
                np.zeros(len(self.move_labels)),
                width=width,
                color=color,
                alpha=0.85,
                label=self.agent_names[aid],
            )
            self.gaze_prob_bars[aid] = self.ax_gaze_probs.bar(
                gaze_positions,
                np.zeros(len(self.gaze_labels)),
                width=width,
                color=color,
                alpha=0.85,
                label=self.agent_names[aid],
            )
        self.ax_move_probs.legend(fontsize=8, frameon=False, loc="upper right")
        self.ax_gaze_probs.legend(fontsize=8, frameon=False, loc="upper right")

    def _draw_static_arena(self):
        self.ax_env.set_xlim(ARENA_X_MIN - 0.03, ARENA_X_MAX + 0.03)
        self.ax_env.set_ylim(-0.26, 0.20)
        self.ax_env.set_aspect("equal", adjustable="box")

        # Remove overlapping labels
        self.ax_env.set_xlabel("x position", fontsize=10, labelpad=10)
        self.ax_env.set_ylabel("")
        self.ax_env.tick_params(axis="x", labelsize=9)
        self.ax_env.tick_params(axis="y", left=False, labelleft=False)

        arena = patches.Rectangle(
            (ARENA_X_MIN, -0.14),
            ARENA_X_MAX - ARENA_X_MIN,
            0.28,
            linewidth=2,
            edgecolor="black",
            facecolor="#f2f2f2",
            alpha=0.9,
        )
        self.ax_env.add_patch(arena)

        self.ax_env.axhline(Y_TOP, color="#bbbbbb", lw=1)
        self.ax_env.axhline(Y_BOTTOM, color="#bbbbbb", lw=1)

        for y in [Y_TOP, Y_BOTTOM]:
            self._create_icon_abox(self.lever_image, self._lever_drawing(), LEVER_X, y, zoom=0.24)
            self._create_icon_abox(self.reward_image, self._droplet_drawing(), PORT_X, y, zoom=0.24)

        self.ax_env.text(
            LEVER_X,
            -0.175,
            "lever",
            ha="center",
            va="top",
            fontsize=8,
            clip_on=False,
        )
        self.ax_env.text(
            PORT_X,
            -0.175,
            "reward",
            ha="center",
            va="top",
            fontsize=8,
            clip_on=False,
        )

        for aid in self.active_agents:
            y = self._arena_y(aid)

            self.lever_cue_markers[aid] = patches.Circle(
                (LEVER_X, y + 0.055),
                radius=0.022,
                facecolor="white",
                edgecolor="#946200",
                linewidth=1.5,
                zorder=7,
            )
            self.reward_cue_markers[aid] = patches.Circle(
                (PORT_X, y + 0.055),
                radius=0.022,
                facecolor="white",
                edgecolor="#2c6b3b",
                linewidth=1.5,
                zorder=7,
            )

            self.ax_env.add_patch(self.lever_cue_markers[aid])
            self.ax_env.add_patch(self.reward_cue_markers[aid])

    def _update_frame(self):
        ep = self._episode()
        ep_len = self._episode_len(ep)
        if ep_len <= 0:
            return

        self.t = max(0, min(self.t, ep_len - 1))

        state_blocks = []
        for aid in self.active_agents:
            s = self._state_from_logs(ep, self.t, aid)
            y = self._arena_y(aid)

            # Actually move the rat sprite
            self._move_rat_artist(self.rat_artists[aid], s.x_pos, y)

            xs = [self._state_from_logs(ep, i, aid).x_pos for i in range(self.t + 1)]
            ys = [y] * len(xs)
            self.agent_paths[aid].set_data(xs, ys)

            move_action, gaze_action = self._actions_at_t(ep, self.t, aid)
            move_action_name = self._move_action_name(move_action)
            gaze_action_name = self._gaze_action_name(gaze_action)
            pred_move_action, pred_gaze_action = self._predicted_actions_at_t(ep, self.t, aid)

            state_blocks.append(
                "\n".join(
                    [
                        f"{self.agent_names[aid]}",
                        f"  x_pos            : {s.x_pos: .4f}",
                        f"  x_vel            : {s.x_vel: .4f}",
                        f"  x_lever_rel      : {s.x_lever_rel: .4f}",
                        f"  x_reward_rel     : {s.x_port_rel: .4f}",
                        f"  x_other_rel      : {s.x_other_rel: .4f}",
                        f"  reward_cue       : {s.reward_cue}",
                        f"  lever_cue        : {s.lever_cue}",
                        f"  lever_action_obs : {s.lever_action_state}",
                        f"  time_since_pull  : {s.time_since_pull}",
                        f"  action(move)     : {move_action_name}",
                        f"  action(gaze)     : {gaze_action_name}",
                        f"  predicted(move)  : {pred_move_action}",
                        f"  predicted(gaze)  : {pred_gaze_action}",
                    ]
                )
            )

        for aid in self.active_agents:
            cue_state = self._state_from_logs(ep, self.t, aid)
            lever_on = cue_state.lever_cue == 1
            reward_on = cue_state.reward_cue == 1

            self.lever_cue_markers[aid].set_facecolor("#f2c94c" if lever_on else "white")
            self.reward_cue_markers[aid].set_facecolor("#4caf50" if reward_on else "white")
            self.lever_cue_markers[aid].set_alpha(1.0 if lever_on else 0.45)
            self.reward_cue_markers[aid].set_alpha(1.0 if reward_on else 0.45)

        flags = self._event_flags(ep, self.t)
        events_line = (
            f"events @ t={self.t}:\n"
            f"  pull0={flags['pull_0']}  pull1={flags['pull_1']}\n"
            f"  reward0={flags['reward_0']}  reward1={flags['reward_1']}  coop={flags['coop']}"
        )

        self.title_text.set_text(
            f"Episode {ep + 1} | timestep {self.t + 1}/{ep_len} | {'PLAY' if self.playing else 'PAUSE'}"
        )

        notes = (
            "Controls\n"
            "  Left/Right : step\n"
            "  Up/Down    : episode\n"
            "  Space      : play/pause\n"
            "\n"
        )

        self.state_text.set_text("\n\n".join(state_blocks + [events_line, notes]))

        self._update_prob_axes(ep)

        xs = list(range(ep_len))
        ymax = 1.0
        ymin = 0.0
        total = [0.0] * ep_len
        for aid in self.active_agents:
            cumsum = self._return_cumsum(ep, ep_len, aid)
            self.reward_lines[aid].set_data(xs, cumsum)
            total = [a + b for a, b in zip(total, cumsum)]
            if cumsum:
                ymax = max(ymax, max(cumsum))
                ymin = min(ymin, min(cumsum))
        self.total_reward_line.set_data(xs, total)
        if total:
            ymax = max(ymax, max(total))
            ymin = min(ymin, min(total))
        self.ax_reward.set_xlim(0, max(1, ep_len - 1))
        if ymax == ymin:
            ymax += 1.0
            ymin -= 1.0
        pad = max(1.0, 0.05 * (ymax - ymin))
        self.ax_reward.set_ylim(ymin - pad, ymax + pad)
        self.reward_cursor.set_xdata([self.t, self.t])

        self.fig.canvas.draw_idle()

    def _update_prob_axes(self, ep: int):
        for aid in self.active_agents:
            move_probs, gaze_probs = self._policy_probs_at_t(ep, self.t, aid)
            for idx, bar in enumerate(self.move_prob_bars[aid]):
                bar.set_height(float(move_probs[idx]) if idx < len(move_probs) else 0.0)
            for idx, bar in enumerate(self.gaze_prob_bars[aid]):
                bar.set_height(float(gaze_probs[idx]) if idx < len(gaze_probs) else 0.0)

    def _tick(self):
        if not self.playing:
            return
        ep = self._episode()
        ep_len = self._episode_len(ep)
        if self.t < ep_len - 1:
            self.t += 1
        else:
            self.playing = False
        self._update_frame()

    def _on_key(self, event):
        ep = self._episode()
        ep_len = self._episode_len(ep)

        if event.key == "right":
            self.playing = False
            self.t = min(ep_len - 1, self.t + 1)
        elif event.key == "left":
            self.playing = False
            self.t = max(0, self.t - 1)
        elif event.key == " ":
            self.playing = not self.playing
        elif event.key == "up":
            self.playing = False
            self.episode_idx = min(len(self.episodes) - 1, self.episode_idx + 1)
            self.t = 0
        elif event.key == "down":
            self.playing = False
            self.episode_idx = max(0, self.episode_idx - 1)
            self.t = 0
        elif event.key == "home":
            self.playing = False
            self.t = 0
        elif event.key == "end":
            self.playing = False
            self.t = max(0, ep_len - 1)
        else:
            return

        self._update_frame()

    def run(self):
        self._build_figure()
        self._update_frame()
        plt.show()

    def export_frames_and_video(
        self,
        output_dir: str,
        episode: Optional[int] = None,
        fps: int = 8,
        make_video: bool = True,
        dpi: int = 160,
        stream_video: bool = False,
        cleanup_frames: bool = False,
    ) -> None:
        self._build_figure()
        if self.timer is not None:
            self.timer.stop()

        episodes = [episode] if episode is not None else self.episodes
        output_dir = os.path.abspath(os.path.expanduser(output_dir))
        model_label = self._model_label()

        if make_video and stream_video:
            self._render_videos_direct(output_dir, episodes, fps, dpi)
            plt.close(self.fig)
            return

        frames_root = os.path.join(output_dir, "frames")
        os.makedirs(frames_root, exist_ok=True)

        for ep in episodes:
            ep_dir = os.path.join(frames_root, f"{model_label}_{self._episode_label(ep)}")
            os.makedirs(ep_dir, exist_ok=True)

            self.episode_idx = ep
            ep_len = self._episode_len(ep)
            for t in range(ep_len):
                self.t = t
                self.playing = False
                self._update_frame()
                frame_path = os.path.join(ep_dir, f"frame_{t + 1:04d}.png")
                self.fig.savefig(frame_path, dpi=dpi)

        if make_video:
            self._render_videos_from_frames(frames_root, output_dir, episodes, fps)
            if cleanup_frames:
                shutil.rmtree(frames_root, ignore_errors=True)

        plt.close(self.fig)

    def _render_videos_direct(
        self,
        output_dir: str,
        episodes: List[int],
        fps: int,
        dpi: int,
    ) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(
                "Direct mp4 export requires `ffmpeg` on PATH. If it is unavailable, use --frames-only or install ffmpeg."
            )

        videos_dir = os.path.join(output_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)
        model_label = self._model_label()

        concat_manifest = os.path.join(videos_dir, "concat_manifest.txt")
        manifest_lines = []

        metadata = {"title": model_label, "artist": "classic_log_sim_viewer"}
        for ep in episodes:
            ep_name = f"{model_label}_{self._episode_label(ep)}"
            ep_video = os.path.join(videos_dir, f"{ep_name}.mp4")
            writer = FFMpegWriter(fps=fps, metadata=metadata)

            self.episode_idx = ep
            ep_len = self._episode_len(ep)
            with writer.saving(self.fig, ep_video, dpi=dpi):
                for t in range(ep_len):
                    self.t = t
                    self.playing = False
                    self._update_frame()
                    writer.grab_frame()

            manifest_lines.append(f"file '{ep_video}'\n")

        if len(episodes) > 1:
            with open(concat_manifest, "w", encoding="utf-8") as handle:
                handle.writelines(manifest_lines)
            combined_video = os.path.join(videos_dir, f"{model_label}_all_episodes.mp4")
            cmd = [
                ffmpeg,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                concat_manifest,
                "-c",
                "copy",
                combined_video,
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _render_videos_from_frames(
        self,
        frames_root: str,
        output_dir: str,
        episodes: List[int],
        fps: int,
    ) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(
                "Saved all frames, but could not create video because `ffmpeg` was not found on PATH."
            )

        videos_dir = os.path.join(output_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)
        model_label = self._model_label()

        concat_manifest = os.path.join(videos_dir, "concat_manifest.txt")
        manifest_lines = []

        for ep in episodes:
            ep_name = f"{model_label}_{self._episode_label(ep)}"
            ep_frames = os.path.join(frames_root, ep_name, "frame_%04d.png")
            ep_video = os.path.join(videos_dir, f"{ep_name}.mp4")
            cmd = [
                ffmpeg,
                "-y",
                "-framerate",
                str(fps),
                "-i",
                ep_frames,
                "-pix_fmt",
                "yuv420p",
                ep_video,
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            manifest_lines.append(f"file '{ep_video}'\n")

        if len(episodes) > 1:
            with open(concat_manifest, "w", encoding="utf-8") as handle:
                handle.writelines(manifest_lines)
            combined_video = os.path.join(videos_dir, f"{model_label}_all_episodes.mp4")
            cmd = [
                ffmpeg,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                concat_manifest,
                "-c",
                "copy",
                combined_video,
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _choose_directory_dialog() -> Optional[str]:
    if tk is None or filedialog is None:
        return None
    root = tk.Tk()
    root.withdraw()
    directory = filedialog.askdirectory(title="Select directory containing all_*.pkl logs")
    root.destroy()
    return directory or None


def parse_args():
    parser = argparse.ArgumentParser(description="Classic interactive log renderer for current repo evaluation logs")
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory with all_observations.pkl/all_positions.pkl and optional other logs",
    )
    parser.add_argument("--interval-ms", type=int, default=150, help="Playback interval in milliseconds")
    parser.add_argument(
        "--reward-value",
        type=float,
        default=100.0,
        help="Reward magnitude added when a reward-delivery step is detected",
    )
    parser.add_argument("--step-penalty", type=float, default=-1.0, help="Per-timestep reward baseline")
    parser.add_argument("--gaze-penalty", type=float, default=0.0, help="Penalty added when gaze action == 1")
    parser.add_argument("--rat-image", type=str, default=None, help="Optional image path for moving rat sprite")
    parser.add_argument("--lever-image", type=str, default=None, help="Optional image path for lever sprite")
    parser.add_argument("--reward-image", type=str, default=None, help="Optional image path for reward sprite")
    parser.add_argument("--export-all", action="store_true", help="Save every timestep frame for every episode")
    parser.add_argument(
        "--export-episode",
        type=int,
        default=None,
        help="Save every timestep frame for one 1-based episode index only",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default="viewer_export",
        help="Output directory used by --export-all or --export-episode",
    )
    parser.add_argument("--export-fps", type=int, default=8, help="Frames per second for exported mp4 videos")
    parser.add_argument("--export-dpi", type=int, default=160, help="Resolution for exported frames and videos")
    parser.add_argument("--frames-only", action="store_true", help="Save PNG frames but skip mp4 creation")
    parser.add_argument(
        "--stream-video",
        action="store_true",
        help="Create mp4 directly with ffmpeg instead of saving PNG frames first",
    )
    parser.add_argument(
        "--cleanup-frames",
        action="store_true",
        help="Delete exported PNG frames after mp4 creation completes",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    log_dir = args.log_dir

    if log_dir is None:
        log_dir = _choose_directory_dialog()
    if log_dir is None:
        raise SystemExit("No log directory provided. Use --log-dir or choose a folder in the file dialog.")

    viewer = LogSimulationViewer(
        log_dir=log_dir,
        interval_ms=args.interval_ms,
        reward_value=args.reward_value,
        step_penalty=args.step_penalty,
        gaze_penalty=args.gaze_penalty,
        rat_image=args.rat_image,
        lever_image=args.lever_image,
        reward_image=args.reward_image,
    )

    if args.export_all and args.export_episode is not None:
        raise SystemExit("Use only one of --export-all or --export-episode.")

    if args.export_all:
        viewer.export_frames_and_video(
            output_dir=args.export_dir,
            episode=None,
            fps=args.export_fps,
            make_video=not args.frames_only,
            dpi=args.export_dpi,
            stream_video=args.stream_video,
            cleanup_frames=args.cleanup_frames,
        )
        return

    if args.export_episode is not None:
        episode_idx = args.export_episode - 1
        if not (0 <= episode_idx < len(viewer.episodes)):
            raise SystemExit(
                f"--export-episode must be between 1 and {len(viewer.episodes)}"
            )
        viewer.export_frames_and_video(
            output_dir=args.export_dir,
            episode=episode_idx,
            fps=args.export_fps,
            make_video=not args.frames_only,
            dpi=args.export_dpi,
            stream_video=args.stream_video,
            cleanup_frames=args.cleanup_frames,
        )
        return

    viewer.run()


if __name__ == "__main__":
    main()

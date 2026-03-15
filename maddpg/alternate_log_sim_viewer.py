#!/usr/bin/env python3
"""Interactive viewer for `main.py --evaluate` logs in this repository."""

from __future__ import annotations

import argparse
import pickle
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

try:
    import tkinter as tk
    from tkinter import filedialog
except Exception:
    tk = None
    filedialog = None


EPS = 1e-9
DEFAULT_RAT_IMAGE = Path(__file__).with_name("rat_img.png!sw800")
POLICY_COLOR = "#2c7fb8"
PRED_COLOR = "#d95f0e"
PATH_COLOR = "#7a7a7a"
TOTAL_COLOR = "#222222"
LEVER_ON = "#f6c945"
LEVER_OFF = "#7aa6c2"
REWARD_ON = "#57b86c"
REWARD_OFF = "#b8d8c1"
AGENT_COLORS = ["#4c72b0", "#c44e52", "#55a868", "#8172b3"]

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
    }
)


@dataclass
class CueLayout:
    has_lever_cue: bool
    has_lever_action_obs: bool
    reward_cue_idx: int
    lever_cue_idx: Optional[int]
    lever_action_obs_idx: Optional[int]


@dataclass
class ProbabilitySummary:
    available: bool
    source: str
    move_probs: np.ndarray
    move_argmax: Optional[int]
    move_confidence: Optional[float]
    gaze_probs: np.ndarray
    gaze_argmax: Optional[int]
    gaze_confidence: Optional[float]


class EvalLogViewer:
    def __init__(
        self,
        log_dir: str,
        interval_ms: int = 150,
        rat_image: Optional[str] = None,
        selected_agent: Optional[int] = None,
        reward_value: Optional[float] = None,
        step_penalty: float = -1.0,
        gaze_penalty: float = 0.0,
    ):
        self.log_dir = self._resolve_log_dir(log_dir)
        self.interval_ms = interval_ms
        self.rat_image_path = rat_image or str(DEFAULT_RAT_IMAGE)

        self.args_obj, self.args_dict = self._load_args()
        self.observations = self._load_pickle("all_observations.pkl")
        self.positions = self._load_pickle("all_positions.pkl", required=False)
        self.rewards = self._load_pickle("all_rewards.pkl", required=False)
        self.move_actions = self._load_pickle("all_move_actions.pkl", required=False)
        self.gaze_actions = self._load_pickle("all_gaze_actions.pkl", required=False)
        self.action_scores = self._load_pickle("all_action_probs.pkl", required=False)
        self.embeds = self._load_pickle("all_embeds.pkl", required=False)
        self.critic = self._load_pickle("all_critic.pkl", required=False)
        self.pulls = self._load_pickle("all_pulls.pkl", required=False) or {}
        self.coops = self._load_pickle("all_coops.pkl", required=False) or {}

        self.observations = self._to_ndarray(self.observations, "all_observations.pkl")
        self.positions = self._to_optional_ndarray(self.positions)
        self.rewards = self._to_optional_ndarray(self.rewards)
        self.move_actions = self._to_optional_ndarray(self.move_actions)
        self.gaze_actions = self._to_optional_ndarray(self.gaze_actions)
        self.action_scores = self._to_optional_ndarray(self.action_scores)
        self.embeds = self._to_optional_ndarray(self.embeds)
        self.critic = self._to_optional_ndarray(self.critic)

        if self.observations.ndim != 4:
            raise ValueError(
                f"Expected all_observations.pkl to have 4 dimensions, got shape {self.observations.shape}"
            )

        self.n_episodes = self.observations.shape[0]
        self.episode_len = self.observations.shape[2]
        self.obs_dim = self.observations.shape[3]
        self.agent_ids = self._infer_active_agents()
        if not self.agent_ids:
            raise ValueError("No active agents found in evaluation logs.")

        if selected_agent is not None and selected_agent in self.agent_ids:
            self.selected_agent = selected_agent
        else:
            self.selected_agent = self.agent_ids[0]
        self.agent_colors = {aid: AGENT_COLORS[idx % len(AGENT_COLORS)] for idx, aid in enumerate(self.agent_ids)}

        self.num_move_actions = self._infer_num_move_actions()
        self.move_labels = self._move_labels(self.num_move_actions)
        self.gaze_labels = ["no_gaze", "gaze"]
        self.cue_layout = self._infer_cue_layout()
        self.reward_value = float(reward_value) if reward_value is not None else float(self.args_dict.get("reward_value", 100.0))
        self.step_penalty = float(step_penalty)
        self.gaze_penalty = float(gaze_penalty)
        self.embed_loss = str(self.args_dict.get("embed_loss", "unknown"))
        self.embed_input = str(self.args_dict.get("embed_input", "none"))

        self.lever_pos, self.reward_pos = self._infer_landmarks()
        self.lane_y = {aid: self._infer_lane_y(aid) for aid in self.agent_ids}
        self.bounds = self._infer_bounds()
        self.rat_rgba = self._load_rat_image(self.rat_image_path)

        self.episode_idx = 0
        self.t = 0
        self.playing = False

        self.fig = None
        self.ax_env = None
        self.ax_trace = None
        self.ax_info = None
        self.timer = None

        self.title_text = None
        self.info_text = None
        self.path_lines: Dict[int, Any] = {}
        self.rat_artists: Dict[int, Any] = {}
        self.rat_offsets: Dict[int, OffsetImage] = {}
        self.lever_patch = None
        self.reward_patch = None
        self.return_line = None
        self.total_return_line = None
        self.trace_cursor = None
        self.pull_scatter = None
        self.reward_scatter = None

    @staticmethod
    def _resolve_log_dir(log_dir: str) -> str:
        path = Path(log_dir).expanduser().resolve()
        if (path / "all_observations.pkl").exists():
            return str(path)
        eval_path = path / "eval"
        if (eval_path / "all_observations.pkl").exists():
            return str(eval_path)
        raise FileNotFoundError(
            f"Could not find all_observations.pkl in {path} or {eval_path}. "
            "Pass the eval directory or the model directory that contains it."
        )

    def _load_pickle(self, filename: str, required: bool = True) -> Any:
        path = Path(self.log_dir) / filename
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Missing required log file: {path}")
            return None
        with open(path, "rb") as handle:
            return pickle.load(handle)

    @staticmethod
    def _to_ndarray(value: Any, label: str) -> np.ndarray:
        arr = np.asarray(value)
        if arr.dtype == object:
            raise ValueError(f"{label} is not an array-based log in this repository format.")
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
            Path(self.log_dir) / "args.npy",
            Path(self.log_dir).parent / "args.npy",
        ]
        for candidate in candidates:
            if not candidate.exists():
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
            return list(range(min(n_agents, self.observations.shape[1])))

        active: List[int] = []
        for agent_id in range(self.observations.shape[1]):
            obs_agent = self.observations[:, agent_id]
            if np.any(np.abs(obs_agent) > EPS):
                active.append(agent_id)
        return active or [0]

    def _infer_num_move_actions(self) -> int:
        if "num_actions" in self.args_dict:
            return int(self.args_dict["num_actions"])
        if self.move_actions is not None and np.any(np.abs(self.move_actions) > EPS):
            return max(3, int(np.nanmax(self.move_actions)) + 1)
        if self.action_scores is not None and self.action_scores.shape[-1] >= 2:
            return max(3, self.action_scores.shape[-1] - 2)
        if self.embeds is not None and self.embeds.shape[-1] in (3, 4, 5):
            return int(self.embeds.shape[-1])
        return 3

    def _infer_cue_layout(self) -> CueLayout:
        extra_dims = max(0, self.obs_dim - 10)
        if "lever_cue" in self.args_dict:
            has_lever_cue = str(self.args_dict["lever_cue"]) != "none"
        else:
            if extra_dims == 3:
                has_lever_cue = True
            elif extra_dims == 2:
                has_lever_cue = self.num_move_actions <= 3
            else:
                has_lever_cue = False

        if "lever_action" in self.args_dict:
            has_lever_action_obs = bool(self.args_dict["lever_action"])
        else:
            has_lever_action_obs = extra_dims in (2, 3) and self.num_move_actions > 3

        if has_lever_cue and has_lever_action_obs:
            return CueLayout(True, True, -3, -2, -1)
        if has_lever_cue:
            return CueLayout(True, False, -2, -1, None)
        if has_lever_action_obs:
            return CueLayout(False, True, -2, None, -1)
        return CueLayout(False, False, -1, None, None)

    def _infer_landmarks(self) -> Tuple[np.ndarray, np.ndarray]:
        for episode in range(self.n_episodes):
            for time_step in range(self.episode_len):
                obs = self._obs_at(episode, self.selected_agent, time_step)
                pos = self._position_at(episode, self.selected_agent, time_step)
                if obs is None or pos is None or obs.shape[0] < 8:
                    continue
                lever = pos + obs[4:6]
                reward = pos + obs[6:8]
                if np.any(np.abs(lever) > EPS) or np.any(np.abs(reward) > EPS):
                    return lever, reward

        env_size = float(self.args_dict.get("env_size", 2.0))
        y_pos = 1.0
        if self.positions is not None and np.any(np.abs(self.positions[:, self.selected_agent, :, 1]) > EPS):
            y_pos = float(self.positions[0, self.selected_agent, 0, 1])
        return np.array([-env_size / 2.0, y_pos]), np.array([env_size / 2.0, y_pos])

    def _infer_lane_y(self, agent_id: int) -> float:
        if self.positions is not None:
            ys = np.asarray(self.positions[:, agent_id, :, 1], dtype=float).reshape(-1)
        else:
            ys = np.asarray(self.observations[:, agent_id, :, 1], dtype=float).reshape(-1)
        finite = ys[np.isfinite(ys)]
        nonzero = finite[np.abs(finite) > EPS]
        if nonzero.size:
            return float(np.median(nonzero))
        if finite.size:
            return float(np.median(finite))
        return float(self.lever_pos[1])

    def _infer_bounds(self) -> Tuple[float, float, float, float]:
        points = [self.lever_pos, self.reward_pos]
        if self.positions is not None:
            for agent_id in self.agent_ids:
                agent_pos = self.positions[:, agent_id, :, :2].reshape(-1, 2)
                mask = np.any(np.abs(agent_pos) > EPS, axis=1)
                if np.any(mask):
                    points.append(agent_pos[mask])

        stacked = np.vstack(points)
        x_min = float(np.min(stacked[:, 0]))
        x_max = float(np.max(stacked[:, 0]))
        y_min = float(np.min(stacked[:, 1]))
        y_max = float(np.max(stacked[:, 1]))
        x_pad = max(0.25, 0.15 * max(x_max - x_min, 1.0))
        y_pad = max(0.2, 0.25 * max(y_max - y_min, 1.0))
        return x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad

    def _obs_at(self, episode: int, agent_id: int, time_step: int) -> np.ndarray:
        return np.asarray(self.observations[episode, agent_id, time_step], dtype=float)

    def _position_at(self, episode: int, agent_id: int, time_step: int) -> np.ndarray:
        if self.positions is not None:
            pos = np.asarray(self.positions[episode, agent_id, time_step], dtype=float)
            if pos.shape[0] >= 2 and np.any(np.abs(pos) > EPS):
                return pos[:2]
        obs = self._obs_at(episode, agent_id, time_step)
        if obs.shape[0] >= 2:
            return obs[:2]
        return np.zeros(2, dtype=float)

    def _reward_at(self, episode: int, agent_id: int, time_step: int) -> float:
        if self.rewards is None:
            return 0.0
        return float(self.rewards[episode, agent_id, time_step])

    def _critic_at(self, episode: int, agent_id: int, time_step: int) -> float:
        if self.critic is None:
            return 0.0
        value = np.asarray(self.critic[episode, agent_id, time_step], dtype=float).reshape(-1)
        return float(value[0]) if value.size else 0.0

    def _cue_state(self, episode: int, agent_id: int, time_step: int) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        obs = self._obs_at(episode, agent_id, time_step)
        reward_cue = int(round(obs[self.cue_layout.reward_cue_idx])) if obs.size else None
        lever_cue = None
        if self.cue_layout.lever_cue_idx is not None:
            lever_cue = int(round(obs[self.cue_layout.lever_cue_idx]))
        lever_action_obs = None
        if self.cue_layout.lever_action_obs_idx is not None:
            lever_action_obs = int(round(obs[self.cue_layout.lever_action_obs_idx]))
        return reward_cue, lever_cue, lever_action_obs

    def _other_rel(self, episode: int, agent_id: int, time_step: int) -> Optional[np.ndarray]:
        obs = self._obs_at(episode, agent_id, time_step)
        if obs.shape[0] >= 10:
            return obs[8:10]
        return None

    @staticmethod
    def _scores_to_probs(scores: Sequence[float]) -> np.ndarray:
        arr = np.asarray(scores, dtype=float).reshape(-1)
        if arr.size == 0:
            return np.zeros(0, dtype=float)
        if np.all(np.isfinite(arr)) and np.all(arr >= 0.0):
            total = float(np.sum(arr))
            if total > EPS and abs(total - 1.0) < 1e-4:
                return arr
        arr = np.where(np.isfinite(arr), arr, -1e9)
        arr = arr - np.max(arr)
        exp = np.exp(arr)
        total = float(np.sum(exp))
        if not np.isfinite(total) or total <= EPS:
            return np.full(arr.shape, 1.0 / arr.size, dtype=float)
        return exp / total

    def _policy_summary(self, episode: int, agent_id: int, time_step: int) -> ProbabilitySummary:
        if self.action_scores is None:
            return ProbabilitySummary(False, "all_action_probs", np.zeros(0), None, None, np.zeros(0), None, None)

        raw = np.asarray(self.action_scores[episode, agent_id, time_step], dtype=float).reshape(-1)
        if raw.size < self.num_move_actions:
            return ProbabilitySummary(False, "all_action_probs", np.zeros(0), None, None, np.zeros(0), None, None)

        move_scores = raw[: self.num_move_actions]
        gaze_scores = raw[-2:] if raw.size >= self.num_move_actions + 2 else np.zeros(0, dtype=float)
        move_probs = self._scores_to_probs(move_scores)
        gaze_probs = self._scores_to_probs(gaze_scores) if gaze_scores.size else np.zeros(0, dtype=float)
        move_idx = int(np.argmax(move_probs)) if move_probs.size else None
        gaze_idx = int(np.argmax(gaze_probs)) if gaze_probs.size else None
        move_conf = float(move_probs[move_idx]) if move_idx is not None else None
        gaze_conf = float(gaze_probs[gaze_idx]) if gaze_idx is not None else None
        return ProbabilitySummary(True, "all_action_probs", move_probs, move_idx, move_conf, gaze_probs, gaze_idx, gaze_conf)

    def _predictive_summary(self, episode: int, agent_id: int, time_step: int) -> ProbabilitySummary:
        if self.embeds is None or self.embed_input == "none":
            return ProbabilitySummary(False, "all_embeds", np.zeros(0), None, None, np.zeros(0), None, None)

        raw = np.asarray(self.embeds[episode, agent_id, time_step], dtype=float).reshape(-1)
        if raw.size == 0 or not np.any(np.abs(raw) > EPS):
            return ProbabilitySummary(False, "all_embeds", np.zeros(0), None, None, np.zeros(0), None, None)

        source = f"all_embeds ({self.embed_loss})"
        if self.embed_loss == "action" and raw.size >= self.num_move_actions + 2:
            move_scores = raw[: self.num_move_actions]
            gaze_scores = raw[-2:]
        elif self.embed_loss in {"move_action", "self_move_action"} and raw.size >= self.num_move_actions:
            move_scores = raw[: self.num_move_actions]
            gaze_scores = np.zeros(0, dtype=float)
        elif raw.size == self.num_move_actions:
            move_scores = raw
            gaze_scores = np.zeros(0, dtype=float)
            source = "all_embeds (assumed move_action)"
        elif raw.size == self.num_move_actions + 2:
            move_scores = raw[: self.num_move_actions]
            gaze_scores = raw[-2:]
            source = "all_embeds (assumed action)"
        else:
            return ProbabilitySummary(False, source, np.zeros(0), None, None, np.zeros(0), None, None)

        move_probs = self._scores_to_probs(move_scores)
        gaze_probs = self._scores_to_probs(gaze_scores) if gaze_scores.size else np.zeros(0, dtype=float)
        move_idx = int(np.argmax(move_probs)) if move_probs.size else None
        gaze_idx = int(np.argmax(gaze_probs)) if gaze_probs.size else None
        move_conf = float(move_probs[move_idx]) if move_idx is not None else None
        gaze_conf = float(gaze_probs[gaze_idx]) if gaze_idx is not None else None
        return ProbabilitySummary(True, source, move_probs, move_idx, move_conf, gaze_probs, gaze_idx, gaze_conf)

    def _pull_event(self, episode: int, agent_id: int, time_step: int) -> bool:
        episode_pulls = self.pulls.get(episode, {})
        agent_pulls = episode_pulls.get(agent_id, []) if isinstance(episode_pulls, dict) else []
        return time_step in agent_pulls

    def _coop_event(self, episode: int, time_step: int) -> bool:
        episode_coops = self.coops.get(episode, [])
        if not isinstance(episode_coops, (list, tuple)):
            return False
        for item in episode_coops:
            if isinstance(item, (list, tuple, np.ndarray)) and time_step in item:
                return True
        return False

    def _reward_event(self, episode: int, agent_id: int, time_step: int) -> bool:
        reward = self._reward_at(episode, agent_id, time_step)
        threshold = max(5.0, 0.5 * self.reward_value)
        return reward >= threshold

    def _display_step_return(self, episode: int, agent_id: int, time_step: int) -> float:
        reward = self.step_penalty
        if self._reward_event(episode, agent_id, time_step):
            reward += self.reward_value
        if self.gaze_actions is not None:
            gaze_action = int(round(float(self.gaze_actions[episode, agent_id, time_step])))
            if gaze_action == 1:
                reward += self.gaze_penalty
        return reward

    @staticmethod
    def _move_labels(num_actions: int) -> List[str]:
        common = {
            3: ["stay", "left", "right"],
            4: ["stay", "left", "right", "press"],
            5: ["stay", "left", "right", "down", "up"],
        }
        if num_actions in common:
            return common[num_actions]
        return [f"move_{idx}" for idx in range(num_actions)]

    def _move_label(self, action_idx: Optional[int]) -> str:
        if action_idx is None:
            return "n/a"
        if 0 <= action_idx < len(self.move_labels):
            return self.move_labels[action_idx]
        return str(action_idx)

    def _gaze_label(self, action_idx: Optional[int]) -> str:
        if action_idx is None:
            return "n/a"
        if 0 <= action_idx < len(self.gaze_labels):
            return self.gaze_labels[action_idx]
        return str(action_idx)

    def _cumulative_return(self, episode: int, agent_id: int) -> np.ndarray:
        per_step = np.array(
            [self._display_step_return(episode, agent_id, time_step) for time_step in range(self.episode_len)],
            dtype=float,
        )
        return np.cumsum(per_step)

    @staticmethod
    def _plot_ready(series: Sequence[float]) -> np.ndarray:
        arr = np.asarray(series, dtype=float).reshape(-1)
        return np.where(np.isfinite(arr), arr, np.nan)

    @staticmethod
    def _finite_min_max(series: Sequence[float], default: Tuple[float, float]) -> Tuple[float, float]:
        arr = np.asarray(series, dtype=float).reshape(-1)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return default
        return float(np.min(finite)), float(np.max(finite))

    @staticmethod
    def _transparent_border_background(image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image).copy()
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            return arr

        if arr.shape[2] == 3:
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
        q: deque[Tuple[int, int]] = deque()
        height, width = dark.shape

        def enqueue(y: int, x: int) -> None:
            if 0 <= y < height and 0 <= x < width and dark[y, x] and not visited[y, x]:
                visited[y, x] = True
                q.append((y, x))

        for x in range(width):
            enqueue(0, x)
            enqueue(height - 1, x)
        for y in range(height):
            enqueue(y, 0)
            enqueue(y, width - 1)

        while q:
            y, x = q.popleft()
            enqueue(y - 1, x)
            enqueue(y + 1, x)
            enqueue(y, x - 1)
            enqueue(y, x + 1)

        arr[..., 3][visited] = alpha_zero
        return arr

    def _load_rat_image(self, path: Optional[str]) -> Optional[np.ndarray]:
        if not path:
            return None
        image_path = Path(path).expanduser()
        if not image_path.exists():
            return None
        image = plt.imread(image_path)
        return self._transparent_border_background(image)

    def _create_rat_artist(self, x: float, y: float) -> Tuple[AnnotationBbox, OffsetImage]:
        if self.rat_rgba is not None:
            offset = OffsetImage(self.rat_rgba, zoom=0.11)
        else:
            fallback = np.ones((20, 20, 4), dtype=float)
            fallback[..., :3] = 0.45
            fallback[..., 3] = 1.0
            offset = OffsetImage(fallback, zoom=0.11)
        artist = AnnotationBbox(offset, (x, y), frameon=False, box_alignment=(0.5, 0.3), zorder=6)
        self.ax_env.add_artist(artist)
        return artist, offset

    def _build_figure(self) -> None:
        self.fig = plt.figure(figsize=(14.5, 8.4), facecolor="white")
        grid = self.fig.add_gridspec(
            2,
            2,
            width_ratios=[2.25, 1.15],
            height_ratios=[1.7, 1.0],
            hspace=0.34,
            wspace=0.28,
        )
        self.ax_env = self.fig.add_subplot(grid[0, 0])
        self.ax_trace = self.fig.add_subplot(grid[1, 0])
        self.ax_info = self.fig.add_subplot(grid[:, 1])

        self._draw_static_env()
        self._init_trace()

        self.ax_info.set_axis_off()
        self.info_text = self.ax_info.text(
            0.02,
            0.98,
            "",
            va="top",
            ha="left",
            family="serif",
            fontsize=11,
            linespacing=1.35,
            wrap=True,
            bbox={"facecolor": "white", "edgecolor": "#d0d0d0", "boxstyle": "round,pad=0.55", "alpha": 0.96},
        )
        self.title_text = self.ax_env.set_title("")
        self.fig.subplots_adjust(top=0.93, bottom=0.09, left=0.07, right=0.97)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        self.timer = self.fig.canvas.new_timer(interval=self.interval_ms)
        self.timer.add_callback(self._tick)
        self.timer.start()

    def _draw_static_env(self) -> None:
        x_min, x_max = self.bounds[0], self.bounds[1]
        lane_positions = [self.lane_y[agent_id] for agent_id in self.agent_ids]
        y_min = min(lane_positions + [self.lever_pos[1], self.reward_pos[1]]) - 0.28
        y_max = max(lane_positions + [self.lever_pos[1], self.reward_pos[1]]) + 0.28
        self.ax_env.set_xlim(x_min, x_max)
        self.ax_env.set_ylim(y_min, y_max)
        self.ax_env.set_aspect("equal", adjustable="box")
        self.ax_env.set_xlabel("x position")
        self.ax_env.set_yticks([])
        self.ax_env.grid(False)

        arena = patches.Rectangle(
            (x_min, y_min + 0.04),
            x_max - x_min,
            (y_max - y_min) - 0.08,
            linewidth=2.0,
            edgecolor="#444444",
            facecolor="#f2f2f2",
            alpha=0.96,
            zorder=0,
        )
        self.ax_env.add_patch(arena)

        for lane_y in lane_positions:
            self.ax_env.axhline(lane_y, color="#b8b8b8", lw=1.2, zorder=1)

        square_size = max(0.12, min(0.22, 0.08 * (x_max - x_min)))
        self.lever_patch = patches.Rectangle(
            (self.lever_pos[0] - square_size / 2, self.lever_pos[1] - square_size / 2),
            square_size,
            square_size,
            facecolor=LEVER_OFF,
            edgecolor="#304858",
            linewidth=2.0,
            zorder=3,
        )
        self.reward_patch = patches.Rectangle(
            (self.reward_pos[0] - square_size / 2, self.reward_pos[1] - square_size / 2),
            square_size,
            square_size,
            facecolor=REWARD_OFF,
            edgecolor="#24563a",
            linewidth=2.0,
            zorder=3,
        )
        self.ax_env.add_patch(self.lever_patch)
        self.ax_env.add_patch(self.reward_patch)
        label_y = y_max - 0.02
        self.ax_env.text(self.lever_pos[0], label_y, "lever", ha="center", va="top", fontsize=12)
        self.ax_env.text(self.reward_pos[0], label_y, "reward", ha="center", va="top", fontsize=12)

        legend = []
        for agent_id in self.agent_ids:
            color = self.agent_colors[agent_id]
            path_line, = self.ax_env.plot([], [], "-", color=color, lw=2.0, alpha=0.7, zorder=2)
            self.path_lines[agent_id] = path_line
            artist, offset = self._create_rat_artist(0.0, 0.0)
            self.rat_artists[agent_id] = artist
            self.rat_offsets[agent_id] = offset
            legend.append(Line2D([0], [0], color=color, lw=2, label=f"agent_{agent_id}"))

        legend.extend(
            [
            Line2D([0], [0], marker="s", color="w", markerfacecolor=LEVER_ON, markeredgecolor="#304858", ms=9, lw=0, label="lever cue"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor=REWARD_ON, markeredgecolor="#24563a", ms=9, lw=0, label="reward cue"),
            ]
        )
        self.ax_env.legend(handles=legend, loc="upper left", frameon=True, facecolor="white")

    def _init_trace(self) -> None:
        self.ax_trace.set_title("Return Over Time", fontsize=13, pad=12)
        self.ax_trace.set_xlabel("timestep")
        self.ax_trace.set_ylabel("cumulative return")
        self.ax_trace.grid(alpha=0.22, color="#d8d8d8")

        self.return_line, = self.ax_trace.plot([], [], color=POLICY_COLOR, lw=2.4, label="selected agent")
        self.total_return_line, = self.ax_trace.plot([], [], color=TOTAL_COLOR, lw=1.9, ls="--", label="total")
        self.trace_cursor = self.ax_trace.axvline(0, color="black", ls="--", lw=1.0)
        self.pull_scatter = self.ax_trace.scatter([], [], marker="v", s=42, color="#c0392b", label="pull")
        self.reward_scatter = self.ax_trace.scatter([], [], marker="*", s=64, color="#2e8b57", label="reward")

        handles = [Line2D([0], [0], color=POLICY_COLOR, lw=2.4, label="selected agent")]
        if len(self.agent_ids) > 1:
            handles.append(Line2D([0], [0], color=TOTAL_COLOR, lw=1.9, ls="--", label="total"))
        handles.extend(
            [
                Line2D([0], [0], marker="v", color="#c0392b", lw=0, label="pull"),
                Line2D([0], [0], marker="*", color="#2e8b57", lw=0, label="reward"),
            ]
        )
        self.ax_trace.legend(handles=handles, loc="upper left", frameon=True, facecolor="white")

    def _update_trace(self) -> None:
        episode = self.episode_idx
        agent_id = self.selected_agent
        xs = np.arange(self.episode_len)
        returns = self._cumulative_return(episode, agent_id)
        returns_plot = self._plot_ready(returns)
        total_returns = np.zeros(self.episode_len, dtype=float)
        for current_agent in self.agent_ids:
            total_returns += self._cumulative_return(episode, current_agent)
        total_plot = self._plot_ready(total_returns)

        self.return_line.set_data(xs, returns_plot)
        self.total_return_line.set_data(xs, total_plot)
        self.total_return_line.set_visible(len(self.agent_ids) > 1)
        self.trace_cursor.set_xdata([self.t, self.t])

        combined = np.concatenate([returns, total_returns]) if total_returns.size else returns
        r_min, r_max = self._finite_min_max(combined, default=(-1.0, 1.0))
        if abs(r_max - r_min) < 1e-6:
            r_max += 1.0
            r_min -= 1.0
        r_pad = max(1.0, 0.06 * (r_max - r_min))
        self.ax_trace.set_xlim(0, max(1, self.episode_len - 1))
        self.ax_trace.set_ylim(r_min - r_pad, r_max + r_pad)

        pull_offsets = []
        reward_offsets = []
        for time_step in range(self.episode_len):
            if self._pull_event(episode, agent_id, time_step) and np.isfinite(returns[time_step]):
                pull_offsets.append([time_step, returns[time_step]])
            if self._reward_event(episode, agent_id, time_step) and np.isfinite(returns[time_step]):
                reward_offsets.append([time_step, returns[time_step]])

        if not pull_offsets:
            pull_offsets = np.empty((0, 2))
        else:
            pull_offsets = np.asarray(pull_offsets, dtype=float)
        if not reward_offsets:
            reward_offsets = np.empty((0, 2))
        else:
            reward_offsets = np.asarray(reward_offsets, dtype=float)

        self.pull_scatter.set_offsets(pull_offsets)
        self.reward_scatter.set_offsets(reward_offsets)

    def _update_env(self) -> None:
        episode = self.episode_idx
        reward_cue, lever_cue, _ = self._cue_state(episode, self.selected_agent, self.t)
        pull_now = self._pull_event(episode, self.selected_agent, self.t)
        reward_now = self._reward_event(episode, self.selected_agent, self.t)

        self.lever_patch.set_facecolor(LEVER_ON if lever_cue else LEVER_OFF)
        self.reward_patch.set_facecolor(REWARD_ON if reward_cue else REWARD_OFF)
        self.lever_patch.set_linewidth(3.0 if pull_now else 2.0)
        self.reward_patch.set_linewidth(3.0 if reward_now else 2.0)
        self.lever_patch.set_edgecolor("#8f1d21" if pull_now else "#304858")
        self.reward_patch.set_edgecolor("#1d6b35" if reward_now else "#24563a")

        for agent_id in self.agent_ids:
            xs = [self._position_at(episode, agent_id, idx)[0] for idx in range(self.t + 1)]
            ys = [self._position_at(episode, agent_id, idx)[1] for idx in range(self.t + 1)]
            self.path_lines[agent_id].set_data(xs, ys)

            pos = self._position_at(episode, agent_id, self.t)
            self.rat_artists[agent_id].xy = (pos[0], pos[1])

    def _update_info(self) -> None:
        episode = self.episode_idx
        agent_id = self.selected_agent
        obs = self._obs_at(episode, agent_id, self.t)
        pos = self._position_at(episode, agent_id, self.t)
        vel = obs[2:4] if obs.shape[0] >= 4 else np.zeros(2)
        lever_rel = obs[4:6] if obs.shape[0] >= 6 else np.zeros(2)
        reward_rel = obs[6:8] if obs.shape[0] >= 8 else np.zeros(2)
        other_rel = self._other_rel(episode, agent_id, self.t)
        reward_cue, lever_cue, lever_action_obs = self._cue_state(episode, agent_id, self.t)

        move_action = None
        gaze_action = None
        if self.move_actions is not None:
            move_action = int(round(float(self.move_actions[episode, agent_id, self.t])))
        if self.gaze_actions is not None:
            gaze_action = int(round(float(self.gaze_actions[episode, agent_id, self.t])))

        logged_reward = self._reward_at(episode, agent_id, self.t)
        display_step_return = self._display_step_return(episode, agent_id, self.t)
        cum_return = self._cumulative_return(episode, agent_id)[self.t]
        critic_value = self._critic_at(episode, agent_id, self.t)
        policy = self._policy_summary(episode, agent_id, self.t)
        pred = self._predictive_summary(episode, agent_id, self.t)
        move_prob_lines = self._probability_lines("Policy move probs", self.move_labels, policy.move_probs)
        gaze_prob_lines = self._probability_lines("Policy gaze probs", self.gaze_labels, policy.gaze_probs)
        pred_move_lines = self._probability_lines("Pred move probs", self.move_labels, pred.move_probs if pred.available else np.zeros(0))
        pred_gaze_lines = self._probability_lines("Pred gaze probs", self.gaze_labels, pred.gaze_probs if pred.available else np.zeros(0))

        self.title_text.set_text(
            f"Episode {episode + 1}/{self.n_episodes} | "
            f"Timestep {self.t + 1}/{self.episode_len} | "
            f"{'PLAY' if self.playing else 'PAUSE'}"
        )

        lines = [
            f"Agent {agent_id}",
            "",
            f"Position: {self._fmt_vec(pos)}",
            f"Velocity: {self._fmt_vec(vel)}",
            f"Lever relative position: {self._fmt_vec(lever_rel)}",
            f"Reward relative position: {self._fmt_vec(reward_rel)}",
            f"Other relative position: {self._fmt_vec(other_rel)}",
            "",
            "Observed cues",
            f"Reward cue: {self._fmt_flag(reward_cue)}",
            f"Lever cue: {self._fmt_flag(lever_cue)}",
            f"Lever action state: {self._fmt_flag(lever_action_obs)}",
            "",
            "Actions",
            f"Actual movement action: {self._move_label(move_action)}",
            f"Actual gaze action: {self._gaze_label(gaze_action)}",
            f"Policy movement argmax: {self._move_label(policy.move_argmax)} {self._fmt_conf(policy.move_confidence)}",
        ]
        if policy.gaze_probs.size:
            lines.append(f"Policy gaze argmax: {self._gaze_label(policy.gaze_argmax)} {self._fmt_conf(policy.gaze_confidence)}")
        if pred.available and pred.move_argmax is not None:
            lines.append(f"Predicted movement action: {self._move_label(pred.move_argmax)} {self._fmt_conf(pred.move_confidence)}")
        else:
            lines.append("Predicted movement action: n/a")
        if pred.available and pred.gaze_probs.size:
            lines.append(f"Predicted gaze action: {self._gaze_label(pred.gaze_argmax)} {self._fmt_conf(pred.gaze_confidence)}")
        else:
            lines.append("Predicted gaze action: n/a")
        lines.extend(
            [
                "",
                "Returns",
                f"Logged reward at this step: {self._fmt_scalar(logged_reward)}",
                f"Displayed return increment: {self._fmt_scalar(display_step_return)}",
                f"Cumulative return: {self._fmt_scalar(cum_return)}",
                f"Critic value: {self._fmt_scalar(critic_value)}",
                "",
                "Policy and prediction details",
                *move_prob_lines,
                *gaze_prob_lines,
                *pred_move_lines,
                *pred_gaze_lines,
                f"Prediction source: {pred.source if pred.available else 'no predictive embedding logged'}",
                "",
                "Events",
                f"Pull this step: {self._pull_event(episode, agent_id, self.t)}",
                f"Reward this step: {self._reward_event(episode, agent_id, self.t)}",
                f"Cooperation this step: {self._coop_event(episode, self.t)}",
                "",
                "Controls",
                "Left/Right: step",
                "Up/Down: change episode",
                "Space: play or pause",
                "Home/End: first or last timestep",
                "Tab: cycle visible agent",
                "Click return plot: jump to timestep",
            ]
        )

        self.info_text.set_text("\n".join(lines))

    @staticmethod
    def _fmt_flag(value: Optional[int]) -> str:
        if value is None:
            return "n/a"
        return str(int(value))

    @staticmethod
    def _fmt_conf(value: Optional[float]) -> str:
        if value is None:
            return ""
        return f"({value:.2f})"

    @staticmethod
    def _fmt_vec(vec: Optional[np.ndarray]) -> str:
        if vec is None or vec.size == 0:
            return "n/a"
        return f"({vec[0]: .3f}, {vec[1]: .3f})"

    @staticmethod
    def _fmt_scalar(value: float) -> str:
        if not np.isfinite(value):
            return "n/a"
        return f"{value: .3f}"

    @staticmethod
    def _probability_lines(title: str, labels: Sequence[str], probs: np.ndarray, chunk_size: int = 3) -> List[str]:
        if probs.size == 0:
            return [f"{title}: n/a"]
        entries = [f"{labels[idx]}={probs[idx]:.2f}" for idx in range(min(len(labels), probs.size))]
        lines = []
        for start in range(0, len(entries), chunk_size):
            prefix = f"{title}: " if start == 0 else "    "
            lines.append(prefix + ", ".join(entries[start : start + chunk_size]))
        return lines

    def _update_frame(self) -> None:
        self.t = max(0, min(self.t, self.episode_len - 1))
        self._update_env()
        self._update_trace()
        self._update_info()
        self.fig.canvas.draw_idle()

    def _tick(self) -> None:
        if not self.playing:
            return
        if self.t < self.episode_len - 1:
            self.t += 1
        else:
            self.playing = False
        self._update_frame()

    def _cycle_agent(self) -> None:
        if len(self.agent_ids) <= 1:
            return
        idx = self.agent_ids.index(self.selected_agent)
        self.selected_agent = self.agent_ids[(idx + 1) % len(self.agent_ids)]

    def _on_key(self, event: Any) -> None:
        if event.key == "right":
            self.playing = False
            self.t = min(self.episode_len - 1, self.t + 1)
        elif event.key == "left":
            self.playing = False
            self.t = max(0, self.t - 1)
        elif event.key == " ":
            self.playing = not self.playing
        elif event.key == "up":
            self.playing = False
            self.episode_idx = min(self.n_episodes - 1, self.episode_idx + 1)
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
            self.t = self.episode_len - 1
        elif event.key == "tab":
            self.playing = False
            self._cycle_agent()
        else:
            return
        self._update_frame()

    def _on_click(self, event: Any) -> None:
        if event.inaxes != self.ax_trace or event.xdata is None:
            return
        self.playing = False
        self.t = int(round(event.xdata))
        self._update_frame()

    def run(self) -> None:
        self._build_figure()
        self._update_frame()
        plt.show()


def _choose_directory_dialog() -> Optional[str]:
    if tk is None or filedialog is None:
        return None
    root = tk.Tk()
    root.withdraw()
    directory = filedialog.askdirectory(title="Select an eval log directory")
    root.destroy()
    return directory or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive viewer for modeling_code_inst evaluation logs")
    parser.add_argument("--log-dir", type=str, default=None, help="Eval directory containing all_*.pkl logs")
    parser.add_argument("--interval-ms", type=int, default=150, help="Playback interval in milliseconds")
    parser.add_argument("--rat-image", type=str, default=str(DEFAULT_RAT_IMAGE), help="Path to the rat sprite image")
    parser.add_argument("--selected-agent", type=int, default=None, help="Agent index to focus on initially")
    parser.add_argument("--reward-value", type=float, default=None, help="Reward magnitude to use in the return plot")
    parser.add_argument("--step-penalty", type=float, default=-1.0, help="Per-timestep baseline used for the return plot")
    parser.add_argument("--gaze-penalty", type=float, default=0.0, help="Per-gaze penalty used for the return plot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_dir = args.log_dir or _choose_directory_dialog()
    if log_dir is None:
        raise SystemExit("No log directory provided. Use --log-dir or choose a folder in the file dialog.")

    viewer = EvalLogViewer(
        log_dir=log_dir,
        interval_ms=args.interval_ms,
        rat_image=args.rat_image,
        selected_agent=args.selected_agent,
        reward_value=args.reward_value,
        step_penalty=args.step_penalty,
        gaze_penalty=args.gaze_penalty,
    )
    viewer.run()


if __name__ == "__main__":
    main()

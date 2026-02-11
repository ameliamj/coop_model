#!/usr/bin/env python3
"""Interactive renderer for MADDPG evaluation logs.

Expected files in a directory:
- all_positions.pkl
- all_actions.pkl
- all_rewards.pkl (optional)
- all_pulls.pkl   (optional)
- all_coops.pkl   (optional)

Controls:
- Right arrow: next timestep
- Left arrow: previous timestep
- Space: play/pause
- Up arrow: next episode
- Down arrow: previous episode
- Home: first timestep
- End: last timestep
"""

import argparse
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib import patches

try:
    import tkinter as tk
    from tkinter import filedialog
except Exception:
    tk = None
    filedialog = None


ARENA_X_MIN = -0.4
ARENA_X_MAX = 0.4
LEVER_X = -0.25
PORT_X = 0.25
Y_TOP = 0.10
Y_BOTTOM = -0.10

AGENT_NAMES = {0: "adversary_0", 1: "agent_0"}
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
    def __init__(self, log_dir: str, interval_ms: int = 150):
        self.log_dir = log_dir
        self.interval_ms = interval_ms

        self.positions = self._load_pickle("all_positions.pkl")
        self.actions = self._load_pickle("all_actions.pkl")
        self.rewards = self._load_pickle("all_rewards.pkl", required=False) or {}
        self.pulls = self._load_pickle("all_pulls.pkl", required=False) or {}
        self.coops = self._load_pickle("all_coops.pkl", required=False) or {}

        self.episodes = sorted(self.positions.keys())
        if not self.episodes:
            raise ValueError("No episodes found in all_positions.pkl")

        self.episode_idx = 0
        self.t = 0
        self.playing = False

        self.fig = None
        self.ax_env = None
        self.ax_state = None
        self.ax_reward = None
        self.timer = None

        self.agent_dots = {}
        self.agent_paths = {}
        self.state_text = None
        self.title_text = None
        self.reward_lines = {}
        self.total_reward_line = None
        self.reward_cursor = None

    def _load_pickle(self, filename: str, required: bool = True):
        path = os.path.join(self.log_dir, filename)
        if not os.path.exists(path):
            if required:
                raise FileNotFoundError(f"Missing required log file: {path}")
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def _episode(self) -> int:
        return self.episodes[self.episode_idx]

    def _episode_len(self, ep: int) -> int:
        seq0 = self.positions[ep][0]
        seq1 = self.positions[ep][1]
        return min(len(seq0), len(seq1))

    def _safe_get(self, series: List[Any], idx: int, default: Any = None) -> Any:
        if 0 <= idx < len(series):
            return series[idx]
        return default

    def _extract_obs_like(self, value: Any) -> Optional[List[float]]:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            if value and isinstance(value[0], (list, tuple)):
                return None
            try:
                return [float(x) for x in value]
            except Exception:
                return None
        return None

    def _state_from_logs(self, ep: int, t: int, agent_id: int) -> AgentState:
        own_series = self.positions[ep][agent_id]
        other_id = 1 - agent_id
        other_series = self.positions[ep][other_id]

        own_raw = self._safe_get(own_series, t, 0.0)
        own_prev_raw = self._safe_get(own_series, max(0, t - 1), own_raw)
        other_raw = self._safe_get(other_series, t, 0.0)

        own_obs = self._extract_obs_like(own_raw)
        own_prev_obs = self._extract_obs_like(own_prev_raw)
        other_obs = self._extract_obs_like(other_raw)

        # If full observation was logged, use it directly.
        if own_obs is not None and len(own_obs) >= 5:
            x_pos = own_obs[0]
            x_vel = own_obs[1]
            x_lever_rel = own_obs[2]
            x_port_rel = own_obs[3]
            x_other_rel = own_obs[4]
            reward_cue = own_obs[5] if len(own_obs) > 5 else "n/a"
            lever_cue = own_obs[6] if len(own_obs) > 6 else "n/a"
            lever_action_state = own_obs[7] if len(own_obs) > 7 else "n/a"
            time_since_pull = own_obs[8] if len(own_obs) > 8 else "n/a"
            return AgentState(
                x_pos=x_pos,
                x_vel=x_vel,
                x_lever_rel=x_lever_rel,
                x_port_rel=x_port_rel,
                x_other_rel=x_other_rel,
                reward_cue=reward_cue,
                lever_cue=lever_cue,
                lever_action_state=lever_action_state,
                time_since_pull=time_since_pull,
            )

        # Fallback for current logging format where positions stores x_lever_rel only.
        own_x_lever_rel = float(own_raw)
        own_prev_x_lever_rel = float(own_prev_raw)

        x_pos = LEVER_X - own_x_lever_rel
        prev_x_pos = LEVER_X - own_prev_x_lever_rel
        x_vel = x_pos - prev_x_pos

        other_x = x_pos
        if other_obs is not None and len(other_obs) >= 1:
            other_x = float(other_obs[0])
        else:
            other_rel_to_own_lever = float(other_raw)
            other_x = LEVER_X - other_rel_to_own_lever

        x_port_rel = PORT_X - x_pos
        x_other_rel = other_x - x_pos

        return AgentState(
            x_pos=x_pos,
            x_vel=x_vel,
            x_lever_rel=own_x_lever_rel,
            x_port_rel=x_port_rel,
            x_other_rel=x_other_rel,
            reward_cue="n/a",
            lever_cue="n/a",
            lever_action_state="n/a",
            time_since_pull="n/a",
        )

    def _actions_at_t(self, ep: int, t: int, agent_id: int) -> Tuple[Any, Any]:
        eps_actions = self.actions.get(ep, {}).get(agent_id, [])
        entry = self._safe_get(eps_actions, t, ("n/a", "n/a"))
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            return entry[0], entry[1]
        return entry, "n/a"

    def _event_flags(self, ep: int, t: int) -> Dict[str, Any]:
        pulls = self.pulls.get(ep, {0: [], 1: []})
        rewards = self.rewards.get(ep, {0: [], 1: []})
        coops = self.coops.get(ep, [])

        return {
            "pull_0": t in pulls.get(0, []),
            "pull_1": t in pulls.get(1, []),
            "reward_0": t in rewards.get(0, []),
            "reward_1": t in rewards.get(1, []),
            "coop": any((isinstance(c, (list, tuple)) and t in c) for c in coops),
        }

    def _arena_y(self, agent_id: int) -> float:
        return Y_TOP if agent_id == 0 else Y_BOTTOM

    def _reward_event_cumsum(self, ep: int, ep_len: int, agent_id: int) -> List[int]:
        reward_steps = set(self.rewards.get(ep, {}).get(agent_id, []))
        running = 0
        out = []
        for t in range(ep_len):
            if t in reward_steps:
                running += 1
            out.append(running)
        return out

    def _build_figure(self):
        self.fig = plt.figure(figsize=(13, 7))
        gs = self.fig.add_gridspec(
            2, 2, width_ratios=[2.3, 1.3], height_ratios=[2.0, 1.0], hspace=0.16, wspace=0.16
        )
        self.ax_env = self.fig.add_subplot(gs[0, 0])
        self.ax_reward = self.fig.add_subplot(gs[1, 0])
        self.ax_state = self.fig.add_subplot(gs[:, 1])

        self._draw_static_arena()

        for aid in [0, 1]:
            y = self._arena_y(aid)
            dot, = self.ax_env.plot([], [], "o", color=AGENT_COLORS[aid], ms=10, label=AGENT_NAMES[aid])
            path, = self.ax_env.plot([], [], "-", color=AGENT_COLORS[aid], lw=1.8, alpha=0.6)
            self.agent_dots[aid] = dot
            self.agent_paths[aid] = path

        self.ax_env.legend(loc="upper left")
        self.title_text = self.ax_env.set_title("")

        self.ax_state.set_axis_off()
        self.state_text = self.ax_state.text(
            0.02,
            0.98,
            "",
            va="top",
            ha="left",
            family="monospace",
            fontsize=10,
        )

        self.ax_reward.set_title("Reward Over Time", fontsize=10)
        self.ax_reward.set_xlabel("timestep")
        self.ax_reward.set_ylabel("cumulative rewards")
        self.ax_reward.grid(alpha=0.25)

        for aid in [0, 1]:
            line, = self.ax_reward.plot([], [], color=AGENT_COLORS[aid], lw=2, label=AGENT_NAMES[aid])
            self.reward_lines[aid] = line
        self.total_reward_line, = self.ax_reward.plot(
            [], [], color="black", lw=2, ls="--", label="total"
        )
        self.reward_cursor = self.ax_reward.axvline(0, color="black", ls="--", lw=1)
        self.ax_reward.legend(loc="upper left", fontsize=8)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self.timer = self.fig.canvas.new_timer(interval=self.interval_ms)
        self.timer.add_callback(self._tick)
        self.timer.start()

    def _draw_static_arena(self):
        self.ax_env.set_xlim(ARENA_X_MIN - 0.03, ARENA_X_MAX + 0.03)
        self.ax_env.set_ylim(-0.20, 0.20)
        self.ax_env.set_aspect("equal", adjustable="box")
        self.ax_env.set_xlabel("x position")
        self.ax_env.set_ylabel("lane")

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
            self.ax_env.plot([LEVER_X], [y], marker="s", color="goldenrod", ms=9)
            self.ax_env.plot([PORT_X], [y], marker="s", color="forestgreen", ms=9)

        self.ax_env.text(LEVER_X, 0.155, "lever", ha="center", va="bottom", fontsize=9)
        self.ax_env.text(PORT_X, 0.155, "reward", ha="center", va="bottom", fontsize=9)

    def _update_frame(self):
        ep = self._episode()
        ep_len = self._episode_len(ep)
        if ep_len <= 0:
            return

        self.t = max(0, min(self.t, ep_len - 1))

        state_blocks = []
        for aid in [0, 1]:
            s = self._state_from_logs(ep, self.t, aid)
            y = self._arena_y(aid)

            # Dot
            self.agent_dots[aid].set_data([s.x_pos], [y])

            # Path
            xs = []
            for i in range(self.t + 1):
                xs.append(self._state_from_logs(ep, i, aid).x_pos)
            ys = [y] * len(xs)
            self.agent_paths[aid].set_data(xs, ys)

            move_action, gaze_action = self._actions_at_t(ep, self.t, aid)

            state_blocks.append(
                "\n".join(
                    [
                        f"{AGENT_NAMES[aid]}",
                        f"  x_pos            : {s.x_pos: .4f}",
                        f"  x_vel            : {s.x_vel: .4f}",
                        f"  x_lever_rel      : {s.x_lever_rel: .4f}",
                        f"  x_reward_rel     : {s.x_port_rel: .4f}",
                        f"  x_other_rel      : {s.x_other_rel: .4f}",
                        f"  reward_cue       : {s.reward_cue}",
                        f"  lever_cue        : {s.lever_cue}",
                        f"  lever_action_obs : {s.lever_action_state}",
                        f"  time_since_pull  : {s.time_since_pull}",
                        f"  action(move)     : {move_action}",
                        f"  action(gaze)     : {gaze_action}",
                    ]
                )
            )

        flags = self._event_flags(ep, self.t)
        events_line = (
            f"events @ t={self.t}: "
            f"pull0={flags['pull_0']} pull1={flags['pull_1']} "
            f"reward0={flags['reward_0']} reward1={flags['reward_1']} coop={flags['coop']}"
        )

        self.title_text.set_text(
            f"Episode {ep} | timestep {self.t + 1}/{ep_len} | {'PLAY' if self.playing else 'PAUSE'}"
        )

        notes = (
            "Controls\n"
            "  Left/Right : step\n"
            "  Up/Down    : episode\n"
            "  Space      : play/pause\n"
            "\n"
        )

        self.state_text.set_text("\n\n".join(state_blocks + [events_line, notes]))

        xs = list(range(ep_len))
        ymax = 1
        total = [0] * ep_len
        for aid in [0, 1]:
            cumsum = self._reward_event_cumsum(ep, ep_len, aid)
            self.reward_lines[aid].set_data(xs, cumsum)
            total = [a + b for a, b in zip(total, cumsum)]
            if cumsum:
                ymax = max(ymax, max(cumsum))
        self.total_reward_line.set_data(xs, total)
        if total:
            ymax = max(ymax, max(total))
        self.ax_reward.set_xlim(0, max(1, ep_len - 1))
        self.ax_reward.set_ylim(0, ymax + 1)
        self.reward_cursor.set_xdata([self.t, self.t])

        self.fig.canvas.draw_idle()

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
        plt.tight_layout()
        plt.show()


def _choose_directory_dialog() -> Optional[str]:
    if tk is None or filedialog is None:
        return None
    root = tk.Tk()
    root.withdraw()
    directory = filedialog.askdirectory(title="Select directory containing all_*.pkl logs")
    root.destroy()
    return directory or None


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive log renderer for coop_model evaluation logs")
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory with all_positions.pkl/all_actions.pkl and optional other logs",
    )
    parser.add_argument("--interval-ms", type=int, default=150, help="Playback interval in milliseconds")
    return parser.parse_args()


def main():
    args = parse_args()
    log_dir = args.log_dir

    if log_dir is None:
        log_dir = _choose_directory_dialog()
    if log_dir is None:
        raise SystemExit("No log directory provided. Use --log-dir or choose a folder in the file dialog.")

    viewer = LogSimulationViewer(log_dir=log_dir, interval_ms=args.interval_ms)
    viewer.run()


if __name__ == "__main__":
    main()

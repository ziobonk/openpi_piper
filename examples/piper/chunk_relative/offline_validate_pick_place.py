#!/usr/bin/env python3
"""Read-only Piper TCP anchored validation of a pick_place chunk-relative policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.piper.chunk_relative.infer_pick_place import training_metadata  # noqa: E402


def pose_matrix(pose: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    # WebSocket action arrays may be views into a read-only receive buffer;
    # scipy Rotation requires a writable input array on some versions.
    rotvec = np.array(pose[3:6], dtype=np.float64, copy=True)
    result[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def matrix_pose(transform: np.ndarray) -> np.ndarray:
    return np.r_[transform[:3, 3], Rotation.from_matrix(transform[:3, :3]).as_rotvec()]


def read_piper_tcp(can_name: str, offset: np.ndarray, timeout: float = 3.0) -> np.ndarray:
    """Receive a complete Piper pose without enabling the arm or sending commands."""
    sdk = ROOT / "piper_sdk"
    if str(sdk) not in sys.path:
        sys.path.insert(0, str(sdk))
    from piper_sdk import C_PiperInterface_V2

    piper = C_PiperInterface_V2(can_name=can_name, judge_flag=False, can_auto_init=True, dh_is_offset=1)
    try:
        piper.ConnectPort(piper_init=False)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            feedback = piper.GetArmEndPoseMsgs()
            if feedback.time_stamp > 0 and feedback.Hz > 0:
                ep = feedback.end_pose
                position = np.array([ep.X_axis, ep.Y_axis, ep.Z_axis], dtype=np.float64) * 1e-6
                rotation = Rotation.from_euler("xyz", np.array([ep.RX_axis, ep.RY_axis, ep.RZ_axis]) * np.pi / 180000.0)
                pose = np.r_[position + rotation.apply(offset), rotation.as_rotvec()]
                if not np.isfinite(pose).all():
                    raise ValueError("Piper TCP feedback contains nonfinite values")
                return pose
            time.sleep(0.02)
        raise TimeoutError(f"no complete Piper pose on {can_name} within {timeout:g} s")
    finally:
        piper.DisconnectPort()


def load_episode(dataset: Path, episode: int, max_frames: int) -> tuple[np.ndarray, np.ndarray, dict]:
    info = json.loads((dataset / "meta/info.json").read_text())
    template = info["data_path"]
    path = dataset / template.format(episode_chunk=episode // int(info["chunks_size"]), episode_index=episode)
    table = pq.read_table(path, columns=["frame_index", "state", "actions"])
    length = min(len(table), max_frames or len(table))
    indices = np.asarray(table["frame_index"].to_pylist()[:length], dtype=np.int64)
    state = np.asarray(table["state"].to_pylist()[:length], dtype=np.float64)
    actions = np.asarray(table["actions"].to_pylist()[:length], dtype=np.float64)
    if length < 2 or not np.array_equal(indices, np.arange(length)):
        raise ValueError("episode needs at least two consecutive frames starting at zero")
    if state.shape != (length, 1) or actions.shape != (length, 7):
        raise ValueError("expected prepared state (N,1) and actions (N,7)")
    if not np.isfinite(state).all() or not np.isfinite(actions).all():
        raise ValueError("episode contains nonfinite state/actions")
    if not np.allclose(state[:, 0], actions[:, 6], atol=1e-4):
        raise ValueError("state and action gripper angles are not same-frame")
    return state, actions, info


class VideoPair:
    def __init__(self, dataset: Path, info: dict, episode: int, needed_frames: int):
        template = info["video_path"]
        videos = []
        for key in ("observation.images.global", "observation.images.camera"):
            path = dataset / template.format(
                episode_chunk=episode // int(info["chunks_size"]), episode_index=episode, video_key=key
            )
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened() or capture.get(cv2.CAP_PROP_FRAME_COUNT) < needed_frames:
                capture.release()
                for prior in videos:
                    prior.release()
                raise ValueError(f"video missing or shorter than episode: {path}")
            videos.append(capture)
        self._videos = videos

    def read(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        images = []
        for capture in self._videos:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"could not decode video frame {index}")
            images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return images[0], images[1]

    def close(self) -> None:
        for capture in self._videos:
            capture.release()


def rollout(
    policy,
    state: np.ndarray,
    dataset_actions: np.ndarray,
    videos: VideoPair,
    start_tcp: np.ndarray,
    prompt: str,
    steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Advance a virtual TCP using predicted chunks; the physical arm stays still."""
    count = len(state)
    start = pose_matrix(start_tcp)
    source0_inv = np.linalg.inv(pose_matrix(dataset_actions[0, :6]))
    expected = np.empty_like(dataset_actions)
    predicted = np.empty_like(dataset_actions)
    predicted[0] = np.r_[start_tcp, dataset_actions[0, 6]]
    for i in range(count):
        expected[i, :6] = matrix_pose(start @ source0_inv @ pose_matrix(dataset_actions[i, :6]))
        expected[i, 6] = dataset_actions[i, 6]

    input_index = np.zeros(count, dtype=np.int64)
    input_images: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    virtual_tcp = start.copy()
    for t in range(0, count - 1, steps):
        base, wrist = videos.read(t)
        # Keep smaller copies for plotting; send the original decoded frames to
        # the policy so the server applies its training image transform.
        input_images[t] = (
            cv2.resize(base, (320, 240), interpolation=cv2.INTER_AREA),
            cv2.resize(wrist, (320, 240), interpolation=cv2.INTER_AREA),
        )
        observation = {
            "observation/state": state[t].astype(np.float32),
            "observation/image": base,
            "observation/wrist_image": wrist,
            "prompt": prompt,
        }
        result = np.asarray(policy.infer(observation)["actions"], dtype=np.float64)
        if result.ndim != 2 or result.shape[1] < 7 or len(result) < 2 or not np.isfinite(result).all():
            raise ValueError(f"policy returned invalid chunk at frame {t}: {result.shape}")
        future = min(steps, count - t - 1, len(result) - 1)
        if future < min(steps, count - t - 1):
            raise ValueError(f"policy chunk too short at frame {t}: {len(result)} actions")
        # action[0] is the same-frame identity. Every future action is relative
        # to the TCP at this query, not relative to the previous action.
        for k in range(1, future + 1):
            predicted[t + k, :6] = matrix_pose(virtual_tcp @ pose_matrix(result[k, :6]))
            predicted[t + k, 6] = result[k, 6]
            input_index[t + k] = t
        virtual_tcp = pose_matrix(predicted[t + future, :6])
        print(f"[Policy] frame {t}: {future} future steps; virtual TCP {predicted[t + future, :3]}")

    return predicted, expected, input_index, input_images


def render(
    predicted: np.ndarray,
    expected: np.ndarray,
    input_index: np.ndarray,
    input_images: dict[int, tuple[np.ndarray, np.ndarray]],
    task: str,
    gif: Path | None,
    gif_fps: int,
    *,
    show: bool,
) -> None:
    import matplotlib as mpl

    if not show:
        mpl.use("Agg")
    import matplotlib.pyplot as plt

    n = len(predicted)
    error_mm = np.linalg.norm(predicted[:, :3] - expected[:, :3], axis=1) * 1000
    fig = plt.figure(figsize=(13, 9))
    grid = fig.add_gridspec(2, 2)
    ax_base = fig.add_subplot(grid[0, 0])
    ax_wrist = fig.add_subplot(grid[0, 1])
    ax_3d = fig.add_subplot(grid[1, 0], projection="3d")
    ax_error = fig.add_subplot(grid[1, 1])
    all_xyz = np.vstack((predicted[:, :3], expected[:, :3]))
    center = (all_xyz.max(axis=0) + all_xyz.min(axis=0)) / 2
    radius = max(float(np.ptp(all_xyz, axis=0).max()) / 2, 0.02) * 1.1

    def draw(i: int) -> None:
        query = int(input_index[i])
        base, wrist = input_images[query]
        for ax, image, title in ((ax_base, base, "Dataset base"), (ax_wrist, wrist, "Dataset wrist")):
            ax.clear()
            ax.imshow(image)
            ax.set_title(f"{title}, model query frame {query}")
            ax.axis("off")
        ax_3d.clear()
        ax_3d.plot(*expected[: i + 1, :3].T, label="dataset anchored", color="tab:orange")
        ax_3d.plot(*predicted[: i + 1, :3].T, label="policy virtual TCP", color="tab:blue")
        ax_3d.scatter(*predicted[0, :3], color="black", marker="x", s=60)
        ax_3d.scatter(*predicted[i, :3], color="tab:blue", s=30)
        ax_3d.set(
            xlim=(center[0] - radius, center[0] + radius),
            ylim=(center[1] - radius, center[1] + radius),
            zlim=(center[2] - radius, center[2] + radius),
            xlabel="X (m)",
            ylabel="Y (m)",
            zlabel="Z (m)",
        )
        ax_3d.legend(loc="upper left", fontsize=8)
        ax_error.clear()
        ax_error.plot(np.arange(i + 1), error_mm[: i + 1], color="tab:purple")
        ax_error.set(
            xlim=(0, max(n - 1, 1)),
            ylim=(0, max(1, error_mm.max() * 1.1)),
            xlabel="Dataset frame",
            ylabel="Position difference (mm)",
        )
        fig.suptitle(f"Episode rollout, frame {i}/{n - 1}, difference {error_mm[i]:.1f} mm\n{task}")

    if gif is not None:
        from matplotlib.animation import FuncAnimation
        from matplotlib.animation import PillowWriter

        gif.parent.mkdir(parents=True, exist_ok=True)
        movie = FuncAnimation(fig, draw, frames=n, interval=1000 / gif_fps)
        movie.save(gif, writer=PillowWriter(fps=gif_fps), dpi=85)
        print(f"[Output] GIF: {gif}")
    if show:
        from matplotlib.widgets import Slider

        fig.subplots_adjust(bottom=0.12)
        slider_ax = fig.add_axes((0.15, 0.03, 0.7, 0.025))
        slider = Slider(slider_ax, "Frame", 0, n - 1, valinit=0, valstep=1)
        slider.on_changed(lambda value: (draw(int(value)), fig.canvas.draw_idle()))
        draw(0)
        plt.show()
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("local/datasets/pick_place_chunk_relative"))
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--can-name", default="can0")
    parser.add_argument("--tcp-offset-m", nargs=3, type=float, default=(0, 0, 0.22), metavar=("X", "Y", "Z"))
    parser.add_argument("--steps", type=int, default=3, help="future steps per model query")
    parser.add_argument("--max-frames", type=int, default=100, help="0 means complete episode")
    parser.add_argument("--gif", type=Path, default=None)
    parser.add_argument("--gif-fps", type=int, default=15)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args(argv)
    offset = np.asarray(args.tcp_offset_m, dtype=np.float64)
    if args.steps < 1 or args.steps > 49 or args.max_frames < 0 or args.gif_fps < 1:
        parser.error("steps must be 1..49, max-frames >= 0, and gif-fps > 0")
    if not np.isfinite(offset).all() or np.linalg.norm(offset) > 1:
        parser.error("tcp-offset-m must be finite and within 1 m")
    if args.no_show and args.gif is None:
        parser.error("--no-show requires --gif")
    if args.gif is not None and args.gif.suffix.lower() != ".gif":
        parser.error("--gif must end in .gif")
    _, _, _, prompt = training_metadata(args.dataset)
    state, actions, info = load_episode(args.dataset, args.episode, args.max_frames)
    from openpi_client import websocket_client_policy

    # This is the only robot interaction. No arm/gripper enable or command path is used.
    start_tcp = read_piper_tcp(args.can_name, offset)
    print(f"[Piper] read-only TCP start [m, rotvec rad]: {start_tcp}")
    policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    videos = VideoPair(args.dataset, info, args.episode, len(state))
    try:
        predicted, expected, input_index, input_images = rollout(
            policy, state, actions, videos, start_tcp, prompt, args.steps
        )
    finally:
        videos.close()
    error_mm = np.linalg.norm(predicted[:, :3] - expected[:, :3], axis=1) * 1000
    print(f"[Eval] {len(state)} frames, position difference mean={error_mm.mean():.1f} mm, max={error_mm.max():.1f} mm")
    if args.gif is not None:
        args.gif.parent.mkdir(parents=True, exist_ok=True)
        log_path = args.gif.with_suffix(".npz")
        np.savez_compressed(
            log_path,
            start_tcp=start_tcp,
            dataset_actions=actions,
            expected_tcp=expected,
            predicted_tcp=predicted,
            model_query_frame=input_index,
            position_error_mm=error_mm,
        )
        print(f"[Output] trace: {log_path}")
    render(predicted, expected, input_index, input_images, prompt, args.gif, args.gif_fps, show=not args.no_show)


if __name__ == "__main__":
    main()

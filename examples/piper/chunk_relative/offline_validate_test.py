#!/usr/bin/env python3
"""Compare a TCP-only chunk-relative checkpoint with recorded dataset actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

HORIZON = 50


def episode_paths(dataset: Path, info: dict, episode: int) -> tuple[Path, Path, Path]:
    values = {"episode_chunk": episode // int(info["chunks_size"]), "episode_index": episode}
    data = dataset / info["data_path"].format(**values)
    video = info["video_path"]
    base = dataset / video.format(**values, video_key="observation.images.global")
    wrist = dataset / video.format(**values, video_key="observation.images.camera")
    return data, base, wrist


def read_frame(capture: cv2.VideoCapture, index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"could not decode frame {index}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def relative_targets(actions: np.ndarray) -> tuple[np.ndarray, Rotation]:
    """Match the relative pose transform used by ChunkRelativeEEFActions."""
    origin = Rotation.from_rotvec(actions[0, 3:6])
    translation = origin.inv().apply(actions[1:, :3] - actions[0, :3])
    rotation = origin.inv() * Rotation.from_rotvec(actions[1:, 3:6])
    return translation, rotation


def evaluate_episode(
    policy, dataset: Path, info: dict, episode: int, prompt: str, stride: int, max_queries: int
) -> dict[str, np.ndarray]:
    data_path, base_path, wrist_path = episode_paths(dataset, info, episode)
    table = pq.read_table(data_path, columns=["frame_index", "state", "actions"])
    indices = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
    state = np.asarray(table["state"].to_pylist(), dtype=np.float32)
    actions = np.asarray(table["actions"].to_pylist(), dtype=np.float64)
    n = len(indices)
    if (
        n < HORIZON + 1
        or not np.array_equal(indices, np.arange(n))
        or state.shape != (n, 1)
        or actions.shape != (n, 7)
        or not np.isfinite(state).all()
        or not np.isfinite(actions).all()
        or not np.allclose(state, 0)
        or not np.allclose(actions[:, 6], 0)
    ):
        raise ValueError(f"episode {episode}: expected consecutive TCP-only frames with zero gripper")

    base = cv2.VideoCapture(str(base_path))
    wrist = cv2.VideoCapture(str(wrist_path))
    if not base.isOpened() or not wrist.isOpened():
        base.release()
        wrist.release()
        raise FileNotFoundError(f"episode {episode}: missing video {base_path} or {wrist_path}")
    position_errors = []
    rotation_errors = []
    query_frames = []
    predicted_tcp = []
    expected_tcp = []
    try:
        # Frame 0 is a copied pose because the original recording has no same-frame TCP there.
        frames = range(1, n - HORIZON + 1, stride)
        for query_number, t in enumerate(frames):
            if max_queries and query_number >= max_queries:
                break
            observation = {
                "observation/state": state[t],
                "observation/image": read_frame(base, t),
                "observation/wrist_image": read_frame(wrist, t),
                "prompt": prompt,
            }
            predicted = np.asarray(policy.infer(observation)["actions"], dtype=np.float64)
            if predicted.ndim != 2 or predicted.shape[0] < HORIZON or predicted.shape[1] < 7:
                raise ValueError(f"episode {episode}, frame {t}: expected at least (50, 7), got {predicted.shape}")
            predicted = predicted[1:HORIZON]
            if not np.isfinite(predicted).all():
                raise ValueError(f"episode {episode}, frame {t}: nonfinite model actions")
            expected_xyz, expected_rotation = relative_targets(actions[t : t + HORIZON])
            position_errors.append(np.linalg.norm(predicted[:, :3] - expected_xyz, axis=1) * 1000)
            angle = (Rotation.from_rotvec(predicted[:, 3:6]).inv() * expected_rotation).magnitude()
            rotation_errors.append(np.rad2deg(angle))
            # Every query uses its recorded TCP as the anchor, matching the error calculation.
            anchor_rotation = Rotation.from_rotvec(actions[t, 3:6])
            absolute_prediction = np.empty((HORIZON, 6))
            absolute_prediction[0] = actions[t, :6]
            absolute_prediction[1:, :3] = anchor_rotation.apply(predicted[:, :3]) + actions[t, :3]
            absolute_prediction[1:, 3:6] = (anchor_rotation * Rotation.from_rotvec(predicted[:, 3:6])).as_rotvec()
            query_frames.append(t)
            predicted_tcp.append(absolute_prediction)
            expected_tcp.append(actions[t : t + HORIZON, :6])
            print(
                f"episode {episode}, frame {t}: 49-step position mean={position_errors[-1].mean():.1f} mm, "
                f"rotation mean={rotation_errors[-1].mean():.1f} deg",
                flush=True,
            )
    finally:
        base.release()
        wrist.release()
    return {
        "episode": np.asarray(episode),
        "query_frames": np.asarray(query_frames),
        "dataset_tcp": actions[:, :6],
        "predicted_tcp": np.asarray(predicted_tcp),
        "expected_tcp": np.asarray(expected_tcp),
        "position_error_mm": np.asarray(position_errors),
        "rotation_error_deg": np.asarray(rotation_errors),
    }


def visualize(trace: dict[str, np.ndarray], output_dir: Path | None, *, show: bool, gif: bool) -> None:
    """Plot independent prediction chunks in the dataset tracking world frame."""
    import matplotlib as mpl

    if not show:
        mpl.use("Agg")
    import matplotlib.pyplot as plt

    episode = int(trace["episode"])
    frames = trace["query_frames"]
    if len(frames) == 0:
        raise ValueError("no query windows to visualize")
    predicted = trace["predicted_tcp"]
    expected = trace["expected_tcp"]
    dataset = trace["dataset_tcp"]
    position = trace["position_error_mm"]
    rotation = trace["rotation_error_deg"]
    fig = plt.figure(figsize=(13, 7))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.4, 1))
    ax_path = fig.add_subplot(grid[:, 0], projection="3d")
    ax_position = fig.add_subplot(grid[0, 1])
    ax_rotation = fig.add_subplot(grid[1, 1])
    xyz = np.vstack((dataset[:, :3], predicted[:, :, :3].reshape(-1, 3)))
    center = (xyz.min(axis=0) + xyz.max(axis=0)) / 2
    radius = max(float(np.ptp(xyz, axis=0).max()) / 2, 0.02) * 1.1
    future_steps = np.arange(1, HORIZON)

    def draw(query: int) -> None:
        ax_path.clear()
        ax_path.plot(*dataset[:, :3].T, color="0.8", alpha=0.6, label="Full dataset episode")
        ax_path.plot(*expected[query, :, :3].T, color="tab:orange", label="Dataset chunk")
        ax_path.plot(*predicted[query, :, :3].T, color="tab:blue", label="Model chunk")
        ax_path.scatter(*expected[query, 0, :3], color="black", marker="x", s=60, label="Recorded TCP anchor")
        ax_path.set(
            xlim=(center[0] - radius, center[0] + radius),
            ylim=(center[1] - radius, center[1] + radius),
            zlim=(center[2] - radius, center[2] + radius),
            xlabel="Tracking world X (m)",
            ylabel="Y (m)",
            zlabel="Z (m)",
        )
        ax_path.set_box_aspect((1, 1, 1))
        ax_path.legend(fontsize=8)
        for ax, values, label in (
            (ax_position, position, "Position error (mm)"),
            (ax_rotation, rotation, "Rotation error (deg)"),
        ):
            ax.clear()
            ax.plot(future_steps, values[query])
            ax.set(
                xlabel="Future step", ylabel=label, xlim=(1, HORIZON - 1), ylim=(0, max(float(values.max()) * 1.1, 1))
            )
            ax.grid(alpha=0.3)
        fig.suptitle(f"Episode {episode}, query frame {frames[query]}: independent 49-step prediction")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = output_dir / f"episode_{episode:06d}"
        np.savez_compressed(stem.with_suffix(".npz"), **trace)
        draw(0)
        fig.savefig(stem.with_suffix(".png"), dpi=150, bbox_inches="tight")
        if gif:
            from matplotlib.animation import FuncAnimation
            from matplotlib.animation import PillowWriter

            animation = FuncAnimation(fig, draw, frames=len(frames), interval=300)
            animation.save(stem.with_suffix(".gif"), writer=PillowWriter(fps=3), dpi=85)
        print(f"[Output] trajectory files: {stem}.*", flush=True)
    if show:
        from matplotlib.widgets import Slider

        fig.subplots_adjust(bottom=0.15)
        if len(frames) > 1:
            slider = Slider(fig.add_axes((0.18, 0.04, 0.6, 0.03)), "Query", 0, len(frames) - 1, valinit=0, valstep=1)
            slider.on_changed(lambda value: (draw(int(value)), fig.canvas.draw_idle()))
        draw(0)
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("local/datasets/test_chunk_relative"))
    parser.add_argument("--episodes", type=int, nargs="+", default=[0])
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--stride", type=int, default=10, help="frames between model queries")
    parser.add_argument("--max-queries-per-episode", type=int, default=20, help="0 evaluates all valid query frames")
    parser.add_argument("--output", type=Path, help="optional JSON metrics path")
    parser.add_argument("--plot-dir", type=Path, help="save trajectory PNG and reusable NPZ per episode")
    parser.add_argument("--gif", action="store_true", help="also save animated query chunks; requires --plot-dir")
    parser.add_argument("--show", action="store_true", help="open a 3D trajectory window with a query slider")
    parser.add_argument("--load-trace", type=Path, help="visualize a saved NPZ without querying the model")
    args = parser.parse_args()
    if args.stride < 1 or args.max_queries_per_episode < 0 or any(e < 0 for e in args.episodes):
        parser.error("stride must be positive; max queries and episode indices must be nonnegative")
    if args.gif and args.plot_dir is None:
        parser.error("--gif requires --plot-dir")
    if args.load_trace:
        if not args.show and args.plot_dir is None:
            parser.error("--load-trace requires --show or --plot-dir")
        with np.load(args.load_trace, allow_pickle=False) as saved:
            trace = {key: saved[key] for key in saved.files}
        visualize(trace, args.plot_dir, show=args.show, gif=args.gif)
        return

    info = json.loads((args.dataset / "meta/info.json").read_text())
    mapping = info.get("gripper_mapping", {})
    if (
        info.get("robot_type") != "piper_chunk_relative_eef"
        or mapping.get("available") is not False
        or info.get("features", {}).get("state", {}).get("shape") != [1]
        or info.get("features", {}).get("actions", {}).get("shape") != [7]
    ):
        raise ValueError("expected prepared TCP-only chunk-relative dataset")
    if any(e >= int(info["total_episodes"]) for e in args.episodes):
        parser.error(f"episode index must be below {info['total_episodes']}")
    tasks = [json.loads(line)["task"] for line in (args.dataset / "meta/tasks.jsonl").read_text().splitlines() if line]
    if len(tasks) != 1:
        raise ValueError("expected a dataset with one task prompt")

    from openpi_client import websocket_client_policy

    policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    position, rotation = [], []
    for episode in args.episodes:
        trace = evaluate_episode(
            policy, args.dataset, info, episode, tasks[0], args.stride, args.max_queries_per_episode
        )
        position.extend(trace["position_error_mm"])
        rotation.extend(trace["rotation_error_deg"])
        if args.plot_dir is not None or args.show:
            visualize(trace, args.plot_dir, show=args.show, gif=args.gif)
    if not position:
        raise ValueError("no full 50-frame query windows found")
    p, r = np.asarray(position), np.asarray(rotation)
    metrics = {
        "episodes": args.episodes,
        "queries": len(p),
        "future_steps_per_query": HORIZON - 1,
        "position_mean_mm": float(p.mean()),
        "position_median_mm": float(np.median(p)),
        "rotation_mean_deg": float(r.mean()),
        "rotation_median_deg": float(np.median(r)),
        "by_future_step": {
            str(k): {"position_mean_mm": float(p[:, k - 1].mean()), "rotation_mean_deg": float(r[:, k - 1].mean())}
            for k in (1, 10, 20, 49)
        },
    }
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

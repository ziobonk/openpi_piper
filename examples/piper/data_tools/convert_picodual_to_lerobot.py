#!/usr/bin/env python3
"""
Convert a raw "picodual" single-arm zarr recorder dump (schema_version 2) into a
LeRobot v2 dataset.

This is NOT the diffusion_policy_piper replay-buffer format that
``convert_dual_demo_to_lerobot.py`` / the C++ ``zarr2parquet`` handle. The
picodual dump stores:

    data/action                 (N, 7) float32  [x, y, z, rx, ry, rz, gripper_width]
    data/action_valid           (N,)   bool
    data/global_camera_img      (N, H, W, 3) uint8   one JPEG frame per chunk
    data/left_img               (N, H, W, 3) uint8   one JPEG frame per chunk
    data/left_gripper_width_mm  (N,)   float32
    data/left_t265_*            (N, 3) float32       T265 tracking-camera state
    data/timestamp_ns           (N,)   int64
    meta/episode_ends           (E,)   int64         cumulative frame counts

The image arrays use the ``imagecodecs_jpeg`` codec, which the installed
zarr/numcodecs cannot decode (``imagecodecs`` is not installed). We therefore
read the raw per-frame JPEG chunk files directly instead of going through zarr.

Output LeRobot v2 layout (single arm, EEF space):

    state   / actions   float32 (7,)  [left_x, left_y, left_z, left_rx, left_ry, left_rz, left_gripper_width]
    image               global_camera_img  (base camera)
    wrist_image         left_img           (left wrist camera)

The ``action_valid`` flag is ignored. Missing pose values in ``data/action``
are filled within each episode from the most recent finite pose; leading
missing poses use the first finite pose. Gripper width is kept for every frame.
The final frame's action holds its current pose so output frame counts match
the source exactly.

The input pose is ``^W T_C``: the PICO controller pose in PICO World, with
position in meters and rotation represented as a rotation vector in radians.
For each episode this converter applies the fixed controller-to-virtual-TCP
mount on the RIGHT and stores the complete first-frame-relative SE(3) pose::

    ^W T_E(t) = ^W T_C(t) @ ^C T_E
    D(t) = inv(^W T_E(0)) @ ^W T_E(t)

No PICO-World-to-robot-base extrinsic is used. State/action poses are D(t)
encoded as translation + rotation vector; action[t] is D(t+1).

Usage:
    python examples/piper/data_tools/convert_picodual_to_lerobot.py \
        --input data/test.zarr \
        --output ./pick_place \
        --controller-to-tcp 0 0 0.15 0 0 0 1 \
        --task "<control mode> end effector <control mode>pick up the red bottle cap and place it into the cup."
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import numcodecs
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from examples.piper.transforms.pico_relative_se3 import invert_transform
from examples.piper.transforms.pico_relative_se3 import pose_to_matrix
from examples.piper.transforms.pico_relative_se3 import relative_transform

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
FALLBACK_FPS = 50
STATE_DIM = 7
STATE_NAMES = [
    "tcp0_x", "tcp0_y", "tcp0_z",
    "tcp0_rx", "tcp0_ry", "tcp0_rz",
    "left_gripper_width",
]
MATRIX_NAMES = [f"m{row}{column}" for row in range(4) for column in range(4)]


@dataclass(frozen=True)
class EpisodePoseData:
    state: np.ndarray
    action: np.ndarray
    world_controller: np.ndarray
    world_tcp: np.ndarray
    episode_relative_tcp: np.ndarray


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def read_image_chunk_meta(img_dir: Path) -> tuple[int, ...]:
    """Return the zarr chunk shape for an image array, straight from .zarray."""
    meta = json.loads((img_dir / ".zarray").read_text(encoding="utf-8"))
    return tuple(meta["chunks"])


def read_numeric_array(array_dir: Path) -> np.ndarray:
    """Read a Zarr v2 numeric array without scanning the large image directories."""
    meta = json.loads((array_dir / ".zarray").read_text(encoding="utf-8"))
    shape = tuple(meta["shape"])
    chunks = tuple(meta["chunks"])
    if chunks[1:] != shape[1:]:
        raise ValueError(f"{array_dir}: unsupported numeric chunk shape {chunks}")
    codec = numcodecs.get_codec(meta["compressor"])
    dtype = np.dtype(meta["dtype"])
    parts = []
    for chunk_index in range((shape[0] + chunks[0] - 1) // chunks[0]):
        name = ".".join([str(chunk_index)] + ["0"] * (len(shape) - 1))
        decoded = codec.decode((array_dir / name).read_bytes())
        part = np.frombuffer(decoded, dtype=dtype).reshape(-1, *shape[1:])
        parts.append(part[: min(chunks[0], shape[0] - chunk_index * chunks[0])])
    return np.concatenate(parts, axis=0) if parts else np.empty(shape, dtype=dtype)


def read_jpeg_frame(img_dir: Path, frame: int, chunks: tuple[int, ...]) -> bytes:
    """Read the raw JPEG bytes for `frame` from a per-frame-chunked image array."""
    if chunks[0] != 1:
        raise RuntimeError(
            f"{img_dir}: expected chunks[0]==1 (per-frame JPEG), got {chunks}"
        )
    # Chunk filename: frame index along dim 0, 0 for every other dim.
    name = ".".join([str(frame)] + ["0"] * (len(chunks) - 1))
    return (img_dir / name).read_bytes()


def _fill_episode_controller_poses(actions: np.ndarray) -> np.ndarray:
    """Hold missing controller poses without changing per-frame gripper values."""
    filled = np.array(actions, dtype=np.float64, copy=True)
    if filled.ndim != 2 or filled.shape[1] != STATE_DIM or len(filled) == 0:
        raise ValueError("episode action must be a nonempty (N, 7) array")
    if not np.isfinite(filled[:, 6]).all():
        raise ValueError("episode gripper width contains non-finite values")
    pose_valid = np.isfinite(filled[:, :6]).all(axis=1)
    if not pose_valid.any():
        raise ValueError("episode has no finite pose to fill missing frames")
    first_valid = int(np.flatnonzero(pose_valid)[0])
    source_index = np.maximum.accumulate(np.where(pose_valid, np.arange(len(filled)), first_valid))
    filled[:, :6] = filled[source_index, :6]
    return filled


def _poses_to_matrices(poses: np.ndarray) -> np.ndarray:
    rotations = Rotation.from_rotvec(poses[:, 3:6]).as_matrix()
    matrices = np.repeat(np.eye(4, dtype=np.float64)[None], len(poses), axis=0)
    matrices[:, :3, :3] = rotations
    matrices[:, :3, 3] = poses[:, :3]
    return matrices


def _matrices_to_poses(matrices: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    poses = np.empty((len(matrices), STATE_DIM), dtype=np.float32)
    poses[:, :3] = matrices[:, :3, 3]
    poses[:, 3:6] = Rotation.from_matrix(matrices[:, :3, :3]).as_rotvec()
    poses[:, 6] = gripper
    return poses


def prepare_episode_pose_data(
    actions: np.ndarray,
    controller_to_tcp: np.ndarray,
    *,
    max_translation_step_m: float | None = None,
    max_rotation_step_rad: float | None = None,
) -> EpisodePoseData:
    """Create World poses and episode-relative virtual-TCP poses for one episode."""
    filled = _fill_episode_controller_poses(actions)
    controller_to_tcp = np.asarray(controller_to_tcp, dtype=np.float64)
    invert_transform(controller_to_tcp)  # Validate it as a proper SE(3) transform.

    world_controller = _poses_to_matrices(filled)
    # ^W T_E = ^W T_C @ ^C T_E: the fixed mount is a RIGHT multiply.
    world_tcp = world_controller @ controller_to_tcp

    for frame in range(1, len(world_tcp)):
        step = relative_transform(world_tcp[frame - 1], world_tcp[frame])
        translation_step = float(np.linalg.norm(step[:3, 3]))
        rotation_step = float(Rotation.from_matrix(step[:3, :3]).magnitude())
        if (max_translation_step_m is not None and translation_step > max_translation_step_m) or (
            max_rotation_step_rad is not None and rotation_step > max_rotation_step_rad
        ):
            raise ValueError(
                f"possible PICO tracking-origin jump at episode frame {frame}: "
                f"translation={translation_step:.3f} m, rotation={np.rad2deg(rotation_step):.1f} deg"
            )

    episode_relative_tcp = invert_transform(world_tcp[0]) @ world_tcp
    state = _matrices_to_poses(episode_relative_tcp, filled[:, 6])
    action = np.concatenate((state[1:], state[-1:]), axis=0)
    return EpisodePoseData(
        state=state,
        action=action,
        world_controller=world_controller,
        world_tcp=world_tcp,
        episode_relative_tcp=episode_relative_tcp,
    )


def prepare_episode_actions(
    actions: np.ndarray, controller_to_tcp: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper returning episode-relative state and next pose."""
    mount = np.eye(4) if controller_to_tcp is None else controller_to_tcp
    data = prepare_episode_pose_data(actions, mount)
    return data.state, data.action


# ---------------------------------------------------------------------------
# Per-episode processing
# ---------------------------------------------------------------------------
def _process_episode_worker(args: tuple) -> dict:
    """Worker: read one episode, write its parquet, return its stats."""
    (
        ep_idx,
        ep_start,
        ep_end,
        raw_actions,
        output_dir,
        task_text,
        fps,
        img_dir_global,
        img_dir_left,
        controller_to_tcp,
        max_translation_step_m,
        max_rotation_step_rad,
    ) = args

    import datasets
    from datasets import Features
    from datasets import Image as HFImage
    from datasets import Sequence
    from datasets import Value
    import pandas as pd

    pose_data = prepare_episode_pose_data(
        raw_actions,
        controller_to_tcp,
        max_translation_step_m=max_translation_step_m,
        max_rotation_step_rad=max_rotation_step_rad,
    )
    ep_state, ep_actions = pose_data.state, pose_data.action
    # 时序平移: state[t] = pose[t], action[t] = pose[t+1] (绝对目标)。
    # 最后一帧的 action=state；训练时 DeltaEEFPoseActions 将其转成零增量。
    n = len(ep_state)
    ep_ts = (np.arange(n, dtype=np.float32) / fps)

    chunks_g = read_image_chunk_meta(img_dir_global)
    chunks_l = read_image_chunk_meta(img_dir_left)

    # Read the raw JPEG bytes (no decode/re-encode) and decode only for stats.
    img_g_raw = [read_jpeg_frame(img_dir_global, i, chunks_g) for i in range(ep_start, ep_end)]
    img_l_raw = [read_jpeg_frame(img_dir_left, i, chunks_l) for i in range(ep_start, ep_end)]

    def encode_row(pos: int) -> dict:
        return {
            "state": ep_state[pos],
            "actions": ep_actions[pos],
            "pico_world_controller": pose_data.world_controller[pos].reshape(-1).astype(np.float32),
            "pico_world_tcp": pose_data.world_tcp[pos].reshape(-1).astype(np.float32),
            "episode_relative_tcp": pose_data.episode_relative_tcp[pos].reshape(-1).astype(np.float32),
            "timestamp": ep_ts[pos],
            "frame_index": pos,
            "episode_index": ep_idx,
            "index": ep_start + pos,
            "task_index": 0,
            "image": {"bytes": img_g_raw[pos], "path": f"frame_{pos:06d}.jpg"},
            "wrist_image": {"bytes": img_l_raw[pos], "path": f"frame_{pos:06d}.jpg"},
        }

    frames = [encode_row(pos) for pos in range(n)]

    df = pd.DataFrame(frames)
    data_dir = Path(output_dir) / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = data_dir / f"episode_{ep_idx:06d}.parquet"

    hf_features = Features({
        "state": Sequence(length=STATE_DIM, feature=Value(dtype="float32")),
        "actions": Sequence(length=STATE_DIM, feature=Value(dtype="float32")),
        "pico_world_controller": Sequence(length=16, feature=Value(dtype="float32")),
        "pico_world_tcp": Sequence(length=16, feature=Value(dtype="float32")),
        "episode_relative_tcp": Sequence(length=16, feature=Value(dtype="float32")),
        "timestamp": Value(dtype="float32"),
        "frame_index": Value(dtype="int64"),
        "episode_index": Value(dtype="int64"),
        "index": Value(dtype="int64"),
        "task_index": Value(dtype="int64"),
        "image": HFImage(),
        "wrist_image": HFImage(),
    })
    hf_ds = datasets.Dataset.from_pandas(df, features=hf_features)
    hf_ds.to_parquet(str(parquet_path))
    print(f"  -> Episode {ep_idx}: {n} frames -> {parquet_path.name}")

    # ---- stats ----
    def _feature_stats(arr: np.ndarray) -> dict:
        return {
            "min": arr.min(axis=0),
            "max": arr.max(axis=0),
            "mean": arr.mean(axis=0),
            "std": arr.std(axis=0),
            "count": np.array([n], dtype=np.int64),
        }

    def _image_stats(raw_frames: list[bytes]) -> dict:
        c = 3
        ch_min = np.full(c, 255.0)
        ch_max = np.zeros(c)
        ch_sum = np.zeros(c)
        ch_sq = np.zeros(c)
        total = 0
        for b in raw_frames:
            image = cv2.imdecode(np.frombuffer(b, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("failed to decode source JPEG frame")
            for channel_idx, channel in enumerate(reversed(cv2.split(image))):
                minimum, maximum, _, _ = cv2.minMaxLoc(channel)
                ch_min[channel_idx] = min(ch_min[channel_idx], minimum)
                ch_max[channel_idx] = max(ch_max[channel_idx], maximum)
            mean_bgr, std_bgr = cv2.meanStdDev(image)
            mean_rgb = mean_bgr.ravel()[::-1]
            std_rgb = std_bgr.ravel()[::-1]
            pixels = image.shape[0] * image.shape[1]
            ch_sum += mean_rgb * pixels
            ch_sq += (np.square(std_rgb) + np.square(mean_rgb)) * pixels
            total += pixels
        mean = (ch_sum / total).reshape(3, 1, 1).astype(np.float32)
        var = (ch_sq / total - (ch_sum / total) ** 2).reshape(3, 1, 1).astype(np.float32)
        return {
            "min": ch_min.reshape(3, 1, 1).astype(np.float32),
            "max": ch_max.reshape(3, 1, 1).astype(np.float32),
            "mean": mean,
            "std": np.sqrt(np.maximum(var, 0)).astype(np.float32),
            "count": np.array([n], dtype=np.int64),
        }

    ep_stats = {
        "state": _feature_stats(ep_state),
        "actions": _feature_stats(ep_actions),
        "image": _image_stats(img_g_raw),
        "wrist_image": _image_stats(img_l_raw),
    }

    return {
        "episode_index": ep_idx,
        "tasks": [task_text],
        "length": n,
        "stats": ep_stats,
    }


# ---------------------------------------------------------------------------
# 主转换
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Convert picodual single-arm zarr to LeRobot v2")
    p.add_argument("--input", default="data/test.zarr", help="input picodual zarr path")
    p.add_argument("--output", default="data/piper_picodual_lerobot", help="output LeRobot dir")
    p.add_argument("--task", default="pick up the black block and place it into the cup.",
                   help="task instruction text")
    p.add_argument("--fps", type=float, default=None,
                   help="frames per second (default: read frequency_hz from meta/.zattrs)")
    mount = p.add_mutually_exclusive_group(required=True)
    mount.add_argument(
        "--controller-to-tcp",
        type=float,
        nargs=7,
        metavar=("X", "Y", "Z", "QX", "QY", "QZ", "QW"),
        help="^C T_E as translation(m) plus xyzw quaternion; applied on the right",
    )
    mount.add_argument(
        "--controller-is-tcp",
        action="store_true",
        help="use identity ^C T_E only when raw poses already describe the virtual TCP",
    )
    p.add_argument(
        "--max-translation-step-m",
        type=float,
        default=0.25,
        help="abort on a larger adjacent-frame TCP step; <=0 disables (default: 0.25)",
    )
    p.add_argument(
        "--max-rotation-step-deg",
        type=float,
        default=60.0,
        help="abort on a larger adjacent-frame rotation; <=0 disables (default: 60)",
    )
    args = p.parse_args()

    controller_to_tcp = (
        np.eye(4, dtype=np.float64)
        if args.controller_is_tcp
        else pose_to_matrix(
            np.asarray(args.controller_to_tcp[:3]),
            np.asarray(args.controller_to_tcp[3:]),
            quaternion_order="xyzw",
        )
    )
    max_translation_step_m = args.max_translation_step_m if args.max_translation_step_m > 0 else None
    max_rotation_step_rad = (
        np.deg2rad(args.max_rotation_step_deg) if args.max_rotation_step_deg > 0 else None
    )

    zarr_path = Path(args.input)
    if not zarr_path.exists():
        print(f"[ERROR] zarr path not found: {zarr_path}")
        sys.exit(1)

    meta = json.loads((zarr_path / "meta" / ".zattrs").read_text(encoding="utf-8"))

    # fps: prefer meta/.zattrs frequency_hz, then CLI, then fallback
    fps = args.fps
    if fps is None:
        fps = float(meta.get("frequency_hz", FALLBACK_FPS))
    fps = float(fps)

    ep_ends = read_numeric_array(zarr_path / "meta" / "episode_ends").astype(np.int64)
    n_episodes = len(ep_ends)
    total_frames = int(ep_ends[-1])
    all_actions = read_numeric_array(zarr_path / "data" / "action")
    if len(all_actions) != total_frames:
        raise ValueError(f"action has {len(all_actions)} frames, episode_ends ends at {total_frames}")

    # Image shapes straight from meta/.zattrs (zarr can't open the jpeg arrays)
    img_shapes = meta.get("image_shapes", {})
    g_shape = tuple(img_shapes.get("global_camera", (total_frames, 720, 1280, 3)))
    l_shape = tuple(img_shapes.get("left", (total_frames, 480, 640, 3)))
    g_h, g_w = g_shape[1], g_shape[2]
    l_h, l_w = l_shape[1], l_shape[2]

    print(f"[INFO] input:  {zarr_path}")
    print(f"[INFO] output: {args.output}")
    print(f"[INFO] fps:    {fps}")
    print(f"[INFO] episodes: {n_episodes}, total frames: {total_frames}")
    print(f"[INFO] global_camera: {g_h}x{g_w}, left: {l_h}x{l_w}")
    print("[INFO] pose frame: PICO World -> virtual TCP -> episode-initial TCP")
    print(f"[INFO] ^C T_E:\n{controller_to_tcp}")
    print("[INFO] keeping all source frames; filling missing poses within each episode")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(exist_ok=True)
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    img_dir_global = zarr_path / "data" / "global_camera_img"
    img_dir_left = zarr_path / "data" / "left_img"

    # Keep every source frame. Missing pose values are filled within each episode.
    worker_args = []
    ep_start = 0
    for ep_idx, ep_end in enumerate(ep_ends):
        ep_end = int(ep_end)
        worker_args.append(
            (
                ep_idx,
                ep_start,
                ep_end,
                all_actions[ep_start:ep_end],
                str(out),
                args.task,
                fps,
                img_dir_global,
                img_dir_left,
                controller_to_tcp,
                max_translation_step_m,
                max_rotation_step_rad,
            )
        )
        ep_start = ep_end

    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures import as_completed

    episode_info = [None] * n_episodes
    max_workers = min(8, n_episodes)
    print(f"[INFO] processing {n_episodes} episodes with {max_workers} processes...")
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process_episode_worker, w): w[1] for w in worker_args}
        for future in as_completed(futures):
            result = future.result()
            episode_info[result["episode_index"]] = result

    # ---- meta files ----
    info = {
        "codebase_version": "v2.1",
        "robot_type": "piper_picodual_eef",
        "pose_coordinate_frame": "episode_initial_virtual_tcp",
        "pose_convention": "column_vectors; ^A_T_C = ^A_T_B @ ^B_T_C",
        "quaternion_order": "xyzw",
        "controller_to_tcp": controller_to_tcp.tolist(),
        "state_pose_semantics": "inv(^W_T_E(0)) @ ^W_T_E(t)",
        "action_pose_semantics": "next episode-relative TCP pose; training computes inv(state) @ future",
        "total_episodes": n_episodes,
        "total_frames": sum(e["length"] for e in episode_info),
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{n_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": None,
        "features": {
            "state": {"dtype": "float32", "shape": [STATE_DIM], "names": STATE_NAMES},
            "actions": {"dtype": "float32", "shape": [STATE_DIM], "names": STATE_NAMES},
            "pico_world_controller": {"dtype": "float32", "shape": [16], "names": MATRIX_NAMES},
            "pico_world_tcp": {"dtype": "float32", "shape": [16], "names": MATRIX_NAMES},
            "episode_relative_tcp": {"dtype": "float32", "shape": [16], "names": MATRIX_NAMES},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "image": {"dtype": "image", "shape": [g_h, g_w, 3], "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": [l_h, l_w, 3], "names": ["height", "width", "channel"]},
        },
    }
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    with open(out / "meta" / "episodes.jsonl", "w", encoding="utf-8") as f:
        for ep in episode_info:
            f.write(json.dumps({
                "episode_index": ep["episode_index"],
                "tasks": ep["tasks"],
                "length": ep["length"],
            }, ensure_ascii=False) + "\n")

    (out / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": args.task}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            return super().default(obj)

    with open(out / "meta" / "episodes_stats.jsonl", "w", encoding="utf-8") as f:
        for ep in episode_info:
            f.write(json.dumps({
                "episode_index": ep["episode_index"],
                "stats": ep["stats"],
            }, cls=NumpyEncoder) + "\n")

    from lerobot.common.datasets.compute_stats import aggregate_stats
    global_stats = aggregate_stats([ep["stats"] for ep in episode_info])
    (out / "meta" / "stats.json").write_text(json.dumps(global_stats, cls=NumpyEncoder, indent=2), encoding="utf-8")

    print(f"\n[DONE] dataset written to {out}/")
    print(f"  Episodes:     {n_episodes}")
    print(f"  Total frames: {sum(e['length'] for e in episode_info)}")
    print(f"  Duration:     {sum(e['length'] for e in episode_info) / fps:.1f}s @ {fps}fps")


if __name__ == "__main__":
    main()

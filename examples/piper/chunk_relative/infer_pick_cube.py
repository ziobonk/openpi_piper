#!/usr/bin/env python3
"""Run pick_cube with a D435 global camera, cropped USB wrist camera and native gripper."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.piper.chunk_relative import infer_test  # noqa: E402


def main() -> None:
    infer_test.main(native_gripper=True)


if __name__ == "__main__":
    main()

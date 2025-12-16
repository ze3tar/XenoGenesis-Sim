"""Rendering helpers for CA grids."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import ffmpeg


def render_frames(states: list[np.ndarray], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for idx, state in enumerate(states):
        frame_path = out_dir / f"frame_{idx:04d}.png"
        plt.imsave(frame_path, state, cmap="viridis", vmin=0, vmax=1)
        frames.append(frame_path)
    if not frames:
        return out_dir / "empty.mp4"
    mp4_path = out_dir / "ca.mp4"
    try:
        (
            ffmpeg
            .input(str(out_dir / 'frame_%04d.png'), framerate=24)
            .output(str(mp4_path), vcodec='libx264', pix_fmt='yuv420p', loglevel='quiet')
            .overwrite_output()
            .run()
        )
    except ffmpeg.Error:
        mp4_path = out_dir / "ca.gif"
        (
            ffmpeg
            .input(str(out_dir / 'frame_%04d.png'), framerate=12)
            .output(str(mp4_path), vf='palettegen', loglevel='quiet')
            .overwrite_output()
            .run()
        )
    return mp4_path

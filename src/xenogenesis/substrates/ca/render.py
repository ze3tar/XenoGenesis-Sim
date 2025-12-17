"""Rendering helpers for CA grids."""
from __future__ import annotations
from pathlib import Path
import shutil
from typing import Iterable, Mapping
import numpy as np
import matplotlib.pyplot as plt
import ffmpeg


def _frame_overlay(
    ax,
    state: np.ndarray,
    idx: int,
    metrics: Mapping[str, float] | None,
    cmap: str,
    *,
    gamma: float = 1.0,
    show_contours: bool = False,
):
    ax.clear()
    ax.set_axis_off()
    vmin = float(np.percentile(state, 1))
    vmax = float(np.percentile(state, 99))
    if np.isclose(vmin, vmax):
        vmin, vmax = 0.0, 1.0
    img = state ** gamma
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear")
    if show_contours:
        ax.contour(state, levels=[0.5], colors="white", linewidths=0.5)
    if metrics:
        text = " | ".join(f"{k}: {v:.3f}" for k, v in metrics.items())
        ax.set_title(f"t={idx} | {text}", fontsize=8)
    return im


def render_frames(
    states: list[np.ndarray],
    out_dir: Path,
    *,
    cmap: str = "magma",
    fps: int = 24,
    gamma: float = 1.0,
    show_contours: bool = False,
    metric_history: Iterable[Mapping[str, float]] | None = None,
) -> Path:
    """Render a sequence of CA states to an MP4 (GIF fallback).

    Parameters
    ----------
    states:
        Sequence of grid snapshots.
    out_dir:
        Directory to store frames and final animation.
    cmap:
        Matplotlib colormap name.
    fps:
        Frames per second for the video.
    metric_history:
        Optional per-frame metrics to annotate the render.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    metrics_iter = list(metric_history) if metric_history else [None] * len(states)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=140)
    for idx, (state, metric) in enumerate(zip(states, metrics_iter)):
        _frame_overlay(ax, state, idx, metric, cmap, gamma=gamma, show_contours=show_contours)
        frame_path = out_dir / f"frame_{idx:04d}.png"
        fig.savefig(frame_path, bbox_inches="tight")
        frames.append(frame_path)
    plt.close(fig)
    if not frames:
        return out_dir / "empty.mp4"
    mp4_path = out_dir / "ca.mp4"
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg is required for rendering; please install it.")
    try:
        (
            ffmpeg
            .input(str(out_dir / "frame_%04d.png"), framerate=fps)
            .output(
                str(mp4_path),
                vcodec="libx264",
                pix_fmt="yuv420p",
                crf=18,
                movflags="+faststart",
            )
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error:
        mp4_path = out_dir / "ca.gif"
        try:
            (
                ffmpeg
                .input(str(out_dir / "frame_%04d.png"), framerate=max(6, fps // 2))
                .output(str(mp4_path), vf="palettegen", loglevel="quiet")
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error as exc:
            raise RuntimeError("FFmpeg is required for rendering; please install it.") from exc
    return mp4_path

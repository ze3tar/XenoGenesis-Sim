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
    contour_level: float = 0.5,
    prev_state: np.ndarray | None = None,
    overlay_delta: bool = False,
    show_polarity_vectors: bool = False,
    vector_stride: int = 8,
    species_color: int | None = None,
):
    ax.clear()
    ax.set_axis_off()
    if state.ndim == 2:
        biomass = state
        resource = None
        polarity_mag = None
    else:
        biomass = state[0]
        resource = state[1] if state.shape[0] > 1 else None
        if state.shape[0] > 3:
            polarity_mag = np.clip(np.hypot(state[2], state[3]), 0.0, 1.0)
        elif state.shape[0] > 2:
            polarity_mag = np.clip(np.abs(state[2]), 0.0, 1.0)
        else:
            polarity_mag = None
    vmin = float(np.percentile(biomass, 1))
    vmax = float(np.percentile(biomass, 99))
    if np.isclose(vmin, vmax):
        vmin, vmax = 0.0, 1.0
    img = biomass ** gamma
    if species_color is not None:
        base_cmap = plt.get_cmap("tab20")
        tint = base_cmap(species_color % base_cmap.N)
        tint_img = np.dstack([img * tint[i] for i in range(3)])
        im = ax.imshow(tint_img, vmin=vmin, vmax=vmax, interpolation="bilinear")
    else:
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear")
    if overlay_delta and prev_state is not None:
        delta = biomass - prev_state
        denom = np.max(np.abs(delta)) + 1e-6
        ax.imshow(delta / denom, cmap="coolwarm", alpha=0.35, vmin=-1, vmax=1)
    if resource is not None:
        ax.imshow(resource, cmap="Greens", alpha=0.25, vmin=0, vmax=1)
    if polarity_mag is not None:
        ax.imshow(polarity_mag, cmap="twilight", alpha=0.25, vmin=0, vmax=1)
    if show_contours:
        ax.contour(biomass, levels=[contour_level], colors="white", linewidths=0.5)
    if show_polarity_vectors and state.shape[0] >= 4:
        skip = vector_stride
        y, x = np.mgrid[0:biomass.shape[0]:skip, 0:biomass.shape[1]:skip]
        ax.quiver(
            x,
            y,
            state[3][::skip, ::skip],
            state[2][::skip, ::skip],
            color="cyan",
            alpha=0.6,
            scale=40,
            width=0.002,
        )
    if metrics:
        text = " | ".join(f"{k}: {v:.3f}" for k, v in metrics.items() if isinstance(v, (int, float)))
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
    contour_level: float = 0.5,
    metric_history: Iterable[Mapping[str, float]] | None = None,
    overlay_delta: bool = True,
    snapshot_name: str = "organism_snapshot.png",
    video_name: str = "alien_life.mp4",
    show_polarity_vectors: bool = False,
    vector_stride: int = 8,
    species_labels: list[int] | None = None,
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
    if species_labels is None:
        species_labels = [None] * len(states)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=140)
    prev = None
    for idx, (state, metric, species_color) in enumerate(zip(states, metrics_iter, species_labels)):
        _frame_overlay(
            ax,
            state,
            idx,
            metric,
            cmap,
            gamma=gamma,
            show_contours=show_contours,
            contour_level=contour_level,
            prev_state=prev,
            overlay_delta=overlay_delta,
            show_polarity_vectors=show_polarity_vectors,
            vector_stride=vector_stride,
            species_color=species_color,
        )
        frame_path = out_dir / f"frame_{idx:04d}.png"
        fig.savefig(frame_path, bbox_inches="tight")
        frames.append(frame_path)
        prev = state[0].copy() if state.ndim > 2 else state.copy()
    plt.close(fig)
    if not frames:
        return out_dir / "empty.mp4"
    mp4_path = out_dir / video_name
    snapshot_path = out_dir / snapshot_name
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
        shutil.copy(frames[-1], snapshot_path)
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
        shutil.copy(frames[-1], snapshot_path)
    return mp4_path

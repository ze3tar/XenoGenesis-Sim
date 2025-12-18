"""Streamlit UI for real-time world manipulation and regime exploration."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from xenogenesis.config import load_config
from xenogenesis.substrates.ca.ca_model import CAStepper
from xenogenesis.substrates.ca.genome import params_from_config, mutate, Genome, CAParams
from xenogenesis.substrates.ca.render import render_frames


PRESETS = {
    "Nutrient-rich": {
        "growth_alpha": 0.95,
        "regen_rate": 0.08,
        "consumption_rate": 0.03,
        "division_threshold": 0.45,
        "resource_affinity": 0.3,
        "resource_capacity": 1.6,
    },
    "Desert": {
        "growth_alpha": 0.55,
        "regen_rate": 0.025,
        "consumption_rate": 0.018,
        "resource_gradient": 0.1,
        "division_threshold": 0.65,
    },
    "Tidal cycles": {
        "growth_alpha": 0.7,
        "regen_rate": 0.06,
        "drift_rate": 0.02,
        "resource_gradient": 0.5,
        "division_fraction": 0.6,
    },
    "Toxic storms": {
        "growth_alpha": 0.85,
        "toxin_rate": 0.08,
        "competition_scale": 0.55,
        "resource_capacity": 1.2,
        "polarity_gain": 0.75,
    },
    "Filamentous worms": {
        "polarity_gain": 0.8,
        "directional_gain": 1.5,
        "fission_assist": 0.2,
        "division_fraction": 0.55,
    },
    "Colonial organisms": {
        "competition_scale": 0.4,
        "resource_gradient": 0.45,
        "resource_capacity": 1.4,
    },
}


def _init_state():
    cfg_path = Path(__file__).parents[2] / "configs" / "alien_life.yaml"
    cfg = load_config(cfg_path if cfg_path.exists() else Path("src/xenogenesis/config/defaults.yaml"))
    ca_params = params_from_config(cfg.ca)
    rng = np.random.default_rng(cfg.seed)
    biomass = rng.random((cfg.ca.grid_size, cfg.ca.grid_size), dtype=np.float32) * 0.1
    resource = np.ones_like(biomass, dtype=np.float32)
    polarity = np.zeros_like(biomass, dtype=np.float32)
    state = np.stack((biomass, resource, polarity, polarity))
    st.session_state.update(
        {
            "cfg": cfg,
            "ca_params": ca_params,
            "stepper": CAStepper(),
            "state": state,
            "history": [state.copy()],
            "metrics": {"mass": [], "resource": [], "reproduction": []},
            "paused": False,
            "show_contours": True,
            "show_resource": True,
            "autoplay": False,
        }
    )


def _append_history(state: np.ndarray):
    hist = st.session_state.get("history", [])
    hist.append(state.copy())
    if len(hist) > 400:
        hist = hist[-400:]
    st.session_state["history"] = hist


def _render_snapshot(state: np.ndarray):
    fig, ax = plt.subplots(figsize=(5, 5))
    biomass = state[0]
    vmin, vmax = np.percentile(biomass, (2, 98))
    ax.imshow(biomass ** st.session_state["ca_params"].render_gamma, cmap="magma", vmin=vmin, vmax=vmax)
    if st.session_state.get("show_resource", True) and state.shape[0] > 1:
        ax.imshow(np.clip(state[1], 0, 1), cmap="Greens", alpha=0.25, vmin=0, vmax=1)
    if st.session_state.get("show_contours", True):
        ax.contour(biomass, levels=[st.session_state["ca_params"].contour_level], colors="white", linewidths=0.5)
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)


def _apply_preset(name: str):
    params = st.session_state["ca_params"]
    preset = PRESETS[name]
    data = {**params.__dict__}
    data.update(preset)
    st.session_state["ca_params"] = CAParams(**data)


def _inject_seed(radius: int = 8):
    state = st.session_state["state"]
    h, w = state.shape[1:]
    cy, cx = h // 2, w // 2
    y0, y1 = max(cy - radius, 0), min(cy + radius, h)
    x0, x1 = max(cx - radius, 0), min(cx + radius, w)
    patch = np.random.default_rng().random((y1 - y0, x1 - x0)).astype(np.float32) * 0.8
    state[0, y0:y1, x0:x1] = patch
    st.session_state["state"] = state
    _append_history(state)


def main():
    st.set_page_config(page_title="XenoGenesis Live UI", layout="wide")
    if "state" not in st.session_state:
        _init_state()

    st.sidebar.header("Life regimes")
    preset = st.sidebar.selectbox("Preset", list(PRESETS.keys()))
    if st.sidebar.button("Apply preset"):
        _apply_preset(preset)

    ca_params = st.session_state["ca_params"]
    st.sidebar.subheader("Environment")
    regen = st.sidebar.slider("Resource regen", 0.0, 0.2, float(ca_params.regen_rate))
    directional = st.sidebar.slider("Polarity strength", 0.1, 2.0, float(ca_params.directional_gain))
    division = st.sidebar.slider("Division threshold", 0.2, 0.9, float(ca_params.division_threshold))
    dt = st.sidebar.slider("dt", 0.01, 0.3, float(ca_params.dt))
    diffusion = st.sidebar.slider("Resource diffusion", 0.0, 0.4, float(ca_params.resource_diffusion))
    mobility = st.sidebar.slider("Polarity mobility", 0.0, 0.5, float(ca_params.polarity_mobility))
    data = {
        **ca_params.__dict__,
        "regen_rate": regen,
        "directional_gain": directional,
        "division_threshold": division,
        "dt": dt,
        "resource_diffusion": diffusion,
        "polarity_mobility": mobility,
    }
    st.session_state["ca_params"] = CAParams(**data)

    st.sidebar.subheader("Overlays")
    st.session_state["show_contours"] = st.sidebar.checkbox("Contours", value=st.session_state.get("show_contours", True))
    st.session_state["show_resource"] = st.sidebar.checkbox("Resource heatmap", value=st.session_state.get("show_resource", True))
    st.session_state["autoplay"] = st.sidebar.checkbox("Auto-run", value=st.session_state.get("autoplay", False))

    cols = st.columns(4)
    if cols[0].button("Step x20"):
        st.session_state["paused"] = False
        for _ in range(20):
            result = st.session_state["stepper"].step(st.session_state["state"], st.session_state["ca_params"])
            st.session_state["state"] = result.state
            _append_history(result.state)
            st.session_state["metrics"]["mass"].append(result.stats.mass)
            st.session_state["metrics"]["resource"].append(result.stats.resource)
            st.session_state["metrics"]["reproduction"].append(result.stats.reproduction_events)
    if cols[1].button("Run 200 steps"):
        for _ in range(200):
            result = st.session_state["stepper"].step(st.session_state["state"], st.session_state["ca_params"])
            st.session_state["state"] = result.state
            _append_history(result.state)
            st.session_state["metrics"]["mass"].append(result.stats.mass)
            st.session_state["metrics"]["resource"].append(result.stats.resource)
            st.session_state["metrics"]["reproduction"].append(result.stats.reproduction_events)
    if cols[2].button("Pause/Resume"):
        st.session_state["paused"] = not st.session_state["paused"]
    if cols[3].button("Rewind 50 steps") and len(st.session_state["history"]) > 50:
        st.session_state["state"] = st.session_state["history"][-50].copy()

    st.markdown("### Live CA view")
    _render_snapshot(st.session_state["state"])

    if st.session_state.get("autoplay"):
        for _ in range(5):
            result = st.session_state["stepper"].step(st.session_state["state"], st.session_state["ca_params"])
            st.session_state["state"] = result.state
            _append_history(result.state)
            st.session_state["metrics"]["mass"].append(result.stats.mass)
            st.session_state["metrics"]["resource"].append(result.stats.resource)
            st.session_state["metrics"]["reproduction"].append(result.stats.reproduction_events)

    st.markdown("### Metrics")
    met_cols = st.columns(3)
    met_cols[0].line_chart(st.session_state["metrics"]["mass"], height=150)
    met_cols[1].line_chart(st.session_state["metrics"]["resource"], height=150)
    met_cols[2].line_chart(st.session_state["metrics"]["reproduction"], height=150)

    st.markdown("Genome controls")
    if st.button("Mutate genome seed"):
        genome = Genome.random(st.session_state["cfg"].genome.length, np.random.default_rng())
        mutated = mutate(genome, np.random.default_rng())
        st.session_state["cfg"].genome.enabled = True
        st.session_state["state"][2:] = 0.0
        st.success(f"Injected mutated genome with hash {hash(mutated.genes.tobytes()) % 10000}")

    export_col1, export_col2, export_col3 = st.columns(3)
    if export_col1.button("Inject biomass seed"):
        _inject_seed()
        st.success("Injected biomass seed at center")
    if export_col2.button("Export PNG"):
        out_dir = Path("runs/ui_exports")
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            render_frames([st.session_state["state"]], out_dir, snapshot_name="live_snapshot.png", video_name="live_snapshot.mp4", multi_panel=True, show_polarity_vectors=True)
            st.success(f"Saved snapshot to {out_dir / 'live_snapshot.png'}")
        except Exception as exc:  # pragma: no cover - UI feedback path
            st.error(f"Export failed: {exc}")
    if export_col3.button("Export last 100 frames") and len(st.session_state["history"]) > 5:
        out_dir = Path("runs/ui_exports")
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            render_frames(st.session_state["history"][-100:], out_dir, video_name="live_history.mp4", multi_panel=True, show_polarity_vectors=True)
            st.success(f"Saved video to {out_dir / 'live_history.mp4'}")
        except Exception as exc:  # pragma: no cover - UI feedback path
            st.error(f"Export failed: {exc}")


if __name__ == "__main__":
    main()

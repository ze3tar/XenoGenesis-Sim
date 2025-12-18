"""Streamlit UI for real-time world manipulation and regime exploration."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from xenogenesis.config import load_config
from xenogenesis.substrates.ca.ca_model import CAStepper
from xenogenesis.substrates.ca.genome import params_from_config, mutate, Genome, CAParams


PRESETS = {
    "Abiogenesis": {
        "growth_alpha": 0.6,
        "regen_rate": 0.04,
        "consumption_rate": 0.02,
        "division_threshold": 0.42,
        "resource_affinity": 0.4,
        "polarity_gain": 0.2,
    },
    "Microbial soup": {
        "growth_alpha": 0.9,
        "regen_rate": 0.06,
        "consumption_rate": 0.03,
        "division_threshold": 0.5,
        "resource_affinity": 0.35,
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
    "Competitive ecosystems": {
        "competition_scale": 0.55,
        "resource_affinity": 0.25,
        "division_threshold": 0.6,
        "polarity_mutation": 0.035,
        "polarity_gain": 0.9,
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
        }
    )


def _render_snapshot(state: np.ndarray):
    fig, ax = plt.subplots(figsize=(5, 5))
    biomass = state[0]
    vmin, vmax = np.percentile(biomass, (2, 98))
    ax.imshow(biomass ** st.session_state["ca_params"].render_gamma, cmap="magma", vmin=vmin, vmax=vmax)
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


def main():
    st.set_page_config(page_title="XenoGenesis Live UI", layout="wide")
    if "state" not in st.session_state:
        _init_state()

    st.sidebar.header("Life regimes")
    preset = st.sidebar.selectbox("Preset", list(PRESETS.keys()))
    if st.sidebar.button("Apply preset"):
        _apply_preset(preset)

    ca_params = st.session_state["ca_params"]
    regen = st.sidebar.slider("Resource regen", 0.0, 0.2, float(ca_params.regen_rate))
    directional = st.sidebar.slider("Polarity strength", 0.1, 2.0, float(ca_params.directional_gain))
    division = st.sidebar.slider("Division threshold", 0.2, 0.9, float(ca_params.division_threshold))
    data = {**ca_params.__dict__, "regen_rate": regen, "directional_gain": directional, "division_threshold": division}
    st.session_state["ca_params"] = CAParams(**data)

    cols = st.columns(3)
    if cols[0].button("Step x20"):
        st.session_state["paused"] = False
        for _ in range(20):
            result = st.session_state["stepper"].step(st.session_state["state"], st.session_state["ca_params"])
            st.session_state["state"] = result.state
            st.session_state["history"].append(result.state.copy())
            st.session_state["metrics"]["mass"].append(result.stats.mass)
            st.session_state["metrics"]["resource"].append(result.stats.resource)
            st.session_state["metrics"]["reproduction"].append(result.stats.reproduction_events)
    if cols[1].button("Pause/Resume"):
        st.session_state["paused"] = not st.session_state["paused"]
    if cols[2].button("Rewind 50 steps") and len(st.session_state["history"]) > 50:
        st.session_state["state"] = st.session_state["history"][-50].copy()

    st.markdown("### Live CA view")
    _render_snapshot(st.session_state["state"])

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


if __name__ == "__main__":
    main()

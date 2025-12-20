"""Phylogeny and lineage visualization utilities."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:  # pragma: no cover
    import networkx as nx


logger = logging.getLogger(__name__)


def _load_lineage(run_dir: Path) -> list[dict]:
    path = run_dir / "lineage.jsonl"
    if not path.exists():
        genome_path = run_dir / "genome.json"
        if genome_path.exists():
            genome = json.loads(genome_path.read_text()).get("genes", [])
        else:
            genome = []
        return [
            {
                "individual_id": "root",
                "generation": 0,
                "parents": [],
                "fitness": {},
                "species_id": None,
                "phenotype_descriptor": [],
                "genome": genome,
            }
        ]
    lines = []
    for line in path.read_text().strip().splitlines():
        try:
            lines.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return lines


def _get_networkx():
    try:
        import networkx as nx  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("networkx is required for phylogeny analysis") from exc
    return nx


def build_lineage_graph(run_dir: Path):
    nx = _get_networkx()
    data = _load_lineage(run_dir)
    g = nx.DiGraph()
    for entry in data:
        node_id = entry.get("individual_id")
        genome_vec = entry.get("genome", [])
        phenotype_vec = entry.get("phenotype_descriptor", [])
        g.add_node(
            node_id,
            generation=entry.get("generation", 0),
            species_id=entry.get("species_id"),
            fitness=entry.get("fitness", {}),
            genome=list(genome_vec),
            phenotype=list(phenotype_vec),
            birth_step=entry.get("birth_step"),
            death_step=entry.get("death_step"),
        )
        for parent in entry.get("parents", []) or []:
            g.add_edge(parent, node_id)
    return g


def compute_branch_lengths(graph, mode: Literal["genome", "phenotype"] = "genome"):
    for u, v in graph.edges():
        source = np.asarray(graph.nodes[u].get("genome" if mode == "genome" else "phenotype"), dtype=float)
        target = np.asarray(graph.nodes[v].get("genome" if mode == "genome" else "phenotype"), dtype=float)
        if source is None or target is None or len(source) == 0 or len(target) == 0:
            length = 1.0
        else:
            length = float(np.linalg.norm(source - target))
        graph.edges[u, v]["length"] = length
    return graph


def export_graphml(graph, path: Path) -> Path:
    nx = _get_networkx()
    serializable = graph.copy()
    for node, data in serializable.nodes(data=True):
        for key, value in list(data.items()):
            if value is None:
                data[key] = "None"
        if "genome" in data:
            data["genome"] = json.dumps(data["genome"])
        if "phenotype" in data:
            data["phenotype"] = json.dumps(data["phenotype"])
        if "fitness" in data and isinstance(data["fitness"], dict):
            data["fitness"] = json.dumps(data["fitness"])
    try:
        nx.write_graphml(serializable, path)
    except Exception as exc:  # pragma: no cover - dependency optional
        fallback = path.with_suffix(".json")
        fallback.write_text(json.dumps(nx.node_link_data(serializable), indent=2))
        logger.warning("GraphML export failed (%s); wrote JSON lineage to %s", exc, fallback)
        return fallback
    return path


def export_newick(graph, path: Path) -> Path:
    roots = [n for n, deg in graph.in_degree() if deg == 0]
    if not roots:
        return path

    def _to_newick(node: str) -> str:
        children = list(graph.successors(node))
        if not children:
            return node
        child_str = ",".join(_to_newick(c) for c in children)
        return f"({child_str}){node}"

    newick_str = _to_newick(roots[0]) + ";"
    path.write_text(newick_str)
    return path


def render_phylogeny(graph, out_path_png: Path, out_path_html: Path | None = None):
    nx = _get_networkx()
    pos = nx.spring_layout(graph, seed=0)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
    species_ids = [graph.nodes[n].get("species_id", -1) for n in graph.nodes]
    nodes = nx.draw_networkx_nodes(graph, pos, node_color=species_ids, cmap="tab20", ax=ax)
    nx.draw_networkx_edges(graph, pos, ax=ax)
    labels = {
        n: f"{n}\nGen {graph.nodes[n].get('generation', 0)} | S{graph.nodes[n].get('species_id', '-') }"
        for n in graph.nodes
    }
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=7, ax=ax)
    ax.axis("off")
    fig.colorbar(nodes, ax=ax, label="species id")
    out_path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path_png, bbox_inches="tight")
    plt.close(fig)
    if out_path_html is not None:
        try:
            import plotly.graph_objects as go
            edge_x = []
            edge_y = []
            for u, v in graph.edges():
                edge_x.extend([pos[u][0], pos[v][0], None])
                edge_y.extend([pos[u][1], pos[v][1], None])
            node_x = [pos[n][0] for n in graph.nodes]
            node_y = [pos[n][1] for n in graph.nodes]
            fig_html = go.Figure()
            fig_html.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="#888")))
            fig_html.add_trace(
                go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode="markers+text",
                    text=[labels[n] for n in graph.nodes],
                    textposition="top center",
                    marker=dict(color=species_ids, colorscale="Turbo", size=10),
                )
            )
            out_path_html.parent.mkdir(parents=True, exist_ok=True)
            fig_html.write_html(out_path_html)
        except Exception:
            pass
    return out_path_png


def run_phylogeny_pipeline(run_dir: Path, *, mode: Literal["genome", "phenotype"] = "genome", out_dir: Path | None = None) -> Path:
    graph = build_lineage_graph(run_dir)
    graph = compute_branch_lengths(graph, mode=mode)
    out_dir = out_dir or (run_dir / "phylogeny")
    out_dir.mkdir(parents=True, exist_ok=True)
    export_graphml(graph, out_dir / "lineage.graphml")
    export_newick(graph, out_dir / "phylogeny.nwk")
    png_path = out_dir / "phylogeny.png"
    html_path = out_dir / "phylogeny.html"
    render_phylogeny(graph, png_path, html_path)
    return png_path

"""Component-level lineage tracking with reproduction detection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np

from xenogenesis.substrates.ca.fitness import _components


@dataclass
class Track:
    track_id: str
    centroid: np.ndarray
    mass: float
    area: float
    birth_step: int
    last_seen: int
    parent_id: str | None = None
    age: int = 0


@dataclass
class LineageEvent:
    event: str
    step: int
    track_id: str
    parent_id: str | None
    mass: float
    area: float


class LineageTracker:
    """Track connected components across time and flag reproduction/survival events."""

    def __init__(self, *, threshold: float = 0.15, match_radius: float = 6.0, persistence: int = 3):
        self.threshold = threshold
        self.match_radius = match_radius
        self.persistence = persistence
        self._tracks: Dict[str, Track] = {}
        self._events: List[LineageEvent] = []
        self._next_id = 0

    @property
    def events(self) -> list[dict]:
        return [event.__dict__ for event in self._events]

    def _new_id(self) -> str:
        tid = f"c{self._next_id:05d}"
        self._next_id += 1
        return tid

    def update(self, biomass: np.ndarray, step: int) -> dict:
        comps = _components(biomass, self.threshold)
        assignments: dict[str, list[dict]] = {}
        for comp in comps:
            centroid = comp["centroid"]
            best_id = None
            best_dist = 1e9
            for tid, track in self._tracks.items():
                dist = float(np.linalg.norm(track.centroid - centroid))
                if dist < best_dist and dist <= self.match_radius:
                    best_dist = dist
                    best_id = tid
            if best_id is not None:
                assignments.setdefault(best_id, []).append(comp)
            else:
                new_id = self._new_id()
                self._tracks[new_id] = Track(
                    track_id=new_id,
                    centroid=centroid,
                    mass=comp["mass"],
                    area=comp["area"],
                    birth_step=step,
                    last_seen=step,
                    parent_id=None,
                )
                self._events.append(LineageEvent("birth", step, new_id, None, comp["mass"], comp["area"]))

        reproduction_events = 0
        for tid, comp_list in assignments.items():
            comp_list = sorted(comp_list, key=lambda c: c["mass"], reverse=True)
            primary = comp_list[0]
            track = self._tracks[tid]
            track.centroid = primary["centroid"]
            track.mass = primary["mass"]
            track.area = primary["area"]
            track.last_seen = step
            track.age += 1
            if len(comp_list) > 1:
                reproduction_events += len(comp_list) - 1
                for child in comp_list[1:]:
                    child_id = self._new_id()
                    self._tracks[child_id] = Track(
                        track_id=child_id,
                        centroid=child["centroid"],
                        mass=child["mass"],
                        area=child["area"],
                        birth_step=step,
                        last_seen=step,
                        parent_id=tid,
                        age=1,
                    )
                    self._events.append(LineageEvent("reproduction", step, child_id, tid, child["mass"], child["area"]))

        deaths: list[str] = []
        for tid in list(self._tracks.keys()):
            track = self._tracks[tid]
            if step - track.last_seen > self.persistence:
                deaths.append(tid)
        for tid in deaths:
            track = self._tracks.pop(tid)
            self._events.append(LineageEvent("death", step, track.track_id, track.parent_id, track.mass, track.area))

        return {
            "component_count": len(comps),
            "reproduction_events": reproduction_events,
            "active_tracks": len(self._tracks),
        }

    def serialize_tracks(self) -> list[dict]:
        return [track.__dict__ for track in self._tracks.values()]

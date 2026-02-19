# builder.py
#
# Stim circuit builder for a surface-code memory experiment in Z basis.
# - Consumes a topology object (e.g. PatchTopology from topology.py)
# - Produces a stim.Circuit with DETECTORs and OBSERVABLE_INCLUDE
# - Does not assume the patch is rectangular; only uses topo.neighbors and topo.coupling_layers (if provided)

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable, Literal

import stim

Qid = int
Coord = Tuple[int, int]
StabType = Literal["x", "z"]


@dataclass(frozen=True)
class MemoryZConfig:
    rounds: int
    include_coords: bool = True
    include_shift_coords: bool = True
    include_inter_round_detectors: bool = True
    include_termination_detectors: bool = True


class SurfaceCodeMemoryZBuilder:
    """
    Builds a Z-basis memory circuit (rotated_memory_z-style) from topology.

    Required topology attributes/methods:
      topo.data: set[int]
      topo.anc_x: set[int]
      topo.anc_z: set[int]
      topo.coords: dict[int, (y,x)]   (or any consistent (y,x) convention)
      topo.neighbors: dict[anc_qid -> list[Optional[data_qid]]] of length 4 (NW,NE,SW,SE) with None allowed
      topo.logical_z: list[int]       (data qubits defining logical Z support for OBSERVABLE_INCLUDE)
      topo.type_of_ancilla(anc_qid) -> "x" or "z"   (or you can infer via anc_x/anc_z sets)

    Optional:
      topo.coupling_layers: list[list[(anc_qid, data_qid)]] with length 4
        If absent/empty, layers are derived from topo.neighbors ordering.
    """

    def __init__(self, topo, *, config: MemoryZConfig):
        if config.rounds < 1:
            raise ValueError("rounds must be >= 1")
        self.topo = topo
        self.cfg = config

        # measurement bookkeeping
        self._m_count: int = 0  # total number of measurements appended so far
        self._anc_meas: List[Dict[Qid, int]] = []  # per round: ancilla -> absolute meas index
        self._data_meas: Dict[Qid, int] = {}       # final: data -> absolute meas index

    # ----------------------------
    # Measurement record utilities
    # ----------------------------
    def _rec(self, meas_index: int) -> stim.GateTarget:
        """
        Convert an absolute measurement index (0..m_count-1) into stim rec[-k].
        """
        return stim.target_rec(meas_index - self._m_count)

    def _sorted(self, xs: Iterable[Qid]) -> List[Qid]:
        return sorted(xs)

    # ----------------------------
    # Topology helpers
    # ----------------------------
    def _anc_type(self, anc: Qid) -> StabType:
        # Prefer method if present
        if hasattr(self.topo, "type_of_ancilla"):
            return self.topo.type_of_ancilla(anc)
        # Fall back to membership
        if anc in self.topo.anc_x:
            return "x"
        if anc in self.topo.anc_z:
            return "z"
        raise KeyError(f"{anc} is not an ancilla")

    def _layers(self) -> List[List[Tuple[Qid, Qid]]]:
        """
        Return 4 coupling layers as lists of (anc,data) edges.

        If topo.coupling_layers exists and has length 4, use it.
        Otherwise derive from topo.neighbors[anc] order (must be length 4).
        """
        layers = getattr(self.topo, "coupling_layers", None)
        if layers and isinstance(layers, list) and len(layers) == 4:
            return layers

        derived: List[List[Tuple[Qid, Qid]]] = [[], [], [], []]
        for anc, nbs in self.topo.neighbors.items():
            if len(nbs) != 4:
                raise ValueError(f"neighbors[{anc}] must have length 4 (NW,NE,SW,SE)")
            for k, dq in enumerate(nbs):
                if dq is None:
                    continue
                derived[k].append((anc, dq))
        return derived

    # ----------------------------
    # Circuit blocks
    # ----------------------------
    def _append_qubit_coords(self, c: stim.Circuit) -> None:
        # Emit coordinates for visualization; stim uses QUBIT_COORDS(x,y) qid by convention.
        # If your topo.coords is (y,x), swap to (x,y).
        for qid, (y, x) in self.topo.coords.items():
            c.append("QUBIT_COORDS", [qid], [float(x), float(y)])

    def _append_init_data(self, c: stim.Circuit) -> None:
        data = self._sorted(self.topo.data)
        if data:
            c.append("RZ", data)
            # <-- noise hook here (reset errors) if you later want
        c.append("TICK")

    def _append_syndrome_round(self, c: stim.Circuit, t: int) -> None:
        anc_all = self._sorted(self.topo.anc_x | self.topo.anc_z)
        anc_x = self._sorted(self.topo.anc_x)

        # Reset ancillas
        if anc_all:
            c.append("RZ", anc_all)
            # <-- noise hook here (reset errors)
        c.append("TICK")

        # Prepare X ancillas (H)
        if anc_x:
            c.append("H", anc_x)
            # <-- noise hook here (1q noise after H)
        c.append("TICK")

        # 4 CX layers
        for layer in self._layers():
            for anc, dq in layer:
                tanc = self._anc_type(anc)
                if tanc == "z":
                    # Z-check: data -> anc
                    c.append("CX", [dq, anc])
                else:
                    # X-check: anc -> data
                    c.append("CX", [anc, dq])
                # <-- noise hook here (2q noise after CX)
            c.append("TICK")

        # Unprepare X ancillas
        if anc_x:
            c.append("H", anc_x)
            # <-- noise hook here (1q noise after H)
        c.append("TICK")

        # Measure ancillas (MZ)
        # <-- noise hook here (meas flip before MZ)
        if anc_all:
            c.append("MZ", anc_all)

        # Record measurement indices for this round
        meas_map: Dict[Qid, int] = {}
        for a in anc_all:
            meas_map[a] = self._m_count
            self._m_count += 1
        self._anc_meas.append(meas_map)

        if self.cfg.include_shift_coords:
            c.append("SHIFT_COORDS", [], [0.0, 0.0, 1.0])

    def _append_inter_round_detectors(self, c: stim.Circuit, t: int) -> None:
        if not self.cfg.include_inter_round_detectors:
            return
        if t <= 0:
            return

        cur = self._anc_meas[t]
        prev = self._anc_meas[t - 1]

        for anc, cur_idx in cur.items():
            prev_idx = prev.get(anc)
            if prev_idx is None:
                continue
            y, x = self.topo.coords.get(anc, (0, 0))
            c.append(
                "DETECTOR",
                [self._rec(cur_idx), self._rec(prev_idx)],
                [float(x), float(y), float(t)],
            )

    def _append_termination(self, c: stim.Circuit) -> None:
        data = self._sorted(self.topo.data)

        # Measure data (MZ)
        # <-- noise hook here (meas flip before MZ)
        if data:
            c.append("MZ", data)

        for q in data:
            self._data_meas[q] = self._m_count
            self._m_count += 1

        # Final detectors for Z stabilizers (tie last ancilla meas to neighbor data meas)
        if self.cfg.include_termination_detectors and self._anc_meas:
            last_round = len(self._anc_meas) - 1
            last = self._anc_meas[last_round]

            for anc in self._sorted(self.topo.anc_z):
                anc_idx = last.get(anc)
                if anc_idx is None:
                    continue

                nbs = self.topo.neighbors.get(anc)
                if nbs is None:
                    continue

                targets: List[stim.GateTarget] = [self._rec(anc_idx)]
                for dq in nbs:
                    if dq is None:
                        continue
                    d_idx = self._data_meas.get(dq)
                    if d_idx is None:
                        continue
                    targets.append(self._rec(d_idx))

                y, x = self.topo.coords.get(anc, (0, 0))
                c.append(
                    "DETECTOR",
                    targets,
                    [float(x), float(y), float(last_round + 1)],
                )

        # Observable include (logical Z on final data measurements)
        if not getattr(self.topo, "logical_z", None):
            raise ValueError("topology.logical_z is empty; set it before building the circuit.")

        obs_targets: List[stim.GateTarget] = []
        for dq in self.topo.logical_z:
            idx = self._data_meas.get(dq)
            if idx is None:
                raise RuntimeError(f"logical_z qubit {dq} has no final measurement index")
            obs_targets.append(self._rec(idx))

        c.append("OBSERVABLE_INCLUDE", obs_targets, 0)

    # ----------------------------
    # Public API
    # ----------------------------
    def build(self) -> stim.Circuit:
        c = stim.Circuit()

        # reset bookkeeping
        self._m_count = 0
        self._anc_meas = []
        self._data_meas = {}

        if self.cfg.include_coords:
            self._append_qubit_coords(c)

        self._append_init_data(c)

        for t in range(self.cfg.rounds):
            self._append_syndrome_round(c, t)
            self._append_inter_round_detectors(c, t)

        self._append_termination(c)
        return c
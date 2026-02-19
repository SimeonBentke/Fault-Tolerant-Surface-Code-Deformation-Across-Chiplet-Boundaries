# topology.py
#
# Pure topology/geometry for planar *rotated* surface-code patches.
# - No Stim
# - No noise
# - No detectors / measurements
#
# This module builds a static graph:
#   data qubits  <-->  ancilla qubits (X-check / Z-check)
# with coordinates and stabilizer neighborhoods.
#
# Coordinate convention (internal):
#   - data qubits live on odd/odd coordinates in a (2d-1) x (2d-1) grid
#   - ancillas live on even/even coordinates in a (2d+1) x (2d+1) grid
#   - an ancilla at (y,x) couples to diagonal data at (y±1, x±1)
#
# NOTE:
#   There are several equivalent rotated-code conventions in the literature.
#   This file picks one consistent convention that is:
#     - easy to generalize to non-rectangular masks
#     - compatible with standard 4-layer diagonal CX scheduling
#
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple, Literal

Qid = int
Coord = Tuple[int, int]  # (y, x)
StabType = Literal["x", "z"]

# Fixed diagonal neighbor ordering used by this topology.
# 0: NW, 1: NE, 2: SW, 3: SE
NEIGHBOR_DIRS: Tuple[Tuple[int, int], ...] = (
    (-1, -1),  # NW
    (-1, +1),  # NE
    (+1, -1),  # SW
    (+1, +1),  # SE
)


@dataclass(frozen=True)
class PatchTopology:
    """
    Pure static topology of a surface-code-like patch.

    Stores:
      - sets of data and ancilla qubits (X / Z)
      - coordinates for each qubit
      - stabilizer neighborhoods: ancilla -> [NW,NE,SW,SE] data qubits (None if missing)
      - canonical logical operator supports (as data-qubit id lists)
      - optional: precomputed coupling layers for syndrome extraction

    Does NOT store:
      - any circuit instructions
      - measurement indexing / detectors
      - noise rates
    """

    name: str
    d: int
    origin: Coord = (0, 0)

    # qubit identity partition
    data: Set[Qid] = field(default_factory=set)
    anc_x: Set[Qid] = field(default_factory=set)
    anc_z: Set[Qid] = field(default_factory=set)

    # geometry
    coords: Dict[Qid, Coord] = field(default_factory=dict)
    coord_to_qid: Dict[Coord, Qid] = field(default_factory=dict)

    # stabilizer graph: anc -> ordered diagonal neighbors [NW,NE,SW,SE]
    neighbors: Dict[Qid, List[Optional[Qid]]] = field(default_factory=dict)

    # optional explicit type map (otherwise infer from anc_x/anc_z)
    stab_type: Dict[Qid, StabType] = field(default_factory=dict)

    # logical operators (supports on data qubits)
    logical_z: List[Qid] = field(default_factory=list)
    logical_x: List[Qid] = field(default_factory=list)

    # optional: conflict-free coupling layers (each layer: list of (anc, data) edges)
    coupling_layers: List[List[Tuple[Qid, Qid]]] = field(default_factory=list)

    # ---------------- Convenience ----------------

    def all_qubits(self) -> Set[Qid]:
        return set(self.coords.keys())

    def ancillas(self) -> Set[Qid]:
        return set(self.anc_x) | set(self.anc_z)

    def type_of_ancilla(self, anc: Qid) -> StabType:
        if anc in self.stab_type:
            return self.stab_type[anc]
        if anc in self.anc_x:
            return "x"
        if anc in self.anc_z:
            return "z"
        raise KeyError(f"qid {anc} is not an ancilla")

    def iter_edges(self) -> Iterable[Tuple[Qid, Qid]]:
        """Iterate all undirected ancilla-data edges implied by neighborhoods."""
        for a, nbs in self.neighbors.items():
            for q in nbs:
                if q is not None:
                    yield (a, q)

    def validate_basic(self) -> None:
        # disjoint partitions
        if (self.data & self.anc_x) or (self.data & self.anc_z) or (self.anc_x & self.anc_z):
            raise ValueError("data/anc_x/anc_z sets must be pairwise disjoint")

        # coords for all known qubits
        for q in (self.data | self.anc_x | self.anc_z):
            if q not in self.coords:
                raise ValueError(f"Missing coords for qid {q}")

        # coord_to_qid consistency
        for q, c in self.coords.items():
            if self.coord_to_qid.get(c) != q:
                raise ValueError("coord_to_qid is inconsistent with coords")

        # neighborhoods consistent
        for a, nbs in self.neighbors.items():
            if a not in self.ancillas():
                raise ValueError(f"neighbors defined for non-ancilla qid {a}")
            if len(nbs) != 4:
                raise ValueError(f"neighbors[{a}] must have length 4 (NW,NE,SW,SE)")
            for q in nbs:
                if q is None:
                    continue
                if q not in self.data:
                    raise ValueError(f"Ancilla {a} has neighbor {q} that is not a data qubit")

        # logical supports are data
        for q in self.logical_z:
            if q not in self.data:
                raise ValueError(f"logical_z contains non-data qid {q}")
        for q in self.logical_x:
            if q not in self.data:
                raise ValueError(f"logical_x contains non-data qid {q}")


def _classify_ancilla_type_rotated(y: int, x: int) -> StabType:
    """
    Checkerboard classification on the even-even ancilla grid.
    This yields an alternating X/Z tiling.

    Many conventions are valid. Keep it consistent across the project.
    """
    # y,x are even. Using (x+y) mod 4 is a common simple rule.
    return "x" if ((x + y) % 4 == 0) else "z"


def make_rotated_planar_patch(
    d: int,
    *,
    origin: Coord = (0, 0),
    name: str = "rotated_planar",
    keep_lonely_ancillas: bool = False,
) -> PatchTopology:
    """
    Build a rectangular rotated planar patch of distance d.

    - data qubits: odd/odd coords inside [1..2d-1]^2 (size d^2)
    - ancillas: even/even coords inside [0..2d]^2, then filtered to those that touch >=2 data qubits

    keep_lonely_ancillas:
      If False (default), ancillas with <2 neighbors are dropped (corner artifacts).
      If True, keep them (rarely useful).
    """
    if d < 3:
        raise ValueError("Typically d>=3 for a meaningful planar surface-code patch.")

    oy, ox = origin

    coords: Dict[Qid, Coord] = {}
    coord_to_qid: Dict[Coord, Qid] = {}
    data: Set[Qid] = set()
    anc_x: Set[Qid] = set()
    anc_z: Set[Qid] = set()
    neighbors: Dict[Qid, List[Optional[Qid]]] = {}
    stab_type: Dict[Qid, StabType] = {}

    def new_qubit(c: Coord) -> Qid:
        q = len(coords)
        coords[q] = c
        coord_to_qid[c] = q
        return q

    # 1) Create data qubits (odd/odd)
    data_coords: Set[Coord] = set()
    for y in range(1, 2 * d, 2):
        for x in range(1, 2 * d, 2):
            c = (oy + y, ox + x)
            data_coords.add(c)
            q = new_qubit(c)
            data.add(q)

    # 2) Create candidate ancillas (even/even)
    anc_coords: List[Coord] = []
    for y in range(0, 2 * d + 1, 2):
        for x in range(0, 2 * d + 1, 2):
            anc_coords.append((oy + y, ox + x))

    # We'll create them now, but may later drop some
    for c in anc_coords:
        q = new_qubit(c)
        # classify by local coords relative to origin (strip offset)
        rel_y, rel_x = c[0] - oy, c[1] - ox
        t = _classify_ancilla_type_rotated(rel_y, rel_x)
        stab_type[q] = t
        (anc_x if t == "x" else anc_z).add(q)

    # 3) Build neighborhoods and optionally prune ancillas
    def coord_is_data(c: Coord) -> Optional[Qid]:
        q = coord_to_qid.get(c)
        if q is None:
            return None
        return q if q in data else None

    to_drop: List[Qid] = []
    for a in list(anc_x | anc_z):
        ay, ax = coords[a]
        nbs: List[Optional[Qid]] = []
        for dy, dx in NEIGHBOR_DIRS:
            nbs.append(coord_is_data((ay + dy, ax + dx)))
        touch = sum(q is not None for q in nbs)
        if (not keep_lonely_ancillas) and touch < 2:
            to_drop.append(a)
        else:
            neighbors[a] = nbs

    if to_drop:
        # Remove dropped ancillas from partitions and maps.
        # NOTE: we keep their qids unused; that’s fine for topology purposes.
        # If you want contiguous qids, build a remapping step.
        for a in to_drop:
            anc_x.discard(a)
            anc_z.discard(a)
            stab_type.pop(a, None)
            neighbors.pop(a, None)

    topo = PatchTopology(
        name=name,
        d=d,
        origin=origin,
        data=data,
        anc_x=anc_x,
        anc_z=anc_z,
        coords=coords,
        coord_to_qid=coord_to_qid,
        neighbors=neighbors,
        stab_type=stab_type,
    )

    # 4) Canonical logical operators (simple “middle line” representatives)
    topo_lz = compute_logical_z_path(topo)
    topo_lx = compute_logical_x_path(topo)

    # 5) Coupling layers (diagonal buckets)
    layers = compute_coupling_layers_from_neighbors(topo)

    # Return a new frozen dataclass instance with these filled
    return PatchTopology(
        **{**topo.__dict__, "logical_z": topo_lz, "logical_x": topo_lx, "coupling_layers": layers}
    )


def make_masked_rotated_patch(
    d: int,
    *,
    mask_data: Callable[[int, int], bool],
    origin: Coord = (0, 0),
    name: str = "masked_rotated",
    min_ancilla_neighbors: int = 2,
) -> PatchTopology:
    """
    Build a rotated patch where the *data qubits* are filtered by a mask.
    Ancillas are included if they touch at least `min_ancilla_neighbors` masked data qubits.

    mask_data(y, x) is evaluated on *relative* coords in the data grid domain:
      y,x in {1,3,...,2d-1} (odd values).
    """
    if d < 3:
        raise ValueError("Typically d>=3 for a meaningful planar surface-code patch.")
    if min_ancilla_neighbors < 1 or min_ancilla_neighbors > 4:
        raise ValueError("min_ancilla_neighbors must be in {1,2,3,4}")

    oy, ox = origin

    coords: Dict[Qid, Coord] = {}
    coord_to_qid: Dict[Coord, Qid] = {}
    data: Set[Qid] = set()
    anc_x: Set[Qid] = set()
    anc_z: Set[Qid] = set()
    neighbors: Dict[Qid, List[Optional[Qid]]] = {}
    stab_type: Dict[Qid, StabType] = {}

    def new_qubit(c: Coord) -> Qid:
        q = len(coords)
        coords[q] = c
        coord_to_qid[c] = q
        return q

    # 1) masked data qubits
    for y in range(1, 2 * d, 2):
        for x in range(1, 2 * d, 2):
            if not mask_data(y, x):
                continue
            c = (oy + y, ox + x)
            q = new_qubit(c)
            data.add(q)

    # 2) candidate ancillas
    anc_candidates: List[Tuple[Coord, StabType]] = []
    for y in range(0, 2 * d + 1, 2):
        for x in range(0, 2 * d + 1, 2):
            c = (oy + y, ox + x)
            t = _classify_ancilla_type_rotated(y, x)
            anc_candidates.append((c, t))

    for c, t in anc_candidates:
        q = new_qubit(c)
        stab_type[q] = t
        (anc_x if t == "x" else anc_z).add(q)

    # 3) neighborhoods & prune ancillas by masked connectivity
    def coord_is_data(c: Coord) -> Optional[Qid]:
        q = coord_to_qid.get(c)
        if q is None:
            return None
        return q if q in data else None

    to_drop: List[Qid] = []
    for a in list(anc_x | anc_z):
        ay, ax = coords[a]
        nbs: List[Optional[Qid]] = []
        for dy, dx in NEIGHBOR_DIRS:
            nbs.append(coord_is_data((ay + dy, ax + dx)))
        touch = sum(q is not None for q in nbs)
        if touch < min_ancilla_neighbors:
            to_drop.append(a)
        else:
            neighbors[a] = nbs

    for a in to_drop:
        anc_x.discard(a)
        anc_z.discard(a)
        stab_type.pop(a, None)
        neighbors.pop(a, None)

    topo = PatchTopology(
        name=name,
        d=d,
        origin=origin,
        data=data,
        anc_x=anc_x,
        anc_z=anc_z,
        coords=coords,
        coord_to_qid=coord_to_qid,
        neighbors=neighbors,
        stab_type=stab_type,
    )

    # logical operators in masked codes are not always obvious.
    # We'll attempt a simple “middle line” representative restricted to available data.
    # If it fails, leave empty and let user set it later.
    lz: List[Qid] = []
    lx: List[Qid] = []
    try:
        lz = compute_logical_z_path(topo)
        lx = compute_logical_x_path(topo)
    except RuntimeError:
        pass

    layers = compute_coupling_layers_from_neighbors(topo)

    return PatchTopology(
        **{**topo.__dict__, "logical_z": lz, "logical_x": lx, "coupling_layers": layers}
    )


def compute_coupling_layers_from_neighbors(topo: PatchTopology) -> List[List[Tuple[Qid, Qid]]]:
    """
    Return 4 layers corresponding to the fixed neighbor ordering (NW,NE,SW,SE).
    Each layer contains (ancilla_qid, data_qid) edges.

    Builders can orient these edges depending on stabilizer type (X vs Z).
    """
    layers: List[List[Tuple[Qid, Qid]]] = [[], [], [], []]
    for a, nbs in topo.neighbors.items():
        for k, q in enumerate(nbs):
            if q is None:
                continue
            layers[k].append((a, q))
    return layers


def compute_logical_z_path(topo: PatchTopology) -> List[Qid]:
    """
    Return a simple canonical logical-Z support: the middle column of data qubits
    (in x) of length d, ordered from south to north.

    This assumes the standard rectangular rotated patch construction. For masked
    patches, this may fail (then raise RuntimeError).
    """
    d = topo.d
    oy, ox = topo.origin

    # data x positions are ox + 1, ox + 3, ..., ox + (2d-1)
    odd_xs = [ox + (2 * i + 1) for i in range(d)]
    x_mid = odd_xs[(d - 1) // 2]

    # collect qids at (y, x_mid) for y=oy+1,oy+3,...,oy+(2d-1)
    path: List[Qid] = []
    for i in range(d):
        y = oy + (2 * i + 1)
        q = topo.coord_to_qid.get((y, x_mid))
        if q is None or q not in topo.data:
            raise RuntimeError("Cannot construct logical_z_path with this topology/convention.")
        path.append(q)
    return path


def compute_logical_x_path(topo: PatchTopology) -> List[Qid]:
    """
    Return a simple canonical logical-X support: the middle row of data qubits
    (in y) of length d, ordered from west to east.

    This assumes the standard rectangular rotated patch construction. For masked
    patches, this may fail (then raise RuntimeError).
    """
    d = topo.d
    oy, ox = topo.origin

    odd_ys = [oy + (2 * i + 1) for i in range(d)]
    y_mid = odd_ys[(d - 1) // 2]

    path: List[Qid] = []
    for i in range(d):
        x = ox + (2 * i + 1)
        q = topo.coord_to_qid.get((y_mid, x))
        if q is None or q not in topo.data:
            raise RuntimeError("Cannot construct logical_x_path with this topology/convention.")
        path.append(q)
    return path


# ---------------------------
# Small utilities (optional)
# ---------------------------

def summary(topo: PatchTopology) -> str:
    """Human-readable summary useful for debugging."""
    return (
        f"PatchTopology(name={topo.name!r}, d={topo.d}, origin={topo.origin})\n"
        f"  qubits: total={len(topo.coords)} data={len(topo.data)} "
        f"anc_x={len(topo.anc_x)} anc_z={len(topo.anc_z)}\n"
        f"  stabilizers: neighborhoods={len(topo.neighbors)}\n"
        f"  logical_z_len={len(topo.logical_z)} logical_x_len={len(topo.logical_x)}\n"
    )

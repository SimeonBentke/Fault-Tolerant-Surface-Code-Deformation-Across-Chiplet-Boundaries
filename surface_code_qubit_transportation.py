from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple, List

import stim
import numpy as np
import pymatching
import matplotlib.pyplot as plt


# -----------------------------
# Utilities: coords + swap layers
# -----------------------------

def extract_qubit_coords(c: stim.Circuit) -> Dict[int, Tuple[float, float]]:
    """Extract (x,y) coords from QUBIT_COORDS instructions."""
    coords: Dict[int, Tuple[float, float]] = {}
    for inst in c:
        if inst.name == "QUBIT_COORDS":
            args = inst.gate_args_copy()
            targets = inst.targets_copy()
            if len(args) >= 2:
                x, y = float(args[0]), float(args[1])
                for t in targets:
                    if t.is_qubit_target:
                        coords[int(t.value)] = (x, y)
        elif inst.name == "REPEAT":
            # Generated circuits sometimes put QUBIT_COORDS at top-level, but be safe:
            coords.update(extract_qubit_coords(inst.body_copy()))
    return coords


def build_swap_layers_for_shift_x(coords: Dict[int, Tuple[float, float]]) -> List[stim.Circuit]:
    """
    Returns two disjoint SWAP layers (parity 0 then parity 1) that together implement a 1-step translation in +x,
    on the graph induced by coordinate-neighbors at (x+1,y).

    We treat x as integer-like (Stim uses integer grid coords for generated surface code).
    """
    # Build lookup from (x,y) to qubit id.
    xy_to_q: Dict[Tuple[int, int], int] = {}
    for q, (x, y) in coords.items():
        xi, yi = int(round(x)), int(round(y))
        xy_to_q[(xi, yi)] = q

    def layer_for_parity(parity: int) -> stim.Circuit:
        used = set()
        circ = stim.Circuit()
        pairs: List[Tuple[int, int]] = []

        for (x, y), q in xy_to_q.items():
            if (x % 2) != parity:
                continue
            nbr = (x + 1, y)
            if nbr not in xy_to_q:
                continue
            q2 = xy_to_q[nbr]
            if q in used or q2 in used:
                continue
            used.add(q)
            used.add(q2)
            pairs.append((q, q2))

        if pairs:
            circ.append("TICK")
            for a, b in pairs:
                circ.append("SWAP", [a, b])
            circ.append("TICK")
        return circ

    return [layer_for_parity(0), layer_for_parity(1)]


def insert_motion_between_rounds(
    base: stim.Circuit,
    n_shifts: int,
) -> stim.Circuit:
    """
    Inserts SWAP layers between syndrome rounds to translate the patch in +x by n_shifts.

    We detect "round boundaries" by looking for SHIFT_COORDS with a nonzero time increment (typically (0,0,1)).
    We insert a full shift (two parity SWAP layers) after each such boundary until n_shifts is exhausted.
    """
    flat = base.flattened()

    coords = extract_qubit_coords(flat)
    swap_layers = build_swap_layers_for_shift_x(coords)

    out = stim.Circuit()
    shifts_done = 0

    for inst in flat:
        out.append(inst)

        # Heuristic: round boundary marker
        if shifts_done < n_shifts and inst.name == "SHIFT_COORDS":
            args = inst.gate_args_copy()
            # Typically SHIFT_COORDS(0,0,1) each round. Be permissive: any positive last arg triggers.
            if len(args) >= 3 and float(args[2]) > 0:
                # Insert one full +x shift: parity-0 swaps then parity-1 swaps.
                out += swap_layers[0]
                out += swap_layers[1]
                shifts_done += 1

    if shifts_done < n_shifts:
        raise ValueError(
            f"Not enough round boundaries found to insert {n_shifts} shifts; only inserted {shifts_done}. "
            f"Increase 'rounds' or adjust boundary detection."
        )

    return out


# -----------------------------
# Noise injection (simple, consistent)
# -----------------------------

ONE_Q_GATES = {
    "H", "S", "SQRT_X", "SQRT_X_DAG", "SQRT_Y", "SQRT_Y_DAG",
    "X", "Y", "Z",
    "RX", "RY", "RZ",  # (usually not present in generated circuits, but harmless)
}
TWO_Q_GATES = {"CX", "CY", "CZ", "ISWAP", "SWAP", "XCZ", "YCZ"}

MEAS_GATES = {"M", "MX", "MY", "MZ"}  # Stim uses M for Z-measure; generated circuits typically use M.
RESET_GATES = {"R", "RX", "RY", "RZ"}  # In Stim, R is Z-basis reset; generators use R.

# Note: We avoid trying to noise-tag everything perfectly; we aim for a stable, working model:
# - After every 1q/2q gate: depolarize.
# - Before every measurement: X_ERROR(p_meas) (for Z-measure this is a flip of the classical outcome).
# - After every reset: X_ERROR(p_reset).


def add_simple_circuit_noise(c: stim.Circuit, p: float, p_meas: float | None = None, p_reset: float | None = None) -> stim.Circuit:
    """
    Adds a simple phenomenological-ish noise model to an existing circuit by rewriting it instruction-by-instruction.

    Defaults:
      p_meas = p
      p_reset = p
    """
    if p_meas is None:
        p_meas = p
    if p_reset is None:
        p_reset = p

    out = stim.Circuit()

    for inst in c:
        name = inst.name

        if name == "REPEAT":
            body_noisy = add_simple_circuit_noise(inst.body_copy(), p, p_meas=p_meas, p_reset=p_reset)
            out.append("REPEAT", inst.repeat_count, body_noisy)
            continue

        # Copy instruction
        out.append(inst)

        # Targets as plain qubit indices (ignore rec[-k], detectors, etc.)
        targs = inst.targets_copy()
        qubits = [int(t.value) for t in targs if t.is_qubit_target]

        if not qubits:
            continue

        if name in ONE_Q_GATES:
            # depolarize each acted-on qubit
            out.append("DEPOLARIZE1", qubits, [p])

        elif name in TWO_Q_GATES:
            # depolarize the pair(s)
            # If multiple targets, Stim interprets them as pairs for 2q gates.
            # We'll just depolarize all qubits involved with DEPOLARIZE2 on each pair in order.
            # (Safe for generated circuits where gates are in 2q pairs.)
            it = iter(qubits)
            pairs = list(zip(it, it))
            for a, b in pairs:
                out.append("DEPOLARIZE2", [a, b], [p])

        elif name in MEAS_GATES or name == "M":
            # Put a classical-bit flip model by inserting X_ERROR before measurement.
            # For Z-basis measurement, X before M flips the outcome bit.
            out.append("X_ERROR", qubits, [p_meas])

        elif name in RESET_GATES or name == "R":
            # After reset, apply a flip error.
            out.append("X_ERROR", qubits, [p_reset])

    return out


# -----------------------------
# Logical error estimation
# -----------------------------

def estimate_logical_error_rate(c: stim.Circuit, shots: int, obs_index: int = 0) -> float:
    """
    Uses PyMatching to decode and estimates Pr[logical failure] for observable obs_index.
    """
    # Build matching graph from the circuit's detector error model.
    dem = c.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)

    sampler = c.compile_detector_sampler()
    dets, obs = sampler.sample(shots=shots, separate_observables=True)

    # predicted corrections for observables
    # pymatching returns corrections that should be XORed with obs to get residual logical flips.
    pred = matching.decode_batch(dets)
    residual = np.logical_xor(pred[:, obs_index], obs[:, obs_index])
    return float(np.mean(residual))


# -----------------------------
# Main: build circuit, sweep p, plot
# -----------------------------

@dataclass
class Params:
    distance: int = 5
    rounds: int = 30
    n_shifts: int = 10
    shots_per_point: int = 30_000

    # Sweep range
    p_min: float = 1e-4
    p_max: float = 3e-2
    num_points: int = 12


def build_moving_rotated_memory_z(params: Params, p: float) -> stim.Circuit:
    # 1) Generate ideal rotated memory-Z surface code
    base = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=params.distance,
        rounds=params.rounds,
        # keep generator noise off; we will inject noise ourselves
        before_round_data_depolarization=0,
        after_clifford_depolarization=0,
        after_reset_flip_probability=0,
        before_measure_flip_probability=0,
    )

    # 2) Insert motion (n_shifts translations) between rounds
    moved = insert_motion_between_rounds(base, n_shifts=params.n_shifts)

    # 3) Add simple circuit-level noise at rate p
    noisy = add_simple_circuit_noise(moved, p=p, p_meas=p, p_reset=p)

    return noisy


def main():
    params = Params()

    # sanity: need enough rounds to fit shifts (we insert at most 1 shift per round boundary)
    if params.n_shifts >= params.rounds:
        print("WARNING: n_shifts >= rounds. Increase rounds to have enough round boundaries.")

    ps = np.geomspace(params.p_min, params.p_max, params.num_points)
    pLs = []

    for p in ps:
        c = build_moving_rotated_memory_z(params, p=p)
        pL = estimate_logical_error_rate(c, shots=params.shots_per_point, obs_index=0)
        print(f"p={p:.3e}  ->  pL≈{pL:.3e}")
        pLs.append(pL)

    # Plot
    plt.figure()
    plt.loglog(ps, pLs, marker="o")
    plt.xlabel("Physical error rate p")
    plt.ylabel("Logical failure rate p_L (memory-Z observable)")
    plt.title(f"rotated_memory_z move +x by n={params.n_shifts} (d={params.distance}, rounds={params.rounds})")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

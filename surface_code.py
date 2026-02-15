# sweep_seam_plot.py
#
# Makes a plot of logical failure rate p_L vs seam 2q error rate p_seam,
# comparing multiple code distances on the same figure.
#
# Requirements:
#   pip install stim pymatching numpy matplotlib
#
# Run:
#   python sweep_seam_plot.py

from __future__ import annotations
import math
import stim
import numpy as np
import pymatching
import matplotlib.pyplot as plt


def extract_qubit_coords(c: stim.Circuit) -> dict[int, tuple[float, float]]:
    coords: dict[int, tuple[float, float]] = {}

    def walk(circ: stim.Circuit) -> None:
        for inst in circ:
            if inst.name == "REPEAT":
                walk(inst.body_copy())
                continue
            if inst.name == "QUBIT_COORDS":
                args = inst.gate_args_copy()
                targets = inst.targets_copy()
                if len(args) >= 2:
                    x, y = float(args[0]), float(args[1])
                    for t in targets:
                        if t.is_qubit_target:
                            coords[int(t.value)] = (x, y)

    walk(c)
    return coords


def add_seam_depolarize2_after_entangling_gates(
    c: stim.Circuit,
    p: float,
    p_seam: float,
    *,
    x_cut: float = 0.0,
    entangling_gates: tuple[str, ...] = ("CX", "CZ"),
) -> stim.Circuit:
    coords = extract_qubit_coords(c)

    def is_seam_pair(q1: int, q2: int) -> bool:
        # seam edge if the pair crosses the vertical cut x=x_cut
        if q1 not in coords or q2 not in coords:
            return False
        x1, _ = coords[q1]
        x2, _ = coords[q2]
        return (x1 <= x_cut < x2) or (x2 <= x_cut < x1)

    def process(circ: stim.Circuit) -> stim.Circuit:
        out = stim.Circuit()
        for inst in circ:
            if inst.name == "REPEAT":
                body_processed = process(inst.body_copy())
                out += stim.Circuit(f"REPEAT {inst.repeat_count} {{\n{body_processed}\n}}")
                continue

            out.append(inst.name, inst.targets_copy(), inst.gate_args_copy())

            if inst.name in entangling_gates:
                qubits = [int(t.value) for t in inst.targets_copy() if t.is_qubit_target]
                if len(qubits) % 2 != 0:
                    raise ValueError(f"Unexpected target count for {inst.name}: {qubits}")
                for q1, q2 in zip(qubits[0::2], qubits[1::2]):
                    p_use = p_seam if is_seam_pair(q1, q2) else p
                    if p_use > 0:
                        out.append("DEPOLARIZE2", [q1, q2], [p_use])
        return out

    return process(c)


def logical_failure_rate_with_pymatching(
    c: stim.Circuit,
    *,
    shots: int = 50_000,
    seed: int | None = 0,
) -> float:
    dem = c.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)

    sampler = c.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(shots, separate_observables=True)

    pred = matching.decode_batch(dets)
    pred = np.asarray(pred)
    obs = np.asarray(obs)

    if pred.ndim == 1:
        pred = pred[:, None]

    failures = np.any(pred != obs, axis=1)
    return float(np.mean(failures))


def logspace(a: float, b: float, n: int) -> np.ndarray:
    # 10^a ... 10^b
    return np.logspace(a, b, n)


def main() -> None:
    # -----------------------------
    # Experiment knobs you may tweak
    # -----------------------------
    task = "surface_code:rotated_memory_z"

    # Compare these distances (all on one plot)
    distances = [3, 5, 7]

    # Number of QEC rounds. Common choice: rounds ~ O(d)
    # Increase this if you want "longer memory".
    rounds_factor = 3  # rounds = rounds_factor * d

    # Baseline 2q depolarizing probability per entangling gate (non-seam)
    p_base_2q = 1e-3

    # Sweep seam 2q error probability
    p_seam_values = logspace(-4, -1, 11)  # 1e-4 ... 1e-1

    # Vertical seam location in generator coordinates
    x_cut = 0.0

    # Monte Carlo shots per point
    shots = 20_000

    # Optional "target" logical error rate line (like your professor drew)
    target_pL = 1e-3  # change to whatever you want (e.g. 1e-6, 1e-12, ...)

    # Random seed base (keeps runs reproducible-ish)
    seed0 = 123

    # -----------------------------
    # Run sweep
    # -----------------------------
    results: dict[int, list[float]] = {}

    for d in distances:
        r = rounds_factor * d
        base = stim.Circuit.generated(task, distance=d, rounds=r)
        y = []
        for i, p_seam in enumerate(p_seam_values):
            noisy = add_seam_depolarize2_after_entangling_gates(
                base,
                p=p_base_2q,
                p_seam=float(p_seam),
                x_cut=x_cut,
                entangling_gates=("CX", "CZ"),
            )
            pL = logical_failure_rate_with_pymatching(noisy, shots=shots, seed=seed0 + 1000 * d + i)
            y.append(pL)
            print(f"d={d:2d} rounds={r:2d} p_seam={p_seam:.2e} -> pL={pL:.3e}")
        results[d] = y

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure()
    for d in distances:
        plt.plot(p_seam_values, results[d], marker="o", label=f"d={d} (rounds={rounds_factor*d})")

    # Target line
    plt.axhline(target_pL, linestyle="--", label=f"target pL={target_pL:g}")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Seam 2-qubit error rate $p_{\\mathrm{seam}}$")
    plt.ylabel("Logical failure rate $p_L$")
    plt.title(f"{task} with baseline 2q error p={p_base_2q:g}, seam at x={x_cut:g}")
    plt.legend()
    plt.grid(True, which="both", linestyle=":", linewidth=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

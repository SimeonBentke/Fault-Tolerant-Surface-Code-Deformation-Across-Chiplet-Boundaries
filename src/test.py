# test.py
import argparse
from pathlib import Path

from topology import make_rotated_planar_patch, summary
from circuit_builder import SurfaceCodeMemoryZBuilder, MemoryZConfig


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--out", type=str, default="circuit", help="base output filename (no extension)")
    args = parser.parse_args()

    topo = make_rotated_planar_patch(args.d)
    topo.validate_basic()
    print(summary(topo))

    cfg = MemoryZConfig(rounds=args.rounds)
    circuit = SurfaceCodeMemoryZBuilder(topo, config=cfg).build()

    c0 = circuit.without_noise()

    base = Path(args.out).resolve()

    # timeslice view (SVG)
    timeslice_svg = str(c0.diagram("timeslice-svg"))
    write_text(base.with_name(base.name + "_timeslice.svg"), timeslice_svg)
    print("Wrote:", base.with_name(base.name + "_timeslice.svg"))

    # timeline 3D (HTML)
    timeline_3d = str(c0.diagram("timeline-3d"))
    write_text(base.with_name(base.name + "_timeline_3d.html"), timeline_3d)
    print("Wrote:", base.with_name(base.name + "_timeline_3d.html"))

    # standard timeline SVG
    timeline_svg = str(c0.diagram("timeline-svg"))
    write_text(base.with_name(base.name + "_timeline.svg"), timeline_svg)
    print("Wrote:", base.with_name(base.name + "_timeline.svg"))


if __name__ == "__main__":
    main()
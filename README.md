# Fault-Tolerant Surface-Code Deformation Across Chiplet Boundaries

This repository contains Python code and a companion notebook for exploring surface-code patches, qubit topologies, and error models in the context of fault-tolerant quantum computing across chiplet boundaries.

## Repository Structure

```text
.
|-- src/
|   |-- patch.py
|   |-- pqubit.py
|   |-- pqubit_error_model.py
|   `-- qubit_topology.py
|-- usage.ipynb
|-- output.png
`-- README.md
```

## What Is Included

- `src/patch.py`: Defines `surf_patch`, a surface-code patch model built on top of the topology abstraction and designed for `stim` simulations.
- `src/qubit_topology.py`: Provides shared topology and qubit-mapping utilities.
- `src/pqubit.py`: Implements a physical-qubit abstraction used throughout the codebase.
- `src/pqubit_error_model.py`: Stores per-qubit error parameters.
- `usage.ipynb`: Main notebook for experimenting with the model and generating plots/results.
- `output.png`: Example generated output from the notebook workflow.

## Requirements

The notebook imports the following Python packages:

- `numpy`
- `stim`
- `sinter`
- `matplotlib`
- `scipy`

A recent Python 3 version is recommended.

## Setup

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy stim sinter matplotlib scipy jupyter
```

If you are working from Windows PowerShell instead of WSL bash, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Usage

Launch Jupyter and open the notebook:

```bash
jupyter notebook usage.ipynb
```

The notebook demonstrates how to construct and analyze surface-code patches using the modules in `src/`.

## Notes

- The source package currently lives in `src/` and is imported directly from the notebook.
- `__pycache__` files may be generated after running the code locally.

## License

No license file is currently included in this repository. Add one if you plan to share or publish the project.

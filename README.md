# Pydeviseis
### A Devito-pySolver FWI Wrapper

A Python framework for performing acoustic and soon elastic Full-Waveform Inversion (FWI). This project integrates the high-performance finite-difference modeling of the [Devito](https://www.devitocodes.com/) framework with the `GenericSolver` optimization library.

## Features

* **Acoustic FWI**: Inverts for P-wave velocity models from seismic data.
* **Devito Backend**: Leverages Devito for fast and efficient wave propagation.
* **pySolver Integration**: Uses the `GenericSolver` optimization suite, including L-BFGS and NLCG algorithms.
* **Multi-shot & Multi-frequency**: Supports standard multi-shot FWI and a frequency continuation workflow for progressing from low to high frequencies.

## Installation

Follow these steps to set up the environment and install the necessary dependencies.

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### 2. Create a Python Environment

It is highly recommended to use a virtual environment. Using `conda` is often easiest for managing compiler dependencies.

```bash
# Using conda
conda create -n fwi_env python=3.10
conda activate fwi_env
```

### 3. Install Devito

Devito has specific system dependencies (like GCC and an MPI library). The most reliable way to install it is by following the **[Official Devito Installation Guide](https://www.devitocodes.com/download)**.

A common installation method is:
```bash
conda install -c conda-forge devito
```

However, I just install it use a pip install (SEE REQUIREMENTS).

### 4. Install Python Dependencies

Install all other required Python packages using the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
```
*(Note: If you have a compatible NVIDIA GPU and CUDA toolkit, you can uncomment the `cupy` line in the requirements file before running.)*

### 5. Install Local Packages

Install the `pydeviseis` and `GenericSolver` packages in "editable" mode. This links the installation to your source code, so any changes you make are immediately available.

```bash
# From the root directory of the project
pip install -e .
pip install -e pydeviseis/packages/pySolver/GenericSolver/python
```

## Quick Start: Check jupyter notebook attached

## Project Structure

* `pydeviseis/`: Main source code for the FWI wrapper.
    * `core/`: Low-level wrappers (e.g., `devito_adapters.py`).
    * `inversion/`: High-level FWI logic (e.g., `fwi.py`, `problems.py`).
    * `wave_equations/`: High-level interface to the physics (`acoustic.py`).
    * `packages/`: Contains the `GenericSolver` library.
    * `examples/notebooks/`: Example Jupyter notebooks demonstrating usage.
* `requirements.txt`: Project dependencies.
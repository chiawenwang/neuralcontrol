# Stateful Tangent Operator for Adjoint RHC of Elastic Rods

Code accompanying our submission. The repository implements adjoint-based
receding-horizon control (RHC) of a quasi-static elastic rod and adds a
**Stateful Tangent Operator (STO)** that caches the implicit-differentiation
inverse and reuses it under cheap, query-independent validity checks.

Three control tasks are included:

| Task | Script | Description |
| ---- | ------ | ----------- |
| 1 | `quasi_static_task_any_node.py` | drive a chosen rod node to a target (x, y) |
| 2 | `quasi_static_task_middle_tracking.py` | track sin / cos / triangle / semicircle / square paths |
| 3 | `quasi_static_task_letter_curve.py` | deform the rod into target letter shapes (C, U, M) |

Each script supports the same `--mode` selector:

| Mode | Solver |
| ---- | ------ |
| `adjoint` | full-horizon exact adjoint |
| `adjoint_sto` | full-horizon adjoint with STO reuse |
| `rhc` | receding-horizon exact adjoint |
| `rhc_sto` | receding-horizon adjoint with STO reuse |
| `spsa` | full-horizon SPSA baseline |
| `jfb` | full-horizon Jacobian-free backprop baseline |

## Layout

```
.
├── stateful_tangent_operator.py            # STO core (validate-before-reuse)
├── sto_adapter.py                          # adapts STO to the rod simulator's J_ff / J_fb
├── common.py                               # seeds, threads, sim IO, SPSA primitives, animations
├── utils.py                                # policy MLP and geometry helpers
├── trajectory.py                           # reference paths for Task 2
├── quasi_static_task_any_node.py           # Task 1
├── quasi_static_task_middle_tracking.py    # Task 2
├── quasi_static_task_letter_curve.py       # Task 3
├── src/                                    # C++ quasi-static rod simulator
├── CMakeLists.txt                          # build config for the C++ extension
├── setup.py                                # registers the compiled extension as `nn_der`
├── inputs/                                 # initial rod configurations
└── targets/                                # target letter shapes for Task 3
```

## Build

The Python scripts depend on a C++ pybind11 extension `nn_der` that wraps
the rod simulator. Build it once with CMake before running any task script.

### Conda (recommended)

```bash
conda create -n rod_sto python=3.10 -y
conda activate rod_sto

conda install -c conda-forge \
    cmake cxx-compiler eigen mkl mkl-devel pybind11 \
    numpy matplotlib liblapack -y
conda install pytorch cpuonly -c pytorch -y

export MKLROOT=$CONDA_PREFIX
export CMAKE_PREFIX_PATH=$CONDA_PREFIX:$CMAKE_PREFIX_PATH
export Eigen3_DIR=$CONDA_PREFIX/share/eigen3/cmake
```

### apt-get (Ubuntu/Debian)

```bash
sudo apt-get install build-essential cmake libeigen3-dev python3-dev python3-pip
# Install Intel oneAPI MKL separately, then
source /opt/intel/oneapi/setvars.sh
pip install numpy torch matplotlib
```

### Compile the extension

```bash
mkdir -p build && cd build
cmake ..
make -j
cd ..
pip install -e .

python -c "import nn_der.nn_der as py_der; print('OK')"
```

This produces `nn_der/nn_der*.so`, which the task scripts import.

## Usage

Each task script takes `--mode` plus optional overrides; all defaults live in
the `CONFIG` dict at the top of the file.

### Task 1 — Any node tracking

```bash
python quasi_static_task_any_node.py --mode rhc_sto --no_show
```

### Task 2 — Middle-node trajectory tracking

```bash
python quasi_static_task_middle_tracking.py --mode rhc_sto --no_show
```

### Task 3 — Letter curve deformation

```bash
python quasi_static_task_letter_curve.py --mode rhc_sto --no_show
# Run only one of the four built-in letter cases:
python quasi_static_task_letter_curve.py --mode rhc --case_indices 1 --no_show
```

Drop `--no_show` to display the matplotlib animation (one frame per MPC step).

### Common flags

| Flag | Meaning |
| ---- | ------- |
| `--mode {adjoint, adjoint_sto, rhc, rhc_sto, spsa, jfb}` | pick the solver |
| `--seed INT` | RNG seed for the policy network and SPSA probes |
| `--max_total_iterations INT` | total optimization iterations per case |
| `--inner_iterations INT` | optimizer steps per MPC horizon |
| `--learning_rate FLOAT` | Adam learning rate |
| `--patience INT` / `--min_delta_rel FLOAT` / `--loss_threshold FLOAT` | early stopping |
| `--log_dir DIR` / `--run_name NAME` | per-iteration trace CSV output |
| `--no_show` | suppress the matplotlib animation |

STO-only flags (used when `--mode` is `adjoint_sto` or `rhc_sto`):

| Flag | Meaning |
| ---- | ------- |
| `--rho_max FLOAT` | probe-residual gate threshold |
| `--kappa_max FLOAT` / `--kappa_warn FLOAT` | conditioning gate thresholds |
| `--kappa_check_period INT` | how often to recompute σ_max in the steady regime |
| `--cooldown INT` | consecutive safe recomputations needed to leave warning mode |
| `--n_probes INT` / `--n_power_iter INT` | probe count and power-iteration depth |
| `--max_reuse INT` | hard cap on cache reuses before forced reinit |
| `--sto_seed INT` | RNG seed for STO probes |

SPSA-only flags (used when `--mode spsa`):

| Flag | Meaning |
| ---- | ------- |
| `--spsa_lr FLOAT` / `--spsa_c FLOAT` / `--spsa_m INT` | learning rate, perturbation size, number of pairs |
| `--spsa_grad_clip FLOAT` | gradient norm clip |
| `--spsa_blocking` / `--spsa_block_tol FLOAT` | reject updates that increase loss beyond this fraction |

## Output

Each script writes one CSV per case to `--log_dir`:

```
<run_name>_case<idx>_<initial>_to_<target>_trace.csv
```

with per-iteration columns including `loss`, `best_loss`, `grad_norm`,
`cumulative_time`, the timing breakdown of the backward pass, and STO
diagnostics (`sto_hit_rate`, `sto_queries`, `sto_cache_hits`,
`sto_invalid_reason_counts`, `sto_last_rho`, `sto_last_kappa`, ...).

## Notes

- The simulator runs single-threaded by default (`configure_threads(1)` in
  `common.py`) for determinism. Each script also calls `set_seed(seed)` with
  cuDNN deterministic mode enabled. Same seed and same machine should give
  the same trace.
- `--mode rhc` plus seed 2 reaches `best_loss ≈ 4e-8` on Task 3 case 1
  (`C_initial → target_C_flipped`) at iter ~500 — most of the drop happens
  in the last ~100 iterations.
- The compiled extension is platform-specific. Rebuild on the target machine
  rather than copying `nn_der/*.so`.

## Troubleshooting

**CMake cannot find MKL** — set `MKLROOT` and `CMAKE_PREFIX_PATH` (see Build).

**`cannot find -llapack`** — `conda install -c conda-forge liblapack`.

**`ImportError: cannot import nn_der`** — rebuild and reinstall:

```bash
rm -rf build && mkdir build && cd build && cmake .. && make -j && cd ..
pip install -e .
```

**Animation does not show** — install a backend (`pip install pyqt5`) or run
with `--no_show` and inspect the trace CSV instead.

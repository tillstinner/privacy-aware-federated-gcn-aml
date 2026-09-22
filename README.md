# Privacy-Aware Federated GCNs for AML

Research code for studying federated graph convolutional networks on synthetic
anti-money-laundering transaction graphs. The repository compares centralized
GCNs, local cut-edge FedAvg, and one- or two-hop pretraining feature aggregation
across bank-partitioned AMLSim graphs.

The implementation was developed for the bachelor's thesis *Privacy-Preserving
Federated Graph Learning for AML*.

## Scope

The main implementation is under `src/federated_gcn_aml/federated/` and supports:

- `cut_edges`: local message passing followed by FedAvg;
- one-hop (`h1`) pretraining aggregation for rows owned by each client;
- two-hop (`h2`) pretraining aggregation for owned and required placeholder rows;
- plain or TenSEAL/CKKS transport for pretraining feature contributions;
- diagnostic Gaussian contribution perturbation and calibrated contribution-row
  DP mechanisms;
- plain FedAvg or pairwise-masked FedAvg model-update aggregation;
- `oracle` and `privacy_clean` AMLSim feature boundaries.

This is a single-process research simulation, not a production federated
deployment. The server is allowed to know graph topology, routing information,
and recipient row plans. CKKS protects plaintext contribution values during
pretraining aggregation, but clients decrypt their routed aggregate rows.
Masked FedAvg hides individual plaintext model updates from the simulated
server aggregation role under non-colluding, no-dropout assumptions; the
aggregate update remains visible. It does not implement dropout recovery,
secret sharing, malicious-server protection, or end-to-end privacy.

## Repository Layout

```text
datasets/                           Curated AMLSim configurations and patches
envs/                               Reproducible CPU and CUDA environments
hpc/examples/                       Machine-independent Slurm examples
src/federated_gcn_aml/data/         Graph construction and feature boundaries
src/federated_gcn_aml/experiments/  Experiment entry points
src/federated_gcn_aml/federated/    Federated and privacy protocols
src/federated_gcn_aml/models/       GCN model definitions
src/federated_gcn_aml/training/     Training, splitting, and metrics
tests/                              Unit and protocol-boundary tests
```

Generated datasets and experiment outputs are intentionally excluded.

## Environment

Python 3.11 is the tested version. For a CPU environment:

```bash
conda env create -f envs/fedgcn-aml-cpu.yml
conda activate fedgcn-aml-cpu
python -m pip install --no-deps -e ".[he,analysis,test]"
```

For CUDA 12.1, use `envs/fedgcn-aml-gpu.yml`. TenSEAL is installed in both
environments because the HE protocol is CPU-bound even when model training uses
a GPU.

Run the tests from the repository root:

```bash
python -m pytest tests -q
```

## Data

Experiments expect AMLSim CSV output containing `accounts.csv`,
`transactions.csv`, and `sar_accounts.csv`. Curated balanced, mildly non-IID,
strongly non-IID, and smoke configurations are included under
`datasets/`; generated outputs are not.

See `datasets/README.md` for the pinned AMLSim base revision, generator
patches, and generation notes. AMLSim and its external runtime dependencies are
not redistributed by this repository.

## Quick Start

Set `DATA_ROOT` to an AMLSim output directory. A short centralized run is:

```bash
aml-gcn \
  --data-root "$DATA_ROOT" \
  --output-dir runs/centralized-smoke \
  --representation-version pass2 \
  --feature-boundary privacy_clean \
  --edge-weight-mode unit \
  --epochs 10 --hidden-channels 16 --num-layers 3 --device cpu
```

A short two-hop federated run with plaintext pretraining is:

```bash
aml-federated-gcn \
  --data-root "$DATA_ROOT" \
  --output-dir runs/h2-plain-smoke \
  --feature-boundary privacy_clean \
  --federated-mode fedgcn_feature_pretrain --num-hops 2 \
  --feature-pretrain-transport plain \
  --feature-pretrain-payload-protection none \
  --model-update-aggregation plain_fedavg \
  --rounds 2 --local-epochs 1 \
  --hidden-channels 16 --num-layers 3 --device cpu
```

Change `--feature-pretrain-transport` to `he` to use CKKS. For masked model
updates, use `--model-update-aggregation masked_fedavg`. Run the entry point
with `--help` for the diagnostic perturbation and formal contribution-row DP modes.

The payload mechanism names intentionally separate empirical diagnostics from
formal privacy claims:

```text
central_diagnostic_perturbation  Gaussian aggregate perturbation diagnostic
local_diagnostic_perturbation    Gaussian source-contribution perturbation diagnostic
central_row_dp                   Calibrated central contribution-row DP
local_row_dp                     Calibrated local contribution-row DP
```

Diagnostic modes use `--payload-clip-norm`,
`--diagnostic-noise-multiplier`, and optionally `--diagnostic-seed`; they do
not imply an epsilon/delta guarantee. Formal row-DP modes instead use
`--payload-clip-norm`, `--payload-dp-epsilon`, `--payload-dp-delta`, and
optionally `--payload-dp-seed` and `--payload-dp-calibration`. Neither family
provides account-, transaction-, topology-, client-, model-update-, or
end-to-end DP.

## Outputs

Experiment directories contain model state, run metadata, and metrics. The
federated runner additionally writes per-round metrics and client diagnostics.
AUPR is the primary reporting and checkpoint-selection metric; AUROC and
threshold-dependent metrics are reporting-only.

Runtime measurements are local runner wall time and exclude scheduler queue
delay. Communication fields are estimates or serialized protocol sizes for the
implemented simulation, not production SecAgg or network measurements.

## References and Attribution

The federated graph-learning method and pretraining workflow build on FedGCN
and FedGraph. This repository implements those foundations for bank-partitioned
AMLSim graphs and adapts the implementation and protocol boundaries for routed
row-block communication and explicit privacy mechanisms. Neither upstream
project is a runtime dependency. Dataset generation is based on IBM AMLSim.
See `NOTICE` for source links, licenses, and the distinction between these
methodological foundations and included modified AMLSim source.

## Citation

Software citation metadata is provided in `CITATION.cff`.

## License

Unless otherwise noted in `NOTICE`, this repository is licensed under the
Apache License 2.0. The AMLSim-derived generator overlay retains the upstream
Apache-2.0 terms.

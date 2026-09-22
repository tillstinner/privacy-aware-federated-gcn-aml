# Curated AMLSim Configurations

This directory contains the AMLSim configurations used for the balanced,
mildly non-IID, strongly non-IID, and smoke graphs. Generated data is excluded.

The generator work was based on IBM AMLSim at revision:

```text
7338a4b (Merge pull request #66 from nelsonjd/iss-65)
```

Repository: https://github.com/IBM/AMLSim

## Generator Changes

Two overlays reproduce the thesis-specific generation behavior:

```text
patches/amlsim_normal_models.patch
patches/transaction_graph_generator.py
```

The first patch adjusts AMLSim normal-model group behavior. The second replaces
`scripts/transaction_graph_generator.py` with the graph generator used for the
curated configurations, including bank-aware edge construction and coverage
calibration. Both are distributed under AMLSim's Apache-2.0 terms.

One way to prepare a generator checkout is:

```bash
git clone https://github.com/IBM/AMLSim.git
cd AMLSim
git checkout 7338a4b
git apply /path/to/privacy-aware-federated-gcn-aml/datasets/patches/amlsim_normal_models.patch
cp /path/to/privacy-aware-federated-gcn-aml/datasets/patches/transaction_graph_generator.py scripts/
cp -R /path/to/privacy-aware-federated-gcn-aml/datasets/paramFiles/* paramFiles/
cp /path/to/privacy-aware-federated-gcn-aml/datasets/conf_*.json .
```

Then follow AMLSim's own build and execution instructions. AMLSim's MASON JAR
and other external runtime components are not redistributed here.

## Included Configurations

```text
conf_1m_3b_diag_i66_x33_combined_v07.json
conf_1m_3b_diag_i66_x33_combined_smoke_v07.json
conf_1m_3b_diag_i66_x33_noniid_mild_v01.json
conf_1m_3b_diag_i66_x33_noniid_mild_smoke_v01.json
conf_1m_3b_diag_i66_x33_noniid_strong_v01_recovered_v01.json
conf_1m_3b_diag_i66_x33_noniid_strong_smoke_v01.json
```

The recovered strong configuration is the retained strong non-IID dataset. It
uses `paramFiles/1m_3b_diag_i66_x33_noniid_strong_v01`.

Place generated CSV files under `datasets/outputs/<dataset-name>/` or
pass their directory explicitly with `--data-root`.

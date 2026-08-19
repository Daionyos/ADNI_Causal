# ADNI Synthetic Data Causal Benchmark

This repository contains the code used to reproduce the cross-sectional ADNI synthetic-data benchmark. The experiments compare conventional synthetic-data quality with preservation of inferred causal structure and simulated intervention behaviour.

Raw ADNI data are not included. To run the pipeline, place `ADNIMERGE.csv` in `data/`.

The `src/onto_cgans/` package is adapted from the Onto-CGAN implementation by Sun Chang and is redistributed under the MIT license included in `LICENSE.txt`. The ADNI preprocessing, benchmarking, holdout utility, and causal-evaluation scripts were added for this project.

## Data Preprocessing

The pipeline uses baseline ADNI visits only (`VISCODE == "bl"`), with one row per participant when `RID` is available. Diagnosis labels are mapped to three classes: CN, MCI, and AD. Numeric fields are cleaned before imputation, including ADNI values such as `>1700`. Numeric missing values are median-imputed after dropping columns with more than 60% missingness.

Ontology conditioning uses:

| Diagnosis | Ontology/control value |
|---|---|
| CN | `ADNI_CN_CONTROL` |
| MCI | `http://purl.obolibrary.org/obo/HP_0100543` |
| AD | `http://www.orpha.net/ORDO/Orphanet_26929` |

CN is represented by a local control token rather than by an unrelated disease ontology term.

## Feature Sets

Main numeric features:

`AGE`, `PTEDUCAT`, `APOE4`, `ABETA`, `TAU`, `PTAU`, `AV45`, `FDG`, `Hippocampus`, `Entorhinal`, `Fusiform`, `MidTemp`, `ICV`, `CDRSB`, `MMSE`, `ADAS13`, `RAVLT_immediate`, `FAQ`, `MOCA`.

Expanded feature set adds:

`Ventricles`, `WholeBrain`, `ADAS11`, `ADASQ4`, `RAVLT_learning`, `RAVLT_forgetting`, `RAVLT_perc_forgetting`, `LDELTOTAL`, `TRABSCOR`, `mPACCdigit`, `mPACCtrailsB`.

## Generators and Seeds

Experiments used seeds `11`, `22`, and `33`.

Generators:

- `ontocgan_decoder`
- `ctgan`
- `tvae`
- `gaussian_copula`
- `independent_marginals`
- `row_bootstrap`

Onto-CGAN was run with decoder conditioning, 300 epochs, batch size 500, and synthetic sample count matching the available real training rows. SDV CTGAN and TVAE were run for 300 epochs.

## Utility Evaluation

Predictive utility uses train-on-synthetic, test-on-real evaluation. The real table is split once into generator-training and held-out test partitions using a 30% test split, stratified by diagnosis, with split seed `123`.

A random forest classifier is trained on synthetic data and evaluated on the same held-out real patients. The real-train/real-test classifier provides the denominator for the relative utility score.

Classifier settings:

- 300 trees
- `balanced_subsample` class weights
- classifier seed `321`
- macro-F1 and accuracy reported on held-out real patients

## Causal Evaluation

Causal structure is inferred using PC-Stable with Fisher-z partial-correlation tests. The inferred graph from each synthetic table is compared against the graph inferred from the corresponding real table. These are comparisons of inferred causal structure under the chosen pipeline, not claims about the true biological causal graph.

Main causal-discovery settings:

- PC-Stable `alpha = 0.01`
- maximum conditioning set size `3`
- bootstrap stability threshold `0.70`
- SCM simulation size `3000` samples per intervention run
- NOTEARS-style DAG learning with domain bounds
- NOTEARS `lambda = 0.01`
- NOTEARS threshold `0.10`
- NOTEARS maximum iterations `10`

Tier ordering used for DAG completion and domain-bounded NOTEARS:

1. `gender_Female`, `AGE`, `PTEDUCAT`, `APOE4`
2. `ABETA`, `TAU`, `PTAU`, `AV45`, `FDG`
3. `ICV`, `Ventricles`, `WholeBrain`, `Hippocampus`, `Entorhinal`, `Fusiform`, `MidTemp`
4. `CDRSB`, `MMSE`, `ADAS11`, `ADAS13`, `ADASQ4`, `RAVLT_immediate`, `RAVLT_learning`, `RAVLT_forgetting`, `RAVLT_perc_forgetting`, `LDELTOTAL`, `TRABSCOR`, `FAQ`, `MOCA`, `mPACCdigit`, `mPACCtrailsB`

The 22-edge domain template is used only as a post-hoc structural sanity check. These edges are not forced during PC-Stable.

| Source | Target |
|---|---|
| `APOE4` | `ABETA` |
| `APOE4` | `TAU` |
| `APOE4` | `PTAU` |
| `APOE4` | `AV45` |
| `AGE` | `Hippocampus` |
| `AGE` | `Entorhinal` |
| `AGE` | `MMSE` |
| `PTEDUCAT` | `MMSE` |
| `ABETA` | `Hippocampus` |
| `TAU` | `Hippocampus` |
| `PTAU` | `Hippocampus` |
| `AV45` | `Hippocampus` |
| `FDG` | `MMSE` |
| `FDG` | `ADAS13` |
| `Hippocampus` | `MMSE` |
| `Hippocampus` | `ADAS13` |
| `Hippocampus` | `RAVLT_immediate` |
| `Entorhinal` | `MMSE` |
| `Entorhinal` | `ADAS13` |
| `Entorhinal` | `FAQ` |
| `Fusiform` | `MMSE` |
| `MidTemp` | `MMSE` |

Intervention variables:

`ABETA`, `TAU`, `PTAU`, `AV45`, `FDG`, `Hippocampus`, `Entorhinal`.

Outcome variables:

`CDRSB`, `MMSE`, `ADAS13`, `RAVLT_immediate`, `FAQ`, `MOCA`.

## Scripts

- `scripts/adni_ontocgan_plus.py`: preprocesses `ADNIMERGE.csv` and trains/samples Onto-CGAN.
- `scripts/adni_generator_benchmark.py`: runs row bootstrap, independent marginals, Gaussian copula, CTGAN, and TVAE baselines and computes conventional fidelity/utility metrics.
- `scripts/adni_holdout_utility_benchmark.py`: repeats predictive utility with generators fitted only on the real training partition and evaluated on a fixed held-out real test set.
- `scripts/causal_adni_experiments.py`: runs PC-Stable, bootstrap-stable graph comparison, NOTEARS-style graph comparison, domain-template scoring, Markov blanket comparison, and SCM intervention benchmarking.

## Example Commands

Prepare main feature-set data and Onto-CGAN output:

```powershell
.\.conda\python.exe scripts\adni_ontocgan_plus.py --feature-set main --epochs 300 --batch-size 500 --n-samples 2419 --mode decoder --seed 11 --overwrite --out-dir output\resit_aligned_experiments\main\real
```

Prepare expanded feature-set data and Onto-CGAN output:

```powershell
.\.conda\python.exe scripts\adni_ontocgan_plus.py --feature-set expanded --epochs 300 --batch-size 500 --n-samples 2419 --mode decoder --seed 11 --overwrite --out-dir output\resit_aligned_experiments\expanded\real
```

Run generator benchmark for a prepared feature set:

```powershell
.\.conda\python.exe scripts\adni_generator_benchmark.py --real output\resit_aligned_experiments\main\real\adni_cgan_train.csv --ontocgan-synthetic output\resit_aligned_experiments\main\real\synthetic_adni_decoder.csv --methods row_bootstrap independent_marginals gaussian_copula ctgan tvae --seeds 11 22 33 --ctgan-epochs 300 --tvae-epochs 300 --out-dir output\resit_aligned_experiments\main\generators
```

Run holdout utility benchmark:

```powershell
.\.conda\python.exe scripts\adni_holdout_utility_benchmark.py --feature-sets main expanded --methods row_bootstrap independent_marginals gaussian_copula ctgan tvae --seeds 11 22 33 --test-size 0.30 --split-seed 123 --classifier-seed 321 --ctgan-epochs 300 --tvae-epochs 300 --include-ontocgan --out-dir output\resit_aligned_experiments\holdout_utility
```

Run causal benchmark for one synthetic table:

```powershell
.\.conda\python.exe scripts\causal_adni_experiments.py --real output\resit_aligned_experiments\main\real\adni_cgan_train.csv --synthetic output\resit_aligned_experiments\main\generators\gaussian_copula\seed_11\synthetic.csv --alpha 0.01 --max-cond-set 3 --n-sim 3000 --bootstrap-runs 100 --bootstrap-threshold 0.70 --run-notears --notears-domain-bounds --notears-lambda 0.01 --notears-threshold 0.10 --notears-max-iter 10 --out-dir output\resit_aligned_experiments\main\causal\gaussian_copula\seed_11
```

## Tested Software Versions

The final runs were performed with:

- Python 3.11.15
- pandas 2.3.3
- numpy 2.4.4
- scikit-learn 1.8.0
- scipy 1.17.1
- torch 2.11.0+cu128
- matplotlib 3.10.9
- networkx 3.6.1
- SDV 1.37.3

Additional dependencies are listed in `requirements.txt`.

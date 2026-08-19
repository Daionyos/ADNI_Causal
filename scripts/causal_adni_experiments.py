"""Causal comparison experiments for real vs synthetic ADNI data.

Pipeline:
1. Learn PC-Stable graphs on real and synthetic ADNI tables.
2. Compare graph structure using skeleton SHD and DAG-style SHD.
3. Complete each graph into a domain-prior DAG.
4. Fit linear-Gaussian SCMs and run matched do-interventions.

The implementation is self-contained. It uses Fisher-z partial-correlation tests, so the
graph discovery step should be interpreted as a Gaussian PC-Stable baseline.

Main feature-set run:
    .\\.conda\\python.exe scripts\\causal_adni_experiments.py --real output\\adni_ontocgan_plus_corrected\\adni_cgan_train.csv --synthetic output\\adni_ontocgan_plus_corrected\\synthetic_adni_decoder.csv --alpha 0.01 --max-cond-set 3 --n-sim 3000 --bootstrap-runs 100 --bootstrap-threshold 0.70 --run-notears --notears-domain-bounds --notears-lambda 0.01 --notears-threshold 0.10 --notears-max-iter 10 --out-dir output\\causal_adni_corrected_full5_boot100

Expanded feature-set run:
    .\\.conda\\python.exe scripts\\causal_adni_experiments.py --real output\\adni_ontocgan_plus_corrected\\expanded\\adni_cgan_train.csv --synthetic output\\adni_ontocgan_plus_corrected\\expanded\\synthetic_adni_decoder.csv --alpha 0.01 --max-cond-set 3 --n-sim 3000 --bootstrap-runs 100 --bootstrap-threshold 0.70 --run-notears --notears-domain-bounds --notears-lambda 0.01 --notears-threshold 0.10 --notears-max-iter 10 --out-dir output\\causal_adni_corrected_expanded_full5_boot100
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats
from sklearn.linear_model import LinearRegression


DEFAULT_REAL = ROOT / "output" / "adni_ontocgan_plus" / "adni_cgan_train.csv"
DEFAULT_SYNTH = ROOT / "output" / "adni_ontocgan_plus" / "synthetic_adni_decoder.csv"
DEFAULT_OUT_DIR = ROOT / "output" / "causal_adni"

META_COLUMNS = {"IRI", "label"}

DEFAULT_OUTCOMES = [
    "CDRSB",
    "MMSE",
    "ADAS13",
    "RAVLT_immediate",
    "FAQ",
    "MOCA",
]

DEFAULT_INTERVENTIONS = [
    "ABETA",
    "TAU",
    "PTAU",
    "AV45",
    "FDG",
    "Hippocampus",
    "Entorhinal",
]

DOMAIN_TIERS = [
    # Broad clinical ordering used when a directed acyclic graph is needed:
    # demographics/genetics -> biomarkers/PET -> MRI neurodegeneration -> cognition.
    # They are used after PC-Stable to turn partially directed graphs into DAGs
    # that can be fitted as SCMs for the intervention benchmark.
    ["gender_Female", "AGE", "PTEDUCAT", "APOE4"],
    ["ABETA", "TAU", "PTAU", "AV45", "FDG"],
    ["ICV", "Ventricles", "WholeBrain", "Hippocampus", "Entorhinal", "Fusiform", "MidTemp"],
    [
        "CDRSB",
        "MMSE",
        "ADAS11",
        "ADAS13",
        "ADASQ4",
        "RAVLT_immediate",
        "RAVLT_learning",
        "RAVLT_forgetting",
        "RAVLT_perc_forgetting",
        "LDELTOTAL",
        "TRABSCOR",
        "FAQ",
        "MOCA",
        "mPACCdigit",
        "mPACCtrailsB",
    ],
]

EXPECTED_AD_EDGES = [
    # Small Alzheimer's disease template used as a domain sanity check. These
    # edges are not forced during PC-Stable; they are scored after learning.
    ("APOE4", "ABETA"),
    ("APOE4", "TAU"),
    ("APOE4", "PTAU"),
    ("APOE4", "AV45"),
    ("AGE", "Hippocampus"),
    ("AGE", "Entorhinal"),
    ("AGE", "MMSE"),
    ("PTEDUCAT", "MMSE"),
    ("ABETA", "Hippocampus"),
    ("TAU", "Hippocampus"),
    ("PTAU", "Hippocampus"),
    ("AV45", "Hippocampus"),
    ("FDG", "MMSE"),
    ("FDG", "ADAS13"),
    ("Hippocampus", "MMSE"),
    ("Hippocampus", "ADAS13"),
    ("Hippocampus", "RAVLT_immediate"),
    ("Entorhinal", "MMSE"),
    ("Entorhinal", "ADAS13"),
    ("Entorhinal", "FAQ"),
    ("Fusiform", "MMSE"),
    ("MidTemp", "MMSE"),
]


@dataclass
class PCStableResult:
    """Container for the partially directed graph returned by PC-Stable."""
    columns: list[str]
    undirected: np.ndarray
    directed: np.ndarray
    sep_sets: dict[tuple[int, int], set[int]]
    pvalue_matrix: np.ndarray
    max_cond_set: int
    alpha: float


@dataclass
class LinearSCM:
    """Fitted linear-Gaussian SCM used for matched intervention simulations."""
    name: str
    columns: list[str]
    dag: np.ndarray
    data: pd.DataFrame
    parents: dict[str, list[str]]
    models: dict[str, LinearRegression | None]
    residuals: dict[str, np.ndarray]
    root_values: dict[str, np.ndarray]
    order: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PC-Stable, SHD comparison, and intervention benchmarks on ADNI real/synthetic data.",
    )
    parser.add_argument("--real", type=Path, default=DEFAULT_REAL)
    parser.add_argument("--synthetic", type=Path, default=DEFAULT_SYNTH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--max-cond-set", type=int, default=3)
    parser.add_argument(
        "--pc-transform",
        choices=["rank-gaussian", "zscore", "none"],
        default="rank-gaussian",
        help="Transform used only for PC-Stable CI tests.",
    )
    parser.add_argument("--intervention-vars", nargs="*", default=DEFAULT_INTERVENTIONS)
    parser.add_argument("--outcomes", nargs="*", default=DEFAULT_OUTCOMES)
    parser.add_argument("--n-sim", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-label", action="store_true", help="Encode ADNI label as an ordinal variable.")
    parser.add_argument("--no-gender", action="store_true")
    parser.add_argument("--bootstrap-runs", type=int, default=0)
    parser.add_argument("--bootstrap-sample-frac", type=float, default=1.0)
    parser.add_argument("--bootstrap-threshold", type=float, default=0.70)
    parser.add_argument("--run-notears", action="store_true")
    parser.add_argument("--notears-lambda", type=float, default=0.03)
    parser.add_argument("--notears-threshold", type=float, default=0.20)
    parser.add_argument("--notears-max-iter", type=int, default=20)
    parser.add_argument(
        "--notears-domain-bounds",
        action="store_true",
        help="Constrain NOTEARS candidate directions to the AD domain-tier order.",
    )
    return parser.parse_args()


def load_raw_tables(real_path: Path, synth_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not real_path.exists():
        raise FileNotFoundError(f"Real ADNI table not found: {real_path}")
    if not synth_path.exists():
        raise FileNotFoundError(f"Synthetic ADNI table not found: {synth_path}")
    return pd.read_csv(real_path), pd.read_csv(synth_path)


def encode_table(df: pd.DataFrame, include_label: bool, include_gender: bool) -> pd.DataFrame:
    """Convert a CGAN CSV into the numeric matrix used for causal discovery."""
    out = pd.DataFrame(index=df.index)

    for col in df.columns:
        if col in META_COLUMNS or col == "gender":
            continue
        out[col] = pd.to_numeric(df[col], errors="coerce")

    if include_gender and "gender" in df.columns:
        gender = df["gender"].astype(str).str.strip().str.lower()
        out.insert(0, "gender_Female", (gender == "female").astype(float))

    if include_label and "label" in df.columns:
        #Ordinal diagnosis severity proxy. Disabled by default because PC's
        #Gaussian CI test is not ideal for categorical diagnosis labels.
        mapping = {"CN": 0.0, "MCI": 1.0, "AD": 2.0}
        out["diagnosis_stage"] = df["label"].map(mapping).astype(float)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def align_real_synthetic(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    include_label: bool,
    include_gender: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only variables that exist and vary in both real and synthetic data."""
    real_enc = encode_table(real, include_label, include_gender)
    synth_enc = encode_table(synth, include_label, include_gender)
    common = [col for col in real_enc.columns if col in synth_enc.columns]

    real_enc = real_enc[common].copy()
    synth_enc = synth_enc[common].copy()

    #Impute with each dataset's median, then drop non-informative columns in either table.
    for table in [real_enc, synth_enc]:
        for col in table.columns:
            median = table[col].median()
            table[col] = table[col].fillna(0.0 if pd.isna(median) else median)

    keep = [
        col
        for col in common
        if real_enc[col].std(ddof=0) > 1e-10 and synth_enc[col].std(ddof=0) > 1e-10
    ]
    return real_enc[keep].astype(float), synth_enc[keep].astype(float)


def normal_score_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Rank-Gaussian transform used before Fisher-z partial correlations."""
    n = len(df)
    transformed = pd.DataFrame(index=df.index)
    for col in df.columns:
        ranks = stats.rankdata(df[col].to_numpy(), method="average")
        probs = (ranks - 0.5) / n
        transformed[col] = stats.norm.ppf(probs)
    return transformed


def zscore(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / df.std(ddof=0).replace(0.0, 1.0)


def transform_for_pc(df: pd.DataFrame, transform: str) -> np.ndarray:
    """Apply the selected preprocessing before PC-Stable tests."""
    if transform == "rank-gaussian":
        work = normal_score_transform(df)
    elif transform == "zscore":
        work = zscore(df)
    else:
        work = df.copy()
    work = zscore(work)
    return work.to_numpy(dtype=float)


def partial_corr_pvalue(data: np.ndarray, x: int, y: int, cond: tuple[int, ...]) -> tuple[float, float]:
    """Fisher-z test for zero partial correlation between two variables."""
    n = data.shape[0]
    if len(cond) == 0:
        r = np.corrcoef(data[:, x], data[:, y])[0, 1]
    else:
        cols = [x, y, *cond]
        corr = np.corrcoef(data[:, cols], rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        corr = corr + np.eye(corr.shape[0]) * 1e-10
        precision = np.linalg.pinv(corr)
        denom = math.sqrt(max(precision[0, 0] * precision[1, 1], 1e-20))
        r = -precision[0, 1] / denom

    r = float(np.clip(r, -0.999999, 0.999999))
    dof = n - len(cond) - 3
    if dof <= 0:
        return r, 1.0

    z_value = abs(0.5 * math.log((1.0 + r) / (1.0 - r)) * math.sqrt(dof))
    p_value = 2.0 * (1.0 - stats.norm.cdf(z_value))
    return r, float(p_value)


def pc_stable(df: pd.DataFrame, alpha: float, max_cond_set: int, transform: str) -> PCStableResult:
    """Learn a PC-Stable-style DAG from a numeric data matrix.

    This implementation follows the PC-Stable idea of freezing the adjacency
    set at each conditioning depth before removing edges.  The maximum
    conditioning size is capped because exhaustive high-order conditioning is
    expensive and unstable for this tabular problem.
    """
    columns = list(df.columns)
    data = transform_for_pc(df, transform)
    p = len(columns)

    skeleton = np.ones((p, p), dtype=bool)
    np.fill_diagonal(skeleton, False)
    sep_sets: dict[tuple[int, int], set[int]] = {}
    pvalue_matrix = np.zeros((p, p), dtype=float)

    for cond_size in range(max_cond_set + 1):
        adjacency_snapshot = skeleton.copy()
        removals: list[tuple[int, int, set[int], float]] = []

        for i in range(p):
            for j in range(i + 1, p):
                if not adjacency_snapshot[i, j]:
                    continue
                neighbors = [k for k in np.where(adjacency_snapshot[i])[0] if k != j]
                if len(neighbors) < cond_size:
                    continue

                for cond in itertools.combinations(neighbors, cond_size):
                    _, p_value = partial_corr_pvalue(data, i, j, cond)
                    pvalue_matrix[i, j] = pvalue_matrix[j, i] = max(pvalue_matrix[i, j], p_value)
                    if p_value > alpha:
                        removals.append((i, j, set(cond), p_value))
                        break

        for i, j, sep, _ in removals:
            skeleton[i, j] = skeleton[j, i] = False
            sep_sets[(i, j)] = sep
            sep_sets[(j, i)] = sep

        max_degree = int(skeleton.sum(axis=0).max()) if p else 0
        if max_degree < cond_size + 1:
            break

    undirected, directed = orient_cpdag(skeleton, sep_sets)
    return PCStableResult(columns, undirected, directed, sep_sets, pvalue_matrix, max_cond_set, alpha)


def is_adjacent(undirected: np.ndarray, directed: np.ndarray, i: int, j: int) -> bool:
    return bool(undirected[i, j] or directed[i, j] or directed[j, i])


def orient_edge(undirected: np.ndarray, directed: np.ndarray, src: int, dst: int) -> None:
    if directed[dst, src]:
        return
    undirected[src, dst] = undirected[dst, src] = False
    directed[src, dst] = True


def orient_cpdag(skeleton: np.ndarray, sep_sets: dict[tuple[int, int], set[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Orient the skeleton using unshielded colliders and a Meek R1 pass."""
    p = skeleton.shape[0]
    undirected = skeleton.copy()
    directed = np.zeros((p, p), dtype=bool)

    # Orient unshielded colliders i -> k <- j.
    for k in range(p):
        neighbors = np.where(skeleton[k])[0] 
        for i, j in itertools.combinations(neighbors, 2):
            if skeleton[i, j]:
                continue
            if k not in sep_sets.get((i, j), set()):
                orient_edge(undirected, directed, i, k)
                orient_edge(undirected, directed, j, k)

    #Meek R1: i -> j - k and i not adjacent k implies j -> k.
    changed = True
    while changed:
        changed = False
        for i in range(p):
            for j in range(p):
                if not directed[i, j]:
                    continue
                for k in range(p):
                    if undirected[j, k] and not is_adjacent(undirected, directed, i, k):
                        orient_edge(undirected, directed, j, k)
                        changed = True

    return undirected, directed


def edge_type(undirected: np.ndarray, directed: np.ndarray, i: int, j: int) -> str:
    if directed[i, j]:
        return "i_to_j"
    if directed[j, i]:
        return "j_to_i"
    if undirected[i, j]:
        return "undirected"
    return "none"


def compare_graphs(real: PCStableResult, synth: PCStableResult) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compare real and synthetic PC-Stable graphs edge by edge."""
    rows = []
    skeleton_shd = 0
    cpdag_shd = 0
    real_edges = 0
    synth_edges = 0
    common_edges = 0
    orientation_diffs = 0

    p = len(real.columns)
    # Keep PC-Stable's adjacencies, but orient unresolved edges according to the
    # tier prior so the later SCM never contains cognition -> genotype style arcs.
    for i, j in itertools.combinations(range(p), 2):
        real_type = edge_type(real.undirected, real.directed, i, j)
        synth_type = edge_type(synth.undirected, synth.directed, i, j)
        real_has = real_type != "none"
        synth_has = synth_type != "none"

        if real_has:
            real_edges += 1
        if synth_has:
            synth_edges += 1
        if real_has and synth_has:
            common_edges += 1
        if real_has != synth_has:
            skeleton_shd += 1
            cpdag_shd += 1
            status = "missing_in_synthetic" if real_has else "extra_in_synthetic"
        elif real_type != synth_type:
            cpdag_shd += 1
            orientation_diffs += 1
            status = "orientation_or_mark_diff"
        else:
            status = "same_edge" if real_has else "same_absent"

        if status != "same_absent":
            rows.append(
                {
                    "source": real.columns[i],
                    "target": real.columns[j],
                    "real_edge_type": real_type,
                    "synthetic_edge_type": synth_type,
                    "status": status,
                }
            )

    precision = common_edges / synth_edges if synth_edges else float("nan")
    recall = common_edges / real_edges if real_edges else float("nan")
    jaccard = common_edges / (real_edges + synth_edges - common_edges) if (real_edges + synth_edges - common_edges) else 1.0
    metrics = {
        "real_edges": real_edges,
        "synthetic_edges": synth_edges,
        "common_skeleton_edges": common_edges,
        "skeleton_shd": skeleton_shd,
        "cpdag_shd": cpdag_shd,
        "orientation_or_mark_differences": orientation_diffs,
        "skeleton_precision_real_reference": precision,
        "skeleton_recall_real_reference": recall,
        "skeleton_jaccard": jaccard,
    }
    return pd.DataFrame(rows), metrics


def skeleton_from_pc(result: PCStableResult) -> np.ndarray:
    return result.undirected | result.directed | result.directed.T


def compare_skeleton_matrices(real_skeleton: np.ndarray, synth_skeleton: np.ndarray) -> dict[str, float]:
    """Compute edge-count, SHD, precision, recall, and Jaccard metrics."""
    p = real_skeleton.shape[0]
    real_edges = synth_edges = common_edges = shd = 0
    for i, j in itertools.combinations(range(p), 2):
        real_has = bool(real_skeleton[i, j])
        synth_has = bool(synth_skeleton[i, j])
        real_edges += int(real_has)
        synth_edges += int(synth_has)
        common_edges += int(real_has and synth_has)
        shd += int(real_has != synth_has)

    precision = common_edges / synth_edges if synth_edges else float("nan")
    recall = common_edges / real_edges if real_edges else float("nan")
    denom = real_edges + synth_edges - common_edges
    return {
        "real_edges": real_edges,
        "synthetic_edges": synth_edges,
        "common_skeleton_edges": common_edges,
        "skeleton_shd": shd,
        "skeleton_precision_real_reference": precision,
        "skeleton_recall_real_reference": recall,
        "skeleton_jaccard": common_edges / denom if denom else 1.0,
    }


def tier_rank(columns: list[str]) -> dict[str, int]:
    """Assign each variable a deterministic order for DAG completion."""
    rank = {}
    offset = 0
    for tier in DOMAIN_TIERS:
        for node in tier:
            if node in columns:
                rank[node] = offset
                offset += 1
    for node in columns:
        if node not in rank:
            rank[node] = offset
            offset += 1
    return rank


def complete_to_domain_dag(result: PCStableResult) -> np.ndarray:
    """Turn a partially directed PC graph into a DAG using domain tiers.
    PC-Stable often leaves undirected edges.  The SCM benchmark needs a DAG, so
    every remaining adjacency is oriented from earlier to later tier.  Within a
    tier, the fixed column order is used only to make the completion acyclic and
    reproducible.
    """
    columns = result.columns
    rank = tier_rank(columns)
    p = len(columns)
    dag = np.zeros((p, p), dtype=bool)

    for i, j in itertools.combinations(range(p), 2):
        if not is_adjacent(result.undirected, result.directed, i, j):
            continue
        if rank[columns[i]] <= rank[columns[j]]:
            dag[i, j] = True
        else:
            dag[j, i] = True

    graph = nx.DiGraph()
    graph.add_nodes_from(range(p))
    graph.add_edges_from((i, j) for i in range(p) for j in range(p) if dag[i, j])
    if not nx.is_directed_acyclic_graph(graph):
        raise RuntimeError("Domain-tier DAG completion unexpectedly produced a cycle.")
    return dag


def fit_linear_scm(name: str, data: pd.DataFrame, dag: np.ndarray) -> LinearSCM:
    """Fit one linear regression per node, using its DAG parents as predictors."""
    columns = list(data.columns)
    rank = tier_rank(columns)
    order = sorted(columns, key=lambda c: rank[c])
    models: dict[str, LinearRegression | None] = {}
    residuals: dict[str, np.ndarray] = {}
    root_values: dict[str, np.ndarray] = {}
    parents: dict[str, list[str]] = {}

    # This SCM is simple. It is a matched comparison device, not a
    # mechanistic model of ADNI biology.
    for col in order:
        idx = columns.index(col)
        parent_cols = [columns[i] for i in np.where(dag[:, idx])[0]]
        parent_cols = sorted(parent_cols, key=lambda c: rank[c])
        parents[col] = parent_cols

        y = data[col].to_numpy(dtype=float)
        if parent_cols:
            model = LinearRegression()
            model.fit(data[parent_cols].to_numpy(dtype=float), y)
            pred = model.predict(data[parent_cols].to_numpy(dtype=float))
            models[col] = model
            residuals[col] = y - pred
        else:
            models[col] = None
            root_values[col] = y
            residuals[col] = y - y.mean()

    return LinearSCM(name, columns, dag, data, parents, models, residuals, root_values, order)


def simulate_scm(
    scm: LinearSCM,
    n: int,
    interventions: dict[str, float] | None,
    seed: int,
) -> pd.DataFrame:
    """Generate samples from a fitted SCM, optionally applying do-interventions."""
    rng = np.random.default_rng(seed)
    interventions = interventions or {}
    generated = pd.DataFrame(index=range(n), columns=scm.columns, dtype=float)

    for col in scm.order:
        if col in interventions:
            generated[col] = float(interventions[col])
            continue

        model = scm.models[col]
        if model is None:
            values = scm.root_values[col]
            generated[col] = rng.choice(values, size=n, replace=True)
        else:
            parent_cols = scm.parents[col]
            pred = model.predict(generated[parent_cols].to_numpy(dtype=float))
            eps = rng.choice(scm.residuals[col], size=n, replace=True)
            generated[col] = pred + eps

    return generated[scm.columns]


def run_interventions(
    real_scm: LinearSCM,
    synth_scm: LinearSCM,
    real_data: pd.DataFrame,
    intervention_vars: list[str],
    outcomes: list[str],
    n_sim: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the same biomarker/MRI interventions to real and synthetic SCMs.
    Intervention values are taken from real-data quartiles so both SCMs are
    tested under the same clinically grounded changes.
    """
    usable_interventions = [v for v in intervention_vars if v in real_scm.columns and v in synth_scm.columns]
    usable_outcomes = [v for v in outcomes if v in real_scm.columns and v in synth_scm.columns]

    # Compare every intervention against each SCM's own baseline, then compare
    # real and synthetic effect vectors.
    baseline_real = simulate_scm(real_scm, n_sim, None, seed)
    baseline_synth = simulate_scm(synth_scm, n_sim, None, seed)
    baselines = {
        "real": baseline_real[usable_outcomes].mean().to_dict(),
        "synthetic": baseline_synth[usable_outcomes].mean().to_dict(),
    }

    rows = []
    detail_rows = []
    for variable in usable_interventions:
        q25 = float(real_data[variable].quantile(0.25))
        q75 = float(real_data[variable].quantile(0.75))
        for level, value in [("q25", q25), ("q75", q75)]:
            do_real = simulate_scm(real_scm, n_sim, {variable: value}, seed + 17)
            do_synth = simulate_scm(synth_scm, n_sim, {variable: value}, seed + 17)

            for outcome in usable_outcomes:
                real_mean = float(do_real[outcome].mean())
                synth_mean = float(do_synth[outcome].mean())
                real_delta = real_mean - baselines["real"][outcome]
                synth_delta = synth_mean - baselines["synthetic"][outcome]
                rows.append(
                    {
                        "intervention": variable,
                        "level": level,
                        "do_value_from_real_data": value,
                        "outcome": outcome,
                        "real_do_mean": real_mean,
                        "synthetic_do_mean": synth_mean,
                        "real_delta_vs_baseline": real_delta,
                        "synthetic_delta_vs_baseline": synth_delta,
                        "delta_difference_synth_minus_real": synth_delta - real_delta,
                        "abs_delta_difference": abs(synth_delta - real_delta),
                    }
                )
            detail_rows.append(
                {
                    "intervention": variable,
                    "level": level,
                    "do_value_from_real_data": value,
                    "real_baseline_n": n_sim,
                    "synthetic_baseline_n": n_sim,
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(detail_rows)


def safe_correlation(x: np.ndarray, y: np.ndarray, method: str) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    if method == "pearson":
        return float(np.corrcoef(x, y)[0, 1])
    if method == "spearman":
        return float(stats.spearmanr(x, y).correlation)
    raise ValueError(f"Unknown correlation method: {method}")


def summarize_intervention_agreement(intervention_results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize whether real and synthetic SCMs predict similar effect changes."""
    if intervention_results.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = intervention_results.copy()
    real = df["real_delta_vs_baseline"].to_numpy(dtype=float)
    synth = df["synthetic_delta_vs_baseline"].to_numpy(dtype=float)
    sign_agree = np.sign(real) == np.sign(synth)
    nonzero = (np.abs(real) > 1e-9) | (np.abs(synth) > 1e-9)

    overall = {
        "n_effects": int(len(df)),
        "sign_agreement_rate": float(sign_agree[nonzero].mean()) if nonzero.any() else float("nan"),
        "mean_abs_delta_difference": float(df["abs_delta_difference"].mean()),
        "median_abs_delta_difference": float(df["abs_delta_difference"].median()),
        "max_abs_delta_difference": float(df["abs_delta_difference"].max()),
        "pearson_delta_correlation": safe_correlation(real, synth, "pearson"),
        "spearman_delta_correlation": safe_correlation(real, synth, "spearman"),
    }

    by_outcome_rows = []
    for outcome, group in df.groupby("outcome"):
        r = group["real_delta_vs_baseline"].to_numpy(dtype=float)
        s = group["synthetic_delta_vs_baseline"].to_numpy(dtype=float)
        signs = np.sign(r) == np.sign(s)
        nz = (np.abs(r) > 1e-9) | (np.abs(s) > 1e-9)
        by_outcome_rows.append(
            {
                "outcome": outcome,
                "n_effects": int(len(group)),
                "sign_agreement_rate": float(signs[nz].mean()) if nz.any() else float("nan"),
                "mean_abs_delta_difference": float(group["abs_delta_difference"].mean()),
                "pearson_delta_correlation": safe_correlation(r, s, "pearson"),
                "spearman_delta_correlation": safe_correlation(r, s, "spearman"),
            }
        )

    return pd.DataFrame([overall]), pd.DataFrame(by_outcome_rows)


def score_domain_template(name: str, result: PCStableResult, dag: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score how many expected AD relationships appear in a learned graph."""
    columns = result.columns
    col_idx = {col: idx for idx, col in enumerate(columns)}
    skeleton = skeleton_from_pc(result)
    rank = tier_rank(columns)

    rows = []
    for source, target in EXPECTED_AD_EDGES:
        if source not in col_idx or target not in col_idx:
            continue
        i, j = col_idx[source], col_idx[target]
        rows.append(
            {
                "dataset": name,
                "source": source,
                "target": target,
                "pc_skeleton_present": bool(skeleton[i, j]),
                "domain_dag_direction_present": bool(dag[i, j]),
            }
        )

    # Count directed PC edges that point against the broad tier order. They are
    # reported as a diagnostic, not automatically changed here.
    directed_violations = []
    for i in range(len(columns)):
        for j in range(len(columns)):
            if result.directed[i, j] and rank[columns[i]] > rank[columns[j]]:
                directed_violations.append(f"{columns[i]}->{columns[j]}")

    edge_df = pd.DataFrame(rows)
    expected_total = len(edge_df)
    summary = {
        "dataset": name,
        "expected_edges_evaluable": expected_total,
        "expected_edge_skeleton_recall": float(edge_df["pc_skeleton_present"].mean()) if expected_total else float("nan"),
        "expected_edge_domain_dag_recall": float(edge_df["domain_dag_direction_present"].mean()) if expected_total else float("nan"),
        "pc_directed_tier_violation_count": len(directed_violations),
        "pc_directed_tier_violations": "; ".join(directed_violations),
    }
    return edge_df, pd.DataFrame([summary])


def markov_blanket_from_dag(columns: list[str], dag: np.ndarray, node: str) -> set[str]:
    """Return parents, children, and spouses of one node in a DAG."""
    if node not in columns:
        return set()
    idx = columns.index(node)
    parents = set(np.where(dag[:, idx])[0])
    children = set(np.where(dag[idx, :])[0])
    spouses = set()
    for child in children:
        spouses.update(np.where(dag[:, child])[0])
    blanket_idx = (parents | children | spouses) - {idx}
    return {columns[i] for i in blanket_idx}


def compare_markov_blankets(columns: list[str], real_dag: np.ndarray, synth_dag: np.ndarray, outcomes: list[str]) -> pd.DataFrame:
    """Compare local causal neighborhoods around cognitive outcomes."""
    rows = []
    for outcome in [node for node in outcomes if node in columns]:
        real_mb = markov_blanket_from_dag(columns, real_dag, outcome)
        synth_mb = markov_blanket_from_dag(columns, synth_dag, outcome)
        union = real_mb | synth_mb
        inter = real_mb & synth_mb
        rows.append(
            {
                "outcome": outcome,
                "real_markov_blanket": "; ".join(sorted(real_mb)),
                "synthetic_markov_blanket": "; ".join(sorted(synth_mb)),
                "shared": "; ".join(sorted(inter)),
                "missing_in_synthetic": "; ".join(sorted(real_mb - synth_mb)),
                "extra_in_synthetic": "; ".join(sorted(synth_mb - real_mb)),
                "real_size": len(real_mb),
                "synthetic_size": len(synth_mb),
                "shared_size": len(inter),
                "jaccard": len(inter) / len(union) if union else 1.0,
            }
        )
    return pd.DataFrame(rows)


def save_graph_outputs(name: str, result: PCStableResult, dag: np.ndarray, out_dir: Path) -> None:
    """Write graph adjacency matrices, edge lists, and a DAG image."""
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = result.columns

    skeleton = (result.undirected | result.directed | result.directed.T).astype(int)
    pd.DataFrame(skeleton, index=columns, columns=columns).to_csv(out_dir / f"{name}_pc_skeleton_adjacency.csv")

    cpdag = np.zeros_like(skeleton, dtype=int)
    cpdag[result.undirected] = 1
    cpdag[result.directed] = 2
    pd.DataFrame(cpdag, index=columns, columns=columns).to_csv(out_dir / f"{name}_pc_cpdag_adjacency.csv")
    pd.DataFrame(dag.astype(int), index=columns, columns=columns).to_csv(out_dir / f"{name}_domain_dag_adjacency.csv")

    edge_rows = []
    for i, j in itertools.combinations(range(len(columns)), 2):
        etype = edge_type(result.undirected, result.directed, i, j)
        if etype == "i_to_j":
            edge_rows.append({"source": columns[i], "target": columns[j], "edge_type": "directed"})
        elif etype == "j_to_i":
            edge_rows.append({"source": columns[j], "target": columns[i], "edge_type": "directed"})
        elif etype == "undirected":
            edge_rows.append({"source": columns[i], "target": columns[j], "edge_type": "undirected"})
    pd.DataFrame(edge_rows).to_csv(out_dir / f"{name}_pc_edges.csv", index=False)

    dag_rows = [
        {"source": columns[i], "target": columns[j]}
        for i in range(len(columns))
        for j in range(len(columns))
        if dag[i, j]
    ]
    pd.DataFrame(dag_rows).to_csv(out_dir / f"{name}_domain_dag_edges.csv", index=False)

    draw_graph(columns, dag, out_dir / f"{name}_domain_dag.png", title=f"{name.title()} Domain-Prior DAG")


def draw_graph(columns: list[str], dag: np.ndarray, path: Path, title: str) -> None:
    graph = nx.DiGraph()
    graph.add_nodes_from(columns)
    graph.add_edges_from(
        (columns[i], columns[j])
        for i in range(len(columns))
        for j in range(len(columns))
        if dag[i, j]
    )
    rank = tier_rank(columns)
    pos = {}
    tiers = {}
    for col in columns:
        tier = next((idx for idx, tier_nodes in enumerate(DOMAIN_TIERS) if col in tier_nodes), len(DOMAIN_TIERS))
        tiers.setdefault(tier, []).append(col)
    for tier, nodes in tiers.items():
        nodes = sorted(nodes, key=lambda c: rank[c])
        for idx, node in enumerate(nodes):
            pos[node] = (idx - (len(nodes) - 1) / 2, -tier)

    plt.figure(figsize=(14, 8))
    nx.draw_networkx_nodes(graph, pos, node_size=1400, node_color="#e8eef7", edgecolors="#30415d")
    nx.draw_networkx_edges(graph, pos, arrows=True, arrowstyle="-|>", arrowsize=14, width=1.4, edge_color="#536878")
    nx.draw_networkx_labels(graph, pos, font_size=8)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def run_bootstrap_stability(
    name: str,
    data: pd.DataFrame,
    alpha: float,
    max_cond_set: int,
    transform: str,
    n_runs: int,
    sample_frac: float,
    seed: int,
    out_dir: Path,
) -> np.ndarray:
    """Estimate how often each PC-Stable skeleton edge appears under resampling."""
    columns = list(data.columns)
    p = len(columns)
    counts = np.zeros((p, p), dtype=float)
    rng = np.random.default_rng(seed)
    n_sample = max(10, int(round(len(data) * sample_frac)))

    for run in range(n_runs):
        sample_idx = rng.choice(len(data), size=n_sample, replace=True)
        boot_df = data.iloc[sample_idx].reset_index(drop=True)
        result = pc_stable(boot_df, alpha=alpha, max_cond_set=max_cond_set, transform=transform)
        counts += skeleton_from_pc(result).astype(float)
        if (run + 1) % max(1, n_runs // 5) == 0:
            print(f"  {name} bootstrap {run + 1}/{n_runs}")

    freq = counts / n_runs
    np.fill_diagonal(freq, 0.0)
    pd.DataFrame(freq, index=columns, columns=columns).to_csv(out_dir / f"{name}_bootstrap_edge_frequency.csv")

    edge_rows = []
    for i, j in itertools.combinations(range(p), 2):
        edge_rows.append({"source": columns[i], "target": columns[j], "frequency": freq[i, j]})
    pd.DataFrame(edge_rows).sort_values("frequency", ascending=False).to_csv(
        out_dir / f"{name}_bootstrap_edge_frequency_long.csv",
        index=False,
    )
    return freq


def compare_bootstrap_stability(
    columns: list[str],
    real_freq: np.ndarray,
    synth_freq: np.ndarray,
    threshold: float,
    out_dir: Path,
) -> dict[str, float]:
    """Compare only edges that pass the bootstrap stability threshold."""
    real_stable = real_freq >= threshold
    synth_stable = synth_freq >= threshold
    np.fill_diagonal(real_stable, False)
    np.fill_diagonal(synth_stable, False)

    rows = []
    for i, j in itertools.combinations(range(len(columns)), 2):
        real_has = bool(real_stable[i, j])
        synth_has = bool(synth_stable[i, j])
        if real_has or synth_has:
            rows.append(
                {
                    "source": columns[i],
                    "target": columns[j],
                    "real_frequency": real_freq[i, j],
                    "synthetic_frequency": synth_freq[i, j],
                    "real_stable": real_has,
                    "synthetic_stable": synth_has,
                    "status": "shared_stable"
                    if real_has and synth_has
                    else ("missing_in_synthetic" if real_has else "extra_in_synthetic"),
                }
            )

    metrics = compare_skeleton_matrices(real_stable, synth_stable)
    metrics = {f"stable_{key}": value for key, value in metrics.items()}
    metrics["stability_threshold"] = threshold
    pd.DataFrame(rows).to_csv(out_dir / "bootstrap_stable_edge_comparison.csv", index=False)
    pd.DataFrame([metrics]).to_csv(out_dir / "bootstrap_stability_metrics.csv", index=False)
    return metrics


def save_dag_outputs(name: str, columns: list[str], dag: np.ndarray, out_dir: Path) -> None:
    pd.DataFrame(dag.astype(int), index=columns, columns=columns).to_csv(out_dir / f"{name}_dag_adjacency.csv")
    rows = [
        {"source": columns[i], "target": columns[j]}
        for i in range(len(columns))
        for j in range(len(columns))
        if dag[i, j]
    ]
    pd.DataFrame(rows).to_csv(out_dir / f"{name}_dag_edges.csv", index=False)
    draw_graph(columns, dag, out_dir / f"{name}_dag.png", title=name.replace("_", " ").title())


def learn_notears_linear(
    data: pd.DataFrame,
    transform: str,
    lambda1: float,
    threshold: float,
    max_iter: int,
    domain_bounds: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Learn a NOTEARS-style linear DAG as a second causal discovery check."""
    columns = list(data.columns)
    X = transform_for_pc(data, transform)
    n, d = X.shape
    rank = tier_rank(columns)

    def h_func(W: np.ndarray) -> float:
        return float(np.trace(linalg.expm(W * W)) - d)

    def objective(w: np.ndarray, rho: float, alpha: float) -> tuple[float, np.ndarray]:
        W = w.reshape(d, d)
        residual = X - X @ W
        loss = 0.5 / n * np.sum(residual * residual)
        h = h_func(W)
        E = linalg.expm(W * W)
        grad_loss = -X.T @ residual / n
        grad_h = E.T * W * 2
        smooth = loss + 0.5 * rho * h * h + alpha * h + lambda1 * np.abs(W).sum()
        grad = grad_loss + (rho * h + alpha) * grad_h + lambda1 * np.sign(W)
        return float(smooth), grad.ravel()

    bounds = []
    for i in range(d):
        for j in range(d):
            if i == j or (domain_bounds and rank[columns[i]] >= rank[columns[j]]):
                bounds.append((0.0, 0.0))
            else:
                bounds.append((None, None))

    W_est = np.zeros((d, d), dtype=float)
    rho, alpha = 1.0, 0.0
    h = float("inf")
    for _ in range(max_iter):
        while True:
            sol = optimize.minimize(
                fun=lambda w: objective(w, rho, alpha),
                x0=W_est.ravel(),
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={"maxiter": 1000, "ftol": 1e-9},
            )
            W_new = sol.x.reshape(d, d)
            h_new = h_func(W_new)
            if h_new <= 0.25 * h or rho >= 1e16:
                break
            rho *= 10.0
        W_est = W_new
        h = h_new
        alpha += rho * h
        if h <= 1e-8 or rho >= 1e16:
            break

    W_thresholded = W_est.copy()
    W_thresholded[np.abs(W_thresholded) < threshold] = 0.0
    dag = W_thresholded != 0.0

    graph = nx.DiGraph()
    graph.add_nodes_from(range(d))
    graph.add_edges_from((i, j) for i in range(d) for j in range(d) if dag[i, j])
    if not nx.is_directed_acyclic_graph(graph):
        # Numerical optimization can leave a small cycle after thresholding.
        # Keep the strongest edges greedily while preserving acyclicity.
        weighted_edges = sorted(
            [(abs(W_thresholded[i, j]), i, j) for i in range(d) for j in range(d) if dag[i, j]],
            reverse=True,
        )
        clean = nx.DiGraph()
        clean.add_nodes_from(range(d))
        for _, i, j in weighted_edges:
            clean.add_edge(i, j)
            if not nx.is_directed_acyclic_graph(clean):
                clean.remove_edge(i, j)
        dag = np.zeros((d, d), dtype=bool)
        for i, j in clean.edges:
            dag[i, j] = True

    return dag, W_est


def write_summary(out_dir: Path, args: argparse.Namespace, metrics: dict[str, float], real_df: pd.DataFrame, synth_df: pd.DataFrame) -> None:
    """Save a compact JSON record of parameters, variables, and key metrics."""
    summary = {
        "real_path": str(args.real),
        "synthetic_path": str(args.synthetic),
        "n_real": int(len(real_df)),
        "n_synthetic": int(len(synth_df)),
        "variables": list(real_df.columns),
        "alpha": args.alpha,
        "max_cond_set": args.max_cond_set,
        "pc_transform": args.pc_transform,
        "bootstrap_runs": args.bootstrap_runs,
        "bootstrap_sample_frac": args.bootstrap_sample_frac,
        "bootstrap_threshold": args.bootstrap_threshold,
        "run_notears": args.run_notears,
        "notears_lambda": args.notears_lambda,
        "notears_threshold": args.notears_threshold,
        "notears_max_iter": args.notears_max_iter,
        "notears_domain_bounds": args.notears_domain_bounds,
        "metrics": metrics,
        "domain_tiers": DOMAIN_TIERS,
        "expected_ad_edges": EXPECTED_AD_EDGES,
    }
    (out_dir / "causal_experiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    """Run the full causal-validity experiment and write all result files."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    real_raw, synth_raw = load_raw_tables(args.real, args.synthetic)
    real_df, synth_df = align_real_synthetic(
        real_raw,
        synth_raw,
        include_label=args.include_label,
        include_gender=not args.no_gender,
    )

    real_df.to_csv(args.out_dir / "real_causal_matrix.csv", index=False)
    synth_df.to_csv(args.out_dir / "synthetic_causal_matrix.csv", index=False)

    print(f"Real causal matrix:      {real_df.shape}")
    print(f"Synthetic causal matrix: {synth_df.shape}")
    print(f"Variables: {', '.join(real_df.columns)}")

    # Main structural comparison: learn the same type of graph from real and
    # synthetic data, then compare the two inferred structures.
    real_pc = pc_stable(real_df, alpha=args.alpha, max_cond_set=args.max_cond_set, transform=args.pc_transform)
    synth_pc = pc_stable(synth_df, alpha=args.alpha, max_cond_set=args.max_cond_set, transform=args.pc_transform)

    real_dag = complete_to_domain_dag(real_pc)
    synth_dag = complete_to_domain_dag(synth_pc)

    save_graph_outputs("real", real_pc, real_dag, args.out_dir)
    save_graph_outputs("synthetic", synth_pc, synth_dag, args.out_dir)

    edge_comparison, metrics = compare_graphs(real_pc, synth_pc)
    edge_comparison.to_csv(args.out_dir / "structural_edge_comparison.csv", index=False)
    pd.DataFrame([metrics]).to_csv(args.out_dir / "structural_metrics.csv", index=False)

    real_template_edges, real_template_summary = score_domain_template("real", real_pc, real_dag)
    synth_template_edges, synth_template_summary = score_domain_template("synthetic", synth_pc, synth_dag)
    pd.concat([real_template_edges, synth_template_edges], ignore_index=True).to_csv(
        args.out_dir / "domain_template_edge_scores.csv",
        index=False,
    )
    pd.concat([real_template_summary, synth_template_summary], ignore_index=True).to_csv(
        args.out_dir / "domain_template_summary.csv",
        index=False,
    )

    markov_blankets = compare_markov_blankets(real_pc.columns, real_dag, synth_dag, args.outcomes)
    markov_blankets.to_csv(args.out_dir / "markov_blanket_comparison.csv", index=False)

    real_scm = fit_linear_scm("real", real_df, real_dag)
    synth_scm = fit_linear_scm("synthetic", synth_df, synth_dag)
    intervention_results, intervention_design = run_interventions(
        real_scm,
        synth_scm,
        real_df,
        args.intervention_vars,
        args.outcomes,
        args.n_sim,
        args.seed,
    )
    intervention_results.to_csv(args.out_dir / "interventional_benchmark.csv", index=False)
    intervention_design.to_csv(args.out_dir / "intervention_design.csv", index=False)
    intervention_overall, intervention_by_outcome = summarize_intervention_agreement(intervention_results)
    intervention_overall.to_csv(args.out_dir / "interventional_effect_agreement_overall.csv", index=False)
    intervention_by_outcome.to_csv(args.out_dir / "interventional_effect_agreement_by_outcome.csv", index=False)

    extra_metrics = {}
    if args.bootstrap_runs > 0:
        print(f"\nBootstrap stability ({args.bootstrap_runs} runs)")
        real_freq = run_bootstrap_stability(
            "real",
            real_df,
            args.alpha,
            args.max_cond_set,
            args.pc_transform,
            args.bootstrap_runs,
            args.bootstrap_sample_frac,
            args.seed,
            args.out_dir,
        )
        synth_freq = run_bootstrap_stability(
            "synthetic",
            synth_df,
            args.alpha,
            args.max_cond_set,
            args.pc_transform,
            args.bootstrap_runs,
            args.bootstrap_sample_frac,
            args.seed + 10000,
            args.out_dir,
        )
        extra_metrics.update(
            compare_bootstrap_stability(
                real_pc.columns,
                real_freq,
                synth_freq,
                args.bootstrap_threshold,
                args.out_dir,
            )
        )

    if args.run_notears:
        print("\nLearning NOTEARS-style linear SEM DAGs")
        real_notears_dag, real_weights = learn_notears_linear(
            real_df,
            args.pc_transform,
            args.notears_lambda,
            args.notears_threshold,
            args.notears_max_iter,
            args.notears_domain_bounds,
        )
        synth_notears_dag, synth_weights = learn_notears_linear(
            synth_df,
            args.pc_transform,
            args.notears_lambda,
            args.notears_threshold,
            args.notears_max_iter,
            args.notears_domain_bounds,
        )
        pd.DataFrame(real_weights, index=real_df.columns, columns=real_df.columns).to_csv(args.out_dir / "real_notears_weights.csv")
        pd.DataFrame(synth_weights, index=synth_df.columns, columns=synth_df.columns).to_csv(args.out_dir / "synthetic_notears_weights.csv")
        save_dag_outputs("real_notears", list(real_df.columns), real_notears_dag, args.out_dir)
        save_dag_outputs("synthetic_notears", list(synth_df.columns), synth_notears_dag, args.out_dir)
        notears_metrics = compare_skeleton_matrices(
            real_notears_dag | real_notears_dag.T,
            synth_notears_dag | synth_notears_dag.T,
        )
        notears_metrics = {f"notears_{key}": value for key, value in notears_metrics.items()}
        extra_metrics.update(notears_metrics)
        pd.DataFrame([notears_metrics]).to_csv(args.out_dir / "notears_structural_metrics.csv", index=False)

    all_metrics = {**metrics, **extra_metrics}
    write_summary(args.out_dir, args, all_metrics, real_df, synth_df)

    print("\nStructural metrics")
    for key, value in all_metrics.items():
        print(f"  {key}: {value}")
    print(f"\nSaved causal experiment outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()

"""Benchmark tabular generators on preprocessed ADNI baseline data.

The script creates synthetic tables with simple references, a Gaussian copula,
and optional SDV CTGAN/TVAE models. It reports conventional fidelity and
classifier utility, and can run the causal benchmark on each synthetic output.

The baselines implemented here run with the packages already used in the
repository:
    row_bootstrap          samples whole real rows with replacement
    independent_marginals samples each feature independently within diagnosis
    gaussian_copula       preserves empirical marginals and rank correlations

If SDV is installed later, the script can also run:
    ctgan
    tvae

Example main-feature run:
    .\\.conda\\python.exe scripts\\adni_generator_benchmark.py --real output\\adni_ontocgan_plus_corrected\\adni_cgan_train.csv --ontocgan-synthetic output\\adni_ontocgan_plus_corrected\\synthetic_adni_decoder.csv --seeds 11 22 33 --out-dir output\\adni_generator_benchmark\\main

Example expanded-feature run:
    .\\.conda\\python.exe scripts\\adni_generator_benchmark.py --real output\\adni_ontocgan_plus_corrected\\expanded\\adni_cgan_train.csv --ontocgan-synthetic output\\adni_ontocgan_plus_corrected\\expanded\\synthetic_adni_decoder.csv --seeds 11 22 33 --out-dir output\\adni_generator_benchmark\\expanded
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REAL = ROOT / "output" / "adni_ontocgan_plus_corrected" / "adni_cgan_train.csv"
DEFAULT_ONTOCGAN = ROOT / "output" / "adni_ontocgan_plus_corrected" / "synthetic_adni_decoder.csv"
DEFAULT_OUT_DIR = ROOT / "output" / "adni_generator_benchmark"

META_COLUMNS = {"IRI", "label", "gender"}
DEFAULT_METHODS = ["row_bootstrap", "independent_marginals", "gaussian_copula"]
OPTIONAL_SDV_METHODS = {"ctgan", "tvae"}


@dataclass
class TableSchema:
    """Lightweight schema inferred from the preprocessed ADNI training table."""

    columns: list[str]
    categorical: list[str]
    numeric: list[str]
    integer_like: set[str]
    label_to_iri: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ADNI synthetic tables with alternative generators and evaluate fidelity/utility.",
    )
    parser.add_argument("--real", type=Path, default=DEFAULT_REAL)
    parser.add_argument("--ontocgan-synthetic", type=Path, default=DEFAULT_ONTOCGAN)
    parser.add_argument(
        "--ontocgan-synthetic-map",
        nargs="*",
        default=[],
        metavar="SEED=PATH",
        help="Optional Onto-CGAN seed/path pairs, e.g. 11=output/.../synthetic_adni_decoder.csv.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--seeds", nargs="*", type=int, default=[11, 22, 33])
    parser.add_argument("--n-samples", type=int, default=None, help="Defaults to the real-table row count.")
    parser.add_argument("--no-include-ontocgan", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--ctgan-epochs", type=int, default=300)
    parser.add_argument("--tvae-epochs", type=int, default=300)
    parser.add_argument("--no-sdv-cuda", action="store_true", help="Run SDV CTGAN/TVAE on CPU.")
    parser.add_argument("--reuse-existing", action="store_true", help="Score existing synthetic CSVs instead of regenerating them.")
    parser.add_argument("--run-causal", action="store_true", help="Run causal_adni_experiments.py for every output.")
    parser.add_argument("--causal-bootstrap-runs", type=int, default=30)
    parser.add_argument("--causal-n-sim", type=int, default=3000)
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    """Seed all stochastic libraries used by the benchmark generators."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except Exception as exc:
        print(f"Warning: could not fully seed torch RNGs: {exc}")


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_ontocgan_inputs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    """Return Onto-CGAN seed/path pairs requested for benchmark inclusion."""
    if args.no_include_ontocgan:
        return []

    if args.ontocgan_synthetic_map:
        pairs = []
        for entry in args.ontocgan_synthetic_map:
            seed, sep, raw_path = entry.partition("=")
            if not sep or not seed.strip() or not raw_path.strip():
                raise ValueError(
                    "--ontocgan-synthetic-map entries must look like SEED=PATH; "
                    f"got {entry!r}."
                )
            pairs.append((seed.strip(), resolve_repo_path(Path(raw_path.strip()))))

        missing = [str(path) for _, path in pairs if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing Onto-CGAN synthetic paths: " + "; ".join(missing))
        return pairs

    path = resolve_repo_path(args.ontocgan_synthetic)
    return [("existing", path)] if path.exists() else []


def load_real_table(path: Path) -> pd.DataFrame:
    """Load the preprocessed ADNI table produced by adni_ontocgan_plus.py."""

    if not path.exists():
        raise FileNotFoundError(f"Real preprocessed ADNI table not found: {path}")
    df = pd.read_csv(path)
    missing = META_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Expected columns missing from {path}: {sorted(missing)}")
    return df


def infer_schema(df: pd.DataFrame) -> TableSchema:
    """Infer the small amount of metadata needed by all generators."""

    numeric = [col for col in df.columns if col not in META_COLUMNS]
    categorical = [col for col in ["IRI", "label", "gender"] if col in df.columns]
    integer_like = {
        col
        for col in numeric
        if np.nanmax(np.abs(pd.to_numeric(df[col], errors="coerce") - np.round(pd.to_numeric(df[col], errors="coerce"))))
        < 1e-8
    }
    label_to_iri = {}
    for label, group in df.groupby("label"):
        mode = group["IRI"].mode(dropna=True)
        label_to_iri[str(label)] = str(mode.iloc[0] if len(mode) else group["IRI"].iloc[0])
    return TableSchema(list(df.columns), categorical, numeric, integer_like, label_to_iri)


def allocate_label_counts(real: pd.DataFrame, n_samples: int) -> dict[str, int]:
    """Match the real diagnosis distribution as closely as possible."""

    counts = real["label"].value_counts().sort_index()
    if n_samples == len(real):
        return {str(label): int(count) for label, count in counts.items()}

    expected = counts / counts.sum() * n_samples
    allocated = np.floor(expected).astype(int)
    remainder = int(n_samples - allocated.sum())
    if remainder > 0:
        order = (expected - allocated).sort_values(ascending=False).index
        for label in order[:remainder]:
            allocated[label] += 1
    return {str(label): int(count) for label, count in allocated.items()}


def reorder_and_clean(synth: pd.DataFrame, real: pd.DataFrame, schema: TableSchema) -> pd.DataFrame:
    """Put synthetic data back into the same shape and support as the real table."""

    synth = synth.copy()
    for col in schema.columns:
        if col not in synth.columns:
            synth[col] = np.nan

    # The IRI is a deterministic condition for each diagnosis, so repair that
    # mapping if a generator emits label and IRI separately.
    if "label" in synth.columns and "IRI" in synth.columns:
        synth["IRI"] = synth["label"].astype(str).map(schema.label_to_iri).fillna(synth["IRI"])

    for col in schema.numeric:
        synth[col] = pd.to_numeric(synth[col], errors="coerce")
        synth[col] = synth[col].fillna(real[col].median())
        synth[col] = synth[col].clip(real[col].min(), real[col].max())
        if col in schema.integer_like:
            synth[col] = synth[col].round()

    for col in schema.categorical:
        synth[col] = synth[col].astype(object).fillna(real[col].mode(dropna=True).iloc[0])

    return synth[schema.columns].reset_index(drop=True)


def generate_row_bootstrap(real: pd.DataFrame, schema: TableSchema, n_samples: int, seed: int) -> pd.DataFrame:
    """Upper reference: resample whole real records within diagnosis groups."""

    rng = np.random.default_rng(seed)
    parts = []
    for label, count in allocate_label_counts(real, n_samples).items():
        group = real[real["label"].astype(str) == label]
        idx = rng.choice(group.index.to_numpy(), size=count, replace=True)
        parts.append(real.loc[idx])
    synth = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed)
    return reorder_and_clean(synth, real, schema)


def generate_independent_marginals(real: pd.DataFrame, schema: TableSchema, n_samples: int, seed: int) -> pd.DataFrame:
    """Lower reference: preserve diagnosis-specific marginals, break dependence."""

    rng = np.random.default_rng(seed)
    rows = []
    for label, count in allocate_label_counts(real, n_samples).items():
        group = real[real["label"].astype(str) == label]
        part = pd.DataFrame(index=range(count))
        part["label"] = label
        part["IRI"] = schema.label_to_iri[label]
        part["gender"] = rng.choice(group["gender"].to_numpy(), size=count, replace=True)
        for col in schema.numeric:
            part[col] = rng.choice(group[col].to_numpy(dtype=float), size=count, replace=True)
        rows.append(part)
    synth = pd.concat(rows, ignore_index=True).sample(frac=1.0, random_state=seed)
    return reorder_and_clean(synth, real, schema)


def _nearest_psd_correlation(corr: np.ndarray) -> np.ndarray:
    """Make a sampled correlation matrix numerically safe for multivariate normal."""

    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.clip(eigvals, 1e-6, None)
    repaired = (eigvecs * eigvals) @ eigvecs.T
    scale = np.sqrt(np.diag(repaired))
    repaired = repaired / np.outer(scale, scale)
    np.fill_diagonal(repaired, 1.0)
    return repaired


def _empirical_quantile(values: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Map uniform probabilities back to an empirical feature distribution."""

    values = np.asarray(values, dtype=float)
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
    return np.quantile(values, probs)


def generate_gaussian_copula(real: pd.DataFrame, schema: TableSchema, n_samples: int, seed: int) -> pd.DataFrame:
    """Gaussian copula baseline fitted separately within each diagnosis group."""

    rng = np.random.default_rng(seed)
    rows = []
    for label, count in allocate_label_counts(real, n_samples).items():
        group = real[real["label"].astype(str) == label].reset_index(drop=True)
        numeric = group[schema.numeric].astype(float)

        # Rank-Gaussianize each feature, fit one correlation matrix, then map
        # sampled values back through empirical quantiles. This is why the
        # copula directly targets dependence structure.
        z_cols = []
        for col in schema.numeric:
            ranks = stats.rankdata(numeric[col].to_numpy(), method="average")
            probs = (ranks - 0.5) / len(numeric)
            z_cols.append(stats.norm.ppf(probs))
        z = np.column_stack(z_cols)
        corr = _nearest_psd_correlation(np.corrcoef(z, rowvar=False))

        sampled_z = rng.multivariate_normal(np.zeros(len(schema.numeric)), corr, size=count)
        sampled_u = stats.norm.cdf(sampled_z)

        part = pd.DataFrame(index=range(count))
        part["label"] = label
        part["IRI"] = schema.label_to_iri[label]
        part["gender"] = rng.choice(group["gender"].to_numpy(), size=count, replace=True)
        for idx, col in enumerate(schema.numeric):
            part[col] = _empirical_quantile(numeric[col].to_numpy(), sampled_u[:, idx])
        rows.append(part)

    synth = pd.concat(rows, ignore_index=True).sample(frac=1.0, random_state=seed)
    return reorder_and_clean(synth, real, schema)


def generate_sdv(real: pd.DataFrame, schema: TableSchema, method: str, n_samples: int, seed: int, args: argparse.Namespace) -> pd.DataFrame:
    """Run SDV CTGAN/TVAE if SDV is installed in the local environment."""
    set_global_seed(seed)

    try:
        from sdv.metadata import SingleTableMetadata
        from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
    except ImportError as exc:
        raise RuntimeError(
            f"Method {method} requires SDV. Install it first, for example: pip install sdv"
        ) from exc

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real)
    for col in schema.categorical:
        metadata.update_column(col, sdtype="categorical")
    for col in schema.numeric:
        metadata.update_column(col, sdtype="numerical")

    cls = CTGANSynthesizer if method == "ctgan" else TVAESynthesizer
    epochs = args.ctgan_epochs if method == "ctgan" else args.tvae_epochs
    kwargs = {"epochs": epochs, "verbose": True}
    if args.no_sdv_cuda:
        kwargs["enable_gpu"] = False
    synthesizer = cls(metadata, **kwargs)
    synthesizer.fit(real)
    synth = synthesizer.sample(n_samples)
    return reorder_and_clean(synth, real, schema)


def generate(method: str, real: pd.DataFrame, schema: TableSchema, n_samples: int, seed: int, args: argparse.Namespace) -> pd.DataFrame:
    if method == "row_bootstrap":
        return generate_row_bootstrap(real, schema, n_samples, seed)
    if method == "independent_marginals":
        return generate_independent_marginals(real, schema, n_samples, seed)
    if method == "gaussian_copula":
        return generate_gaussian_copula(real, schema, n_samples, seed)
    if method in OPTIONAL_SDV_METHODS:
        return generate_sdv(real, schema, method, n_samples, seed, args)
    raise ValueError(f"Unknown generator method: {method}")


def total_variation(real: pd.Series, synth: pd.Series) -> float:
    real_dist = real.astype(str).value_counts(normalize=True)
    synth_dist = synth.astype(str).value_counts(normalize=True)
    cats = sorted(set(real_dist.index) | set(synth_dist.index))
    return float(0.5 * sum(abs(real_dist.get(cat, 0.0) - synth_dist.get(cat, 0.0)) for cat in cats))


def fidelity_metrics(real: pd.DataFrame, synth: pd.DataFrame, schema: TableSchema, method: str, seed: str) -> dict[str, float | str]:
    """Conventional distributional similarity checks for one synthetic table."""

    rows = {
        "method": method,
        "seed": seed,
        "n_real": int(len(real)),
        "n_synthetic": int(len(synth)),
        "label_tvd": total_variation(real["label"], synth["label"]),
        "gender_tvd": total_variation(real["gender"], synth["gender"]),
    }

    mean_diffs = []
    std_ratio_errors = []
    ks_stats = []
    range_violations = []
    for col in schema.numeric:
        r = pd.to_numeric(real[col], errors="coerce").to_numpy(dtype=float)
        s = pd.to_numeric(synth[col], errors="coerce").to_numpy(dtype=float)
        r_std = np.nanstd(r)
        s_std = np.nanstd(s)
        mean_diffs.append(abs(np.nanmean(s) - np.nanmean(r)) / (r_std + 1e-8))
        std_ratio_errors.append(abs(math.log((s_std + 1e-8) / (r_std + 1e-8))))
        ks_stats.append(stats.ks_2samp(r, s).statistic)
        range_violations.append(np.mean((s < np.nanmin(r)) | (s > np.nanmax(r))))

    r_corr = real[schema.numeric].astype(float).corr().to_numpy()
    s_corr = synth[schema.numeric].astype(float).corr().to_numpy()
    upper = np.triu_indices_from(r_corr, k=1)
    corr_rmse = float(np.sqrt(np.nanmean((r_corr[upper] - s_corr[upper]) ** 2)))

    rows.update(
        {
            "mean_abs_standardized_mean_diff": float(np.mean(mean_diffs)),
            "mean_abs_log_std_ratio": float(np.mean(std_ratio_errors)),
            "mean_ks_statistic": float(np.mean(ks_stats)),
            "max_ks_statistic": float(np.max(ks_stats)),
            "correlation_rmse": corr_rmse,
            "numeric_range_violation_rate": float(np.mean(range_violations)),
        }
    )
    return rows


def make_encoder(categorical_cols: list[str], numeric_cols: list[str]) -> ColumnTransformer:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        transformers=[
            ("categorical", encoder, categorical_cols),
            ("numeric", SimpleImputer(strategy="median"), numeric_cols),
        ],
        remainder="drop",
    )


def utility_metrics(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    schema: TableSchema,
    method: str,
    seed: str,
    test_size: float,
) -> dict[str, float | str]:
    """Train a diagnosis classifier on synthetic data and test it on real rows."""

    feature_cols = [col for col in schema.columns if col not in {"IRI", "label"}]
    categorical_cols = [col for col in feature_cols if col == "gender"]
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]

    real_train, real_test = train_test_split(
        real,
        test_size=test_size,
        random_state=123,
        stratify=real["label"],
    )

    def _fit_and_score(train_df: pd.DataFrame) -> tuple[float, float]:
        model = Pipeline(
            steps=[
                ("preprocess", make_encoder(categorical_cols, numeric_cols)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=321,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        model.fit(train_df[feature_cols], train_df["label"])
        pred = model.predict(real_test[feature_cols])
        return (
            float(accuracy_score(real_test["label"], pred)),
            float(f1_score(real_test["label"], pred, average="macro")),
        )

    real_acc, real_f1 = _fit_and_score(real_train)
    synth_acc, synth_f1 = _fit_and_score(synth)
    return {
        "method": method,
        "seed": seed,
        "real_train_real_test_accuracy": real_acc,
        "synthetic_train_real_test_accuracy": synth_acc,
        "accuracy_ratio_vs_real": synth_acc / real_acc if real_acc else float("nan"),
        "real_train_real_test_macro_f1": real_f1,
        "synthetic_train_real_test_macro_f1": synth_f1,
        "macro_f1_ratio_vs_real": synth_f1 / real_f1 if real_f1 else float("nan"),
    }


def write_causal_commands(real_path: Path, synthetic_paths: list[tuple[str, str, Path]], args: argparse.Namespace) -> None:
    """Write a PowerShell helper to run causal evaluation for all generated tables."""

    lines = [
        "# Run these from the repository root after synthetic generation.",
        "# Bootstrap runs are modest by default so this is practical for multiple seeds.",
        "",
    ]
    for method, seed, synth_path in synthetic_paths:
        out_dir = args.out_dir / "causal" / method / f"seed_{seed}"
        lines.append(
            ".\\.conda\\python.exe scripts\\causal_adni_experiments.py "
            f"--real {real_path} "
            f"--synthetic {synth_path} "
            "--alpha 0.01 --max-cond-set 3 "
            f"--n-sim {args.causal_n_sim} "
            f"--bootstrap-runs {args.causal_bootstrap_runs} "
            "--bootstrap-threshold 0.70 "
            "--run-notears --notears-domain-bounds "
            "--notears-lambda 0.01 --notears-threshold 0.10 --notears-max-iter 10 "
            f"--out-dir {out_dir}"
        )
    (args.out_dir / "run_causal_for_generators.ps1").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_causal_commands(real_path: Path, synthetic_paths: list[tuple[str, str, Path]], args: argparse.Namespace) -> None:
    """Optionally execute the generated causal commands immediately."""

    for method, seed, synth_path in synthetic_paths:
        out_dir = args.out_dir / "causal" / method / f"seed_{seed}"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "causal_adni_experiments.py"),
            "--real",
            str(real_path),
            "--synthetic",
            str(synth_path),
            "--alpha",
            "0.01",
            "--max-cond-set",
            "3",
            "--n-sim",
            str(args.causal_n_sim),
            "--bootstrap-runs",
            str(args.causal_bootstrap_runs),
            "--bootstrap-threshold",
            "0.70",
            "--run-notears",
            "--notears-domain-bounds",
            "--notears-lambda",
            "0.01",
            "--notears-threshold",
            "0.10",
            "--notears-max-iter",
            "10",
            "--out-dir",
            str(out_dir),
        ]
        print(f"\nRunning causal benchmark for {method} seed {seed}")
        subprocess.run(cmd, check=True)


def summarize_by_method(df: pd.DataFrame, out_path: Path) -> None:
    numeric_cols = [col for col in df.columns if col not in {"method", "seed"}]
    summary = df.groupby("method")[numeric_cols].agg(["mean", "std"])
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.to_flat_index()]
    summary.reset_index().to_csv(out_path, index=False)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    real = load_real_table(args.real)
    schema = infer_schema(real)
    n_samples = args.n_samples or len(real)

    fidelity_rows = []
    utility_rows = []
    generated_paths: list[tuple[str, str, Path]] = []

    # Onto-CGAN is generated by its own training script, then scored alongside
    # the tabular references and SDV models.
    ontocgan_inputs = load_ontocgan_inputs(args)
    for seed, synth_path in ontocgan_inputs:
        ontocgan = reorder_and_clean(pd.read_csv(synth_path), real, schema)
        method = "ontocgan_decoder"
        if seed == "existing":
            out_path = args.out_dir / method / "synthetic.csv"
        else:
            out_path = args.out_dir / method / f"seed_{seed}" / "synthetic.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ontocgan.to_csv(out_path, index=False)
        fidelity_rows.append(fidelity_metrics(real, ontocgan, schema, method, seed))
        utility_rows.append(utility_metrics(real, ontocgan, schema, method, seed, args.test_size))
        generated_paths.append((method, seed, out_path))

    for seed in args.seeds:
        for method in args.methods:
            out_path = args.out_dir / method / f"seed_{seed}" / "synthetic.csv"
            if args.reuse_existing and out_path.exists():
                print(f"\nReusing {method} seed {seed}: {out_path}")
                synth = reorder_and_clean(pd.read_csv(out_path), real, schema)
            else:
                print(f"\nGenerating {method} seed {seed}")
                set_global_seed(seed)
                synth = generate(method, real, schema, n_samples, seed, args)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            synth.to_csv(out_path, index=False)

            seed_label = str(seed)
            fidelity_rows.append(fidelity_metrics(real, synth, schema, method, seed_label))
            utility_rows.append(utility_metrics(real, synth, schema, method, seed_label, args.test_size))
            generated_paths.append((method, seed_label, out_path))

    fidelity = pd.DataFrame(fidelity_rows)
    utility = pd.DataFrame(utility_rows)
    fidelity.to_csv(args.out_dir / "fidelity_metrics.csv", index=False)
    utility.to_csv(args.out_dir / "utility_metrics.csv", index=False)
    summarize_by_method(fidelity, args.out_dir / "fidelity_metrics_by_method.csv")
    summarize_by_method(utility, args.out_dir / "utility_metrics_by_method.csv")
    write_causal_commands(args.real, generated_paths, args)

    summary = {
        "real_path": str(args.real),
        "n_real": int(len(real)),
        "n_samples_per_synthetic": int(n_samples),
        "methods": args.methods,
        "seeds": args.seeds,
        "included_ontocgan": bool(ontocgan_inputs),
        "ontocgan_inputs": {seed: str(path) for seed, path in ontocgan_inputs},
        "reuse_existing": bool(args.reuse_existing),
        "sdv_cuda": bool(not args.no_sdv_cuda),
        "outputs": [str(path) for _, _, path in generated_paths],
    }
    (args.out_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.run_causal:
        run_causal_commands(args.real, generated_paths, args)

    print(f"\nSaved generator benchmark outputs to: {args.out_dir}")
    print(f"Fidelity metrics: {args.out_dir / 'fidelity_metrics.csv'}")
    print(f"Utility metrics:  {args.out_dir / 'utility_metrics.csv'}")
    print(f"Causal commands:  {args.out_dir / 'run_causal_for_generators.ps1'}")


if __name__ == "__main__":
    main()

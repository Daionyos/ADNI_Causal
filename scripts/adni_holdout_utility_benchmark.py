"""Train-on-synthetic, test-on-real utility benchmark with a fixed holdout set.

The real ADNI table is split once. Generators are fitted on the training
partition, the diagnosis classifier is trained on synthetic rows, and all
scores are computed on the same held-out real patients.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import adni_generator_benchmark as bench


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "output" / "resit_aligned_experiments"
DEFAULT_OUT_DIR = DEFAULT_ROOT / "holdout_utility"
DEFAULT_FEATURE_SETS = ["main", "expanded"]
DEFAULT_METHODS = ["row_bootstrap", "independent_marginals", "gaussian_copula", "ctgan", "tvae"]
DEFAULT_SEEDS = [11, 22, 33]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate predictive utility with generators fitted on a real training split.",
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--feature-sets", nargs="*", default=DEFAULT_FEATURE_SETS)
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument("--classifier-seed", type=int, default=321)
    parser.add_argument("--n-samples", type=int, default=None, help="Defaults to len(real_train).")
    parser.add_argument("--ctgan-epochs", type=int, default=300)
    parser.add_argument("--tvae-epochs", type=int, default=300)
    parser.add_argument("--no-sdv-cuda", action="store_true", help="Run SDV CTGAN/TVAE on CPU.")
    parser.add_argument("--include-ontocgan", action="store_true", help="Also train Onto-CGAN on the real split.")
    parser.add_argument("--ontocgan-epochs", type=int, default=300)
    parser.add_argument("--ontocgan-batch-size", type=int, default=500)
    parser.add_argument("--ontocgan-mode", choices=["ontology", "decoder", "decoder_contrast"], default="decoder")
    parser.add_argument("--ontocgan-no-cuda", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true", help="Score existing synthetic CSVs if present.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate synthetic CSVs even if present.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def set_seed(seed: int) -> None:
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


def split_real(real: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the stratified real train/test split used for utility scoring."""

    train, test = train_test_split(
        real,
        test_size=args.test_size,
        random_state=args.split_seed,
        stratify=real["label"],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def utility_on_fixed_test(
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synth_train: pd.DataFrame,
    schema: bench.TableSchema,
    method: str,
    seed: str,
    args: argparse.Namespace,
) -> dict[str, float | str | int]:
    """Score synthetic training data against the fixed real test set."""

    feature_cols = [col for col in schema.columns if col not in {"IRI", "label"}]
    categorical_cols = [col for col in feature_cols if col == "gender"]
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]

    def fit_and_score(train_df: pd.DataFrame) -> tuple[float, float]:
        model = Pipeline(
            steps=[
                ("preprocess", bench.make_encoder(categorical_cols, numeric_cols)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=args.classifier_seed,
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

    # Use the real-train baseline on the same test patients as the denominator.
    real_acc, real_f1 = fit_and_score(real_train)
    synth_acc, synth_f1 = fit_and_score(synth_train)
    return {
        "method": method,
        "seed": seed,
        "n_real_train": int(len(real_train)),
        "n_real_test": int(len(real_test)),
        "n_synthetic_train": int(len(synth_train)),
        "real_train_real_test_accuracy": real_acc,
        "synthetic_train_real_test_accuracy": synth_acc,
        "accuracy_ratio_vs_real": synth_acc / real_acc if real_acc else float("nan"),
        "real_train_real_test_macro_f1": real_f1,
        "synthetic_train_real_test_macro_f1": synth_f1,
        "macro_f1_ratio_vs_real": synth_f1 / real_f1 if real_f1 else float("nan"),
    }


def train_ontocgan(
    real_train: pd.DataFrame,
    schema: bench.TableSchema,
    n_samples: int,
    seed: int,
    out_dir: Path,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Train Onto-CGAN on the real training partition."""

    from adni_ontocgan_plus import (
        DEFAULT_EMB_DIR,
        postprocess_synthetic,
        set_global_seed,
        train_or_load_model,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    oc_args = SimpleNamespace(
        out_dir=out_dir,
        mode=args.ontocgan_mode,
        emb_dir=DEFAULT_EMB_DIR,
        epochs=args.ontocgan_epochs,
        batch_size=args.ontocgan_batch_size,
        overwrite=args.overwrite,
        no_cuda=args.ontocgan_no_cuda,
    )
    set_global_seed(seed, use_cuda=not args.ontocgan_no_cuda)
    model, _ = train_or_load_model(real_train, oc_args)
    synth = model.sample(n_samples)
    synth = postprocess_synthetic(synth, real_train, clip=True)
    return bench.reorder_and_clean(synth, real_train, schema)


def summarize(df: pd.DataFrame, out_path: Path) -> None:
    numeric_cols = [col for col in df.columns if col not in {"feature_set", "method", "seed"}]
    summary = df.groupby(["feature_set", "method"], sort=False)[numeric_cols].agg(["mean", "std"])
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.to_flat_index()]
    summary.reset_index().to_csv(out_path, index=False)


def run_feature_set(feature_set: str, args: argparse.Namespace) -> list[dict[str, float | str | int]]:
    root = resolve(args.root)
    out_root = resolve(args.out_dir)
    real_path = root / feature_set / "real" / "adni_cgan_train.csv"
    if not real_path.exists():
        raise FileNotFoundError(f"Missing prepared real table: {real_path}")

    real = bench.load_real_table(real_path)
    schema = bench.infer_schema(real)
    real_train, real_test = split_real(real, args)
    n_samples = args.n_samples or len(real_train)

    fs_out = out_root / feature_set
    fs_out.mkdir(parents=True, exist_ok=True)
    real_train.to_csv(fs_out / "real_generator_train.csv", index=False)
    real_test.to_csv(fs_out / "real_heldout_test.csv", index=False)

    metadata = {
        "feature_set": feature_set,
        "real_path": str(real_path),
        "n_full_real": int(len(real)),
        "n_real_train": int(len(real_train)),
        "n_real_test": int(len(real_test)),
        "n_samples_per_synthetic": int(n_samples),
        "test_size": args.test_size,
        "split_seed": args.split_seed,
        "seeds": args.seeds,
        "methods": args.methods,
        "include_ontocgan": bool(args.include_ontocgan),
    }
    (fs_out / "split_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    rows: list[dict[str, float | str | int]] = []
    generator_args = SimpleNamespace(
        ctgan_epochs=args.ctgan_epochs,
        tvae_epochs=args.tvae_epochs,
        no_sdv_cuda=args.no_sdv_cuda,
    )

    for seed in args.seeds:
        for method in args.methods:
            synth_path = fs_out / method / f"seed_{seed}" / "synthetic.csv"
            if synth_path.exists() and args.reuse_existing and not args.overwrite:
                print(f"Reusing {feature_set} {method} seed {seed}: {synth_path}")
                synth = bench.reorder_and_clean(pd.read_csv(synth_path), real_train, schema)
            else:
                print(f"Generating {feature_set} {method} seed {seed} from real_train")
                set_seed(seed)
                synth = bench.generate(method, real_train, schema, n_samples, seed, generator_args)
                synth_path.parent.mkdir(parents=True, exist_ok=True)
                synth.to_csv(synth_path, index=False)
            row = utility_on_fixed_test(real_train, real_test, synth, schema, method, str(seed), args)
            row["feature_set"] = feature_set
            rows.append(row)

        if args.include_ontocgan:
            method = "ontocgan_decoder"
            synth_path = fs_out / method / f"seed_{seed}" / "synthetic.csv"
            model_dir = fs_out / method / f"seed_{seed}" / "model"
            if synth_path.exists() and args.reuse_existing and not args.overwrite:
                print(f"Reusing {feature_set} {method} seed {seed}: {synth_path}")
                synth = bench.reorder_and_clean(pd.read_csv(synth_path), real_train, schema)
            else:
                print(f"Training {feature_set} {method} seed {seed} from real_train")
                synth = train_ontocgan(real_train, schema, n_samples, seed, model_dir, args)
                synth_path.parent.mkdir(parents=True, exist_ok=True)
                synth.to_csv(synth_path, index=False)
            row = utility_on_fixed_test(real_train, real_test, synth, schema, method, str(seed), args)
            row["feature_set"] = feature_set
            rows.append(row)

    pd.DataFrame(rows).to_csv(fs_out / "holdout_utility_metrics.csv", index=False)
    return rows


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, float | str | int]] = []
    for feature_set in args.feature_sets:
        all_rows.extend(run_feature_set(feature_set, args))

    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(out_dir / "holdout_utility_metrics_all.csv", index=False)
    summarize(all_df, out_dir / "holdout_utility_summary.csv")
    print(f"Wrote {out_dir / 'holdout_utility_metrics_all.csv'}")
    print(f"Wrote {out_dir / 'holdout_utility_summary.csv'}")


if __name__ == "__main__":
    main()

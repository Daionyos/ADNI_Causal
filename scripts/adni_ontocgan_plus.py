"""Prepare ADNI baseline data and train/sample Onto-CGAN+.

This script is ADNI-specific. The repository's
evaluation/generate.py workflow is aimed at zero-shot experiments where one
disease is withheld. Here, all baseline ADNI diagnosis classes are used and
sampled back out.

Expected training format for onto_cgan+:
    IRI, label, gender, <numeric clinical features...>

Run from the repository root, for example:
    .\\.conda\\python.exe scripts\\adni_ontocgan_plus.py --quick --overwrite

Main feature-set run:
    .\\.conda\\python.exe scripts\\adni_ontocgan_plus.py --feature-set main --epochs 300 --batch-size 500 --n-samples 2419 --mode decoder --overwrite --out-dir output\\adni_ontocgan_plus_corrected

Expanded feature-set run:
    .\\.conda\\python.exe scripts\\adni_ontocgan_plus.py --feature-set expanded --epochs 300 --batch-size 500 --n-samples 2419 --mode decoder --overwrite --out-dir output\\adni_ontocgan_plus_corrected\\expanded
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _add_local_import_paths() -> None:
    """Make the script runnable even when the local package is not installed."""
    src_path = str(ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


_add_local_import_paths()

import numpy as np
import pandas as pd


DEFAULT_ADNIMERGE = ROOT / "data" / "ADNIMERGE.csv"
DEFAULT_EMB_DIR = ROOT / "data" / "ontology_emb"
DEFAULT_OUT_DIR = ROOT / "output" / "adni_ontocgan_plus"


DX_TO_IRI = {
    # CN is not a disease and should not be mapped to an unrelated ORDO disease.
    # OntologyEmbedding will return a zero vector for this explicit local control
    # token, which is preferable to false biomedical semantics.
    "CN": "ADNI_CN_CONTROL",
    # ADNI MCI is a clinical stage rather than a rare disease entity, so use the
    # broad HPO cognitive impairment phenotype available in the ORDO+HPO vectors.
    "MCI": "http://purl.obolibrary.org/obo/HP_0100543",
    "AD": "http://www.orpha.net/ORDO/Orphanet_26929",
}

DX_TO_IRI_NOTES = {
    "CN": "non-disease control baseline; zero-vector conditioning",
    "MCI": "HPO cognitive impairment phenotype proxy",
    "AD": "ORDO Alzheimer disease",
}


DX_NORMALIZATION = {
    "CN": "CN",
    "NL": "CN",
    "MCI": "MCI",
    "EMCI": "MCI",
    "LMCI": "MCI",
    "CN to MCI": "MCI",
    "MCI to Dementia": "AD",
    "MCI to AD": "AD",
    "Dementia": "AD",
    "AD": "AD",
    "Alzheimer's Disease": "AD",
    "Alzheimer's": "AD",
}


MAIN_NUMERIC_FEATURES = [
    # Genetic and demographic risk
    "AGE",
    "PTEDUCAT",
    "APOE4",
    # CSF biomarkers
    "ABETA",
    "TAU",
    "PTAU",
    # PET
    "AV45",
    "FDG",
    # Structural MRI
    "Hippocampus",
    "Entorhinal",
    "Fusiform",
    "MidTemp",
    "ICV",
    # Cognitive and functional outcomes
    "CDRSB",
    "MMSE",
    "ADAS13",
    "RAVLT_immediate",
    "FAQ",
    "MOCA",
]

EXPANDED_EXTRA_NUMERIC_FEATURES = [
    # Additional structural MRI sensitivity variables
    "Ventricles",
    "WholeBrain",
    # Additional cognitive and functional sensitivity variables
    "ADAS11",
    "ADASQ4",
    "RAVLT_learning",
    "RAVLT_forgetting",
    "RAVLT_perc_forgetting",
    "LDELTOTAL",
    "TRABSCOR",
    "mPACCdigit",
    "mPACCtrailsB",
]

# The main feature set keeps the core ADNI variables. The expanded set adds
# extra MRI and cognitive measures for a higher-dimensional sensitivity run.
FEATURE_SETS = {
    "main": MAIN_NUMERIC_FEATURES,
    "expanded": MAIN_NUMERIC_FEATURES + EXPANDED_EXTRA_NUMERIC_FEATURES,
}

INTEGER_LIKE_FEATURES = {
    "PTEDUCAT",
    "APOE4",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare ADNIMERGE.csv, train Onto-CGAN+, and save synthetic ADNI rows.",
    )
    parser.add_argument("--adnimerge", type=Path, default=DEFAULT_ADNIMERGE)
    parser.add_argument("--emb-dir", type=Path, default=DEFAULT_EMB_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--feature-set",
        choices=sorted(FEATURE_SETS),
        default="main",
        help="Use the main variables or the expanded sensitivity-analysis variables.",
    )
    parser.add_argument("--mode", choices=["ontology", "decoder", "decoder_contrast"], default="decoder")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--n-samples", type=int, default=None, help="Defaults to the number of training rows.")
    parser.add_argument("--missing-threshold", type=float, default=0.60)
    parser.add_argument("--limit-rows", type=int, default=None, help="Optional stratified row cap for smoke tests.")
    parser.add_argument("--quick", action="store_true", help="Smoke test: epochs=2, batch_size=50, n_samples=100.")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--no-clip", action="store_true", help="Do not clip synthetic numeric values to real-data ranges.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_global_seed(seed: int, use_cuda: bool) -> None:
    """Seed Python, NumPy, and Torch before a generation run."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if use_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except Exception as exc:
        print(f"Warning: could not fully seed torch RNGs: {exc}")


def _clean_numeric(series: pd.Series) -> pd.Series:
    """Convert ADNI numeric-ish fields, including values like '>1700'."""
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace(r"^[<>]=?\s*", "", regex=True)
        .replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "-4": pd.NA})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _select_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one baseline row per participant.

    ADNI is longitudinal, but this pipeline is cross-sectional.
    Keeping only baseline visits avoids treating repeated visits from the same
    participant as independent samples.
    """
    if "VISCODE" not in df.columns:
        raise ValueError("ADNIMERGE.csv must contain VISCODE so baseline rows can be selected.")

    baseline = df[df["VISCODE"].astype("string").str.lower().eq("bl")].copy()
    if "RID" in baseline.columns:
        baseline = baseline.drop_duplicates(subset=["RID"], keep="first")
    return baseline.reset_index(drop=True)


def _normalise_diagnosis(df: pd.DataFrame) -> pd.Series:
    """Map ADNI diagnosis variants onto the three classes used for generation."""
    dx = pd.Series(pd.NA, index=df.index, dtype="string")
    if "DX" in df.columns:
        dx = df["DX"].astype("string")
    if "DX_bl" in df.columns:
        dx = dx.fillna(df["DX_bl"].astype("string"))

    if dx.isna().all():
        raise ValueError("Could not find usable DX or DX_bl diagnosis values.")

    return dx.str.strip().map(DX_NORMALIZATION)


def _impute_numeric(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Median-impute numeric ADNI features after sparse columns are removed."""
    for col in numeric_cols:
        median = df[col].median()
        if pd.isna(median):
            raise ValueError(f"Cannot impute {col}: all values are missing.")
        df[col] = df[col].fillna(median).astype(float)
    return df


def prepare_adni(args: argparse.Namespace) -> pd.DataFrame:
    """Create the tabular input consumed by Onto-CGAN+.

    The output keeps IRI and label as the first two columns because Onto-CGAN+
    uses the first column as the ontology/control condition and the second as
    the human-readable class label.  Gender is kept as a categorical covariate;
    the remaining columns are numeric clinical features.
    """
    if not args.adnimerge.exists():
        raise FileNotFoundError(f"ADNIMERGE.csv not found: {args.adnimerge}")

    # ADNI data are read locally and are not written to the repository.
    raw = pd.read_csv(args.adnimerge, low_memory=False)
    baseline = _select_baseline(raw)
    baseline["label"] = _normalise_diagnosis(baseline)
    baseline = baseline.dropna(subset=["label"]).copy()
    baseline = baseline[baseline["label"].isin(DX_TO_IRI)].copy()

    numeric_features = FEATURE_SETS[args.feature_set]
    available_numeric = [col for col in numeric_features if col in baseline.columns]
    missing_numeric = [col for col in numeric_features if col not in baseline.columns]
    if missing_numeric:
        print("Skipping unavailable ADNI feature columns:", ", ".join(missing_numeric))

    if "PTGENDER" not in baseline.columns:
        raise ValueError("PTGENDER is required so Onto-CGAN+ has a second non-IRI categorical column.")

    # Onto-CGAN consumes the ontology/control condition first. CN is represented
    # by a local zero-vector control IRI instead of an unrelated disease term.
    out = pd.DataFrame(index=baseline.index)
    out["IRI"] = baseline["label"].map(DX_TO_IRI).astype(object)
    out["label"] = baseline["label"].astype(object)
    out["gender"] = baseline["PTGENDER"].astype(str).str.strip().replace({"": "Unknown"}).astype(object)

    for col in available_numeric:
        out[col] = _clean_numeric(baseline[col])

    # Very sparse variables are more likely to be artifacts of imputation than
    # useful clinical signal, so they are excluded before median imputation.
    missingness = out[available_numeric].isna().mean().sort_values(ascending=False)
    sparse_cols = missingness[missingness > args.missing_threshold].index.tolist()
    if sparse_cols:
        print(
            "Dropping columns above missingness threshold "
            f"{args.missing_threshold:.0%}: {', '.join(sparse_cols)}"
        )
        out = out.drop(columns=sparse_cols)
        available_numeric = [col for col in available_numeric if col not in sparse_cols]
        missingness = missingness.drop(index=sparse_cols)

    missingness.rename("missing_fraction").reset_index().rename(
        columns={"index": "feature"}
    ).to_csv(args.out_dir / "adni_missingness_before_imputation.csv", index=False)

    out = _impute_numeric(out, available_numeric)

    if args.limit_rows is not None and args.limit_rows < len(out):
        # Smoke tests should still contain CN/MCI/AD examples, so the optional
        # row cap samples approximately within each diagnosis group.
        sampled_parts = []
        for _, group in out.groupby("label", sort=False):
            n_group = max(1, round(args.limit_rows * len(group) / len(out)))
            sampled_parts.append(group.sample(min(n_group, len(group)), random_state=args.seed))
        out = pd.concat(sampled_parts, ignore_index=True)
        out = out.sample(frac=1, random_state=args.seed).head(args.limit_rows).reset_index(drop=True)

    # Keep the exact column order expected by Onto-CGAN+.
    out = out[["IRI", "label", "gender"] + available_numeric].reset_index(drop=True)
    out[["IRI", "label", "gender"]] = out[["IRI", "label", "gender"]].astype(object)
    return out


def ensure_embedding_dicts(emb_dir: Path) -> tuple[Path, Path, Path]:
    """Return ontology embedding files, rebuilding dictionaries when needed.

    Some local setups have a full ontology.embeddings file but damaged or tiny
    HPO/ORDO dictionaries. Rebuilding from the embedding vocabulary keeps the
    IRI lookup aligned with the embedding model.
    """
    emb_path = emb_dir / "ontology.embeddings"
    hp_dict = emb_dir / "HPO.dict"
    ordo_dict = emb_dir / "ORDO.dict"

    if not emb_path.exists():
        raise FileNotFoundError(
            f"Missing {emb_path}. Download the ontology embeddings referenced in the README "
            "and place ontology.embeddings under data/ontology_emb/."
        )

    emb_dir.mkdir(parents=True, exist_ok=True)

    def _line_count(path: Path) -> int:
        if not path.exists():
            return 0
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())

    # Some local copies only ship ontology.embeddings.  If the dictionaries are
    # missing or tiny placeholders, rebuild label;IRI dictionaries from the
    # embedding vocabulary instead of creating minimal files.
    if _line_count(hp_dict) < 100 or _line_count(ordo_dict) < 100:
        from gensim.models import KeyedVectors

        kv = KeyedVectors.load(str(emb_path))
        keys = [str(key) for key in kv.index_to_key]
        ordo_iris = sorted(
            {key for key in keys if key.startswith("http://www.orpha.net/ORDO/Orphanet_")},
            key=lambda iri: int(iri.rsplit("_", 1)[-1]) if iri.rsplit("_", 1)[-1].isdigit() else iri,
        )
        hpo_iris = sorted(
            {key for key in keys if key.startswith("http://purl.obolibrary.org/obo/HP_")},
            key=lambda iri: iri.rsplit("/", 1)[-1],
        )

        if not ordo_iris or not hpo_iris:
            raise ValueError(f"Could not rebuild ontology dictionaries from {emb_path}.")

        ordo_dict.write_text(
            "".join(f"{iri.rsplit('/', 1)[-1]};{iri}\n" for iri in ordo_iris),
            encoding="utf-8",
        )
        hp_dict.write_text(
            "".join(f"{iri.rsplit('/', 1)[-1]};{iri}\n" for iri in hpo_iris),
            encoding="utf-8",
        )
        print(f"Rebuilt ORDO.dict ({len(ordo_iris)} entries) and HPO.dict ({len(hpo_iris)} entries).")

    return emb_path, hp_dict, ordo_dict


def build_embedding(emb_dir: Path):
    """Load ontology embeddings and print a sanity check for each diagnosis."""
    from onto_cgans import OntologyEmbedding

    emb_path, hp_dict, ordo_dict = ensure_embedding_dicts(emb_dir)
    embedding = OntologyEmbedding(
        embedding_path=str(emb_path),
        embedding_size=100,
        hp_dict_fn=str(hp_dict),
        rd_dict_fn=str(ordo_dict),
    )

    for label, iri in DX_TO_IRI.items():
        vec = embedding.get_embedding(iri)
        norm = float(np.linalg.norm(vec))
        if norm == 0.0 and label != "CN":
            raise ValueError(f"Embedding lookup returned a zero vector for {label}: {iri}")
        note = DX_TO_IRI_NOTES.get(label, "")
        status = "zero-vector baseline" if norm == 0.0 else "embedding ok"
        print(f"{status}: {label} -> {iri.split('/')[-1]} (norm={norm:.3f}) {note}")

    return embedding


def train_or_load_model(train_df: pd.DataFrame, args: argparse.Namespace):
    """Fit Onto-CGAN+ unless a compatible saved model is already present."""
    from onto_cgans import Onto_DP_CGAN

    model_path = args.out_dir / f"gan_model_adni_{args.mode}.pkl"
    transformer_path = args.out_dir / f"fitted_transformer_adni_{args.mode}.pkl"

    if args.overwrite:
        for path in [model_path, transformer_path]:
            if path.exists():
                path.unlink()

    if model_path.exists():
        print(f"Loading existing model: {model_path}")
        with model_path.open("rb") as fh:
            return pickle.load(fh), model_path

    embedding = build_embedding(args.emb_dir)
    # Decoder conditioning is the default experiment mode. Other modes are kept
    # available because they are implemented by onto_cgan+.
    model = Onto_DP_CGAN(
        log_file_path=None,
        embedding=embedding,
        epochs=args.epochs,
        batch_size=args.batch_size,
        noise_dim=128,
        generator_dim=(256, 256, 256),
        discriminator_dim=(256, 256, 256),
        discriminator_steps=10,
        generator_lr=2e-4,
        generator_decay=1e-6,
        discriminator_lr=2e-4,
        discriminator_decay=1e-6,
        log_frequency=True,
        verbose=True,
        private=False,
        cuda=not args.no_cuda,
        wandb=False,
        conditioning_mode=args.mode,
        saved_transformer=str(transformer_path),
    )
    model.fit(train_df)

    with model_path.open("wb") as fh:
        pickle.dump(model, fh)
    print(f"Saved model: {model_path}")
    return model, model_path


def write_run_artifacts(train_df: pd.DataFrame, synth_df: pd.DataFrame | None, args: argparse.Namespace) -> None:
    """Save the files needed to reproduce and audit a generation run."""
    train_path = args.out_dir / "adni_cgan_train.csv"
    train_df.to_csv(train_path, index=False)
    print(f"Saved training CSV: {train_path}")

    numeric_cols = [col for col in train_df.columns if col not in {"IRI", "label", "gender"}]
    missing_report = train_df[numeric_cols].isna().mean().rename("missing_fraction").reset_index()
    missing_report = missing_report.rename(columns={"index": "feature"})
    missing_report.to_csv(args.out_dir / "adni_missingness_after_imputation.csv", index=False)

    summary = {
        "n_train_rows": int(len(train_df)),
        "columns": list(train_df.columns),
        "label_counts": train_df["label"].value_counts().to_dict(),
        "gender_counts": train_df["gender"].value_counts().to_dict(),
        "feature_set": args.feature_set,
        "seed": int(args.seed),
        "diagnosis_iri_mapping": DX_TO_IRI,
        "diagnosis_iri_notes": DX_TO_IRI_NOTES,
        "mode": args.mode,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "synthetic_numeric_clipped": not args.no_clip,
    }
    if synth_df is not None:
        summary["n_synthetic_rows"] = int(len(synth_df))
        summary["synthetic_label_counts"] = synth_df["label"].value_counts().to_dict()

        rows = []
        for col in numeric_cols:
            if col not in synth_df.columns:
                continue
            rows.append(
                {
                    "feature": col,
                    "real_mean": train_df[col].mean(),
                    "synthetic_mean": synth_df[col].mean(),
                    "real_std": train_df[col].std(),
                    "synthetic_std": synth_df[col].std(),
                }
            )
        pd.DataFrame(rows).to_csv(args.out_dir / "real_vs_synthetic_summary.csv", index=False)

    (args.out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def postprocess_synthetic(synth_df: pd.DataFrame, train_df: pd.DataFrame, clip: bool = True) -> pd.DataFrame:
    """Clean generated rows before causal analysis.

    The generator may emit numeric values just outside the empirical ADNI range.
    Clipping keeps the synthetic data inside observed support without changing
    the learned graph code downstream.
    """
    synth_df = synth_df.copy()
    numeric_cols = [col for col in train_df.columns if col not in {"IRI", "label", "gender"}]

    # Keep synthetic values inside the support observed in the training split.
    # The causal benchmark still tests whether dependencies were preserved.
    for col in numeric_cols:
        if col not in synth_df.columns:
            continue
        synth_df[col] = pd.to_numeric(synth_df[col], errors="coerce")
        if clip:
            synth_df[col] = synth_df[col].clip(train_df[col].min(), train_df[col].max())
        if col in INTEGER_LIKE_FEATURES:
            synth_df[col] = synth_df[col].round()

    return synth_df


def main() -> None:
    """Command-line entry point for preprocessing, training, and sampling."""
    args = _parse_args()
    if args.quick:
        args.epochs = 2
        args.batch_size = 50
        args.n_samples = 100 if args.n_samples is None else args.n_samples
        args.limit_rows = 300 if args.limit_rows is None else args.limit_rows

    if args.out_dir is None:
        if args.quick:
            args.out_dir = DEFAULT_OUT_DIR / (
                "quick_smoke" if args.feature_set == "main" else "quick_smoke_expanded"
            )
        elif args.feature_set == "expanded":
            args.out_dir = DEFAULT_OUT_DIR / "expanded"
        else:
            args.out_dir = DEFAULT_OUT_DIR

    if args.batch_size % 2 != 0 or args.batch_size % 10 != 0:
        raise ValueError("--batch-size must be even and divisible by 10 for the default pac=10 discriminator.")

    set_global_seed(args.seed, use_cuda=not args.no_cuda)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare baseline ADNI, train Onto-CGAN, sample rows, then write artifacts.
    train_df = prepare_adni(args)
    print(f"Prepared ADNI CGAN data: {train_df.shape[0]} rows x {train_df.shape[1]} columns")
    print("Label counts:")
    print(train_df["label"].value_counts().to_string())

    if args.prepare_only:
        write_run_artifacts(train_df, None, args)
        return

    n_samples = args.n_samples or len(train_df)
    model, _ = train_or_load_model(train_df, args)
    synth_df = model.sample(n_samples)
    synth_df = postprocess_synthetic(synth_df, train_df, clip=not args.no_clip)

    synth_path = args.out_dir / f"synthetic_adni_{args.mode}.csv"
    synth_df.to_csv(synth_path, index=False)
    print(f"Saved synthetic CSV: {synth_path}")
    print("Synthetic label counts:")
    print(synth_df["label"].value_counts().to_string())

    write_run_artifacts(train_df, synth_df, args)


if __name__ == "__main__":
    main()

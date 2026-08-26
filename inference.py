#!/usr/bin/env python3
"""
inference.py — AI Usage Patterns & Prompt Quality Analysis
Production SageMaker Processing script.

Loads a pre-trained prompt-efficiency classifier from a configurable model
path, scans every CSV in the input directory, classifies each CSV by its
column signature into one of four known sources (OASST / LMSYS / ChatGPT /
student survey), and writes four Parquet tables to the output directory:

    unified_prompt_efficiency.parquet
    oasst_user_dependency.parquet
    lmsys_user_dependency.parquet
    survey_dependency.parquet

Design notes carried over from the source notebook (see project history):
- oasst.user_id, lmsys.judge_user_id, and survey.Student_Name are three
  disjoint, unrelated populations. No cross-source "unified user_id" is
  ever created. The dependency tables are independent and are never
  joined to one another.
- ChatGPT prompts have no user identity and therefore only ever appear in
  unified_prompt_efficiency, never in a dependency table.
- The classifier is trained elsewhere (a SageMaker Training job) on
  text-only features to avoid leakage from the columns used to build
  ground-truth labels. This script only loads and applies that model —
  it does not train.

Run non-interactively, e.g. as the entry_point of an SKLearnProcessor:

    python inference.py \\
        --input-dir /opt/ml/processing/input \\
        --output-dir /opt/ml/processing/output \\
        --model-dir /opt/ml/processing/model \\
        --model-filename model.joblib

Exit codes:
    0  — completed; at least one output table was written
    1  — fatal error (bad args, no CSVs found, all tables failed, etc.)
"""

from __future__ import annotations

import argparse
import logging
import re
import string
import sys
import traceback
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Logging — plain stdout/stderr so CloudWatch Logs captures it without any
# extra handler configuration inside the processing container.
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("ai_usage_patterns.inference")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
FEATURE_COLS = [
    "has_question_mark", "has_code_keywords", "has_please", "avg_word_length",
    "unique_word_ratio", "punct_density", "capital_ratio", "starts_with_wh",
]

CODE_KEYWORDS = {
    "code", "python", "function", "class", "import", "def", "javascript",
    "java", "c++", "sql", "html", "css", "api", "algorithm", "bug", "error",
    "compile", "debug", "script", "variable", "array", "loop", "syntax",
}
WH_WORDS = ("what", "why", "how", "when", "where", "who", "which", "whom")

# Expected schemas, used to auto-classify each CSV found in --input-dir.
# Filename hints are tried first (cheap, unambiguous); column-signature
# matching is the fallback for arbitrarily-named files.
SOURCE_SCHEMAS = {
    "oasst": {
        "required_columns": {
            "message_id", "user_id", "text", "Word_Count", "Prompts_In_Tree",
            "Is_Repeated_Prompt",
        },
        "filename_hints": ("oasst",),
    },
    "lmsys": {
        "required_columns": {
            "question_id", "judge_user_id", "prompt_text", "turn_number",
            "total_turns_in_battle", "Total_Prompts_By_User",
        },
        "filename_hints": ("lmsys",),
    },
    "chatgpt": {
        "required_columns": {"act", "prompt", "Prompt_Length", "Prompt_Category"},
        "filename_hints": ("chatgpt",),
    },
    "survey": {
        "required_columns": {
            "Student_Name", "Daily_Usage_Hours", "Trust_in_AI_Tools",
            "Usage_Bucket", "Academic_Use_Flag", "Num_AI_Tools_Used",
        },
        "filename_hints": ("survey", "student"),
    },
}

USAGE_BUCKET_RANK = {"Low (<1h)": 0, "Medium (1-3h)": 1, "High (3h+)": 2}

W_OASST = {"volume": 0.30, "repeat": 0.25, "turns": 0.25, "low_eff": 0.20}
W_LMSYS = {"volume": 0.40, "turns": 0.30, "low_eff": 0.30}
W_SURVEY = {"hours": 0.30, "trust": 0.20, "academic": 0.20, "tools": 0.15, "bucket": 0.15}


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI usage patterns / prompt quality processing job")
    parser.add_argument(
        "--input-dir", type=str, default="/opt/ml/processing/input",
        help="Directory containing input CSVs (scanned recursively).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="/opt/ml/processing/output",
        help="Directory to write output Parquet tables to.",
    )
    parser.add_argument(
        "--model-dir", type=str, default="/opt/ml/processing/model",
        help="Directory containing the pre-trained classifier artifact.",
    )
    parser.add_argument(
        "--model-filename", type=str, default="model.joblib",
        help="Filename of the serialized classifier inside --model-dir.",
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# Text feature extraction (must exactly match the features the model was
# trained on — order and definitions are load-bearing).
# --------------------------------------------------------------------------- #
def extract_text_features(texts: pd.Series) -> pd.DataFrame:
    texts = texts.fillna("").astype(str)

    def per_text(t: str) -> pd.Series:
        words = t.split()
        n_words = max(len(words), 1)
        n_chars = max(len(t), 1)
        lower = t.lower()
        letters = [c for c in t if c.isalpha()]
        n_letters = max(len(letters), 1)
        punct_count = sum(1 for c in t if c in string.punctuation)
        cap_count = sum(1 for c in letters if c.isupper())
        unique_words = {w.lower().strip(string.punctuation) for w in words}

        return pd.Series({
            "has_question_mark": int("?" in t),
            "has_code_keywords": int(any(kw in lower for kw in CODE_KEYWORDS)),
            "has_please": int("please" in lower),
            "avg_word_length": sum(len(w) for w in words) / n_words if words else 0.0,
            "unique_word_ratio": len(unique_words) / n_words if words else 0.0,
            "punct_density": punct_count / n_chars,
            "capital_ratio": cap_count / n_letters,
            "starts_with_wh": int(lower.strip().startswith(WH_WORDS)),
        })

    return texts.apply(per_text)


# --------------------------------------------------------------------------- #
# CSV discovery / classification
# --------------------------------------------------------------------------- #
def classify_csv(path: Path, columns: set[str]) -> Optional[str]:
    """Identify which known source a CSV belongs to, by filename hint first,
    falling back to column-signature match. Returns None if unrecognized."""
    name_lower = path.name.lower()
    for source, schema in SOURCE_SCHEMAS.items():
        if any(hint in name_lower for hint in schema["filename_hints"]):
            if schema["required_columns"].issubset(columns):
                return source
            logger.warning(
                "%s: filename suggests source '%s' but required columns are missing (%s) — "
                "falling back to column-signature matching.",
                path.name, source, schema["required_columns"] - columns,
            )
    for source, schema in SOURCE_SCHEMAS.items():
        if schema["required_columns"].issubset(columns):
            return source
    return None


def discover_sources(input_dir: Path) -> dict[str, pd.DataFrame]:
    """Scan input_dir recursively for CSVs, classify each, and concatenate
    same-source files. Unrecognized CSVs are logged and skipped."""
    csv_paths = sorted(input_dir.rglob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under {input_dir}")

    logger.info("Found %d CSV file(s) under %s", len(csv_paths), input_dir)

    frames: dict[str, list[pd.DataFrame]] = {}
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
        except Exception:
            logger.error("Failed to read %s — skipping.\n%s", path, traceback.format_exc())
            continue

        source = classify_csv(path, set(df.columns))
        if source is None:
            logger.warning("Could not classify %s (columns=%s) — skipping.", path.name, list(df.columns))
            continue

        logger.info("Classified %s as source='%s' (%d rows)", path.name, source, len(df))
        frames.setdefault(source, []).append(df)

    combined = {
        source: (pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0])
        for source, dfs in frames.items()
    }
    for source, df in combined.items():
        logger.info("Source '%s': %d total row(s) after combining files", source, len(df))
    return combined


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_model(model_dir: Path, model_filename: str):
    model_path = model_dir / model_filename
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}")
    logger.info("Loading model from %s", model_path)
    model = joblib.load(model_path)

    n_features_expected = getattr(model, "n_features_in_", None)
    if n_features_expected is not None and n_features_expected != len(FEATURE_COLS):
        logger.warning(
            "Loaded model expects %d feature(s) but this script produces %d (%s). "
            "Predictions may be invalid if the feature set has drifted.",
            n_features_expected, len(FEATURE_COLS), FEATURE_COLS,
        )
    return model


def build_predictions(model, texts: pd.Series, source: str, prefix: str) -> pd.DataFrame:
    feats = extract_text_features(texts)[FEATURE_COLS]
    proba = model.predict_proba(feats)
    pred = np.asarray(model.classes_)[np.argmax(proba, axis=1)]
    confidence = proba.max(axis=1)
    return pd.DataFrame({
        "prompt_key": [f"{prefix}_{i}" for i in texts.index],
        "source": source,
        "text": texts.values,
        "predicted_label": pred,
        "confidence_score": confidence,
    })


# --------------------------------------------------------------------------- #
# Scoring helpers
# --------------------------------------------------------------------------- #
def minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi - lo == 0:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


def tier_from_score(score: pd.Series) -> pd.Series:
    bins = score.quantile([0, 1 / 3, 2 / 3, 1]).to_numpy().copy()
    bins[0] -= 1e-9
    return pd.cut(score, bins=bins, labels=["Low", "Moderate", "High"])


def to_snake(col: str) -> str:
    # Inputs are already underscore-separated (e.g. "Trust_in_AI_Tools");
    # just lowercase and collapse repeated underscores rather than inserting
    # new ones before every capital letter (that would split acronyms like
    # "AI" into "a_i" and double up existing underscores).
    return re.sub(r"_+", "_", col).lower()


def snake_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [to_snake(c) for c in out.columns]
    return out


# --------------------------------------------------------------------------- #
# Dependency table builders
# --------------------------------------------------------------------------- #
def build_oasst_dependency(oasst: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    df = oasst[["message_id", "user_id", "Prompts_In_Tree", "Is_Repeated_Prompt"]].copy()
    df["predicted_label"] = predictions["predicted_label"].values

    grouped = df.groupby("user_id").agg(
        prompt_volume=("message_id", "count"),
        repeat_rate=("Is_Repeated_Prompt", "mean"),
        avg_turns=("Prompts_In_Tree", "mean"),
        low_efficiency_rate=("predicted_label", lambda s: (s == "Low").mean()),
    ).reset_index()

    grouped["dependency_score"] = 100 * (
        W_OASST["volume"] * minmax(grouped["prompt_volume"])
        + W_OASST["repeat"] * minmax(grouped["repeat_rate"])
        + W_OASST["turns"] * minmax(grouped["avg_turns"])
        + W_OASST["low_eff"] * minmax(grouped["low_efficiency_rate"])
    )
    grouped["dependency_tier"] = tier_from_score(grouped["dependency_score"])
    return grouped[[
        "user_id", "prompt_volume", "repeat_rate", "avg_turns",
        "low_efficiency_rate", "dependency_score", "dependency_tier",
    ]]


def build_lmsys_dependency(lmsys: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    df = lmsys[["question_id", "judge_user_id", "Total_Prompts_By_User", "total_turns_in_battle"]].copy()
    df["predicted_label"] = predictions["predicted_label"].values

    grouped = df.groupby("judge_user_id").agg(
        prompt_volume=("Total_Prompts_By_User", "max"),
        avg_turns=("total_turns_in_battle", "mean"),
        low_efficiency_rate=("predicted_label", lambda s: (s == "Low").mean()),
    ).reset_index()

    grouped["dependency_score"] = 100 * (
        W_LMSYS["volume"] * minmax(grouped["prompt_volume"])
        + W_LMSYS["turns"] * minmax(grouped["avg_turns"])
        + W_LMSYS["low_eff"] * minmax(grouped["low_efficiency_rate"])
    )
    grouped["dependency_tier"] = tier_from_score(grouped["dependency_score"])
    grouped["repeat_rate"] = None  # not available in this source — not fabricated
    return grouped[[
        "judge_user_id", "prompt_volume", "avg_turns", "repeat_rate",
        "low_efficiency_rate", "dependency_score", "dependency_tier",
    ]]


def build_survey_dependency(survey: pd.DataFrame) -> pd.DataFrame:
    df = survey[[
        "Student_Name", "Daily_Usage_Hours", "Trust_in_AI_Tools", "Usage_Bucket",
        "Academic_Use_Flag", "Num_AI_Tools_Used",
    ]].copy()

    found_buckets = set(df["Usage_Bucket"].unique())
    unknown = found_buckets - set(USAGE_BUCKET_RANK)
    if unknown:
        raise ValueError(f"Unrecognized Usage_Bucket value(s): {unknown}")

    hours_n = minmax(df["Daily_Usage_Hours"])
    trust_n = minmax(df["Trust_in_AI_Tools"])
    tools_n = minmax(df["Num_AI_Tools_Used"])
    academic_n = df["Academic_Use_Flag"].astype(int)
    bucket_n = minmax(df["Usage_Bucket"].map(USAGE_BUCKET_RANK))

    df["dependency_score"] = 100 * (
        W_SURVEY["hours"] * hours_n
        + W_SURVEY["trust"] * trust_n
        + W_SURVEY["academic"] * academic_n
        + W_SURVEY["tools"] * tools_n
        + W_SURVEY["bucket"] * bucket_n
    )
    df["dependency_tier"] = tier_from_score(df["dependency_score"])
    return df


# --------------------------------------------------------------------------- #
# Output writing
# --------------------------------------------------------------------------- #
def write_table(df: pd.DataFrame, output_dir: Path, name: str) -> None:
    out = snake_columns(df)
    path = output_dir / f"{name}.parquet"
    out.to_parquet(path, index=False, engine="pyarrow")
    logger.info("Wrote %s (%d rows, %d cols)", path, len(out), len(out.columns))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)

    logger.info("Starting AI usage patterns processing job")
    logger.info("input_dir=%s output_dir=%s model_dir=%s model_filename=%s",
                input_dir, output_dir, model_dir, args.model_filename)

    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        sources = discover_sources(input_dir)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1

    if not sources:
        logger.error("No recognized source files found — nothing to process.")
        return 1

    prompt_sources = {k: v for k, v in sources.items() if k in ("oasst", "lmsys", "chatgpt")}
    survey_df = sources.get("survey")

    model = None
    if prompt_sources:
        try:
            model = load_model(model_dir, args.model_filename)
        except FileNotFoundError as exc:
            logger.error(
                "%s — unified_prompt_efficiency and any dependent tables (oasst/lmsys) "
                "cannot be produced without it.", exc,
            )

    tables_written = []
    tables_failed = []

    predictions_by_source: dict[str, pd.DataFrame] = {}
    if model is not None:
        text_col = {"oasst": "text", "lmsys": "prompt_text", "chatgpt": "prompt"}
        unified_parts = []
        for source, df in prompt_sources.items():
            try:
                preds = build_predictions(model, df[text_col[source]], source, source)
                predictions_by_source[source] = preds
                unified_parts.append(preds)
                logger.info("Scored %d prompt(s) from source '%s'", len(preds), source)
            except Exception:
                logger.error("Failed to score source '%s':\n%s", source, traceback.format_exc())
                tables_failed.append(f"predictions:{source}")

        if unified_parts:
            try:
                unified = pd.concat(unified_parts, ignore_index=True)
                write_table(unified, output_dir, "unified_prompt_efficiency")
                tables_written.append("unified_prompt_efficiency")
            except Exception:
                logger.error("Failed to write unified_prompt_efficiency:\n%s", traceback.format_exc())
                tables_failed.append("unified_prompt_efficiency")
        else:
            logger.error("No prompt sources were successfully scored — skipping unified_prompt_efficiency.")
            tables_failed.append("unified_prompt_efficiency")

    if "oasst" in prompt_sources and "oasst" in predictions_by_source:
        try:
            oasst_dep = build_oasst_dependency(prompt_sources["oasst"], predictions_by_source["oasst"])
            write_table(oasst_dep, output_dir, "oasst_user_dependency")
            tables_written.append("oasst_user_dependency")
        except Exception:
            logger.error("Failed to build oasst_user_dependency:\n%s", traceback.format_exc())
            tables_failed.append("oasst_user_dependency")
    elif "oasst" in prompt_sources:
        logger.warning("OASST data present but not scored — skipping oasst_user_dependency.")
        tables_failed.append("oasst_user_dependency")

    if "lmsys" in prompt_sources and "lmsys" in predictions_by_source:
        try:
            lmsys_dep = build_lmsys_dependency(prompt_sources["lmsys"], predictions_by_source["lmsys"])
            write_table(lmsys_dep, output_dir, "lmsys_user_dependency")
            tables_written.append("lmsys_user_dependency")
        except Exception:
            logger.error("Failed to build lmsys_user_dependency:\n%s", traceback.format_exc())
            tables_failed.append("lmsys_user_dependency")
    elif "lmsys" in prompt_sources:
        logger.warning("LMSYS data present but not scored — skipping lmsys_user_dependency.")
        tables_failed.append("lmsys_user_dependency")

    if survey_df is not None:
        try:
            survey_dep = build_survey_dependency(survey_df)
            write_table(survey_dep, output_dir, "survey_dependency")
            tables_written.append("survey_dependency")
        except Exception:
            logger.error("Failed to build survey_dependency:\n%s", traceback.format_exc())
            tables_failed.append("survey_dependency")

    logger.info("Job summary: %d table(s) written (%s), %d failed/skipped (%s)",
                len(tables_written), tables_written, len(tables_failed), tables_failed)

    if not tables_written:
        logger.error("No output tables were produced — failing the job.")
        return 1

    logger.info("Processing job completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

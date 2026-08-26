#!/usr/bin/env python
# coding: utf-8

# # AI Usage Patterns & Prompt Quality Analysis
# 
# This notebook builds two deliverables from four heterogeneous prompt/survey
# datasets, and writes the outputs as Parquet so they can be queried with
# **AWS Athena**.
# 
# **Sources (all independently sourced, verified against the repo before use):**
# 1. `oasst_prompts_clean.csv` (15,708 rows) — OASST prompts, real `user_id`,
#    behavioral columns (`Word_Count`, `Prompts_In_Tree`, `Is_Repeated_Prompt`).
#    The only source with everything needed to build rule-based ground-truth labels.
# 2. `lmsys_prompts_clean.csv` (39,316 rows) — LMSYS arena prompts, has
#    `judge_user_id` and turn/volume info, but different schema/semantics than OASST.
# 3. `chatgpt_prompts_clean.csv` (153 rows) — prompt text + category only, **no
#    user identity, no turn data**.
# 4. `student_survey_clean.csv` (3,614 rows) — self-reported survey, one row
#    per student, not per prompt.
# 
# ## Hard constraint: no fabricated cross-source identity
# `oasst.user_id`, `lmsys.judge_user_id`, and `survey.Student_Name` are three
# **disjoint, unrelated populations** — there is no key linking a person across
# them. This notebook never creates a "unified user_id." The only surrogate key
# used is `prompt_key` (`f"{source}_{row_index}"`), which standardizes an ID for
# a prompt *row* — it does not claim identity linkage across sources.
# 
# Dependency scores are therefore built and kept in **three separate tables**:
# `oasst_user_dependency`, `lmsys_user_dependency`, `survey_dependency`. ChatGPT
# has no user identity at all, so it never appears in a dependency table — only
# in the unified prompt-efficiency table.

# In[1]:


import re
import string
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

pd.set_option("display.max_columns", 50)
np.random.seed(42)

RAW_BASE = (
    "https://raw.githubusercontent.com/Ananya5105/"
    "AI-Usage-Patterns-Prompt-Quality-Analysis/main/clean_data"
)
OUT_DIR = Path("athena_data")
OUT_DIR.mkdir(exist_ok=True)

# ## Load data
# 
# Loaded directly from the repo's `clean_data/` folder so the notebook is
# reproducible from source.

# In[2]:


oasst = pd.read_csv("clean_data/oasst_prompts_clean.csv")
lmsys = pd.read_csv("clean_data/lmsys_prompts_clean.csv")
chatgpt = pd.read_csv("clean_data/chatgpt_prompts_clean.csv")
survey = pd.read_csv("clean_data/student_survey_clean.csv")

print("oasst  ", oasst.shape)
print("lmsys  ", lmsys.shape)
print("chatgpt", chatgpt.shape)
print("survey ", survey.shape)

# ## Schema verification
# 
# Verified against the live repo before writing any logic below. Asserting
# here so the notebook fails loudly (rather than silently mis-joining) if the
# upstream files ever change.

# In[3]:


EXPECTED = {
    "oasst": [
        "message_id", "parent_id", "user_id", "created_date", "text", "role",
        "lang", "deleted", "rank", "message_tree_id", "tree_state", "split",
        "Word_Count", "Char_Count", "Prompts_In_Tree", "Repeat_Count",
        "Is_Repeated_Prompt",
    ],
    "lmsys": [
        "question_id", "judge_user_id", "turn_number", "total_turns_in_battle",
        "prompt_text", "language", "tstamp", "model_a", "model_b", "winner",
        "flagged_moderation", "datetime", "hour_of_day", "date", "weekday",
        "Word_Count", "Total_Prompts_By_User",
    ],
    "chatgpt": ["act", "prompt", "Prompt_Length", "Prompt_Category"],
    "survey": [
        "Student_Name", "College_Name", "Stream", "Year_of_Study",
        "AI_Tools_Used", "Daily_Usage_Hours", "Use_Cases", "Trust_in_AI_Tools",
        "Impact_on_Grades", "Do_Professors_Allow_Use", "Preferred_AI_Tool",
        "Awareness_Level", "Willing_to_Pay_for_Access", "State", "Device_Used",
        "Internet_Access", "Primary_AI_Tool", "Num_AI_Tools_Used",
        "Academic_Use_Flag", "Usage_Bucket",
    ],
}

for name, df in [("oasst", oasst), ("lmsys", lmsys), ("chatgpt", chatgpt), ("survey", survey)]:
    missing = set(EXPECTED[name]) - set(df.columns)
    extra = set(df.columns) - set(EXPECTED[name])
    assert not missing, f"{name}: missing expected columns {missing}"
    if extra:
        print(f"{name}: extra columns not in spec (informational only): {extra}")
print("All schemas verified OK.")

# ## Part 1 — Unified prompt efficiency classifier
# 
# ### Step 1: source-agnostic text features
# 
# These eight features are derived purely from prompt *text*, so a model
# trained on them generalizes to any source — OASST, LMSYS, or ChatGPT —
# even though only OASST has the behavioral columns needed for labeling.

# In[4]:


CODE_KEYWORDS = {
    "code", "python", "function", "class", "import", "def", "javascript",
    "java", "c++", "sql", "html", "css", "api", "algorithm", "bug", "error",
    "compile", "debug", "script", "variable", "array", "loop", "syntax",
}
WH_WORDS = ("what", "why", "how", "when", "where", "who", "which", "whom")


def extract_text_features(texts: pd.Series) -> pd.DataFrame:
    texts = texts.fillna("").astype(str)

    def per_text(t):
        words = t.split()
        n_words = max(len(words), 1)
        n_chars = max(len(t), 1)
        lower = t.lower()
        letters = [c for c in t if c.isalpha()]
        n_letters = max(len(letters), 1)
        punct_count = sum(1 for c in t if c in string.punctuation)
        cap_count = sum(1 for c in letters if c.isupper())
        unique_words = set(w.lower().strip(string.punctuation) for w in words)

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


FEATURE_COLS = [
    "has_question_mark", "has_code_keywords", "has_please", "avg_word_length",
    "unique_word_ratio", "punct_density", "capital_ratio", "starts_with_wh",
]

# ### Step 2: rule-based ground-truth labels (OASST only)
# 
# Labels are built **only** from OASST's behavioral columns
# (`Word_Count`, `Prompts_In_Tree`, `Is_Repeated_Prompt`) — never from the
# text features above, to avoid circularity.
# 
# Rule (documented, tunable):
# - `Is_Repeated_Prompt == True` → automatically penalized (repeating an
#   identical prompt is a sign of low-effort/inefficient prompting).
# - Longer, more detailed prompts (`Word_Count`) → higher efficiency, capped
#   at the 95th percentile to limit outlier influence.
# - Prompts needing a large `Prompts_In_Tree` (many follow-ups to resolve) →
#   treated as *less* efficient, since more back-and-forth was needed.
# - These are combined into a single 0-1 composite score, then bucketed into
#   equal-sized tertiles → Low / Moderate / High.

# In[5]:


wc_cap = oasst["Word_Count"].quantile(0.95)
wc_norm = (oasst["Word_Count"].clip(upper=wc_cap) / wc_cap).clip(0, 1)

tree_cap = oasst["Prompts_In_Tree"].quantile(0.95)
tree_norm = (oasst["Prompts_In_Tree"].clip(upper=tree_cap) / tree_cap).clip(0, 1)

repeat_penalty = oasst["Is_Repeated_Prompt"].astype(int) * 0.5

composite_score = (0.5 * wc_norm + 0.5 * (1 - tree_norm)) - repeat_penalty
composite_score = composite_score.clip(0, 1)

oasst["_composite_score"] = composite_score
label_bins = oasst["_composite_score"].quantile([0, 1 / 3, 2 / 3, 1]).to_numpy().copy()
label_bins[0] -= 1e-9  # ensure the minimum value is included in the first bin
oasst["efficiency_label"] = pd.cut(
    oasst["_composite_score"], bins=label_bins, labels=["Low", "Moderate", "High"]
)

print(oasst["efficiency_label"].value_counts())

# ### Step 3: train RandomForestClassifier on text features only
# 
# `Word_Count`, `Char_Count`, `Prompts_In_Tree`, `Is_Repeated_Prompt` (or any
# near-proxy such as raw character count) are deliberately **excluded** as
# model features — they built the labels, so including them would be data
# leakage and the model would just memorize the labeling rule instead of
# learning generalizable text signal.

# In[6]:


X = extract_text_features(oasst["text"])
y = oasst["efficiency_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight="balanced")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# In[7]:


import joblib

joblib.dump(clf, "model.joblib")
print("Saved model.joblib")

# In[8]:


import joblib

test_model = joblib.load("model.joblib")

print(type(test_model))
print("Features:", test_model.n_features_in_)

# ### Step 4: feature importance sanity check
# 
# Confirms no single feature dominates (a red flag for leakage or a
# degenerate rule). Threshold: ~60%+ on one feature is worth investigating.

# In[9]:


importances = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
print(importances)

top_importance = importances.iloc[0]
if top_importance >= 0.60:
    print(f"\nWARNING: '{importances.index[0]}' accounts for {top_importance:.1%} of importance — investigate for leakage.")
else:
    print(f"\nOK: top feature ('{importances.index[0]}') accounts for {top_importance:.1%} — no single-feature dominance.")

# ### Step 5: apply the trained model to all three prompt sources
# 
# Because the model only ever saw text-derived features, it can be applied
# to LMSYS and ChatGPT prompts even though they lack OASST's behavioral
# columns.

# In[10]:


def build_predictions(texts: pd.Series, source: str, prefix: str) -> pd.DataFrame:
    feats = extract_text_features(texts)
    proba = clf.predict_proba(feats)
    pred = clf.classes_[np.argmax(proba, axis=1)]
    confidence = proba.max(axis=1)
    return pd.DataFrame({
        "prompt_key": [f"{prefix}_{i}" for i in texts.index],
        "source": source,
        "text": texts.values,
        "predicted_label": pred,
        "confidence_score": confidence,
    })


oasst_preds = build_predictions(oasst["text"], "oasst", "oasst")
lmsys_preds = build_predictions(lmsys["prompt_text"], "lmsys", "lmsys")
chatgpt_preds = build_predictions(chatgpt["prompt"], "chatgpt", "chatgpt")

unified_prompt_efficiency = pd.concat(
    [oasst_preds, lmsys_preds, chatgpt_preds], ignore_index=True
)
print(unified_prompt_efficiency["source"].value_counts())
print(unified_prompt_efficiency["predicted_label"].value_counts())
unified_prompt_efficiency.head()

# ## Part 2 — Dependency scores (kept separate per source, never merged)
# 
# Each table uses its own available columns. Weights are documented and
# tunable; each component is min-max normalized to 0-1 before blending into
# a 0-100 score, then bucketed into Low / Moderate / High tiers by terciles.

# In[11]:


def minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi - lo == 0:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


def tier_from_score(score: pd.Series) -> pd.Series:
    bins = score.quantile([0, 1 / 3, 2 / 3, 1]).to_numpy().copy()
    bins[0] -= 1e-9
    return pd.cut(score, bins=bins, labels=["Low", "Moderate", "High"])

# ### `oasst_user_dependency`
# 
# Grouped by real `user_id`. Weights (tunable): volume 30 / repeat_rate 25 /
# avg_turns 25 / low_efficiency_rate 20.

# In[12]:


oasst_with_pred = oasst[["message_id", "user_id", "Word_Count", "Prompts_In_Tree", "Is_Repeated_Prompt"]].copy()
oasst_with_pred["predicted_label"] = oasst_preds["predicted_label"].values

oasst_user_dependency = oasst_with_pred.groupby("user_id").agg(
    prompt_volume=("message_id", "count"),
    repeat_rate=("Is_Repeated_Prompt", "mean"),
    avg_turns=("Prompts_In_Tree", "mean"),
    low_efficiency_rate=("predicted_label", lambda s: (s == "Low").mean()),
).reset_index()

oasst_user_dependency["volume_n"] = minmax(oasst_user_dependency["prompt_volume"])
oasst_user_dependency["repeat_n"] = minmax(oasst_user_dependency["repeat_rate"])
oasst_user_dependency["turns_n"] = minmax(oasst_user_dependency["avg_turns"])
oasst_user_dependency["low_eff_n"] = minmax(oasst_user_dependency["low_efficiency_rate"])

W_OASST = {"volume": 0.30, "repeat": 0.25, "turns": 0.25, "low_eff": 0.20}
oasst_user_dependency["dependency_score"] = 100 * (
    W_OASST["volume"] * oasst_user_dependency["volume_n"]
    + W_OASST["repeat"] * oasst_user_dependency["repeat_n"]
    + W_OASST["turns"] * oasst_user_dependency["turns_n"]
    + W_OASST["low_eff"] * oasst_user_dependency["low_eff_n"]
)
oasst_user_dependency["dependency_tier"] = tier_from_score(oasst_user_dependency["dependency_score"])

oasst_user_dependency = oasst_user_dependency[[
    "user_id", "prompt_volume", "repeat_rate", "avg_turns", "low_efficiency_rate",
    "dependency_score", "dependency_tier",
]]
print(oasst_user_dependency["dependency_tier"].value_counts())
oasst_user_dependency.head()

# ### `lmsys_user_dependency`
# 
# Grouped by `judge_user_id`. **No `repeat_rate` component** — LMSYS has no
# repeat-prompt semantics, so it is omitted rather than faked. Weights
# redistributed across the remaining three components (tunable):
# volume 40 / avg_turns 30 / low_efficiency_rate 30.

# In[13]:


lmsys_with_pred = lmsys[["question_id", "judge_user_id", "Total_Prompts_By_User", "turn_number", "total_turns_in_battle"]].copy()
lmsys_with_pred["predicted_label"] = lmsys_preds["predicted_label"].values

lmsys_user_dependency = lmsys_with_pred.groupby("judge_user_id").agg(
    prompt_volume=("Total_Prompts_By_User", "max"),
    avg_turns=("total_turns_in_battle", "mean"),
    low_efficiency_rate=("predicted_label", lambda s: (s == "Low").mean()),
).reset_index()

lmsys_user_dependency["volume_n"] = minmax(lmsys_user_dependency["prompt_volume"])
lmsys_user_dependency["turns_n"] = minmax(lmsys_user_dependency["avg_turns"])
lmsys_user_dependency["low_eff_n"] = minmax(lmsys_user_dependency["low_efficiency_rate"])

W_LMSYS = {"volume": 0.40, "turns": 0.30, "low_eff": 0.30}
lmsys_user_dependency["dependency_score"] = 100 * (
    W_LMSYS["volume"] * lmsys_user_dependency["volume_n"]
    + W_LMSYS["turns"] * lmsys_user_dependency["turns_n"]
    + W_LMSYS["low_eff"] * lmsys_user_dependency["low_eff_n"]
)
lmsys_user_dependency["dependency_tier"] = tier_from_score(lmsys_user_dependency["dependency_score"])
lmsys_user_dependency["repeat_rate"] = None  # explicitly absent in this source, not fabricated

lmsys_user_dependency = lmsys_user_dependency[[
    "judge_user_id", "prompt_volume", "avg_turns", "repeat_rate", "low_efficiency_rate",
    "dependency_score", "dependency_tier",
]]
print(lmsys_user_dependency["dependency_tier"].value_counts())
lmsys_user_dependency.head()

# ### `survey_dependency`
# 
# Built directly from the survey's own self-reported columns — real ground
# truth, no model needed. A different grain (one row per student) and no
# identity overlap with the prompt-log sources, so this stays in its own
# table. Weights (tunable): usage hours 30 / trust 20 / academic use 20 /
# number of tools used 15 / usage bucket 15.

# In[14]:


survey_dep = survey[[
    "Student_Name", "Daily_Usage_Hours", "Trust_in_AI_Tools", "Usage_Bucket",
    "Academic_Use_Flag", "Num_AI_Tools_Used",
]].copy()

# NOTE: alphabetical sort would put "High (3h+)" before "Low (<1h)" — wrong.
# Order explicitly by usage intensity instead.
found_buckets = set(survey_dep["Usage_Bucket"].unique())
USAGE_BUCKET_RANK = {"Low (<1h)": 0, "Medium (1-3h)": 1, "High (3h+)": 2}
assert found_buckets == set(USAGE_BUCKET_RANK), (
    f"Usage_Bucket categories changed: found {found_buckets}, expected {set(USAGE_BUCKET_RANK)}"
)
USAGE_BUCKET_ORDER = USAGE_BUCKET_RANK
print("Usage_Bucket categories found:", USAGE_BUCKET_ORDER)

survey_dep["hours_n"] = minmax(survey_dep["Daily_Usage_Hours"])
survey_dep["trust_n"] = minmax(survey_dep["Trust_in_AI_Tools"])
survey_dep["tools_n"] = minmax(survey_dep["Num_AI_Tools_Used"])
survey_dep["academic_n"] = survey_dep["Academic_Use_Flag"].astype(int)
survey_dep["bucket_n"] = minmax(survey_dep["Usage_Bucket"].map(USAGE_BUCKET_ORDER))

W_SURVEY = {"hours": 0.30, "trust": 0.20, "academic": 0.20, "tools": 0.15, "bucket": 0.15}
survey_dep["dependency_score"] = 100 * (
    W_SURVEY["hours"] * survey_dep["hours_n"]
    + W_SURVEY["trust"] * survey_dep["trust_n"]
    + W_SURVEY["academic"] * survey_dep["academic_n"]
    + W_SURVEY["tools"] * survey_dep["tools_n"]
    + W_SURVEY["bucket"] * survey_dep["bucket_n"]
)
survey_dep["dependency_tier"] = tier_from_score(survey_dep["dependency_score"])

survey_dependency = survey_dep[[
    "Student_Name", "Daily_Usage_Hours", "Trust_in_AI_Tools", "Usage_Bucket",
    "Academic_Use_Flag", "Num_AI_Tools_Used", "dependency_score", "dependency_tier",
]]
print(survey_dependency["dependency_tier"].value_counts())
survey_dependency.head()

# ## Write outputs for Athena
# 
# Each table is written to its own Parquet folder (Hive-style, one file per
# table) with lowercase snake_case column names — Athena is case-insensitive
# but this avoids ambiguity. `unified_prompt_efficiency` includes `source` as
# a real column (not a partition) since row counts are small enough that
# partitioning isn't needed; Athena can still filter/GROUP BY on it directly.

# In[15]:


%pip install pyarrow

# In[16]:


def to_snake(col: str) -> str:
    # Columns are already underscore-separated (e.g. "Trust_in_AI_Tools");
    # just lowercase and collapse any incidental repeated underscores rather
    # than inserting new ones before every capital (that would split
    # acronyms like "AI" into "a_i" and double up existing underscores).
    return re.sub(r"_+", "_", col).lower()


tables = {
    "unified_prompt_efficiency": unified_prompt_efficiency,
    "oasst_user_dependency": oasst_user_dependency,
    "lmsys_user_dependency": lmsys_user_dependency,
    "survey_dependency": survey_dependency,
}

for name, df in tables.items():
    out = df.copy()
    out.columns = [to_snake(c) for c in out.columns]
    path = OUT_DIR / f"{name}.parquet"
    out.to_parquet(path, index=False, engine="pyarrow")
    print(f"wrote {path}  ({len(out)} rows, {len(out.columns)} cols)")

# ### Athena DDL (for reference)
# 
# After uploading the four Parquet files under `athena_data/` to S3 (one
# object per table, e.g. `s3://<bucket>/ai_usage/unified_prompt_efficiency/`),
# create external tables, e.g.:
# 
# ```sql
# CREATE EXTERNAL TABLE ai_usage.unified_prompt_efficiency (
#   prompt_key        string,
#   source            string,
#   text              string,
#   predicted_label   string,
#   confidence_score  double
# )
# STORED AS PARQUET
# LOCATION 's3://<bucket>/ai_usage/unified_prompt_efficiency/';
# 
# CREATE EXTERNAL TABLE ai_usage.oasst_user_dependency (
#   user_id               string,
#   prompt_volume         bigint,
#   repeat_rate           double,
#   avg_turns             double,
#   low_efficiency_rate   double,
#   dependency_score      double,
#   dependency_tier       string
# )
# STORED AS PARQUET
# LOCATION 's3://<bucket>/ai_usage/oasst_user_dependency/';
# 
# CREATE EXTERNAL TABLE ai_usage.lmsys_user_dependency (
#   judge_user_id         string,
#   prompt_volume         bigint,
#   avg_turns             double,
#   repeat_rate           string,   -- always NULL: not available in this source
#   low_efficiency_rate   double,
#   dependency_score      double,
#   dependency_tier       string
# )
# STORED AS PARQUET
# LOCATION 's3://<bucket>/ai_usage/lmsys_user_dependency/';
# 
# CREATE EXTERNAL TABLE ai_usage.survey_dependency (
#   student_name        string,
#   daily_usage_hours    double,
#   trust_in_ai_tools    bigint,
#   usage_bucket         string,
#   academic_use_flag    boolean,
#   num_ai_tools_used    bigint,
#   dependency_score     double,
#   dependency_tier      string
# )
# STORED AS PARQUET
# LOCATION 's3://<bucket>/ai_usage/survey_dependency/';
# ```

# ## Summary & caveats
# 
# - **`unified_prompt_efficiency`**: one row per prompt across OASST, LMSYS,
#   and ChatGPT, labeled by a model trained only on text-derived features
#   (no leakage from the columns that built the ground-truth labels).
# - **`oasst_user_dependency`** / **`lmsys_user_dependency`** /
#   **`survey_dependency`**: three separate, never-merged tables — they
#   describe three disjoint populations. Do **not** join them on any
#   surrogate "user" key; none exists.
# - ChatGPT prompts have no user identity and appear **only** in the
#   efficiency table, by design.
# - Labeling rule, dependency weights, and tier cutoffs are documented above
#   and intended to be tuned against domain judgment, not treated as fixed.

# In[ ]:




# In[ ]:




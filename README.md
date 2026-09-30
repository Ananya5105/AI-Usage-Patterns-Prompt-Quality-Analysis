# AI Usage Patterns & Prompt Quality Analysis

Event-driven, cloud-based analytics pipeline on Amazon Web Services (AWS) that measures **how efficiently people prompt AI systems** and **how dependent users are on AI**, across 100K+ conversational AI records from four independent datasets. Upload a CSV and the pipeline classifies it, scores it, writes Parquet, catalogues it, and makes it queryable and ready for dashboards, with no manual steps.

## Stack

Python · Pandas · Scikit-learn · Amazon Simple Storage Service (S3) · AWS Lambda · Amazon SageMaker · AWS Glue · Amazon Athena · Power BI

## Datasets

| Source | Contains | Used for |
|---|---|---|
| LMSYS Chatbot Arena | Prompts, turns, judge user identifiers | Prompt efficiency + user dependency |
| OpenAssistant (OASST1) | Messages, prompt trees, user identifiers | Model training labels + prompt efficiency + user dependency |
| ChatGPT Prompts Dataset | Role-based prompts, no user identity | Prompt efficiency only |
| Student AI Usage Survey | Self-reported usage hours, trust, tools | Survey dependency score |

The three user populations (OASST `user_id`, LMSYS `judge_user_id`, survey `Student_Name`) are unrelated. No cross-source user identifier is created and the dependency tables are never joined.

## What the project produces

### 1. Prompt efficiency classifier
- Random Forest trained on **eight text-only features** (question mark, code keywords, "please", average word length, unique word ratio, punctuation density, capital ratio, starts with a "wh" word).
- Ground-truth labels come from OASST behavioural columns (word count, prompts in tree, repeated prompt). Those columns are **excluded from the features** to avoid data leakage.
- Because the model only sees text, it is applied to OASST, LMSYS and ChatGPT prompts alike.
- Accuracy: about 80% on the held-out set. See `confusion_matrix.png`.

### 2. AI dependency scores (0–100, tiered Low / Moderate / High)
Each source uses only the columns it really has. Every component is min-max normalised, then blended with documented, tunable weights.

| Table | Grain | Weights |
|---|---|---|
| `oasst_user_dependency` | per user | volume 30 · repeat rate 25 · average turns 25 · low-efficiency rate 20 |
| `lmsys_user_dependency` | per judge user | volume 40 · average turns 30 · low-efficiency rate 30 (no repeat component, LMSYS has no repeat semantics) |
| `survey_dependency` | per student | usage hours 30 · trust 20 · academic use 20 · tools 15 · usage bucket 15 |

### 3. Power BI dashboards
1. Executive Overview
2. User Behavior Analytics
3. Prompt Analytics
4. AI Dependency Dashboard
5. Student AI Usage Analytics
6. Recommendations Dashboard

Files: `ananya_ai_usage.pbix` (report) and `ai_usage_analytics_dashboards.pdf` (exported pages).

## Architecture

```
CSV upload
  → Amazon S3 (input bucket)
  → AWS Lambda (triggered by the S3 upload event)
  → Amazon SageMaker Processing job (runs inference.py with the trained model)
  → Parquet tables written to Amazon S3 (output bucket)
  → AWS Glue crawler (detects schema, updates the Data Catalog)
  → Amazon Athena (SQL over the Parquet tables)
  → Power BI dashboards
```

| Stage | What it does |
|---|---|
| Amazon S3 input | Receives the raw CSV. The upload event starts the pipeline. |
| AWS Lambda | Reads the event, then launches the SageMaker Processing job for the new file. |
| Amazon SageMaker | Loads the pre-trained Random Forest model, detects the CSV's source, scores every row, and applies the dependency scoring. |
| Parquet output | Four columnar tables (see below) written back to Amazon S3. |
| AWS Glue crawler | Crawls the output location and creates or updates the table definitions. |
| Amazon Athena | Serverless SQL queries over the tables. |
| Power BI | Six dashboards built on the Athena tables. |

Offline development flow (how the model and logic were built):
`Raw data → Python / Pandas cleaning (notebooks/cleaning.ipynb) → classifier training and scoring (notebooks/ai_usage_patterns_analysis_1.ipynb)`

## Repository layout

```
clean_data/         Cleaned CSVs for all four sources (+ OASST predictions)
notebooks/          cleaning.ipynb, ai_usage_patterns_analysis_1.ipynb
parquets/           The four output tables (Athena-ready)
inference.py        Non-interactive batch script (SageMaker Processing entry point)
lambda/             AWS Lambda function that starts the SageMaker job on S3 upload
ai_usage_patterns_analysis_1.py   Script version of the analysis notebook
prompt_classifier.pkl, label_encoder.pkl   Trained classifier + label encoder
confusion_matrix.png              Classifier evaluation
ananya_ai_usage.pbix, ai_usage_analytics_dashboards.pdf   Power BI report and export
requirements.txt
```

## Output tables (Parquet)

- `unified_prompt_efficiency.parquet` — one row per prompt across OASST, LMSYS and ChatGPT, with predicted efficiency label
- `oasst_user_dependency.parquet`
- `lmsys_user_dependency.parquet`
- `survey_dependency.parquet`

## Run it

```bash
pip install -r requirements.txt
```

`inference.py` loads a pre-trained model, detects each input CSV's source by file name or column signature, and writes the four Parquet tables. It does **not** train.

```bash
python inference.py \
  --input-dir clean_data \
  --output-dir out \
  --model-dir . \
  --model-filename model.joblib
```

Defaults match the Amazon SageMaker Processing paths (`/opt/ml/processing/input`, `/output`, `/model`). Exit code `0` means at least one table was written; `1` means a fatal error.

> `model.joblib` is not committed (see `.gitignore`). Regenerate it by running the training section of `notebooks/ai_usage_patterns_analysis_1.ipynb`.



## Caveats

- Efficiency labels are rule-based on OASST, so the classifier learns that rule from text features. It is a proxy for prompt quality, not a human-judged measure.
- Dependency scores are relative rankings within each source, not comparable across sources.
- Survey data is self-reported.

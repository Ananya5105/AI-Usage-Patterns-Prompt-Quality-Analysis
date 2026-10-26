# AI Usage Patterns & Prompt Quality Analysis

Event-driven, cloud-based analytics pipeline on Amazon Web Services (AWS) that measures **how efficiently people prompt AI systems** and **how dependent users are on AI**, across 58,791 conversational AI and survey records from four independent datasets. Upload a CSV and the pipeline classifies it, scores it, writes Parquet, catalogues it, and makes it queryable and ready for dashboards, with no manual steps.

## White Paper
The full technical methodology, empirical validation benchmarks, and architectural design are documented in the [White Paper (PDF)](paper/whitepaper.pdf).

## Stack
Python · Pandas · Scikit-learn · Amazon Simple Storage Service (S3) · AWS Lambda · Amazon SageMaker · AWS Glue · Amazon Athena · PostgreSQL · Power BI · LaTeX

## Datasets

| Source | Rows | Grain | Key Signals | Used For |
|---|---|---|---|---|
| **LMSYS Chatbot Arena** | 39,316 | Prompt | Turns, timestamps, model votes, judge ID | Prompt efficiency + judge dependency |
| **OpenAssistant (OASST1)** | 15,708 | Prompt | Word count, conversation tree size, repeat flag, user ID | Ground-truth labels + prompt efficiency + user dependency |
| **ChatGPT Prompts Dataset** | 153 | Prompt | Curated role templates, text length, category | Prompt efficiency only |
| **Student AI Usage Survey** | 3,614 | Student | Daily hours, trust level, academic flag, tool counts | Survey dependency score |
| **Total Cleaned Records** | **58,791** | | | |

The three user populations (OASST `user_id`, LMSYS `judge_user_id`, survey `Student_Name`) describe disjoint cohorts. No cross-source user identifier is fabricated and dependency tables are never joined.

## Validation
- **Ablation & Baseline**: The 8-feature text Random Forest reaches 59.1% test accuracy and 0.576 macro-F1 on 3,142 held-out OASST prompts.
- **Length Leakage**: A length-only baseline (`Word_Count` alone) achieves 66.3% accuracy, confirming prompt length accounts for the primary variance in behavioural labels.
- **Feature Decoupling**: Removing length-coupled features drops accuracy to 46.5%, while residualising features against log word count yields 66.2%.
- **Cross-Source Evaluation**: Applied across 39,316 LMSYS prompts (47.0% Low) and 153 ChatGPT prompts (98.7% High) with steady model confidence (0.57–0.78).
- **Weight Sensitivity**: 200 random ±30% weight perturbations confirm dependency ranking stability (mean Spearman $\rho \ge 0.990$, 5th percentile $\ge 0.977$).

## What the Project Produces

### 1. Prompt Efficiency Classifier
- Random Forest (300 trees, max depth 10, balanced class weights, seed 42) trained on **eight text-only features** (question mark, code keywords, "please", average word length, unique word ratio, punctuation density, capital ratio, starts with a "wh" word).
- Ground-truth labels are derived exclusively from OASST behavioural columns (word count, tree size, repeat flag). These behavioural columns are **excluded from the model features** to prevent circular data leakage.
- Because the model inspects only text markers, it generalizes to OASST, LMSYS, and ChatGPT prompts alike.

### 2. AI Dependency Scores (0–100, Tiered Low / Moderate / High)
Each source uses only the signals it actually contains. Every component is min--max normalised to $[0, 1]$, then blended with documented, tunable weights into a 0--100 score partitioned into equal-sized tercile tiers:

| Table | Grain | Weights |
|---|---|---|
| `oasst_user_dependency` | per user | volume 30 · repeat rate 25 · average turns 25 · low-efficiency rate 20 |
| `lmsys_user_dependency` | per judge user | volume 40 · average turns 30 · low-efficiency rate 30 (no repeat component fabricated) |
| `survey_dependency` | per student | usage hours 30 · trust 20 · academic use 20 · tool count 15 · usage bucket 15 |

### 3. Power BI Dashboards
1. Executive Overview
2. User Behavior Analytics
3. Prompt Analytics
4. AI Dependency Dashboard
5. Student AI Usage Analytics
6. Recommendations Dashboard

Files: `ananya_ai_usage.pbix` (interactive report) and `ai_usage_analytics_dashboards.pdf` (exported dashboard pages).

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
| Parquet output | Four columnar tables written back to Amazon S3. |
| AWS Glue crawler | Crawls the output location and creates or updates the table definitions. |
| Amazon Athena | Serverless SQL queries over the tables. |
| Power BI | Six dashboards built on the Athena tables. |

## Repository Layout

```
clean_data/         Cleaned CSVs for all four sources (+ OASST predictions)
notebooks/          cleaning.ipynb, ai_usage_patterns_analysis_1.ipynb
paper/              LaTeX white paper source, tables, and figures
  tables/           classreport.tex, importance.tex, ablation.tex, source_dist.tex, dependency.tex, sensitivity.tex
  figures/          confusion_matrix.png and dashboard screenshots
  whitepaper.tex    LaTeX whitepaper root document
  whitepaper.pdf    Compiled publication-ready PDF whitepaper
experiments/        Validation experiments and baseline suites (run_experiments.py)
parquets/           The four output tables (Athena-ready)
inference.py        SageMaker Processing batch entry point
lambda/             AWS Lambda function that starts the SageMaker job on S3 upload
ai_usage_patterns_analysis_1.py   Training and analysis pipeline script
model.joblib, prompt_classifier.pkl, label_encoder.pkl   Trained model artifacts
confusion_matrix.png              Classifier evaluation plot
ananya_ai_usage.pbix, ai_usage_analytics_dashboards.pdf   Power BI report and export
results.json        All experimental numbers, metrics, and correlations
requirements.txt
```

## How to Reproduce

### 1. Setup Environment & Run Validation Experiments
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt pyarrow joblib matplotlib scipy

# Run experiments to compute metrics, baselines, and output LaTeX tables / results.json
python experiments/run_experiments.py
```

### 2. Build LaTeX White Paper
```bash
cd paper
pdflatex -interaction=nonstopmode whitepaper.tex
pdflatex -interaction=nonstopmode whitepaper.tex
```
Output compiled at [`paper/whitepaper.pdf`](paper/whitepaper.pdf).

### 3. Run Batch Scoring Pipeline
```bash
python inference.py \
  --input-dir clean_data \
  --output-dir out \
  --model-dir . \
  --model-filename model.joblib
```

## Caveats
- Efficiency labels are rule-based proxies derived from OASST behavioural metrics, rather than human expert annotations.
- Dependency scores are relative internal rankings within each population and are not numerically comparable across different sources.
- Survey findings are based on self-reported usage hours and perception.

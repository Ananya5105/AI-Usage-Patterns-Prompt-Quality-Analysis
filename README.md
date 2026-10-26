# AI Usage Patterns & Prompt Quality Analysis

End-to-end analytics pipeline analyzing 58,791 conversational AI interaction and survey records across 4 cleaned datasets.

## White Paper
The complete technical methodology, validation benchmarks, and architectural design are documented in the [White Paper (PDF)](paper/whitepaper.pdf).

## Stack
Python · Pandas · Scikit-learn · PostgreSQL · Power BI · AWS Athena / Parquet · LaTeX

## Datasets
- **LMSYS Chatbot Arena**: 39,316 prompts (turn-level comparisons and user volume)
- **OpenAssistant (OASST1)**: 15,708 prompts (conversation trees, word counts, repeat flags)
- **ChatGPT Prompts Dataset**: 153 prompts (curated prompt templates)
- **Student AI Usage Survey**: 3,614 students (self-reported hours, trust, academic usage)
- **Total**: 58,791 cleaned records

## Pipeline
- **Analytics Path**: Raw Data → Python/Pandas (cleaning & EDA) → PostgreSQL (star schema) → Power BI (6 dashboards)
- **Modelling Path**: Clean CSVs → Text Feature Extraction → Random Forest Scoring → Dependency Scoring → Parquet → S3 / Athena

## Validation
- **Ablation & Baseline**: The 8-feature text Random Forest reaches 59.1% test accuracy and 0.576 macro-F1 on 3,142 held-out OASST prompts.
- **Length Leakage**: A length-only baseline (`Word_Count` only) achieves 66.3% accuracy, confirming prompt length accounts for the primary variance in behavioural labels.
- **Feature Decoupling**: Removing length-coupled features drops accuracy to 46.5%, while residualising features against log word count yields 66.2%.
- **Cross-Source Evaluation**: Applied across 39,316 LMSYS prompts (47.0% Low) and 153 ChatGPT prompts (98.7% High) with steady model confidence (0.57–0.78).
- **Weight Sensitivity**: 200 random ±30% weight perturbations confirm dependency ranking stability (mean Spearman $\rho \ge 0.990$, 5th percentile $\ge 0.977$).

## Dashboards
1. Executive Overview
2. User Behavior Analytics
3. Prompt Analytics
4. AI Dependency Dashboard
5. Student AI Usage Analytics
6. Recommendations Dashboard

## How to Reproduce

### 1. Setup Environment & Run Experiments
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt pyarrow joblib matplotlib scipy

# Run experiments and generate LaTeX results/tables
python experiments/run_experiments.py
```

### 2. Build LaTeX White Paper
```bash
cd paper
pdflatex -interaction=nonstopmode whitepaper.tex
pdflatex -interaction=nonstopmode whitepaper.tex
```
Output compiled at [`paper/whitepaper.pdf`](paper/whitepaper.pdf).

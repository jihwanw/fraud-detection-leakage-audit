# Does the Boardroom Network Predict Corporate Fraud?
## A Leakage-Controlled Evaluation with a Cautionary Demonstration — Replication & Audit Pipeline

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22271884.svg)](https://doi.org/10.5281/zenodo.22271884)

This repository contains the replication pipeline for *Does the Boardroom
Network Predict Corporate Fraud? A Leakage-Controlled Evaluation with a
Cautionary Demonstration* (under review). It reconstructs every empirical
result from licensed source data and public data, and provides a reusable
leakage-audit template for network-based fraud-detection research.

## Design rules

1. **Reveal-date discipline (features).** Fraud-exposure features at year *t*
   use only fraud whose first public revelation precedes *t*.
2. **Reveal-date discipline (labels).** Positives at *t* are misstatements
   ongoing at *t* and revealed after *t*.
3. **Temporal evaluation.** Models are never evaluated with random
   cross-validation or a resampled test distribution. Paired cluster bootstrap
   by firm is used only after scoring to quantify uncertainty in model
   differences.
4. **Rare-event metrics.** ROC-AUC, PR-AUC, and precision@top-1% are computed
   on the intact temporally held-out test sample.
5. **No synthetic or placeholder inputs.** Random seeds control learner
   stochasticity, bootstrap inference, and graph layout; they never generate
   financial variables, labels, or network observations.

## Single source of truth

`code/20_build_numbers_json.py` recomputes panel and label facts from source
parquets and parses model results from execution logs and aggregate diagnostic
JSON files. It writes `results/paper_numbers.json`. Numeric tables are generated
by `code/21_make_tables.py`; figures by `code/16_paper_figures.py` and
`code/17_network_visual.py`. No empirical result statistic is manually entered
into a table or figure generator.

The confidential manuscript source is not distributed here. Before release,
the canonical JSON and generated artifacts passed 50 JSON-derived checks
against the manuscript; the archived audit output is included in
`results/numeric_audit_report.txt`.

## Result-to-script map

| Paper item | Script | Audit artifact |
|---|---|---|
| Licensed data collection | `01_collect.py`, `05_collect_bao_items.py` | `logs/collection_log.txt` |
| Leak-free network panel | `02_build_panel.py`, `08_extend_network.py` | `logs/panel_build.log` |
| Ongoing-misstatement labels and baseline | `04_ongoing_fraud_eval.py`, `06_bao_eval.py` | `logs/eval04.log`, `logs/eval06.log` |
| Fixed split, firm-cluster bootstrap, rolling origin | `09_final_eval.py`, `07_bootstrap_test.py` | `logs/final.log`, `logs/boot.log` |
| Leakage dose-response | `10_leakage_demo.py`, `12_referee_experiments.py` | `logs/leak.log`, `logs/referee.log` |
| Homonym sensitivity | `11_homonym_sensitivity.py`, `15_final_referee.py` | `logs/homonym.log`, `logs/final_ref.log` |
| Construction grid (Supplementary Table S1) | `13_construction_grid.py` | `logs/grid.log` |
| SEC AAER validation | `14_aaer_overlap.py`, `15_final_referee.py` | `logs/aaer.log`, `logs/final_ref.log` |
| Industry controls | `15_final_referee.py` | `logs/final_ref.log` |
| Network figure and topology | `17_network_visual.py` | `logs/netviz.log` |
| BoardEx trial aggregate diagnostic | `18_boardex_trial_diagnostic.py` | `data/boardex_trial_diag.json` |
| Missing-data sensitivity (Supplementary Table S2) | `19_missingness_sensitivity.py` | `results/missingness_sensitivity.json`, `logs/missingness.log` |
| Canonical results and internal manuscript audit | `20_build_numbers_json.py` | `results/paper_numbers.json`, `results/numeric_audit_report.txt` |
| Generated tables and figures | `21_make_tables.py`, `16_paper_figures.py` | generated artifacts |

## Data access and redistribution

Audit Analytics restatements and director/officer records and Compustat
fundamentals are licensed through WRDS and cannot be redistributed. Derived
row-level panels retaining licensed identifiers are also excluded. Researchers
with the required WRDS subscriptions can reconstruct them with the collection
and processing scripts.

Redistributable artifacts included here:

- `data/aaer_public.parquet` — public SEC AAER index (3,317 releases,
  1997–2026), used for label validation.
- `data/boardex_trial_diag.json` — aggregate counts only; no licensed BoardEx
  rows.
- `logs/` — aggregate execution logs used as audit evidence.
- `results/paper_numbers.json` and `results/missingness_sensitivity.json` —
  aggregate result artifacts.
- `results/numeric_audit_report.txt`, `results/final_submission_audit.json`, and
  `results/reference_audit_final.md` — pre-submission numeric, data, randomness,
  and bibliography audit evidence.

## WRDS authentication

Set environment variables:

```bash
export WRDS_USERNAME='your_username'
export WRDS_PASSWORD='your_password'
```

Alternatively, set `WRDS_CONFIG` to an INI file with a `[wrds]` section or a
simple `USERNAME=...` / `PASSWORD=...` file. Credentials are never printed or
stored by the scripts.

## Reproduction

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
mkdir -p data logs results paper/tables paper/figures

python code/01_collect.py | tee logs/collect_stdout.log
python code/05_collect_bao_items.py
python code/02_build_panel.py | tee logs/panel_build.log
python code/08_extend_network.py
python code/09_final_eval.py | tee logs/final.log
python code/07_bootstrap_test.py | tee logs/boot.log
python code/10_leakage_demo.py | tee logs/leak.log
python code/11_homonym_sensitivity.py | tee logs/homonym.log
python code/12_referee_experiments.py | tee logs/referee.log
python code/13_construction_grid.py | tee logs/grid.log
python code/14_aaer_overlap.py | tee logs/aaer.log
python code/15_final_referee.py | tee logs/final_ref.log
python code/17_network_visual.py | tee logs/netviz.log
python code/18_boardex_trial_diagnostic.py
python code/19_missingness_sensitivity.py | tee logs/missingness.log
python code/20_build_numbers_json.py
python code/21_make_tables.py
python code/16_paper_figures.py
```

`code/18` requires access to the WRDS `boardex_trial` library. If that library
is unavailable, the released aggregate diagnostic JSON can be used to recreate
the submitted canonical results without redistributing trial rows.

## Citation

Woo, Jihwan. (2026). *fraud-detection-leakage-audit: Leakage-controlled
evaluation pipeline for network-based fraud detection* (v1.0.0). Zenodo.
https://doi.org/10.5281/zenodo.22271884

## License

MIT for code. Licensed source data remain subject to their providers' terms.

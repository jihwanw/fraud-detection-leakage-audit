# Does the Boardroom Network Predict Corporate Fraud?
## A Leakage-Controlled Evaluation with a Cautionary Demonstration — Replication & Audit Pipeline

This repository contains the complete pipeline for the paper
*"Does the Boardroom Network Predict Corporate Fraud? A Leakage-Controlled
Evaluation with a Cautionary Demonstration"* (under review). Its purpose is
twofold: to reproduce every number in the paper, and to serve as a reusable
**leakage-audit template** for network-based fraud-detection research.

## Design rules implemented here

1. **Reveal-date discipline (features).** Fraud-exposure features at year *t*
   use only fraud whose first public revelation precedes *t*.
2. **Reveal-date discipline (labels).** Positives at *t* are misstatements
   ongoing at *t* and revealed after *t* (Dechow et al., 2011 convention).
3. **Temporal evaluation only.** No random cross-validation; no resampling of
   test data.
4. **Rare-event metrics.** ROC-AUC, PR-AUC and precision@top-1% on the intact
   test distribution.

## Single source of truth

Every statistic, table and figure in the paper derives from
[`results/paper_numbers.json`](results/paper_numbers.json), which is built by
`code/20_build_numbers_json.py` from raw data and the experiment logs, and is
verified by 45 automated cross-checks against the manuscript. Tables are
generated as LaTeX fragments by `code/21_make_tables.py`; figures by
`code/16_paper_figures.py` and `code/17_network_visual.py`. No number is typed
by hand.

## Result-to-script map

| Paper item | Script | Log (audit evidence) |
|---|---|---|
| Data collection (restatements, directors, fundamentals) | `01_collect.py`, `05_collect_bao_items.py` | `logs/collection_log.txt` |
| Leak-free network panel | `02_build_panel.py`, `08_extend_network.py` | `logs/panel_build.log` |
| Ongoing-misstatement labels & baseline | `04_ongoing_fraud_eval.py`, `06_bao_eval.py` | `logs/eval06.log` |
| Main results (Table: fixed split; rolling; bootstrap) | `09_final_eval.py`, `07_bootstrap_test.py` | `logs/final.log`, `logs/boot.log` |
| Leakage dose-response (Table: leak) | `10_leakage_demo.py`, `12_referee_experiments.py` | `logs/leak.log`, `logs/referee.log` |
| Construction grid (Supp. Table S1) | `13_construction_grid.py` | `logs/grid.log` |
| AAER label validation | `14_aaer_overlap.py` | `logs/aaer.log` |
| Industry controls, homonym sensitivity, AAER-anchored label | `15_final_referee.py`, `11_homonym_sensitivity.py` | `logs/final_ref*.log`, `logs/homonym.log` |
| Network figure & topology table | `17_network_visual.py` | `logs/netviz.log` |
| Canonical numbers + 45 cross-checks | `20_build_numbers_json.py` | (prints check results) |
| Generated tables / figures | `21_make_tables.py`, `16_paper_figures.py` | — |

## Data access

Audit Analytics (restatements, feed 39; directors & officers, feed 17) and
Compustat fundamentals are licensed products accessed via
[WRDS](https://wrds-www.wharton.upenn.edu/) and **cannot be redistributed
here**. Researchers with WRDS access can rebuild every dataset by running the
collection scripts (WRDS credentials required). Included in this repository:

- `data/aaer_public.parquet` — the SEC AAER index scraped from the public SEC
  website (3,317 releases, 1997–2026), used for label validation.
- `logs/` — complete experiment logs containing only aggregate statistics;
  these are the audit trail cited in the paper.

## Reproduction

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# WRDS credentials file: see code/01_collect.py header
python code/01_collect.py          # ~10 min (WRDS)
python code/05_collect_bao_items.py
python code/02_build_panel.py
python code/08_extend_network.py
python code/09_final_eval.py       # main results
python code/20_build_numbers_json.py   # canonical JSON + 45 checks
python code/21_make_tables.py      # LaTeX tables
python code/16_paper_figures.py    # figures
```

## Related project

[`board-interlock-contagion`](https://github.com/jihwanw/board-interlock-contagion)
studies a complementary question — whether director mobility transmits
restatement risk (a behavioral-channel design) — on the same class of data.
The present repository asks the prediction question under leakage control;
the two designs are distinct and the findings are complementary.

## Citation

Please cite the paper (details to be added on publication) and this
repository via its Zenodo DOI (badge to be added on archiving).

## License

MIT (code). Note the data-access restrictions above.

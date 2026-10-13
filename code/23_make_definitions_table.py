#!/usr/bin/env python3
"""23_make_definitions_table.py — Key-concept definitions table (Scientific Reports R4,
editor request 2026-10-13): how each key concept was defined, the source of the
definition, and where it is applied in the study.

Two fragments are written from the same row list so the main text and the
Supplementary Information carry identical content:
  tab_defs.tex       main text (natbib \\citep keys; \\ref cross-references)
  tab_defs_supp.tex  Supplementary Table S3 (plain-text sources; literal numbers)

Every numeric value comes from results/paper_numbers.json; nothing is typed by hand.
Usage: 23_make_definitions_table.py [out_dir]   (default: paper/tables)
"""
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = json.load(open(f"{BASE}/results/paper_numbers.json"))
OUT = sys.argv[1] if len(sys.argv) > 1 else f"{BASE}/paper/tables"
os.makedirs(OUT, exist_ok=True)


def n(x):
    return f"{x:,}".replace(",", "{,}")


P, R, A, Z, S, M = J["panel"], J["restatements"], J["aaer_label"], J["dose_response"], J["seeds"]["rusboost"], J["missingness"]["methods"]["zero"]
BOOT = M["cluster_bootstrap"]["replicates"]
pct_fraud = f"{P['fraud_pos'] / P['firm_years'] * 100:.2f}"

# (cite keys for the main text, plain text for the supplement)
SRC = {
    "standards": (r"PCAOB AS~2401; IAASB ISA~240\citep{pcaob2401, isa240}", "PCAOB AS 2401; IAASB ISA 240"),
    "aa": (r"Audit Analytics\citep{auditanalytics2013}", "Audit Analytics (2013)"),
    "dechow": (r"Dechow et al.\citep{dechow2011predicting}", "Dechow et al. (2011)"),
    "dechow_bao": (r"Dechow et al.; Bao et al.\citep{dechow2011predicting, bao2020detecting}", "Dechow et al. (2011); Bao et al. (2020)"),
    "bao": (r"Bao et al.; Cecchini et al.\citep{bao2020detecting, cecchini2010}", "Bao et al. (2020); Cecchini et al. (2010)"),
    "leak": (r"Kaufman et al.; Kapoor \& Narayanan\citep{kaufman2012, kapoor2023leakage}", "Kaufman et al. (2012); Kapoor \\& Narayanan (2023)"),
    "saito": (r"Saito \& Rehmsmeier\citep{saito2015}", "Saito \\& Rehmsmeier (2015)"),
    "karpoff": (r"Karpoff et al.\citep{karpoff2017}", "Karpoff et al. (2017)"),
    "seiffert": (r"Seiffert et al.; Bao et al.\citep{seiffert2010, bao2020detecting}", "Seiffert et al. (2010); Bao et al. (2020)"),
    "study": ("This study", "This study"),
}

# Cross-references: (main-text LaTeX, supplement plain text)
REF = {
    "tab_desc": (r"Table~\ref{tab:desc}", "Table 2"),
    "tab_main": (r"Table~\ref{tab:main}", "Table 4"),
    "tab_leak": (r"Table~\ref{tab:leak}", "Table 5"),
    "fig_net": (r"Fig.~\ref{fig:network}", "Fig. 1"),
    "fig_roll": (r"Fig.~\ref{fig:rolling}", "Fig. 2"),
    "fig_dec": (r"Fig.~\ref{fig:decomp}", "Fig. 3"),
    "fig_dose": (r"Fig.~\ref{fig:dose}", "Fig. 4"),
}


def rows(main: bool):
    s = {k: v[0 if main else 1] for k, v in SRC.items()}
    r = {k: v[0 if main else 1] for k, v in REF.items()}
    return [
        ("Fraud (conceptual definition)",
         "Intentional financial-statement misstatement. PCAOB AS~2401 (paras.~5--6) defines fraud for audit purposes as an intentional act that results in a material misstatement of the financial statements and names two relevant types, fraudulent financial reporting and misappropriation of assets; ISA~240 (para.~11(a)) defines fraud more broadly as an intentional act involving deception to obtain an unjust or illegal advantage and (paras.~2--3) names the same two types of intentional misstatement. The standards supply the conceptual scope; neither they nor this study assess intent, materiality or legal liability for individual observations.",
         s["standards"], "Sect.~3.3"),
        ("Fraud-flagged restatement (event)",
         f"A restatement record in Audit Analytics feed 39 with field \\texttt{{res\\_fraud}}=1, a category the provider describes as a restatement caused by intentional manipulation of financial data or misappropriation of assets; {n(R['res_fraud'])} filings ({R['res_fraud_pct_of_restatements']:.1f} percent of restatements). The other feed-39 flags queried (\\texttt{{res\\_accounting}}, \\texttt{{res\\_clerical\\_errors}}, \\texttt{{res\\_sec\\_investigation}}, \\texttt{{res\\_adverse}}) do not qualify a record on their own.",
         s["aa"] + " (category); " + s["study"] + " (field selection)", f"Sects.~3.1, 3.3; {r['tab_desc']}"),
        ("Positive firm year (label $y_{c,t}=1$)",
         f"Firm $c$ in fiscal year $t$ is positive when a fraud-flagged restatement of $c$ has a misstatement period covering $t$ ($b \\le t \\le e$) and a filing year later than $t$ ($f>t$). The firm-year unit follows the misstatement-prediction convention; the three-condition rule was developed for this study. {n(P['fraud_pos'])} positive firm years, a base rate of {pct_fraud} percent of {n(P['firm_years'])}.",
         s["dechow"] + " (unit); " + s["study"] + " (rule)", f"Sects.~3.3, 4.2 (Eq.~2); {r['tab_desc']}, {r['tab_main']}"),
        ("Negative firm year ($y_{c,t}=0$)",
         "A firm year with no qualifying spell. It records the absence of a qualifying restatement, not verified absence of misconduct: undetected or unclassified misconduct can produce observed negatives, which the study treats as one-sided label noise (an assumption about the negative class, not a claim that the positive class is error-free).",
         s["study"], "Sects.~3.3, 7.1"),
        ("Broad (secondary) outcome",
         f"A firm year positive under the same three conditions when the restatement carries \\texttt{{res\\_fraud}}=1 or \\texttt{{res\\_sec\\_investigation}}=1; {n(P['broad_pos'])} positive firm years. Always reported separately from the primary outcome.",
         s["study"], "Sects.~3.3, 5.1"),
        ("AAER-associated outcome (sensitivity)",
         f"An accounting-restatement spell covering $t$ and filed after $t$, at a firm matched by normalized respondent name to at least one public SEC Accounting and Auditing Enforcement Release; {n(A['pos'])} positive firm years. A firm-associated proxy, not an event-level AAER--restatement match.",
         s["dechow_bao"] + " (enforcement labels in prior work); " + s["study"] + " (construction)", "Sects.~3.4, 5.1, 5.2, 7.5"),
        ("Prediction year $t$; misstatement period $[b,e]$; filing year $f$",
         "$t$ is the fiscal year whose firm-year label is predicted; $b$ and $e$ are the calendar years of the feed-39 fields \\texttt{res\\_begin\\_date} and \\texttt{res\\_end\\_date}; $f$ is the calendar year of \\texttt{file\\_date}, the date the restatement was filed with the SEC.",
         s["aa"] + " (fields queried by the collection script); " + s["study"] + " (conversion to years; rule)", "Sects.~3.3, 4.2 (Eq.~2)"),
        ("Revealed (reveal date)",
         "A fraud-flagged restatement is treated as publicly revealed in its filing year $f$; the filing year is the study's proxy for public revelation. No other revelation channel (press, enforcement) is used.",
         s["study"], "Sects.~3.3, 4.1 (rules 1--2), 6"),
        ("CIK",
         "The SEC Central Index Key, the registrant identifier used to link restatement, financial-statement, director and public enforcement records.",
         s["study"] + " (usage)", "Sects.~3.4, 3.6"),
        ("$F_t$ (previously revealed fraud firms)",
         "The set of firms whose first fraud-flagged restatement was filed before year $t$. In leakage-controlled specifications only members of $F_t$ contribute fraud status to the exposure features of year $t$.",
         s["study"], f"Sect.~4.2 (Eq.~1); {r['tab_desc']} (Panel A)"),
        ("$F_\\infty$ (complete-database fraud membership)",
         "The set of all firms with a fraud-flagged restatement anywhere in the restatement database, regardless of filing date. The indicator $\\mathbf{1}[c \\in F_\\infty]$ therefore incorporates revelations unavailable at prediction time.",
         s["study"], f"Sect.~6; {r['tab_leak']}; {r['fig_dose']}"),
        ("Board interlock network $G_t$; activity window",
         "Nodes are SEC registrants; an edge joins two registrants whenever one person holds board links to both within the activity window $[t-w,\\,t]$ of director-change events (integer years, both endpoints included; $w=4$ in the baseline). The window approximates active service; the feed records changes, not tenure spells. In the weighted variant the edge weight is the number of shared persons; the primary specification is unweighted.",
         s["study"], f"Sect.~4.2; {r['tab_desc']} (Panel B); {r['fig_net']}"),
        ("Graph statistics (degree, PageRank, density, clustering, giant component)",
         "Degree: number of distinct adjacent registrants. PageRank: \\texttt{networkx} PageRank with default damping 0.85, unweighted in the primary specification (edge weights used in the weighted grid cells). Density: edges over possible edges. Clustering: \\texttt{networkx} average clustering coefficient. Giant component: the largest connected component, reported as a share of graph registrants.",
         s["study"] + " (choices); \\texttt{networkx} implementation", f"Sects.~3.2, 4.2; {r['tab_desc']} (Panel B); {r['fig_net']}"),
        ("Fraud exposure; NET4 feature set",
         "Fraud exposure of firm $c$ at $t$: the count of its neighbors in $F_t$, $|N_t(c)\\cap F_t|$, and the fraud-neighbor ratio $|N_t(c)\\cap F_t|/|N_t(c)|$ (zero for isolated firms). NET4 combines these two exposure measures with degree and PageRank.",
         s["study"], f"Sect.~4.2 (Eq.~1); {r['tab_main']}"),
        ("NET7 feature set",
         "NET4 plus (i) the two-hop exposure ratio, the share of nodes at distance exactly two from $c$ (neighbors of neighbors, excluding $c$ and its neighbors) that belong to $F_t$, zero when that set is empty; and (ii) the count and share of $c$'s board members active in the window who had been linked to a firm in $F_t$ no later than that firm's revelation year (the fraud-associated director channel), the share being zero when no board member is recorded.",
         s["study"], f"Sect.~4.2; {r['tab_main']}"),
        ("Financial baseline (FIN)",
         "The 28 financial predictors of Supplementary Note~1: the raw Compustat annual items of Bao et al.\\ (2020), originating in Cecchini et al.\\ (2010); items other than total assets, fiscal-year closing price and shares outstanding are divided by total assets, total assets is replaced by its logarithm, and missing items are set to zero in the primary specification. The financial-only model uses the learner specified for each experiment (RUSBoost in the primary analysis).",
         s["bao"] + " (predictor set; zero-fill)", f"Sects.~3.1, 4.3, 5.1; {r['tab_main']}; Suppl.\\ Note~1"),
        ("Learners",
         f"RUSBoost: AdaBoost over decision trees (\\texttt{{min\\_samples\\_leaf}}=5) with random undersampling of the majority class in each boosting round; 300 estimators, learning rate 0.1, pre-specified seed 42 (the ten-seed experiment varies only the seed). Balanced histogram gradient boosting: class-weighted (\\texttt{{class\\_weight}}=\\texttt{{balanced}}). Class-weighted regularized logistic regression on standardized predictors.",
         s["seiffert"] + " (RUSBoost and its use in misstatement prediction); " + s["study"] + " (settings)", f"Sect.~4.3; {r['tab_main']}"),
        ("Network increment ($\\Delta$ROC-AUC)",
         f"Test ROC-AUC of the network-augmented model minus that of its matched baseline, with sample, split, learner and outcome held fixed within the comparison: FIN in the primary experiments, FIN$+$SIC in the industry-controlled comparison. Reported at the pre-specified seed, averaged over {S['n']} seeds (RUSBoost and gradient boosting), and with paired firm-cluster bootstrap intervals for the fixed-split NET4 and NET7 comparisons.",
         s["study"], f"Sect.~5.2; {r['tab_main']}; {r['fig_roll']}, {r['fig_dec']}"),
        ("Leak-free (leakage-controlled) estimate",
         "Study-specific shorthand for an estimate produced under the four design rules of Sect.~4.1: exposure features use only $F_t$ (rule 1); a positive label requires $f>t$ (rule 2); temporal train/test split with no random cross-validation (rule 3); rare-event metrics on the original test distribution (rule 4). It does not establish point-in-time availability of every financial input or of the training labels at each training cutoff.",
         s["leak"] + " (leakage concept); " + s["study"] + " (four-rule protocol)", f"Sects.~3.3, 4.1, 5; {r['tab_leak']} (first row)"),
        ("Leakage",
         "Use, in feature construction or evaluation, of information that would not have been available at the stated prediction time, including outcome information that enters through features (target leakage) or through non-temporal evaluation.",
         s["leak"], "Sects.~1, 2.2, 4.1, 6"),
        ("Leaked estimate",
         f"An estimate from the same pipeline after relaxing reveal-date discipline. Neighbor-side: the condition $f<t$ on neighbors' fraud filings is replaced by $f<t+h$ for $h=1,2,3$ (admitting filings through year $t$, $t+1$ and $t+2$ respectively) or dropped entirely. Target-side: the own-firm indicator $\\mathbf{{1}}[c \\in F_\\infty]$ is added (ROC-AUC {Z['self_leak_clean']:.3f} against the ladder's leak-free {Z['clean']:.3f}).",
         s["study"], f"Sect.~6; {r['tab_leak']}; {r['fig_dose']}"),
        ("Leakage ladder (dose--response); rung",
         "The seven configurations of Sect.~6, one pipeline and one learner with increasing reveal-date violations in feature construction: the leak-free rung; three neighbor-side horizons; neighbor reveal dates ignored; and the two rungs adding the own-firm indicator. Each rung is compared with the ladder's own leak-free rung, not with the FIN$+$NET4 estimate of the main table.",
         s["study"], f"Sect.~6; {r['tab_leak']}; {r['fig_dose']}"),
        ("Temporal evaluation",
         f"Fixed split: train on fiscal years 2004--2017, test on 2018--2023 ({n(M['test_rows'])} test firm years, {M['test_positives']} positives); rolling-origin splits move the boundary across four test windows. No random cross-validation; point estimates use the original test distribution, and test firms are resampled only in the paired bootstrap used for uncertainty.",
         s["dechow_bao"] + " (practice); " + s["study"], f"Sects.~4.1 (rule 3), 5.2; {r['fig_roll']}"),
        ("Rare-event metrics",
         "ROC-AUC (area under the receiver operating characteristic curve); PR-AUC, computed as average precision (\\texttt{scikit-learn} \\texttt{average\\_precision\\_score}); and precision in the top one percent, the share of positives among the $\\lfloor 0.01\\,N_{\\mathrm{test}}\\rfloor$ highest-scored test firm years (descending score order). Accuracy and F1 at a threshold, common in the applied literature, are not used.",
         s["saito"] + " (rationale); " + s["study"] + " (implementation)", f"Sect.~4.1 (rule 4); {r['tab_main']}"),
        ("Industry-controlled comparison; industry-adjusted exposure",
         "The financial baseline augmented with one-digit Standard Industrial Classification (SIC) dummies (FIN$+$SIC), with NET4 added on top (FIN$+$SIC$+$NET4) or with the industry-adjusted exposure feature: the fraud-neighbor ratio minus its mean within the one-digit-SIC $\\times$ year cell, entered with degree and PageRank. One-digit codes are used because the fixed-split test contains few positives.",
         s["study"], "Sect.~5.3"),
        ("Industry-period risk set (proposed)",
         "Evaluation of the increment within groups of firms exposed to the same industry-period conditions. Named as future work; not implemented in this study.",
         s["study"], "Sects.~5.3, 7.4; Suppl.\\ Note~2"),
        ("Construction grid",
         "The six interlock-network constructions obtained by crossing the activity window $w\\in\\{3,4,5\\}$ with unweighted and weighted edges (weight = number of shared persons, used in degree and PageRank); the increment is re-estimated in each cell on that cell's merged sample.",
         s["study"], f"Sects.~4.2, 5.3; {r['fig_dec']}(a); Suppl.\\ Table~S1"),
        ("Person identification; homonym strictness",
         f"Directors are matched across firms by normalized full name (first, middle, last) because the director feed lacks usable person keys. A homonym merger is two or more distinct people sharing one normalized name key; the {n(J['name_matching']['names_5plus_firms'])} of {n(J['name_matching']['names'])} matched names attached to five or more firms is a diagnostic of this risk, not a verified false-merge rate. Strictness rules: baseline full-name matching; dropping names linked to five or more firms; requiring a non-empty middle name.",
         s["study"], f"Sects.~3.5, 5.3; {r['fig_dec']}(b); Suppl.\\ Note~2"),
        ("Inference: paired firm-cluster bootstrap; seed variation",
         f"{n(BOOT)} resamples of test firms (clusters) for the paired fixed-split increment; {S['n']} learner seeds for training stochasticity; rolling-origin windows for temporal stability. Window-specific intervals are not reported.",
         s["study"], f"Sects.~4.3, 5.2; {r['tab_main']}"),
        ("Missing-data specifications",
         "Zero-fill of absent financial items (primary, following Bao et al.\\ 2020); train-only median imputation; median imputation with missingness indicators; complete cases.",
         s["bao"] + " (zero-fill); " + s["study"] + " (alternatives)", "Sects.~4.3, 5.2, 7.5; Suppl.\\ Table~S2"),
    ]


HEADER = r"""\begin{longtable}{>{\raggedright\arraybackslash}p{2.5cm}>{\raggedright\arraybackslash}p{6.5cm}>{\raggedright\arraybackslash}p{2.6cm}>{\raggedright\arraybackslash}p{2.6cm}}
"""


def body(main: bool):
    out = []
    for c, d, s, w in rows(main):
        out.append(f"{c} & {d} & {s} & {w} \\\\\n\\addlinespace")
    return "\n".join(out)


def fragment(main: bool):
    cap_main = (r"\caption{Key concepts of the study: definition, source of the definition, and where each is applied. "
                r"Counts refer to the source-data universe or analysis subset identified in each row; methodological settings describe the stated specifications. Sources marked ``This study'' are operational definitions developed for this paper; the cited works supply the concept or the practice adopted, as labelled.}\label{tab:defs}\\")
    cap_supp = (r"\caption*{\textbf{Supplementary Table S3.} Key concepts of the study: definition, source of the definition, and where each is applied (identical to Table~3 of the main text). "
                r"Sources marked ``This study'' are operational definitions developed for this paper; the cited works supply the concept or the practice adopted. Section, table and figure numbers refer to the main text.}\\")
    head = r"\toprule Concept & Definition in this study & Source & Where applied \\ \midrule \endfirsthead" + "\n" + \
           r"\multicolumn{4}{l}{\footnotesize\emph{(continued)}}\\ \toprule Concept & Definition in this study & Source & Where applied \\ \midrule \endhead" + "\n" + \
           r"\midrule \multicolumn{4}{r}{\footnotesize\emph{continued on next page}}\\ \endfoot" + "\n" + r"\bottomrule \endlastfoot" + "\n"
    return "{\\footnotesize\n" + HEADER + (cap_main if main else cap_supp) + "\n" + head + body(main) + "\n\\end{longtable}}\n"


for name, main in (("tab_defs.tex", True), ("tab_defs_supp.tex", False)):
    with open(f"{OUT}/{name}", "w") as f:
        f.write("% AUTO-GENERATED by code/23_make_definitions_table.py from results/paper_numbers.json\n"
                "% DO NOT EDIT BY HAND — regenerate instead.\n" + fragment(main))
    print("wrote", f"{OUT}/{name}")

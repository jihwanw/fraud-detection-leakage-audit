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
# Per-name citations: (main-text form with its own superscript, plain form for the supplement)
CITE = {
    "pcaob": (r"PCAOB AS~2401\citep{pcaob2401}", "PCAOB AS 2401"),
    "isa": (r"IAASB ISA~240\citep{isa240}", "IAASB ISA 240"),
    "aa_cat": (r"Audit Analytics\citep{auditanalytics2013}", "Audit Analytics (2013)"),
    "aa_feed": (r"Audit Analytics feed 39 via WRDS\citep{auditanalytics_feed39}", r"Audit Analytics feed 39 via WRDS (table audit.feed39\allowbreak\_financial\allowbreak\_restatements; accessed 2 September 2026)"),
    "dechow": (r"Dechow et al.\citep{dechow2011predicting}", "Dechow et al. (2011)"),
    "bao": (r"Bao et al.\citep{bao2020detecting}", "Bao et al. (2020)"),
    "cecchini": (r"Cecchini et al.\citep{cecchini2010}", "Cecchini et al. (2010)"),
    "kaufman": (r"Kaufman et al.\citep{kaufman2012}", "Kaufman et al. (2012)"),
    "kapoor": (r"Kapoor \& Narayanan\citep{kapoor2023leakage}", "Kapoor \\& Narayanan (2023)"),
    "saito": (r"Saito \& Rehmsmeier\citep{saito2015}", "Saito \\& Rehmsmeier (2015)"),
    "karpoff": (r"Karpoff et al.\citep{karpoff2017}", "Karpoff et al. (2017)"),
    "seiffert": (r"Seiffert et al.\citep{seiffert2010}", "Seiffert et al. (2010)"),
    "networkx": (r"NetworkX\citep{hagberg2008networkx}", "NetworkX (Hagberg et al., 2008)"),
    "sklearn": (r"scikit-learn\citep{pedregosa2011scikit}", "scikit-learn (Pedregosa et al., 2011)"),
    "imblearn": (r"imbalanced-learn\citep{lemaitre2017imbalanced}", "imbalanced-learn (Lema\\^{i}tre et al., 2017)"),
    "fuller": (r"Fuller\citep{fuller1987measurement}", "Fuller (1987)"),
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
    c = {k: v[0 if main else 1] for k, v in CITE.items()}
    r = {k: v[0 if main else 1] for k, v in REF.items()}
    return [
        ("Fraud (conceptual definition)",
         "Intentional financial-statement misstatement. PCAOB AS~2401 (paras.~5--6) defines fraud for audit purposes as an intentional act that results in a material misstatement of the financial statements and names two relevant types, fraudulent financial reporting and misappropriation of assets; ISA~240 (para.~11(a)) defines fraud more broadly as an intentional act involving deception to obtain an unjust or illegal advantage and (paras.~2--3) names the same two types of intentional misstatement. The standards establish the conceptual scope; they do not establish that individual study observations meet their definitions, and this study makes no independent assessment of intent, materiality or legal liability.",
         f"{c['pcaob']}; {c['isa']}", "Sect.~3.3"),
        ("Fraud-flagged restatement (event)",
         f"A restatement record in Audit Analytics feed 39 with field \\texttt{{res\\_fraud}}=1, a category the provider describes as a restatement caused by intentional manipulation of financial data or misappropriation of assets; {n(R['res_fraud'])} filings ({R['res_fraud_pct_of_restatements']:.1f} percent of restatements). The other feed-39 flags queried (\\texttt{{res\\_accounting}}, \\texttt{{res\\_clerical\\_errors}}, \\texttt{{res\\_sec\\_investigation}}; \\texttt{{res\\_adverse}}, retained but unused) do not qualify a record on their own.",
         f"{c['aa_cat']} (category description); {c['aa_feed']} (flags); {c['study']} (field selection)", f"Sects.~3.1, 3.3; {r['tab_desc']}"),
        ("Positive firm year (label $y_{c,t}=1$)",
         f"Firm $c$ in fiscal year $t$ is positive when a fraud-flagged restatement of $c$ has a misstatement period covering $t$ ($b \\le t \\le e$) and a filing year later than $t$ ($f>t$). The firm-year unit follows the misstatement-prediction convention; the three-condition rule was developed for this study. {n(P['fraud_pos'])} positive firm years of {n(P['firm_years'])}.",
         f"{c['dechow']} (unit); {c['study']} (rule)", f"Sects.~3.3, 4.2 (Eq.~2); {r['tab_desc']}, {r['tab_main']}"),
        ("Negative firm year ($y_{c,t}=0$)",
         "A firm year with no qualifying spell. It records the absence of a qualifying restatement, not verified absence of misconduct: undetected or unclassified misconduct can produce observed negatives, which the study treats as one-sided label noise (an assumption about the negative class, not a claim that the positive class is error-free).",
         c["study"], "Sects.~3.3, 7.1"),
        ("Observed prevalence (base rate)",
         f"The number of positive firm years divided by the number of firm years in the stated outcome and sample ({pct_fraud} percent for the primary outcome in the full panel). It is the rate of provider-flagged, later-filed misstatement, not the incidence of all underlying misconduct.",
         c["study"], f"Sects.~2.2, 3.3, 4.1, 5.1; {r['tab_desc']}"),
        ("Broad (secondary) outcome",
         f"A firm year positive under the same three conditions when the restatement carries \\texttt{{res\\_fraud}}=1 or \\texttt{{res\\_sec\\_investigation}}=1; {n(P['broad_pos'])} positive firm years. Always reported separately from the primary outcome.",
         c["study"], "Sects.~3.3, 5.1"),
        ("AAER-associated outcome (sensitivity)",
         f"An accounting-restatement spell covering $t$ and filed after $t$, at a firm matched by normalized respondent name to at least one public SEC Accounting and Auditing Enforcement Release; {n(A['pos'])} positive firm years. A firm-associated proxy, not an event-level AAER--restatement match.",
         f"{c['dechow']}; {c['bao']} (enforcement labels in prior work); {c['study']} (construction)", "Sects.~3.4, 5.1, 5.2, 7.5"),
        ("Prediction year $t$; misstatement period $[b,e]$; filing year $f$",
         "$t$ is the fiscal year whose firm-year label is predicted; $b$ and $e$ are the calendar years of the feed-39 date fields \\texttt{res\\_begin\\_date} and \\texttt{res\\_end\\_date}; $f$ is the calendar year of \\texttt{file\\_date}, the date the restatement was filed with the SEC.",
         f"{c['aa_feed']} (date fields); {c['study']} (conversion to years; rule)", "Sects.~3.3, 4.2 (Eq.~2)"),
        ("Revealed (reveal date)",
         "A fraud-flagged restatement is treated as publicly revealed in its filing year $f$; the filing year is the study's proxy for public revelation. No other revelation channel (press, enforcement) is used.",
         c["study"], "Sects.~3.3, 4.1 (rules 1--2), 6"),
        ("CIK",
         "The SEC Central Index Key, the registrant identifier used to link restatement, financial-statement, director and public enforcement records.",
         f"SEC EDGAR (identifier); {c['study']} (usage)", "Sects.~3.4, 3.6"),
        ("$F_t$ (previously revealed fraud firms)",
         "The set of firms whose first fraud-flagged restatement was filed before year $t$. In leakage-controlled specifications only members of $F_t$ contribute fraud status to the exposure features of year $t$.",
         c["study"], f"Sect.~4.2 (Eq.~1); {r['tab_desc']} (Panel A)"),
        ("$F_\\infty$ (complete-database fraud membership)",
         "The set of all firms with a fraud-flagged restatement anywhere in the restatement database, regardless of filing date. The indicator $\\mathbf{1}[c \\in F_\\infty]$ therefore incorporates revelations unavailable at prediction time.",
         c["study"], f"Sect.~6; {r['tab_leak']}; {r['fig_dose']}"),
        ("Board interlock network $G_t$; activity window",
         "Nodes are SEC registrants; an edge joins two registrants whenever one person holds board links to both within the activity window $[t-w,\\,t]$ of director-change events (integer years, both endpoints included; $w=4$ in the baseline). The window approximates active service; the feed records changes, not tenure spells. In the weighted variant the edge weight $w_{cc'}$ is the number of shared persons; the primary specification is unweighted.",
         c["study"], f"Sect.~4.2; {r['tab_desc']} (Panel B); {r['fig_net']}"),
        ("Graph statistics (degree, PageRank, density, clustering, giant component)",
         "Degree: $|N_t(c)|$, the number of distinct adjacent registrants; in weighted specifications the degree feature is $\\sum_{c'\\in N_t(c)} w_{cc'}$. PageRank: the stationary importance score of the random-surfer model, computed with NetworkX \\texttt{pagerank} (damping 0.85; unweighted in the primary specification, edge weights in the weighted grid cells). Density: $2|E_t|/[|V_t|(|V_t|-1)]$. Clustering: for node $c$ with degree $k_c$ and $T_c$ triangles through it, the unweighted local coefficient $C_c = 2T_c/[k_c(k_c-1)]$ (zero for $k_c<2$), averaged over all nodes including zeros (NetworkX \\texttt{average\\_clustering}). Connected component: a maximal set of nodes joined by paths; the giant component is the largest, reported as a share of graph registrants.",
         f"{c['networkx']} (implementation); {c['study']} (choices)", f"Sects.~3.2, 4.2; {r['tab_desc']} (Panel B); {r['fig_net']}"),
        ("Fraud exposure; NET4 feature set",
         "Fraud exposure of firm $c$ at $t$: the count of its neighbors in $F_t$, $|N_t(c)\\cap F_t|$, and the fraud-neighbor ratio $|N_t(c)\\cap F_t|/|N_t(c)|$ (zero for isolated firms). NET4 combines these two exposure measures with degree and PageRank.",
         c["study"], f"Sect.~4.2 (Eq.~1); {r['tab_main']}"),
        ("NET7 feature set",
         "NET4 plus (i) the two-hop exposure ratio, the share of nodes at distance exactly two from $c$ (neighbors of neighbors, excluding $c$ and its neighbors) that belong to $F_t$, zero when that set is empty; and (ii) the count and share of $c$'s board members active in the window (distinct persons $B_t(c)=\\{p:(p,c)\\in D_t\\}$) who had been linked to a firm in $F_t$ no later than that firm's revelation year (the fraud-associated director channel); the share divides the count by $|B_t(c)|$ and is zero when $B_t(c)$ is empty.",
         c["study"], f"Sect.~4.2; {r['tab_main']}"),
        ("Financial baseline (FIN)",
         "The 28 financial predictors of Supplementary Note~1: the raw Compustat annual items of Bao et al.\\ (2020), originating in Cecchini et al.\\ (2010); items other than total assets, fiscal-year closing price and shares outstanding are divided by total assets, total assets is replaced by its logarithm, and missing items are set to zero in the primary specification. The financial-only model uses the learner specified for each experiment (RUSBoost in the primary analysis).",
         f"{c['bao']} (adopted predictor practice and primary zero-fill); {c['cecchini']} (originating predictor set); {c['study']} (learner choice)", f"Sects.~3.1, 4.3, 5.1; {r['tab_main']}; Suppl.\\ Note~1"),
        ("F-score (literature benchmark)",
         "The misstatement-risk score of Dechow et al.\\ (2011): a logistic model's predicted misstatement probability scaled by the sample's unconditional misstatement probability. A benchmark of the prior literature, distinct from the FIN predictors, from this study's model outputs, and from the F1 classification metric; not used as a predictor here.",
         c["dechow"], "Sect.~2.2; Table~1" if main else "Sect. 2.2; Table 1"),
        ("Learners",
         "RUSBoost: AdaBoost over decision trees (\\texttt{min\\_samples\\_leaf}=5) with random undersampling of the majority class in each boosting round; 300 estimators, learning rate 0.1, pre-specified seed 42 (the ten-seed experiment varies only the seed). RUSBoost undersamples the majority class to the minority-class size without replacement (\\texttt{imbalanced-learn} defaults). Balanced histogram gradient boosting: gradient-boosted trees on binned predictor values with class weights inversely proportional to class frequency (\\texttt{HistGradientBoostingClassifier}, \\texttt{class\\_weight}=\\texttt{balanced}). Class-weighted L2-regularized logistic regression (\\texttt{LogisticRegression}, lbfgs solver, default penalty strength, \\texttt{class\\_weight}=\\texttt{balanced}, 3{,}000 iterations) on predictors standardized with training-set means and standard deviations. Training-set undersampling (RUSBoost) and class weighting (the other learners) are the only imbalance treatments; synthetic oversampling (generating artificial minority cases) and evaluation rebalancing (altering the test class distribution) are not used; outcome-based sampling (selecting observations by their labels) is not tested by the own-firm-indicator experiment.",
         f"{c['seiffert']} (RUSBoost); {c['bao']} (its use in misstatement prediction); {c['imblearn']}; {c['sklearn']} (implementations); {c['study']} (settings)", f"Sect.~4.3; {r['tab_main']}"),
        ("Network increment ($\\Delta$ROC-AUC)",
         "Test ROC-AUC of the network-augmented model minus that of its matched baseline, with sample, split, learner and outcome held fixed within the comparison: FIN in the primary experiments, FIN$+$SIC in the industry-controlled comparison. Reported at the pre-specified seed; ten-seed summaries concern FIN$+$NET4 against FIN for RUSBoost and histogram gradient boosting; paired firm-cluster bootstrap intervals are reported for the fixed-split NET4 and NET7 comparisons.",
         c["study"], f"Sect.~5.2; {r['tab_main']}; {r['fig_roll']}, {r['fig_dec']}"),
        ("Leak-free (leakage-controlled) estimate",
         "Study-specific shorthand for an estimate produced under the four design rules of Sect.~4.1: exposure features use only $F_t$ (rule 1); a positive label requires $f>t$ (rule 2); temporal train/test split with no random cross-validation (rule 3); rare-event metrics on the original test distribution (rule 4). It does not establish point-in-time availability of every financial input or of the training labels at each training cutoff.",
         f"{c['kaufman']}; {c['kapoor']} (leakage concept); {c['study']} (four-rule protocol)", f"Sects.~3.3, 4.1, 5; {r['tab_leak']} (first row)"),
        ("Leakage",
         "Use, in feature construction or evaluation, of information that would not have been available at the stated prediction time, including outcome information that enters through features (target leakage) or through non-temporal evaluation.",
         f"{c['kaufman']}; {c['kapoor']}", "Sects.~1, 2.2, 4.1, 6"),
        ("Leaked estimate",
         f"An estimate from the same pipeline after relaxing reveal-date discipline. Neighbor-side: the condition $f<t$ on neighbors' fraud filings is replaced by $f<t+h$ for $h=1,2,3$ (admitting filings through year $t$, $t+1$ and $t+2$ respectively) or dropped entirely. Target-side: the own-firm indicator $\\mathbf{{1}}[c \\in F_\\infty]$ is added (ROC-AUC {Z['self_leak_clean']:.3f} against the ladder's leak-free {Z['clean']:.3f}).",
         c["study"], f"Sect.~6; {r['tab_leak']}; {r['fig_dose']}"),
        ("Leakage ladder (dose--response); rung",
         "The seven configurations of Sect.~6, one pipeline and one learner with increasing reveal-date violations in feature construction: the leak-free rung; three neighbor-side horizons; neighbor reveal dates ignored; and the two rungs adding the own-firm indicator. Each rung is compared with the ladder's own leak-free rung, not with the FIN$+$NET4 estimate of the main table.",
         c["study"], f"Sect.~6; {r['tab_leak']}; {r['fig_dose']}"),
        ("Temporal evaluation",
         f"Fixed split: train on fiscal years 2004--2017, test on 2018--2023 ({n(M['test_rows'])} test firm years, {M['test_positives']} positives); rolling-origin splits move the boundary across four test windows. No random cross-validation; point estimates use the original test distribution, and test firms are resampled only in the paired bootstrap used for uncertainty.",
         f"{c['dechow']}; {c['bao']} (temporal evaluation in prior work); {c['study']}", f"Sects.~4.1 (rule 3), 5.2; {r['fig_roll']}"),
        ("Rare-event metrics",
         "ROC-AUC: area under the curve of true-positive rate against false-positive rate across score thresholds. PR-AUC: average precision (\\texttt{average\\_precision\\_score}). Precision in the top one percent: the share of positives among the $k=\\lfloor 0.01\\,N_{\\mathrm{test}}\\rfloor$ highest-scored test firm years, selected as the first $k$ indices of \\texttt{numpy.argsort} applied to the negated score (default sort; equal scores are ordered by the algorithm, with no fractional allocation of ties). Accuracy, $(TP+TN)/N$, and F1, $2TP/(2TP+FP+FN)$, at a chosen threshold are metrics of the applied literature and are not used here.",
         f"{c['saito']} (rationale); {c['sklearn']} (implementation); {c['study']} (top-one-percent rule)", f"Sects.~2.2, 4.1 (rule 4), 5.2; {r['tab_main']}"),
        ("Industry-controlled comparison; industry-adjusted exposure",
         "The financial baseline augmented with one-digit Standard Industrial Classification (SIC) dummies (FIN$+$SIC), with NET4 added on top (FIN$+$SIC$+$NET4) or with the industry-adjusted exposure feature: the fraud-neighbor ratio minus its mean within the one-digit-SIC $\\times$ year cell, entered with degree and PageRank. One-digit codes are used because the fixed-split test contains few positives.",
         c["study"], "Sect.~5.3"),
        ("Industry-period risk set (proposed)",
         "Evaluation of the increment within groups of firms exposed to the same industry-period conditions. Named as future work; not implemented in this study.",
         c["study"], "Sects.~5.3, 7.4; Suppl.\\ Note~2"),
        ("Construction grid",
         "The six interlock-network constructions obtained by crossing the activity window $w\\in\\{3,4,5\\}$ with unweighted and weighted edges (weight = number of shared persons, used in degree and PageRank); the increment is re-estimated in each cell on that cell's merged sample.",
         c["study"], f"Sects.~4.2, 5.3; {r['fig_dec']}(a); Suppl.\\ Table~S1"),
        ("Person identification; homonym strictness",
         f"Directors are matched across firms by normalized full name (first, middle, last) because the director feed lacks usable person keys. A homonym merger is two or more distinct people sharing one normalized name key; the {n(J['name_matching']['names_5plus_firms'])} of {n(J['name_matching']['names'])} matched names attached to five or more firms is a diagnostic of this risk, not a verified false-merge rate. Strictness rules: baseline full-name matching; dropping names linked to five or more firms; requiring a non-empty middle name.",
         c["study"], f"Sects.~3.5, 5.3; {r['fig_dec']}(b); Suppl.\\ Note~2"),
        ("Measurement-error attenuation; reliability ratio",
         "Attenuation: a possible shrinkage of an estimated association toward zero under specified measurement-error assumptions. Under the classical additive model $X_{\\mathrm{obs}} = X^{*} + U$, with zero-mean error uncorrelated with $X^{*}$, the reliability ratio is $\\lambda = \\mathrm{Var}(X^{*})/[\\mathrm{Var}(X^{*})+\\mathrm{Var}(U)]$; in scalar linear regression with error also uncorrelated with the disturbance it is the slope-attenuation factor, and it is not a general correction for ROC-AUC increments. The monotone increase of the increment under stricter matching is described as consistent with attenuation; the study estimates neither $\\lambda$ nor an attenuation-corrected increment, and Supplementary Note~2 makes that estimation a target of the person-resolved replication.",
         f"{c['fuller']} (classical model); {c['study']} (interpretation; non-estimation)", f"Sects.~5.3, 7.5; {r['fig_dec']}(b); Suppl.\\ Note~2"),
        ("Inference: paired firm-cluster bootstrap; seed variation; power approximation",
         f"Bootstrap: {n(BOOT)} resamples draw test firms with replacement, keep all test firm years of each drawn firm together, and score both models on the same resample from their once-fitted predictions; the interval is the 2.5th--97.5th percentile of the resampled increment (resamples with fewer than five positives are skipped). Seed variation: {S['n']} learner seeds for training stochasticity. Power approximation: statistical power is the probability of rejecting the null of zero increment at a specified true increment; the increment detectable with 80 percent power in a two-sided 5 percent test is $2.80\\times$ a standard error inferred from the bootstrap interval as (upper $-$ lower)$/3.92$; a normal approximation, not a bootstrap power simulation.",
         c["study"], f"Sects.~4.3, 5.2, 7.5; {r['tab_main']}"),
        ("Missing-data specifications",
         "Zero-fill of absent financial items (primary, following Bao et al.\\ 2020); train-only median imputation; median imputation with missingness indicators; complete cases.",
         f"{c['bao']} (zero-fill); {c['study']} (alternatives)", "Sects.~4.3, 5.2, 7.5; Suppl.\\ Table~S2; Suppl.\\ Note~4"),
    ]


HEADER = r"""\begin{longtable}{>{\raggedright\arraybackslash}p{2.5cm}>{\raggedright\arraybackslash}p{6.5cm}>{\raggedright\arraybackslash}p{2.6cm}>{\raggedright\arraybackslash}p{2.6cm}}
"""


def body(main: bool):
    out = []
    for c, d, s, w in rows(main):
        out.append(f"{c} & {d} & {s} & {w} \\\\\n\\addlinespace")
    return "\n".join(out)


def fragment(main: bool):
    common = (r"Counts refer to the source-data universe or analysis subset identified in each row; methodological settings describe the stated specifications. "
              r"Sources marked ``This study'' identify study-specific operational definitions or choices; cited works supply the concept, practice or implementation adopted, as labeled. "
              r"Cross-references point to the main text unless marked Supplementary.")
    cap_main = r"\caption{Key concepts of the study: definition, source of the definition, and where each is applied. " + common + r"}\label{tab:defs}\\"
    cap_supp = (r"\caption*{\textbf{Supplementary Table S3.} Key concepts of the study: definition, source of the definition, and where each is applied; the row content reproduces Table~3 of the main text with sources given as author--year or standard identifiers. "
                + common + r"}\\")
    head = r"\toprule Concept & Definition in this study & Source & Where applied \\ \midrule \endfirsthead" + "\n" + \
           r"\multicolumn{4}{l}{\footnotesize\emph{(continued)}}\\ \toprule Concept & Definition in this study & Source & Where applied \\ \midrule \endhead" + "\n" + \
           r"\midrule \multicolumn{4}{r}{\footnotesize\emph{continued on next page}}\\ \endfoot" + "\n" + r"\bottomrule \endlastfoot" + "\n"
    return "{\\footnotesize\n" + HEADER + (cap_main if main else cap_supp) + "\n" + head + body(main) + "\n\\end{longtable}}\n"


for name, main in (("tab_defs.tex", True), ("tab_defs_supp.tex", False)):
    with open(f"{OUT}/{name}", "w") as f:
        f.write("% AUTO-GENERATED by code/23_make_definitions_table.py from results/paper_numbers.json\n"
                "% DO NOT EDIT BY HAND — regenerate instead.\n" + fragment(main))
    print("wrote", f"{OUT}/{name}")

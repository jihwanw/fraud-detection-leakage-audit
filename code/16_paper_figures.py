#!/usr/bin/env python3
"""16_paper_figures.py — Publication figures from verified experiment logs.
All numbers traceable to data/*.log (referee.log, grid.log, final_ref*.log, final.log).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import json

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(BASE, "paper", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.dpi": 200, "savefig.bbox": "tight"})
J = json.load(open(os.path.join(BASE, "results", "paper_numbers.json")))

# ---- Fig 1: leakage dose-response (referee.log [B] + leak.log self-leak) ----
labels = ["Leak-free\n(reveal $<t$)", "reveal\n$<t{+}1$", "reveal\n$<t{+}2$",
          "reveal\n$<t{+}3$", "All fraud\n(dates ignored)", "+ own future\nrevelation"]
D = J["dose_response"]
vals = [D["clean"], D["t1"], D["t2"], D["t3"], D["all"], D["self_leak"]]
colors = ["#2b7a3e"] + ["#b8a83a"] * 4 + ["#b03030"]
fig, ax = plt.subplots(figsize=(8, 4.2))
bars = ax.bar(range(6), vals, color=colors)
ax.axhline(J["fixed_split"]["FIN"]["roc"], ls="--", c="gray", lw=1)
ax.text(5.45, 0.635, "FIN baseline", ha="right", fontsize=9, color="gray")
for i, v in enumerate(vals):
    ax.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=9)
ax.set_ylim(0.6, 1.03)
ax.set_ylabel("Test ROC-AUC")
ax.set_xticks(range(6))
ax.set_xticklabels(labels, fontsize=9)
ax.set_title("Leakage dose-response: one pipeline, increasing reveal-date violations")
plt.savefig(f"{FIG}/fig1_dose_response.pdf")
plt.close()

# ---- Fig 2: construction grid (grid.log) ----
G = J["grid"]
grid = np.array([[G["w3_unw"], G["w3_wt"]], [G["w4_unw"], G["w4_wt"]], [G["w5_unw"], G["w5_wt"]]])
fig, ax = plt.subplots(figsize=(5.2, 4))
im = ax.imshow(grid, cmap="RdBu_r", vmin=-0.06, vmax=0.06)
for i in range(3):
    for j in range(2):
        ax.text(j, i, f"{grid[i, j]:+.3f}", ha="center", va="center", fontsize=12,
                color="black")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Unweighted", "Weighted"])
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(["3-year\nwindow", "4-year\nwindow", "5-year\nwindow"])
ax.set_title("Network increment ($\\Delta$ROC-AUC) across\nconstruction choices")
plt.colorbar(im, ax=ax, shrink=0.8, label="$\\Delta$ROC-AUC (NET4 $-$ FIN)")
plt.savefig(f"{FIG}/fig2_construction_grid.pdf")
plt.close()

# ---- Fig 3: rolling windows (final.log) ----
wins = ["2012-14\n(74 pos)", "2015-17\n(51 pos)", "2018-20\n(25 pos)", "2021-23\n(20 pos)"]
fin = J["rolling"]["fin_roc"]
d4 = J["rolling"]["d_net4"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8))
a1.plot(range(4), fin, "o-", color="#333", label="FIN baseline")
a1.plot(range(4), [f + d for f, d in zip(fin, d4)], "s--", color="#2b5fa8",
        label="FIN+NET4")
a1.set_xticks(range(4)); a1.set_xticklabels(wins, fontsize=9)
a1.set_ylabel("Test ROC-AUC"); a1.legend(fontsize=9)
a1.set_title("(a) Rolling-origin performance")
a2.bar(range(4), d4, color=["#b03030" if x < 0 else "#2b7a3e" for x in d4])
a2.axhline(0, c="gray", lw=1)
a2.set_xticks(range(4)); a2.set_xticklabels(wins, fontsize=9)
a2.set_ylabel("$\\Delta$ROC-AUC")
a2.set_title("(b) Network increment by window")
plt.savefig(f"{FIG}/fig3_rolling.pdf")
plt.close()

# ---- Fig 4: homonym monotonicity (final_ref.log [D]) ----
rules = ["Full-name\nmatching", "Drop names\nat $\\geq$5 firms", "Require\nmiddle name"]
H = J["homonym_common"]
dd = [H["V0"]["d"], H["V1"]["d"], H["V2"]["d"]]
fig, ax = plt.subplots(figsize=(5.5, 3.6))
ax.bar(range(3), dd, color=["#b03030" if x < 0 else "#2b7a3e" for x in dd])
ax.axhline(0, c="gray", lw=1)
for i, v in enumerate(dd):
    ax.text(i, v + (0.001 if v > 0 else -0.0025), f"{v:+.3f}", ha="center", fontsize=10)
ax.set_xticks(range(3)); ax.set_xticklabels(rules, fontsize=9)
ax.set_ylabel("$\\Delta$ROC-AUC (common sample)")
ax.set_title("Stricter person identification $\\Rightarrow$ larger increment\n(measurement-error attenuation)")
plt.savefig(f"{FIG}/fig4_homonym.pdf")
plt.close()

print("figures saved:", os.listdir(FIG))

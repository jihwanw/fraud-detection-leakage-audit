# Final Reference Audit — 2026-09-04

## Result

**PASS: 45/45 references accounted for; no fabricated or undefined reference remains.**

- 45 unique bibliography entries.
- 45 unique in-text citation keys.
- 0 undefined citations.
- 0 uncited bibliography entries.
- First-citation order matches bibliography order exactly, as required for numbered Nature-style references.

## External verification

- Scholarly article titles were queried against Crossref/OpenAlex; 39 title-level matches were returned automatically.
- Apparent year mismatches were online-first or working-paper dates selected by the API rather than errors in the manuscript's final issue year.
- Journal volume, issue, and page fields were compared against Crossref records. Entries for which Crossref ranked an SSRN version, review, digest, or book chapter above the final publication were checked against the final publisher metadata.
- Kaufman et al. (2012), which the first title parser missed, was verified directly by DOI `10.1145/2382577.2382579`: *ACM Transactions on Knowledge Discovery from Data*, 6(4), 1–21.
- Non-Crossref sources were verified against official sources: McCrum's *Money Men* (2022), ACFE's *Occupational Fraud 2024*, the U.S. Treasury release of 17 October 2024, Regulation (EU) 2024/1689, and the SEC/Federal Register withdrawal notice (90 FR 25531–25533).

## Corrections made during final audit

- Added the official 2025 Federal Register notice documenting withdrawal of the SEC predictive-data-analytics proposal.
- Revised the policy discussion to state that the proposal was withdrawn rather than pending.
- Revised the EU AI Act discussion to acknowledge the specific financial-fraud-detection exception in Annex III's creditworthiness category.
- Corrected the Treasury description from fraud alone to “fraud and improper payments.”

Machine-readable intermediate reports are stored in `results/reference_audit.json` and `results/reference_field_audit.json`; API false positives were resolved using the official sources described above.

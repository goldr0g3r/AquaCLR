# AquaCLR / LEGION-DeSnow — M.Tech Dissertation

This folder contains the chapter-per-file source of the dissertation
manuscript. The compiled master entry point is
[`../../DISSERTATION.md`](../../DISSERTATION.md).

## Chapter manifest

| # | File | Title | Approx. lines |
| --- | --- | --- | --- |
| 01 | [`01_introduction.md`](01_introduction.md) | Introduction | ~270 |
| 02 | [`02_background.md`](02_background.md) | Background & Literature Review | ~360 |
| 03 | [`03_theory.md`](03_theory.md) | Theoretical Foundation | ~430 |
| 04 | [`04_architecture.md`](04_architecture.md) | System Architecture | ~440 |
| 05 | [`05_implementation.md`](05_implementation.md) | Implementation | ~430 |
| 06 | [`06_datasets.md`](06_datasets.md) | Datasets and Methodology | ~370 |
| 07 | [`07_training.md`](07_training.md) | Training Methodology | ~440 |
| 08 | [`08_evaluation.md`](08_evaluation.md) | Evaluation Methodology | ~330 |
| 09 | [`09_deployment.md`](09_deployment.md) | Deployment | ~360 |
| 10 | [`10_results.md`](10_results.md) | Results, Discussion, Limitations | ~290 |
| 11 | [`11_conclusion.md`](11_conclusion.md) | Conclusion & Future Work | ~190 |
| 12 | [`12_references.md`](12_references.md) | References | ~330 |
| A | [`A_math.md`](A_math.md) | Math Derivations | ~140 |
| B | [`B_glossary.md`](B_glossary.md) | Glossary | ~200 |
| C | [`C_code_reference.md`](C_code_reference.md) | Code Reference | ~260 |
| D | [`D_reproducibility.md`](D_reproducibility.md) | Reproducibility Checklist | ~180 |

Total: ~5,000 lines of substantive prose + diagrams + tables across
16 files.

## Conventions

Every chapter follows the same hybrid Diátaxis + Cornell + Microsoft
Style structure:

1. **Learning objectives** at the top.
2. **TL;DR** below them.
3. Numbered sections (`§N.M`) for navigation.
4. Mermaid diagrams inline (renderable in GitHub, exportable via
   `mermaid-cli`).
5. **Asides** (`> **Aside:**`) for context that's interesting but
   skippable.
6. **Pitfalls** (`> **Pitfall:**`) for traps and how to avoid them.
7. **Worked examples** with concrete numbers.
8. **Key takeaways** block at chapter end (suitable for spaced
   revision).
9. **Cross-references** to neighbouring chapters and code.

## Compilation to a single PDF

A helper script is provided at
[`../../scripts/build_dissertation.sh`](../../scripts/build_dissertation.sh).
It concatenates the chapters in order, renders mermaid blocks via
`mermaid-cli`, and runs pandoc with thesis-friendly settings.

```bash
# From the project root:
bash scripts/build_dissertation.sh
# produces DISSERTATION.pdf
```

For a quick PDF without thesis styling, the VS Code "Markdown PDF"
extension (`yzane.markdown-pdf`) works well: open
`DISSERTATION_FULL.md` (the concatenated file the script generates),
then `right-click → Markdown PDF: Export (pdf)`.

## Reading order recommendations

| Reader | Suggested route |
| --- | --- |
| Examiner / external reviewer | Straight through, Ch. 1 → Appendix D |
| Practitioner | Ch. 1 → Ch. 4 → Ch. 9 → Appendix C |
| Reproducer | Ch. 6 → Ch. 7 → Ch. 8 → Appendix D |
| Quickstart user | Project [`README.md`](../../README.md) + Ch. 9 |

## Citation conventions

Inline citations use abbreviated tags `[Author Year]` (e.g.
`[Sato 2023]`); full BibTeX entries are in [`12_references.md`](12_references.md).
Code citations are markdown links to the file path with optional
line range:

- `[`src/aquaclr/models/model.py`](../../src/aquaclr/models/model.py)`
- `[`docs/DEPLOYMENT_FEDORA.md`](../DEPLOYMENT_FEDORA.md)`

## Update protocol

When updating any chapter:

1. Bump the line-count entry in this index if substantial.
2. Re-run `bash scripts/build_dissertation.sh` to verify the merged
   PDF still compiles.
3. Add a short entry to the dissertation changelog (when one exists
   — currently the dissertation is a single submission artefact).

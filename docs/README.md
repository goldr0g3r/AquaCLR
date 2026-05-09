# AquaCLR / Project LEGION — Documentation Index

| Doc | What's in it |
| --- | --- |
| [`../README.md`](../README.md) | Project overview, quickstart, the canonical pointer to everything else. |
| [`../DISSERTATION.md`](../DISSERTATION.md) | **M.Tech dissertation master entry point** with full TOC and submission guide. |
| [`../MODEL_CARD.md`](../MODEL_CARD.md) | Intended use, training data, evaluation, ethical considerations, dataset bias. |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Long-form **why** of every design decision: physics, network, loss, datasets, tooling, perf budget. |
| [`DEPLOYMENT_FEDORA.md`](DEPLOYMENT_FEDORA.md) | End-to-end runbook for **Fedora 44 host + Ubuntu 24.04 / ROS2 Jazzy container** (distrobox or pure podman) with NVIDIA passthrough. |
| [`dissertation/`](dissertation/) | The 12 chapters + 4 appendices that make up the full M.Tech dissertation. |
| [`../CONTRIBUTING.md`](../CONTRIBUTING.md) | Branching, commit style, lint/type/test bar, dataset-contribution guide. |

## Suggested reading order

1. **Just want to run it?** Start at the project [`README.md`](../README.md),
   skip to *Quickstart*.
2. **On Fedora?** Jump straight to [`DEPLOYMENT_FEDORA.md`](DEPLOYMENT_FEDORA.md).
3. **Reading the dissertation?** Open
   [`../DISSERTATION.md`](../DISSERTATION.md) and follow its TOC into
   [`dissertation/01_introduction.md`](dissertation/01_introduction.md).
4. **Reviewing a PR or designing M2?** Read [`ARCHITECTURE.md`](ARCHITECTURE.md)
   end-to-end — every non-trivial choice is justified there.
5. **Citing or auditing the model?** [`../MODEL_CARD.md`](../MODEL_CARD.md)
   has the intended-use and limitations sections.
6. **Reproducing every number?** [`dissertation/D_reproducibility.md`](dissertation/D_reproducibility.md).

## Convention

All non-trivial public APIs in `src/aquaclr/` carry a Google-style
docstring with an explicit **"Automotive SiL parallel"** paragraph
that maps the underwater concept onto its automotive ADAS analogue.
This is intentional cross-pollination — Project LEGION's perception
team comes from automotive backgrounds, and several M2 stretch goals
involve back-porting LEGION-DeSnow to a rain-augmented automotive
dataset.

## Building the dissertation as a PDF

```bash
# From the project root:
bash scripts/build_dissertation.sh           # produces DISSERTATION.pdf
bash scripts/build_dissertation.sh --no-pdf  # only concatenate to DISSERTATION_FULL.md
```

Requires `pandoc` and (optionally) `pandoc-mermaid-filter` /
`mermaid-cli`. See the script's header for installation hints.

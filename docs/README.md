# AquaCLR / Project LEGION — Documentation Index

| Doc | What's in it |
| --- | --- |
| [`../README.md`](../README.md) | Project overview, quickstart, and the canonical pointer to everything else. |
| [`../MODEL_CARD.md`](../MODEL_CARD.md) | Intended use, training data, evaluation, ethical considerations, dataset bias. |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Long-form **why** of every design decision: physics, network, loss, datasets, tooling, perf budget. |
| [`DEPLOYMENT_FEDORA.md`](DEPLOYMENT_FEDORA.md) | End-to-end runbook for **Fedora 44 host + Ubuntu 24.04 / ROS2 Jazzy container** (distrobox or pure podman) with NVIDIA passthrough. |
| [`../CONTRIBUTING.md`](../CONTRIBUTING.md) | Branching, commit style, lint/type/test bar, dataset-contribution guide. |

## Suggested reading order

1. **Just want to run it?** Start at the project [`README.md`](../README.md),
   skip to *Quickstart*.
2. **On Fedora?** Jump straight to [`DEPLOYMENT_FEDORA.md`](DEPLOYMENT_FEDORA.md).
3. **Reviewing a PR or designing M2?** Read [`ARCHITECTURE.md`](ARCHITECTURE.md)
   end-to-end — every non-trivial choice is justified there.
4. **Citing or auditing the model?** [`MODEL_CARD.md`](../MODEL_CARD.md)
   has the intended-use and limitations sections.

## Convention

All non-trivial public APIs in `src/aquaclr/` carry a Google-style
docstring with an explicit **"Automotive SiL parallel"** paragraph
that maps the underwater concept onto its automotive ADAS analogue.
This is intentional cross-pollination — Project LEGION's perception
team comes from automotive backgrounds, and several M2 stretch goals
involve back-porting LEGION-DeSnow to a rain-augmented automotive
dataset.

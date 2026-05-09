#!/usr/bin/env bash
# Build the dissertation as a single PDF / standalone Markdown.
#
# Outputs:
#   DISSERTATION_FULL.md   - chapters concatenated in order (intermediate)
#   DISSERTATION.pdf       - final PDF (if pandoc + LaTeX are installed)
#
# Required tools:
#   - pandoc (Markdown -> PDF). On Fedora: sudo dnf install -y pandoc texlive-scheme-medium
#   - mermaid-cli (mermaid blocks -> PNG). Optional but recommended.
#       npm install -g @mermaid-js/mermaid-cli
#
# Usage:
#   bash scripts/build_dissertation.sh
#   bash scripts/build_dissertation.sh --no-pdf      # only build the .md
#   bash scripts/build_dissertation.sh --eisvogel    # use the Eisvogel pandoc template

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DISS_DIR="${ROOT}/docs/dissertation"
OUT_MD="${ROOT}/DISSERTATION_FULL.md"
OUT_PDF="${ROOT}/DISSERTATION.pdf"

USE_EISVOGEL=0
BUILD_PDF=1
for arg in "$@"; do
    case "$arg" in
        --no-pdf) BUILD_PDF=0 ;;
        --eisvogel) USE_EISVOGEL=1 ;;
        -h|--help)
            sed -n '1,/^set -euo/p' "$0"
            exit 0
            ;;
        *) echo "Unknown arg: $arg" >&2; exit 1 ;;
    esac
done

# Chapters in canonical order. Title page comes from DISSERTATION.md.
CHAPTERS=(
    "${ROOT}/DISSERTATION.md"
    "${DISS_DIR}/01_introduction.md"
    "${DISS_DIR}/02_background.md"
    "${DISS_DIR}/03_theory.md"
    "${DISS_DIR}/04_architecture.md"
    "${DISS_DIR}/05_implementation.md"
    "${DISS_DIR}/06_datasets.md"
    "${DISS_DIR}/07_training.md"
    "${DISS_DIR}/08_evaluation.md"
    "${DISS_DIR}/09_deployment.md"
    "${DISS_DIR}/10_results.md"
    "${DISS_DIR}/11_conclusion.md"
    "${DISS_DIR}/12_references.md"
    "${DISS_DIR}/A_math.md"
    "${DISS_DIR}/B_glossary.md"
    "${DISS_DIR}/C_code_reference.md"
    "${DISS_DIR}/D_reproducibility.md"
)

echo "[build] concatenating ${#CHAPTERS[@]} chapters -> ${OUT_MD}"
{
    for f in "${CHAPTERS[@]}"; do
        if [[ ! -f "$f" ]]; then
            echo "[build] WARN: missing $f, skipping" >&2
            continue
        fi
        # Page break between chapters in the PDF output.
        printf '\n\n\\newpage\n\n'
        cat "$f"
    done
} > "${OUT_MD}"

echo "[build] wrote $(wc -l < "${OUT_MD}") lines to ${OUT_MD}"

if [[ "${BUILD_PDF}" -eq 0 ]]; then
    echo "[build] --no-pdf set; stopping after Markdown concat"
    exit 0
fi

if ! command -v pandoc >/dev/null 2>&1; then
    echo "[build] pandoc not found. Install on Fedora with:"
    echo "          sudo dnf install -y pandoc texlive-scheme-medium"
    exit 1
fi

PANDOC_ARGS=(
    "${OUT_MD}"
    "-o" "${OUT_PDF}"
    "--from=gfm+yaml_metadata_block"
    "--standalone"
    "--toc"
    "--toc-depth=3"
    "--number-sections"
    "--metadata" "title=AquaCLR / LEGION-DeSnow — M.Tech Dissertation"
    "--metadata" "author=Project LEGION"
    "--metadata" "date=$(date +%Y-%m-%d)"
    "--variable=papersize:a4"
    "--variable=geometry:margin=2.5cm"
    "--variable=fontsize:11pt"
    "--variable=linkcolor:blue"
    "--variable=urlcolor:blue"
    "--pdf-engine=xelatex"
    "--filter" "pandoc-mermaid-filter"
)

if [[ "${USE_EISVOGEL}" -eq 1 ]]; then
    PANDOC_ARGS+=("--template=eisvogel")
fi

# pandoc-mermaid-filter is optional; if missing, drop the filter arg.
if ! command -v pandoc-mermaid-filter >/dev/null 2>&1; then
    echo "[build] pandoc-mermaid-filter not found; mermaid blocks will appear as code"
    NEW_ARGS=()
    skip_next=0
    for a in "${PANDOC_ARGS[@]}"; do
        if [[ "$skip_next" -eq 1 ]]; then
            skip_next=0
            continue
        fi
        if [[ "$a" == "--filter" ]]; then
            skip_next=1
            continue
        fi
        NEW_ARGS+=("$a")
    done
    PANDOC_ARGS=("${NEW_ARGS[@]}")
fi

echo "[build] running pandoc..."
pandoc "${PANDOC_ARGS[@]}"
echo "[build] wrote ${OUT_PDF}"

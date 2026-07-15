# TTMD6 waveform relative-reliability analysis

[![CI](https://github.com/hmk-228674/TTMD6-functional-G-theory-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/hmk-228674/TTMD6-functional-G-theory-analysis/actions/workflows/ci.yml)

This repository contains the complete analysis code, selected reference outputs, figure source data, and publication-ready figures for a secondary analysis of the TTMD6 table-tennis motion-capture archive.

The action-specific estimand is the $L^2$-trace relative reliability

$$
R_{L^2,a}(n)=\frac{B_a}{B_a+W_a/n},
$$

where $B_a$ and $W_a$ are the integrated between-athlete and within-athlete variance components for fixed action $a$. Generalizability theory motivates the $W_a/n$ design relation, but this trace ratio is **not presented as a standard functional G coefficient**.

For the detailed Chinese instructions, see [docs/README_zh-CN.md](docs/README_zh-CN.md).

## Reproducibility status

- Clean end-to-end run: **PASS** on 2026-07-15 (UTC).
- Whole-athlete bootstrap: 5,000 replicates.
- Within-cell balanced resampling: 1,000 replicates.
- REML numerical self-tests: 13/13 passed.
- Figure QA: 7/7 passed; fixed 162 mm width, 600 dpi LZW TIFF, editable SVG text, and zero detected text-boundary violations.
- Random seed: `20260712` (assumption diagnostics use the separately recorded seed `20260713`).

The machine-readable record is [reference_results/reproduction/REPRODUCTION_STATUS.json](reference_results/reproduction/REPRODUCTION_STATUS.json). The archived reference outputs are checks against a known successful run; the pipeline does not read them when reproducing the analysis.

## Data provenance and scope

The primary publication reports 30 participants and 9,000 TTMD6 strokes. The archive examined here contains 12,000 paired racket/body files indexed by codes 1–40. The analysis therefore applies a deterministic rule:

- codes 1–30 form the primary 9,000-pair analysis block;
- codes 31–40 are quarantined because their provenance is not explained in the primary article;
- this operational rule does **not** claim that codes 1–30 have been author-confirmed as a one-to-one mapping to the 30 reported participants.

Additional disclosed archive issues include 50 duplicated 400-row blocks for each waveform family, eight exact paired-record matches spanning the primary and quarantined blocks, 1,180 primary trials with nominal lengths above 200 frames, and structural zero triplets in 190 body trials. These conditions are audited explicitly and examined through sensitivity analyses; they are not silently discarded.

The third-party coordinate archive is **not redistributed** in this repository. Obtain it lawfully from the [primary Scientific Reports article](https://doi.org/10.1038/s41598-024-54150-5) or its linked data record. The pipeline accepts only this exact input identity:

| Property | Required value |
|---|---|
| Filename | `TTMD6.rar` |
| Bytes | `341074031` |
| MD5 | `1c9ce9cbf79dd35dd22f16a7199e2a8c` |
| SHA-256 | `93d1b52a470f14b9dc0ba0600959bff921be891a3da1b71e609bd328224b354d` |

The current Figshare record (concept DOI [10.6084/m9.figshare.31746358](https://doi.org/10.6084/m9.figshare.31746358)) has been reported with a different byte count. Do not assume byte identity across records; verify the hash before analysis. The program stops before extraction if the input identity differs.

## Run the analysis

Requirements: Python 3.10 or newer, a `bsdtar` build with RAR5 support, and the packages in `requirements.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python reproduce_all.py \
  --rar /absolute/path/TTMD6.rar \
  --out /absolute/path/ttmd6_reproduced
```

The output directory must be new or empty. A successful run ends with `Reproduction PASS` and writes `REPRODUCTION_STATUS.json`, `FILE_INVENTORY.csv`, all analysis tables, and all seven figures. For a non-analytical input/dependency check:

```bash
python reproduce_all.py \
  --rar /absolute/path/TTMD6.rar \
  --out /absolute/path/ttmd6_reproduced \
  --plan-only
```

Changing `--n-bootstrap` or `--n-balanced-resamples` is intended only for development smoke tests. Manuscript results use the defaults of 5,000 and 1,000.

## Repository map

| Path | Contents |
|---|---|
| `reproduce_all.py` | Single validated entry point |
| `scripts/` | Archive audit, waveform derivation, REML, bootstrap, sensitivity analyses, and figure generation |
| `tests/` | Data-free mathematical and repository-contract tests |
| `reference_results/` | Selected summaries from the clean default run; no raw coordinates or cached arrays |
| `figure_source_data/` | Machine-readable source data and automated figure-QA records |
| `figures/` | PNG, PDF, editable SVG, and 600 dpi LZW TIFF exports |
| `docs/` | Methods, provenance, output dictionary, and Chinese reproduction guide |

## Inferential boundaries

- The independent unit for population-level uncertainty is the athlete code, not the individual trial.
- The six archive labels are treated as a fixed finite set, not as a random sample of all table-tennis techniques.
- The waveform is displacement magnitude per adjacent frame; it is not physical speed because no frame interval is applied.
- The body representation is the mean displacement magnitude across available joints; it is not centre-of-mass motion or a coordination metric.
- Archive rank is only an ordering proxy and is not asserted to be verified acquisition time.
- Results describe within-session relative distinguishability for this archive and do not establish cross-day reliability, absolute agreement, responsiveness, or classification performance.

## Licensing

Original code is released under the [MIT License](LICENSE). Documentation, figures, and derived tabular outputs created for this analysis are released under [CC BY 4.0](LICENSE-DATA). Neither license applies to the third-party TTMD6 raw archive.

## Citation

Until the associated article receives its final bibliographic record, cite this repository by title, owner, version or commit, URL, and access date. Release metadata are provided in `CITATION.cff`.

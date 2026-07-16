# TTMD6 waveform relative-reliability analysis

[![CI](https://github.com/hmk-228674/TTMD6-functional-G-theory-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/hmk-228674/TTMD6-functional-G-theory-analysis/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21382966.svg)](https://doi.org/10.5281/zenodo.21382966)

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
- Release status: `v1.0.0` is published from commit `f5c0562d8b0abfe79cbd20971efc6dc2ea6fd022`; the version-specific DOI is [10.5281/zenodo.21382967](https://doi.org/10.5281/zenodo.21382967), and the all-versions DOI is [10.5281/zenodo.21382966](https://doi.org/10.5281/zenodo.21382966).

The machine-readable record is [reference_results/reproduction/REPRODUCTION_STATUS.json](reference_results/reproduction/REPRODUCTION_STATUS.json). The archived reference outputs are checks against a known successful run; the pipeline does not read them when reproducing the analysis.

## Data provenance and scope

The primary publication reports 30 participants and 9,000 TTMD6 strokes. The archive examined here contains 12,000 paired racket/body files indexed by codes 1–40. The analysis therefore applies a deterministic rule:

- codes 1–30 form the primary 9,000-pair analysis block;
- codes 31–40 are quarantined because their provenance is not explained in the primary article;
- this operational rule does **not** claim that codes 1–30 have been author-confirmed as a one-to-one mapping to the 30 reported participants.

Additional disclosed archive issues include 50 duplicated 400-row blocks for each waveform family, eight exact paired-record matches spanning the primary and quarantined blocks, 1,180 primary trials with nominal lengths above 200 frames, and structural zero triplets in 190 body trials. These conditions are audited explicitly and examined through sensitivity analyses; they are not silently discarded.

The third-party coordinate archive is **not redistributed** in this repository. The exact archive analyzed here is publicly downloadable from the publisher-hosted [Scientific Reports Supplementary Information](https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-024-54150-5/MediaObjects/41598_2024_54150_MOESM1_ESM.rar) associated with the [primary article](https://doi.org/10.1038/s41598-024-54150-5). The pipeline accepts only this exact input identity:

| Property | Required value |
|---|---|
| Filename | `TTMD6.rar` |
| Bytes | `341074031` |
| MD5 | `1c9ce9cbf79dd35dd22f16a7199e2a8c` |
| SHA-256 | `93d1b52a470f14b9dc0ba0600959bff921be891a3da1b71e609bd328224b354d` |

On 2026-07-15, the publisher-hosted file was downloaded in full and compared with the local analysis input. Its byte count, MD5, and SHA-256 matched the values above, and a byte-for-byte comparison returned no differences. This verifies that the official Springer Nature supplementary file is the exact archive analyzed here, irrespective of the download filename.

A later Figshare record (concept DOI [10.6084/m9.figshare.31746358](https://doi.org/10.6084/m9.figshare.31746358)) has a different byte count and was not used for the archived results. Do not substitute a similarly named or later-version file without verifying the exact identity above. The program stops before extraction if the byte count or either digest differs.

## Run the analysis

Exact reproduction of the archived results requires Python 3.12, a `bsdtar` build with RAR5 support, and the packages recorded in `requirements-lock.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt

python reproduce_all.py \
  --rar /absolute/path/TTMD6.rar \
  --out /absolute/path/ttmd6_reproduced
```

For exact reproduction of the archived manuscript results, use Python 3.12 and `requirements-lock.txt`; the recorded clean run used Python 3.12.13. The broader `requirements.txt` defines supported dependency ranges for Python 3.10 or newer and is used for compatibility-oriented testing. It is not a byte-for-byte specification of the archived software environment.

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

## License

Source code in this repository is licensed under the [MIT License](LICENSE). Author-generated documentation, figures, figure source data, and derived tabular outputs are licensed under the [Creative Commons Attribution 4.0 International License](LICENSE-DATA). The third-party TTMD6 raw archive is not redistributed by this repository and is not covered by either repository license.

## Citation

For exact reproduction of the manuscript analysis, cite the frozen `v1.0.0` record:

> Han, M. (2026). *TTMD6 waveform relative-reliability analysis* (Version v1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.21382967

Use [10.5281/zenodo.21382966](https://doi.org/10.5281/zenodo.21382966) only when citing the evolving software record across all versions. The version DOI above resolves permanently to the exact archived release used for the manuscript; `CITATION.cff` contains the corresponding machine-readable metadata.

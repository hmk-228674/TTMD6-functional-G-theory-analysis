# Data provenance and archive decisions

## Source record

The analysis is a secondary analysis of the TTMD6 coordinate archive associated with:

> Table tennis motion recognition based on the bat trajectory using varying-length-input convolution neural networks. *Scientific Reports* (2024). https://doi.org/10.1038/s41598-024-54150-5

The raw archive is third-party material and is not redistributed here. The exact analyzed archive is available from the publisher-hosted [Scientific Reports Supplementary Information](https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-024-54150-5/MediaObjects/41598_2024_54150_MOESM1_ESM.rar). The accepted input is fixed by byte count and two hashes in `reproduce_all.py`; a mismatch stops the workflow before extraction.

## Exact publisher-file verification

On 2026-07-15, the Springer Nature supplementary file was downloaded in full and compared with the local analysis input. The two files were identical under all four checks:

- byte count: `341074031`;
- MD5: `1c9ce9cbf79dd35dd22f16a7199e2a8c`;
- SHA-256: `93d1b52a470f14b9dc0ba0600959bff921be891a3da1b71e609bd328224b354d`;
- direct byte-for-byte comparison: no differences.

The publisher download name (`41598_2024_54150_MOESM1_ESM.rar`) and the local analysis name (`TTMD6.rar`) refer to the same byte sequence. The archive is linked here rather than copied into the repository, so readers can obtain it from the source publication while independently checking the exact analyzed identity.

## Reported cohort versus archive structure

The source article reports 30 participants and 9,000 strokes. The examined archive contains 12,000 racket files and 12,000 body files organized as 40 numeric codes × 6 action labels × 50 trials.

The analysis therefore uses a predeclared deterministic operational rule:

1. Codes 1–30 form the primary analysis block (30 × 6 × 50 = 9,000 paired trials).
2. Codes 31–40 are quarantined and are never silently included in the primary inferential analysis.
3. The rule matches the reported participant count but is not represented as an author-verified identity mapping.

This unresolved provenance is an inferential limitation, not merely a file-format detail.

## Participant-code robustness scenarios

Because the public materials do not contain a participant-code crosswalk, the repository does not try to infer identities from numeric order. In addition to the operational codes 1–30 primary block, `v1.0.1` runs three explicitly descriptive scenarios:

1. the same-width codes 11–40 window;
2. all complete archive codes 1–40;
3. all codes after removing the codes 31–40-side record from each of the eight exact paired-record duplicate groups.

The last scenario uses unbalanced profile REML after the eight record removals. Across the 12 racket/body action conditions, the alternative 30-code window changes the primary $L^2$-trace 0.90 integer threshold by 0–4 trials. The all-code and exact-pair-deduplicated scenarios have identical integer thresholds under both the trace and phase-mean pointwise summaries. These results show that the eight exact duplicates do not drive the reported integer conclusions; they do not prove that all 40 codes are distinct participants or that codes 1–30 are the 30 recruited participants. Machine-readable outputs are in `reference_results/cohort_estimand/`.

## Archive audit findings

- 50 racket and 50 body files contain exact duplicated 200-row halves within 400-row matrices; the duplicated half is deterministically collapsed.
- Eight exact paired-record groups span primary and quarantined codes; no exact whole-trial duplicates occur entirely inside the primary block.
- The filename nominal length exceeds the public 200-row matrix boundary for 1,180 primary trials. The main analysis uses the observable 200 rows and does not claim to reconstruct unavailable frames.
- Structural `[0,0,0]` joint triplets affect 190 body trials. For each adjacent-frame interval, a joint is omitted when either endpoint is structural zero; remaining joint displacement magnitudes are averaged. No coordinate gap filling is performed.
- The public archive contains 50 trials per cell, whereas the source description refers to 55 consecutive strokes. The selection mechanism for the five unpublished strokes is unknown.

Every item above is emitted in machine-readable audit tables during reproduction.

## Data-record caveat

The later Figshare record at https://doi.org/10.6084/m9.figshare.31746358 has a byte count different from the verified publisher archive and was not used for the archived results. A shared dataset title or concept DOI is not proof of byte identity. Users must verify the exact archive hash.

The analyzed archive is identified by byte count, MD5, and SHA-256; the expected local filename is documented for clarity but does not override content identity. If another source record supplies a byte-distinct archive, the workflow deliberately stops before extraction and does not treat the files as analytically interchangeable. This is an explicit provenance and version boundary: the repository verifies the identity of the input actually analyzed but does not claim that later TTMD6 distributions are byte-identical.

## Software and derived-output archive

The historical analysis release is GitHub tag `v1.0.0` at commit `f5c0562d8b0abfe79cbd20971efc6dc2ea6fd022`. Zenodo archived that release as [version DOI 10.5281/zenodo.21382967](https://doi.org/10.5281/zenodo.21382967). It is not overwritten or retagged. The revised manuscript analysis is released separately as `v1.0.1` with version DOI [10.5281/zenodo.21400718](https://doi.org/10.5281/zenodo.21400718) and its own immutable commit. [Concept DOI 10.5281/zenodo.21382966](https://doi.org/10.5281/zenodo.21382966) resolves to the latest software version and should not replace the matching version DOI when exact reproduction is required.

The Zenodo release file `hmk-228674/TTMD6-functional-G-theory-analysis-v1.0.0.zip` was downloaded and validated after publication:

- byte count: `9201472`;
- MD5: `81e2dbd99d85ce45d18dbb8d60aa6438` (matching the checksum displayed by Zenodo);
- SHA-256: `6da90f49aaebe52662f94f14b55c7e2b5126f125f3ab3b6fd25beb53c96c2230`;
- ZIP integrity test: passed with no errors.

The `v1.0.0` checksum values above identify only that historical Zenodo ZIP; they must not be copied into `v1.0.1` metadata. Each software archive contains code, documentation, figures, figure source data, and selected derived results but not the third-party TTMD6 raw coordinate archive. Raw-input identity and software-release identity are deliberately documented as separate provenance layers.

## Privacy and ethics boundary

The repository contains no names, direct identifiers, raw coordinate files, or key linking athlete codes to personal characteristics. The original article reports an age range including minors. This repository does not infer or invent consent procedures that are absent from the public source record; manuscript-level ethics statements remain the responsibility of the submitting authors and their institution.

# Data provenance and archive decisions

## Source record

The analysis is a secondary analysis of the TTMD6 coordinate archive associated with:

> Table tennis motion recognition based on the bat trajectory using varying-length-input convolution neural networks. *Scientific Reports* (2024). https://doi.org/10.1038/s41598-024-54150-5

The raw archive is third-party material and is not redistributed here. The accepted input is fixed by byte count and two hashes in `reproduce_all.py`; a mismatch stops the workflow before extraction.

## Reported cohort versus archive structure

The source article reports 30 participants and 9,000 strokes. The examined archive contains 12,000 racket files and 12,000 body files organized as 40 numeric codes × 6 action labels × 50 trials.

The analysis therefore uses a predeclared deterministic operational rule:

1. Codes 1–30 form the primary analysis block (30 × 6 × 50 = 9,000 paired trials).
2. Codes 31–40 are quarantined and are never silently included in the primary inferential analysis.
3. The rule matches the reported participant count but is not represented as an author-verified identity mapping.

This unresolved provenance is an inferential limitation, not merely a file-format detail.

## Archive audit findings

- 50 racket and 50 body files contain exact duplicated 200-row halves within 400-row matrices; the duplicated half is deterministically collapsed.
- Eight exact paired-record groups span primary and quarantined codes; no exact whole-trial duplicates occur entirely inside the primary block.
- The filename nominal length exceeds the public 200-row matrix boundary for 1,180 primary trials. The main analysis uses the observable 200 rows and does not claim to reconstruct unavailable frames.
- Structural `[0,0,0]` joint triplets affect 190 body trials. For each adjacent-frame interval, a joint is omitted when either endpoint is structural zero; remaining joint displacement magnitudes are averaged. No coordinate gap filling is performed.
- The public archive contains 50 trials per cell, whereas the source description refers to 55 consecutive strokes. The selection mechanism for the five unpublished strokes is unknown.

Every item above is emitted in machine-readable audit tables during reproduction.

## Data-record caveat

The later Figshare record at https://doi.org/10.6084/m9.figshare.31746358 has been reported with a byte count different from the archive analyzed here. A shared dataset title or concept DOI is not proof of byte identity. Users must verify the exact archive hash.

## Privacy and ethics boundary

The repository contains no names, direct identifiers, raw coordinate files, or key linking athlete codes to personal characteristics. The original article reports an age range including minors. This repository does not infer or invent consent procedures that are absent from the public source record; manuscript-level ethics statements remain the responsibility of the submitting authors and their institution.

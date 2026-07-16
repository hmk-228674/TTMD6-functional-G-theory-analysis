# TTMD6 joint and action-label evidence map

## Published 14-label anatomical order

The source article's Figure 1 labels the 14-point skeleton in the order below.
The archive human CSV files have 42 unnamed numeric columns; the analysis reads
them as 14 consecutive three-coordinate blocks in this published order. Because
the displacement magnitude is Euclidean, the calculation is invariant to a
permutation of the three coordinate axes. The source labels identify anatomical
segments but do not supply an exact marker-centre definition for each derived
point.

| Index | Source label | Chinese mapping | Archive columns (1-based) |
|---:|---|---|---:|
| 1 | Hips | 髋部/骨盆 | 1--3 |
| 2 | Head | 头部 | 4--6 |
| 3 | LeftShoulder | 左肩 | 7--9 |
| 4 | LeftArm | 左上臂 | 10--12 |
| 5 | LeftForeArm | 左前臂 | 13--15 |
| 6 | RightShoulder | 右肩 | 16--18 |
| 7 | RightArm | 右上臂 | 19--21 |
| 8 | RightForeArm | 右前臂 | 22--24 |
| 9 | LeftUpLeg | 左大腿 | 25--27 |
| 10 | LeftLeg | 左小腿 | 28--30 |
| 11 | LeftFoot | 左足 | 31--33 |
| 12 | RightUpLeg | 右大腿 | 34--36 |
| 13 | RightLeg | 右小腿 | 37--39 |
| 14 | RightFoot | 右足 | 40--42 |

Source: Zhang et al., *Scientific Reports* 14, 3549 (2024), Figure 1,
https://doi.org/10.1038/s41598-024-54150-5.

## Why the body summary is an equal-weight mean

For each adjacent-frame interval, the analysis calculates the Euclidean
displacement magnitude of every anatomical label observed at both endpoints
and averages those magnitudes with equal weight. This produces one scalar
whole-configuration motion-magnitude waveform per trial, avoids cancellation
across signed axes, and allows the body and racket waveforms to enter the same
functional variance framework. Equal weighting is transparent and reproducible
because TTMD6 provides no segment masses or inertial parameters. It must not be
interpreted as centre-of-mass motion, a joint-angle waveform, or a coordination
index. Fixed 8-label, fixed 13-label, and complete-case sensitivities examine
the dependence of conclusions on this aggregation choice.

## Limited evidence for the six action labels

The source study selected 30 racket-trajectory animations (five per archived
action label) and asked 40 professional table-tennis athletes to classify them.
Mean recognition accuracy was 92.63%; attack and drive were the main confusion
pairs, while the two push labels had the highest reported accuracies. This
supports coarse recognizability of a small animation sample based on racket
trajectory. It does not provide trial-level adjudication of all 9,000 primary
records, independent validation of the body-coordinate labels, or validation
of the archive code-to-participant mapping. The secondary analysis therefore
treats the six labels as fixed archive labels rather than error-free clinical
or biomechanical constructs.

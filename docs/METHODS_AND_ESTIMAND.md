# Methods and estimand contract

## Operational waveform definitions

- **Racket waveform:** Euclidean magnitude of frame-to-frame displacement of the archived racket centroid coordinate.
- **Body waveform:** mean Euclidean frame-to-frame displacement magnitude across joints usable at both adjacent frames.
- **Phase normalization:** the observable segment is interpolated to 200 equally spaced phase nodes.

The historical cache filename contains `speed`, but no frame interval is applied. The quantity is displacement per adjacent frame, not physical speed.

## Action-specific model

For each of six fixed archive action labels and each waveform family, the nodewise model is

$$
y_{ij}(q)=\mu(q)+u_i(q)+e_{ij}(q),
$$

with athlete-code variance $\sigma^2_{A,a}(q)$ and within-athlete trial variance $\sigma^2_{E,a}(q)$. Nodewise non-negative REML estimates are integrated with trapezoidal weights:

$$
B_a=\int \sigma^2_{A,a}(q)\,dq, \qquad
W_a=\int \sigma^2_{E,a}(q)\,dq.
$$

The reported trace relative reliability for the mean of $n$ trials is

$$
R_{L^2,a}(n)=\frac{B_a}{B_a+W_a/n}.
$$

The continuous trial-count threshold for target $r$ is

$$
n_a^*(r)=\frac{rW_a}{(1-r)B_a},
$$

and the integer recommendation is the ceiling with a minimum of one trial. This trace ratio is not the integral of the pointwise ratios and is not labeled a standard functional G coefficient.

## Uncertainty

Population-level uncertainty uses 5,000 whole-athlete cluster-bootstrap replicates. A replicate resamples the 30 athlete codes, retains each sampled athlete's complete action/trial structure, and uses the same cluster draw across waveform families and actions.

Trials are not treated as independent population-level sampling units.

## Sensitivity analyses

The workflow includes:

- exact unbalanced profile REML after excluding nominal lengths above 200;
- 1,000 within-athlete/action balanced subsamples after that exclusion;
- a prespecified one-sided Hampel-type local-high-value rule;
- fixed 8-joint and fixed 13-joint body representations;
- complete-case structural-zero sensitivity;
- archive-order lag correlation and permutation reference ranges;
- AR(1) working scenarios, which are scenarios rather than fitted dependence parameters;
- robust racket-peak registration applied to paired racket/body waveforms;
- leave-one-athlete-out influence;
- percentile, basic, BCa, and Monte Carlo bootstrap diagnostics;
- pointwise marginal bootstrap intervals, which are not simultaneous bands and are not used for local significance claims.

## Fixed-label contextual decomposition

A separate across-label decomposition describes dispersion for the observed finite set of six labels. The action component is a noise-corrected finite-set contrast dispersion; it is not interpreted as variance from a randomly sampled universe of actions. Action-specific models carry the trial-count design inference.
